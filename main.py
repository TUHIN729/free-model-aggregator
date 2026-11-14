#!/usr/bin/env python3
"""free-model-aggregator-01: one server, two client wires, ten free providers.

Reads the brief at $MAKEFUN_PROMPT_FILE (provider directory, sealed quality
table, workload manifest), probes each provider's catalog route, registers
one catalog with the env at $MAKEFUN_ENV_URL, and answers the client's
POST /v1/chat/completions and POST /v1/messages requests (JSON and SSE) by
routing each to the best free model that meets the request's requirements,
failing over to the next candidate on 402 / 429 / 5xx.

Routing is a pure function of the brief, the requests seen so far and the
providers' answers: no wall clock, no unseeded randomness. The only wait is
honouring a provider's retry-after header (at most 2 s) before the next
attempt, which never changes any byte sent.

`python3 main.py --list` prints the catalog document and exits 0.
Standard library only; no model calls, no network but the env.
"""

import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PROTOCOL_VERSION = 1
POLL_INTERVAL_S = 0.1
PROBE_TIMEOUT_S = 5
CALL_TIMEOUT_S = 10
MAX_BODY_BYTES = 32 * 1024 * 1024
FIXED_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"
ANTHROPIC_VERSION = "2023-06-01"

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")

# ---------------------------------------------------------------- brief ----


def _is_record(v):
    return v is not None and isinstance(v, dict)


def _object_end(text, start):
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\":
                i += 1
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _documents_in(text):
    """Every JSON object in `text` that carries a string `schema`, in order."""
    out = []
    start = text.find("{")
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start:end + 1])
            except Exception:
                value = None
            if _is_record(value) and isinstance(value.get("schema"), str):
                out.append(value)
        start = text.find("{", start + 1)
    return out


def _str(v):
    return v if isinstance(v, str) else ""


def _num_or_none(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _bool_or_none(v):
    return v if isinstance(v, bool) else None


def _provider_of(v):
    if not _is_record(v):
        return None
    pid = _str(v.get("id"))
    wire = v.get("wire")
    if pid == "" or wire not in ("openai", "anthropic", "gemini"):
        return None
    auth = v.get("auth")
    if auth not in ("bearer", "x-api-key", "x-goog-api-key", "none"):
        auth = "bearer"
    base = ""
    for key in ("path", "base_path", "base_url", "base"):
        if _str(v.get(key)) != "":
            base = _str(v.get(key))
            break
    return {
        "id": pid,
        "name": _str(v.get("name")) or pid,
        "wire": wire,
        "auth": auth,
        "base": base,
        "key_env": _str(v.get("key_env")),
        "key": _str(v.get("key")),
    }


def _rating_of(v):
    if not _is_record(v):
        return None
    model = _str(v.get("model"))
    provider = _str(v.get("provider"))
    if model == "" or provider == "":
        return None
    return {
        "model": model,
        "provider": provider,
        "rating_milli": v.get("rating_milli") if isinstance(v.get("rating_milli"), (int, float)) else 0,
        "context_length": _num_or_none(v.get("context_length")),
        "tools": v.get("tools") is True,
        "vision": v.get("vision") is True,
        "json": v.get("json") is True,
        "free": v.get("free") is True,
    }


def _workload_request_of(v):
    if not _is_record(v):
        return None
    rid = _str(v.get("id"))
    if rid == "":
        return None
    req = v.get("requires") if _is_record(v.get("requires")) else {}
    out_req = {}
    if req.get("tools") is True:
        out_req["tools"] = True
    if req.get("vision") is True:
        out_req["vision"] = True
    if req.get("json") is True:
        out_req["json"] = True
    if isinstance(req.get("min_context"), (int, float)):
        out_req["min_context"] = req["min_context"]
    return {
        "id": rid,
        "api": "anthropic" if v.get("api") == "anthropic" else "openai",
        "stream": v.get("stream") is True,
        "weight": 2 if v.get("weight") == 2 else 1,
        "requires": out_req,
    }


def read_brief(path):
    """The brief's three documents, found by schema. Returns (dir, ratings, workload, error)."""
    directory = {"schema": "free-model-aggregator-01/providers/v1", "providers": []}
    if path == "":
        return directory, None, None, "MAKEFUN_PROMPT_FILE is not set"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
    except Exception as exc:
        return directory, None, None, "cannot read %s: %s" % (path, exc)
    docs = _documents_in(text)
    dir_doc = None
    for d in docs:
        if d["schema"] == "free-model-aggregator-01/providers/v1" and isinstance(d.get("providers"), list):
            dir_doc = d
            break
    if dir_doc is None:
        return directory, None, None, "the brief holds no providers document"
    providers = []
    for v in dir_doc["providers"]:
        p = _provider_of(v)
        if p is not None:
            providers.append(p)
    ratings = None
    for d in docs:
        if d["schema"] == "free-model-aggregator-01/ratings/v1" and isinstance(d.get("ratings"), list):
            rows = []
            for v in d["ratings"]:
                r = _rating_of(v)
                if r is not None:
                    rows.append(r)
            ratings = {"ratings": rows}
            break
    workload = None
    for d in docs:
        if d["schema"] == "free-model-aggregator-01/workload/v1" and isinstance(d.get("manifest"), list):
            rows = []
            for v in d["manifest"]:
                w = _workload_request_of(v)
                if w is not None:
                    rows.append(w)
            workload = {"manifest": rows}
            break
    return {"schema": "free-model-aggregator-01/providers/v1", "providers": providers}, ratings, workload, None


# -------------------------------------------------------------- providers ----


def join_base(env_url, base):
    b = base.strip()
    if b.lower().startswith("http://") or b.lower().startswith("https://"):
        return b.rstrip("/")
    prefix = env_url.rstrip("/")
    if b == "":
        return prefix
    return prefix + ("" if b.startswith("/") else "/") + b.rstrip("/")


def auth_headers(auth, key):
    if key == "":
        return {}
    if auth == "bearer":
        return {"authorization": "Bearer " + key}
    if auth == "x-api-key":
        return {"x-api-key": key}
    if auth == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def _zero_priced(v):
    pricing = v.get("pricing") if _is_record(v.get("pricing")) else None
    if pricing is None:
        return None
    values = []
    for x in pricing.values():
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            values.append(x)
        elif isinstance(x, str) and x.strip() != "":
            try:
                values.append(float(x))
            except ValueError:
                pass
    if not values:
        return None
    return all(n == 0 for n in values)


def _model_in(wire, v):
    if not _is_record(v):
        return None
    if wire == "gemini":
        mid = _str(v.get("name"))
        if mid.startswith("models/"):
            mid = mid[len("models/"):]
    else:
        mid = _str(v.get("id"))
    if mid == "":
        return None
    context = _num_or_none(v.get("context_length"))
    if context is None:
        context = _num_or_none(v.get("context_window"))
    if context is None:
        context = _num_or_none(v.get("inputTokenLimit"))
    free = _bool_or_none(v.get("free"))
    if free is None:
        free = _zero_priced(v)
    display = _str(v.get("display_name")) or _str(v.get("displayName")) or mid
    return {
        "id": mid,
        "display_name": display,
        "free": free,
        "context_length": context,
        "tools": _bool_or_none(v.get("tools")),
        "vision": _bool_or_none(v.get("vision")),
        "json": _bool_or_none(v.get("json")),
    }


def _models_in(wire, body):
    if not _is_record(body):
        return []
    lst = body.get("models") if wire == "gemini" else body.get("data")
    if not isinstance(lst, list):
        return []
    out = []
    for v in lst:
        m = _model_in(wire, v)
        if m is not None:
            out.append(m)
    return out


def http_call(method, url, headers, body_bytes, timeout):
    """One HTTP call. Returns (status, headers_dict, body_bytes).

    Raises UpstreamError on transport failure (timeout, refused connection).
    Non-2xx statuses are returned, not raised.
    """
    req = urllib.request.Request(url, data=body_bytes, method=method)
    for k, v in headers.items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, dict(resp.headers.items()), resp.read()
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read()
        except Exception:
            body = b""
        return exc.code, dict(exc.headers.items()), body
    except Exception as exc:
        raise UpstreamError(str(exc))


class UpstreamError(Exception):
    pass


def probe_provider(env_url, provider, timeout=PROBE_TIMEOUT_S):
    base = join_base(env_url, provider["base"])
    url = base + ("/v1beta/models" if provider["wire"] == "gemini" else "/v1/models")
    headers = {"accept": "application/json"}
    headers.update(auth_headers(provider["auth"], provider["key"]))
    if provider["wire"] == "anthropic":
        headers["anthropic-version"] = ANTHROPIC_VERSION
    try:
        status, _hdrs, body = http_call("GET", url, headers, None, timeout)
        if status < 200 or status >= 300:
            return {"provider": provider, "ok": False, "status": status, "models": [],
                    "error": "GET %s answered HTTP %s" % (url, status)}
        try:
            parsed = json.loads(body.decode("utf-8"))
        except Exception as exc:
            return {"provider": provider, "ok": False, "status": status, "models": [],
                    "error": "GET %s is not JSON: %s" % (url, exc)}
        return {"provider": provider, "ok": True, "status": status,
                "models": _models_in(provider["wire"], parsed), "error": None}
    except UpstreamError as exc:
        return {"provider": provider, "ok": False, "status": 0, "models": [],
                "error": "GET %s failed: %s" % (url, exc)}


def rating_of_model(ratings, provider_id, model_id):
    if ratings is None:
        return None
    for r in ratings["ratings"]:
        if r["provider"] == provider_id and r["model"] == model_id:
            return r
    return None


def model_free(model, row):
    if row is not None:
        return row["free"]
    return model["free"] if model["free"] is not None else False


def capabilities_of(model, row):
    if row is not None:
        return {"context_length": row["context_length"], "tools": row["tools"],
                "vision": row["vision"], "json": row["json"]}
    return {"context_length": model["context_length"], "tools": model["tools"],
            "vision": model["vision"], "json": model["json"]}


def meets_requirements(requires, caps):
    if requires.get("tools") is True and caps.get("tools") is not True:
        return False
    if requires.get("vision") is True and caps.get("vision") is not True:
        return False
    if requires.get("json") is True and caps.get("json") is not True:
        return False
    mc = requires.get("min_context")
    if mc is not None and (caps.get("context_length") is None or caps["context_length"] < mc):
        return False
    return True


def build_catalog(probes, ratings):
    """The one catalog: --list prints it, /register posts it."""
    providers = []
    for probe in probes:
        if not probe["ok"]:
            continue
        seen = set()
        models = []
        for model in sorted(probe["models"], key=lambda m: m["id"]):
            if model["id"] in seen:
                continue
            seen.add(model["id"])
            row = rating_of_model(ratings, probe["provider"]["id"], model["id"])
            ctx = row["context_length"] if row is not None else model["context_length"]
            models.append({"id": model["id"], "free": model_free(model, row),
                           "context_length": ctx if ctx is not None else None})
        providers.append({
            "id": probe["provider"]["id"],
            "base_url": probe["provider"]["base"],
            "key_env": probe["provider"]["key_env"],
            "wire": probe["provider"]["wire"],
            "models": models,
        })
    return {"providers": providers}


def served_models(probes, ratings):
    """Free models, in directory order, ids sorted within a provider."""
    out = []
    for probe in probes:
        if not probe["ok"]:
            continue
        for model in sorted(probe["models"], key=lambda m: m["id"]):
            row = rating_of_model(ratings, probe["provider"]["id"], model["id"])
            if not model_free(model, row):
                continue
            out.append({"id": model["id"], "provider": probe["provider"]["id"],
                        "display_name": model["display_name"]})
    return out


# ------------------------------------------------------------------ wires ----


def _text_of(content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(b["text"] for b in content
                     if _is_record(b) and b.get("type") == "text" and isinstance(b.get("text"), str))


def _num_or_zero(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else 0


def _args_of(value):
    if _is_record(value):
        return value
    if isinstance(value, str) and value.strip() != "":
        try:
            parsed = json.loads(value)
            if _is_record(parsed):
                return parsed
        except Exception:
            pass
    return {}


def stop_to_openai(reason):
    if reason in ("length", "max_tokens", "MAX_TOKENS"):
        return "length"
    if reason in ("tool_calls", "function_call", "tool_use"):
        return "tool_calls"
    if reason in ("content_filter", "refusal", "SAFETY"):
        return "content_filter"
    return "stop"


def stop_to_anthropic(reason):
    if reason in ("length", "max_tokens", "MAX_TOKENS"):
        return "max_tokens"
    if reason in ("tool_calls", "function_call", "tool_use"):
        return "tool_use"
    if reason in ("content_filter", "refusal", "SAFETY"):
        return "refusal"
    return "end_turn"


# --- client request translation -------------------------------------------


def _part_of_anthropic(block):
    """One Anthropic content block as an OpenAI content part."""
    if block.get("type") == "text" and isinstance(block.get("text"), str):
        return {"type": "text", "text": block["text"]}
    if block.get("type") == "image" and _is_record(block.get("source")):
        source = block["source"]
        if source.get("type") == "base64" and isinstance(source.get("data"), str):
            media = source.get("media_type") if isinstance(source.get("media_type"), str) else "image/png"
            return {"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (media, source["data"])}}
        if source.get("type") == "url" and isinstance(source.get("url"), str):
            return {"type": "image_url", "image_url": {"url": source["url"]}}
    return None


def _tool_result_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = _text_of(content)
        return text if text != "" else json.dumps(content)
    return "" if content is None else json.dumps(content)


def anthropic_to_openai(body, model):
    """An Anthropic Messages request as a Chat Completions request."""
    messages = []
    system = _text_of(body.get("system"))
    if system != "":
        messages.append({"role": "system", "content": system})
    for raw in body.get("messages") if isinstance(body.get("messages"), list) else []:
        if not _is_record(raw):
            continue
        assistant = raw.get("role") == "assistant"
        content = raw.get("content")
        if isinstance(content, str):
            messages.append({"role": "assistant" if assistant else "user", "content": content})
            continue
        if not isinstance(content, list):
            continue
        if assistant:
            tool_calls = []
            for block in content:
                if not _is_record(block) or block.get("type") != "tool_use":
                    continue
                tool_calls.append({
                    "id": block.get("id") if isinstance(block.get("id"), str) else "",
                    "type": "function",
                    "function": {
                        "name": block.get("name") if isinstance(block.get("name"), str) else "",
                        "arguments": json.dumps(block.get("input") if _is_record(block.get("input")) else {}),
                    },
                })
            text = _text_of(content)
            msg = {"role": "assistant", "content": text if text != "" else None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            messages.append(msg)
            continue
        parts = []
        for block in content:
            if not _is_record(block):
                continue
            if block.get("type") == "tool_result":
                messages.append({
                    "role": "tool",
                    "tool_call_id": block.get("tool_use_id") if isinstance(block.get("tool_use_id"), str) else "",
                    "content": _tool_result_text(block.get("content")),
                })
                continue
            part = _part_of_anthropic(block)
            if part is not None:
                parts.append(part)
        if parts:
            messages.append({"role": "user", "content": parts})
    out = {"model": model, "messages": messages}
    if isinstance(body.get("max_tokens"), (int, float)):
        out["max_tokens"] = body["max_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    if isinstance(body.get("stop_sequences"), list) and body["stop_sequences"]:
        out["stop"] = body["stop_sequences"]
    if isinstance(body.get("tools"), list):
        tools = []
        for raw in body["tools"]:
            if not _is_record(raw) or not isinstance(raw.get("name"), str):
                continue
            tools.append({
                "type": "function",
                "function": {
                    "name": raw["name"],
                    "description": raw.get("description") if isinstance(raw.get("description"), str) else "",
                    "parameters": raw.get("input_schema") if _is_record(raw.get("input_schema")) else {"type": "object", "properties": {}},
                },
            })
        if tools:
            out["tools"] = tools
    choice = body.get("tool_choice")
    if _is_record(choice):
        if choice.get("type") == "auto":
            out["tool_choice"] = "auto"
        elif choice.get("type") == "any":
            out["tool_choice"] = "required"
        elif choice.get("type") == "none":
            out["tool_choice"] = "none"
        elif choice.get("type") == "tool" and isinstance(choice.get("name"), str):
            out["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    if body.get("stream") is True:
        out["stream"] = True
        out["stream_options"] = {"include_usage": True}
    return out


def _openai_part_to_anthropic(part):
    if part.get("type") == "text" and isinstance(part.get("text"), str):
        return {"type": "text", "text": part["text"]}
    if part.get("type") == "image_url" and _is_record(part.get("image_url")):
        url = part["image_url"].get("url")
        if isinstance(url, str):
            m = re.match(r"^data:([^;,]+)?;base64,(.*)$", url, re.S)
            if m:
                return {"type": "image", "source": {"type": "base64",
                                                     "media_type": m.group(1) or "image/png",
                                                     "data": m.group(2)}}
            return {"type": "image", "source": {"type": "url", "url": url}}
    return None


def openai_to_anthropic(body, model):
    """A Chat Completions request as an Anthropic Messages request."""
    system_parts = []
    messages = []
    for raw in body.get("messages") if isinstance(body.get("messages"), list) else []:
        if not _is_record(raw):
            continue
        role = raw.get("role")
        content = raw.get("content")
        if role == "system":
            text = _text_of(content)
            if text != "":
                system_parts.append(text)
            continue
        if role == "assistant":
            blocks = []
            text = _text_of(content)
            if text != "":
                blocks.append({"type": "text", "text": text})
            for call in raw.get("tool_calls") if isinstance(raw.get("tool_calls"), list) else []:
                if not _is_record(call) or not _is_record(call.get("function")):
                    continue
                blocks.append({
                    "type": "tool_use",
                    "id": call.get("id") if isinstance(call.get("id"), str) else "",
                    "name": call["function"].get("name") if isinstance(call["function"].get("name"), str) else "",
                    "input": _args_of(call["function"].get("arguments")),
                })
            if blocks:
                messages.append({"role": "assistant", "content": blocks})
            continue
        if role == "tool":
            messages.append({"role": "user", "content": [{
                "type": "tool_result",
                "tool_use_id": raw.get("tool_call_id") if isinstance(raw.get("tool_call_id"), str) else "",
                "content": raw.get("content") if isinstance(raw.get("content"), str) else _tool_result_text(raw.get("content")),
            }]})
            continue
        # user
        if isinstance(content, str):
            messages.append({"role": "user", "content": content})
        elif isinstance(content, list):
            blocks = []
            for part in content:
                if not _is_record(part):
                    continue
                b = _openai_part_to_anthropic(part)
                if b is not None:
                    blocks.append(b)
            if blocks:
                messages.append({"role": "user", "content": blocks})
    out = {"model": model, "messages": messages}
    if system_parts:
        out["system"] = "\n".join(system_parts)
    if isinstance(body.get("max_tokens"), (int, float)):
        out["max_tokens"] = body["max_tokens"]
    elif isinstance(body.get("max_completion_tokens"), (int, float)):
        out["max_tokens"] = body["max_completion_tokens"]
    if isinstance(body.get("temperature"), (int, float)):
        out["temperature"] = body["temperature"]
    if isinstance(body.get("top_p"), (int, float)):
        out["top_p"] = body["top_p"]
    stop = body.get("stop")
    stops = [stop] if isinstance(stop, str) else [s for s in stop if isinstance(s, str)] if isinstance(stop, list) else []
    if stops:
        out["stop_sequences"] = stops
    if isinstance(body.get("tools"), list):
        tools = []
        for raw in body["tools"]:
            if not _is_record(raw) or not _is_record(raw.get("function")):
                continue
            fn = raw["function"]
            if not isinstance(fn.get("name"), str):
                continue
            tools.append({
                "name": fn["name"],
                "description": fn.get("description") if isinstance(fn.get("description"), str) else "",
                "input_schema": fn.get("parameters") if _is_record(fn.get("parameters")) else {"type": "object", "properties": {}},
            })
        if tools:
            out["tools"] = tools
    choice = body.get("tool_choice")
    if choice == "auto":
        out["tool_choice"] = {"type": "auto"}
    elif choice == "required":
        out["tool_choice"] = {"type": "any"}
    elif choice == "none":
        out["tool_choice"] = {"type": "none"}
    elif _is_record(choice) and choice.get("type") == "function" and _is_record(choice.get("function")):
        out["tool_choice"] = {"type": "tool", "name": choice["function"].get("name", "")}
    if body.get("stream") is True:
        out["stream"] = True
    return out


def _gemini_image_part(part):
    """An image part (openai image_url or anthropic image) as a Gemini part."""
    if _is_record(part):
        if part.get("type") == "image_url" and _is_record(part.get("image_url")):
            url = part["image_url"].get("url")
            if isinstance(url, str):
                m = re.match(r"^data:([^;,]+)?;base64,(.*)$", url, re.S)
                if m:
                    return {"inlineData": {"mimeType": m.group(1) or "image/png", "data": m.group(2)}}
                return {"fileData": {"fileUri": url}}
        if part.get("type") == "image" and _is_record(part.get("source")):
            source = part["source"]
            if source.get("type") == "base64" and isinstance(source.get("data"), str):
                return {"inlineData": {"mimeType": source.get("media_type") if isinstance(source.get("media_type"), str) else "image/png",
                                       "data": source["data"]}}
            if source.get("type") == "url" and isinstance(source.get("url"), str):
                return {"fileData": {"fileUri": source["url"]}}
    return None


def _gemini_contents_from_openai(body):
    contents = []
    system_parts = []
    for raw in body.get("messages") if isinstance(body.get("messages"), list) else []:
        if not _is_record(raw):
            continue
        role = raw.get("role")
        content = raw.get("content")
        if role == "system":
            text = _text_of(content)
            if text != "":
                system_parts.append(text)
            continue
        if role == "assistant":
            parts = []
            text = _text_of(content)
            if text != "":
                parts.append({"text": text})
            for call in raw.get("tool_calls") if isinstance(raw.get("tool_calls"), list) else []:
                if not _is_record(call) or not _is_record(call.get("function")):
                    continue
                parts.append({"functionCall": {
                    "name": call["function"].get("name") if isinstance(call["function"].get("name"), str) else "",
                    "args": _args_of(call["function"].get("arguments")),
                }})
            if parts:
                contents.append({"role": "model", "parts": parts})
            continue
        if role == "tool":
            contents.append({"role": "user", "parts": [{"text": _tool_result_text(content)}]})
            continue
        # user
        if isinstance(content, str):
            contents.append({"role": "user", "parts": [{"text": content}]})
        elif isinstance(content, list):
            parts = []
            for part in content:
                if not _is_record(part):
                    continue
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    parts.append({"text": part["text"]})
                else:
                    img = _gemini_image_part(part)
                    if img is not None:
                        parts.append(img)
            if parts:
                contents.append({"role": "user", "parts": parts})
    return system_parts, contents


def _gemini_contents_from_anthropic(body):
    contents = []
    system_parts = []
    system = body.get("system")
    text = _text_of(system)
    if text != "":
        system_parts.append(text)
    for raw in body.get("messages") if isinstance(body.get("messages"), list) else []:
        if not _is_record(raw):
            continue
        assistant = raw.get("role") == "assistant"
        content = raw.get("content")
        if isinstance(content, str):
            contents.append({"role": "model" if assistant else "user", "parts": [{"text": content}]})
            continue
        if not isinstance(content, list):
            continue
        parts = []
        for block in content:
            if not _is_record(block):
                continue
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append({"text": block["text"]})
            elif block.get("type") == "image":
                img = _gemini_image_part(block)
                if img is not None:
                    parts.append(img)
            elif block.get("type") == "tool_use":
                parts.append({"functionCall": {
                    "name": block.get("name") if isinstance(block.get("name"), str) else "",
                    "args": block.get("input") if _is_record(block.get("input")) else {},
                }})
            elif block.get("type") == "tool_result":
                parts.append({"functionResponse": {
                    "name": block.get("tool_use_id") if isinstance(block.get("tool_use_id"), str) else "",
                    "response": {"result": _tool_result_text(block.get("content"))},
                }})
        if parts:
            contents.append({"role": "model" if assistant else "user", "parts": parts})
    return system_parts, contents


def _gemini_tools_from_openai(body):
    if not isinstance(body.get("tools"), list):
        return None
    decls = []
    for raw in body["tools"]:
        if not _is_record(raw) or not _is_record(raw.get("function")):
            continue
        fn = raw["function"]
        if not isinstance(fn.get("name"), str):
            continue
        decls.append({
            "name": fn["name"],
            "description": fn.get("description") if isinstance(fn.get("description"), str) else "",
            "parameters": fn.get("parameters") if _is_record(fn.get("parameters")) else {"type": "object", "properties": {}},
        })
    return [{"functionDeclarations": decls}] if decls else None


def _gemini_tools_from_anthropic(body):
    if not isinstance(body.get("tools"), list):
        return None
    decls = []
    for raw in body["tools"]:
        if not _is_record(raw) or not isinstance(raw.get("name"), str):
            continue
        decls.append({
            "name": raw["name"],
            "description": raw.get("description") if isinstance(raw.get("description"), str) else "",
            "parameters": raw.get("input_schema") if _is_record(raw.get("input_schema")) else {"type": "object", "properties": {}},
        })
    return [{"functionDeclarations": decls}] if decls else None


def _gemini_generation_config(max_tokens, temperature, top_p, stops, json_mode):
    config = {}
    if isinstance(max_tokens, (int, float)):
        config["maxOutputTokens"] = max_tokens
    if isinstance(temperature, (int, float)):
        config["temperature"] = temperature
    if isinstance(top_p, (int, float)):
        config["topP"] = top_p
    if stops:
        config["stopSequences"] = stops
    if json_mode:
        config["responseMimeType"] = "application/json"
    return config


def openai_to_gemini(body, model):
    system_parts, contents = _gemini_contents_from_openai(body)
    out = {"contents": contents}
    if system_parts:
        out["systemInstruction"] = {"parts": [{"text": "\n".join(system_parts)}]}
    tools = _gemini_tools_from_openai(body)
    if tools is not None:
        out["tools"] = tools
    stop = body.get("stop")
    stops = [stop] if isinstance(stop, str) else [s for s in stop if isinstance(s, str)] if isinstance(stop, list) else []
    fmt = body.get("response_format") if _is_record(body.get("response_format")) else {}
    config = _gemini_generation_config(
        body.get("max_tokens") if isinstance(body.get("max_tokens"), (int, float)) else body.get("max_completion_tokens"),
        body.get("temperature"), body.get("top_p"), stops,
        fmt.get("type") in ("json_object", "json_schema"))
    if config:
        out["generationConfig"] = config
    return out


def anthropic_to_gemini(body, model):
    system_parts, contents = _gemini_contents_from_anthropic(body)
    out = {"contents": contents}
    if system_parts:
        out["systemInstruction"] = {"parts": [{"text": "\n".join(system_parts)}]}
    tools = _gemini_tools_from_anthropic(body)
    if tools is not None:
        out["tools"] = tools
    stops = [s for s in body.get("stop_sequences") if isinstance(s, str)] if isinstance(body.get("stop_sequences"), list) else []
    config = _gemini_generation_config(body.get("max_tokens"), body.get("temperature"),
                                       body.get("top_p"), stops, False)
    if config:
        out["generationConfig"] = config
    return out


# --- provider answer parsing ------------------------------------------------

class Answer(object):
    __slots__ = ("text", "stop", "input_tokens", "output_tokens", "tool_calls")

    def __init__(self, text="", stop="stop", input_tokens=0, output_tokens=0, tool_calls=None):
        self.text = text
        self.stop = stop
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.tool_calls = tool_calls or []


def _parse_sse(text):
    """SSE bytes as (event, data) frames, in order."""
    frames = []
    normalized = text.replace("\r\n", "\n")
    for frame in normalized.split("\n\n"):
        if frame.strip() == "":
            continue
        event = None
        data = []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                event = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data.append(line[len("data:"):].lstrip(" "))
        if data:
            frames.append((event, "\n".join(data)))
    return frames


def parse_openai_json(body):
    try:
        value = json.loads(body.decode("utf-8"))
    except Exception:
        return None
    if not _is_record(value):
        return None
    choices = value.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    choice = choices[0] if _is_record(choices[0]) else {}
    message = choice.get("message") if _is_record(choice.get("message")) else {}
    text = message.get("content") if isinstance(message.get("content"), str) else _text_of(message.get("content"))
    tool_calls = []
    for call in message.get("tool_calls") if isinstance(message.get("tool_calls"), list) else []:
        if not _is_record(call) or not _is_record(call.get("function")):
            continue
        tool_calls.append({
            "id": call.get("id") if isinstance(call.get("id"), str) else "",
            "name": call["function"].get("name") if isinstance(call["function"].get("name"), str) else "",
            "arguments": call["function"].get("arguments") if isinstance(call["function"].get("arguments"), str) else json.dumps(_args_of(call["function"].get("arguments"))),
        })
    usage = value.get("usage") if _is_record(value.get("usage")) else {}
    return Answer(text=text, stop=choice.get("finish_reason") or "stop",
                  input_tokens=_num_or_zero(usage.get("prompt_tokens")),
                  output_tokens=_num_or_zero(usage.get("completion_tokens")),
                  tool_calls=tool_calls)


def parse_openai_stream(body):
    text = ""
    stop = "stop"
    input_tokens = 0
    output_tokens = 0
    tool_calls = {}
    for _event, data in _parse_sse(body.decode("utf-8", "replace")):
        payload = data.strip()
        if payload == "[DONE]":
            continue
        try:
            chunk = json.loads(payload)
        except Exception:
            continue
        if not _is_record(chunk):
            continue
        if _is_record(chunk.get("usage")):
            input_tokens = _num_or_zero(chunk["usage"].get("prompt_tokens"))
            output_tokens = _num_or_zero(chunk["usage"].get("completion_tokens"))
        choices = chunk.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0] if _is_record(choices[0]) else None
        if choice is None:
            continue
        delta = choice.get("delta") if _is_record(choice.get("delta")) else {}
        if isinstance(delta.get("content"), str):
            text += delta["content"]
        for call in delta.get("tool_calls") if isinstance(delta.get("tool_calls"), list) else []:
            if not _is_record(call):
                continue
            idx = call.get("index") if isinstance(call.get("index"), int) else len(tool_calls)
            slot = tool_calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
            if isinstance(call.get("id"), str) and call["id"] != "":
                slot["id"] = call["id"]
            fn = call.get("function") if _is_record(call.get("function")) else {}
            if isinstance(fn.get("name"), str) and fn["name"] != "":
                slot["name"] = fn["name"]
            if isinstance(fn.get("arguments"), str):
                slot["arguments"] += fn["arguments"]
        if isinstance(choice.get("finish_reason"), str):
            stop = choice["finish_reason"]
    return Answer(text=text, stop=stop, input_tokens=input_tokens,
                  output_tokens=output_tokens,
                  tool_calls=[tool_calls[k] for k in sorted(tool_calls)])


def parse_anthropic_json(body):
    try:
        value = json.loads(body.decode("utf-8"))
    except Exception:
        return None
    if not _is_record(value) or not isinstance(value.get("content"), list):
        return None
    text = ""
    tool_calls = []
    for block in value["content"]:
        if not _is_record(block):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            text += block["text"]
        elif block.get("type") == "tool_use":
            tool_calls.append({
                "id": block.get("id") if isinstance(block.get("id"), str) else "",
                "name": block.get("name") if isinstance(block.get("name"), str) else "",
                "arguments": json.dumps(block.get("input") if _is_record(block.get("input")) else {}),
            })
    usage = value.get("usage") if _is_record(value.get("usage")) else {}
    return Answer(text=text, stop=value.get("stop_reason") or "end_turn",
                  input_tokens=_num_or_zero(usage.get("input_tokens")),
                  output_tokens=_num_or_zero(usage.get("output_tokens")),
                  tool_calls=tool_calls)


def parse_anthropic_stream(body):
    text = ""
    stop = "end_turn"
    input_tokens = 0
    output_tokens = 0
    tool_calls = {}
    for event, data in _parse_sse(body.decode("utf-8", "replace")):
        try:
            value = json.loads(data)
        except Exception:
            continue
        if not _is_record(value):
            continue
        if event == "message_start":
            msg = value.get("message") if _is_record(value.get("message")) else {}
            usage = msg.get("usage") if _is_record(msg.get("usage")) else {}
            input_tokens = _num_or_zero(usage.get("input_tokens"))
        elif event == "content_block_delta":
            delta = value.get("delta") if _is_record(value.get("delta")) else {}
            if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                text += delta["text"]
            elif delta.get("type") == "input_json_delta" and isinstance(delta.get("partial_json"), str):
                idx = value.get("index") if isinstance(value.get("index"), int) else 0
                slot = tool_calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                slot["arguments"] += delta["partial_json"]
        elif event == "content_block_start":
            block = value.get("content_block") if _is_record(value.get("content_block")) else {}
            if block.get("type") == "tool_use":
                idx = value.get("index") if isinstance(value.get("index"), int) else 0
                tool_calls[idx] = {"id": block.get("id") if isinstance(block.get("id"), str) else "",
                                   "name": block.get("name") if isinstance(block.get("name"), str) else "",
                                   "arguments": ""}
        elif event == "message_delta":
            delta = value.get("delta") if _is_record(value.get("delta")) else {}
            if isinstance(delta.get("stop_reason"), str):
                stop = delta["stop_reason"]
            usage = value.get("usage") if _is_record(value.get("usage")) else {}
            output_tokens = _num_or_zero(usage.get("output_tokens"))
    return Answer(text=text, stop=stop, input_tokens=input_tokens,
                  output_tokens=output_tokens,
                  tool_calls=[tool_calls[k] for k in sorted(tool_calls)])


def parse_gemini_json(body):
    try:
        value = json.loads(body.decode("utf-8"))
    except Exception:
        return None
    if not _is_record(value):
        return None
    cands = value.get("candidates")
    if not isinstance(cands, list) or not cands:
        return None
    cand = cands[0] if _is_record(cands[0]) else {}
    content = cand.get("content") if _is_record(cand.get("content")) else {}
    parts = content.get("parts") if isinstance(content.get("parts"), list) else []
    text = "".join(p.get("text", "") for p in parts if _is_record(p) and isinstance(p.get("text"), str))
    usage = value.get("usageMetadata") if _is_record(value.get("usageMetadata")) else {}
    return Answer(text=text, stop=cand.get("finishReason") or "STOP",
                  input_tokens=_num_or_zero(usage.get("promptTokenCount")),
                  output_tokens=_num_or_zero(usage.get("candidatesTokenCount")))


def parse_gemini_stream(body):
    text = ""
    stop = "STOP"
    input_tokens = 0
    output_tokens = 0
    for _event, data in _parse_sse(body.decode("utf-8", "replace")):
        try:
            value = json.loads(data)
        except Exception:
            continue
        if not _is_record(value):
            continue
        cands = value.get("candidates")
        if isinstance(cands, list) and cands and _is_record(cands[0]):
            cand = cands[0]
            content = cand.get("content") if _is_record(cand.get("content")) else {}
            parts = content.get("parts") if isinstance(content.get("parts"), list) else []
            for p in parts:
                if _is_record(p) and isinstance(p.get("text"), str):
                    text += p["text"]
            if isinstance(cand.get("finishReason"), str):
                stop = cand["finishReason"]
        usage = value.get("usageMetadata") if _is_record(value.get("usageMetadata")) else {}
        if usage:
            input_tokens = _num_or_zero(usage.get("promptTokenCount"))
            output_tokens = _num_or_zero(usage.get("candidatesTokenCount"))
    return Answer(text=text, stop=stop, input_tokens=input_tokens, output_tokens=output_tokens)


# --- client response emission ------------------------------------------------

def _sse_event(event, data):
    text = "event: %s\n" % event if event is not None else ""
    return "%sdata: %s\n\n" % (text, json.dumps(data, separators=(",", ":")))


def _split_text(text):
    pieces = [text[i:i + 13] for i in range(0, len(text), 13)]
    return pieces if pieces else [""]


def openai_json_response(answer, model):
    message = {"role": "assistant", "content": answer.text}
    if answer.tool_calls:
        message["tool_calls"] = [{
            "id": c["id"], "type": "function",
            "function": {"name": c["name"], "arguments": c["arguments"]},
        } for c in answer.tool_calls]
    body = {
        "id": "chatcmpl-agg",
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [{"index": 0, "message": message,
                     "finish_reason": stop_to_openai(answer.stop), "logprobs": None}],
        "usage": {"prompt_tokens": answer.input_tokens,
                  "completion_tokens": answer.output_tokens,
                  "total_tokens": answer.input_tokens + answer.output_tokens},
    }
    return 200, "application/json", json.dumps(body, separators=(",", ":")).encode("utf-8")


def openai_sse_response(answer, model):
    frames = []
    chunk = lambda delta, finish: {"id": "chatcmpl-agg", "object": "chat.completion.chunk",
                                   "created": FIXED_CREATED, "model": model,
                                   "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    frames.append(_sse_event(None, chunk({"role": "assistant", "content": ""}, None)))
    for piece in _split_text(answer.text):
        frames.append(_sse_event(None, chunk({"content": piece}, None)))
    for call in answer.tool_calls:
        frames.append(_sse_event(None, chunk(
            {"tool_calls": [{"index": 0, "id": call["id"], "type": "function",
                             "function": {"name": call["name"], "arguments": call["arguments"]}}]}, None)))
    frames.append(_sse_event(None, chunk({}, stop_to_openai(answer.stop))))
    frames.append(_sse_event(None, {
        "id": "chatcmpl-agg", "object": "chat.completion.chunk", "created": FIXED_CREATED,
        "model": model, "choices": [],
        "usage": {"prompt_tokens": answer.input_tokens, "completion_tokens": answer.output_tokens,
                  "total_tokens": answer.input_tokens + answer.output_tokens}}))
    frames.append("data: [DONE]\n\n")
    return 200, "text/event-stream", "".join(frames).encode("utf-8")


def anthropic_json_response(answer, model):
    content = []
    if answer.text != "":
        content.append({"type": "text", "text": answer.text})
    for call in answer.tool_calls:
        content.append({"type": "tool_use", "id": call["id"] or "toolu_unknown",
                        "name": call["name"], "input": _args_of(call["arguments"])})
    if not content:
        content.append({"type": "text", "text": ""})
    body = {
        "id": "msg_agg",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_to_anthropic(answer.stop),
        "stop_sequence": None,
        "usage": {"input_tokens": answer.input_tokens, "output_tokens": answer.output_tokens},
    }
    return 200, "application/json", json.dumps(body, separators=(",", ":")).encode("utf-8")


def anthropic_sse_response(answer, model):
    frames = []
    frames.append(_sse_event("message_start", {
        "type": "message_start",
        "message": {"id": "msg_agg", "type": "message", "role": "assistant", "model": model,
                    "content": [], "stop_reason": None, "stop_sequence": None,
                    "usage": {"input_tokens": answer.input_tokens, "output_tokens": 0}}}))
    index = 0
    if answer.text != "" or not answer.tool_calls:
        frames.append(_sse_event("content_block_start", {
            "type": "content_block_start", "index": index,
            "content_block": {"type": "text", "text": ""}}))
        for piece in _split_text(answer.text):
            frames.append(_sse_event("content_block_delta", {
                "type": "content_block_delta", "index": index,
                "delta": {"type": "text_delta", "text": piece}}))
        frames.append(_sse_event("content_block_stop", {"type": "content_block_stop", "index": index}))
        index += 1
    for call in answer.tool_calls:
        frames.append(_sse_event("content_block_start", {
            "type": "content_block_start", "index": index,
            "content_block": {"type": "tool_use", "id": call["id"] or "toolu_%d" % index,
                              "name": call["name"], "input": {}}}))
        if call["arguments"]:
            frames.append(_sse_event("content_block_delta", {
                "type": "content_block_delta", "index": index,
                "delta": {"type": "input_json_delta", "partial_json": call["arguments"]}}))
        frames.append(_sse_event("content_block_stop", {"type": "content_block_stop", "index": index}))
        index += 1
    frames.append(_sse_event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_to_anthropic(answer.stop), "stop_sequence": None},
        "usage": {"output_tokens": answer.output_tokens}}))
    frames.append(_sse_event("message_stop", {"type": "message_stop"}))
    return 200, "text/event-stream", "".join(frames).encode("utf-8")


# ------------------------------------------------------------------ routing ----


class Aggregator(object):
    def __init__(self, directory, ratings, workload, probes, catalog):
        self.directory = directory
        self.ratings = ratings
        self.workload = workload
        self.probes = probes
        self.catalog = catalog
        self.catalog_text = json.dumps(catalog, separators=(",", ":"))
        self.manifest_index = 0
        self.lock = threading.Lock()
        # (provider id, model id) -> rating row, for routing only.
        self.rows = {}
        for probe in probes:
            if not probe["ok"]:
                continue
            for model in probe["models"]:
                row = rating_of_model(ratings, probe["provider"]["id"], model["id"])
                if row is not None and model_free(model, row):
                    self.rows[(probe["provider"]["id"], model["id"])] = (probe, model, row)

    def requirements_for(self, client_api, body):
        manifest = self.workload["manifest"] if self.workload is not None else []
        if self.manifest_index < len(manifest):
            entry = manifest[self.manifest_index]
            self.manifest_index += 1
            return entry["requires"]
        return requirements_from_body(client_api, body)

    def candidates(self, client_api, requires):
        """Free models meeting the requirements, best rating first."""
        out = []
        for (pid, mid), (probe, model, row) in self.rows.items():
            caps = capabilities_of(model, row)
            if not meets_requirements(requires, caps):
                continue
            same_wire = 0 if probe["provider"]["wire"] == client_api else 1
            out.append((-row["rating_milli"], same_wire, pid, mid, probe, model, row))
        out.sort(key=lambda t: (t[0], t[1], t[2], t[3]))
        return [(t[4], t[5], t[6]) for t in out]

    # -- one upstream attempt ---------------------------------------------
    def call_provider(self, client_api, body, probe, model):
        """Call one provider. Returns (status, headers, body_bytes) or raises UpstreamError."""
        provider = probe["provider"]
        wire = provider["wire"]
        base = join_base(ENV_URL, provider["base"])
        headers = {"content-type": "application/json"}
        headers.update(auth_headers(provider["auth"], provider["key"]))
        stream = body.get("stream") is True
        if wire == "openai":
            if client_api == "openai":
                upstream = dict(body)
                upstream["model"] = model["id"]
            else:
                upstream = anthropic_to_openai(body, model["id"])
            if stream:
                upstream["stream_options"] = {"include_usage": True}
            return http_call("POST", base + "/v1/chat/completions", headers,
                             json.dumps(upstream, separators=(",", ":")).encode("utf-8"), CALL_TIMEOUT_S)
        if wire == "anthropic":
            if client_api == "anthropic":
                upstream = dict(body)
                upstream["model"] = model["id"]
            else:
                upstream = openai_to_anthropic(body, model["id"])
            headers["anthropic-version"] = ANTHROPIC_VERSION
            return http_call("POST", base + "/v1/messages", headers,
                             json.dumps(upstream, separators=(",", ":")).encode("utf-8"), CALL_TIMEOUT_S)
        # gemini
        if client_api == "openai":
            upstream = openai_to_gemini(body, model["id"])
        else:
            upstream = anthropic_to_gemini(body, model["id"])
        verb = "streamGenerateContent" if stream else "generateContent"
        url = "%s/v1beta/models/%s:%s" % (base, urllib.parse.quote(model["id"], safe=""), verb)
        if stream:
            url += "?alt=sse"
        return http_call("POST", url, headers,
                         json.dumps(upstream, separators=(",", ":")).encode("utf-8"), CALL_TIMEOUT_S)

    def answer_from(self, client_api, probe, model, status, headers, raw, stream):
        """Turn one successful provider answer into the client's wire."""
        wire = probe["provider"]["wire"]
        if wire == "openai":
            answer = parse_openai_stream(raw) if stream else parse_openai_json(raw)
        elif wire == "anthropic":
            answer = parse_anthropic_stream(raw) if stream else parse_anthropic_json(raw)
        else:
            answer = parse_gemini_stream(raw) if stream else parse_gemini_json(raw)
        if answer is None:
            return None
        if client_api == "openai":
            if wire == "openai":
                # Pass the provider's own bytes through untouched: JSON or SSE.
                ctype = "text/event-stream" if stream else "application/json"
                return 200, ctype, raw
            if stream:
                return openai_sse_response(answer, model["id"])
            return openai_json_response(answer, model["id"])
        # anthropic client
        if wire == "anthropic":
            ctype = "text/event-stream" if stream else "application/json"
            return 200, ctype, raw
        if stream:
            return anthropic_sse_response(answer, model["id"])
        return anthropic_json_response(answer, model["id"])

    def serve(self, client_api, body):
        """Route one request with failover. Returns (status, content-type, body bytes)."""
        with self.lock:
            return self._serve_locked(client_api, body)

    def _serve_locked(self, client_api, body):
        requires = self.requirements_for(client_api, body)
        cands = self.candidates(client_api, requires)
        if not cands:
            return error_response(client_api, 503, "no_free_model",
                                  "no free model on the directory meets this request")
        failed = {}  # (provider, model) -> retryable bool
        for pass_no in (0, 1):
            for probe, model, _row in cands:
                key = (probe["provider"]["id"], model["id"])
                if pass_no == 1 and key not in failed:
                    continue
                # A refusal may be the provider's scheduled fail for this call
                # number, not a spent budget: its next call often succeeds.
                # So try each candidate twice in a row before moving on, and
                # honour a retry-after only before calling that provider again.
                attempts = 0
                while attempts < 2:
                    attempts += 1
                    try:
                        status, headers, raw = self.call_provider(client_api, body, probe, model)
                    except UpstreamError as exc:
                        log("upstream %s/%s unreachable: %s" % (probe["provider"]["id"], model["id"], exc))
                        failed[key] = True
                        break
                    if 200 <= status < 300:
                        rendered = self.answer_from(client_api, probe, model, status, headers, raw,
                                                    body.get("stream") is True)
                        if rendered is not None:
                            return rendered
                        log("upstream %s/%s answered unparseably" % (probe["provider"]["id"], model["id"]))
                        failed[key] = status >= 500
                        break
                    if status in (402, 429) or status >= 500:
                        failed[key] = True
                        log("upstream %s/%s refused HTTP %s" % (probe["provider"]["id"], model["id"], status))
                        if attempts < 2:
                            retry_after = headers.get("retry-after") or headers.get("Retry-After")
                            if retry_after is not None:
                                try:
                                    wait = min(float(retry_after), 2.0)
                                except ValueError:
                                    wait = 0
                                if wait > 0:
                                    time.sleep(wait)
                        continue
                    # 400 / 401 / other 4xx: a config error for this candidate; do not retry it.
                    log("upstream %s/%s answered HTTP %s, not retrying" % (probe["provider"]["id"], model["id"], status))
                    failed[key] = False
                    break
            if pass_no == 0 and not any(failed.values()):
                break
        return error_response(client_api, 503, "no_free_model",
                              "every free model meeting this request refused it")


def requirements_from_body(client_api, body):
    if client_api == "anthropic":
        messages = body.get("messages") if isinstance(body.get("messages"), list) else []
        vision = False
        for msg in messages:
            if not _is_record(msg) or not isinstance(msg.get("content"), list):
                continue
            for block in msg["content"]:
                if _is_record(block) and block.get("type") == "image":
                    vision = True
        return {"tools": isinstance(body.get("tools"), list) and len(body["tools"]) > 0,
                "vision": vision, "json": False}
    messages = body.get("messages") if isinstance(body.get("messages"), list) else []
    vision = False
    for msg in messages:
        if not _is_record(msg) or not isinstance(msg.get("content"), list):
            continue
        for block in msg["content"]:
            if _is_record(block) and block.get("type") in ("image", "image_url"):
                vision = True
    fmt = body.get("response_format") if _is_record(body.get("response_format")) else {}
    return {"tools": isinstance(body.get("tools"), list) and len(body["tools"]) > 0,
            "vision": vision,
            "json": fmt.get("type") in ("json_object", "json_schema")}


def error_response(client_api, status, code, message):
    if client_api == "anthropic":
        body = {"type": "error", "error": {"type": code, "message": message}}
    else:
        body = {"error": {"message": message, "type": code, "code": None}}
    return status, "application/json", json.dumps(body, separators=(",", ":")).encode("utf-8")


# ------------------------------------------------------------------- server ----


AGG = None


def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *args):
        pass

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY_BYTES:
            return None
        raw = self.rfile.read(length)
        try:
            value = json.loads(raw.decode("utf-8"))
        except Exception:
            return None
        return value if _is_record(value) else None

    def _send(self, status, content_type, body):
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, client_api, status, code, message):
        s, ctype, body = error_response(client_api, status, code, message)
        self._send(s, ctype, body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/v1/models":
            models = served_models(AGG.probes, AGG.ratings)
            if isinstance(self.headers.get("x-api-key"), str):
                body = {"data": [{"type": "model", "id": m["id"], "display_name": m["display_name"],
                                  "created_at": FIXED_ANTHROPIC_CREATED_AT} for m in models],
                        "has_more": False}
            else:
                body = {"object": "list",
                        "data": [{"id": m["id"], "object": "model", "created": FIXED_CREATED,
                                  "owned_by": m["provider"]} for m in models]}
            self._send(200, "application/json", json.dumps(body, separators=(",", ":")).encode("utf-8"))
            return
        self._send_error("openai", 404, "not_found", "no route GET %s" % parsed.path)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/v1/chat/completions":
            body = self._read_json()
            if body is None:
                self._send_error("openai", 400, "invalid_request_error", "the request body is not one JSON object")
                return
            try:
                status, ctype, out = AGG.serve("openai", body)
            except Exception as exc:
                log("chat completions failed: %s" % exc)
                self._send_error("openai", 500, "internal_error", "the server failed to answer")
                return
            self._send(status, ctype, out)
            return
        if parsed.path == "/v1/messages":
            body = self._read_json()
            if body is None:
                self._send_error("anthropic", 400, "invalid_request_error", "the request body is not one JSON object")
                return
            try:
                status, ctype, out = AGG.serve("anthropic", body)
            except Exception as exc:
                log("messages failed: %s" % exc)
                self._send_error("anthropic", 500, "internal_error", "the server failed to answer")
                return
            self._send(status, ctype, out)
            return
        client_api = "anthropic" if parsed.path == "/v1/messages" else "openai"
        self._send_error(client_api, 404, "not_found", "no route POST %s" % parsed.path)


def load():
    directory, ratings, workload, error = read_brief(PROMPT_FILE)
    if error is not None:
        log(error)
    probes = [probe_provider(ENV_URL, p) for p in directory["providers"]]
    for probe in probes:
        if not probe["ok"] and probe["error"] is not None:
            log(probe["error"])
    catalog = build_catalog(probes, ratings)
    return Aggregator(directory, ratings, workload, probes, catalog)


def register(port, catalog_text):
    status, _headers, raw = http_call(
        "POST", ENV_URL + "/register",
        {"content-type": "application/json"},
        json.dumps({"protocol_version": PROTOCOL_VERSION, "port": port,
                    "catalog": json.loads(catalog_text)}, separators=(",", ":")).encode("utf-8"),
        PROBE_TIMEOUT_S)
    if status < 200 or status >= 300:
        raise RuntimeError("POST %s/register answered HTTP %s: %s" % (ENV_URL, status, raw[:500].decode("utf-8", "replace")))
    try:
        reply = json.loads(raw.decode("utf-8"))
    except Exception:
        reply = {}
    if _is_record(reply) and isinstance(reply.get("requests"), int):
        log("registered: the workload holds %d requests" % reply["requests"])


def poll_until_done(server):
    while True:
        time.sleep(POLL_INTERVAL_S)
        try:
            status, _headers, raw = http_call("GET", ENV_URL + "/work", {}, None, 2)
            if status != 200:
                continue
            work = json.loads(raw.decode("utf-8"))
        except Exception:
            continue
        if _is_record(work) and work.get("done") is True:
            server.shutdown()
            return


def main():
    global AGG
    AGG = load()
    if "--list" in sys.argv[1:]:
        # The same document the register body carries, byte for byte.
        sys.stdout.write(AGG.catalog_text + "\n")
        sys.stdout.flush()
        return 0
    if ENV_URL == "":
        raise RuntimeError("MAKEFUN_ENV_URL is not set")
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    server.daemon_threads = True
    port = server.server_address[1]
    if port == 8080:
        raise RuntimeError("the OS picked 8080, which the env owns on one layout: restart the trial")
    log("listening on 0.0.0.0:%d with %d answered providers and %d free models"
        % (port, sum(1 for p in AGG.probes if p["ok"]), len(served_models(AGG.probes, AGG.ratings))))
    register(port, AGG.catalog_text)
    poller = threading.Thread(target=poll_until_done, args=(server,), daemon=True)
    poller.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        log(str(exc))
        sys.exit(1)
