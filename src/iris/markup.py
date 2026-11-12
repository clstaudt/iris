"""Markup a model may write into its reply that is never to be said or shown:
its reasoning, and tool calls typed out as text instead of made. Small
models do the latter now and then, in the shape their training used:

- <tool_call>{"name": ..., "arguments": {...}}</tool_call>
- <function=name>{...}</function>, or <parameter=...> tags inside it
- <|tool_call|>[{"name": ..., "arguments": {...}}], <|python_tag|>{...}
- [TOOL_CALLS][{...}], or [TOOL_CALLS]name[ARGS]{...}
- a JSON object (or a list of them) that is plainly a call, as
  {"name": ..., "arguments": {...}}, alone or in a ``` fence
- what is left of such markup: a closing tag without its opening one
  (</tool_call>), a special token (<|im_end|>)

MarkupFilter takes the reply as it streams in and lets only the rest
through. Markup can be split across chunks, so text that may begin some is
held back until the chunks after it show whether it does: no part of it is
ever shown or spoken. The calls found are kept, so a caller can still run
one that is unambiguous.
"""

import json
import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class TypedCall:
    """A tool call written into a reply as text."""

    # "" when the markup did not say
    name: str
    # None when they did not parse
    arguments: Optional[dict]


# Tags dropped with all they enclose, and whether that is tool calls
PAIRED = {"think": False, "tool_call": True, "tool_calls": True}
# Closing tags left over, dropped alone
CLOSING = ("</think>", "</tool_call>", "</tool_calls>", "</function>", "</parameter>")
FUNCTION = "<function="
FUNCTION_TAG = re.compile(r"<function=([^>\s]{1,80})>")
FUNCTION_BLOCK = re.compile(r"<function=([^>\s]{1,80})>(.*?)(?:</function>|$)", re.DOTALL)
PARAMETER = re.compile(r"<parameter=([^>\s]{1,80})>(.*?)(?:</parameter>|$)", re.DOTALL)
# Special tokens after which tool calls follow
CALL_TOKENS = ("<|tool_call|>", "<|python_tag|>", "[TOOL_CALLS]")
SPECIAL_TOKEN = re.compile(r"<\|[A-Za-z_/]{1,30}\|>")
SPECIAL_TOKEN_START = re.compile(r"<\|[A-Za-z_/]{0,30}\|?")
NAMED_ARGS = re.compile(r"([\w.\-]{1,80})\[ARGS\]")
NAMED_ARGS_START = re.compile(r"[\w.\-]{0,80}(?:\[(?:A(?:R(?:G(?:S)?)?)?)?)?")
# The keys a JSON tool call opens with
CALL_KEYS = ('"name"', '"arguments"', '"parameters"', '"function"', '"type"')
CALL_FIELDS = {"name", "arguments", "parameters", "type", "id"}
# Fences whose content is a call; a plain or json fence is one only when a
# call follows
CALL_FENCES = ("tool_call", "tool_code")
FENCE_INFO_MAX = 20
# Where markup may begin
SPECIAL = re.compile(r"[<{\[`]")

# A verdict on the text held: None waits for more, ("text", n) or
# ("drop", n) say what its first n characters are
WAIT = None


def json_end(text: str, start: int = 0) -> int:
    """The index after the JSON object or list that opens at text[start], or
    -1 while it is not closed."""
    depth, in_string, escaped = 0, False, False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
            if depth == 0:
                return index + 1
    return -1


def _arguments(raw) -> Optional[dict]:
    """A call's arguments as given (an object, or JSON text of one)."""
    if raw is None:
        return {}
    if isinstance(raw, str):
        if not raw.strip():
            return {}
        try:
            raw = json.loads(raw)
        except ValueError:
            return None
    return raw if isinstance(raw, dict) else None


def _call_of(item, loose: bool) -> Optional[TypedCall]:
    """The call a JSON value stands for. Strictly, only an object with a
    name, arguments and nothing more is one; loosely (inside markup that
    says it holds a call), any object with a name."""
    if not isinstance(item, dict):
        return None
    if isinstance(item.get("function"), dict):
        if not loose and set(item) - {"type", "function", "id"}:
            return None
        item = item["function"]
    name = item.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    if not loose and ("arguments" not in item and "parameters" not in item):
        return None
    if not loose and set(item) - CALL_FIELDS:
        return None
    return TypedCall(name.strip(), _arguments(item.get("arguments", item.get("parameters"))))


def calls_in_json(text: str, loose: bool = False) -> Optional[list[TypedCall]]:
    """The calls a piece of JSON stands for, or None if it is not calls."""
    try:
        value = json.loads(text)
    except ValueError:
        return None
    items = value if isinstance(value, list) else [value]
    calls = [_call_of(item, loose) for item in items]
    if not calls or any(call is None for call in calls):
        return None
    return calls


def function_arguments(body: str) -> Optional[dict]:
    """The arguments inside <function=name>...</function>: JSON, or
    <parameter=key>value</parameter> tags, or nothing."""
    body = body.strip()
    if not body:
        return {}
    if body.startswith("{"):
        return _arguments(body)
    parameters = PARAMETER.findall(body)
    if parameters:
        return {key: value.strip() for key, value in parameters}
    return None


def calls_in_text(body: str) -> list[TypedCall]:
    """The calls inside markup that holds calls; one without a name when
    they cannot be read."""
    body = body.strip()
    if body[:1] in ("{", "["):
        end = json_end(body)
        calls = calls_in_json(body[:end], loose=True) if end > 0 else None
        if calls:
            return calls
    functions = FUNCTION_BLOCK.findall(body)
    if functions:
        return [TypedCall(name, function_arguments(inner)) for name, inner in functions]
    return [TypedCall("", None)]


def _call_start(text: str) -> Optional[bool]:
    """Whether the JSON object opening text may be a call (by its first
    key); None while that is not yet known."""
    rest = text[1:].lstrip()
    if not rest:
        return None
    if any(rest.startswith(key) for key in CALL_KEYS):
        return True
    if any(key.startswith(rest) for key in CALL_KEYS):
        return None
    return False


class MarkupFilter:
    """Lets a streamed reply through without its markup (see the module).

    feed() each chunk and send on what it returns, then flush() at the end.
    calls holds the tool calls found in the markup, in order.
    """

    def __init__(self):
        self.held = ""
        self.calls: list[TypedCall] = []

    def feed(self, text: str) -> str:
        self.held += text
        return self._release(final=False)

    def flush(self) -> str:
        """The text held back at the end; markup left open is dropped."""
        out = self._release(final=True)
        self.held = ""
        return out

    def _release(self, final: bool) -> str:
        out = []
        while self.held:
            found = SPECIAL.search(self.held)
            if found is None:
                out.append(self.held)
                self.held = ""
                break
            out.append(self.held[: found.start()])
            self.held = self.held[found.start() :]
            verdict = self._verdict(final)
            if verdict is WAIT:
                break
            kind, length = verdict
            if kind == "text":
                out.append(self.held[:length])
            self.held = self.held[length:]
        return "".join(out)

    def _verdict(self, final: bool):
        """What the held text, which opens with a character markup may
        begin with, starts with. At the end (final) there is no waiting."""
        first = self.held[0]
        if first == "<":
            verdict = self._angle(final)
        elif first == "`":
            verdict = self._fence(final)
        else:
            verdict = self._bracket(final)
        if verdict is WAIT and final:
            return ("text", 1)
        return verdict

    def _drop(self, end: int, calls: list[TypedCall]):
        self.calls += calls
        return ("drop", end)

    def _enclosed(self, opening: str, closing: str, final: bool):
        """(end, body) of markup from opening to closing; WAIT while it is
        open. Left open at the end, it runs to the end."""
        end = self.held.find(closing, len(opening))
        if end < 0:
            if not final:
                return WAIT
            return len(self.held), self.held[len(opening) :]
        return end + len(closing), self.held[len(opening) : end]

    def _angle(self, final: bool):
        held = self.held
        for tag, holds_calls in PAIRED.items():
            opening = f"<{tag}>"
            if held.startswith(opening):
                enclosed = self._enclosed(opening, f"</{tag}>", final)
                if enclosed is WAIT:
                    return WAIT
                end, body = enclosed
                return self._drop(end, calls_in_text(body) if holds_calls else [])
        function = FUNCTION_TAG.match(held)
        if function:
            enclosed = self._enclosed(function.group(0), "</function>", final)
            if enclosed is WAIT:
                return WAIT
            end, body = enclosed
            return self._drop(end, [TypedCall(function.group(1), function_arguments(body))])
        for token in CALL_TOKENS[:2]:
            if held.startswith(token):
                return self._after_token(token, final)
        special = SPECIAL_TOKEN.match(held)
        if special:
            return self._drop(special.end(), [])
        for closing in CLOSING:
            if held.startswith(closing):
                return self._drop(len(closing), [])
        openings = [f"<{tag}>" for tag in PAIRED] + list(CLOSING) + [FUNCTION]
        if any(opening.startswith(held) for opening in openings):
            return WAIT
        if held.startswith(FUNCTION) and ">" not in held and len(held) < len(FUNCTION) + 80:
            return WAIT  # the name is still coming
        if SPECIAL_TOKEN_START.fullmatch(held):
            return WAIT
        return ("text", 1)

    def _after_token(self, token: str, final: bool):
        """Calls after a special token: JSON, or name[ARGS]{...}. A token
        followed by neither is dropped alone."""
        held = self.held

        def left_open(call: TypedCall):
            # Calls still coming; at the end, dropped as they are
            return WAIT if not final else self._drop(len(held), [call])

        body = held[len(token) :].lstrip()
        start = len(held) - len(body)
        if not body:
            return WAIT
        if body[0] in "{[":
            end = json_end(held, start)
            if end < 0:
                return left_open(TypedCall("", None))
            calls = calls_in_json(held[start:end], loose=True)
            return self._drop(end, calls or [TypedCall("", None)])
        named = NAMED_ARGS.match(body)
        if named:
            name = named.group(1)
            rest = held[start + named.end() :].lstrip()
            if not rest:
                return left_open(TypedCall(name, None))
            if rest[0] != "{":
                return self._drop(start + named.end(), [TypedCall(name, None)])
            open_at = len(held) - len(rest)
            end = json_end(held, open_at)
            if end < 0:
                return left_open(TypedCall(name, None))
            return self._drop(end, [TypedCall(name, _arguments(held[open_at:end]))])
        if token == "[TOOL_CALLS]" and NAMED_ARGS_START.fullmatch(body):
            return WAIT
        return self._drop(len(token), [])

    def _json_call(self, start: int, final: bool):
        """The JSON at held[start], if it is a call: WAIT, ("call", end,
        calls), ("json", end) for JSON that is not one, or ("no",) when it
        does not open like one."""
        held = self.held
        opening = held[start:]
        if opening[0] == "[":
            inner = opening[1:].lstrip()
            if not inner:
                return WAIT
            if inner[0] != "{":
                return ("no",)
            maybe = _call_start(inner)
        else:
            maybe = _call_start(opening)
        if maybe is None:
            return WAIT
        if not maybe:
            return ("no",)
        end = json_end(held, start)
        if end < 0:
            # A call left open at the end is dropped
            return WAIT if not final else ("call", len(held), [TypedCall("", None)])
        calls = calls_in_json(held[start:end])
        return ("call", end, calls) if calls else ("json", end)

    def _bracket(self, final: bool):
        held = self.held
        if held.startswith(CALL_TOKENS[2]):
            return self._after_token(CALL_TOKENS[2], final)
        if CALL_TOKENS[2].startswith(held):
            return WAIT
        found = self._json_call(0, final)
        if found is WAIT:
            return WAIT
        if found[0] == "call":
            return self._drop(found[1], found[2])
        if found[0] == "json":
            return ("text", found[1])
        return ("text", 1)

    def _fence(self, final: bool):
        """A ``` fence holding a call: one of CALL_FENCES, or a plain or
        json one with a call in it. Other fences are text."""
        held = self.held
        if not held.startswith("```"):
            return WAIT if "```".startswith(held) else ("text", 1)
        line_end = held.find("\n", 3)
        if line_end < 0:
            return WAIT if len(held) <= 3 + FENCE_INFO_MAX else ("text", 3)
        info = held[3:line_end].strip().lower()
        if info in CALL_FENCES:
            enclosed = self.held.find("```", line_end)
            if enclosed < 0 and not final:
                return WAIT
            end = enclosed + 3 if enclosed >= 0 else len(held)
            body = held[line_end + 1 : enclosed if enclosed >= 0 else len(held)]
            return self._drop(end, calls_in_text(body))
        if info not in ("", "json"):
            return ("text", 3)
        body = held[line_end + 1 :].lstrip()
        if not body:
            return WAIT
        if body[0] not in "{[":
            return ("text", 3)
        found = self._json_call(len(held) - len(body), final)
        if found is WAIT:
            return WAIT
        if found[0] != "call":
            return ("text", 3)
        end = found[1]
        rest = held[end:].lstrip()
        if rest.startswith("```"):
            end = len(held) - len(rest) + 3
        elif "```".startswith(rest) and not final:
            return WAIT  # the closing fence may be coming
        return self._drop(end, found[2])


def strip_markup(text: str) -> str:
    """The text without its markup, trimmed."""
    markup = MarkupFilter()
    return (markup.feed(text) + markup.flush()).strip()
