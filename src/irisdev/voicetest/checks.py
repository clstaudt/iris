"""Checks the scenarios make of a turn beyond the usual ones.

Each function here makes a script.Check: a name (which
Conversation.expected_failures can list) and a function of the turn's
result and its View, returning a failure or None. Times are those of the
space (time.monotonic()); a View gives the turn's events, the scripted
model's requests, the scripted web's calls and the speaker's clips.
"""

import re
from typing import Optional

from irisdev.voicetest.script import Check
from irisdev.voicetest.scripted import GLUED, View, chime_of, flag_windows, state_timeline
from irisdev.voicetest.wording import contains, digits, heard_share, normalized, term_found

# Values in the reports are rounded; comparisons allow for it
SLACK = 0.02


# What a view holds, in absolute times ----------------------------------------------


def windows(view: View, flag: str) -> list[tuple[float, float]]:
    """When a flag was on during the turn."""
    return flag_windows(view.log.since(0, "flags", view.end), view.running.since, view.end).get(flag, [])


def tool_uses(view: View, start: Optional[float] = None) -> list[tuple[float, str, str]]:
    return [(at, d.get("tool", ""), d.get("detail", "")) for at, _, d in view.events("tool.use", start)]


def reply_clips(view: View, start: Optional[float] = None, end: Optional[float] = None):
    start = view.running.ref_end if start is None else start
    return [c for c in view.speaker.clips_between(start, view.end if end is None else end) if not chime_of(c)]


def first_sound(view: View, start: Optional[float] = None, end: Optional[float] = None) -> Optional[float]:
    found = view.runner._first_reply_sound(view.running.ref_end if start is None else start,
                                           view.end if end is None else end)
    return found


def dones(view: View, start: Optional[float] = None, end: Optional[float] = None) -> list[tuple[float, dict]]:
    return [(at, d) for at, _, d in view.events("chat.done", start, end)]


def users(view: View, source: Optional[str] = None, start: Optional[float] = None) -> list[tuple[float, dict]]:
    return [(at, d) for at, _, d in view.events("chat.user", start)
            if source is None or d.get("source", "voice") == source]


def actions(view: View, name: str, start: float = 0.0) -> list[float]:
    """When the runner did an action ("stop", "pause"...), since a moment."""
    return [at for at, _, _ in view.log.since(start, f"action.{name}", view.end)]


def states(view: View, start: float, end: float) -> list[tuple[float, str]]:
    return state_timeline(view.log.since(0, "state", end), start, end)


def turn_result(view: View, number: int):
    """An earlier turn's result in this conversation."""
    return next((r for r in view.runner.results_so_far if r.turn == number and not r.skipped), None)


def turn_start(view: View, number: int) -> Optional[float]:
    moments = view.runner._moments.get(number)
    return moments[0] if moments else None


def last_assistant(messages: list[dict]) -> str:
    found = [m.get("content") or "" for m in messages if m.get("role") == "assistant"]
    return found[-1] if found else ""


def check(name: str):
    """Make a function of (result, view) a Check of this name."""
    def wrap(fn):
        return Check(name, fn)
    return wrap


# Turns ---------------------------------------------------------------------------


def one_user(name: str = "one_turn") -> Check:
    """Exactly one chat.user of the turn's own kind (voice or text)."""
    @check(name)
    def fn(result, view):
        source = "text" if view.running.turn.typed else "voice"
        heard = users(view, source)
        return None if len(heard) == 1 else f"{len(heard)} turns heard: {[d.get('text') for _, d in heard]}"
    return fn


def no_continued(name: str = "continued") -> Check:
    @check(name)
    def fn(result, view):
        return f"{result.continued} chat.continued" if result.continued else None
    return fn


def nothing_played_while_speaking(name: str = "played_early", until: str = "end", margin: float = 0.0) -> Check:
    """No reply sound from the user's first word to their last (or to the
    start of a part given by its text)."""
    @check(name)
    def fn(result, view):
        said = view.running.said
        if said is None:
            return None
        end = said.speech_end + margin
        played = view.runner._reply_sounds(said.speech_start, end)
        if played:
            return f"the voice played {played[0][0] - said.speech_start:.2f} s into the user's turn"
        return None
    return fn


def nothing_played_between(name: str, start_of: int, end_of: int, margin: float = 0.3) -> Check:
    """No reply sound from the speech end of one turn to the speech end of
    another (plus a margin)."""
    @check(name)
    def fn(result, view):
        first = view.runner._moments.get(start_of)
        last = view.runner._moments.get(end_of)
        if first is None or last is None:
            return "the turns were not both held"
        played = view.runner._reply_sounds(first[1], last[1] + margin)
        return f"the voice played at {played[0][0] - first[1]:.2f} s" if played else None
    return fn


def silent_after_user_began(limit: float = 1.0, name: str = "barge_in") -> Check:
    @check(name)
    def fn(result, view):
        if result.barge_in is None:
            return "the voice did not stop when the user spoke" if view.runner._voice_playing(
                view.running.said.speech_start) else None
        return None if result.barge_in <= limit else f"silent {result.barge_in:.2f} s after the user began"
    return fn


def heard_at_least(share: float, of: Optional[str] = None, name: str = "heard") -> Check:
    """At least this share of the words heard (of a given text, or the turn's)."""
    @check(name)
    def fn(result, view):
        got = heard_share(result.heard, of or view.running.judged_text)
        if got >= share - 1e-9:
            return None
        return f"heard {got:.0%} of {of or view.running.judged_text!r}: {result.heard!r}"
    return fn


def first_word_after(limit: float, name: str = "first_word") -> Check:
    @check(name)
    def fn(result, view):
        if result.first_word is None:
            return "no first word"
        return None if result.first_word <= limit + SLACK else f"the first word came after {result.first_word:.2f} s"
    return fn


def first_word_within_turn(number: int, plus: float, name: str = "first_word") -> Check:
    """The first word no later than an earlier turn's, plus some seconds."""
    @check(name)
    def fn(result, view):
        base = turn_result(view, number)
        if base is None or base.first_word is None or result.first_word is None:
            return "no first word to compare"
        if result.first_word > base.first_word + plus + SLACK:
            return f"first word {result.first_word:.2f} s, against {base.first_word:.2f} s + {plus} s in turn {number}"
        return None
    return fn


def reply_not_interrupted(name: str = "interrupted") -> Check:
    @check(name)
    def fn(result, view):
        cut = [d.get("text", "")[:60] for _, d in dones(view) if d.get("interrupted")]
        return f"the reply was cut: {cut}" if cut else None
    return fn


def history_holds_reply(text: str, name: str = "history") -> Check:
    """The history's last assistant message is this whole reply."""
    @check(name)
    def fn(result, view):
        kept = last_assistant(view.history())
        return None if kept.strip() == text.strip() else f"the history keeps {kept[:80]!r}"
    return fn


def history_excludes(terms: list[str], name: str = "history") -> Check:
    @check(name)
    def fn(result, view):
        found = [t for t in terms if any(t.lower() in (m.get("content") or "").lower() for m in view.history())]
        return f"the history holds {found}" if found else None
    return fn


def history_alternates(name: str = "history") -> Check:
    @check(name)
    def fn(result, view):
        roles = [m.get("role") for m in view.history() if m.get("role") in ("user", "assistant")]
        bad = any(a == b for a, b in zip(roles, roles[1:]))
        empty = any(m.get("role") == "assistant" and not (m.get("content") or "").strip() for m in view.history())
        if bad:
            return f"the history does not alternate: {roles}"
        return "an empty assistant message in the history" if empty else None
    return fn


def last_assistant_is_reply(name: str = "history") -> Check:
    """The history's last assistant message is the turn's final reply."""
    @check(name)
    def fn(result, view):
        complete = [d.get("text", "") for _, d in dones(view) if not d.get("interrupted")]
        if not complete:
            return "no complete reply"
        kept = last_assistant(view.history())
        return None if normalized(kept) == normalized(complete[-1]) else f"the history keeps {kept[:80]!r}"
    return fn


def kept_as_prefix(of_turn: int, name: str = "kept_prefix") -> Check:
    """An earlier turn's cut reply is kept as a word-prefix of what it would
    have said, no longer than what was heard allows (4.5 words a second + 3)."""
    @check(name)
    def fn(result, view):
        earlier = turn_result(view, of_turn)
        if earlier is None:
            return "the earlier turn was not held"
        start = turn_start(view, of_turn)
        cut = [d for _, d in dones(view, start) if d.get("interrupted")]
        if not cut:
            return f"turn {of_turn}'s reply was not cut"
        full = normalized(cut[0].get("text", ""))
        history = view.history()
        said = normalized(earlier.heard or earlier.said)
        kept = None
        for index, message in enumerate(history):
            if message.get("role") == "user" and normalized(message.get("content", "")) == said:
                following = history[index + 1] if index + 1 < len(history) else {}
                kept = following.get("content") if following.get("role") == "assistant" else ""
        if kept is None:
            return f"turn {of_turn} is not in the history"
        kept_words = normalized(kept).split()
        if not kept_words:
            return f"turn {of_turn}'s reply was kept empty"
        if not full.startswith(" ".join(kept_words)):
            return f"kept {kept[:80]!r}, not a prefix of what it was saying"
        audible = earlier.audible or 0.0
        if len(kept_words) > 4.5 * audible + 3:
            return f"kept {len(kept_words)} words of a reply heard for {audible:.1f} s"
        return None
    return fn


# Requests to the scripted model -------------------------------------------------------


def requests_count(n: int, name: str = "model_requests") -> Check:
    @check(name)
    def fn(result, view):
        found = [r for r in view.requests() if r.kind != "summary"]
        return None if len(found) == n else f"{len(found)} requests to the model, expected {n}"
    return fn


def second_request_has_result(text: str, name: str = "tool_result") -> Check:
    """The turn's second request carries the tool message with this text."""
    @check(name)
    def fn(result, view):
        found = [r for r in view.requests() if r.kind != "summary"]
        if len(found) < 2:
            return f"{len(found)} requests to the model"
        tools = found[1].tool_messages()
        return None if any(text.lower() in t.lower() for t in tools) else f"the second request's tool messages: {tools}"
    return fn


def tool_message(starts: Optional[str] = None, equals: Optional[str] = None, name: str = "tool_message") -> Check:
    """The tool message the model received starts with, or equals, a text."""
    @check(name)
    def fn(result, view):
        found = [t for r in view.requests() for t in r.tool_messages()]
        if not found:
            return "the model received no tool message"
        last = found[-1]
        if starts is not None and not last.startswith(starts):
            return f"the tool message is {last[:80]!r}"
        if equals is not None and last != equals:
            return f"the tool message is {last[:80]!r}"
        return None
    return fn


def no_request_after(action: str, carrying: list[str], name: str = "stale_request",
                     tool_messages: bool = True) -> Check:
    """No request made after an action (or a turn's start, "T<n>") carries a
    tool message or any of these texts."""
    @check(name)
    def fn(result, view):
        if view.stub is None:
            return None
        if action.startswith("T"):
            moment = turn_start(view, int(action[1:]))
        else:
            done = actions(view, action)
            moment = done[0] if done else None
        if moment is None:
            return None
        stale = []
        for request in view.stub.requests_between(moment, view.end):
            if tool_messages and request.tool_messages():
                stale.append(f"request {request.index} carries a tool result")
            stale += [f"request {request.index} carries {t!r}" for t in carrying if request.carries(t)]
        return "; ".join(stale) or None
    return fn


def request_holds(texts: list[str], name: str = "request", absent: Optional[list[str]] = None) -> Check:
    """The turn's first request holds all of these texts, and none of absent."""
    @check(name)
    def fn(result, view):
        found = view.requests()
        if not found:
            return "no request to the model"
        missing = [t for t in texts if not found[0].carries(t)]
        extra = [t for t in absent or [] if found[0].carries(t)]
        problems = ([f"it lacks {missing}"] if missing else []) + ([f"it holds {extra}"] if extra else [])
        return "; ".join(problems) or None
    return fn


def one_user_message_with(texts: list[str], name: str = "merged_request") -> Check:
    """The turn's last request has one user message holding all the texts."""
    @check(name)
    def fn(result, view):
        found = [r for r in view.requests() if r.kind == "reply"]
        if not found:
            return "no request to the model"
        last = found[-1].users()[-1] if found[-1].users() else ""
        missing = [t for t in texts if t.lower() not in last.lower()]
        return f"the last user message {last!r} lacks {missing}" if missing else None
    return fn


def assistant_before_last_user(text: str, name: str = "history") -> Check:
    """In the turn's request, the assistant message before the user's last
    one is this text."""
    @check(name)
    def fn(result, view):
        found = view.requests()
        if not found:
            return "no request to the model"
        messages = found[0].messages
        indexes = [i for i, m in enumerate(messages) if m.get("role") == "user"]
        if not indexes:
            return "no user message"
        before = [m for m in messages[: indexes[-1]] if m.get("role") == "assistant"]
        got = before[-1].get("content", "") if before else ""
        return None if normalized(got) == normalized(text) else f"the assistant message before it is {got[:80]!r}"
    return fn


# Searching ---------------------------------------------------------------------------


def one_search(term: str, name: str = "one_search") -> Check:
    @check(name)
    def fn(result, view):
        heard = users(view)
        after = heard[-1][0] if heard else view.running.since
        found = [(t, d) for at, t, d in tool_uses(view) if at >= after]
        if len(found) != 1 or found[0][0] != "search":
            return f"tools used after the turn was heard: {found}"
        return None if term_found(found[0][1], term) else f"searched for {found[0][1]!r}"
    return fn


def search_window_at_least(seconds: float, name: str = "search_window") -> Check:
    @check(name)
    def fn(result, view):
        spans = windows(view, "searching")
        if not spans:
            return "never searching"
        length = spans[0][1] - spans[0][0]
        return None if length >= seconds else f"searching for {length:.2f} s"
    return fn


def lead_in_while_searching(name: str = "lead_in") -> Check:
    """The lead-in is heard while the search runs, as a clip of its own, and
    the answer comes once the search is over."""
    @check(name)
    def fn(result, view):
        spans = windows(view, "searching")
        if not spans:
            return "never searching"
        on, off = spans[0]
        first = first_sound(view)
        if first is None:
            return "nothing was said"
        problems = []
        if not (first < off and first <= on + 1.0):
            problems.append(f"the first sound came {first - on:+.2f} s from the search's start "
                            f"(it ran {off - on:.2f} s)")
        clips = reply_clips(view)
        lead = [c for c in clips if c.started < off]
        answer = [c for c in clips if c.started >= off]
        if not lead:
            problems.append("no clip of its own for the lead-in")
        if not answer:
            problems.append("no answer after the search")
        if lead and answer:
            lead_end = max(c.started + (c.sounds()[-1][1] - c.started if c.sounds() else 0) for c in lead)
            answer_start = view.runner._first_reply_sound(off, view.end) or answer[0].started
            if answer_start - lead_end < 0.3:
                problems.append(f"only {answer_start - lead_end:.2f} s between the lead-in and the answer")
            if not off - SLACK <= answer_start <= off + 1.5:
                problems.append(f"the answer began {answer_start - off:+.2f} s after the search ended")
        return "; ".join(problems) or None
    return fn


def no_glued_sentences(name: str = "glued") -> Check:
    @check(name)
    def fn(result, view):
        texts = [d.get("text", "") for _, d in dones(view)] + [last_assistant(view.history())]
        glued = [GLUED.search(t).group(0) for t in texts if GLUED.search(t)]
        return f"sentences run together: {glued}" if glued else None
    return fn


def searching_after_lead_in(name: str = "searching_state") -> Check:
    """From the lead-in's end until the search ends, the state is searching."""
    @check(name)
    def fn(result, view):
        spans = windows(view, "searching")
        if not spans:
            return "never searching"
        on, off = spans[0]
        lead = [c for c in reply_clips(view) if c.started < off and c.sounds()]
        start = max([c.sounds()[-1][1] for c in lead] + [on]) + 0.7  # the state's hold over speech
        if start >= off:
            return None
        wrong = [(round(at - on, 2), s) for at, s in states(view, start, off) if s in ("listening", "idle")]
        return f"the state while searching: {wrong}" if wrong else None
    return fn


def flags_follow_search(name: str = "search_flag") -> Check:
    """The searching flag rises within 0.4 s of the model's tool call, and
    falls within 0.1 s of the web's answer."""
    @check(name)
    def fn(result, view):
        spans = windows(view, "searching")
        sent = [r.tool_call_sent for r in view.requests() if r.tool_call_sent]
        calls = view.web.between(view.running.since, view.end) if view.web else []
        if not spans or not sent or not calls:
            return "no search to follow"
        on, off = spans[0]
        problems = []
        if on - sent[0] > 0.4:
            problems.append(f"the flag rose {on - sent[0]:.2f} s after the tool call")
        if calls[0].end is not None and off - calls[0].end > 0.1:
            problems.append(f"the flag fell {off - calls[0].end:.2f} s after the web answered")
        return "; ".join(problems) or None
    return fn


def search_added(baseline: int, limit: float, name: str = "search_added") -> Check:
    """What the search added to the first word, over a turn without one."""
    @check(name)
    def fn(result, view):
        base = turn_result(view, baseline)
        if base is None or base.first_word is None or result.first_word is None:
            return "no first word to compare"
        added = result.first_word - base.first_word
        return None if added <= limit + SLACK else f"the search added {added:.2f} s"
    return fn


def reply_stage_at_least(seconds: float, name: str = "reply_stage") -> Check:
    @check(name)
    def fn(result, view):
        reply = result.stages.get("reply")
        return None if reply is not None and reply / 1000 >= seconds else f"the model stage took {reply} ms"
    return fn


def web_queries(count: int, first_without: Optional[str] = None, name: str = "web_queries") -> Check:
    @check(name)
    def fn(result, view):
        start = turn_start(view, 1) or view.running.since
        calls = [c for c in (view.web.searches() if view.web else []) if c.start >= start]
        if len(calls) != count:
            return f"{len(calls)} searches: {[c.query for c in calls]}"
        if first_without and term_found(calls[0].query, first_without):
            return f"the first search was for {calls[0].query!r}"
        return None
    return fn


def last_search_mentions(term: str, name: str = "search_mentions") -> Check:
    @check(name)
    def fn(result, view):
        found = tool_uses(view, turn_start(view, 1))
        if not found:
            return "no search"
        return None if term_found(found[-1][2], term) else f"the last search was for {found[-1][2]!r}"
    return fn


def state_held_during_search(state: str = "searching", name: str = "searching_state",
                             forbid: tuple = ("listening", "idle", "thinking")) -> Check:
    """While each web call ran, the state stayed this one (allowing for
    speech over it)."""
    @check(name)
    def fn(result, view):
        if view.web is None:
            return None
        problems = []
        for call in view.web.between(view.running.since, view.end):
            if call.end is None:
                continue
            seen = [(round(at - call.start, 2), s) for at, s in states(view, call.start + 0.3, call.end)
                    if s in forbid]
            if seen:
                problems.append(f"while {call.query!r} ran: {seen}")
        return "; ".join(problems) or None
    return fn


def answer_after_web(limit: float, name: str = "answer_time") -> Check:
    """The answer's first word within so many seconds of the web's answer."""
    @check(name)
    def fn(result, view):
        calls = view.web.between(view.running.since, view.end) if view.web else []
        if not calls or calls[-1].end is None:
            return "no web call"
        answer = first_sound(view, calls[-1].end)
        if answer is None:
            return "nothing said after the web answered"
        return None if answer - calls[-1].end <= limit else f"the answer came {answer - calls[-1].end:.2f} s after"
    return fn


def stopped_during_search(name: str = "search_stopped") -> Check:
    """After Stop during a search: the reply is given up at once, and Iris
    leaves the searching state within a second."""
    @check(name)
    def fn(result, view):
        pressed = actions(view, "stop", view.running.since)
        if not pressed:
            return "Stop was not pressed"
        stop = pressed[0]
        problems = []
        done = [at for at, d in dones(view, stop) if d.get("interrupted")]
        if not done or done[0] - stop > 0.5:
            problems.append("no interrupted chat.done within 0.5 s of Stop")
        later = view.log.since(stop + 1.0, "state", view.end)
        first_state = view.log.last_before(stop + 1.0, "state")
        still = (first_state and first_state[2].get("state") == "searching") or any(
            d.get("state") == "searching" for _, _, d in later)
        if still:
            problems.append("still searching 1 s after Stop")
        on = windows(view, "searching")
        if on and (on[-1][1] - stop > 1.0):
            problems.append(f"the searching flag stayed on {on[-1][1] - stop:.1f} s after Stop")
        return "; ".join(problems) or None
    return fn


def no_tool_use_after(action: str, name: str = "tool_after_stop") -> Check:
    @check(name)
    def fn(result, view):
        done = actions(view, action)
        if not done:
            return None
        found = view.log.since(done[0], "tool.use", view.end)
        return f"{len(found)} tool uses after {action}" if found else None
    return fn


def no_searching_after_done(name: str = "searching_after") -> Check:
    @check(name)
    def fn(result, view):
        complete = [at for at, d in dones(view) if not d.get("interrupted")]
        if not complete:
            return None
        found = [d for _, _, d in view.log.since(complete[-1], "state", view.end) if d.get("state") == "searching"]
        return "searching after the final reply" if found else None
    return fn


def threads_back(number: int, slack: int = 1, name: str = "threads") -> Check:
    """As many threads as before an earlier turn, give or take one."""
    @check(name)
    def fn(result, view):
        before = view.runner._threads_before.get(number)
        if before is None:
            return None
        now = len(result.threads)
        return None if abs(now - before) <= slack else f"{now} threads, {before} before turn {number}"
    return fn


def spoken_mentions(terms: list[str], name: str = "spoken") -> Check:
    """The final complete reply mentions one of the terms."""
    @check(name)
    def fn(result, view):
        complete = [d.get("text", "") for _, d in dones(view) if not d.get("interrupted")]
        if not complete:
            return "no complete reply"
        return None if any(t.lower() in complete[-1].lower() for t in terms) else f"the reply: {complete[-1][:100]!r}"
    return fn


def no_complete_reply_mentions(terms: list[str], name: str = "excludes") -> Check:
    @check(name)
    def fn(result, view):
        bad = [d.get("text", "")[:80] for _, d in dones(view, turn_start(view, 1))
               if not d.get("interrupted") and any(t.lower() in d.get("text", "").lower() for t in terms)]
        return f"replies mention {terms}: {bad}" if bad else None
    return fn


def no_time_of_day(name: str = "time_of_day") -> Check:
    """The reply gives no time of day (a guess where the search failed)."""
    @check(name)
    def fn(result, view):
        text = result.reply
        if re.search(r"\b\d{1,2}:\d\d\b", text) or re.search(r"\b\d{1,2}\s*(?:am|pm|a\.m\.|p\.m\.)", text, re.I):
            return f"the reply gives a time: {text[:100]!r}"
        return None
    return fn


def silent_wait(limit: float = 8.0, name: str = "silent_wait") -> Check:
    """A long wait with nothing said is reported (Iris has no "still
    looking" cue), not failed."""
    @check(name)
    def fn(result, view):
        if result.busy_silence and result.busy_silence > limit and result.first_word and result.first_word > limit:
            view.runner.report(f"  note: {result.conversation} · turn {result.turn}: silent wait of "
                               f"{result.busy_silence:.1f} s")
        return None
    return fn


def listening_after(seconds: float = 2.0, name: str = "listening") -> Check:
    """Listening again within so many seconds of the reply's last sound."""
    @check(name)
    def fn(result, view):
        last = view.speaker.last_sound(view.running.ref_end, view.end)
        if last is None:
            return None
        found = [at for at, _, d in view.log.since(last, "state", view.end) if d.get("state") == "listening"]
        current = view.log.last_before(last + 0.01, "state")
        if current and current[2].get("state") == "listening":
            return None
        return None if found and found[0] - last <= seconds else "not listening after the reply"
    return fn


# What tool markup written as text leaves in a reply (see iris.markup)
MARKUP = ("<tool_call", "</tool_call", "<function", "</function", "<|", "[TOOL_CALLS]", "arguments")


def no_markup(name: str = "markup") -> Check:
    """No tool markup in the reply, as shown while it streams in or once
    complete."""
    @check(name)
    def fn(result, view):
        streamed = "".join(d.get("text", "") for _, _, d in view.events("chat.chunk"))
        texts = [d.get("text", "") for _, d in dones(view)] + [streamed]
        bad = [t[:80] for t in texts if any(mark in t for mark in MARKUP)]
        return f"tool markup in the reply: {bad}" if bad else None
    return fn


def no_tool_offer(name: str = "tools_offered") -> Check:
    """The turn's requests offer no tools, and the system prompt has no
    search guidance."""
    @check(name)
    def fn(result, view):
        problems = []
        for request in view.requests():
            # ollama's client sends an empty list when no tools are offered
            if request.tools:
                problems.append(f"request {request.index} offers {request.tools}")
            if "'search' function" in request.system:
                problems.append(f"request {request.index}'s prompt holds the search guidance")
        return "; ".join(problems) or None
    return fn


def offers(names: list[str], name: str = "tools_offered") -> Check:
    @check(name)
    def fn(result, view):
        found = view.requests()
        if not found:
            return "no request"
        missing = [n for n in names if n not in found[0].tools]
        return f"the request lacks {missing}" if missing else None
    return fn


def no_searching_flags(name: str = "no_search") -> Check:
    @check(name)
    def fn(result, view):
        if windows(view, "searching") or windows(view, "browsing"):
            return "searching or browsing"
        bad = [d.get("state") for _, _, d in view.events("state") if d.get("state") in ("searching", "browsing")]
        return f"states {bad}" if bad else None
    return fn


def mentions_in_order(tools: list[str], name: str = "tool_order") -> Check:
    @check(name)
    def fn(result, view):
        used = [t for _, t, _ in tool_uses(view)]
        return None if used == tools else f"tools used: {used}"
    return fn


def browse_of_search_href(name: str = "browse_href") -> Check:
    """The page read is the first address the search found."""
    @check(name)
    def fn(result, view):
        uses = tool_uses(view)
        searches = view.web.searches() if view.web else []
        browse = [d for _, t, d in uses if t == "browse"]
        found = [r for c in searches if c.start >= view.running.since for r in (c.result or [])
                 if isinstance(c.result, list)]
        if not browse or not found:
            return "no search followed by a page"
        return None if browse[0] == found[0].get("href") else f"read {browse[0]!r}, found {found[0].get('href')!r}"
    return fn


def flags_in_sequence(first: str, second: str, name: str = "flag_order") -> Check:
    @check(name)
    def fn(result, view):
        a, b = windows(view, first), windows(view, second)
        if not a or not b:
            return f"{first}: {a}, {second}: {b}"
        if a[0][1] > b[0][0] + SLACK:
            return f"{second} began before {first} ended"
        left_on = [f for f in (first, second) if windows(view, f) and windows(view, f)[-1][1] >= view.end]
        return f"left on: {left_on}" if left_on else None
    return fn


def last_request_note(text: str, name: str = "last_round") -> Check:
    @check(name)
    def fn(result, view):
        found = [r for r in view.requests() if r.kind != "summary"]
        if not found:
            return "no request"
        last = found[-1]
        problems = []
        if last.tools:
            problems.append(f"the last request offers {last.tools}")
        if not last.carries(text):
            problems.append(f"the last request lacks {text!r}")
        return "; ".join(problems) or None
    return fn


def uses_count(n: int, name: str = "tool_count") -> Check:
    @check(name)
    def fn(result, view):
        found = tool_uses(view)
        return None if len(found) == n else f"{len(found)} tool uses"
    return fn


def query_words_known(name: str = "query_words") -> Check:
    """The query holds no word that is neither said in the turn nor already
    in the conversation (a TV's words, say)."""
    @check(name)
    def fn(result, view):
        known = set(normalized(" ".join(m.get("content") or "" for m in view.history())).split())
        known |= set(normalized(view.running.result.said).split())
        problems = []
        for _, _, detail in tool_uses(view):
            strange = [w for w in normalized(detail).split() if w not in known]
            if strange:
                problems.append(f"{detail!r} has {strange}")
        return "; ".join(problems) or None
    return fn


def word_in_order_or_whole(letters: list[str], whole: str, name: str = "spelled") -> Check:
    @check(name)
    def fn(result, view):
        heard = normalized(result.heard).split()
        if whole in heard:
            return None
        position = 0
        for letter in letters:
            try:
                position = heard.index(letter, position) + 1
            except ValueError:
                return f"the letters {letters} are not heard in order: {result.heard!r}"
        return None
    return fn


def digits_heard(expected: str, name: str = "digits") -> Check:
    @check(name)
    def fn(result, view):
        from_heard = digits(result.heard)
        return None if from_heard == expected else f"heard the digits {from_heard!r}"
    return fn


def no_user_for(seconds: float, name: str = "no_turn") -> Check:
    """No chat.user from the turn's speech end for so many seconds."""
    @check(name)
    def fn(result, view):
        found = view.log.since(view.running.ref_end, "chat.user", view.running.ref_end + seconds)
        return f"took it for a turn: {[d.get('text') for _, _, d in found]}" if found else None
    return fn


def heard_contains(terms: list[str], name: str = "heard_terms") -> Check:
    @check(name)
    def fn(result, view):
        missing = [t for t in terms if not contains(result.heard, t)]
        return f"did not hear {missing}" if missing else None
    return fn
