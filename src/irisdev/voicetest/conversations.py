"""The conversations the voice test holds with Iris.

Each is a sequence of turns, said by the user into the virtual room. A turn
says when the user speaks (after Iris has finished, while it talks, while
it searches, after pressing Stop), what it says (pauses included) and what
is expected: that Iris heard it as one turn, what its reply mentions, which
tools it used, and that it answered or stayed silent (see script.py).

CONVERSATIONS are the everyday ones. GROUPS holds them all by name, one
group per thing tested: "search" (web search), "turn-taking",
"interruptions", "hearing", "conversation", "performance", "model" (a
model server in trouble), and "slow" and "network" for the long runs and
the real web.
"""

import re

from iris.llm_client import READ_TIMEOUT
from irisdev.voicetest import checks as c
from irisdev.voicetest.fakeweb import Each
from irisdev.voicetest.script import (
    At,
    Check,
    Conversation,
    Echo,
    Noise,
    Say,
    Silence,
    Sound,
    Turn,
)
from irisdev.voicetest.scripted import holds, inner_silences
from irisdev.voicetest.stub_llm import (
    LONG,
    SHORT,
    Call,
    Rule,
    browse,
    fill,
    last_tool,
    last_user,
    notes_of,
    results,
    search,
    user_messages,
)
from irisdev.voicetest.wording import normalized

__all__ = ["CONVERSATIONS", "Conversation", "GROUPS", "Turn", "all_conversations", "group"]

# The everyday conversations ----------------------------------------------------------

CONVERSATIONS = [
    Conversation(
        "a question and a follow-up",
        [
            Turn(["What is the capital of France?"], reply_mentions=["Paris"]),
            Turn(["And roughly how many people live there?"], reply_mentions=["million"]),
        ],
    ),
    Conversation(
        "answering at once",
        [
            Turn(["What is the capital of France?"], reply_mentions=["Paris"]),
            # A brisk "Yes", a third of a second
            Turn(["Yes."], answer_after=0.3, speed=1.8),
            Turn(["And how many people live there?"], reply_mentions=["million"], answer_after=0.4),
        ],
    ),
    Conversation(
        "a pause in the middle of a sentence",
        [
            Turn(
                ["I'd like to know", 0.8, "how tall the Eiffel Tower is."],
                reply_mentions=["meter", "metre", "feet", "330", "324", "300"],
            ),
        ],
    ),
    Conversation(
        "thinking aloud",
        [
            Turn(
                ["Can you tell me about", 1.2, "the first landing on the moon?"],
                reply_mentions=["1969", "Apollo", "Armstrong"],
            ),
        ],
    ),
    Conversation(
        "a sentence, then a question",
        [
            Turn(
                ["I'm cooking pasta tonight.", 0.6, "How long should I boil spaghetti?"],
                reply_mentions=["minute"],
            ),
        ],
    ),
    Conversation(
        "interrupting a long answer",
        [
            Turn(["Tell me a long story about a lighthouse keeper."]),
            Turn(["Sorry, stop. What is two plus two?"], interrupt_after=3.0, reply_mentions=["four", "4"]),
        ],
    ),
    Conversation(
        "listening to a long answer to its end",
        [
            Turn(["Explain in detail how a rainbow forms."], listen_through=True),
        ],
    ),
    Conversation(
        "the stop button",
        [
            Turn(["Tell me a long story about the sea."], stop_after=2.0),
            Turn(["What day comes after Monday?"], reply_mentions=["Tuesday"]),
        ],
    ),
    Conversation(
        "the wake word",
        [
            Turn(["What's the weather like today?"], no_reply=True),
            Turn(["Iris, what is the capital of Italy?"], reply_mentions=["Rome"]),
            Turn(["And of Spain?"], reply_mentions=["Madrid"]),
        ],
        wake_word="Iris",
    ),
    Conversation(
        "quick questions back to back",
        [
            Turn(["What color is the sky on a clear day?"], reply_mentions=["blue"]),
            Turn(["How many legs does a spider have?"], reply_mentions=["eight", "8"]),
            Turn(["What is the opposite of hot?"], reply_mentions=["cold"]),
        ],
    ),
]


# What the scripted model reads in a conversation ---------------------------------------


def latest(pattern_by_name: dict, text: str):
    """The name whose pattern is found latest in the text, or None."""
    best, where = None, -1
    for name, pattern in pattern_by_name.items():
        found = [m.start() for m in re.finditer(pattern, text or "", re.IGNORECASE)]
        if found and found[-1] > where:
            best, where = name, found[-1]
    return best


def assistant_before_last_user(messages: list[dict]) -> str:
    users = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if not users:
        return ""
    before = [m.get("content") or "" for m in messages[: users[-1]] if m.get("role") == "assistant"]
    return before[-1] if before else ""


def asked(reply: str) -> bool:
    """Iris ended its reply with a question."""
    return reply.rstrip().endswith("?")


def quick(**more) -> dict:
    """What a one-word answer said at once is expected to be."""
    return dict(exact=True, turn_end=(0, 0.8), deaf_limit=0.05, first_word=4.0,
                checks=[c.one_user()], **more)


# Answering Iris's question at once, in one word -------------------------------------------

DINNER_RULES = [
    Rule("plan dinner", "Happy to help. Should it be vegetarian?"),
    Rule(r"^\W*yes\b", "Great. How many people are eating?"),
    Rule(r"^\W*(four|4)\b", "Got it. Any allergies I should know about?"),
    Rule(r"^\W*no\b", "Then a mushroom risotto would be perfect."),
    Rule(r"\b(six|6)\b", "Six it is. I will scale the recipe."),
]


def answering_at_once(shift: float = 0.0) -> Conversation:
    """One-word answers said the moment Iris has asked, the gaps shifted."""
    name = "answering at once" + (f" (+{shift:g} s)" if shift else "")
    return Conversation(
        name,
        [
            Turn(["Can you help me plan dinner?"], deaf_limit=0.05, checks=[c.one_user()],
                 real_parts=["Ask me one short yes-or-no question at a time to plan dinner."]),
            Turn(["Yes."], at=At.reply_end(0.1 + shift), when=asked, **quick()),
            Turn(["Four."], at=At.reply_end(0.25 + shift), when=asked, **quick()),
            Turn(["No."], at=At.reply_end(0.0 + shift), when=asked, **quick()),
            Turn(["Actually, make it for six people."], at=At.reply_end(-0.2 + shift),
                 must_hear=["actually", "six"], deaf_limit=0.05, first_word=4.0,
                 checks=[c.one_user(), c.silent_after_user_began(1.0)]),
        ],
        stub_rules=DINNER_RULES,
        tags=["turn-taking", "quick", f"shift {shift:g}"],
    )


def springfield_state(messages: list[dict]) -> str:
    found = re.search(r"illinois|missouri", last_user(messages), re.IGNORECASE)
    return found.group(0).title() if found else ""


def answer_by(limit: float, name: str = "answer_time") -> Check:
    """The answer after the search starts within so many seconds of the
    user's speech end."""
    def fn(result, view):
        spans = c.windows(view, "searching")
        if not spans:
            return "never searching"
        answer = c.first_sound(view, spans[0][1])
        if answer is None:
            return "no answer after the search"
        took = answer - view.running.ref_end
        return None if took <= limit else f"the answer began {took:.2f} s after the user stopped"
    return Check(name, fn)


SPRINGFIELD = Conversation(
    "a one-word answer starts a search",
    [
        Turn(["What's the weather in Springfield?"], deaf_limit=0.05, checks=[c.one_user()]),
        Turn(["Illinois."], at=At.reply_end(0.3), when=asked, search_mentions=["springfield", "illinois"],
             reply_mentions=["12"], check_with_stub=True,
             checks=[c.one_user(), c.uses_count(1), answer_by(4.0 + 1.5 + 1.0)], **{
                 k: v for k, v in quick().items() if k != "checks"}),
    ],
    tools={"web": True},
    web={"results": {r"springfield.*illinois|illinois.*springfield": "Springfield, Illinois: windy, 12 degrees."},
         "delay": 1.5},
    stub_rules=[
        Rule("weather in Springfield", "There are a few. Do you mean Springfield in Illinois or in Missouri?"),
        Rule(lambda m: bool(springfield_state(m)) and "Do you mean" in assistant_before_last_user(m),
             lead_in=lambda m: f"Let me check Springfield, {springfield_state(m)}.\n\n",
             tool=search(lambda m: f"weather Springfield {springfield_state(m)}"),
             after_tool="Here is what I found. {first}"),
    ],
    tags=["turn-taking", "search", "quick"],
)


# Going on after a pause --------------------------------------------------------------

AUSTRALIA = "What's the capital of Australia?"
CANBERRA = Rule("australia", "Canberra is the capital of Australia. It was built as the capital.")


def fragment_not_kept(earlier_users: int, name: str = "fragment") -> Check:
    """No exchange is kept for a fragment the user went on from, unless some
    of its reply was heard."""
    def fn(result, view):
        requests = [r for r in view.requests() if r.kind == "reply"]
        if not requests:
            return None
        count = len(requests[-1].users())
        heard_before = view.runner._reply_sounds(view.running.said.speech_start, view.running.said.speech_end)
        allowed = earlier_users + 1 + (1 if heard_before else 0)
        return None if count <= allowed else f"{count} user messages in the request, {allowed} at most"
    return Check(name, fn)


def report_verdicts(name: str = "verdicts") -> Check:
    """Smart Turn's verdicts in the turn, reported; a pause none of whose
    verdicts said finished did not exercise the turn's end."""
    def fn(result, view):
        if result.verdicts:
            ended = [p for _, p in result.verdicts if p is not None and p > 0.5]
            view.runner.report(f"  note: {result.conversation} · turn {result.turn}: Smart Turn's verdicts "
                               f"{[p for _, p in result.verdicts]}" + ("" if ended else " (not exercised)"))
        return None
    return Check(name, fn)


def going_on_after(pause: float) -> Conversation:
    """The question with a pause before its last word, after a control."""
    turn_checks = [report_verdicts(), fragment_not_kept(1)]
    if pause <= 1.8:
        turn_checks += [c.one_user(), c.no_continued(), c.nothing_played_while_speaking(),
                        c.first_word_within_turn(1, 0.5)]
        expect, allow = "answer", 0
    elif pause <= 2.6:
        turn_checks += [nothing_before_last_part(), c.first_word_within_turn(1, 0.5)]
        expect, allow = "answer", 1
    else:
        turn_checks += [silent_soon_after_last_part(1.0)]
        expect, allow = "answer", 1
    return Conversation(
        f"a pause of {pause:g} s before the last word",
        [
            Turn([AUSTRALIA], reply_mentions=["Canberra"], deaf_limit=0.05),
            Turn(["What's the capital of,", pause, "Australia?"], splice=True, must_hear=["australia"],
                 reply_mentions=["Canberra"], deaf_limit=0.05, expect=expect, allow_continued=allow,
                 checks=turn_checks + [c.one_user_message_with(["Australia"])]),
        ],
        stub_rules=[CANBERRA],
        tags=["turn-taking", "going on", f"pause {pause:g}"],
    )


def nothing_before_last_part(name: str = "played_early") -> Check:
    def fn(result, view):
        said = view.running.said
        start = last_part(said)
        played = view.runner._reply_sounds(said.speech_start, start)
        return f"the voice played {played[0][0] - said.speech_start:.2f} s into the turn" if played else None
    return Check(name, fn)


def silent_soon_after_last_part(limit: float, name: str = "answered_fragment") -> Check:
    """If Iris began answering the fragment, its voice falls silent soon
    after the user goes on."""
    def fn(result, view):
        said = view.running.said
        start = last_part(said)
        if not view.runner._reply_sounds(said.speech_start, start):
            return None
        quiet = view.speaker.silent_after(start)
        if quiet is None or quiet - start > limit:
            return "the voice went on over the user"
        return None
    return Check(name, fn)


def last_part(said) -> float:
    """When an utterance's last part begins: after its last pause of 0.3 s
    or more (its speech start if it has none)."""
    found = inner_silences(said.audio, 0.3)
    return said.start + found[-1][1] if found else said.speech_start


TRAILING = "I was wondering if you could,"
CATS = "tell me a joke about cats."


def nothing_before_turn(number: int, name: str = "played_early") -> Check:
    """No reply sound from an earlier turn's speech end to this turn's start."""
    def fn(result, view):
        earlier = view.runner._moments.get(number)
        if earlier is None or view.running.said is None:
            return None
        played = view.runner._reply_sounds(earlier[1], view.running.said.speech_start)
        return f"the voice played {played[0][0] - earlier[1]:.2f} s after turn {number}" if played else None
    return Check(name, fn)


TRAILING_OFF = [
    Conversation(
        "trailing off",
        [Turn([TRAILING], unsaid=[CATS], turn_end=(3.2, 4.0), checks=[c.one_user()])],
        tags=["turn-taking", "going on"],
    ),
    Conversation(
        "trailing off, then going on",
        [Turn([TRAILING, 2.5, CATS], splice=True, turn_end=(0, 1.0),
              checks=[c.one_user(), c.no_continued()], deaf_limit=0.05)],
        tags=["turn-taking", "going on"],
    ),
    Conversation(
        "trailing off, going on while Iris understands",
        [
            Turn([TRAILING], unsaid=[CATS]),
            Turn([CATS], at=At.state("understanding", 0.5), expect="either", first_word=4.0,
                 checks=[nothing_before_turn(1)]),
        ],
        tags=["turn-taking", "going on"],
    ),
    Conversation(
        "trailing off, going on once heard",
        [
            Turn([TRAILING], unsaid=[CATS]),
            Turn([CATS], at=At.event("chat.user", 0.3), expect="continued", checks=[nothing_before_turn(1)]),
        ],
        tags=["turn-taking", "going on"],
    ),
    Conversation(
        "trailing off, starting over",
        [
            Turn([TRAILING], unsaid=[CATS]),
            Turn(["Sorry, tell me a joke about cats."], at=At.reply_audible(1.0), must_not_hear=["wondering"],
                 checks=[c.one_user(), c.silent_after_user_began(1.0)]),
        ],
        tags=["turn-taking", "going on"],
    ),
]


def addition_while_thinking(delay: float, either: bool = False) -> Conversation:
    """A second question said while Iris works on the first."""
    expect = "either" if either else "continued"
    turn_checks = [c.one_user_message_with(["Australia", "people"])] if not either else []
    if not either:
        turn_checks += [c.nothing_played_between("played_early", 1, 2, 0.3), c.reply_not_interrupted()]
    return Conversation(
        f"an addition {delay:g} s after the question",
        [
            Turn([AUSTRALIA]),
            Turn(["And how many people live there?"], at=At.speech_end(delay), expect=expect,
                 checks=turn_checks),
        ],
        stub_rules=[CANBERRA],
        tags=["turn-taking", "going on", "while thinking"],
    )


GOING_ON = (
    [going_on_after(p) for p in (0.4, 1.0, 1.8, 2.6, 4.0, 7.0)]
    + TRAILING_OFF
    + [addition_while_thinking(d) for d in (0.5, 1.5, 2.5)]
)


# Several sentences in one breath ---------------------------------------------------------

LISBON_TRIP = ["I just got back from Lisbon.", "The food was wonderful.",
               "We had grilled sardines every night.", "What else should I try next time?"]


def one_breath(speed: float, gap: float) -> Conversation:
    parts = []
    for sentence in LISBON_TRIP:
        parts += [sentence, gap]
    return Conversation(
        f"several sentences at speed {speed:g}, gaps of {gap:g} s",
        [Turn(parts[:-1], speed=speed, must_hear=["lisbon", "wonderful", "sardines", "try"], deaf_limit=0.05,
              first_word=4.0, checks=[c.one_user(), c.no_continued(), c.nothing_played_while_speaking(),
                                      c.heard_at_least(0.9, name="heard_all"), report_verdicts()])],
        # The middle of the turn is lost after Smart Turn ends it at a gap
        # between sentences (being diagnosed separately)
        expected_failures=["must_hear", "heard_all", "deaf"],
        tags=["turn-taking", "one breath", f"speed {speed:g}"],
    )


ONE_BREATH = [one_breath(speed, gap) for speed in (0.8, 1.0, 1.3) for gap in (0.3, 0.6)] + [
    Conversation(
        "a plan and a question, fast",
        [Turn(["I'm planning a trip.", 0.25, "Which month is best for Japan?"], speed=1.3,
              must_hear=["trip", "japan"], deaf_limit=0.05, first_word=4.0,
              checks=[c.one_user(), c.no_continued(), c.nothing_played_while_speaking(),
                      c.heard_at_least(0.9, name="heard_all"), report_verdicts()])],
        expected_failures=["must_hear", "heard_all", "deaf"],
        tags=["turn-taking", "one breath", "speed 1.3"],
    ),
]


# Lists, codes, numbers and spelled names ---------------------------------------------------

FLIGHT_WEB = {"results": {r"lh ?402": "LH402 Frankfurt to Lisbon: on time, departs 14:05."},
              "delay": 1.0}
PAUSED_ITEMS = {
    "a list": dict(parts=["I need to buy eggs,", 0.7, "milk,", 0.7, "flour,", 0.9, "and butter.", 0.4,
                          "What can I bake with that?"],
                   must_hear=["eggs", "milk", "flour", "butter", "bake"]),
    "a code": dict(parts=["My locker code is", 0.6, "four,", 0.5, "nine,", 0.5, "two."],
                   must_hear_in_order=["4", "9", "2"]),
    "fillers": dict(parts=["So, um,", 1.2, "what's the, uh,", 1.0, "tallest mountain in Europe?"],
                    must_hear=["tallest", "mountain", "europe"]),
    "a flight number": dict(parts=["Is flight", 0.2, "L", 0.35, "H", 0.6, "four", 0.3, "oh", 0.3, "two", 0.4,
                                   "on time today?"],
                            heard_terms=["lh402"], search_mentions=["lh402"], reply_mentions=["14:05"],
                            check_with_stub=True),
    "a spelled name": dict(parts=["My name is Varga.", 0.5, "That's V,", 0.35, "A,", 0.35, "R,", 0.35, "G,",
                                  0.35, "A."],
                           heard_terms=["varga"],
                           checks=[c.word_in_order_or_whole(["v", "a", "r", "g", "a"], "varga")]),
    "a date": dict(parts=["What day of the week is the 3rd of March 2027?"],
                   heard_terms=["3|third", "march", "2027"], reply_mentions=["Wednesday"]),
    "a phone number": dict(parts=["Call me back on", 0.3, "oh one seven six,", 0.6, "two four five,", 0.6,
                                  "eight nine one three."],
                           checks=[c.digits_heard("01762458913")]),
}
# What must pass over a fan at 15 dB; the rest is reported
PASSES_IN_NOISE = ("a list", "a flight number", "a date")


def paused_item(name: str, noisy: bool, speed: float = 1.0) -> Conversation:
    spec = dict(PAUSED_ITEMS[name])
    parts = spec.pop("parts")
    extra = spec.pop("checks", [])
    turns = [Turn(parts, splice=len([p for p in parts if isinstance(p, str)]) > 1, speed=speed, deaf_limit=0.05,
                  checks=[c.one_user(), c.no_continued(), c.nothing_played_while_speaking(), report_verdicts()]
                  + extra, **spec)]
    if name == "a code":
        turns.append(Turn(["What was my code again?"], reply_mentions=["492", "4, 9, 2", "four, nine, two"]))
    flight = name == "a flight number"
    label = f"{name} said with pauses" + (" over a fan" if noisy else "")
    label += f" at speed {speed:g}" if speed != 1 else ""
    return Conversation(
        label,
        turns,
        tools={"web": True} if flight else {},
        web=FLIGHT_WEB if flight else None,
        stub_rules=[Rule("flight", tool=search())] if flight else None,
        noise=[Noise("fan", snr=15)] if noisy else [],
        expected_failures=["T1:*"] if noisy and name not in PASSES_IN_NOISE else [],
        tags=["turn-taking", "lists and codes", "fan 15 dB" if noisy else "quiet"],
    )


LISTS_AND_CODES = [paused_item(name, noisy) for noisy in (False, True) for name in PAUSED_ITEMS] + [
    paused_item("a list", noisy, 0.8) for noisy in (False, True)
]


# Turn detection, switched per conversation -----------------------------------------------


def turn_end_agrees(name: str = "turn_end_component") -> Check:
    """What turn.timing says ended the turn agrees with Smart Turn's
    verdicts: a verdict above 0.5 at the end means smart_turn."""
    def fn(result, view):
        component = result.components.get("turn_end")
        ended_by_verdict = any(p is not None and p > 0.5 for _, p in result.verdicts)
        if component == "smart_turn" and not ended_by_verdict and result.stages.get("turn_end", 0) > 3000:
            return "reported smart_turn, but the long silence ended the turn"
        if component == "silence" and ended_by_verdict:
            return "reported silence, but Smart Turn ended the turn"
        return None
    return Check(name, fn)


TURN_DETECTION = [
    Conversation(
        "ending turns after a silence",
        [
            Turn(["Tell me,", 2.5, "how far is the Moon from the Earth?"], turn_end=(3.3, 3.9),
                 reply_mentions=["384", "kilomet", "238", "miles"], checks=[c.one_user()]),
            Turn(["I need eggs,", 1.0, "milk,", 1.0, "and flour."], must_hear=["eggs", "milk", "flour"],
                 turn_end=(3.3, 3.9), checks=[c.one_user()]),
        ],
        turn_detection="silence",
        tags=["turn-taking", "turn detection"],
    ),
    Conversation(
        "switching turn detection while listening",
        [
            Turn(["What is the capital of Japan?"], turn_end=(0, 1.0), checks=[turn_end_agrees()]),
            Turn(["What is the capital of Kenya?"], actions=[(At.finished(0.0), ("turn_detection", "silence"))],
                 at=At.event("config", 0.5), turn_end=(3.3, 3.9), checks=[turn_end_agrees()]),
            Turn(["What is the capital of Peru?"], actions=[(At.finished(0.0), ("turn_detection", "smart_turn"))],
                 at=At.event("config", 0.5), turn_end=(0, 1.0), checks=[turn_end_agrees()]),
        ],
        # The listen() already waiting keeps its old tracker, and the
        # components report "silence"
        expected_failures=["T2:*"],
        tags=["turn-taking", "turn detection"],
    ),
]

WHISPER_TURNS = [
    ["What is the capital of France?"],
    ["How many legs does a spider have?"],
    ["Thanks."],
    ["Could you explain why the days are longer in summer than in winter in the northern hemisphere?"],
    ["I'd like to know", 0.8, "how tall the Eiffel Tower is."],
    ["What day comes after Monday?"],
]


def cell_components(size: str, mode: str, name: str = "components") -> Check:
    def fn(result, view):
        recognition = result.components.get("recognition", "")
        problems = []
        if not recognition.endswith(size):
            problems.append(f"recognized with {recognition!r}")
        if result.components.get("turn_end") not in (mode, None):
            problems.append(f"turns ended by {result.components.get('turn_end')!r}")
        return "; ".join(problems) or None
    return Check(name, fn)


def cell_summary(size: str, mode: str, name: str = "cell") -> Check:
    """Over the cell's turns: the cold turn's recognition against the steady
    median, and the turn ends expected of the detection."""
    def fn(result, view):
        turns = [r for r in view.runner.results_so_far if r.stages] + [result]
        recognition = [r.stages.get("recognition") for r in turns if r.stages.get("recognition") is not None]
        problems = []
        if len(recognition) > 2:
            steady = sorted(recognition[1:])[len(recognition[1:]) // 2]
            if recognition[0] > 1.5 * steady:
                problems.append(f"the cold turn's recognition took {recognition[0]} ms, steady {steady} ms")
        ends = [r.stages.get("turn_end", 0) / 1000 for r, words in zip(turns, WHISPER_TURNS)
                if len([p for p in words if isinstance(p, str)]) == 1]
        if ends:
            middle = sorted(ends)[len(ends) // 2]
            if mode == "smart_turn" and middle > 0.6:
                problems.append(f"turn end median {middle:.2f} s")
            if mode == "silence" and not all(3.3 <= e <= 3.7 for e in ends):
                problems.append(f"turn ends {ends}")
        return "; ".join(problems) or None
    return Check(name, fn)


def whisper_cell(size: str, mode: str) -> Conversation:
    """One size of Whisper with one way of ending turns."""
    heard = 0.9 if size not in ("tiny", "base") else 0.75
    turns = [Turn(parts, checks=[cell_components(size, mode), c.heard_at_least(heard)]) for parts in WHISPER_TURNS]
    turns[-1].checks.append(cell_summary(size, mode))
    return Conversation(
        f"Whisper {size} with {mode.replace('_', ' ')}",
        turns,
        stt_model=size,
        turn_detection=mode,
        tags=["turn-taking", "whisper", f"cell {size} {mode}", f"whisper {size}", mode],
    )


WHISPER_SIZES = ("tiny", "base", "small", "medium", "large-v3", "large-v3-turbo")


def whisper_cells(sizes=WHISPER_SIZES) -> list[Conversation]:
    return [whisper_cell(size, mode) for size in sizes for mode in ("smart_turn", "silence")]


# Back-channels and room noises -------------------------------------------------------------


def holds_per_aside(name: str = "holds") -> Check:
    """At most one hold per aside, short, and resumed with a little of what
    was heard repeated."""
    def fn(result, view):
        problems = []
        if len(result.holds) > max(1, len(view.running.asides)):
            problems.append(f"{len(result.holds)} holds for {len(view.running.asides)} asides")
        for at, silence, rewind in result.holds:
            if silence > 0.8:
                problems.append(f"silent {silence:.2f} s at {at:.1f} s")
            if rewind is not None and not 0.2 <= rewind <= 0.6:
                problems.append(f"repeated {rewind:.2f} s at {at:.1f} s")
        return "; ".join(problems) or None
    return Check(name, fn)


def ended_naturally(name: str = "ended_naturally") -> Check:
    def fn(result, view):
        clips = c.reply_clips(view)
        if not clips:
            return "nothing was said"
        return "the reply's last clip was cut" if clips[-1].stopped is not None else None
    return Check(name, fn)


def kept_whole(text: str, name: str = "kept_whole") -> Check:
    """With the scripted model, the history holds the whole reply."""
    def fn(result, view):
        if view.stub is None:
            return None
        kept = c.last_assistant(view.history())
        return None if normalized(kept) == normalized(text) else f"the history keeps {kept[:80]!r}"
    return Check(name, fn)


def unbroken(text: str) -> list[Check]:
    return [c.reply_not_interrupted(), kept_whole(text), holds_per_aside(), ended_naturally()]


BACK_CHANNELS = [
    Conversation(
        "back-channels during a long answer",
        [
            Turn(["Tell me a long story about a lighthouse keeper."], listen_through=True,
                 during=[(At.reply_audible(2.0), Say(["Mm-hm."])), (At.in_gap(2), Say(["Okay."])),
                         (At.reply_audible(9.0), Say(["Right."])), (At.reply_audible(14.0), Say(["Yeah."])),
                         (At.reply_audible(18.0), Say(["Mm-hm."], gain=0.35))],
                 checks=unbroken(LONG)),
            Turn(["What was the keeper's name?"]),
        ],
        tags=["interruptions", "back-channels"],
    ),
    Conversation(
        "noise during the answer",
        [Turn(["Tell me a long story about the sea."], listen_through=True,
              during=[(At.reply_audible(1.5), Sound("door")), (At.reply_audible(3.0), Sound("keyboard", seconds=2.0)),
                      (At.in_gap(2), Sound("mug")), (At.reply_audible(8.0), Sound("cough"))],
              checks=unbroken(LONG))],
        tags=["interruptions", "back-channels", "noise"],
    ),
] + [
    Conversation(
        f"{label} while Iris thinks",
        [
            Turn(["What is the capital of France?"]),
            Turn(["What is the capital of France?"], during=[(At.speech_end(0.8), item)], model_requests=1,
                 checks=[c.first_word_within_turn(1, 0.5), c.reply_not_interrupted(), kept_whole(SHORT)]),
        ],
        stub_rules=[Rule("capital of France", SHORT, first_token=2.5)],
        tags=["interruptions", "back-channels", "while thinking"],
    )
    for label, item in (("a door", Sound("door")), ("a cough", Sound("cough")),
                        ("typing", Sound("keyboard", seconds=2.0)), ("a hmm", Say(["Hmm."])))
] + [
    Conversation(
        "noises while Iris listens",
        [Turn(["What is two plus two?"], at=At.start(9.0),
              during=[(At.start(1.0), Sound("door")), (At.start(3.0), Sound("keyboard", seconds=2.0)),
                      (At.start(5.0), Sound("cough"))])],
        tags=["interruptions", "back-channels", "ambient"],
    ),
]


# Barge-in at five moments, twice in a row, hesitating -------------------------------------

BARGE_RULES = [
    Rule("two plus two", "Two plus two is four."),
    Rule("short fact", "Octopuses have three hearts. Two of them pump blood to the gills."),
    Rule("two more", "Here are two more. A snail can sleep for three years. Honey never spoils."),
]
FACT = "Octopuses have three hearts. Two of them pump blood to the gills."


def first_word_heard(name: str = "first_word_heard") -> Check:
    def fn(result, view):
        said = normalized(view.running.result.said).split()
        heard = normalized(result.heard).split()
        return None if said and said[0] in heard else f"the first word {said[:1]} is not in {result.heard!r}"
    return Check(name, fn)


def quiet_until_heard(name: str = "quiet_until_heard") -> Check:
    """No sound between the voice falling silent and the next chat.user."""
    def fn(result, view):
        said = view.running.said
        if said is None or result.barge_in is None:
            return None
        silent = said.speech_start + result.barge_in
        heard = [at for at, d in c.users(view, "voice") if at > silent]
        if not heard:
            return None
        played = view.runner._reply_sounds(silent + 0.3, heard[0])
        return f"the voice played {played[0][0] - silent:.2f} s after falling silent" if played else None
    return Check(name, fn)


def one_cut_reply(of_turn: int, name: str = "cut_reply") -> Check:
    def fn(result, view):
        start = c.turn_start(view, of_turn)
        cut = [d for _, d in c.dones(view, start, view.running.ref_end + 5.0) if d.get("interrupted")]
        return None if len(cut) == 1 else f"{len(cut)} interrupted replies to turn {of_turn}"
    return Check(name, fn)


def interrupting(at: At, parts=None, **more) -> Turn:
    return Turn(parts or ["Sorry, what is two plus two?"], at=at, reply_mentions=["four", "4"],
                checks=[first_word_heard(), quiet_until_heard(), one_cut_reply(1), c.kept_as_prefix(1)], **more)


def barge_moment(label: str, at: At, first=None, second=None, fresh: bool = False, run: int = 1,
                 cut: bool = True) -> Conversation:
    first = first or ["Tell me a long story about the sea."]
    turn = interrupting(at, second) if cut else Turn(
        second, at=at, checks=[first_word_heard(), c.heard_at_least(1.0, name="heard_all")])
    turns = [Turn(first), turn]
    if not cut:
        turns[0].checks += [c.reply_not_interrupted(), kept_whole(FACT)]
    return Conversation(
        f"barge-in {label}" + (" on a fresh Iris" if fresh else "") + (f" (run {run})" if run > 1 else ""),
        turns,
        stub_rules=BARGE_RULES,
        fresh=fresh,
        # The echo canceller lets too much through before it has heard 3 s
        # of the voice (SETTLE), so the first reply is not interrupted
        expected_failures=["T2:barge_in"] if fresh else [],
        tags=["interruptions", "barge-in", f"moment {label}"],
    )


def barge_moments(runs: int = 3) -> list[Conversation]:
    out = []
    for run in range(1, runs + 1):
        out += [
            barge_moment("at the first word", At.reply_audible(0.15), run=run),
            barge_moment("after a sentence's start", At.reply_audible(1.2), run=run),
            barge_moment("in a pause between sentences", At.in_gap(1, after=1.0), run=run),
            barge_moment("just before the end", At.before_end(0.4), ["Tell me a short fact."], run=run),
            barge_moment("just after the end", At.reply_end(0.1), ["Tell me a short fact."], ["Two more, please."],
                         run=run, cut=False),
        ]
    out.append(barge_moment("at the first word", At.reply_audible(0.15), fresh=True))
    return out


def hesitation_quiet(name: str = "played_early") -> Check:
    """Once the voice has stopped for "Wait,", nothing until the user has
    finished (and 0.3 s more)."""
    def fn(result, view):
        said = view.running.said
        silent = said.speech_start + (result.barge_in if result.barge_in is not None else 1.0)
        played = view.runner._reply_sounds(silent + 0.2, said.speech_end + 0.3)
        return f"the voice played {played[0][0] - said.speech_start:.2f} s into the turn" if played else None
    return Check(name, fn)


BARGE_IN = barge_moments() + [
    Conversation(
        "two interruptions in a row",
        [
            Turn(["Tell me a long story about a dragon."]),
            Turn(["Actually, tell me a long story about a robot instead."], at=At.reply_audible(2.5),
                 checks=[first_word_heard(), quiet_until_heard(), c.kept_as_prefix(1)]),
            Turn(["Sorry, stop. What is two plus two?"], at=At.reply_audible(1.0), reply_mentions=["four", "4"],
                 must_not_hear=["robot", "dragon"],
                 checks=[first_word_heard(), quiet_until_heard(), c.kept_as_prefix(2), c.history_alternates()]),
        ],
        stub_rules=BARGE_RULES,
        tags=["interruptions", "barge-in"],
    ),
    Conversation(
        "an interruption that hesitates",
        [
            Turn(["Tell me a long story about a lighthouse keeper."]),
            Turn(["Wait,", 1.0, "what is the capital of Japan?"], splice=True, at=At.reply_audible(2.0),
                 allow_continued=1, reply_mentions=["Tokyo"],
                 checks=[c.kept_as_prefix(1), hesitation_quiet()]),
        ],
        stub_rules=BARGE_RULES,
        tags=["interruptions", "barge-in"],
    ),
    Conversation(
        "stop, then go on while Iris thinks",
        [
            Turn(["Tell me a long story about the sea."]),
            Turn(["Stop."], at=At.reply_audible(2.0), expect="silenced"),
            Turn(["Tell me about Rome instead."], at=At.speech_end(1.2), expect="continued",
                 must_hear=["stop", "rome"], reply_mentions=["Rome"]),
        ],
        stub_rules=BARGE_RULES,
        tags=["interruptions", "barge-in"],
    ),
    Conversation(
        "a reply cut before any sound",
        [
            Turn(["Tell me everything about volcanoes."], actions=[(At.event("chat.user", 0.1), "stop")]),
            Turn(["What is the opposite of hot?"], reply_mentions=["cold"],
                 checks=[c.heard_at_least(1.0, name="heard_all"), first_word_heard(),
                         c.history_excludes(["volcanoes"]), c.history_alternates()]),
        ],
        stub_rules=BARGE_RULES,
        tags=["interruptions", "barge-in"],
    ),
]



# Stop pressed late, twice, or when idle ----------------------------------------------------

AFTER_STOP = dict(reply_mentions=["Tuesday"], checks=[c.reply_not_interrupted()])


def stop_after_reply(gap: float, typed: bool = False) -> Conversation:
    return Conversation(
        f"stop {gap:g} s after the reply" + (", then a typed question" if typed else ""),
        [
            Turn(["What is the capital of France?"], listen_through=True, actions=[(At.reply_end(gap), "stop")]),
            Turn(["What day comes after Monday?"], typed=typed, **AFTER_STOP),
        ],
        # interrupt_event stays set while listen() blocks, and the next
        # spoken turn is dropped; a typed one is answered
        expected_failures=[] if typed else ["T2:*"],
        tags=["interruptions", "stop"],
    )


STOP_LATE = [
    stop_after_reply(0.3),
    stop_after_reply(1.0),
    Conversation(
        "stop pressed twice",
        [
            Turn(["Tell me a long story about the sea."],
                 actions=[(At.reply_audible(2.0), "stop"), (At.event("action.stop", 0.4), "stop")]),
            Turn(["What day comes after Monday?"], **AFTER_STOP),
        ],
        expected_failures=["T2:*"],
        tags=["interruptions", "stop"],
    ),
    Conversation(
        "stop when nothing is happening",
        [Turn(["What is the opposite of hot?"], actions=[(At.start(2.0), "stop")], at=At.event("action.stop", 1.0),
              reply_mentions=["cold"], checks=[c.reply_not_interrupted()])],
        expected_failures=["T1:*"],
        tags=["interruptions", "stop"],
    ),
    stop_after_reply(0.3, typed=True),
    Conversation(
        "a spoken stop",
        [
            Turn(["Tell me a long story about the sea."]),
            Turn(["Stop."], at=At.reply_audible(2.0), expect="silenced"),
            Turn(["What day comes after Monday?"], **AFTER_STOP),
        ],
        tags=["interruptions", "stop"],
    ),
]


# Pause and resume, the microphone off and on ------------------------------------------------


def silent_while_paused(name: str = "paused") -> Check:
    """Silent from 0.2 s after each pause until its resume, a paused state at
    the pause, and the voice back within 0.6 s of resuming, repeating a
    little of what was heard."""
    def fn(result, view, resume_within=0.6):
        pauses = c.actions(view, "pause", view.running.since)
        resumes = c.actions(view, "resume", view.running.since)
        problems = []
        for pause, resume in zip(pauses, resumes):
            played = view.speaker.sound_heard(pause + 0.2, resume)
            if played > 0:
                problems.append(f"played {played:.1f} s while paused")
            if not any(d.get("state") == "paused" for _, _, d in view.log.since(pause - 0.05, "state", pause + 0.5)):
                problems.append("no paused state")
            back = view.speaker.first_sound_after(resume)
            if back is None or back - resume > resume_within:
                problems.append(f"back {'never' if back is None else f'{back - resume:.2f} s'} after resuming")
            cut = [k for k in c.reply_clips(view, pause - 1.0, resume)
                   if k.stopped and pause <= k.stopped <= pause + 0.5]
            again = [k for k in c.reply_clips(view, resume, resume + 2.0) if k.started >= resume]
            if cut and again:
                found = holds([cut[0], again[0]]) if again[0].started - cut[0].stopped <= 1.2 else []
                rewind = found[0].rewind if found else None
                if rewind is not None and not 0.2 <= rewind <= 0.6:
                    problems.append(f"repeated {rewind:.2f} s on resuming")
        return "; ".join(problems) or None
    return Check(name, fn)


def muted_soon(name: str = "muted") -> Check:
    """A muted state within 0.3 s of each microphone switched off."""
    def fn(result, view):
        problems = []
        for moment in [at for at, _, d in view.log.since(view.running.since - 30, "action.mic", view.end)
                       if d.get("args") == [False]]:
            if not any(d.get("state") == "muted" for _, _, d in view.log.since(moment, "state", moment + 0.3)):
                last = view.log.last_before(moment + 0.3, "state")
                if not (last and last[2].get("state") == "muted"):
                    problems.append("not muted 0.3 s after the microphone was switched off")
        return "; ".join(problems) or None
    return Check(name, fn)


def still_audible_after_aside(seconds: float, name: str = "kept_talking") -> Check:
    def fn(result, view):
        if not view.running.asides:
            return "no aside"
        moment = view.running.asides[0].start + seconds
        return None if view.speaker.audible(moment) else f"silent {seconds} s after the aside began"
    return Check(name, fn)


BERLIN_WEB = {"results": {"berlin": "Berlin today: 14 degrees and cloudy."}, "delay": 3.0}

PAUSE_AND_MIC = [
    Conversation(
        "pause, resume, interrupt",
        [
            Turn(["Tell me a long story about a lighthouse keeper."],
                 actions=[(At.reply_audible(3.0), "pause"), (At.event("action.pause", 4.0), "resume")],
                 during=[(At.event("action.pause", 1.0), Say(["Hang on, I'm on the phone."]))],
                 checks=[silent_while_paused()]),
            Turn(["Sorry, what is two plus two?"], at=At.event("action.resume", 3.0), reply_mentions=["four", "4"]),
        ],
        stub_rules=BARGE_RULES,
        # listen_on ends on _mic_closed and nothing restarts it on resume
        expected_failures=["T2:*"],
        tags=["interruptions", "pause"],
    ),
    Conversation(
        "quick pause and resume",
        [Turn(["Tell me a long story about the sea."], listen_through=True,
              actions=[(At.reply_audible(2.0), "pause"), (At.event("action.pause", 0.3), "resume"),
                       (At.reply_audible(4.0), "pause"), (At.event("action.pause", 1.0, n=2), "resume")],
              checks=[silent_while_paused(), c.reply_not_interrupted(), kept_whole(LONG)])],
        tags=["interruptions", "pause"],
    ),
    Conversation(
        "pause during a search",
        [
            Turn(["What's the weather in Berlin today?"],
                 actions=[(At.event("tool.use", 0.5), "pause"), (At.event("action.pause", 6.0), "resume")],
                 checks=[Check("paused", lambda r, v: silent_while_paused().fn(r, v, resume_within=1.0))]),
            Turn(["And tomorrow?"], at=At.event("action.resume", 1.5)),
        ],
        tools={"web": True},
        web=BERLIN_WEB,
        stub_rules=[Rule("berlin", tool=search())],
        expected_failures=["T2:*"],
        tags=["interruptions", "pause", "search"],
    ),
    Conversation(
        "the microphone off and on",
        [
            Turn(["What is the capital of France?"]),
            Turn(["What is the capital of Germany?"], actions=[("mic", False)], expect="silent", checks=[muted_soon()]),
            Turn(["And of Spain?"], typed=True),
            Turn(["Tell me a long story about a lighthouse keeper."], typed=True,
                 during=[(At.reply_audible(2.0), Say(["Stop, please."]))],
                 checks=[still_audible_after_aside(2.0), c.reply_not_interrupted()]),
            Turn(["Explain in detail how a rainbow forms."], actions=[
                ("mic", True), (At.reply_audible(1.5), ("mic", False)), (At.reply_audible(3.5), ("mic", True))],
                checks=[muted_soon()]),
            Turn(["Sorry, stop. What colour is the sky?"], at=At.reply_audible(5.0)),
            Turn(["What is the tallest mountain", 0.6, "in the world?"], expect="silent",
                 actions=[(At.speech_start(0.8), ("mic", False)), (At.speech_end(2.0), ("mic", True))]),
            Turn(["And of Portugal?"], must_not_hear=["germany", "stop", "please", "mountain", "world"],
                 checks=[c.request_holds(["capital of France", "Spain", "lighthouse"],
                                         absent=["Germany", "Stop, please", "tallest mountain"])]),
        ],
        expected_failures=["T6:*"],
        tags=["interruptions", "microphone"],
    ),
]


# A search turn's timing ---------------------------------------------------------------------

LISBON_RESULT = [{"title": "Lisbon weather today", "body": "Sunny, 19 degrees, light wind from the north.",
                  "href": "https://weather.example/lisbon"}]
LISBON_ANSWER = "In Lisbon today it is sunny and 19 degrees. There is a light wind from the north."
LEAD_INS = ["", "Let me look that up.", "Let me look that up for you.",
            "One moment, I'll check the forecast for Lisbon.", "Let me look that up for you. ",
            "Let me look that up.\n\n"]


def lead_in_for_turn(messages: list[dict]) -> str:
    """The lead-in of the turn the request belongs to (its user messages
    count the turns: the second is the first search)."""
    index = len(user_messages(messages)) - 2
    return LEAD_INS[index] if 0 <= index < len(LEAD_INS) else ""


def search_turn_checks(lead_in: bool, slow: bool) -> list[Check]:
    found = [c.one_search("lisbon"), c.no_glued_sentences(), c.requests_count(2),
             c.second_request_has_result("Sunny, 19 degrees"), c.flags_follow_search()]
    if slow:
        found.append(c.search_window_at_least(2.9))
    if lead_in:
        found += [c.lead_in_while_searching(), c.searching_after_lead_in(), c.first_word_within_turn(1, 0.5)]
    return found


SEARCH_TIMING = Conversation(
    "search timing",
    [Turn(["What is the capital of France?"], tools_used=[])]
    + [Turn(["What's the weather in Lisbon today?"], reply_mentions=["19"], check_with_stub=True,
            checks=search_turn_checks(bool(lead), slow=index > 0)
            + ([c.search_added(1, 1.0 + 0.3 + 0.6), c.reply_stage_at_least(1.0)] if index == 0 else []))
       for index, lead in enumerate(LEAD_INS)],
    tools={"web": True},
    web={"results": {"lisbon": LISBON_RESULT}, "delay": [1.0, 3.0]},
    stub_rules=[Rule("weather in Lisbon", lead_in=lead_in_for_turn, tool=search(), after_tool=LISBON_ANSWER)],
    stub_options={"first_token": 0.3},
    # split_speech only cuts at "[.!?] " and nothing flushes the buffer when a
    # tool starts: the lead-in is glued to the answer ("up.In")
    expected_failures=[f"T{n}:{check}" for n in (3, 4, 5) for check in ("lead_in", "glued", "first_word")],
    needs=("stub",),
    tags=["search", "timing"],
)


def real_search_timing() -> Conversation:
    """With a real model and the real web: what the search adds."""
    return Conversation(
        "search timing with a real model",
        [Turn(["What is the capital of France?"]),
         Turn(["Search the web for the weather in Lisbon today."], tools_used=["search"],
              checks=[c.search_added(1, 6.0, name="search_added")])],
        tools={"web": True},
        needs=("real_model", "real_web"),
        tags=["search", "timing", "network"],
    )


# Stop and barge-in at each phase of a search turn -----------------------------------------------

KESTREL_RESULT = [{"title": "Kestrel Cup final report",
                   "body": "Harbor City beat Northfield 3-1 in the Kestrel Cup final on Sunday; "
                           "Mara Okafor scored twice.",
                   "href": "https://sport.example/kestrel-final"}]
KESTREL_SHORT = "Harbor City won the Kestrel Cup final. They beat Northfield."
KESTREL_DETAIL = " ".join(f"Sentence {n} about the Kestrel Cup final, which Harbor City won." for n in range(1, 9))
KESTREL_QUESTION = "Who won the Kestrel Cup final yesterday?"


def kestrel_answer(messages: list[dict]) -> str:
    return KESTREL_DETAIL if re.search(r"in detail", last_user(messages), re.I) else KESTREL_SHORT


def kestrel(delay: float, lead_in: str = "") -> dict:
    return dict(
        tools={"web": True},
        web={"results": {"kestrel": KESTREL_RESULT}, "delay": delay},
        stub_rules=[Rule("kestrel", lead_in=lead_in, tool=search(), after_tool=kestrel_answer)] + BARGE_RULES,
    )


def lead_in_kept_as_heard(of_turn: int, name: str = "lead_in_kept") -> Check:
    """A turn cut during its lead-in is kept as far as the lead-in was heard,
    and nothing of the search's answer was said."""
    def fn(result, view):
        problems = []
        if any("harbor" in (d.get("text") or "").lower() for _, d in c.dones(view, c.turn_start(view, of_turn))):
            problems.append("the answer was said")
        kept = [m.get("content") or "" for m in view.history() if m.get("role") == "assistant"]
        first = kept[0] if kept else ""
        if first and not normalized("Sure, let me look that up for you.").startswith(normalized(first)):
            problems.append(f"kept {first!r} for the lead-in")
        return "; ".join(problems) or None
    return Check(name, fn)


def cut_answer_kept(of_turn: int, name: str = "kept_prefix") -> Check:
    """In the turn's request, the cut answer is a non-empty word-prefix of
    what it was saying, shorter than it, and no tool message is there."""
    def fn(result, view):
        requests = view.requests()
        if not requests:
            return None
        messages = requests[0].messages
        if any(m.get("role") == "tool" for m in messages):
            return "a tool message in the request"
        cut = [d.get("text", "") for _, d in c.dones(view, c.turn_start(view, of_turn)) if d.get("interrupted")]
        kept = assistant_before_last_user(messages)
        if not kept.strip():
            return "the cut answer was not kept"
        whole, part = (normalized(cut[0]), normalized(kept)) if cut else ("", "")
        if cut and not (whole.startswith(part) and len(part) < len(whole)):
            return f"kept {kept[:80]!r}"
        return None
    return Check(name, fn)


SEARCH_STOPS = [
    Conversation(
        "stop during the search",
        [
            Turn([KESTREL_QUESTION], actions=[(At.event("tool.use", 1.0), "stop")],
                 checks=[c.stopped_during_search()]),
            Turn(["Tell me a long story about a lighthouse keeper."], at=At.event("action.stop", 1.0),
                 listen_through=True,
                 checks=[Check("pipeline_after_stop", lambda r, v: None if r.voice_pipeline is None or
                               r.voice_pipeline <= v.runner.budgets["voice_pipeline"] + 0.3 else
                               f"voice pipeline {r.voice_pipeline:.2f} s after Stop")]),
            Turn(["What was that story about?"],
                 checks=[c.no_tool_use_after("stop"), c.no_request_after("stop", [KESTREL_QUESTION]),
                         c.history_excludes(["Harbor", "Kestrel"]), c.assistant_before_last_user(LONG),
                         c.threads_back(1)]),
        ],
        # The state stays searching until the web answers, and the next turn's
        # chat_stream clears _cancelled: a stale tool round reaches the model
        # and conversation.reply can be overwritten for the next turn
        expected_failures=["T1:search_stopped", "T3:stale_request", "T3:history", "T3:tool_after_stop"],
        tags=["search", "stop"],
        **kestrel(5.0),
    ),
    Conversation(
        "stop during the lead-in",
        [Turn([KESTREL_QUESTION], actions=[(At.reply_audible(0.8), "stop")]),
         Turn(["What is two plus two?"], reply_mentions=["four", "4"],
              checks=[c.no_request_after("stop", [KESTREL_QUESTION]), lead_in_kept_as_heard(1)])],
        tags=["search", "stop"],
        **kestrel(5.0, "Sure, let me look that up for you.\n\n"),
    ),
    Conversation(
        "barging in during the lead-in",
        [Turn([KESTREL_QUESTION]),
         Turn(["Never mind, what is two plus two?"], at=At.reply_audible(0.8), reply_mentions=["four", "4"],
              checks=[c.no_request_after("T2", [KESTREL_QUESTION]), lead_in_kept_as_heard(1)])],
        tags=["search", "barge-in"],
        **kestrel(5.0, "Sure, let me look that up for you.\n\n"),
    ),
    Conversation(
        "stop the searched answer",
        [Turn(["Search the web and tell me in detail about the Kestrel Cup final."],
              actions=[(At.reply_audible(2.0), "stop")]),
         Turn(["Who scored for Harbor City?"], reply_mentions=["Okafor"],
              checks=[c.no_request_after("stop", ["in detail about the Kestrel"], tool_messages=False),
                      cut_answer_kept(1)])],
        tags=["search", "stop"],
        **kestrel(1.0),
    ),
    Conversation(
        "interrupting the searched answer",
        [Turn(["Search the web and tell me in detail about the Kestrel Cup final."]),
         Turn(["Sorry, just the score please."], at=At.reply_audible(3.0),
              reply_mentions=["3-1", "3 to 1", "three to one"], checks=[cut_answer_kept(1)])],
        tags=["search", "barge-in"],
        **kestrel(1.0),
    ),
]




# Talking while Iris searches ---------------------------------------------------------------

CITIES = {"lisbon": r"lisbon", "porto": r"porto"}
CITY_RESULTS = {"lisbon": "Lisbon today: 19 degrees and sunny.", "porto": "Porto today: 17 degrees and cloudy."}
LOOK_IT_UP = "Let me look that up.\n\n"


def city_rule(lead_in: str = "") -> Rule:
    """A search for the weather in the city named last, unless the user
    gave up on it."""
    def wants(messages):
        text = last_user(messages)
        return latest(CITIES, text) is not None and not re.search(r"never mind", text, re.I)

    return Rule(wants, lead_in=lead_in,
                tool=search(lambda m: f"weather in {latest(CITIES, last_user(m)).title()} today"),
                after_tool="Here is what I found. {first}")


def while_searching(lead_in: bool, **web) -> dict:
    return dict(
        tools={"web": True},
        web={"results": dict(CITY_RESULTS), **web},
        stub_rules=[Rule("two plus two", "Two plus two is four."), city_rule(LOOK_IT_UP if lead_in else "")],
    )


def asides_while_searching() -> list[Conversation]:
    out = []
    for lead_in in (False, True):
        for label, item in (("okay", Say(["Okay."])), ("hmm", Say(["Hmm."])), ("a cough", Sound("cough")),
                            ("a door", Sound("door")), ("typing", Sound("keyboard", seconds=2.0))):
            out.append(Conversation(
                f"{label} while Iris searches" + (" after a lead-in" if lead_in else ""),
                [Turn(["What's the weather in Lisbon today?"], during=[(At.state("searching", 1.5), item)],
                      reply_mentions=["19"], check_with_stub=True,
                      spoken_once=["look that up"] if lead_in else [],
                      checks=[c.uses_count(1), c.web_queries(1), c.no_continued(), c.reply_not_interrupted(),
                              c.state_held_during_search(forbid=("listening", "idle")), c.answer_after_web(1.5),
                              Check("tool_rounds", lambda r, v: None if sum(bool(q.tool_messages())
                                    for q in v.requests()) == 1 else "not one request with the tool's result")])],
                # listen_on holds only while the voice speaks: the user's own
                # "okay" and "hmm" are taken up; a cough and a door are reported
                expected_failures=["T1:*"] if label in ("okay", "hmm", "a cough", "a door") else [],
                tags=["search", "while searching", "aside"],
                **while_searching(lead_in, delay=4.0),
            ))
    return out


def correction_checks(second_delay: float) -> list[Check]:
    return [c.web_queries(2, first_without="porto"), c.last_search_mentions("porto"),
            c.spoken_mentions(["17"]), c.no_complete_reply_mentions(["19"]),
            c.no_request_after("T2", ["Lisbon today"], tool_messages=False), c.last_assistant_is_reply(),
            c.state_held_during_search(forbid=("listening", "idle", "thinking")),
            c.first_word_after(4.0 + second_delay)]


def corrections_while_searching() -> list[Conversation]:
    out = []
    for first, second in ((1, 1), (6, 6), (8, 1)):
        for lead_in in (False, True):
            expected = ["T2:searching_state"] if first > 1 else []
            if (first, second) == (8, 1):
                expected += ["T2:stale_request", "T2:history"]
            turn = Turn(["Actually, I meant Porto."], at=At.state("searching", 1.0),
                        expect="answer" if lead_in else "continued", checks=correction_checks(second))
            if lead_in:
                turn.checks.append(Check("history", lambda r, v: None if any(
                    normalized(m.get("content") or "") == normalized("Let me look that up.")
                    for m in v.history() if m.get("role") == "assistant") else "the lead-in was not kept"))
            out.append(Conversation(
                f"a correction while Iris searches ({first} s, then {second} s)"
                + (" after a lead-in" if lead_in else ""),
                [Turn(["What's the weather in Lisbon today?"]), turn],
                expected_failures=expected,
                tags=["search", "while searching", "correction"],
                **while_searching(lead_in, delay=[first, second]),
            ))
    out.append(Conversation(
        "a correction over the lead-in",
        [Turn(["What's the weather in Lisbon today?"]),
         Turn(["Actually, I meant Porto."], at=At.reply_audible(0.4), checks=correction_checks(1.0))],
        tags=["search", "while searching", "correction", "barge-in"],
        **while_searching(True, delay=[1.0, 1.0]),
    ))
    return out


def one_answer_after(number: int, name: str = "answers") -> Check:
    def fn(result, view):
        start = c.turn_start(view, number)
        complete = [d for _, d in c.dones(view, start) if not d.get("interrupted")]
        return None if len(complete) == 1 else f"{len(complete)} complete replies after turn {number}"
    return Check(name, fn)


STILL_THERE_WEB = {"results": {"lisbon": Each(["Sunny, 22 degrees.", "Sunny, 23 degrees."]),
                               "porto": CITY_RESULTS["porto"]},
                   "delay": [8.0, 1.0]}

TALKING_WHILE_SEARCHING = asides_while_searching() + corrections_while_searching() + [
    Conversation(
        "are you still there",
        [Turn(["What's the weather in Lisbon today?"]),
         Turn(["Hello? Are you still there?"], at=At.state("searching", 2.0), expect="continued",
              reply_mentions=["23"], check_with_stub=True, reply_excludes=["22"],
              checks=[one_answer_after(2), c.no_request_after("T2", ["22 degrees"], tool_messages=False),
                      c.last_assistant_is_reply(), c.no_searching_after_done()])],
        tags=["search", "while searching"],
        **{**while_searching(False), "web": STILL_THERE_WEB},
    ),
    Conversation(
        "never mind while Iris searches",
        [Turn(["What's the weather in Lisbon today?"]),
         Turn(["Never mind. What is two plus two?"], at=At.state("searching", 1.0), expect="either",
              reply_mentions=["four", "4"]),
         Turn(["What did you just tell me?"], typed=True,
              checks=[c.request_holds(["Never mind"], absent=["Lisbon today"])])],
        tags=["search", "while searching"],
        **while_searching(False, delay=8.0),
    ),
]


# Correcting and following up on what a search found ------------------------------------------

FOLLOW_UP_RESULTS = {
    "paris": "- Paris weather: tomorrow 22 degrees and clear (weather.example/paris)",
    "lyon": "Lyon weather: tomorrow 14 degrees and light rain; the day after 19 degrees and sunny "
            "(weather.example/lyon)",
    "kestrel": KESTREL_RESULT,
}
TOPICS = {"Paris": r"paris", "Lyon": r"lyon", "Kestrel Cup": r"kestrel|harbor"}


def topic_query(messages: list[dict]) -> str:
    """The city or the cup named last in any user message, and the last message."""
    named = None
    for text in user_messages(messages):
        named = latest(TOPICS, text) or named
    return f"{named or ''} {last_user(messages)}".strip()


def follow_up_rule(after_tool: str = "Here is what I found. {first}") -> Rule:
    pattern = r"weather|meant|day after|website|who won|beat|score"
    return Rule(lambda m: bool(re.search(pattern, last_user(m), re.I)), tool=search(topic_query),
                after_tool=after_tool)


def searched_turn(**more) -> Turn:
    """A turn whose answer comes from a search."""
    parts = more.pop("parts")
    extra = more.pop("checks", [])
    return Turn(parts, model_requests=2, checks=[web_got_query(), was_searching(), history_clean(),
                                                 c.listening_after(2.0)] + extra, **more)


def web_got_query(name: str = "web_query") -> Check:
    """The scripted web was asked what the tool was called on."""
    def fn(result, view):
        uses = [d for _, t, d in c.tool_uses(view) if t == "search"]
        calls = [w.query for w in view.web.between(view.running.since, view.end)] if view.web else []
        if not uses:
            return "no search"
        return None if uses[-1] in calls else f"searched {uses[-1]!r}, the web got {calls}"
    return Check(name, fn)


def was_searching(name: str = "searching_state") -> Check:
    def fn(result, view):
        return None if any(s == "searching" for _, s in result.states) else "never in the searching state"
    return Check(name, fn)


def history_clean(name: str = "history") -> Check:
    """The model is given the previous turn as said: its user text, and its
    whole spoken reply as one assistant message; no tool message is kept."""
    def fn(result, view):
        requests = view.requests()
        if not requests or view.stub is None:
            return None
        messages = requests[0].messages
        if any(m.get("role") == "tool" for m in messages):
            return "a tool message was kept"
        before = c.turn_result(view, view.running.number - 1)
        if before is None or not before.reply:
            return None
        kept = assistant_before_last_user(messages)
        return None if normalized(kept) == normalized(before.reply) else f"the reply before was kept as {kept[:80]!r}"
    return Check(name, fn)


def score_not_guessed(name: str = "guessed") -> Check:
    """With a real model: the score comes from a search, or from a history
    that holds it; another score, or one with nothing to go on, is a guess."""
    def fn(result, view):
        if view.stub is not None:
            return None
        searched = any(t == "search" for _, t, _ in c.tool_uses(view))
        says = re.search(r"\b\d+\s*(?:-|to)\s*\d+\b|three to one", result.reply or "", re.I)
        known = any("3-1" in (m.get("content") or "") for m in view.history())
        view.runner.report(f"  note: {result.conversation} · turn {result.turn}: "
                           f"{'searched again' if searched else 'did not search'}, first word {result.first_word}")
        if says and not re.search(r"3\s*(?:-|to)\s*1|three to one", says.group(0), re.I):
            return f"gave the score {says.group(0)!r}"
        if says and not searched and not known:
            return "gave the score without searching, with nothing to go on"
        return None
    return Check(name, fn)


def kestrel_in_query(name: str = "query") -> Check:
    def fn(result, view):
        uses = [d for _, _, d in c.tool_uses(view)]
        bad = [d for d in uses if not re.search(r"kestrel|harbor", d, re.I)]
        return f"searched without the subject: {bad}" if bad and view.stub is None else None
    return Check(name, fn)


def probe(note: str, name: str = "probe") -> Check:
    """Reported, not failed."""
    def fn(result, view):
        view.runner.report(f"  note: {result.conversation} · turn {result.turn}: {note}: {result.reply[:120]!r}")
        return None
    return Check(name, fn)


FOLLOWING_UP = [
    Conversation(
        "correcting and following up on the weather",
        [
            searched_turn(parts=["What's the weather going to be like in Paris tomorrow?"], search_mentions=["paris"],
                          reply_mentions=["22"], check_with_stub=True),
            searched_turn(parts=["No, sorry, I meant Lyon."], search_mentions=["lyon"], reply_mentions=["14", "rain"],
                          check_with_stub=True, reply_excludes=["22"]),
            searched_turn(parts=["And the day after?"], search_mentions=["lyon"], reply_mentions=["19", "sunny"],
                          check_with_stub=True, reply_excludes=["22"],
                          checks=[Check("query", lambda r, v: "paris was searched for" if any(
                              re.search("paris", d, re.I) for _, _, d in c.tool_uses(v)) else None)]),
            Turn(["Which city did I ask you about first?"], typed=True, reply_mentions=["Paris"]),
            Turn(["Which website did you get that from?"], checks=[probe("should name weather.example")]),
        ],
        tools={"web": True},
        web={"results": FOLLOW_UP_RESULTS, "delay": 1.5},
        stub_rules=[follow_up_rule("Here is what I found. {first}")],
        tags=["search", "follow-up"],
    ),
    Conversation(
        "following up on a search's result",
        [
            searched_turn(parts=["Who won the Kestrel Cup final?"], reply_mentions=["Harbor City"],
                          check_with_stub=True),
            Turn(["And who did they beat?"], reply_mentions=["Northfield"], checks=[kestrel_in_query()]),
            Turn(["What was the score?"], checks=[score_not_guessed()]),
        ],
        tools={"web": True},
        web={"results": FOLLOW_UP_RESULTS, "delay": 1.5},
        # The score is left out on purpose: the model must search again for it
        stub_rules=[follow_up_rule(KESTREL_SHORT)],
        tags=["search", "follow-up"],
    ),
]


# A search that fails, finds nothing, or hangs -----------------------------------------------


def failing_answer(messages: list[dict]) -> str:
    _, content = last_tool(messages)
    if content.startswith("Error:"):
        return "I couldn't reach the web just now, so I can't tell you."
    if content == "No results found.":
        return "I found nothing about that."
    return fill("Here is what I found. {first}", messages)


def calm_after_reply(name: str = "settled") -> Check:
    """Not searching, and listening, within 1 s of the reply's last sound."""
    def fn(result, view):
        last = view.speaker.last_sound(view.running.ref_end, view.end)
        if last is None:
            return None
        problems = []
        on = c.windows(view, "searching")
        if on and on[-1][1] > last + 1.0:
            problems.append("still searching after the reply")
        state = view.log.last_before(last + 1.0, "state")
        if not state or state[2].get("state") not in ("listening", "dormant"):
            problems.append(f"{state[2].get('state') if state else 'no state'} 1 s after the reply")
        return "; ".join(problems) or None
    return Check(name, fn)


def within_search(delay: float) -> Check:
    return Check("first_word", lambda r, v: None if r.first_word is None or
                 r.first_word <= delay + 0.3 + v.runner.budgets["voice_pipeline"] else
                 f"the first word came after {r.first_word:.2f} s")


FAILING_SEARCH = [
    Conversation(
        "a search that fails, finds nothing, or hangs",
        [
            Turn(["Search the web for the opening hours of the Kestrel Museum."], tools_used=["search"],
                 reply_mentions=["couldn't", "could not", "unable", "can't", "sorry"], check_with_stub=True,
                 checks=[c.tool_message(starts="Error:"), within_search(1.5), calm_after_reply(),
                         c.no_time_of_day()]),
            Turn(["Then look up the Zephyr Lake ferry timetable."], tools_used=["search"],
                 reply_mentions=["nothing", "no results", "couldn't find", "didn't find"], check_with_stub=True,
                 checks=[c.tool_message(equals="No results found."), within_search(1.0), calm_after_reply()]),
            Turn(["What is two plus two?"], tools_used=[], checks=[calm_after_reply()]),
            Turn(["Look up the Old Lighthouse tour times."], tools_used=["search"],
                 checks=[c.state_held_during_search(forbid=("listening", "idle")), c.silent_wait()]),
        ],
        tools={"web": True},
        web={"results": {"museum": ConnectionError("timed out"), "zephyr": [],
                         "lighthouse": "Old Lighthouse tours: hourly from 10:00."},
             "delay": {"museum": 1.5, "zephyr": 1.0, "lighthouse": 15.0}},
        stub_rules=[Rule(r"search|look up", tool=search(), after_tool=failing_answer),
                    Rule("two plus two", "Two plus two is four.")],
        reply_timeout=60.0,
        tags=["search", "failures"],
    ),
]


# Web search off, on and off again; tool markup written as text ----------------------------------

MARS = "Search the web for today's news about the Mars mission."
MARKUP = '<tool_call>{"name": "search", "arguments": {"query": "Mars mission news"}}</tool_call>'


def no_lookup_claim(name: str = "claims_search") -> Check:
    def fn(result, view):
        if view.stub is not None:
            return None
        if re.search(r"\bI (?:searched|looked it up|looked that up)\b|according to the web", result.reply or "", re.I):
            return f"claims to have looked it up: {result.reply[:100]!r}"
        return None
    return Check(name, fn)


WEB_SWITCH = [
    Conversation(
        "web search off and on",
        [
            Turn([MARS], tools_used=[], reply_mentions=["can't", "cannot", "unable", "not able", "don't have",
                                                         "no access"],
                 checks=[c.no_searching_flags(), c.no_tool_offer(), no_lookup_claim()]),
            Turn([MARS], actions=[(At.finished(0.0), ("tool", "web", True))], at=At.event("config", 0.6),
                 tools_used=["search"], checks=[c.offers(["search", "browse"]), c.uses_count(1)]),
            Turn(["Look it up once more."], actions=[(At.finished(0.0), ("tool", "web", False))],
                 at=At.event("config", 0.6), tools_used=[],
                 checks=[c.no_searching_flags(), c.no_tool_offer(), no_lookup_claim()]),
        ],
        web={"results": {"mars": "Mars mission news: the rover reached the crater rim."}, "delay": 1.0},
        stub_rules=[Rule(r"search|look it up", tool=search(lambda m: "Mars mission news"))],
        tags=["search", "switch"],
    ),
    Conversation(
        "tool markup written as text",
        [Turn([MARS], spoken_excludes=["<tool_call>", "arguments"], checks=[c.no_markup()])],
        stub_rules=[Rule("search", tool=search(), no_tools=MARKUP)],
        # UnspokenFilter leaves the reply empty, and nothing is spoken
        expected_failures=["T1:*"],
        needs=("stub",),
        tags=["search", "switch"],
    ),
]


# Search, then read the page; a model that keeps searching ----------------------------------------

KESTREL_PAGE = "The winning goal came in the 88th minute, a header by Lu Brandt from a corner."
KESTREL_SEARCH = [{"title": "Kestrel Cup final report", "body": "Harbor City beat Northfield 3-1.",
                   "href": "https://sport.example/kestrel-final"}]
PAGES = {
    "https://sport.example/kestrel-final": KESTREL_PAGE,
    "https://sport.example": "Front page: Harbor City celebrate the Kestrel Cup.",
    "https://sport.example/missing": None,
}


def read_after(messages: list[dict]):
    """After a search, read its first result; after reading, answer."""
    tool, content = last_tool(messages)
    if tool == "search":
        found = results(content)
        return Call("browse", {"url": found[0]["href"] if found else "https://sport.example"})
    if content.startswith("Error:"):
        return "I could not open the report."
    return "The winning goal came in the 88th minute: Lu Brandt headed in a corner."


def states_in_order(first: str, second: str, name: str = "state_order") -> Check:
    def fn(result, view):
        seen = [s for _, s in result.states]
        if first not in seen or second not in seen or seen.index(first) > len(seen) - 1 - seen[::-1].index(second):
            return f"states {seen}"
        return None
    return Check(name, fn)


READ_RULES = [
    Rule(r"\bread\b", tool=search(), after_tool=read_after),
    Rule("front page of sport.example", tool=browse("https://sport.example"),
         after_tool="Here is the front page. {result}"),
]


def reading(name: str, turns: list, results_=None) -> Conversation:
    return Conversation(
        name,
        turns,
        tools={"web": True},
        web={"results": {"kestrel": results_ or KESTREL_SEARCH}, "pages": PAGES, "delay": 1.0, "browse_delay": 1.5},
        stub_rules=READ_RULES,
        tags=["search", "reading"],
    )


READING = [
    reading("search, then read the page", [
        Turn(["Find the report on the Kestrel Cup final and read me how the winning goal was scored."],
             tools_used=["search", "browse"], reply_mentions=["88th", "Brandt"], check_with_stub=True,
             checks=[c.browse_of_search_href(), c.flags_in_sequence("searching", "browsing"),
                     states_in_order("searching", "browsing"),
                     Check("first_word", lambda r, v: None if r.first_word is None or r.first_word <=
                           1.0 + 1.5 + 3 * 0.3 + v.runner.budgets["voice_pipeline"] else
                           f"the first word came after {r.first_word:.2f} s")]),
        Turn(["What's on the front page of sport.example?"], tools_used=["browse"]),
    ]),
    Conversation(
        "a model that keeps searching",
        [Turn(["Who won the Kestrel Cup final?"],
              checks=[c.uses_count(3), c.last_request_note("Answer the user now"),
                      Check("first_word", lambda r, v: None if r.first_word is None or r.first_word <=
                            3 * 1.0 + 4 * 0.3 + v.runner.budgets["voice_pipeline"] + 1.0 else
                            f"the first word came after {r.first_word:.2f} s")])],
        tools={"web": True},
        web={"results": {"kestrel": KESTREL_SEARCH}, "delay": 1.0},
        stub_rules=[Rule("who won", tool=search(), loop=5, after_tool="Harbor City won the Kestrel Cup final.")],
        needs=("stub",),
        tags=["search", "reading"],
    ),
    reading("a page that cannot be fetched", [
        Turn(["Find the report on the Kestrel Cup final and read me how the winning goal was scored."],
             checks=[c.tool_message(starts="Error: Could not fetch")]),
    ], results_=[{**KESTREL_SEARCH[0], "href": "https://sport.example/missing"}]),
]


# Web search over the real network ---------------------------------------------------------

LOOK_UPS = [
    "Look up the population of Lisbon.", "Look up who wrote Middlemarch.",
    "Look up the height of Mount Kilimanjaro.", "Look up the year the Eiffel Tower was finished.",
    "Look up the capital of Mongolia.", "Look up the length of the river Danube.",
    "Look up who painted the Night Watch.", "Look up the boiling point of ethanol.",
    "Look up the population of Reykjavik.", "Look up the author of Don Quixote.",
    "Look up the depth of Lake Baikal.", "Look up when the Golden Gate Bridge opened.",
]
READS = [
    "Read the page at https://en.wikipedia.org/wiki/Lisbon",
    "Read the page at https://en.wikipedia.org/wiki/List_of_cities_by_population",
    "Read the page at https://10.255.255.1/",
]


def real_answer(messages: list[dict]) -> str:
    _, content = last_tool(messages)
    if content.startswith("Error:") or content == "No results found.":
        return "Sorry, the search did not work."
    return fill("I found this: {title}.", messages) if results(content) else "I read the page."


def over_the_network(name: str = "network") -> Check:
    """Over the turns so far: errors in at most 1 of the look-ups, nothing
    found in at most 2, and the search windows' spread."""
    def fn(result, view):
        turns = view.runner.results_so_far + [result]
        windows = [b - a for r in turns for a, b in (r.flags.get("searching") or []) if b is not None]
        calls = [w for r in turns for w in r.web_calls if w["kind"] == "search"]
        errors = sum(w["outcome"] == "error" for w in calls)
        empty = sum(w["outcome"] == "nothing" for w in calls)
        problems = []
        if errors > 1:
            problems.append(f"{errors} searches failed")
        if empty > 2:
            problems.append(f"{empty} searches found nothing")
        if windows:
            ordered = sorted(windows)
            middle, p90, worst = ordered[len(ordered) // 2], ordered[int(0.9 * (len(ordered) - 1))], ordered[-1]
            view.runner.report(f"  note: search windows: median {middle:.2f} s, p90 {p90:.2f} s, worst {worst:.2f} s")
            if middle > 2.5 or p90 > 4.0 or worst > 10.0:
                problems.append(f"search windows median {middle:.2f}, p90 {p90:.2f}, worst {worst:.2f} s")
        return "; ".join(problems) or None
    return Check(name, fn)


def page_within(seconds: float, name: str = "page_time") -> Check:
    def fn(result, view):
        reads = [w for w in (view.web.between(view.running.since, view.end) if view.web else []) if w.kind == "browse"]
        if not reads or reads[0].end is None:
            return "no page was read"
        took = reads[0].end - reads[0].start
        return None if took <= seconds else f"the page took {took:.1f} s"
    return Check(name, fn)


def truncated_page(name: str = "truncated") -> Check:
    def fn(result, view):
        found = [t for r in view.requests() for t in r.tool_messages()]
        if not found:
            return "no page reached the model"
        page = found[-1]
        if len(page) > 4000 + len("\n...(truncated)"):
            return f"{len(page)} characters reached the model"
        return None if page.endswith("...(truncated)") else "the long page was not cut"
    return Check(name, fn)


PLAIN_QUESTIONS = [
    ("What is the capital of France?", ["Paris"]), ("How many legs does a spider have?", ["eight", "8"]),
    ("What is the opposite of hot?", ["cold"]), ("What colour is the sky on a clear day?", ["blue"]),
    ("What day comes after Monday?", ["Tuesday"]), ("How many days are in a week?", ["seven", "7"]),
    ("What is two plus two?", ["four", "4"]), ("What do bees make?", ["honey"]),
    ("What is frozen water called?", ["ice"]), ("Which planet do we live on?", ["Earth"]),
]
CURRENT_QUESTIONS = [
    "What's the weather in Berlin right now?", "What's the latest stable Python release?",
    "What's the euro to dollar exchange rate today?", "What's in the news today?", "When is the next full moon?",
]


def model_rounds_at_most(n: int, name: str = "rounds") -> Check:
    return Check(name, lambda r, v: None if r.model_rounds <= n else f"{r.model_rounds} model rounds")


def real_pass(web_on: bool, run: int) -> Conversation:
    turns = [Turn([q], reply_mentions=m) for q, m in PLAIN_QUESTIONS]
    if web_on:
        turns += [Turn([q], checks=[model_rounds_at_most(3), c.no_markup()]) for q in CURRENT_QUESTIONS]
    return Conversation(
        f"a real model with the web {'on' if web_on else 'off'} (run {run})",
        turns,
        tools={"web": web_on},
        needs=("real_model", "real_web"),
        reply_timeout=120.0,
        tags=["search", "network", "web on" if web_on else "web off"],
    )


NETWORK = [
    Conversation(
        "look-ups over the real network",
        [Turn([q]) for q in LOOK_UPS[:-1]] + [Turn([LOOK_UPS[-1]], checks=[over_the_network()])]
        + [Turn([READS[0]], checks=[page_within(5.0)]), Turn([READS[1]], checks=[truncated_page()]),
           Turn([READS[2]], checks=[page_within(10.0, name="unreachable")])],
        tools={"web": True},
        stub_rules=[Rule(r"^read the page at", tool=browse(lambda m: last_user(m).split()[-1]),
                         after_tool=real_answer),
                    Rule(r".", tool=search(lambda m: re.sub(r"(?i)\blook up\b", "", last_user(m)).strip(" ."))
                         , after_tool=real_answer)],
        # trafilatura waits 30 s for a host that never answers
        expected_failures=["T15:unreachable"],
        needs=("stub", "real_web"),
        reply_timeout=90.0,
        tags=["search", "network"],
    ),
] + [real_pass(on, run) for run in (1, 2) for on in (False, True)] + [real_search_timing()]

SEARCH = ([SPRINGFIELD, SEARCH_TIMING] + SEARCH_STOPS + TALKING_WHILE_SEARCHING + FOLLOWING_UP
          + FAILING_SEARCH + WEB_SWITCH + READING)


# Asking over a fan, a café, music or a TV -------------------------------------------------

RIJKS_RESULT = "Rijksmuseum: open daily 9:00 to 17:00, also on 26 December. Tickets 23.50 euros."
PLACES = {"Rijksmuseum": r"rijksmuseum"}


def place_query(messages: list[dict]) -> str:
    named = None
    for text in user_messages(messages):
        named = latest(PLACES, text) or named
    text = last_user(messages)
    return text if not named or named.lower() in text.lower() else f"{text} {named}"


def quiet_before_first_turn(wake: bool, name: str = "lead") -> Check:
    """With only the noise in the room, before the first turn: no turn, no
    tool, nothing played (and no waking for the TV)."""
    def fn(result, view):
        start, end = view.runner._start, view.running.since
        problems = [k for _, k, _ in view.log.since(start, None, end) if k in ("chat.user", "tool.use")]
        if view.speaker.sound_heard(start, end) > 0:
            problems.append("sound")
        if wake and view.log.since(start, "wake", end):
            problems.append("wake")
        return f"before the first turn: {problems}" if problems else None
    return Check(name, fn)


def precision_at_least(share: float, name: str = "precision") -> Check:
    return Check(name, lambda r, v: None if r.heard_precision is None or r.heard_precision >= share else
                 f"{1 - r.heard_precision:.0%} of the words heard were not said: {r.heard!r}")


def in_noise(sound: str, snr: float) -> Conversation:
    tv = sound == "tv"
    wake = "Iris, " if tv else ""
    end_by = 1.0 if snr >= 20 else 2.0
    common = dict(turn_end=(0, end_by), checks=[precision_at_least(0.8), c.query_words_known()])
    first = Turn([f"{wake}Search the web for the opening hours of the Rijksmuseum", 0.5, "on December 26th."],
                 heard_terms=["rijksmuseum", "26|twenty.?six", "dec"],
                 search_mentions=["rijksmuseum", "26|twenty.?six", "dec"],
                 tools_used=["search"], expect="wake" if tv else "answer", **common)
    first.checks = first.checks + [quiet_before_first_turn(tv)]
    must_pass = sound == "fan" or (sound in ("babble", "music") and snr >= 10)
    return Conversation(
        f"asking over {'a ' if sound in ('fan', 'tv') else ''}{sound} at {snr:g} dB",
        [
            first,
            Turn([f"{wake}And how much is a ticket?"], tools_used=["search"], reply_mentions=["23"],
                 check_with_stub=True, **common),
            Turn(["Thanks, that helps."], tools_used=[], **common),
        ],
        noise=[Noise(sound, snr=snr, lead=8.0)],
        wake_word="Iris" if tv else None,
        tools={"web": True},
        web={"results": {"rijksmuseum": RIJKS_RESULT}, "delay": 1.0},
        stub_rules=[Rule(r"search|ticket|how much", tool=search(place_query))],
        reply_timeout=30.0,
        # At 5 dB, and over the TV, what is heard is reported, not judged
        expected_failures=[] if must_pass else ["T1:*", "T2:*", "T3:*"],
        tags=["hearing", "noise", sound, f"{snr:g} dB"],
    )


NOISE = [in_noise(sound, snr) for sound in ("fan", "babble", "music", "tv") for snr in (20, 10, 5)]


# A distant user, and loud speakers --------------------------------------------------------

DISTANCES = {
    "at arm's length": (1.0, None),
    "1 m away": (0.3, (0.4, 6.0)),
    "3 m away": (0.1, (0.6, 0.0)),
    "5 m away": (0.05, (0.7, -4.0)),
}


def no_silent_drop(name: str = "dropped") -> Check:
    """An understanding state after the user's speech, with no turn heard
    before Iris listens again, is a turn recognized as nothing."""
    def fn(result, view):
        events = view.events("state", view.running.ref_end)
        for index, (at, _, data) in enumerate(events):
            if data.get("state") != "understanding":
                continue
            following = [t for t, _, d in events[index + 1:] if d.get("state") in ("listening", "dormant")]
            until = following[0] if following else view.end
            if not view.log.since(at, "chat.user", until):
                return "recognized nothing, said nothing"
        return None
    return Check(name, fn)


def no_wake_event(name: str = "woke") -> Check:
    return Check(name, lambda r, v: "it woke" if v.events("wake") else None)


def from_a_distance(label: str) -> Conversation:
    level, room = DISTANCES[label]
    said = dict(level=level, room=room, checks=[no_silent_drop()])
    return Conversation(
        f"a user {label}",
        [
            Turn(["What's the time in Tokyo right now?"], expect="silent", **{**said, "checks": [
                no_silent_drop(), no_wake_event()]}),
            Turn(["Iris,", 0.4, "look up the weather in Lisbon for tomorrow."], expect="wake",
                 search_mentions=["lisbon", "tomorrow"], **{**said, "checks": [no_silent_drop(), c.one_user()]}),
            Turn(["And in Porto?"], search_mentions=["porto"], **said),
            Turn(["Okay, that's all, thanks."], expect="sleep", **said),
        ],
        wake_word="Iris",
        noise=[Noise("fan", rms=0.002)],
        tools={"web": True},
        web={"results": dict(CITY_RESULTS), "delay": 1.0},
        stub_rules=[Rule(r"lisbon|porto", tool=search())],
        expected_failures=["T1:*", "T2:*", "T3:*", "T4:*"] if label == "5 m away" else [],
        tags=["hearing", "distance", label],
    )


def holds_per_minute(limit: float, name: str = "holds") -> Check:
    def fn(result, view):
        minutes = max(1 / 60, (result.audible or 0) / 60)
        rate = len(result.holds) / minutes
        return None if rate <= limit or len(result.holds) <= 1 else f"{len(result.holds)} holds in {minutes:.1f} min"
    return Check(name, fn)


def loud(gain: float, drive) -> Conversation:
    return Conversation(
        f"loud speakers at {gain:g}" + (f", driven at {drive:g}" if drive else ""),
        [
            Turn(["Tell me a long story about a lighthouse keeper."]),
            Turn(["Wait, stop.", 0.3, "Search the web for the tide times in Brest today."], at=At.reply_audible(3.0),
                 listen_through=True, heard_terms=["brest", "tide"], search_mentions=["brest", "tide"],
                 spoken_once=["look that up"], checks=[c.uses_count(1), c.web_queries(1), c.no_continued()]),
            Turn(["Explain in detail how tides work."], listen_through=True,
                 checks=[c.one_user(), holds_per_minute(1)]),
            Turn(["Thanks."], level=0.3),
        ],
        echo=Echo(gain=gain, room=0.5, drive=drive),
        fresh=True,
        tools={"web": True},
        web={"results": {"brest": "Brest tides today: high water at 06:42 and 19:05."}, "delay": 2.0},
        stub_rules=[Rule(r"search the web", lead_in=LOOK_IT_UP, tool=search())],
        expected_failures=["T2:*", "T3:*", "T4:*"] if gain > 1.0 else [],
        tags=["hearing", "loud speakers", f"echo {gain:g}"],
    )


DISTANCE = [from_a_distance(label) for label in DISTANCES] + [
    loud(0.3, None), loud(0.9, None), loud(0.9, 2.0), loud(1.5, None), loud(1.5, 2.0)]


# Other user voices -------------------------------------------------------------------------

OTHER_VOICES = ("emma", "nicole", "indian_en", "french_en", "pocket_female")
EVERYDAY_FOR_VOICES = ("a question and a follow-up", "a pause in the middle of a sentence",
                       "interrupting a long answer", "the wake word")


def in_voice(conversation: Conversation, voice: str) -> Conversation:
    copy = Conversation(**{**conversation.__dict__})
    copy.name = f"{conversation.name} ({voice})"
    copy.voice = voice
    copy.tags = list(conversation.tags) + ["voices", voice]
    if voice == "pocket_female":
        # Close to Iris's own voice: its barge-in is reported on its own
        copy.expected_failures = list(conversation.expected_failures) + ["barge_in"]
    return copy


def voice_search(voice: str) -> Conversation:
    return Conversation(
        f"a search and an interruption ({voice})",
        [
            Turn(["Search the web for the population of Reykjavík."], search_mentions=["reykjav"]),
            Turn(["Tell me a long story about Iceland."]),
            Turn(["Sorry, stop. How cold is it there in winter?"], at=At.reply_audible(3.0)),
        ],
        voice=voice,
        tools={"web": True},
        web={"results": {"reykjav": "Reykjavik: population about 140,000."}, "delay": 1.0},
        stub_rules=[Rule("search", tool=search())],
        expected_failures=["barge_in"] if voice == "pocket_female" else [],
        tags=["hearing", "voices", voice],
    )


VOICES_GROUP = [in_voice(conv, voice) for voice in OTHER_VOICES for conv in CONVERSATIONS
                if conv.name in EVERYDAY_FOR_VOICES] + [voice_search(voice) for voice in OTHER_VOICES]


# Languages ---------------------------------------------------------------------------------

LANGUAGE_NAMES = {"German": "de", "French": "fr", "Italian": "it", "Spanish": "es", "Japanese": "ja",
                  "Dutch": "nl", "Portuguese": "pt", "Chinese (Mandarin)": "zh"}
CANNED = {
    "en": "Here is my answer. {first}",
    "de": "Hier ist meine Antwort. {first}",
    "fr": "Voici ma réponse. {first}",
    "it": "Ecco la mia risposta. {first}",
    "es": "Aquí está mi respuesta. {first}",
    "ja": "お答えします。{first}",
    "nl": "Hier is mijn antwoord. {first}",
}
SEARCH_VERBS = r"\bsearch\b|\blook up\b|\bsuch\b|\bcherche\b|\bbusca\b|\bcerca\b|調べ"


def asked_language(messages: list[dict]) -> str:
    """The language Iris's note asks for ("Respond in German."), English
    without a note."""
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        for note in notes_of(message):
            found = re.search(r"Respond in (.+?)\.$", note)
            if found:
                return LANGUAGE_NAMES.get(found.group(1), "en")
        return "en"
    return "en"


def in_language(messages: list[dict]) -> str:
    language = asked_language(messages)
    return CANNED.get(language, CANNED["en"]).format(first="").strip()


def in_language_after_tool(messages: list[dict]) -> str:
    return fill(CANNED.get(asked_language(messages), CANNED["en"]), messages)


LANGUAGE_RULES = [Rule(SEARCH_VERBS, tool=search(), after_tool=in_language_after_tool),
                  Rule(r".", reply=in_language)]
LANGUAGE_WEB = {"results": {
    "neuschwanstein": "Schloss Neuschwanstein: open 9:00 to 18:00, tickets 21 euros.",
    "orsay": "Musée d'Orsay: tickets 16 euros.",
    "hamburg": "Hamburg tomorrow: 11 degrees and windy.",
    "bohemian|rhapsody": "Bohemian Rhapsody: released in 1975 by Queen.",
    "n(ü|ue|u)rnberg|nuremberg": "Christkindlesmarkt Nürnberg 2026: opens on 27 November.",
    "strasbourg": "Marché de Noël de Strasbourg: opens on 21 November.",
    "madrid|sevilla|seville": "Madrid to Sevilla by train: 45 euros.",
    "東京|tokyo": "東京の天気: 明日は晴れ、18度。",
}, "delay": 1.0}


def asks_for_english(name: str = "english_note") -> Check:
    """Back in English, the model is told to answer in English."""
    def fn(result, view):
        requests = view.requests()
        if not requests:
            return None
        return None if requests[0].carries("Respond in English") else "nothing tells the model to answer in English"
    return Check(name, fn)


def english_with_names(noisy: bool) -> Conversation:
    turns = [
        Turn(["What's a good name for a grey cat?"]), Turn(["Yes."]), Turn(["No, the other one."]),
        Turn(["Search for the opening hours of Schloss Neuschwanstein."], search_mentions=["neuschwanstein"]),
        Turn(["Look up tickets for the Musée d'Orsay."], search_mentions=["orsay"]),
        Turn(["Who is Jürgen Klopp?"]), Turn(["Okay."]), Turn(["Hmm, go on."]),
    ]
    for turn in turns:
        turn.reply_language = "en"
    return Conversation(
        "english with names" + (" over a fan" if noisy else ""), turns,
        noise=[Noise("fan", snr=15)] if noisy else [],
        tools={"web": True}, web=LANGUAGE_WEB, stub_rules=LANGUAGE_RULES,
        tags=["hearing", "languages", "en"],
    )


LANGUAGES = [
    english_with_names(False),
    english_with_names(True),
    Conversation(
        "german with short answers",
        [Turn(["Such im Internet nach dem Wetter morgen in Hamburg."], search_mentions=["hamburg"],
              reply_language="de"),
         Turn(["Ja."], reply_language="de"), Turn(["Nein, danke."], reply_language="de"),
         Turn(["Genau."], reply_language="de"), Turn(["Okay, und übermorgen?"], reply_language="de"),
         Turn(["Such mal nach dem Song Bohemian Rhapsody."], search_mentions=["bohemian", "rhapsody"],
              reply_language="de")],
        voice="de", tools={"web": True}, web=LANGUAGE_WEB, stub_rules=LANGUAGE_RULES,
        tags=["hearing", "languages", "de"],
    ),
    Conversation(
        "across languages",
        [
            Turn(["What is the capital of Australia?"], reply_mentions=["Canberra"], reply_language="en"),
            Turn(["Such im Internet, wann der Christkindlesmarkt in Nürnberg 2026 öffnet."], voice="de",
                 search_mentions=["n(u|ue)rnberg|nuremberg", "2026"], reply_language="de"),
            Turn(["Cherche aussi quand ouvre le marché de Noël de Strasbourg."], voice="fr",
                 search_mentions=["strasbourg"], reply_language="fr"),
            Turn(["Grazie mille!"], voice="it", reply_language="it"),
            Turn(["Busca cuánto cuesta un billete de tren de Madrid a Sevilla."], voice="es",
                 search_mentions=["madrid", "sevilla|seville"], reply_language="es"),
            Turn(["明日の東京の天気を調べてください。"], voice="ja", search_mentions=["東京|tokyo"],
                 reply_language="ja"),
            Turn(["Okay, back to English. What time is it in New York now?"], reply_language="en",
                 checks=[asks_for_english()]),
        ],
        tools={"web": True}, web=LANGUAGE_WEB, stub_rules=LANGUAGE_RULES,
        # llm.py adds a language note only for languages other than English
        expected_failures=["T7:english_note"],
        tags=["hearing", "languages", "mixed"],
    ),
]

HEARING = NOISE + DISTANCE + VOICES_GROUP + LANGUAGES


# Typed and spoken turns in one conversation ------------------------------------------------


def typed_as_sent(name: str = "typed") -> Check:
    def fn(result, view):
        typed = c.users(view, "text")
        if not typed:
            return "no typed turn"
        shown = typed[-1][1].get("text")
        return None if shown == view.running.result.said else f"shown as {shown!r}"
    return Check(name, fn)


def typed_and_spoken(name: str = "both") -> Check:
    """A typed message sent while the user speaks: both are taken, the
    spoken one whole, and both answered aloud."""
    def fn(result, view):
        typed, spoken = c.users(view, "text"), c.users(view, "voice")
        problems = []
        if len(typed) != 1 or len(spoken) != 1:
            problems.append(f"{len(typed)} typed and {len(spoken)} spoken turns")
        complete = [d for _, d in c.dones(view) if not d.get("interrupted")]
        if len(complete) != 2:
            problems.append(f"{len(complete)} complete replies")
        return "; ".join(problems) or None
    return Check(name, fn)


def users_in_order(texts: list[str], name: str = "history") -> Check:
    def fn(result, view):
        requests = view.requests()
        if not requests:
            return None
        found = [u.lower() for u in requests[0].users()]
        position = 0
        for text in texts:
            hits = [i for i, u in enumerate(found) if i >= position and text.lower() in u]
            if not hits:
                return f"{text!r} is not in the request in order"
            position = hits[0] + 1
        return None
    return Check(name, fn)


TYPED_AND_SPOKEN = [
    Conversation(
        "typed and spoken turns",
        [
            Turn(["What is the capital of Japan?"], reply_mentions=["Tokyo"]),
            Turn(["And roughly how many people live there?"], typed=True, reply_mentions=["million"],
                 checks=[typed_as_sent()]),
            Turn(["Tell me a long story about a lighthouse keeper."], typed=True),
            Turn(["Sorry, stop. What is two plus two?"], at=At.reply_audible(3.0)),
            Turn(["Can you tell me about", 1.2, "the first landing on the moon?"],
                 actions=[(At.speech_start(1.0), ("send", "Also, what is the opposite of hot?"))],
                 checks=[typed_and_spoken(), c.no_continued()]),
            Turn(["Which was the first thing I asked you today?"], reply_mentions=["Japan", "Tokyo"],
                 checks=[users_in_order(["capital of Japan", "how many people", "lighthouse", "two plus two",
                                         "landing on the moon"])]),
        ],
        # A typed turn's reply is spoken with no listener, so the voice runs on;
        # a queued typed turn ends listen() and throws the utterance away
        expected_failures=["T4:*", "T5:*"],
        tags=["conversation", "typed"],
    ),
]


# Waking and sleeping ----------------------------------------------------------------------


def woke_within(seconds: float, name: str = "woke") -> Check:
    def fn(result, view):
        woke = view.events("wake")
        if not woke:
            return "did not wake"
        took = woke[0][0] - view.running.ref_end
        return None if took <= seconds else f"woke {took:.1f} s after the user stopped"
    return Check(name, fn)


def request_heard(request: str, name: str = "request") -> Check:
    """The request after the wake word heard, its first words included."""
    def fn(result, view):
        problems = []
        share = c.heard_share(result.heard, request)
        if share < 0.75:
            problems.append(f"heard {share:.0%} of the request: {result.heard!r}")
        first = normalized(request).split()[:2]
        if first and not all(w in normalized(result.heard).split() for w in first):
            problems.append(f"the request's first words {first} are missing")
        return "; ".join(problems) or None
    return Check(name, fn)


def sleep_turn() -> Turn:
    return Turn(["That's all."], expect="sleep", chime="falling")


def wake_with(parts: list) -> Turn:
    request = parts[-1]
    return Turn(parts, expect="wake", checks=[request_heard(request), c.one_user()])


WAKE_FORMS = Conversation(
    "wake word forms",
    [
        Turn(["I was telling my sister about Iris yesterday."], expect="silent", checks=[no_wake_event()]),
        Turn(["Irish coffee is my favourite drink."], expect="silent", checks=[no_wake_event()]),
        Turn(["Irises are blooming in my garden."], expect="silent", checks=[no_wake_event()]),
        Turn(["Iris."], expect="wake", chime="rising"),
        Turn(["What is the capital of Italy?"], at=At.reply_end(1.5)),
        sleep_turn(),
        wake_with(["Iris", 0.5, "what is the capital of Spain?"]), sleep_turn(),
        wake_with(["Iris", 0.9, "what is the capital of Portugal?"]), sleep_turn(),
        wake_with(["Iris", 2.5, "what is the capital of Greece?"]), sleep_turn(),
        wake_with(["Hey Iris, what is the capital of Austria?"]), sleep_turn(),
        Turn(["So, Iris, what is the capital of Norway?"], expect="wake", checks=[probe("a wake word inside")]),
    ],
    wake_word="Iris",
    # A per-word similarity of 0.75 takes "Irish" and "Irises" for "Iris"
    expected_failures=["T2:*", "T3:*", "T15:*"],
    tags=["conversation", "wake word"],
)

WAKE_LANGUAGES = {
    "de": (["Hallo Iris,", 0.3, "such mal das Wetter in Berlin."], "berlin", "Was gibt es heute Neues?"),
    "es": (["Hola Iris,", 0.3, "busca el tiempo en Madrid."], "madrid", None),
    "nl": (["Hoi Iris,", 0.3, "zoek het weer in Amsterdam."], "amsterdam", None),
    "it": (["Ciao Iris,", 0.3, "cerca il meteo a Roma."], "roma", None),
    "fr": (["Salut Iris,", 0.3, "cherche la météo à Lyon."], "lyon", None),
    "ja": (["アイリス、", 0.3, "大阪の天気を調べて。"], "大阪|osaka", None),
    "french_en": (["Hey Iris,", 0.3, "search the weather in Paris."], "paris", None),
}


def wake_in(language: str) -> Conversation:
    parts, city, before = WAKE_LANGUAGES[language]
    spoken = "en" if language == "french_en" else language
    turns = [Turn([before], expect="silent")] if before else []
    turns.append(Turn(parts, expect="wake", reply_language=spoken, search_mentions=[city], checks=[woke_within(4.0)]))
    return Conversation(
        f"waking Iris in {language}",
        turns,
        wake_word="Iris",
        voice=language,
        tools={"web": True},
        web={"results": {city: f"The weather in {city.split('|')[0].title()}: 15 degrees."}, "delay": 1.0},
        stub_rules=[Rule(SEARCH_VERBS + r"|zoek|cerca|busca", tool=search(), after_tool=in_language_after_tool)],
        # GREETINGS lacks hallo, hola and hoi, and Whisper writes the name in katakana
        expected_failures=["*"] if language in ("de", "es", "nl", "ja") else [],
        tags=["conversation", "wake word", language],
    )


def slept_on_time(after: float, name: str = "slept") -> Check:
    """Asleep the window's length after it was last started (+0.6/-0.2 s),
    a falling chime with it, and a dormant state."""
    def fn(result, view):
        windows = [at for at, _, d in view.log.since(0, "wake.window", view.running.since)
                   if d.get("remaining", 0) > 0]
        slept = view.events("sleep", 0)
        slept = [at for at, _, _ in slept if windows and at > windows[-1]]
        if not windows or not slept:
            return "did not fall asleep"
        late = slept[0] - windows[-1] - after
        problems = [] if -0.2 <= late <= 0.6 else [f"fell asleep {late:+.2f} s off the window"]
        chimes = [k for k in view.speaker.clips_between(slept[0] - 0.1, slept[0] + 0.3)
                  if c.chime_of(k) == "falling"]
        if not chimes:
            problems.append("no falling chime")
        return "; ".join(problems) or None
    return Check(name, fn)


def window_from_reply_end(seconds: float, name: str = "window") -> Check:
    def fn(result, view):
        done = view.events("chat.done")
        if not done:
            return "no reply"
        after = [d.get("remaining") for _, _, d in view.log.since(done[-1][0], "wake.window", view.end)]
        return None if seconds in after else f"no wake.window of {seconds} s after the reply: {after}"
    return Check(name, fn)


def no_sleep_during_speech(name: str = "slept_mid_turn") -> Check:
    def fn(result, view):
        said = view.running.said
        if view.log.since(said.speech_start, "sleep", said.speech_end):
            return "fell asleep while the user spoke"
        chimes = [k for k in view.speaker.clips_between(said.speech_start, said.speech_end) if c.chime_of(k)]
        return "a chime while the user spoke" if chimes else None
    return Check(name, fn)


AWAKE_WINDOW = Conversation(
    "the awake window",
    [
        Turn(["Iris, what is the capital of Italy?"], expect="wake"),
        Turn(["And of Spain?"], at=At.finished(3.0)),
        Turn(["And what about the capital of", 0.4, "Portugal?"], at=At.event("wake.window", -0.7),
             checks=[no_sleep_during_speech()]),
        Turn(["And of Greece?"], at=At.event("wake.window", 2.0), expect="silent", checks=[slept_on_time(8.0)]),
        Turn(["Iris, look up the weather in Lyon tomorrow."], expect="wake", reply_mentions=["14"],
             check_with_stub=True, checks=[window_from_reply_end(8.0)]),
        Turn(["And the day after?"], at=At.finished(5.0)),
        Turn(["Iris, what is the capital of France?"], at=At.event("wake.window", 0.0), expect="wake",
             checks=[no_sleep_during_speech()]),
        Turn(["Tell me a long story about the sea."]),
        Turn(["That's all, thanks."], at=At.reply_audible(3.0), expect="sleep", chime="falling",
             checks=[c.silent_after_user_began(1.0)]),
        Turn(["What is two plus two?"], expect="silent"),
        Turn(["Iris, that's all I wanted to know about Lisbon. What about Porto?"], expect="wake"),
        Turn(["That is all, Iris."], expect="sleep"),
        Turn(["Iris, which capitals did we talk about?"], expect="wake", reply_mentions=["Rome", "Madrid"],
             checks=[c.request_holds(["Italy", "Spain", "Portugal", "Lyon", "France"],
                                     absent=["That's all", "That is all"])]),
    ],
    wake_word="Iris",
    wake_options={"window": 8.0},
    tools={"web": True},
    web={"results": {"lyon": FOLLOW_UP_RESULTS["lyon"]}, "delay": 6.0},
    stub_rules=[Rule(r"look up|day after", tool=search(topic_query))],
    # The sleep word's lists differ: "That is all" does not put it to sleep
    expected_failures=["T12:*"],
    tags=["conversation", "wake word", "window"],
)

WAKING = [WAKE_FORMS] + [wake_in(language) for language in WAKE_LANGUAGES] + [AWAKE_WINDOW]


# Switching persona mid-conversation ---------------------------------------------------------


def voice_is(voice: str, name: str = "voice") -> Check:
    return Check(name, lambda r, v: None if not r.components or r.components.get("voice") == voice else
                 f"spoken by {r.components.get('voice')!r}")


def no_clip_after_done(name: str = "late_clip") -> Check:
    def fn(result, view):
        done = c.dones(view)
        if not done:
            return None
        late = [k for k in c.reply_clips(view, done[-1][0]) if k.started > done[-1][0] + 0.2 and done[-1][1].get(
            "interrupted")]
        return f"{len(late)} clips after the reply was given up" if late else None
    return Check(name, fn)


def notes_in_order(texts: list[str], name: str = "notes") -> Check:
    def fn(result, view):
        requests = view.requests()
        if not requests:
            return None
        joined = " ".join(m.get("content") or "" for m in requests[0].messages)
        positions = [joined.find(t) for t in texts]
        if -1 in positions or positions != sorted(positions):
            return f"the notes {texts} are not in the request in order"
        return None
    return Check(name, fn)


PERSONAS = [
    Conversation(
        "switching persona mid-conversation",
        [
            Turn(["What is the capital of France?"], reply_mentions=["Paris"], checks=[voice_is("kyutai:alba")]),
            Turn(["And roughly how many people live there?"], actions=[(At.finished(0.0), ("persona", "hal"))],
                 at=At.event("config", 0.6), reply_mentions=["million"],
                 checks=[voice_is("kokoro:bm_lewis"), c.request_holds(["HAL 9000", "You are now HAL"])]),
            Turn(["Explain in detail how a rainbow forms."], listen_through=True, checks=[voice_is("kokoro:bm_lewis")]),
            Turn(["Sorry, stop. What is two plus two?"], at=At.reply_audible(3.0),
                 checks=[voice_is("kokoro:bm_lewis")]),
            Turn(["Tell me a long story about the sea."], actions=[(At.reply_audible(2.0), ("persona", "computer"))],
                 checks=[no_clip_after_done()]),
            Turn(["What day comes after Monday?"], reply_mentions=["Tuesday"],
                 checks=[voice_is("kokoro:af_sarah"), notes_in_order(["You are now HAL", "You are now Computer"])]),
            Turn(["What was the first thing I asked you?"], actions=[(At.finished(0.0), ("persona", "iris"))],
                 at=At.event("config", 0.6), reply_mentions=["France", "Paris"], checks=[voice_is("kyutai:alba")]),
        ],
        tags=["conversation", "persona"],
    ),
]

CONVERSATION = TYPED_AND_SPOKEN + WAKING + PERSONAS


# A slow model -----------------------------------------------------------------------------


def reply_stage_near(seconds: float, name: str = "reply_stage") -> Check:
    def fn(result, view):
        reply = result.stages.get("reply")
        if reply is None:
            return "no model stage measured"
        return None if abs(reply / 1000 - seconds) <= 0.5 else f"the model stage took {reply / 1000:.2f} s"
    return Check(name, fn)


def thinking_until_first_word(name: str = "thinking") -> Check:
    def fn(result, view):
        states = result.states
        understood = [at for at, s in states if s == "understanding"]
        if not understood or result.first_word is None:
            return None
        after = [s for at, s in states if understood[-1] < at < result.first_word and s != "understanding"]
        bad = [s for s in after if s not in ("thinking", "speaking")]
        return f"states while it thought: {bad}" if bad else None
    return Check(name, fn)


def stub_closed_after_stop(limit: float, name: str = "closed") -> Check:
    def fn(result, view):
        pressed = c.actions(view, "stop", view.running.since)
        requests = view.requests()
        if not pressed or not requests:
            return None
        closed = [r.closed for r in requests if r.closed]
        if not closed:
            return "the model's request was not cut off"
        return None if closed[0] - pressed[0] <= limit else f"cut off {closed[0] - pressed[0]:.2f} s after Stop"
    return Check(name, fn)


def first_request_closed_soon(limit: float, name: str = "closed") -> Check:
    def fn(result, view):
        previous = c.turn_start(view, view.running.number - 1)
        requests = view.stub.requests_between(previous, view.end) if view.stub and previous else []
        said = view.running.said
        if not requests or said is None:
            return None
        if requests[0].closed is None:
            return "the first request was not cut off"
        took = requests[0].closed - said.speech_start
        return None if took <= limit else f"the first request was cut off {took:.2f} s after the user went on"
    return Check(name, fn)


def slow_model(seconds: float, japan: float) -> Conversation:
    rules = [
        Rule("tides", "Tides come from the Moon's pull on the oceans.", first_token=seconds),
        Rule("sky blue", "The sky is blue because air scatters blue light most.", first_token=seconds),
        Rule("volcano", "Volcanoes are openings where molten rock reaches the surface.", first_token=seconds),
        Rule("two plus two", "Two plus two is four.", first_token=0.3),
        Rule("japan|korea", "Seoul is a big city in Korea.", first_token=japan),
    ]
    return Conversation(
        f"a slow model ({seconds:g} s)",
        [
            Turn(["Explain how the tides work."], checks=[reply_stage_near(seconds), thinking_until_first_word()]),
            Turn(["Why is the sky blue?"], during=[(At.speech_end(4.0), Say(["Hmm."]))], model_requests=1),
            Turn(["Tell me about volcanoes."], actions=[(At.speech_end(3.0), "stop")],
                 checks=[stub_closed_after_stop(1.0)]),
            Turn(["What is two plus two?"], at=At.event("action.stop", 1.0),
                 checks=[Check("first_word", lambda r, v: None if r.first_word is None or r.first_word <=
                               v.runner.budgets["voice_pipeline"] + 0.8 else
                               f"the first word came after {r.first_word:.2f} s")]),
            Turn(["Name a big city in Japan."]),
            Turn(["Actually, make that a city in Korea."], at=At.speech_end(5.0), expect="continued",
                 checks=[c.one_user_message_with(["Japan", "Korea"]), first_request_closed_soon(0.5)]),
        ],
        stub_rules=rules,
        stub_options={"serial": True},
        reply_timeout=90.0,
        # Speech while Iris is silent goes straight to on_speech, and the
        # question is asked again
        expected_failures=["T2:model_requests", "T2:aside"],
        tags=["performance", "slow model"],
    )


SLOW_MODEL = [slow_model(15.0, 12.0), slow_model(5.0, 5.0), slow_model(20.0, 20.0)]


# A long conversation that outgrows the context --------------------------------------------

SEED = []
for topic in ("gardening", "trains", "cooking", "music", "astronomy", "chess"):
    SEED += [{"role": "user", "content": f"Tell me about {topic}, please. " * 4},
             {"role": "assistant", "content": f"Here is a long answer about {topic}. " * 13}]


def summary_used(name: str = "summary") -> Check:
    """A reply request after a summary carries the summary, not the fallback."""
    def fn(result, view):
        requests = [r for r in view.requests()]
        summaries = [r for r in requests if r.kind == "summary"]
        replies = [r for r in requests if r.kind == "reply"]
        if not summaries or not replies:
            return None
        if any(r.carries("Previous conversation summary: SUMMARY") for r in replies):
            return None
        return "the reply was asked without the summary"
    return Check(name, fn)


def summaries_spread(name: str = "summaries") -> Check:
    """At most one summary per four turns once the context is full."""
    def fn(result, view):
        turns = view.runner.results_so_far + [result]
        counts = [sum(q["kind"] == "summary" for q in r.requests) for r in turns]
        crossed = next((i for i, n in enumerate(counts) if n), None)
        if crossed is None:
            return None
        after = counts[crossed:]
        allowed = 1 + (len(after) - 1) // 4
        return None if sum(after) <= allowed else f"{sum(after)} summaries in {len(after)} turns"
    return Check(name, fn)


LONG_CONTEXT = [
    Conversation(
        "a conversation that outgrows the context",
        [
            Turn(["Tell me in detail about rivers."], listen_through=True, checks=[summary_used()]),
            Turn(["Tell me in detail about mountains."], listen_through=True, checks=[summary_used()]),
            Turn(["Tell me in detail about deserts."], listen_through=True, checks=[summary_used()]),
            Turn(["What is the capital of France?"], checks=[summary_used()]),
            Turn(["How many legs does a spider have?"], checks=[summary_used()]),
            Turn(["What is the opposite of hot?"], checks=[summary_used()]),
            Turn(["What colour is the sky?"], checks=[summary_used()]),
            Turn(["What day comes after Monday?"], checks=[summary_used()]),
            Turn(["Tell me in detail about forests."],
                 actions=[(At.event("stub.request", 1.0, where=lambda d: d.get("kind") == "summary"), "stop")],
                 checks=[c.no_request_after("stop", ["about forests"], tool_messages=False)]),
            Turn(["Tell me a long story about the sea."], at=At.event("action.stop", 1.0), listen_through=True),
            Turn(["What was that about?"], checks=[c.assistant_before_last_user(LONG), summaries_spread()]),
        ],
        seed_history=SEED,
        stub_options={"context_length": 2048, "summary_delay": 3.0, "first_token": 0.3},
        # manage_context neither keeps the summary nor trims the history, and
        # Stop during the summary does not stop the reply request
        expected_failures=["summaries", "T9:stale_request"],
        needs=("stub",),
        tags=["performance", "context"],
    ),
]


# A session of 60 turns ----------------------------------------------------------------------

ONE_WORD = ["Yes.", "No.", "Thanks.", "Okay.", "Go on.", "Really?", "Sure, why not?"]
SHORT_QUESTIONS = ["What is the capital of France?", "How many legs does a spider have?",
                   "What is the opposite of hot?", "What colour is the sky?", "What day comes after Monday?",
                   "Which planet is closest to the sun?", "What do bees make?", "How many days are in a week?"]
LONG_QUESTIONS = [
    "Could you explain why the days are longer in summer than in winter in the northern hemisphere please?",
    "I would really like to understand how airplanes manage to stay up in the air for such a long time.",
    "Can you tell me why bread goes stale so quickly when it is left out on the kitchen counter overnight?",
    "What is the reason that the ocean is salty while most of the rivers that flow into it are fresh?",
    "Please tell me how a refrigerator keeps food cold even though the kitchen around it is quite warm.",
]
STORY_TOPICS = ["a lighthouse keeper", "the sea", "a dragon", "a robot", "a garden"]
LOOK_UP_TOPICS = ["the weather in Lisbon", "the weather in Porto", "the weather in Paris",
                  "the weather in Lyon", "the weather in Berlin"]
LENGTHS = (30, 80, 150, 250)


def by_request_index(messages: list[dict]) -> str:
    """A reply whose first sentence is 30, 80, 150 or 250 characters long,
    in turn, so the voice's first piece varies in length."""
    count = len(user_messages(messages))
    length = LENGTHS[count % len(LENGTHS)]
    words = ("This sentence is here to be read aloud " * 12).split()
    first = ""
    for word in words:
        if len(first) + len(word) + 1 >= length - 1:
            break
        first = f"{first} {word}".strip()
    return f"{first}. That is all."


def session_checks(name: str = "session") -> Check:
    """Over the session: the voice pipeline's spread, turn ends per kind of
    turn, drift of the first word, and the process's resources."""
    def fn(result, view):
        turns = [r for r in view.runner.results_so_far if not r.skipped] + [result]
        budget = view.runner.budgets["voice_pipeline"]
        problems = []
        pipeline = sorted(r.voice_pipeline for r in turns if r.voice_pipeline is not None)
        if pipeline:
            middle, p90, worst = pipeline[len(pipeline) // 2], pipeline[int(0.9 * (len(pipeline) - 1))], pipeline[-1]
            if middle > budget or p90 > budget + 0.5 or worst > middle + 1.5:
                problems.append(f"voice pipeline median {middle:.2f}, p90 {p90:.2f}, worst {worst:.2f} s")
        for label in ("one-word", "short", "long"):
            ends = sorted(r.stages.get("turn_end", 0) / 1000 for r in turns if r.label == label and r.stages)
            if ends and ends[int(0.9 * (len(ends) - 1))] > 0.8:
                problems.append(f"{label} turns: turn end p90 {ends[int(0.9 * (len(ends) - 1))]:.2f} s")
        firsts = [(i, r.first_word) for i, r in enumerate(turns) if r.first_word is not None and r.label == "short"]
        if len(firsts) > 10:
            xs = [i for i, _ in firsts]
            ys = [f for _, f in firsts]
            mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
            slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / max(
                1e-9, sum((x - mean_x) ** 2 for x in xs))
            if slope > 0.010:
                problems.append(f"the first word drifts {slope * 1000:.0f} ms per turn")
        samples = [r.resources for r in turns if r.resources]
        if len(samples) > 30:
            rss = [s.get("rss_mb") for s in samples if s.get("rss_mb")]
            if rss and rss[-1] - rss[min(4, len(rss) - 1)] > 150:
                problems.append(f"memory grew {rss[-1] - rss[4]:.0f} MB")
            if samples[-1].get("new_wavs", 0) > 1:
                problems.append(f"{samples[-1]['new_wavs']} temporary WAVs left behind")
            fds = [s.get("fds") for s in samples if s.get("fds")]
            if fds and max(fds[5:]) - min(fds[5:]) > 6:
                problems.append(f"open files from {min(fds[5:])} to {max(fds[5:])}")
        return "; ".join(problems) or None
    return Check(name, fn)


def session_of_60() -> Conversation:
    turns: list = [Turn([q], name="short") for q in SHORT_QUESTIONS[:5]]
    turns.append(Silence(120.0))
    for block in range(5):
        block_turns = []
        for i in range(3):
            block_turns.append(Turn([ONE_WORD[(block * 3 + i) % len(ONE_WORD)]], exact=True, name="one-word"))
        for i in range(3):
            block_turns.append(Turn([SHORT_QUESTIONS[(block * 3 + i) % len(SHORT_QUESTIONS)]], name="short"))
        block_turns.append(Turn([LONG_QUESTIONS[block]], name="long"))
        block_turns.append(Turn([f"Tell me a long story about {STORY_TOPICS[block]}."], listen_through=True,
                                name="story"))
        block_turns.append(Turn(["Tell me a long story about the sea."], stop_after=1.5, name="stop"))
        block_turns.append(Turn(["Tell me a long story about a dragon."], name="story"))
        block_turns.append(Turn(["Sorry, stop. What is two plus two?"], at=At.reply_audible(2.0), name="interrupt"))
        block_turns.append(Turn([f"Can you look up {LOOK_UP_TOPICS[block]}?"], name="search"))
        for index, turn in enumerate(block_turns):
            if turn.at is None and turn.interrupt_after is None and index % 2 == 1:
                turn.at = At.reply_end(0.25)
        turns += block_turns
    turns.append(Silence(30.0))
    last = next(t for t in reversed(turns) if isinstance(t, Turn))
    last.checks = list(last.checks) + [session_checks()]
    return Conversation(
        "a session of 60 turns",
        turns,
        tools={"web": True},
        web={"results": {"weather": "The weather: mild, 18 degrees."}, "delay": 1.0},
        stub_rules=[Rule(r"look up", tool=search()), Rule(r"two plus two", "Two plus two is four."),
                    Rule(lambda m: not re.search(r"\b(story|long|in detail)\b", last_user(m), re.I),
                         reply=by_request_index)],
        stub_options={"first_token": 0.3},
        tags=["performance", "session"],
    )


PERFORMANCE = SLOW_MODEL + LONG_CONTEXT


# A model server in trouble ------------------------------------------------------------------

# What a reply that stalls sends: a sentence for the voice, and a word more
STALLED = "The bicycle was invented in the early nineteenth century. And"


def gave_up_on_the_stall(name: str = "gave_up") -> Check:
    """The reply stalled, and Iris hung up once nothing had come for the
    read timeout: not sooner, and not much later."""
    def fn(result, view):
        stalled = [r for r in view.requests() if r.stalled]
        if not stalled:
            return "the reply never stalled"
        if stalled[0].closed is None:
            return "Iris did not give up on the reply"
        waited = stalled[0].closed - stalled[0].stalled
        if not READ_TIMEOUT - 1 <= waited <= READ_TIMEOUT + 5:
            return f"Iris gave up {waited:.1f} s after the reply stalled"
        return None
    return Check(name, fn)


MODEL_TROUBLE = [
    Conversation(
        "a model server that stops answering",
        [
            # The first sentence is said; then nothing comes, and the turn
            # ends with an error that says so
            Turn(["Tell me about the history of the bicycle."], error_mentions=["stopped answering"],
                 checks=[gave_up_on_the_stall()]),
            Turn(["What is the capital of France?"], reply_mentions=["Paris"], check_with_stub=True),
        ],
        stub_rules=[Rule("bicycle", f"{STALLED} then came the pedals.", stall_after=len(STALLED.split()))],
        reply_timeout=READ_TIMEOUT + 60,
        needs=("stub",),
        tags=["model", "timeout"],
    ),
]


# The groups --------------------------------------------------------------------------------

TURN_TAKING = ([answering_at_once(shift) for shift in (0.0, 0.3, 0.6)] + [SPRINGFIELD] + GOING_ON + ONE_BREATH
               + LISTS_AND_CODES + TURN_DETECTION)
INTERRUPTIONS = BACK_CHANNELS + BARGE_IN + STOP_LATE + PAUSE_AND_MIC

GROUPS = {
    "basic": CONVERSATIONS,
    "search": SEARCH,
    "turn-taking": TURN_TAKING,
    "interruptions": INTERRUPTIONS,
    "hearing": HEARING,
    "conversation": CONVERSATION,
    "performance": PERFORMANCE,
    "model": MODEL_TROUBLE,
    # Long runs (pytest -m voice_slow): Whisper's sizes, a session of 60 turns
    "slow": whisper_cells() + [session_of_60()],
    # The real web (pytest -m voice_network, IRIS_VOICETEST_WEB=real)
    "network": NETWORK,
}
# The groups held by default (pytest -m voice)
DEFAULT_GROUPS = ("basic", "search", "turn-taking", "interruptions", "hearing", "conversation", "performance",
                  "model")


def group(*names: str) -> list[Conversation]:
    """The conversations of these groups, each once, in order."""
    seen, out = set(), []
    for name in names:
        if name not in GROUPS:
            raise ValueError(f"No group called {name!r}; there are {', '.join(GROUPS)}")
        for conversation in GROUPS[name]:
            if conversation.name not in seen:
                seen.add(conversation.name)
                out.append(conversation)
    return out


def all_conversations() -> list[Conversation]:
    return group(*GROUPS)
