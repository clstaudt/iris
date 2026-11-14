"""Long-term memory: conversations in threads, each remembered by topic.

A thread is a conversation with one persona, with a memory of its own: its
turns, its topics, the index in its prompt, recall and search. It is stored
with its first turn, so no thread is ever empty. Every turn is stored. The
LLM sees the thread's most recent turns verbatim (the live context) plus a
compact index of the thread's topics, one line each. When the user returns
to something, the model notices it in the index and calls the recall tool
itself; search_memory runs a full-text search over everything ever said in
the thread, for details the index does not show. There are no
embeddings and no similarity thresholds: deciding what to retrieve is the
model's judgment.

Turns are filed into topics by the same LLM, in small focused steps that a
3B model handles reliably: one decision per exchange (worth remembering?
which topic, or a new one?), then one rewrite of each touched topic's
summary and index line.

The conversation always comes first. Filing shares the LLM (and its GPU)
with the conversation, so it only runs once the conversation has been quiet
for a while, checks again before every call, and stops the moment the user
speaks, types or a reply is underway: a call already in flight is cut off
mid-generation (closing its connection makes Ollama stop at once), so a turn
never waits behind filing. Every step is saved as it completes, so stopping
loses nothing; the next quiet spell carries on.
"""

import json
import re
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import ollama

from iris.llm_client import MODEL_TIMEOUT, OpenAIClient, is_openai

DEFAULT_PATH = Path.home() / ".iris" / "memory.db"

# Token budgets (about four characters per token)
INDEX_TOKENS = 1500  # topic index in every prompt, most recent topics first
LIVE_TOKENS = 4000  # verbatim recent turns
PINNED_TOKENS = 1000  # summaries of topics recalled in the last few turns
RECALL_TOKENS = 1500  # what a recall hands the model for the current turn

# Filing starts only after this many seconds without any conversation
QUIET_BEFORE_FILING = 30.0
# Output cap per filing call
FILING_MAX_TOKENS = 400
# How quickly an in-flight filing call is cut off once the conversation resumes
ABORT_CHECK_SECONDS = 0.05

# A recalled topic's summary stays in context for this many user turns
PIN_TURNS = 6

# A thread's title, from its first words until memory has filed a topic
TITLE_LENGTH = 48

SCHEMA = """
-- A thread's id is never given again, so nothing that names a forgotten
-- thread (filing that read it, say) can reach another
CREATE TABLE IF NOT EXISTS threads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    persona TEXT NOT NULL,
    created REAL NOT NULL,
    -- its last turn
    updated REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS topics (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    gist TEXT NOT NULL,
    summary TEXT NOT NULL,
    created REAL NOT NULL,
    updated REAL NOT NULL,
    thread_id INTEGER REFERENCES threads(id)
);
CREATE TABLE IF NOT EXISTS turns (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    source TEXT,
    topic_id INTEGER REFERENCES topics(id) ON DELETE SET NULL,
    filed INTEGER NOT NULL DEFAULT 0,
    -- 0 while the turn is filed under a topic whose summary does not yet
    -- include it
    summarized INTEGER NOT NULL DEFAULT 1,
    thread_id INTEGER REFERENCES threads(id)
);
CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(
    text, content='turns', content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS turns_ai AFTER INSERT ON turns BEGIN
    INSERT INTO turns_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS turns_ad AFTER DELETE ON turns BEGIN
    INSERT INTO turns_fts(turns_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
"""


def estimate_tokens(text: str) -> int:
    return len(text) // 4


def shortened(text: str, length: int = TITLE_LENGTH) -> str:
    """The text on one line, cut at a word within this length."""
    text = " ".join(text.split())
    if len(text) <= length:
        return text
    cut = text[:length].rsplit(" ", 1)[0] or text[:length]
    return cut.rstrip(" ,.;:!?-") + "…"


def relative_day(ts: float, now: Optional[float] = None) -> str:
    days = int(((now or time.time()) - ts) // 86400)
    if days <= 0:
        return "today"
    if days == 1:
        return "yesterday"
    return f"{days} days ago"


@dataclass
class Exchange:
    """A user turn and the reply to it (plus any turns that followed)."""

    turn_ids: list[int]
    text: str
    thread_id: Optional[int] = None


@dataclass
class Topic:
    id: int
    title: str
    gist: str
    summary: str
    created: float
    updated: float
    thread_id: Optional[int] = None
    turns: list[dict] = field(default_factory=list)


class MemoryStore:
    """SQLite storage for threads, their turns and topics, with full-text
    search. What is read for a conversation is read for one thread."""

    def __init__(self, path: str | Path = DEFAULT_PATH):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute("PRAGMA foreign_keys = ON")
            self._db.executescript(SCHEMA)
            columns = {row[1] for row in self._db.execute("PRAGMA table_info(turns)")}
            if "summarized" not in columns:  # databases from before the column
                self._db.execute(
                    "ALTER TABLE turns ADD COLUMN summarized INTEGER NOT NULL DEFAULT 1"
                )
            self._into_threads()
            self._db.execute("CREATE INDEX IF NOT EXISTS turns_thread ON turns(thread_id)")
            self._db.execute("CREATE INDEX IF NOT EXISTS topics_thread ON topics(thread_id)")
            self._db.commit()

    def _into_threads(self):
        """A database from before threads: everything in it becomes the
        first thread, with Iris."""
        for table in ("turns", "topics"):
            columns = {row[1] for row in self._db.execute(f"PRAGMA table_info({table})")}
            if "thread_id" not in columns:
                self._db.execute(
                    f"ALTER TABLE {table} ADD COLUMN thread_id INTEGER REFERENCES threads(id)"
                )
        left = self._db.execute(
            "SELECT (SELECT COUNT(*) FROM turns WHERE thread_id IS NULL)"
            " + (SELECT COUNT(*) FROM topics WHERE thread_id IS NULL)"
        ).fetchone()[0]
        if not left:
            return
        first, last = self._db.execute(
            "SELECT MIN(ts), MAX(ts) FROM turns WHERE thread_id IS NULL"
        ).fetchone()
        now = time.time()
        thread = self._db.execute(
            "INSERT INTO threads (persona, created, updated) VALUES ('iris', ?, ?)",
            (first or now, last or now),
        ).lastrowid
        self._db.execute("UPDATE turns SET thread_id = ? WHERE thread_id IS NULL", (thread,))
        self._db.execute("UPDATE topics SET thread_id = ? WHERE thread_id IS NULL", (thread,))

    def _query(self, sql: str, args: tuple = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._db.execute(sql, args).fetchall()

    def _write(self, sql: str, args: tuple = ()) -> int:
        with self._lock:
            cursor = self._db.execute(sql, args)
            self._db.commit()
            return cursor.lastrowid

    # -- threads ------------------------------------------------------------

    def create_thread(self, persona: str) -> int:
        """A new thread with this persona (its id), as its first turn is stored."""
        now = time.time()
        return self._write(
            "INSERT INTO threads (persona, created, updated) VALUES (?, ?, ?)",
            (persona, now, now),
        )

    def threads(self) -> list[dict]:
        """Every thread, the one talked in last first: {id, persona, title,
        created, updated}."""
        rows = self._query(self._THREADS + " ORDER BY th.updated DESC, th.id DESC")
        return [self._titled(row) for row in rows]

    def thread(self, thread_id: Optional[int]) -> Optional[dict]:
        """One thread, as threads() lists it, or None."""
        rows = self._query(self._THREADS + " WHERE th.id = ?", (thread_id,))
        return self._titled(rows[0]) if rows else None

    def last_thread(self) -> Optional[dict]:
        """The thread talked in last, or None when there is none."""
        threads = self._query(self._THREADS + " ORDER BY th.updated DESC, th.id DESC LIMIT 1")
        return self._titled(threads[0]) if threads else None

    # A thread with what its title is made of: its earliest topic, else the
    # user's first words. Nothing of Iris's own: neither its replies nor an
    # opening it made itself, such as its greeting
    _THREADS = """
        SELECT th.*,
            (SELECT title FROM topics WHERE thread_id = th.id ORDER BY created, id LIMIT 1)
                AS topic,
            (SELECT text FROM turns WHERE thread_id = th.id AND role = 'user'
                ORDER BY id LIMIT 1) AS asked
        FROM threads th
    """

    @staticmethod
    def _titled(row: sqlite3.Row) -> dict:
        """The thread with its title, "" until the user has said anything."""
        thread = dict(row)
        topic, asked = thread.pop("topic"), thread.pop("asked")
        thread["title"] = topic or shortened(asked or "")
        return thread

    def forget_thread(self, thread_id: int):
        """Delete a thread with everything in it: its turns (and their search
        entries) and its topics."""
        with self._lock:
            self._db.execute("DELETE FROM turns WHERE thread_id = ?", (thread_id,))
            self._db.execute("DELETE FROM topics WHERE thread_id = ?", (thread_id,))
            self._db.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
            self._db.commit()

    # -- turns --------------------------------------------------------------

    def add_turn(self, thread: int, role: str, text: str, source: Optional[str] = None) -> int:
        """Store one turn of a thread: "user", "assistant", or "system" for notes."""
        now = time.time()
        with self._lock:
            cursor = self._db.execute(
                "INSERT INTO turns (ts, role, text, source, thread_id) VALUES (?, ?, ?, ?, ?)",
                (now, role, text, source, thread),
            )
            self._db.execute("UPDATE threads SET updated = ? WHERE id = ?", (now, thread))
            self._db.commit()
            return cursor.lastrowid

    def recent_turns(self, thread: Optional[int], budget: Optional[int] = None) -> list[dict]:
        """The thread's newest turns that fit the token budget, oldest first."""
        budget = LIVE_TOKENS if budget is None else budget
        turns, used = [], 0
        rows = self._query(
            "SELECT * FROM turns WHERE thread_id = ? ORDER BY id DESC LIMIT 400", (thread,)
        )
        for row in rows:
            cost = estimate_tokens(row["text"])
            if turns and used + cost > budget:
                break
            turns.append(dict(row))
            used += cost
        return turns[::-1]

    @staticmethod
    def _exchanges(rows, key: str) -> dict:
        """Turns grouped into exchanges, per value of a column (the thread,
        or the topic)."""
        grouped: dict = {}
        for row in rows:
            exchanges = grouped.setdefault(row[key], [])
            line = f"{'User' if row['role'] == 'user' else 'Assistant'}: {row['text']}"
            if row["role"] == "user" or not exchanges:
                exchanges.append(Exchange([row["id"]], line, row["thread_id"]))
            else:
                exchanges[-1].turn_ids.append(row["id"])
                exchanges[-1].text += "\n" + line
        return grouped

    def unfiled_exchanges(self) -> list[Exchange]:
        """Unfiled turns grouped into exchanges, thread by thread; notes are
        left out."""
        rows = self._query(
            "SELECT * FROM turns WHERE filed = 0 AND role != 'system' ORDER BY thread_id, id"
        )
        return [e for exchanges in self._exchanges(rows, "thread_id").values() for e in exchanges]

    def mark_filed(self, thread: int, turn_ids: list[int], topic_id: Optional[int]):
        """File a thread's turns under a topic (None: kept, but under no
        topic). Turns filed under a topic wait for its summary to take them
        in. Turns of a thread forgotten meanwhile are gone, and so is the
        filing."""
        if not turn_ids:
            return
        marks = ",".join("?" * len(turn_ids))
        self._write(
            "UPDATE turns SET filed = 1, topic_id = ?, summarized = ? "
            f"WHERE thread_id = ? AND id IN ({marks})",
            (topic_id, 0 if topic_id else 1, thread, *turn_ids),
        )

    def pending_summaries(self) -> dict[int, list[Exchange]]:
        """Per topic, the filed exchanges its summary does not include yet."""
        rows = self._query(
            "SELECT * FROM turns WHERE filed = 1 AND summarized = 0 AND topic_id IS NOT NULL "
            "ORDER BY id"
        )
        return self._exchanges(rows, "topic_id")

    def mark_summarized(self, thread: int, turn_ids: list[int]):
        if not turn_ids:
            return
        marks = ",".join("?" * len(turn_ids))
        self._write(
            f"UPDATE turns SET summarized = 1 WHERE thread_id = ? AND id IN ({marks})",
            (thread, *turn_ids),
        )

    def mark_notes_filed(self):
        self._write("UPDATE turns SET filed = 1 WHERE role = 'system' AND filed = 0")

    # -- topics -------------------------------------------------------------

    def create_topic(self, thread: int, title: str, gist: str, summary: str) -> Optional[int]:
        """A new topic of the thread (its id), or None when the thread was
        forgotten meanwhile."""
        now = time.time()
        with self._lock:
            cursor = self._db.execute(
                "INSERT INTO topics (title, gist, summary, created, updated, thread_id) "
                "SELECT ?, ?, ?, ?, ?, id FROM threads WHERE id = ?",
                (title, gist, summary, now, now, thread),
            )
            self._db.commit()
            return cursor.lastrowid if cursor.rowcount else None

    def update_topic(self, thread: int, topic_id: int, gist: str, summary: str):
        self._write(
            "UPDATE topics SET gist = ?, summary = ?, updated = ? WHERE id = ? AND thread_id = ?",
            (gist, summary, time.time(), topic_id, thread),
        )

    def topics(self, thread: Optional[int]) -> list[Topic]:
        """The thread's topics, the most recently discussed first."""
        rows = self._query(
            "SELECT * FROM topics WHERE thread_id = ? ORDER BY updated DESC", (thread,)
        )
        return [Topic(**dict(row)) for row in rows]

    def topic(self, topic_id: int, thread: Optional[int] = None) -> Optional[Topic]:
        """A topic with its turns, or None; given a thread, only one of its own."""
        if thread is None:
            rows = self._query("SELECT * FROM topics WHERE id = ?", (topic_id,))
        else:
            rows = self._query(
                "SELECT * FROM topics WHERE id = ? AND thread_id = ?", (topic_id, thread)
            )
        if not rows:
            return None
        topic = Topic(**dict(rows[0]))
        topic.turns = [
            dict(r)
            for r in self._query(
                "SELECT id, ts, role, text, source FROM turns WHERE topic_id = ? ORDER BY id",
                (topic_id,),
            )
        ]
        return topic

    def turn_counts(self, thread: Optional[int]) -> dict[int, int]:
        rows = self._query(
            "SELECT topic_id, COUNT(*) AS n FROM turns "
            "WHERE thread_id = ? AND topic_id IS NOT NULL GROUP BY topic_id",
            (thread,),
        )
        return {row["topic_id"]: row["n"] for row in rows}

    def forget(self, topic_id: int):
        """Delete a topic together with every turn filed under it."""
        with self._lock:
            self._db.execute("DELETE FROM turns WHERE topic_id = ?", (topic_id,))
            self._db.execute("DELETE FROM topics WHERE id = ?", (topic_id,))
            self._db.commit()

    def index_lines(self, thread: Optional[int], budget: Optional[int] = None) -> list[str]:
        """One line per topic of the thread for the prompt, newest first,
        within budget.

        Topics that do not fit are not lost: search_memory still finds them.
        """
        budget = INDEX_TOKENS if budget is None else budget
        lines, used, now = [], 0, time.time()
        for topic in self.topics(thread):
            line = f"#{topic.id} {topic.title}: {topic.gist} ({relative_day(topic.updated, now)})"
            cost = estimate_tokens(line)
            if used + cost > budget:
                break
            lines.append(line)
            used += cost
        return lines

    # -- search -------------------------------------------------------------

    def search(self, thread: Optional[int], query: str, limit: int = 6) -> list[dict]:
        """Full-text search over every turn of the thread, best matches first."""
        words = re.findall(r"\w+", query.lower())
        if not words or thread is None:
            return []
        # Quoted terms joined by OR: any word may match, BM25 ranks the rest
        match = " OR ".join(f'"{w}"' for w in words)
        rows = self._query(
            """
            SELECT turns.id, turns.ts, turns.role, turns.topic_id, topics.title,
                   snippet(turns_fts, 0, '[', ']', ' … ', 16) AS snippet
            FROM turns_fts
            JOIN turns ON turns.id = turns_fts.rowid
            LEFT JOIN topics ON topics.id = turns.topic_id
            WHERE turns_fts MATCH ? AND turns.thread_id = ?
            ORDER BY bm25(turns_fts)
            LIMIT ?
            """,
            (match, thread, limit),
        )
        return [dict(row) for row in rows]


# -- filing ------------------------------------------------------------------

# chat(messages, schema, abort) -> parsed JSON object; raises FilingInterruptedError
# when abort() turns true while the call runs
ChatFn = Callable[..., dict]


class FilingInterruptedError(Exception):
    """The conversation needed the LLM, so a filing call was cut off."""

SUMMARY_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}, "gist": {"type": "string"}},
    "required": ["summary", "gist"],
}


class Filer:
    """Files new exchanges into topics with small, focused LLM calls.

    Each exchange gets one decision of its own: worth remembering, and if
    so, which topic it belongs to or the title of a new one. Asking a small
    model to split a whole batch into subjects at once lumps unrelated
    exchanges together, and the resulting catch-all topic then attracts
    everything after it. Afterwards each topic that received exchanges gets
    its summary and index line rewritten, once.

    should_yield is asked before every call; when it says the conversation
    needs the LLM, filing stops. Each decision and each summary is saved as
    it completes, so the next run picks up exactly where this one stopped.
    """

    def __init__(self, store: MemoryStore, chat: ChatFn):
        self.store = store
        self.chat = chat
        self._abort: Callable[[], bool] = lambda: False
        self._changed = False

    def file(self, should_yield: Callable[[], bool] = lambda: False) -> bool:
        """File unfiled exchanges, then update the summaries they touched,
        until done or until the conversation needs the LLM.

        Returns whether anything changed.
        """
        self._abort = should_yield
        self._changed = False
        try:
            self._file(should_yield)
        except FilingInterruptedError:
            pass  # the interrupted step saved nothing; the next run redoes it
        return self._changed

    def _file(self, should_yield: Callable[[], bool]):
        self.store.mark_notes_filed()
        # Per thread, the exchange said just before: each is filed into
        # topics of its own thread
        before: dict[Optional[int], str] = {}
        for exchange in self.store.unfiled_exchanges():
            if should_yield():
                return
            thread = exchange.thread_id
            decision = self._decide(exchange, before.get(thread, ""))
            before[thread] = exchange.text
            topic_id = None
            if decision.get("remember"):
                topic_id = decision.get("topic_id") or 0
                if self.store.topic(topic_id, thread) is None:
                    title = (decision.get("title") or "").strip() or exchange.text[6:46]
                    topic_id = self.store.create_topic(thread, title, title, "")
                    if topic_id is None:  # the thread was forgotten meanwhile
                        continue
            # Small talk is kept and searchable too, just under no topic
            self.store.mark_filed(thread, exchange.turn_ids, topic_id)
            self._changed = True

        for topic_id, exchanges in self.store.pending_summaries().items():
            if should_yield():
                return
            self._summarize(topic_id, exchanges)
            thread = exchanges[0].thread_id
            self.store.mark_summarized(thread, [t for e in exchanges for t in e.turn_ids])
            self._changed = True

    def _decide(self, exchange: Exchange, before: str) -> dict:
        topics = self.store.topics(exchange.thread_id)[:40]
        listing = "\n".join(f"#{t.id} {t.title}: {t.gist}" for t in topics) or "(none yet)"
        context = f"Said just before:\n{before}\n\n" if before else ""
        prompt = (
            f"Topics in memory:\n{listing}\n\n{context}Exchange:\n{exchange.text}\n\n"
            "Is this exchange worth remembering, rather than small talk such as greetings "
            "or thanks? If it is, which topic above is it about: give its number, or 0 if "
            "it is a new subject. Different subjects that merely share a date or a person "
            "are different topics. Also give a short title for the exchange's subject."
        )
        schema = {
            "type": "object",
            "properties": {
                "remember": {"type": "boolean"},
                "topic_id": {"enum": [t.id for t in topics] + [0]},
                "title": {"type": "string"},
            },
            "required": ["remember", "topic_id", "title"],
        }
        return self.chat([{"role": "user", "content": prompt}], schema, abort=self._abort)

    def _summarize(self, topic_id: int, exchanges: list[Exchange]):
        """Fold the new exchanges into the topic's summary and index line."""
        topic = self.store.topic(topic_id)
        if topic is None:  # forgotten meanwhile
            return
        text = "\n\n".join(e.text for e in exchanges)
        known = f"What is known so far:\n{topic.summary}\n\n" if topic.summary else ""
        prompt = (
            f"Topic: {topic.title}\n\n{known}New exchanges:\n{text}\n\n"
            "Write the topic's summary for long-term memory: the facts, names, dates, "
            "preferences and decisions about this topic worth recalling later, in a few "
            "sentences. Then write the gist: one short line naming the key people, "
            "places, dates and decisions of this topic only."
        )
        result = self.chat(
            [{"role": "user", "content": prompt}], SUMMARY_SCHEMA, abort=self._abort
        )
        summary = result.get("summary", "").strip() or topic.summary or text[:400]
        gist = result.get("gist", "").strip() or topic.gist
        self.store.update_topic(topic.thread_id, topic_id, gist, summary)


class MemoryKeeper:
    """Files during quiet spells, on a background thread, and gives way to
    the conversation the moment it resumes."""

    def __init__(
        self,
        store: MemoryStore,
        chat: ChatFn,
        on_change: Optional[Callable[[], None]] = None,
        busy: Callable[[], bool] = lambda: False,
    ):
        """
        Args:
            busy: True while the conversation is active (a turn underway,
                speech playing, the user speaking or typing); filing never
                starts a call then
        """
        self.store = store
        self.filer = Filer(store, chat)
        self.on_change = on_change
        self.busy = busy
        self._filing = threading.Lock()
        self._last_activity = time.monotonic()
        self._stop = threading.Event()

    def note_turn(self):
        """Call after each exchange: the quiet period starts over."""
        self._last_activity = time.monotonic()

    def should_yield(self) -> bool:
        """The conversation needs the LLM, now or quite possibly soon."""
        if self.busy():
            self._last_activity = time.monotonic()
            return True
        return time.monotonic() - self._last_activity < QUIET_BEFORE_FILING

    def _file(self):
        if not self._filing.acquire(blocking=False):
            return
        try:
            if self.filer.file(self.should_yield) and self.on_change:
                self.on_change()
        except Exception as e:
            # Nothing is lost: what was not filed is filed in a later quiet spell
            print(f"[memory] filing failed: {e}")
        finally:
            self._filing.release()

    def start(self):
        """Watch for quiet spells and file in them."""

        def watch():
            while not self._stop.wait(5):
                if self.should_yield():
                    continue
                if self.store.unfiled_exchanges() or self.store.pending_summaries():
                    self._file()

        threading.Thread(target=watch, daemon=True).start()

    def stop(self):
        self._stop.set()


def _shape(schema: dict) -> str:
    """The schema as a one-line example, for models that ignore `format`."""
    parts = []
    for name, spec in schema["properties"].items():
        if "enum" in spec:
            value = "<one of " + ", ".join(json.dumps(v) for v in spec["enum"]) + ">"
        elif spec.get("type") == "boolean":
            value = "<true or false>"
        else:
            value = '"<text>"'
        parts.append(f'"{name}": {value}')
    return "{" + ", ".join(parts) + "}"


# Reasoning a model writes inline before its answer
REASONING = re.compile(r"<think>.*?(</think>|$)", re.DOTALL)


def parse_json(text: str) -> Optional[dict]:
    """The first JSON object in a reply, tolerating fences and extra prose."""
    start = text.find("{")
    while start != -1:
        try:
            value, _ = json.JSONDecoder().raw_decode(text[start:])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
        start = text.find("{", start + 1)
    return None


def _abortable_chat(
    host: Optional[str], model: str, messages: list[dict], schema: dict, options: dict,
    abort: Callable[[], bool], think: bool,
) -> str:
    """One streamed chat call on a connection of its own, cut off as soon as
    abort() turns true. Closing the connection makes the server stop
    generating, which frees the model for the conversation at once."""
    if is_openai(host):
        client = OpenAIClient(host)
    else:
        client = ollama.Client(host=host, timeout=MODEL_TIMEOUT)
    done = threading.Event()
    cut = threading.Event()

    def watch():
        while not done.wait(ABORT_CHECK_SECONDS):
            if abort():
                cut.set()
                client._client.close()  # the client's httpx.Client
                return

    threading.Thread(target=watch, daemon=True).start()
    kwargs = {"think": False} if not think else {}
    try:
        stream = client.chat(
            model=model, messages=messages, format=schema, options=options, stream=True, **kwargs
        )
        reply = "".join(chunk.message.content or "" for chunk in stream)
        return REASONING.sub("", reply)
    except Exception:
        if cut.is_set():
            raise FilingInterruptedError() from None
        raise
    finally:
        done.set()


def chat_json(get_host: Callable[[], Optional[str]], get_model: Callable[[], str]) -> ChatFn:
    """A ChatFn on the conversation's server and model, looked up per
    call so model and server switches apply.

    Local models are held to the schema by `format`; cloud models ignore it
    and answer in prose, so the shape is also spelled out in the prompt, the
    reply is parsed tolerantly, and a reply without the JSON is asked again
    once. No thinking and no sampling noise: these are quick, plain calls,
    and each can be cut off when the conversation needs the model.
    """

    def chat(messages: list[dict], schema: dict, abort: Callable[[], bool] = lambda: False):
        host, model = get_host(), get_model()
        options = {"temperature": 0, "num_predict": FILING_MAX_TOKENS}
        messages = messages[:-1] + [
            {
                **messages[-1],
                "content": messages[-1]["content"]
                + f"\n\nReply with JSON only, in this shape:\n{_shape(schema)}",
            }
        ]
        reply = ""
        for _ in range(2):
            if abort():
                raise FilingInterruptedError()
            try:
                reply = _abortable_chat(host, model, messages, schema, options, abort, think=False)
            except FilingInterruptedError:
                raise
            except Exception:
                # Models that cannot switch thinking off: ask without the flag
                reply = _abortable_chat(host, model, messages, schema, options, abort, think=True)
            parsed = parse_json(reply)
            if parsed is not None and all(k in parsed for k in schema.get("required", [])):
                return parsed
            messages = messages + [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": "Reply with only the JSON object, nothing else."},
            ]
        raise ValueError(f"{model} gave no usable JSON: {reply[:120]!r}")

    return chat
