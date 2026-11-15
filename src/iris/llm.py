"""LLM conversation management, on Ollama or an OpenAI-compatible server."""

import itertools
import logging
import re
import threading
import types
import warnings
from collections.abc import Callable
from typing import Optional

import httpx
import markdown
import ollama
from bs4 import BeautifulSoup

from iris.attachments import Attachment, recorded_text
from iris.llm_client import MODEL_TIMEOUT, READ_TIMEOUT, OpenAIClient, is_openai, normalize_host
from iris.llm_client import effective_host as effective_host  # imported from here by callers
from iris.markup import MarkupFilter, TypedCall, strip_markup
from iris.memory import LIVE_TOKENS, PIN_TURNS, PINNED_TOKENS, MemoryKeeper, estimate_tokens
from iris.personas import Persona, get_persona
from iris.roles import find_role

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
# Rounds of tool calls one reply may take, e.g. a search, then reading a page
MAX_TOOL_ROUNDS = 3
# Told the model when a round has to answer without tools: the last one, and
# one asked again after a reply that was nothing but a call it cannot make
NO_MORE_TOOLS_NOTE = "Answer the user now from what you have. No more tools can be used."
NO_TOOLS_NOTE = "No tools can be used now. Answer the user in words."
# Told the model in place of images no model at hand can see: shared by
# the user, or captured by a tool (the camera, the screen)
IMAGES_UNSEEN = (
    "(The images cannot be shown to you: no model that can see images is "
    "installed. Let the user know.)"
)
IMAGE_UNSEEN = (
    "(The image cannot be shown to you: no model that can see images is "
    "installed. Let the user know.)"
)
# How Ollama refuses a request with tools for a model that cannot take them
TOOLS_REFUSED = "does not support tools"
# The source of a turn Iris opens itself, such as its greeting at the end of
# the walkthrough: what it was asked to do is kept as a note, not as the
# user's words
OPENING = "opening"


def ollama_client(host: Optional[str] = None, **kwargs):
    """A client for a server URL as resolve_host keeps it: an OpenAIClient
    for an OpenAI-compatible server, ollama's own client for Ollama, or the
    ollama module itself (whose default client follows OLLAMA_HOST) when no
    server is configured. Unless told otherwise, a client gives up on a
    server that stops answering (MODEL_TIMEOUT)."""
    host = normalize_host(host)
    if is_openai(host):
        return OpenAIClient(host, **kwargs)
    if not host and not kwargs:
        return ollama
    return ollama.Client(host=host, **{"timeout": MODEL_TIMEOUT, **kwargs})


def model_missing(error: Exception) -> bool:
    """Whether the server said it has no such model: it was removed after it
    was chosen, or never there. Ollama answers 404 "model '...' not found",
    OpenAI-compatible servers 404 "The model ... does not exist"."""
    return (
        isinstance(error, ollama.ResponseError)
        and error.status_code == 404
        and "model" in str(error.error).lower()
    )


def failure_message(error: Exception, host: Optional[str], model: Optional[str] = None) -> str:
    """What the user is told when the model could not answer. A server out
    of reach, one that stopped answering or ran into a problem of its own,
    and a model it does not have are said to be so in words that fit any
    kind of server; other errors are passed on as they are.

    Args:
        error: What the request raised
        host: The server, as kept
        model: The model asked, which the server may not have
    """
    server = f"The model server at {effective_host(host)}"
    if isinstance(error, (httpx.ConnectError, httpx.ConnectTimeout, ConnectionError)):
        return f"{server} cannot be reached."
    if isinstance(error, httpx.ReadTimeout):
        return f"{server} stopped answering: nothing came for {READ_TIMEOUT:g} seconds."
    if isinstance(error, httpx.TimeoutException):
        return f"{server} did not answer in time."
    if isinstance(error, (httpx.ReadError, httpx.WriteError, httpx.RemoteProtocolError)):
        return f"The connection to the model server at {effective_host(host)} was lost."
    if isinstance(error, ollama.ResponseError):
        # Ollama's client adds the status code to the message, -1 for an
        # error the server streamed once the reply had begun
        reason = str(error.error).strip() or "it gave no reason"
        if model_missing(error):
            named = f"The model {model}" if model else "The model"
            return (
                f"{named} is not on the server at {effective_host(host)}. "
                "Choose another one in Settings."
            )
        if error.status_code >= 500 or error.status_code == -1:
            return f"{server} ran into a problem: {reason}"
        return f"The language model could not answer: {reason}"
    return f"The language model could not answer: {error}"


# A bullet or numbered list marker at the start of a line
LIST_ITEM = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+")
# A line already ending in punctuation that makes the voice pause, perhaps
# followed by closing quotes or brackets (Japanese and Chinese ones, and the
# Hindi danda, too)
PAUSED_END = re.compile(r"[.!?…:;,。！？、，：；।॥][\"'”’)\]」』）]*$")
# Emoji, which the voice does not say: pictographs and the symbols and
# dingbats used as emoji, with their skin tones, presentation selector and
# joiner. Only these blocks: a range across them once took every Japanese
# and Chinese character for an emoji, and those replies were never spoken.
EMOJI = re.compile(
    "["
    "\U0001f000-\U0001faff"  # pictographs, emoticons, transport, flags, skin tones
    "\u2600-\u27bf"  # miscellaneous symbols and dingbats
    "\u2b00-\u2bff"  # arrows, stars and squares
    "\u24c2\ufe0f\u200d\u20e3"  # circled M, emoji presentation, joiner, keycap
    "]+"
)
# Where streamed text may be cut for speech: after a sentence or at a line.
# Japanese and Chinese end a sentence with 。！？ and go on without a space,
# so the text after it must have begun (a closing quote or bracket stays with
# the sentence); Hindi ends one with a danda.
CLOSING = "」』）】〉》”’\"')\\]"
SPEECH_BREAK = re.compile(
    rf"[.!?](?= )|[।॥](?=\s)|[。！？]+[{CLOSING}]*(?=[^{CLOSING}。！？])|\n"
)
# The number of a numbered list item, which looks like a sentence end
LIST_NUMBER = re.compile(r"(?:^|\n)\s*\d+\.$")


def separate_lists(text: str) -> str:
    """Put a blank line before each list, which markdown needs to see one;
    models often start a list right under the line introducing it."""
    lines = text.split("\n")
    for i in range(len(lines) - 1, 0, -1):
        above = lines[i - 1]
        if LIST_ITEM.match(lines[i]) and above.strip() and not LIST_ITEM.match(above):
            lines.insert(i, "")
    return "\n".join(lines)


def pause_at_line_ends(text: str) -> str:
    """End every line with punctuation the voice pauses at.

    The voice reads a line break as a space, so list items, headings and
    paragraphs without closing punctuation would run into one sentence.
    """
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(
        line + "." if line.strip() and not PAUSED_END.search(line) else line
        for line in lines
    )


def speech_break(text: str, start: int, end: int, first: bool = False) -> int:
    """Index after the last (or first) sentence end or line break in
    text[start:end], where the text can be cut for speech, or -1 if there is
    none."""
    cut = -1
    for match in SPEECH_BREAK.finditer(text, start, end):
        if not LIST_NUMBER.search(text, 0, match.end()):
            cut = match.end()
            if first:
                break
    return cut


# The first piece of a reply ends at its first sentence (one this long at
# least, so "Sure." is not a piece of its own): the voice starts as soon as
# the sentence is there
FIRST_PIECE_MIN = 24
# Later pieces end at the last sentence within this span, once there is more
# text than the minimum: long pieces keep the voice flowing
PIECE_MIN = 180
PIECE_SENTENCE_FROM = 90
PIECE_SENTENCE_TO = 280
# ...but while the piece before was short, they are scaled down: one is at
# most this many times as long as the one before. Synthesizing a piece takes
# about a quarter of its playing time, so it is ready before the one before
# has been spoken, and the voice never waits for a long piece.
PIECE_GROWTH = 3


def split_speech(buffer: str, previous: int) -> tuple[str, str]:
    """Split streamed reply text into a piece ready to be spoken and the rest,
    which waits for more text. Returns ("", buffer) while no piece is ready.

    previous is the length of the piece before, 0 for the first piece of the
    reply. A paragraph is always a piece of its own.
    """
    while "\n\n" in buffer:
        head, buffer = buffer.split("\n\n", 1)
        if head.strip():
            return head.strip(), buffer
    if not previous:
        cut = speech_break(buffer, FIRST_PIECE_MIN, len(buffer), first=True)
    else:
        scale = min(1.0, PIECE_GROWTH * previous / PIECE_SENTENCE_TO)
        start, end = int(PIECE_SENTENCE_FROM * scale), int(PIECE_SENTENCE_TO * scale)
        cut = -1
        if len(buffer) > PIECE_MIN * scale:
            cut = speech_break(buffer, start, end)
        if cut < 0 and len(buffer) > end:
            # One long sentence: it ends the piece wherever it ends
            cut = speech_break(buffer, start, len(buffer), first=True)
    if cut > 0:
        return buffer[:cut].strip(), buffer[cut:]
    return "", buffer


def system_first(messages: list) -> list:
    """The messages with a system message only at the start.

    Many chat templates refuse a system message anywhere else, and the server
    then refuses the whole request. Notes Iris adds along the way (the persona
    changed, the language to answer in, no more tools) go into the user
    message that follows them, or into one of their own at the end.
    """
    out: list = []
    notes: list[str] = []
    for index, message in enumerate(messages):
        role = message.get("role")
        if role == "system" and index > 0:
            notes.append(message.get("content") or "")
            continue
        if role == "user" and notes:
            message = {**message, "content": _noted(notes) + (message.get("content") or "")}
            notes = []
        out.append(message)
    if notes:
        out.append({"role": "user", "content": _noted(notes).rstrip()})
    return out


def _noted(notes: list[str]) -> str:
    return "".join(f"(Note: {note})\n\n" for note in notes)


class ConversationManager:
    """Manages LLM conversations with context and summarization."""

    # The turn being answered, stored as its reply began: (thread, turn)
    _asked: Optional[tuple[int, int]] = None

    def __init__(
        self,
        model: str = "qwen3:1.7b",
        persona: Optional[Persona] = None,
        tools=None,
        vision_model: Optional[str] = None,
        host: Optional[str] = None,
        memory: Optional[MemoryKeeper] = None,
    ):
        """
        Initialize conversation manager.

        Args:
            model: Ollama model name (primary/text model)
            persona: Persona to use (defaults to iris)
            tools: Optional Tools instance for function calling
            vision_model: Optional vision model to switch to when using vision tools
            host: Ollama server URL; None follows OLLAMA_HOST
            memory: Long-term memory; when given, the conversation is stored
                in a thread and the context is built from it (see iris.memory)
        """
        self.host = normalize_host(host)
        self.client = ollama_client(self.host)
        # Set by cancel(): the reply being streamed is not to go on
        self._cancelled = threading.Event()
        # The client replies stream through while self.client is the ollama
        # module, whose shared client must stay open
        self._own_client = None
        # The client of the reply being streamed: all its rounds go to the
        # server it began on, and cancel() cuts that one off, also after a
        # switch to another server
        self._streaming = None
        # The model last asked for a reply, which an error may be about
        self.asked: Optional[str] = None
        self.memory = memory
        # The thread the conversation is in, or None for a new one, which is
        # stored with its first turn (see open_thread), and the thread's role
        self.thread: Optional[int] = None
        self.role: Optional[str] = None
        # Told when a turn was stored or taken back: a new thread is listed
        # with its first turn, as its reply begins
        self.on_stored: Optional[Callable[[], None]] = None
        # Recalled topics stay in context for a few turns: {topic_id: turns left}
        self.pinned: dict[int, int] = {}
        # Amnesia mode, the open thread's: the conversation goes on, but
        # nothing of it is written to memory; its turns live here and vanish
        # when it ends or the thread changes
        self.amnesia = False
        self.unremembered: list[dict] = []
        # Memory's own switch, for every thread: off, nothing is written,
        # turns are held back as in amnesia mode, and memory is neither in
        # the prompt nor offered as tools
        self.remember = True
        # How many times everything was forgotten, and how many times when
        # the reply being given began: a reply begun before is not kept
        self._forgotten = 0
        self._reply_began = 0
        self._keeping = threading.Lock()
        # Files shared in this conversation, kept for a few turns and never
        # stored: [(attachment, turns left)], oldest first
        self.shared: list[tuple[Attachment, int]] = []
        self._capability_cache: dict[str, set] = {}
        # Models that refused a request because of its tools
        self._no_tools: set[str] = set()
        self.model = model
        self.vision_model = vision_model
        # The language the user spoke or typed last (Whisper code), so the
        # model is told when the user goes back to English
        self.language = "en"
        self.conversation_history = []
        # The last complete reply, as it is kept in the conversation
        self.reply: Optional[str] = None
        self.persona = persona if persona else get_persona("iris")
        self.tools = tools
        self.max_context_tokens = self._get_model_context_length()

        # Not fatal: the app starts without a server and says what is missing,
        # and a server can be set later (set_host)
        try:
            self.client.list()
            self.reachable = True
        except Exception as e:
            self.reachable = False
            logger.warning("Ollama is not reachable at %s: %s", effective_host(self.host), e)

    @staticmethod
    def list_available_models(host: Optional[str] = None) -> list[dict]:
        """List Ollama models that can hold a conversation.

        Embedding-only models are excluded. Each entry reports whether the
        model supports tool calling, which Iris needs for vision/search tools.
        """
        client = ollama_client(host)
        try:
            models = []
            for entry in client.list().models:
                name = entry.model
                try:
                    capabilities = list(client.show(name).capabilities or [])
                except Exception:
                    capabilities = []
                if "embedding" in capabilities:
                    continue
                models.append(
                    {
                        "name": name,
                        "tools": "tools" in capabilities,
                        "vision": "vision" in capabilities,
                    }
                )
            return sorted(models, key=lambda m: m["name"])
        except Exception:
            return []

    def set_host(self, host: Optional[str]):
        """Switch to another server, keeping the conversation history. A
        reply being streamed goes on with the server it began on."""
        self.host = normalize_host(host)
        self.client = ollama_client(self.host)
        self.reachable = True  # set_host is only called for a server that answered
        self._capability_cache = {}
        self._no_tools = set()
        self.max_context_tokens = self._get_model_context_length()

    def set_model(self, model: str):
        """Switch the primary model, keeping conversation history intact."""
        self.model = model
        self.max_context_tokens = self._get_model_context_length()

    def _capabilities(self, model: str) -> Optional[set]:
        """What the model can do ("vision", "tools", ...), or None if unknown."""
        if model not in self._capability_cache:
            try:
                self._capability_cache[model] = set(self.client.show(model).capabilities or [])
            except Exception:
                return None
        return self._capability_cache[model]

    def _takes_tools(self, model: str) -> bool:
        """Whether to offer the model tools: not once it refused them, nor
        when the server says it cannot take them. A model chosen for its
        sight is offered them only when the server says it can."""
        if model in self._no_tools or getattr(self.client, "tools_refused", False) is True:
            return False
        if model != self.model:
            return "tools" in (self._capabilities(model) or set())
        # Known from when the model was chosen; unknown, it is offered tools
        # until it refuses them
        capabilities = self._capability_cache.get(model)
        return not capabilities or "tools" in capabilities

    def image_model(self) -> Optional[str]:
        """The model that answers a message with images, or None if no
        model that can see is at hand."""
        capabilities = self._capabilities(self.model)
        if capabilities is None or "vision" in capabilities:
            return self.model
        return self.vision_model

    def _files_context(self, new: list[Attachment]) -> str:
        """The shared documents as text for the model, newest first within
        a budget, and a line naming the images that go with the message."""
        budget = max(2000, self.max_context_tokens // 2)
        blocks = []
        for attachment, _ in reversed(self.shared):
            if attachment.kind != "document":
                continue
            when = "with this message" if any(attachment is a for a in new) else "earlier"
            text = attachment.text or ""
            if estimate_tokens(text) > budget:
                text = text[: budget * 4] + "\n[The rest of the file did not fit.]"
            budget -= estimate_tokens(text)
            blocks.insert(0, f'<file name="{attachment.name}" shared="{when}">\n{text}\n</file>')
            if budget <= 0:
                break
        images = [a.name for a, _ in self.shared if a.kind == "image"]
        if images:
            blocks.append("Images the user shared, in order: " + ", ".join(images))
        if not blocks:
            return ""
        return "\n\nFiles the user shared in this conversation:\n\n" + "\n\n".join(blocks)

    def _get_model_context_length(self) -> int:
        """
        Get the model's context length from Ollama.
        Uses 80% of the actual context length as a safety margin.

        Returns:
            Maximum context tokens to use (80% of model's limit)
        """
        try:
            model_info = self.client.show(self.model)
            # What it can do comes with it (see _takes_tools)
            capabilities = set(model_info.capabilities or [])
            if capabilities:
                self._capability_cache[self.model] = capabilities
            modelinfo = model_info.modelinfo

            # Look for context length in model info
            context_keys = [
                f"{model_info.details.family}.context_length",
                "context_length",
                "max_position_embeddings",
            ]

            context_length = None
            for key in context_keys:
                if key in modelinfo:
                    context_length = modelinfo[key]
                    break

            if context_length:
                # Use 80% of context to leave room for safety
                usable_context = int(context_length * 0.8)
                return usable_context
            else:
                # Fallback: use a conservative default
                return 25600

        except Exception:
            return 25600

    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.
        Uses simple heuristic: ~4 characters per token for English.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        return len(text) // 4

    def get_conversation_token_count(self, messages: list) -> int:
        """
        Estimate total tokens in conversation.

        Args:
            messages: List of message dicts

        Returns:
            Estimated token count
        """
        return sum(self.estimate_tokens(msg.get("content", "")) for msg in messages)

    def clean_markdown(self, text: str) -> str:
        """
        Remove markdown formatting and emojis to prepare text for speech.

        Args:
            text: Text potentially containing markdown and emojis

        Returns:
            Clean text suitable for speech
        """
        # First convert markdown to plain text
        html = markdown.markdown(separate_lists(text))
        clean_text = BeautifulSoup(html, "html.parser").get_text()

        clean_text = EMOJI.sub("", clean_text)

        return pause_at_line_ends(clean_text).strip()

    def summarize_old_messages(self, messages: list) -> str:
        """
        Summarize old conversation messages to save context.

        Args:
            messages: List of old messages to summarize

        Returns:
            Summary text
        """
        # Create a prompt to summarize
        conversation_text = "\n".join(
            [f"{msg['role'].upper()}: {msg['content']}" for msg in messages]
        )

        try:
            # Streamed, so the client's timeout limits the wait between
            # chunks, not the time the whole summary takes
            stream = self.client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Summarize this conversation in 2-3 sentences, "
                        "capturing the key points and context.",
                    },
                    {
                        "role": "user",
                        "content": f"Summarize this conversation:\n\n{conversation_text}",
                    },
                ],
                stream=True,
            )
            summary = "".join(chunk["message"]["content"] or "" for chunk in stream)
            return summary or "Previous conversation context."
        except Exception as e:
            print(f"Warning: Could not summarize: {e}")
            # Fallback: just take first message of each type
            return "Previous conversation context."

    def _memory_context(self) -> str:
        """The memory part of the system prompt: what Iris remembers of the
        thread (its topic index) and the topics it recalled in the last few
        turns."""
        store = self.memory.store
        lines = store.index_lines(self.thread)
        text = (
            "\n\nYou remember earlier conversations with the user. They are listed below "
            "by topic. When the user refers to something from before that is not in the "
            "recent messages, call 'recall' with the topic number before you answer, and "
            "use 'search_memory' for details the list does not show. Never pretend to "
            "remember what you have not recalled.\n\nMemory index:\n"
            + ("\n".join(lines) if lines else "(nothing yet)")
        )
        used = 0
        for topic_id in list(self.pinned):
            topic = store.topic(topic_id, self.thread)
            if topic is None:
                del self.pinned[topic_id]
                continue
            block = f"\n\nRecalled #{topic.id} {topic.title}: {topic.summary}"
            if used + estimate_tokens(block) > PINNED_TOKENS:
                break
            text += block
            used += estimate_tokens(block)
        return text

    def set_amnesia(self, on: bool):
        """Stop (or resume) writing the open thread to memory, kept with the
        thread. Turns spoken in amnesia mode are forgotten entirely once it
        ends."""
        self.amnesia = bool(on)
        self.unremembered = []
        if self.memory is not None and self.thread is not None:
            self.memory.store.set_amnesia(self.thread, self.amnesia)

    def set_remember(self, on: bool):
        """Switch memory on or off for every thread. Turns held back while it
        was off are forgotten once it is on again."""
        self.remember = bool(on)
        self.unremembered = []

    @property
    def _holding(self) -> bool:
        """Turns are held back, not written: amnesia mode, or memory off."""
        return self.memory is not None and (self.amnesia or not self.remember)

    def open_thread(
        self, thread: Optional[int], role: Optional[str] = None, amnesia: bool = False
    ):
        """Go on in another thread: its turns and topics are the
        conversation's memory from now on, its role (an id, or None) the
        focus, in amnesia mode if it is. None is a new thread, stored with
        its first turn. What belonged to the thread left (turns held back by
        amnesia, topics recalled, files shared) stays behind."""
        self.thread = thread
        self.role = role
        self.amnesia = bool(amnesia)
        self.unremembered = []
        self.pinned = {}
        self.shared = []
        self.conversation_history = []

    def _store(self, role: str, text: str, source: Optional[str] = None) -> Optional[int]:
        """Store a turn in the thread, which a new thread's first turn creates."""
        store = self.memory.store
        if self.thread is None:
            self.thread = store.create_thread(self.persona.id, self.role, self.amnesia)
        return store.add_turn(self.thread, role, text, source)

    def _store_asked(self, prompt: str, source: Optional[str]):
        """Store the turn being answered as its reply begins, so a new thread
        is listed at once; its reply follows once it has been heard
        (_record), or the turn is taken back (forgo). In amnesia mode nothing
        is stored, but a new thread is listed all the same; with memory off,
        nothing at all."""
        with self._keeping:
            self._asked = None
            if self.memory is None or not self.remember:
                return
            if self.amnesia:
                if self.thread is not None:
                    return
                self.thread = self.memory.store.create_thread(
                    self.persona.id, self.role, amnesia=True
                )
            else:
                role = "system" if source == OPENING else "user"
                turn = self._store(role, prompt, None if role == "system" else source)
                if turn is not None:
                    self._asked = (self.thread, turn)
        if self.on_stored:
            self.on_stored()

    def forgo(self):
        """Nothing of the reply is kept (stopped before a word of it was
        heard, or failed): the turn stored as it began is taken back, and a
        new thread it started goes with it, so no thread is left empty."""
        with self._keeping:
            asked, self._asked = self._asked, None
            if asked is None:
                return
            thread, turn = asked
            if self.memory.store.drop_turn(thread, turn) and self.thread == thread:
                self.thread = None
        if self.on_stored:
            self.on_stored()

    def forget(self):
        """Start the conversation over, as if nothing had been said, in a
        new thread, which is not in amnesia mode: no recent turns, recalled
        topics or shared files go to the model any more, nor does the
        language spoken last. A reply begun before is not kept, should it end
        later. What is stored in memory is the store's to forget
        (MemoryStore.forget_all)."""
        with self._keeping:
            self._forgotten += 1
            self.thread = None
            self.amnesia = False
            self._asked = None
            self.conversation_history = []
            self.unremembered = []
            self.pinned = {}
            self.shared = []
            self.reply = None
            self.language = "en"

    def _record(self, prompt: str, answer: str, source: Optional[str]):
        """Keep the exchange: in memory when there is one, else in history.
        A turn Iris opened itself keeps what it was asked as a note."""
        # Only what was said: a tool call typed out as text, left in the
        # history, would be imitated in later replies
        answer = strip_markup(answer)
        with self._keeping:
            asked, self._asked = self._asked, None
            if self._reply_began != self._forgotten:
                return  # everything was forgotten while it was being given
            role = "system" if source == OPENING else "user"
            if asked is not None:
                # Stored as the reply began, in the thread it began in: the
                # reply joins it
                self.memory.store.add_turn(asked[0], "assistant", answer)
            elif self._holding:
                self.unremembered.append({"role": role, "content": prompt})
                self.unremembered.append({"role": "assistant", "content": answer})
                return
            elif self.memory is None:
                self.conversation_history.append({"role": role, "content": prompt})
                self.conversation_history.append({"role": "assistant", "content": answer})
                return
            else:
                self._store(role, prompt, None if role == "system" else source)
                self._store("assistant", answer)
        self.memory.note_turn()
        if self.on_stored:
            self.on_stored()

    def add_note(self, text: str):
        """A system note in the conversation, e.g. that the persona changed.
        A new thread has said nothing yet for a note to be about."""
        if self._holding:
            self.unremembered.append({"role": "system", "content": text})
        elif self.memory is None:
            self.conversation_history.append({"role": "system", "content": text})
        elif self.thread is not None:
            self.memory.store.add_turn(self.thread, "system", text)

    def manage_context(self, with_tools: bool = True) -> list:
        """
        Manage conversation history to fit within context window.
        Summarizes old messages if needed.

        Args:
            with_tools: The model is offered tools, so the system prompt
                says how to use them

        Returns:
            Managed list of messages
        """
        # Build system prompt with persona description, then the thread's
        # focus, as the role says it now
        system_content = self.persona.description
        if role := find_role(self.role):
            system_content += f"\n\nYour role in this conversation: {role.name}. {role.focus}"

        # How to use the tools that are on, from each tool's guidance
        if self.tools and with_tools:
            system_content += self.tools.guidance()

        if self.memory is not None:
            # What it remembers, and how to recall it, while memory is on
            if self.remember:
                system_content += self._memory_context()
            # The live context: the thread's stored turns, then any held back
            # by amnesia mode, together within the live budget
            unremembered, used = [], 0
            for turn in reversed(self.unremembered):
                used += estimate_tokens(turn["content"])
                if unremembered and used > LIVE_TOKENS:
                    break
                unremembered.insert(0, turn)
            recent = self.memory.store.recent_turns(self.thread, max(0, LIVE_TOKENS - used))
            stored = [{"role": turn["role"], "content": turn["text"]} for turn in recent]
            return [{"role": "system", "content": system_content}] + stored + unremembered

        system_prompt = {
            "role": "system",
            "content": system_content,
        }

        if not self.conversation_history:
            return [system_prompt]

        # Estimate current token usage
        messages = [system_prompt] + self.conversation_history
        current_tokens = self.get_conversation_token_count(messages)

        # If under limit, return as-is
        if current_tokens < self.max_context_tokens:
            return messages

        # Need to compress - keep as many recent messages as possible
        system_token_estimate = self.estimate_tokens(system_prompt["content"])
        available_tokens = (
            self.max_context_tokens - system_token_estimate - 500
        )  # Reserve 500 for summary

        # Start from the end and work backwards to keep as much recent context as possible
        recent_messages = []
        tokens_used = 0
        for msg in reversed(self.conversation_history):
            msg_tokens = self.estimate_tokens(msg["content"])
            if tokens_used + msg_tokens < available_tokens:
                recent_messages.insert(0, msg)
                tokens_used += msg_tokens
            else:
                break

        # Everything not kept needs to be summarized
        if len(recent_messages) < len(self.conversation_history):
            old_messages = self.conversation_history[: -len(recent_messages)]
            summary_text = self.summarize_old_messages(old_messages)

            # Build new message list with summary
            managed_messages = [
                system_prompt,
                {
                    "role": "system",
                    "content": f"Previous conversation summary: {summary_text}",
                },
            ] + recent_messages
        else:
            # All messages fit (shouldn't happen as we already checked, but safety)
            managed_messages = messages

        return managed_messages

    def chat_stream(
        self,
        prompt: str,
        response_language: str = "en",
        on_tool_use=None,
        source=None,
        attachments: Optional[list[Attachment]] = None,
        keep: bool = True,
        on_tool_result=None,
    ):
        """
        Send prompt to LLM and stream response with tool support.
        Automatically switches to vision model when vision tools are used.

        Args:
            prompt: User prompt text
            response_language: Language the assistant should respond in
                (Whisper ISO 639-1 code, e.g. 'en', 'fr', 'ja')
            on_tool_use: Optional callback(tool_name, tool_args) called before each tool executes
            on_tool_result: Optional callback(tool_name, result) called after each tool ran
            attachments: Files sent with the message; they stay in context for
                a few turns, and only their names are recorded
            keep: Keep the exchange in the conversation once the reply is
                complete. False leaves that to the caller (keep()), which
                knows how much of the reply was heard; the complete reply,
                as it would be kept, is then in self.reply.

        Yields:
            Chunks of response text, without reasoning or tool calls a model
            wrote out as text (see iris.markup)
        """
        self.reply = None
        self._reply_began = self._forgotten
        self._cancelled.clear()
        client = self._streaming = self._reply_client()
        try:
            yield from self._reply_stream(
                prompt, response_language, on_tool_use, source, attachments or [], on_tool_result
            )
        finally:
            if self._streaming is client:
                self._streaming = None
        # A reply cut off (cancel()) never completed: there is nothing to keep
        if keep and self.reply is not None:
            self._record(recorded_text(prompt, attachments or []), self.reply, source)
        elif keep:
            self.forgo()

    def cancel(self):
        """Stop the reply being streamed, from another thread: the request
        is cut off, so the server stops generating at once, and no further
        tool or tool round starts. The stream then ends or fails; whoever
        cancelled no longer listens to it."""
        self._cancelled.set()
        client = self._streaming or self._reply_client()
        if client is self._own_client:
            self._own_client = None
        elif client is self.client:
            self.client = ollama_client(self.host)
            if getattr(client, "tools_refused", False):
                self.client.tools_refused = True
        # Otherwise the reply came from a server since switched away from,
        # whose client is no longer kept
        try:
            client._client.close()  # the client's httpx.Client
        except Exception:
            pass

    def _reply_client(self):
        """The client replies are streamed through: one whose connection
        cancel() can cut. The ollama module's shared client, used when no
        server is configured, stays open for everything else."""
        if not isinstance(self.client, types.ModuleType):
            return self.client
        if self._own_client is None:
            self._own_client = ollama.Client(timeout=MODEL_TIMEOUT)
        return self._own_client

    def _ask(self, model: str, messages: list, **extra):
        """Ask the model for a streamed reply to these messages, on the
        server the reply began on."""
        self.asked = model
        return (self._streaming or self._reply_client()).chat(
            model=model, messages=system_first(messages), stream=True, **extra
        )

    def _open(self, model: str, messages: list, schemas: Optional[list]):
        """The model's streamed reply, with these tools on offer, and the
        tools it was asked with in the end.

        An Ollama model that cannot take tools refuses a request with them
        (400, "... does not support tools"). It is asked again without them,
        and without the system prompt's guidance on them, and is not
        offered tools again (OpenAIClient does the same for its servers).
        """
        if not schemas:
            return self._ask(model, messages), None
        stream = iter(self._ask(model, messages, tools=schemas))
        try:
            first = next(stream)
        except StopIteration:
            return iter(()), schemas
        except ollama.ResponseError as error:
            if TOOLS_REFUSED not in str(error.error).lower():
                raise
            logger.info("%s does not support tools; asking it without them", model)
            self._no_tools.add(model)
            guidance = self.tools.guidance()
            if guidance and messages[0].get("role") == "system":
                content = messages[0]["content"].replace(guidance, "", 1)
                messages[0] = {**messages[0], "content": content}
            return self._ask(model, messages), None
        return itertools.chain([first], stream), schemas

    def keep(self, prompt: str, answer: str, source: Optional[str] = None):
        """Keep an exchange streamed with keep=False: its reply as far as
        the user heard it, or self.reply when it was heard to the end. The
        prompt is recorded as given, so with attachments it is the one
        recorded_text() makes."""
        self._record(prompt, answer, source)

    def _reply_stream(
        self,
        prompt: str,
        response_language: str,
        on_tool_use,
        source,
        attachments: list[Attachment],
        on_tool_result=None,
    ):
        """The model's reply as it streams in, tool rounds included."""
        # Language names for the LLM instruction
        _lang_names = {
            "en": "English", "fr": "French", "it": "Italian",
            "ja": "Japanese", "zh": "Chinese (Mandarin)", "de": "German",
            "es": "Spanish", "pt": "Portuguese", "nl": "Dutch", "hi": "Hindi",
        }

        # Recalled topics and shared files fade from context after a few turns
        self.pinned = {t: n - 1 for t, n in self.pinned.items() if n > 1}
        self.shared = [(a, n - 1) for a, n in self.shared if n > 1]
        self.shared += [(a, PIN_TURNS) for a in attachments]

        # Shared images are answered by a model that can see them
        images = [a.image for a, _ in self.shared if a.image]
        sees = self.image_model() if images else None
        model = sees or self.model

        # Tools, and how to use them, for a model that takes them
        use_tools = bool(self.tools and self.tools.has_tools()) and self._takes_tools(model)
        schemas = self.tools.get_schemas() if use_tools else None

        # Get managed conversation context
        messages = self.manage_context(with_tools=use_tools)
        # The turn is stored now that the context is made without it
        self._store_asked(recorded_text(prompt, attachments), source)

        # Inject a language instruction so the LLM replies in the right
        # language: any but English, and English when the user goes back to
        # it, or the model carries on in the language the conversation was in
        if response_language != "en" or self.language != "en":
            lang_name = _lang_names.get(response_language, response_language)
            messages.append({
                "role": "system",
                "content": f"The user is speaking in {lang_name}. Respond in {lang_name}.",
            })
        self.language = response_language

        # Add current user message, with the files still in context
        user_message = {"role": "user", "content": prompt + self._files_context(attachments)}
        if sees:
            user_message["images"] = images
        elif any(a.image for a in attachments):
            # Said with the pictures; they stay among the files shared for a
            # few turns, and saying it on each of them had the model tell the
            # user again on every reply
            user_message["content"] += "\n\n" + IMAGES_UNSEEN
        messages.append(user_message)

        # The reply streams as it comes, so the voice can start before the
        # model has finished; text before a tool call ("Let me look that
        # up") is said while the tool runs. Each round may end in tool
        # calls, whose results the next round answers.
        spoken: list[str] = []
        rounds = 0
        response_model = model
        asked_for_words = False
        while True:
            if self._cancelled.is_set():
                return
            stream, schemas = self._open(response_model, messages, schemas)
            markup = MarkupFilter()
            said, tool_calls = [], None
            for chunk in stream:
                message_chunk = chunk.get("message", {})
                if text := markup.feed(message_chunk.get("content") or ""):
                    said.append(text)
                    yield text
                # Tool calls may come in any chunk, not just the last
                if message_chunk.get("tool_calls"):
                    tool_calls = message_chunk["tool_calls"]
            if text := markup.flush():
                said.append(text)
                yield text
            spoken += said

            # A call written out as text is run as if made, when it is
            # unambiguous; otherwise it is only left unsaid. A reply that
            # was nothing but such a call is asked for again, in words.
            if not tool_calls and markup.calls:
                tool_calls = self._typed_calls(markup.calls, schemas)
                if not tool_calls and not "".join(said).strip() and not asked_for_words:
                    asked_for_words = True
                    schemas = None
                    messages.append({"role": "system", "content": NO_TOOLS_NOTE})
                    continue
            if not tool_calls:
                break

            rounds += 1
            messages.append(
                {"role": "assistant", "content": "".join(said), "tool_calls": tool_calls}
            )
            ran = self._run_tools(tool_calls, messages, on_tool_use, on_tool_result)
            if ran is None:
                return  # cancelled
            uses_vision, ends = ran
            # A call that ends the reply (going to sleep) came with what the
            # reply had to say: asked again, the model would say it twice.
            # Said nothing yet, it answers once more, without tools.
            if ends and "".join(spoken).strip():
                break
            # A tool's image is answered by a model that sees it: the model
            # of the conversation if it can, else the vision model. Without
            # either, the image was left out and the model answers in words.
            response_model = (self.image_model() or model) if uses_vision else model

            # Tools stay on offer, so a search can be followed by reading
            # a page: without them a model writes its next call out as
            # text. A separate vision model, and the last round, answer
            # without tools.
            offer_tools = not ends and rounds < MAX_TOOL_ROUNDS and response_model == model
            schemas = self.tools.get_schemas() if offer_tools else None
            if not offer_tools:
                messages.append({"role": "system", "content": NO_MORE_TOOLS_NOTE})

        answer = "".join(spoken)
        # Without tools, kept in the conversation history as cleaned text
        self.reply = answer if use_tools else self.clean_markdown(answer)

    @staticmethod
    def _typed_calls(calls: list[TypedCall], schemas: Optional[list]) -> Optional[list]:
        """Calls written out as text that can be run as if made: of a tool
        offered with the request, with arguments that parse."""
        offered = {schema["function"]["name"] for schema in schemas or []}
        runnable = [c for c in calls if c.name in offered and c.arguments is not None]
        for call in calls:
            if call not in runnable:
                logger.info("Left out a tool call written as text: %r", call)
        if not runnable:
            return None
        return [{"function": {"name": c.name, "arguments": c.arguments}} for c in runnable]

    def _run_tools(
        self, tool_calls: list, messages: list, on_tool_use, on_tool_result=None
    ) -> Optional[tuple[bool, bool]]:
        """Run a round's tool calls, adding their results to the messages.
        Returns whether one returned an image a model can see and whether
        every call ends the reply (going to sleep), or None if the reply was
        cancelled before all had run. An image no model at hand can see is
        left out, and the model is told so."""
        uses_vision = False
        ends = True
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            tool_args = tool_call["function"].get("arguments", {})

            if self._cancelled.is_set():
                return None
            if on_tool_use:
                on_tool_use(tool_name, tool_args)

            # Execute the tool (it shows its activity meanwhile)
            result = self.tools.execute(tool_name, **tool_args)
            if on_tool_result:
                on_tool_result(tool_name, result)
            ends = ends and bool(result.get("success") and result.get("ends_reply"))
            if result.get("success") and result.get("pin_topic") is not None:
                self.pinned[result["pin_topic"]] = PIN_TURNS

            # A failed tool's error goes to the model; an image is shown to it
            tool_msg = {"role": "tool", "content": ""}
            if not result.get("success"):
                tool_msg["content"] = f"Error: {result.get('message', 'Tool execution failed')}"
            elif result.get("image") and self.image_model() is None:
                tool_msg["content"] = result.get("message", "Image captured") + " " + IMAGE_UNSEEN
            elif result.get("image"):
                uses_vision = True
                tool_msg["images"] = [result["image"]]
                tool_msg["content"] = result.get("message", "Image captured")
            elif result.get("result"):
                tool_msg["content"] = str(result["result"])
            else:
                tool_msg["content"] = result.get("message", "Tool executed")
            messages.append(tool_msg)
        return uses_vision, ends
