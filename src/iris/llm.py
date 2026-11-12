"""LLM conversation management, on Ollama or an OpenAI-compatible server."""

import re
import logging
import threading
import types
import warnings
from typing import Optional

import markdown
import ollama
from bs4 import BeautifulSoup

from iris.attachments import Attachment, recorded_text
from iris.llm_client import OpenAIClient, is_openai, normalize_host
from iris.llm_client import effective_host as effective_host  # imported from here by callers
from iris.memory import LIVE_TOKENS, PIN_TURNS, PINNED_TOKENS, MemoryKeeper, estimate_tokens
from iris.personas import Persona, get_persona

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
# Rounds of tool calls one reply may take, e.g. a search, then reading a page
MAX_TOOL_ROUNDS = 3


def ollama_client(host: Optional[str] = None, **kwargs):
    """A client for a server URL as resolve_host keeps it: an OpenAIClient
    for an OpenAI-compatible server, ollama's own client for Ollama, or the
    ollama module itself (whose default client follows OLLAMA_HOST) when no
    server is configured."""
    host = normalize_host(host)
    if is_openai(host):
        return OpenAIClient(host, **kwargs)
    if not host and not kwargs:
        return ollama
    return ollama.Client(host=host, **kwargs)


# Markup some models write into a reply that is never meant to be spoken:
# their reasoning, and tool calls typed out as text instead of made
UNSPOKEN_TAGS = ("think", "tool_call")


class UnspokenFilter:
    """Drops <think>...</think> and <tool_call>...</tool_call> from streamed
    text. A tag can be split across chunks, so text that might start one is
    held back until the next chunk shows whether it does."""

    def __init__(self):
        self.held = ""
        self.closing = None  # the closing tag being waited for

    def feed(self, text: str) -> str:
        self.held += text
        out = []
        while self.held:
            if self.closing:
                end = self.held.find(self.closing)
                if end < 0:
                    # Keep only what could be the start of the closing tag
                    self.held = self.held[-(len(self.closing) - 1) :]
                    break
                self.held = self.held[end + len(self.closing) :]
                self.closing = None
                continue
            start = self.held.find("<")
            if start < 0:
                out.append(self.held)
                self.held = ""
                break
            out.append(self.held[:start])
            self.held = self.held[start:]
            tag = next((t for t in UNSPOKEN_TAGS if self.held.startswith(f"<{t}>")), None)
            if tag:
                self.held = self.held[len(tag) + 2 :]
                self.closing = f"</{tag}>"
            elif any(f"<{t}>".startswith(self.held) for t in UNSPOKEN_TAGS):
                break  # may still become a tag
            else:
                out.append("<")
                self.held = self.held[1:]
        return "".join(out)

    def flush(self) -> str:
        """Text held back at the end; an unclosed tag is dropped."""
        rest = "" if self.closing else self.held
        self.held, self.closing = "", None
        return rest


def strip_unspoken(text: str) -> str:
    unspoken = UnspokenFilter()
    return (unspoken.feed(text) + unspoken.flush()).strip()


# A bullet or numbered list marker at the start of a line
LIST_ITEM = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+")
# A line already ending in punctuation that makes the voice pause, perhaps
# followed by closing quotes or brackets
PAUSED_END = re.compile(r"[.!?…:;,][\"'”’)\]]*$")
# Where streamed text may be cut for speech: after a sentence or at a line
SPEECH_BREAK = re.compile(r"[.!?](?= )|\n")
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
                and the context is built from it (see iris.memory)
        """
        self.host = normalize_host(host)
        self.client = ollama_client(self.host)
        # Set by cancel(): the reply being streamed is not to go on
        self._cancelled = threading.Event()
        # The client replies stream through while self.client is the ollama
        # module, whose shared client must stay open
        self._own_client = None
        self.memory = memory
        # Recalled topics stay in context for a few turns: {topic_id: turns left}
        self.pinned: dict[int, int] = {}
        # Amnesia mode: the conversation goes on, but nothing of it is
        # written to memory; its turns live here and vanish when it ends
        self.amnesia = False
        self.unremembered: list[dict] = []
        # Files shared in this conversation, kept for a few turns and never
        # stored: [(attachment, turns left)], oldest first
        self.shared: list[tuple[Attachment, int]] = []
        self._capability_cache: dict[str, set] = {}
        self.model = model
        self.vision_model = vision_model
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
        """Switch to another server, keeping the conversation history."""
        self.host = normalize_host(host)
        self.client = ollama_client(self.host)
        self.reachable = True  # set_host is only called for a server that answered
        self._capability_cache = {}
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

        # Remove emojis using regex (matches all emoji characters)
        emoji_pattern = re.compile(
            "["
            "\U0001f600-\U0001f64f"  # emoticons
            "\U0001f300-\U0001f5ff"  # symbols & pictographs
            "\U0001f680-\U0001f6ff"  # transport & map symbols
            "\U0001f1e0-\U0001f1ff"  # flags (iOS)
            "\U00002702-\U000027b0"
            "\U000024c2-\U0001f251"
            "]+",
            flags=re.UNICODE,
        )
        clean_text = emoji_pattern.sub("", clean_text)

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
            summary_response = self.client.chat(
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
            )
            return summary_response.get("message", {}).get(
                "content", "Previous conversation context."
            )
        except Exception as e:
            print(f"Warning: Could not summarize: {e}")
            # Fallback: just take first message of each type
            return "Previous conversation context."

    def _memory_context(self) -> str:
        """The memory part of the system prompt: what Iris remembers (the
        topic index) and the topics it recalled in the last few turns."""
        store = self.memory.store
        lines = store.index_lines()
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
            topic = store.topic(topic_id)
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
        """Stop (or resume) writing the conversation to memory. Turns spoken
        in amnesia mode are forgotten entirely once it ends."""
        self.amnesia = bool(on)
        self.unremembered = []

    def _record(self, prompt: str, answer: str, source: Optional[str]):
        """Keep the exchange: in memory when there is one, else in history."""
        # Only what was said: a tool call typed out as text, left in the
        # history, would be imitated in later replies
        answer = strip_unspoken(answer)
        if self.memory is not None and self.amnesia:
            self.unremembered.append({"role": "user", "content": prompt})
            self.unremembered.append({"role": "assistant", "content": answer})
            return
        if self.memory is None:
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": answer})
            return
        self.memory.store.add_turn("user", prompt, source)
        self.memory.store.add_turn("assistant", answer)
        self.memory.note_turn()

    def add_note(self, text: str):
        """A system note in the conversation, e.g. that the persona changed."""
        if self.memory is not None and self.amnesia:
            self.unremembered.append({"role": "system", "content": text})
        elif self.memory is None:
            self.conversation_history.append({"role": "system", "content": text})
        else:
            self.memory.store.add_turn("system", text)

    def manage_context(self) -> list:
        """
        Manage conversation history to fit within context window.
        Summarizes old messages if needed.

        Returns:
            Managed list of messages
        """
        # Build system prompt with persona description
        system_content = self.persona.description

        # How to use the tools that are on, from each tool's guidance
        if self.tools:
            system_content += self.tools.guidance()

        if self.memory is not None:
            system_content += self._memory_context()
            # The live thread: stored turns, then any held back by amnesia
            # mode, together within the live budget
            unremembered, used = [], 0
            for turn in reversed(self.unremembered):
                used += estimate_tokens(turn["content"])
                if unremembered and used > LIVE_TOKENS:
                    break
                unremembered.insert(0, turn)
            stored = [
                {"role": turn["role"], "content": turn["text"]}
                for turn in self.memory.store.recent_turns(max(0, LIVE_TOKENS - used))
            ]
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
    ):
        """
        Send prompt to LLM and stream response with tool support.
        Automatically switches to vision model when vision tools are used.

        Args:
            prompt: User prompt text
            response_language: Language the assistant should respond in
                (Whisper ISO 639-1 code, e.g. 'en', 'fr', 'ja')
            on_tool_use: Optional callback(tool_name, tool_args) called before each tool executes
            attachments: Files sent with the message; they stay in context for
                a few turns, and only their names are recorded
            keep: Keep the exchange in the conversation once the reply is
                complete. False leaves that to the caller (keep()), which
                knows how much of the reply was heard; the complete reply,
                as it would be kept, is then in self.reply.

        Yields:
            Chunks of response text, without reasoning or tool calls a model
            wrote out as text
        """
        self.reply = None
        self._cancelled.clear()
        unspoken = UnspokenFilter()
        for chunk in self._reply_stream(
            prompt, response_language, on_tool_use, source, attachments or []
        ):
            if text := unspoken.feed(chunk):
                yield text
        if rest := unspoken.flush():
            yield rest
        if keep:
            self._record(recorded_text(prompt, attachments or []), self.reply, source)

    def cancel(self):
        """Stop the reply being streamed, from another thread: the request
        is cut off, so the server stops generating at once, and no further
        tool or tool round starts. The stream then ends or fails; whoever
        cancelled no longer listens to it."""
        self._cancelled.set()
        client = self._reply_client()
        if client is self._own_client:
            self._own_client = None
        else:
            self.client = ollama_client(self.host)
            if getattr(client, "tools_refused", False):
                self.client.tools_refused = True
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
            self._own_client = ollama.Client()
        return self._own_client

    def _ask(self, model: str, messages: list, **extra):
        """Ask the model for a streamed reply to these messages."""
        return self._reply_client().chat(
            model=model, messages=system_first(messages), stream=True, **extra
        )

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
    ):
        """The model's reply as it streams in, tool rounds included."""
        # Language names for the LLM instruction
        _lang_names = {
            "en": "English", "fr": "French", "it": "Italian",
            "ja": "Japanese", "zh": "Chinese (Mandarin)", "de": "German",
            "es": "Spanish", "pt": "Portuguese", "nl": "Dutch",
        }

        # Recalled topics and shared files fade from context after a few turns
        self.pinned = {t: n - 1 for t, n in self.pinned.items() if n > 1}
        self.shared = [(a, n - 1) for a, n in self.shared if n > 1]
        self.shared += [(a, PIN_TURNS) for a in attachments]

        # Get managed conversation context
        messages = self.manage_context()

        # Inject a language instruction so the LLM replies in the right language
        if response_language != "en":
            lang_name = _lang_names.get(response_language, response_language)
            messages.append({
                "role": "system",
                "content": f"The user is speaking in {lang_name}. Respond in {lang_name}.",
            })

        # Add current user message, with the files still in context
        user_message = {"role": "user", "content": prompt + self._files_context(attachments)}
        model = self.model
        images = [a.image for a, _ in self.shared if a.image]
        if images:
            model = self.image_model()
            if model:
                user_message["images"] = images
            else:
                model = self.model
                user_message["content"] += (
                    "\n\n(The images cannot be shown to you: no model that can see "
                    "images is installed. Let the user know.)"
                )
        messages.append(user_message)

        # A model chosen for its sight may not take tools
        use_tools = self.tools and self.tools.has_tools()
        if use_tools and model != self.model:
            use_tools = "tools" in (self._capabilities(model) or set())

        # If tools available, let LLM decide if it needs them
        if use_tools:
            stream = self._ask(model, messages, tools=self.tools.get_schemas())

            # The reply streams as it comes, so the voice can start before
            # the model has finished; text before a tool call ("Let me look
            # that up") is said while the tool runs
            full_response = []
            tool_calls = None

            for chunk in stream:
                message_chunk = chunk.get("message", {})
                chunk_text = message_chunk.get("content") or ""
                if chunk_text:
                    full_response.append(chunk_text)
                    yield chunk_text

                # Check for tool calls - they might appear in any chunk, not just the last
                if message_chunk.get("tool_calls"):
                    tool_calls = message_chunk["tool_calls"]

            # If LLM wants to use tools, execute them and get new response
            if tool_calls:
                spoken = list(full_response)
                rounds = 0
                while tool_calls:
                    rounds += 1
                    # Build message with tool call
                    messages.append(
                        {
                            "role": "assistant",
                            "content": "".join(full_response),
                            "tool_calls": tool_calls,
                        }
                    )

                    # A tool that returned an image is answered by the vision model
                    uses_vision = False
                    for tool_call in tool_calls:
                        tool_name = tool_call["function"]["name"]
                        tool_args = tool_call["function"].get("arguments", {})

                        if self._cancelled.is_set():
                            return
                        if on_tool_use:
                            on_tool_use(tool_name, tool_args)

                        # Execute the tool (it shows its activity meanwhile)
                        result = self.tools.execute(tool_name, **tool_args)
                        if result.get("success") and result.get("pin_topic") is not None:
                            self.pinned[result["pin_topic"]] = PIN_TURNS

                        # Format tool response for LLM
                        tool_msg = {"role": "tool", "content": ""}

                        # Check if tool execution failed
                        if not result.get("success"):
                            # Tool failed - tell LLM about the error
                            tool_msg["content"] = (
                                f"Error: {result.get('message', 'Tool execution failed')}"
                            )
                            messages.append(tool_msg)
                            continue

                        # If tool returned an image, include it
                        if result.get("image"):
                            uses_vision = True
                            tool_msg["images"] = [result["image"]]
                            tool_msg["content"] = result.get("message", "Image captured")
                        elif result.get("result"):
                            tool_msg["content"] = str(result["result"])
                        else:
                            tool_msg["content"] = result.get("message", "Tool executed")

                        messages.append(tool_msg)

                    # Choose model: use vision model if vision was used, otherwise primary model
                    response_model = (
                        self.vision_model if (uses_vision and self.vision_model) else model
                    )

                    # Tools stay on offer, so a search can be followed by reading
                    # a page: without them a model writes its next call out as
                    # text, which would be shown and spoken. A separate vision
                    # model, and the last round, answer without tools.
                    offer_tools = rounds < MAX_TOOL_ROUNDS and response_model == model
                    extra = {"tools": self.tools.get_schemas()} if offer_tools else {}
                    if not offer_tools:
                        messages.append(
                            {
                                "role": "system",
                                "content": "Answer the user now from what you have. "
                                "No more tools can be used.",
                            }
                        )
                    if self._cancelled.is_set():
                        return
                    stream = self._ask(response_model, messages, **extra)

                    # Stream the answer as it comes; a round may end in more
                    # tool calls instead
                    full_response, tool_calls = [], None
                    for chunk in stream:
                        message_chunk = chunk.get("message", {})
                        chunk_text = message_chunk.get("content") or ""
                        if chunk_text:
                            full_response.append(chunk_text)
                            spoken.append(chunk_text)
                            yield chunk_text
                        if message_chunk.get("tool_calls"):
                            tool_calls = message_chunk["tool_calls"]

                # Store in history
                answer = "".join(spoken)
                self.reply = answer
                return
            else:
                answer = "".join(full_response)
                self.reply = answer
                return

        # No tools - standard streaming
        stream = self._ask(model, messages)

        full_response = []
        for chunk in stream:
            chunk_text = chunk["message"]["content"]
            full_response.append(chunk_text)
            yield chunk_text

        # Kept in the conversation history as cleaned text
        self.reply = self.clean_markdown("".join(full_response))
