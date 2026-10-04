"""LLM conversation management using Ollama."""

import os
import re
import warnings
from typing import Optional

import markdown
import ollama
from bs4 import BeautifulSoup

from iris.memory import LIVE_TOKENS, PIN_TURNS, PINNED_TOKENS, MemoryKeeper, estimate_tokens
from iris.personas import Persona, get_persona

warnings.filterwarnings("ignore")

# Fallback shown when neither a server URL nor OLLAMA_HOST is configured
DEFAULT_OLLAMA_URL = "http://localhost:11434"
# Rounds of tool calls one reply may take, e.g. a search, then reading a page
MAX_TOOL_ROUNDS = 3


def normalize_host(url: Optional[str]) -> Optional[str]:
    """Clean up a server URL as people copy it around.

    The OpenAI-compatible endpoint (".../v1") is what most tools ask for, but
    the native client appends its own /api paths and must get the bare server.
    Returns None for an empty value, meaning "use the default".
    """
    url = (url or "").strip().rstrip("/")
    for suffix in ("/v1", "/api"):
        if url.endswith(suffix):
            url = url[: -len(suffix)].rstrip("/")
    return url or None


def ollama_client(host: Optional[str] = None, **kwargs):
    """A client for the given server, or the ollama module itself — whose
    default client follows OLLAMA_HOST — when no server is configured."""
    if not host and not kwargs:
        return ollama
    return ollama.Client(host=normalize_host(host), **kwargs)


def effective_host(host: Optional[str]) -> str:
    """The server URL actually in use, for display."""
    return normalize_host(host) or os.environ.get("OLLAMA_HOST") or DEFAULT_OLLAMA_URL


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


def speech_break(text: str, start: int, end: int) -> int:
    """Index after the last sentence end or line break in text[start:end],
    where the text can be cut for speech, or -1 if there is none."""
    cut = -1
    for match in SPEECH_BREAK.finditer(text, start, end):
        if not LIST_NUMBER.search(text, 0, match.end()):
            cut = match.end()
    return cut


class ConversationManager:
    """Manages LLM conversations with context and summarization."""

    def __init__(
        self,
        model: str = "qwen3:1.7b",
        persona: Optional[Persona] = None,
        tools=None,
        vision_model: Optional[str] = None,
        quiet: bool = True,
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
            quiet: If True, suppress verbose output (setup wizard handles this)
            host: Ollama server URL; None follows OLLAMA_HOST
            memory: Long-term memory; when given, the conversation is stored
                and the context is built from it (see iris.memory)
        """
        self.host = normalize_host(host)
        self.client = ollama_client(self.host)
        self.memory = memory
        # Recalled topics stay in context for a few turns: {topic_id: turns left}
        self.pinned: dict[int, int] = {}
        # Amnesia mode: the conversation goes on, but nothing of it is
        # written to memory; its turns live here and vanish when it ends
        self.amnesia = False
        self.unremembered: list[dict] = []
        self.model = model
        self.vision_model = vision_model
        self.conversation_history = []
        self.quiet = quiet
        self.persona = persona if persona else get_persona("iris")
        self.tools = tools
        self.max_context_tokens = self._get_model_context_length()

        # Test Ollama connection
        if not quiet:
            print(f"Testing Ollama connection with model '{model}'...")
        # Not fatal: the app starts without a server and says what is missing,
        # and a server can be set later (set_host)
        try:
            self.client.list()
            self.reachable = True
            if not quiet:
                print("Ollama connected")
        except Exception as e:
            self.reachable = False
            print(f"Ollama connection failed: {e}")
            print("Make sure Ollama is running: brew services start ollama")

        if not quiet:
            print(f"Model context length: {self.max_context_tokens:,} tokens")

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
        """Switch to another Ollama server, keeping the conversation history."""
        self.host = normalize_host(host)
        self.client = ollama_client(self.host)
        self.reachable = True  # set_host is only called for a server that answered
        self.max_context_tokens = self._get_model_context_length()

    def set_model(self, model: str):
        """Switch the primary model, keeping conversation history intact."""
        self.model = model
        self.max_context_tokens = self._get_model_context_length()

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
                if not self.quiet:
                    print(
                        f"Detected model context: {context_length:,} tokens "
                        f"(using {usable_context:,})"
                    )
                return usable_context
            else:
                # Fallback: use a conservative default
                if not self.quiet:
                    print("Could not detect model context length, using default: 25,600 tokens")
                return 25600

        except Exception as e:
            if not self.quiet:
                print(
                    f"Warning: Could not query model info ({e}), "
                    f"using default context: 25,600 tokens"
                )
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

        # Add tool awareness if tools are available
        if self.tools and self.tools.has_tools():
            tool_names = list(self.tools._enabled_tools.keys())

            # Add general tool instruction
            tool_instructions = (
                "\n\nIMPORTANT - You have access to tools that you must use appropriately:\n"
            )

            if "look" in tool_names:
                tool_instructions += (
                    "- 'look' function: Your CAMERA. You CANNOT see through the camera without calling this.\n"
                    "  * YOU MUST call 'look' for ANY question about physical objects, gestures, or camera view\n"
                    "  * NEVER describe what the camera sees without calling this function first\n"
                    "  * Call it EVERY TIME they say 'look again' - always capture fresh\n"
                )

            if "capture_screen" in tool_names:
                tool_instructions += (
                    "- 'capture_screen' function: Screenshot of the user's PHYSICAL computer screen.\n"
                    "  * ONLY use when they ask about what is on their screen, desktop, or open applications\n"
                    "  * NEVER use this to read websites - use 'browse' for websites\n"
                    "  * 'page', 'article', 'headline', 'website' refer to the WEB, not the screen\n"
                )

            if "search" in tool_names:
                tool_instructions += (
                    "- 'search' function: Search the web for CURRENT information.\n"
                    "  * YOU MUST call this for questions about current events, news, weather, prices, or recent facts\n"
                    "  * Use when the user asks about something you're unsure of or that changes over time\n"
                    "  * NEVER guess at current facts - search first\n"
                )

            if "browse" in tool_names:
                tool_instructions += (
                    "- 'browse' function: Fetch and read the contents of a specific webpage.\n"
                    "  * When the user names a website (e.g. 'The Guardian', 'BBC News'), infer its URL and call this\n"
                    "  * Use to read, scan, or check any web page, article, or headline\n"
                    "  * Use to read full article content from URLs found via 'search'\n"
                    "  * Provide the full URL including https://\n"
                )

            tool_instructions += "\nIMPORTANT: You MUST use these functions when appropriate. NEVER make up or guess what you see."

            system_content += tool_instructions

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
        self, prompt: str, response_language: str = "en", on_tool_use=None, source=None
    ):
        """
        Send prompt to LLM and stream response with tool support.
        Automatically switches to vision model when vision tools are used.

        Args:
            prompt: User prompt text
            response_language: Language the assistant should respond in
                (Whisper ISO 639-1 code, e.g. 'en', 'fr', 'ja')
            on_tool_use: Optional callback(tool_name, tool_args) called before each tool executes

        Yields:
            Chunks of response text, without reasoning or tool calls a model
            wrote out as text
        """
        unspoken = UnspokenFilter()
        for chunk in self._reply_stream(prompt, response_language, on_tool_use, source):
            if text := unspoken.feed(chunk):
                yield text
        if rest := unspoken.flush():
            yield rest

    def _reply_stream(self, prompt: str, response_language: str, on_tool_use, source):
        """The model's reply as it streams in, tool rounds included."""
        # Language names for the LLM instruction
        _lang_names = {
            "en": "English", "fr": "French", "it": "Italian",
            "ja": "Japanese", "zh": "Chinese (Mandarin)",
        }

        # Recalled topics fade from context after a few turns
        self.pinned = {t: n - 1 for t, n in self.pinned.items() if n > 1}

        # Get managed conversation context
        messages = self.manage_context()

        # Inject a language instruction so the LLM replies in the right language
        if response_language != "en":
            lang_name = _lang_names.get(response_language, response_language)
            messages.append({
                "role": "system",
                "content": f"The user is speaking in {lang_name}. Respond in {lang_name}.",
            })

        # Add current user message
        messages.append({"role": "user", "content": prompt})

        # If tools available, let LLM decide if it needs them
        if self.tools and self.tools.has_tools():
            # Use streaming even for tool check - only buffer if we see tool calls
            stream = self.client.chat(
                model=self.model, messages=messages, tools=self.tools.get_schemas(), stream=True
            )

            # Collect response and check for tool calls
            full_response = []
            tool_calls = None

            for chunk in stream:
                message_chunk = chunk.get("message", {})

                # Collect content
                if "content" in message_chunk:
                    full_response.append(message_chunk["content"])

                # Check for tool calls - they might appear in any chunk, not just the last
                if message_chunk.get("tool_calls"):
                    tool_calls = message_chunk["tool_calls"]

            # If LLM wants to use tools, execute them and get new response
            if tool_calls:
                spoken = []
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

                    # Check if any tool is a vision tool (look/camera)
                    uses_vision = False
                    for tool_call in tool_calls:
                        tool_name = tool_call["function"]["name"]
                        tool_args = tool_call["function"].get("arguments", {})

                        # Check if this is a vision tool and set status BEFORE executing
                        if tool_name == "look":
                            uses_vision = True
                            # Set looking status immediately when we know LLM wants to look
                            if self.tools.status_manager:
                                with self.tools.status_manager.status_lock:
                                    self.tools.status_manager.status_looking = True
                        elif tool_name == "capture_screen":
                            uses_vision = True
                            if self.tools.status_manager:
                                with self.tools.status_manager.status_lock:
                                    self.tools.status_manager.status_capturing_screen = True

                        if on_tool_use:
                            on_tool_use(tool_name, tool_args)

                        # Execute the tool
                        result = self.tools.execute(tool_name, **tool_args)
                        if tool_name == "recall" and result.get("success"):
                            self.pinned[result["topic_id"]] = PIN_TURNS

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
                            tool_msg["images"] = [result["image"]]
                            tool_msg["content"] = result.get("message", "Image captured")
                        elif result.get("result"):
                            tool_msg["content"] = str(result["result"])
                        else:
                            tool_msg["content"] = result.get("message", "Tool executed")

                        messages.append(tool_msg)

                    # Choose model: use vision model if vision was used, otherwise primary model
                    response_model = (
                        self.vision_model if (uses_vision and self.vision_model) else self.model
                    )

                    # Tools stay on offer, so a search can be followed by reading
                    # a page: without them a model writes its next call out as
                    # text, which would be shown and spoken. A separate vision
                    # model, and the last round, answer without tools.
                    offer_tools = rounds < MAX_TOOL_ROUNDS and response_model == self.model
                    extra = {"tools": self.tools.get_schemas()} if offer_tools else {}
                    if not offer_tools:
                        messages.append(
                            {
                                "role": "system",
                                "content": "Answer the user now from what you have. "
                                "No more tools can be used.",
                            }
                        )
                    stream = self.client.chat(
                        model=response_model, messages=messages, stream=True, **extra
                    )

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
                self._record(prompt, answer, source)
                return
            else:
                # No tools needed - yield the response we already collected
                answer = "".join(full_response)
                for chunk_text in full_response:
                    if chunk_text:
                        yield chunk_text

                self._record(prompt, answer, source)
                return

        # No tools - standard streaming
        stream = self.client.chat(model=self.model, messages=messages, stream=True)

        full_response = []
        for chunk in stream:
            chunk_text = chunk["message"]["content"]
            full_response.append(chunk_text)
            yield chunk_text

        # Store in conversation history (cleaned version)
        answer = "".join(full_response)
        answer_clean = self.clean_markdown(answer)

        self._record(prompt, answer_clean, source)
