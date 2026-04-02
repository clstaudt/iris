"""LLM conversation management using Ollama."""

import re
import warnings
from typing import Optional

import markdown
import ollama
from bs4 import BeautifulSoup

from iris.personas import Persona, get_persona

warnings.filterwarnings("ignore")


class ConversationManager:
    """Manages LLM conversations with context and summarization."""

    def __init__(
        self,
        model: str = "qwen3:1.7b",
        persona: Optional[Persona] = None,
        tools=None,
        vision_model: Optional[str] = None,
        quiet: bool = True,
    ):
        """
        Initialize conversation manager.

        Args:
            model: Ollama model name (primary/text model)
            persona: Persona to use (defaults to iris)
            tools: Optional Tools instance for function calling
            vision_model: Optional vision model to switch to when using vision tools
            quiet: If True, suppress verbose output (setup wizard handles this)
        """
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
        try:
            ollama.list()
            if not quiet:
                print("Ollama connected")
        except Exception as e:
            print(f"Ollama connection failed: {e}")
            print("Make sure Ollama is running: brew services start ollama")
            raise

        if not quiet:
            print(f"Model context length: {self.max_context_tokens:,} tokens")

    def _get_model_context_length(self) -> int:
        """
        Get the model's context length from Ollama.
        Uses 80% of the actual context length as a safety margin.

        Returns:
            Maximum context tokens to use (80% of model's limit)
        """
        try:
            model_info = ollama.show(self.model)
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
        html = markdown.markdown(text)
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

        return clean_text.strip()

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
            summary_response = ollama.chat(
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
                    "- 'capture_screen' function: Your ONLY way to see the COMPUTER SCREEN.\n"
                    "  * YOU MUST call this function for ANY question about the screen, windows, or applications\n"
                    "  * NEVER describe what's on screen without calling this function first\n"
                    "  * If asked about 'screen', 'monitor', 'display', 'window', 'application' - USE THIS TOOL\n"
                    "  * Do NOT guess or make up what's on screen - you CANNOT see it without this function\n"
                )

            if "search" in tool_names:
                tool_instructions += "- 'search' function: Use this to find current information, news, or facts you don't know.\n"

            tool_instructions += "\nIMPORTANT: You MUST use these functions when appropriate. NEVER make up or guess what you see."

            system_content += tool_instructions

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

    def chat_stream(self, prompt: str, response_language: str = "en"):
        """
        Send prompt to LLM and stream response with tool support.
        Automatically switches to vision model when vision tools are used.

        Args:
            prompt: User prompt text
            response_language: Language the assistant should respond in
                (Whisper ISO 639-1 code, e.g. 'en', 'fr', 'ja')

        Yields:
            Chunks of response text
        """
        # Language names for the LLM instruction
        _lang_names = {
            "en": "English", "fr": "French", "it": "Italian",
            "ja": "Japanese", "zh": "Chinese (Mandarin)",
        }

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
            stream = ollama.chat(
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

                    # Execute the tool
                    result = self.tools.execute(tool_name, **tool_args)

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

                # Now get final response with tool results (streaming)
                stream = ollama.chat(model=response_model, messages=messages, stream=True)

                full_response = []
                for chunk in stream:
                    chunk_text = chunk["message"]["content"]
                    full_response.append(chunk_text)
                    yield chunk_text

                # Store in history
                answer = "".join(full_response)
                self.conversation_history.append({"role": "user", "content": prompt})
                self.conversation_history.append({"role": "assistant", "content": answer})
                return
            else:
                # No tools needed - yield the response we already collected
                answer = "".join(full_response)
                for chunk_text in full_response:
                    if chunk_text:
                        yield chunk_text

                self.conversation_history.append({"role": "user", "content": prompt})
                self.conversation_history.append({"role": "assistant", "content": answer})
                return

        # No tools - standard streaming
        stream = ollama.chat(model=self.model, messages=messages, stream=True)

        full_response = []
        for chunk in stream:
            chunk_text = chunk["message"]["content"]
            full_response.append(chunk_text)
            yield chunk_text

        # Store in conversation history (cleaned version)
        answer = "".join(full_response)
        answer_clean = self.clean_markdown(answer)

        self.conversation_history.append({"role": "user", "content": prompt})
        self.conversation_history.append({"role": "assistant", "content": answer_clean})
