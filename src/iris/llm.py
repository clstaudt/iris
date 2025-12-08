"""LLM conversation management using Ollama."""

import re
import threading
import warnings

import markdown
import ollama
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")


class ConversationManager:
    """Manages LLM conversations with context and summarization."""

    def __init__(self, model: str = "qwen3:8b"):
        """
        Initialize conversation manager.

        Args:
            model: Ollama model name
        """
        self.model = model
        self.conversation_history = []
        self.max_context_tokens = self._get_model_context_length()

        # Test Ollama connection
        print(f"Testing Ollama connection with model '{model}'...")
        try:
            ollama.list()
            print("Ollama connected")
        except Exception as e:
            print(f"Ollama connection failed: {e}")
            print("Make sure Ollama is running: brew services start ollama")
            raise

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
                print(
                    f"Detected model context: {context_length:,} tokens "
                    f"(using {usable_context:,})"
                )
                return usable_context
            else:
                # Fallback: use a conservative default
                print(
                    "Could not detect model context length, using default: 25,600 tokens"
                )
                return 25600

        except Exception as e:
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
        total = 0
        for msg in messages:
            total += self.estimate_tokens(msg.get("content", ""))
        return total

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
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
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
            return summary_response["message"]["content"]
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
        if not self.conversation_history:
            return []

        system_prompt = {
            "role": "system",
            "content": "You are a voice assistant. Your responses are spoken aloud "
            "through text-to-speech.",
        }

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

    def chat_stream(self, prompt: str):
        """
        Send prompt to LLM and stream response.

        Args:
            prompt: User prompt text

        Yields:
            Chunks of response text
        """
        # Get managed conversation context
        messages = self.manage_context()

        # Add current user message with concise rules reminder
        rules_reminder = (
            "[Reply rules: NO markdown/emojis/abbreviations. "
            "Keep SHORT (2-4 sentences). Invite follow-ups.]"
        )
        messages.append({"role": "user", "content": f"{prompt}\n\n{rules_reminder}"})

        # Stream the response
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

