"""The core as the app's end-to-end test (ui/e2e) runs it: iris.rpc_server,
with a virtual room for its audio in place of the computer's microphone and
speaker.

The room holds nothing but Iris's own voice, which comes back into its
microphone as from a laptop's speakers. What goes on in the real room (a video
playing, people talking) never becomes a turn, the test still hears the voice
come back as the app would, and nothing sounds on the speakers.

The test looks into the core through methods of its own:

- e2e.file_memory: files the conversation into topics now, as Iris does once
  it has been quiet for a while, so the test need not wait for that;
- e2e.memory: what the core remembers (the turns and topics stored, the words
  in their search index) and what the next prompt would carry of the
  conversation.

    python -m irisdev.e2e_core
"""

import os
import sqlite3

from iris import audio, rpc_server
from iris.controller import AssistantController
from iris.memory import DEFAULT_PATH
from irisdev.voicetest.acoustics import Space


class Controller(AssistantController):
    """Iris's own controller, kept where the test's methods find it."""

    current: "Controller | None" = None

    def __init__(self, bus):
        super().__init__(bus)
        Controller.current = self


def _assistant():
    assistant = Controller.current.assistant if Controller.current else None
    if assistant is None or assistant.memory is None:
        raise RuntimeError("Iris is not running with memory")
    return assistant


def file_memory(params: dict) -> dict:
    """File what was said into topics, on the conversation's model, as
    Iris does after a quiet spell, and say which topics there are then. The
    test asks while the conversation rests, so filing does not wait for it."""
    memory = _assistant().memory
    with memory._filing:
        changed = memory.filer.file()
    if changed and memory.on_change:
        memory.on_change()
    return {"topics": [topic.title for topic in memory.store.topics()]}


def _words_indexed() -> int:
    """How many words the search index of the memory file holds, read as
    another program would read the file."""
    db = sqlite3.connect(os.environ.get("IRIS_MEMORY_DB") or DEFAULT_PATH)
    try:
        db.execute("CREATE VIRTUAL TABLE temp.words USING fts5vocab(main, turns_fts, row)")
        return db.execute("SELECT COUNT(*) FROM temp.words").fetchone()[0]
    finally:
        db.close()


def remembered(params: dict) -> dict:
    """What the core remembers, and what the next prompt would carry: the
    messages of the conversation after the system prompt, and the whole
    prompt as text (the system prompt holds the topics and those recalled,
    the files shared go with the next message)."""
    assistant = _assistant()
    store = assistant.memory.store
    conversation = assistant.conversation
    messages = conversation.manage_context()
    prompt = "\n\n".join(m["content"] for m in messages) + conversation._files_context([])
    return {
        "turns": store.turn_count(),
        "topics": len(store.topics()),
        "words": _words_indexed(),
        "context": messages[1:],
        "prompt": prompt,
    }


def main():
    # The microphone and the speaker are taken from here when the assistant
    # is built, after the app has asked for it
    audio.SYSTEM = Space().endpoint("Iris")
    rpc_server.AssistantController = Controller
    rpc_server.main({"e2e.file_memory": file_memory, "e2e.memory": remembered})


if __name__ == "__main__":
    main()
