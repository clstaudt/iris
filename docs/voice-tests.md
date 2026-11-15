# Testing the voice

`irisdev.voicetest` tests Iris by talking to it. Iris runs as in the app, with its real recognition, Smart Turn, echo canceller, language model and voice, but its microphone and speaker belong to a virtual acoustic space: what the microphone hears is scripted, the speaker records what it plays, and each speaker comes back into its own microphone as an echo and is heard by the others. Nothing is played on the computer's speakers, and nothing in the room reaches the test. Iris keeps its personas, memory and plugins in a folder of its own, so yours are left alone.

```bash
# Scripted conversations: a synthesized user talks to Iris, turn by turn
uv run python -m irisdev.voicetest conversations
uv run python -m irisdev.voicetest conversations --group search
uv run python -m irisdev.voicetest conversations --group hearing --only babble --noise fan:15
uv run python -m irisdev.voicetest conversations --list
uv run python -m irisdev.voicetest --llm-host <server URL> --llm-model <model> conversations --only "follow-up"

# Self-talk: two Irises (Iris and Test Kokoro) talk to each other, unscripted
uv run python -m irisdev.voicetest --llm-host <server URL> --llm-model <model> selftalk --minutes 5

# The same conversations as pytest tests
uv run pytest -m voice            # the default groups
uv run pytest -m voice_slow       # Whisper's sizes, a session of 60 turns
IRIS_VOICETEST_WEB=real uv run pytest -m voice_network   # web search over the real network
```

- **Conversations** are held in turns and grouped by what they test: `basic` (a question and a follow-up, a pause in the middle of a sentence, interrupting a long answer, Stop, the wake word...), `search` (web search: the lead-in heard while the search runs, Stop and barge-in at each phase of a search, talking while Iris searches, corrections and follow-ups, searches that fail or hang, web search switched on and off, reading a page), `turn-taking`, `interruptions`, `hearing` (noise, distance, loud speakers, other voices, other languages), `conversation` (typed and spoken turns, waking and sleeping, switching persona), `performance` (a slow model, a context that fills up) and `model` (a model server in trouble: one that stops answering mid-reply, a model that writes its tool calls into the reply as text, a model without tool support). Each turn is checked (heard as said, as one turn, answered aloud, the tools it should use, a reply that fits) and timed from the end of the user's speech: to the first word heard, without the model's time (the voice pipeline), to silence when interrupted or stopped.
- A turn is said at a moment (`script.At`): once Iris has finished, over its reply, at the reply's last syllable, in a pause between its sentences, while it searches, after an event. Actions (Stop, pause, the microphone, a persona or a tool switched) and asides (a back-channel, a cough, a door) happen at moments of their own. Iris runs without long-term memory unless a conversation asks for it, as the app runs it; between turns, Iris can be quit and started again (`script.Restart`), and the conversation it shows again is checked.
- Without a server a scripted model answers (`StubModel`, Ollama's API on a local port), which tests the voice without a model's time. Its rules make it call tools as a model with tools would: a search, reading a page, with something said first. Searches go to a scripted web (`FakeWeb`) unless `--web real` (or `IRIS_VOICETEST_WEB=real`) is given. Facts that come from the scripted model or web are checked in replies; other replies are checked only with a real model.
- Checks known to fail today are listed with each conversation and reported as expected failures (pytest xfail); one that passes is reported as such.
- **Self-talk** reports how much of each reply the other heard, whether it took a turn in the pauses between sentences, how long each took to answer, how long both spoke at once, and whether a voice cut itself off.
- Each run writes `report.json` (with medians, 90th percentiles and worst cases per stage, by tag) and, per conversation, what each microphone heard and each speaker played, as WAV files to listen to.
- Budgets (`voice_pipeline`, `barge_in`, `stop`, `turn_end`, `search_added`, `new_language`) are set with `--budgets voice_pipeline=3.5,barge_in=1` or `IRIS_VOICETEST_BUDGETS`.
- The parts are reusable: `Space` and its endpoints are audio devices as Iris takes them (`iris.audio.SoundDevice`), `ScriptedConversations` and `SelfTalk` talk to any `AssistantUnderTest`, and `IrisUnderTest` is Iris's.
