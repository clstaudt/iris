# Getting and using Iris

## Getting Iris

Iris needs a language model server in any case. The simplest is [Ollama](https://ollama.com) on the same computer, with a small model:

```bash
ollama pull qwen3:1.7b
```

Install Ollama from [ollama.com/download](https://ollama.com/download) (on macOS also `brew install ollama && brew services start ollama`, on Linux `curl -fsSL https://ollama.com/install.sh | sh`). A server on another computer works too, Ollama or OpenAI-compatible: the walkthrough asks for its address.

### The macOS app

Iris ships as a self-contained app for Apple Silicon Macs (macOS 14 or later), packed into a disk image ([building it](building-the-macos-app.md)). It brings its own Python; the speech models are downloaded at first launch.

1. Have a language model server, as above.
2. Open the disk image and drag **Iris** into **Applications**.
3. Open Iris. Released disk images are signed with a Developer ID and notarized by Apple, so the app opens like any other. A disk image built without a Developer ID is signed ad hoc only: macOS then refuses it the first time ("Apple could not verify..."). Click **Done**, then go to **System Settings > Privacy & Security**, scroll down to the message about Iris, click **Open Anyway**, and confirm. This is needed only once. If macOS instead says the app "is damaged", run `xattr -dr com.apple.quarantine /Applications/Iris.app` in Terminal and open it again.
4. Follow the walkthrough. The speech models download once (about 2 GB) into `~/.cache/huggingface` and `~/.iris/models`, while you go on with the other steps.
5. Allow the microphone when the walkthrough asks for it. The camera and screen recording are asked for the first time Iris uses them.

### From source

On any platform, with Python 3.11 to 3.13, [uv](https://docs.astral.sh/uv/getting-started/installation/), and Node.js with npm:

```bash
git clone https://github.com/clstaudt/iris.git
cd iris
uv sync                 # the Python core
cd ui
npm ci
npm run dev             # starts Vite, Electron and the Python core
```

The first time it opens, Iris walks you through getting ready, a step at a time; nothing starts before the last one, and every step has a default, so Continue all the way works:

1. **Language model.** Conversations need a large language model. The walkthrough looks for Ollama on this computer and for OpenAI-compatible servers at their usual ports here (LM Studio, llama.cpp, vLLM, Jan), shows which answer, and takes any server you type. It offers the server's models, preferring `qwen3:1.7b`, then one that calls tools. Without a server it says what to install.
2. **Speech models.** They download in the background from here on, the voices first (Kyutai's Pocket TTS with the personas' voices, Kokoro), then Smart Turn and Whisper, about 2 GB together; models already downloaded are reused. Each shows its progress in bytes, and a failed one can be tried again. Folders you already have can be used instead.
3. **Microphone.** The walkthrough asks for the microphone (on macOS, the system's prompt), then shows its level and says when it hears your voice.
4. **Voice.** The persona you talk to, Iris unless you made others; its voice can be played as soon as its engine is downloaded.
5. **Wake word and tools**, both optional: whether Iris waits for its name, and whether it may use the camera, the screen and web search.
6. **Ready.** A summary of the choices, each a click away. Iris starts, the iris bud at the top, which opened as the walkthrough came along, blooms into the orb, and Iris greets you in its persona's words. Downloads still running go on under the orb. **Take the tour** starts Iris the same way, then goes through talking, compact mode, personas, voices, roles, threads, memory and the pipeline, a few sentences each at the controls they are about, which can be tried as it goes.

Each step can be skipped or gone back to, and everything can be changed later in Settings, where **Set up** runs it again, and **Tour** the tour.

## Using Iris

The window is a conversation: your messages and Iris's replies appear as bubbles, replies render markdown, and each of your messages is marked as spoken or typed. Below the message box sits the orb, which shows what Iris is doing through color and motion. It swells softly with your voice while listening and with its own voice while speaking, turns into a slowly rotating spinner while transcribing or thinking, and opens a camera aperture while looking through the camera or at the screen.

### Talking with Iris

Iris starts listening as soon as the models are loaded. Speaking and typing share one conversation.

- Iris answers as soon as you have finished, not after a fixed silence. A voice activity detector (Silero) hears when you pause, and [Smart Turn](https://github.com/pipecat-ai/smart-turn), a small model from the Pipecat project, judges from your words and intonation whether you have finished or are only thinking. Should you go on after it took a pause for the end, the answer begun is dropped and your whole turn is answered. Without Smart Turn (offline on the first run) a turn ends after 3.5 seconds of silence.
- Iris starts speaking as soon as the first sentence of its reply is there. While a reply is long in coming (a model that thinks first, a web search), a low, soft ping sounds every few seconds until Iris speaks.
- Speak while Iris is talking to interrupt it: the voice pauses as soon as you start, and stops if you go on. Iris keeps what it got to say, as far as you heard it, and answers what you said. An echo canceller takes Iris's own voice out of what the microphone hears; it first learns the room from a few seconds of the voice, and a single short word said over loud speakers may not get through. Should the voice pause too often with your speakers, switch **Interrupt by voice** off under Settings > Pipeline > Output; the microphone then closes while Iris speaks.
- Typed and spoken messages are answered one at a time, in order. A message sent while Iris is still answering waits as a queued bubble. While you type, the microphone stays closed, and typing anywhere in the window goes to the message box.
- **Pause** (or a click on the orb) holds the whole conversation: speech stops mid-sentence, the reply stops streaming, the microphone closes, and queued messages wait. **Resume** picks up where it stopped. **Mic** switches listening off while still answering typed messages; **Voice** switches spoken replies off for a text-only chat; **Stop** cancels the current reply.
- **Wake word** (Settings): Iris sleeps until addressed by name. Each persona has its own wake word, its name unless it is given another, and switching persona switches the word; whether the wake word is on is one switch for all of them. "Iris, what's the weather?" wakes it and is answered at once; a bare "Iris" plays a soft chime and waits for the request. Follow-ups need no name. Iris stays awake while you talk and, with **Sleep when idle** on, for a while after each reply (30 seconds by default); over the last 10 seconds the orb drifts toward its sleeping tone, then Iris goes back to sleep with a soft falling chime. With it off, Iris stays awake until told to sleep. A persona can also have a **sleep word**, such as "That's all": said on its own ("That's all, Iris"), it always sends Iris to sleep right away, without asking the model, also with the wake word off; Iris then sleeps until called by the persona's wake word, and listens to everything again once woken. Both words are set beside the wake word switch, in the walkthrough and in the persona editor. Iris may also go to sleep by itself once you are clearly done, when you say goodbye or ask it to stop listening, as the model judges: it says its goodbye, and falls asleep once the voice has died away, so its own words cannot wake it (the **Going to sleep** tool, offered only while the wake word is on). While asleep, only speech reaches the smallest Whisper model, and the wake word only counts at the start of an utterance, so "tell Iris I said hi" doesn't wake it.

### Presence mode

For using Iris by voice and vision alone, the shrink button beside Memory and Settings turns Iris into presence mode: only the orb, round and without a window around it, floating above other windows. Clicks beside it go to whatever is underneath. Drag it anywhere; a click on the orb still pauses and resumes. While the pointer is on the orb, a small expand mark shows inside the ring; a click on it, or Escape, brings the conversation back. When something needs your attention, such as a microphone that delivers only silence, the orb turns red and shows the message when you point at it, and a click opens the conversation. Iris remembers the mode and where the orb was.

### Tools and files

The camera, screen capture, web search and going to sleep are on by default and can be switched off under Settings > Pipeline > Tools; a tool whose optional dependencies are missing shows as not installed. Iris decides itself when to use them:

- **Camera:** "What am I holding?", "Read the text on this paper".
- **Screen:** "What's on my screen right now?", "Can you read the code I have open?" On macOS, screen recording must be allowed for Iris (in development, for the app that started it) in **System Settings > Privacy & Security > Screen Recording**.
- **Web search:** current facts, news, and pages it should read.
- **Going to sleep:** "Thanks, that's all for now", "Good night". Iris says goodbye and sleeps until called by its wake word; it is offered only while a wake word is set.

Images, from the camera, the screen or an attachment, go to a model that can see, such as `qwen3-vl:8b`, when the server has one.

Typed messages can carry files: images, PDFs and text files, added with the paperclip beside Send, by pasting, or by dropping them anywhere in the window. A message holds up to 10 files of up to 25 MB each. They stay in the model's context for six turns, so follow-up questions work, and only their names are kept in memory.

### Memory

Iris keeps one continuous conversation that survives restarts and is remembered by topic in `~/.iris/memory.db`. The model always sees the recent turns plus a one-line index of every topic it remembers. When you come back to something, it decides itself to `recall` that topic or to `search_memory`, a full-text search over everything said; there are no embeddings and no similarity thresholds. Conversations are sorted into topics by the same model, but only during quiet spells: filing gives way the moment you speak or type, so replies never wait for it. The **Memory** panel (top right) lists the topics with their summaries and conversations, and can forget a topic. **Amnesia** in the message box keeps the current conversation out of memory: Iris follows it as usual, but writes nothing, and forgets it entirely when amnesia is switched off.

### Personas

A persona shapes your assistant's character and style: a name, a personality, a voice, and the words that wake it and send it to sleep. Iris ships one, **Iris**, friendly and helpful. Under Settings > Personas you edit it or create others: name, personality, wake word and sleep word, and voice, with every voice playable, the speech engine, the effect, and the speed where the engine takes one. Edits are saved to `~/.iris/personas`; an edited Iris can be reset. Persona files can also be written by hand, in `~/.iris/personas` (or `IRIS_PERSONAS_DIR`); a change to the file of the persona in use is taken on while Iris runs, from its next reply:

```yaml
name: "Juno"
engine: "kokoro"       # or "kyutai"
voice: "af_sarah"      # a voice of that engine
speed: 1.1             # for engines whose voices take a speed (Kokoro)
audio_filter: "intercom"
wake_word: "Hey Juno"  # optional; the name when left out
sleep_word: "Goodbye"  # optional; none when left out
description: |
  You are calm, precise and to the point.
```

A persona keeps its voice in every language. The voice decides which languages are spoken: a reply in a language the persona's voice does not speak properly is shown and not spoken, and the conversation says so once; no other voice or engine speaks in its place. Kyutai's Pocket TTS sounds the most natural. It speaks English, French, German, Italian, Spanish, Portuguese and Dutch, each through that language's model with the voice's own prompt, and no Hindi, Japanese or Chinese. All its voices speak these languages but French, which 12 of its 27 voices do not speak properly: they keep so much of their accent that Whisper now and then hears English. Kokoro speaks English, French, Italian, Spanish, Portuguese, Hindi, Japanese and Chinese, but each of its voices only some of them properly: nearly every one English and the language it was made for, most Spanish, Portuguese and Mandarin, many Italian, few Hindi, five French and only the five Japanese voices Japanese; none speaks German or Dutch. Which languages each voice speaks properly was measured: Whisper hears the voice's sentences in the language, about as many of their words as from the voices made for it (`tests/unit/fixtures/voice_languages.json`). The persona editor lists the languages the persona's voice speaks and those it does not, shows each voice's languages, and filters the voices by the languages you need. Each part of a reply is spoken in the language it is written in, told from its words, so an Italian answer to an English question is not read with an English accent. A Kyutai voice can also be the path to a short audio file of a voice to clone.

A Kyutai persona can also speak in a voice cloned in the app: under the voices in the persona editor, Clone a voice records a text read aloud, or takes an audio file (WAV, MP3, M4A, FLAC). Kyutai hands out the model that clones only to a Hugging Face account that accepted its terms on [huggingface.co/kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts); the editor links there and asks for an access token, which Hugging Face's own login keeps. A cloned voice is kept in `~/.iris/voices`, whether or not you save a persona with it, until you delete it under Settings, Voices. It is the same voice in every language Pocket TTS speaks; each language's cloning model is fetched the first time the voice speaks it.

### Settings

Settings (top right) open as a window of their own. All choices are remembered between launches.

- **Pipeline** shows how a turn travels through Iris, as one line from top to bottom: what it hears (microphone, wake word, turn end, recognition), how it thinks (the language model, with the persona, tools and memory it draws on) and how it speaks (the voice and the output). Click a stage to change it in the inspector beside the line: the recognition model, Smart Turn or a silence for the turn end, the model server and model, the tool switches, and so on.
  Beside each timed stage is how long it took in the latest turn (turn end, recognition, the model's first sentence, the voice starting), on one scale, and below the line the time from the end of your speech to Iris's first word. A stage changed since shows no time until the next turn measures it.
- **Personas** holds who Iris is: each persona's name, personality and voice, with every voice playable before you pick it, and the languages each voice speaks.
- **About** shows Iris's version, copyright and license, with the license in full, and the notices: who made the speech models and voices Iris downloads, and their licenses ([NOTICES.md](../NOTICES.md)).

A **setup** (top of the window) is a named set of choices for how Iris works: the model server and model, the recognition model, the turn end and the tools. Changes go into the setup in use; make a new one to keep the current choices under another name, and switch between setups to compare them.

Two stages in more detail:

- **Model server** (Language model): Ollama, or any server with the OpenAI-compatible API (vLLM, llama.cpp, LM Studio, MLX servers such as MTPLX, LiteLLM, OpenAI). Empty uses `OLLAMA_HOST` (or `http://localhost:11434`). Iris asks the server what it is: Ollama, with or without `/v1` in the URL, is used through its own API, which tells each model's context length and what it can do; any other server through its OpenAI-compatible API at `.../v1`, where each model is taken to support tools unless the server refuses them. A server that needs a key gets the one given when it was added, which Iris keeps on this computer alone (in `~/.iris/server-keys.json`, readable only by you) and forgets when the server is removed; a server added without one gets the key in the `IRIS_LLM_API_KEY` environment variable, if set. A server that answers with the one model it has loaded, whatever model is asked for (MTPLX), has that model in use. The server is checked before Iris switches to it.
- **Model folders** (Recognition and Voice): point Iris at speech models you already have instead of downloading them. Recognition takes a Whisper folder, either an MLX conversion (`weights.safetensors` or `weights.npz`) or a faster-whisper model (`model.bin`); Voice takes a folder holding `kokoro-v1.0.onnx` and `voices-v1.0.bin`. A folder without them is refused at once. **Use a download** goes back to the downloaded models.
