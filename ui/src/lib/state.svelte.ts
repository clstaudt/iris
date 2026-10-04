/** Central reactive store, fed by Python events over the RPC bridge. */

import { onEvent, rpc, type AppInfo } from "./bridge";

export type AssistantState =
  | "idle"
  | "starting"
  | "listening"
  | "understanding"
  | "thinking"
  | "speaking"
  | "looking"
  | "capturing_screen"
  | "searching"
  | "browsing"
  | "summarizing"
  | "paused"
  | "muted"
  | "typing"
  | "dormant";

export interface ChatMessage {
  role: "user" | "assistant" | "tool" | "error" | "system";
  text: string;
  streaming: boolean;
  tool?: string;
  /** How a user turn arrived */
  source?: "voice" | "text";
  /** Error category, so a resolved warning can be retracted */
  kind?: string;
}

/** A topic in long-term memory, as listed in the memory panel. */
export interface MemoryTopic {
  id: number;
  title: string;
  gist: string;
  updated: number;
  turns: number;
}

export interface MemoryTurn {
  id: number;
  ts: number;
  role: "user" | "assistant" | "system";
  text: string;
  source: "voice" | "text" | null;
}

export interface MemoryTopicDetail extends Omit<MemoryTopic, "turns"> {
  summary: string;
  turns: MemoryTurn[];
}

/** A typed message waiting for the current turn to finish. */
export interface PendingTurn {
  id: string;
  text: string;
}

/** Orb colors: muted tones, so state reads from the hue without glaring. */
export const STATE_COLORS: Record<string, string> = {
  idle: "#d6d7dd",
  starting: "#868b98",
  listening: "#e07c88",
  understanding: "#a996e6",
  thinking: "#76c1d4",
  speaking: "#e3b072",
  looking: "#d295d4",
  capturing_screen: "#d295d4",
  searching: "#82b6d6",
  browsing: "#82b6d6",
  summarizing: "#90a6e4",
  paused: "#62666e",
  muted: "#8d919b",
  typing: "#a9b1c9",
  dormant: "#5f6377",
};

export const STATE_LABELS: Record<string, string> = {
  idle: "Ready",
  starting: "Loading models",
  listening: "Listening",
  understanding: "Transcribing",
  thinking: "Thinking",
  speaking: "Speaking",
  looking: "Looking through the camera",
  capturing_screen: "Looking at your screen",
  searching: "Searching the web",
  browsing: "Reading a web page",
  summarizing: "Summarizing",
  paused: "Paused",
  muted: "Microphone off",
  typing: "Microphone waits while you type",
  dormant: "Sleeping",
};

/** States in which a reply is underway and can be stopped. */
export const BUSY_STATES = new Set([
  "understanding",
  "thinking",
  "speaking",
  "looking",
  "capturing_screen",
  "searching",
  "browsing",
  "summarizing",
]);

/** A switchable tool group as the core reports it. */
export interface ToolState {
  enabled: boolean;
  available: boolean;
}

export interface AssistantConfig {
  persona: string;
  llm_model: string;
  whisper_model: string;
  whisper_device: string;
  whisper_engine?: string;
  tts_voice: string;
  input_device: string;
  speak_replies?: boolean;
  wake_word?: string;
  wake_window?: number;
  sleep_word?: string;
  ollama_host?: string;
  ollama_url?: string;
  mic_on?: boolean;
  paused?: boolean;
  memory?: boolean;
  amnesia?: boolean;
  tools?: Record<string, ToolState>;
}

/** User choices remembered across launches. */
interface SavedSettings {
  llm_model?: string;
  persona?: string;
  input_device?: string;
  speak_replies?: boolean;
  tools?: Record<string, boolean>;
  ollama_host?: string;
  amnesia?: boolean;
  /** "" means the wake word is off */
  wake_word?: string;
  wake_window?: number;
  sleep_word?: string;
}

const SETTINGS_KEY = "iris.settings";

function loadSettings(): SavedSettings {
  try {
    return JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? "{}") as SavedSettings;
  } catch {
    return {};
  }
}

function saveSettings(patch: SavedSettings) {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify({ ...loadSettings(), ...patch }));
  } catch {
    // Storage unavailable: settings simply aren't remembered
  }
}

class IrisStore {
  state = $state<AssistantState>("starting");
  running = $state(false);
  personas = $state<string[]>([]);
  /** Display names keyed by persona id */
  personaNames = $state<Record<string, string>>({});
  persona = $state("iris");
  config = $state<AssistantConfig | null>(null);
  inputDevices = $state<{ index: number; name: string }[]>([]);
  models = $state<{ name: string; tools: boolean; vision: boolean }[]>([]);
  messages = $state<ChatMessage[]>([]);
  pending = $state<PendingTurn[]>([]);
  speakReplies = $state(true);
  /** The side panel beside the conversation, if one is open */
  panel = $state<"settings" | "memory" | null>(null);
  memoryTopics = $state<MemoryTopic[]>([]);
  /** When Iris falls asleep (performance.now() ms), or null when it isn't awake */
  awakeUntil = $state<number | null>(null);

  /** Mic input level, driving the meter and spinner while listening. */
  level = $state(0);
  /** TTS playback level, driving the meter while the assistant speaks. */
  outputLevel = $state(0);

  private streamingIndex: number | null = null;
  /** Typed turns that started before their send() call returned. */
  private startedTurns = new Set<string>();
  private holding = false;
  private micRestored = false;

  get busy() {
    return BUSY_STATES.has(this.state);
  }

  get micOn() {
    return this.config?.mic_on ?? true;
  }

  get paused() {
    return this.state === "paused";
  }

  get amnesia() {
    return this.config?.amnesia ?? false;
  }

  handleEvent(type: string, data: Record<string, unknown>) {
    switch (type) {
      case "config": {
        const previous = this.config?.llm_model;
        this.config = data as unknown as AssistantConfig;
        this.persona = this.config.persona.toLowerCase();
        this.speakReplies = this.config.speak_replies ?? true;
        if (previous && previous !== this.config.llm_model) {
          this.messages = this.messages.filter((m) => m.kind !== "model_missing");
        }
        this.restoreMic();
        break;
      }

      case "state":
        this.state = (data.state as AssistantState) ?? "idle";
        this.running = this.state !== "idle";
        break;

      case "audio.level":
        this.level = (data.rms as number) ?? 0;
        break;

      case "audio.output":
        this.outputLevel = (data.rms as number) ?? 0;
        break;

      case "chat.user": {
        const id = data.id as string | undefined;
        if (id) {
          this.startedTurns.add(id);
          this.pending = this.pending.filter((p) => p.id !== id);
        }
        this.streamingIndex = null;
        this.messages.push({
          role: "user",
          text: (data.text as string) ?? "",
          source: (data.source as "voice" | "text") ?? "voice",
          streaming: false,
        });
        break;
      }

      case "chat.chunk": {
        const chunk = (data.text as string) ?? "";
        if (
          this.streamingIndex === null ||
          this.messages[this.streamingIndex]?.role !== "assistant"
        ) {
          this.messages.push({ role: "assistant", text: "", streaming: true });
          this.streamingIndex = this.messages.length - 1;
        }
        this.messages[this.streamingIndex].text += chunk;
        break;
      }

      case "chat.done": {
        if (this.streamingIndex !== null && this.messages[this.streamingIndex]) {
          this.messages[this.streamingIndex].text = (data.text as string) ?? "";
          this.messages[this.streamingIndex].streaming = false;
        } else if ((data.text as string)?.trim()) {
          this.messages.push({
            role: "assistant",
            text: data.text as string,
            streaming: false,
          });
        }
        this.streamingIndex = null;
        break;
      }

      case "wake.window": {
        const remaining = (data.remaining as number) ?? 0;
        this.awakeUntil = remaining > 0 ? performance.now() + remaining * 1000 : null;
        break;
      }

      case "sleep":
        this.awakeUntil = null;
        break;

      case "memory.updated":
        if (this.panel === "memory") this.loadMemory();
        break;

      case "tool.use":
        this.messages.push({
          role: "tool",
          text: (data.detail as string) ?? "",
          tool: (data.tool as string) ?? "tool",
          streaming: false,
        });
        break;

      case "error":
        this.streamingIndex = null;
        this.messages.push({
          role: "error",
          text: (data.message as string) ?? "Unknown error",
          kind: data.kind as string | undefined,
          streaming: false,
        });
        break;
    }
  }

  // ------------------------------------------------------------------
  // Commands
  // ------------------------------------------------------------------

  async send(text: string) {
    const value = text.trim();
    if (!value) return;
    const result = await rpc<{ ok: boolean; id?: string }>("chat.send_text", {
      text: value,
    });
    if (result.ok && result.id && !this.startedTurns.has(result.id)) {
      this.pending.push({ id: result.id, text: value });
    }
  }

  /** Close the mic while a draft is being typed; reopen it when it is gone. */
  async setTyping(on: boolean) {
    if (on === this.holding) return;
    this.holding = on;
    try {
      await rpc("assistant.hold", { on });
    } catch (err) {
      console.error("[store] assistant.hold failed:", err);
    }
  }

  async toggleMic() {
    await rpc("mic.set", { on: !this.micOn });
  }

  /** While on, nothing of the conversation is written to memory. */
  async toggleAmnesia() {
    const on = !this.amnesia;
    saveSettings({ amnesia: on });
    await rpc("memory.set_amnesia", { on });
  }

  /** Pause or resume the whole conversation: listening, reply and speech. */
  async togglePause() {
    await rpc(this.paused ? "assistant.resume" : "assistant.pause");
  }

  async toggleVoice() {
    const enabled = !this.speakReplies;
    this.speakReplies = enabled;
    saveSettings({ speak_replies: enabled });
    await rpc("voice.set_output", { enabled });
  }

  async interrupt() {
    await rpc("assistant.interrupt");
  }

  async setPersona(name: string) {
    this.persona = name;
    saveSettings({ persona: name });
    await rpc("persona.set", { name });
  }

  togglePanel(name: "settings" | "memory") {
    this.panel = this.panel === name ? null : name;
    if (this.panel === "memory") this.loadMemory();
  }

  async loadMemory() {
    const result = await rpc<{ topics: MemoryTopic[] }>("memory.topics");
    this.memoryTopics = result.topics;
  }

  async memoryTopic(id: number): Promise<MemoryTopicDetail | null> {
    const result = await rpc<MemoryTopicDetail & { ok: boolean }>("memory.topic", { id });
    return result.ok ? result : null;
  }

  async forgetTopic(id: number) {
    await rpc("memory.forget", { id });
    await this.loadMemory();
  }

  /** Listen only for this word until addressed; "" listens to everything. */
  async setWakeWord(word: string) {
    saveSettings({ wake_word: word });
    this.awakeUntil = null;
    await rpc("wake.set", { word });
  }

  /** How long Iris stays awake after a turn, and its sleep word. */
  async setWakeOptions(options: { window?: number; sleep_word?: string }) {
    saveSettings({
      ...(options.window !== undefined ? { wake_window: options.window } : {}),
      ...(options.sleep_word !== undefined ? { sleep_word: options.sleep_word } : {}),
    });
    await rpc("wake.options", options);
  }

  async setTool(name: string, enabled: boolean) {
    saveSettings({ tools: { ...loadSettings().tools, [name]: enabled } });
    await rpc("tools.set", { name, enabled });
  }

  /** Re-read personas, so files added to ~/.iris/personas show up. */
  async refreshPersonas() {
    const result = await rpc<{ personas: string[]; names: Record<string, string> }>(
      "persona.list"
    );
    this.personas = result.personas;
    this.personaNames = result.names ?? {};
  }

  /** Point the core at another Ollama server ("" = default); on success the
   *  model list is refreshed, since it differs per server. */
  async setOllamaHost(url: string): Promise<string | null> {
    const result = await rpc<{ ok: boolean; error?: string; host?: string }>(
      "llm.set_host",
      { url }
    );
    if (!result.ok) return result.error ?? "Could not connect";
    saveSettings({ ollama_host: result.host ?? "" });
    this.models = [];
    const llms = await rpc<{ models: { name: string; tools: boolean; vision: boolean }[] }>(
      "llm.list"
    );
    this.models = llms.models;
    return null;
  }

  async setModel(name: string) {
    saveSettings({ llm_model: name });
    await rpc("llm.set", { name });
  }

  async setMic(index: number | null) {
    const device = this.inputDevices.find((d) => d.index === index);
    saveSettings({ input_device: device?.name });
    await rpc("audio.set_input", { index });
  }

  /** Reselect the remembered microphone once both it and the assistant exist. */
  private restoreMic() {
    if (this.micRestored || !this.config || this.inputDevices.length === 0) return;
    this.micRestored = true;
    const saved = loadSettings().input_device;
    if (!saved || saved === this.config.input_device) return;
    const device = this.inputDevices.find((d) => d.name === saved);
    if (device) rpc("audio.set_input", { index: device.index }).catch(() => {});
  }

  async init() {
    onEvent((type, data) => this.handleEvent(type, data));

    const micStatus = await window.iris.micStatus();
    if (micStatus !== "granted") {
      this.messages.push({
        role: "error",
        text:
          `Microphone access ${micStatus}. Enable it under System Settings > ` +
          `Privacy & Security > Microphone for the app that launched this dev ` +
          `server (Cursor, VS Code, Terminal or iTerm), then relaunch it.`,
        streaming: false,
      });
    }

    try {
      const info = await rpc<AppInfo>("app.info");
      this.personas = info.personas;
      this.personaNames = info.persona_names ?? {};
      this.state = info.state as AssistantState;
      this.running = info.running;
      if (info.config) this.config = info.config;

      // After a reload the core still holds the conversation; rebuild it
      if (info.running) {
        try {
          const history = await rpc<{ events: { type: string; data: Record<string, unknown> }[] }>(
            "chat.history"
          );
          for (const event of history.events) this.handleEvent(event.type, event.data);
        } catch (err) {
          console.warn("[store] chat.history unavailable:", err);
        }
      }

      // Start before listing models: listing asks Ollama about each one,
      // which is slow against a remote server
      if (!info.running) {
        const saved = loadSettings();
        const options: Record<string, unknown> = {};
        if (saved.llm_model) options.llm_model = saved.llm_model;
        if (saved.persona) options.persona = saved.persona;
        if (saved.speak_replies !== undefined) options.speak_replies = saved.speak_replies;
        if (saved.tools) options.tools = saved.tools;
        if (saved.ollama_host) options.ollama_host = saved.ollama_host;
        if (saved.amnesia) options.amnesia = true;
        if (saved.wake_word) options.wake_word = saved.wake_word;
        if (saved.wake_window) options.wake_window = saved.wake_window;
        if (saved.sleep_word !== undefined) options.sleep_word = saved.sleep_word;
        this.speakReplies = saved.speak_replies ?? true;
        await rpc("assistant.start", options);
      }

      rpc<{ devices: { index: number; name: string }[] }>("audio.list_inputs").then(
        (inputs) => {
          this.inputDevices = inputs.devices;
          this.restoreMic();
        }
      );
      rpc<{ models: { name: string; tools: boolean; vision: boolean }[] }>(
        "llm.list"
      ).then((llms) => {
        this.models = llms.models;
      });
    } catch (err) {
      console.error("[store] init failed:", err);
    }
  }
}

export const store = new IrisStore();
