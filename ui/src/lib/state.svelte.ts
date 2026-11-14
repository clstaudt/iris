/** Central reactive store, fed by Python events over the RPC bridge. */

import { onEvent, rpc, type AppInfo, type TurnTiming } from "./bridge";

export type { TurnTiming };

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

/** A file sent with a typed message, as the conversation shows it. */
export interface AttachmentInfo {
  name: string;
  kind: "image" | "document";
  mime: string;
  /** An image's picture, as a URL the page can show */
  preview?: string;
}

/** A file waiting in the composer, about to be sent. */
export interface DraftFile {
  file: File;
  info: AttachmentInfo;
}

export interface ChatMessage {
  role: "user" | "assistant" | "tool" | "error" | "system";
  text: string;
  streaming: boolean;
  tool?: string;
  /** What a tool did, as the core words it (plugin tools) */
  label?: string;
  /** How a user turn arrived */
  source?: "voice" | "text";
  /** Files sent with a typed turn */
  attachments?: AttachmentInfo[];
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
  attachments: AttachmentInfo[];
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

/** A plugin's tools as the core reports them, switchable in Settings. */
export interface ToolState {
  enabled: boolean;
  available: boolean;
  /** The plugin's name, e.g. "Camera" */
  name?: string;
}

export interface AssistantConfig {
  persona: string;
  persona_id?: string;
  llm_model: string;
  whisper_model: string;
  whisper_device: string;
  whisper_engine?: string;
  /** Speech recognition engine, e.g. "whisper" */
  recognition_engine?: string;
  tts_engine?: SpeechEngine;
  tts_voice: string;
  input_device: string;
  speak_replies?: boolean;
  barge_in?: boolean;
  barge_in_available?: boolean;
  wake_word?: string;
  wake_window?: number;
  sleep_word?: string;
  stt_model_path?: string;
  tts_models_dir?: string;
  /** Whether Smart Turn or a silence ends a turn */
  turn_detection?: TurnDetection;
  /** Seconds of silence that end a turn without Smart Turn */
  silence_duration?: number;
  ollama_host?: string;
  ollama_url?: string;
  mic_on?: boolean;
  paused?: boolean;
  memory?: boolean;
  amnesia?: boolean;
  tools?: Record<string, ToolState>;
  /** Swappable engines per slot (speech recognition, voice, ...), once the
   *  core reports them */
  engines?: Record<string, ModuleInfo[]>;
  /** Plugins the core has loaded, once it reports them */
  plugins?: ModuleInfo[];
}

/** An engine or plugin, as listed in the expert settings. */
export interface ModuleInfo {
  id: string;
  label: string;
  /** The engine in use for its slot, or a plugin switched on */
  active?: boolean;
  /** False when its dependencies are missing */
  available?: boolean;
  /** Where a plugin came from: "built-in", an installed package or the
   *  plugin folder */
  source?: string;
}

/** A named set of choices for how Iris works: the model server and model,
 *  the recognition model, the turn detection and the tools. Who it is (the
 *  persona) is not part of it. */
export interface Setup {
  name: string;
  ollama_host?: string;
  llm_model?: string;
  whisper_model?: string;
  stt_model_path?: string;
  turn_detection?: TurnDetection;
  tools?: Record<string, boolean>;
}

/** The settings a setup holds. */
const SETUP_KEYS = [
  "ollama_host",
  "llm_model",
  "whisper_model",
  "stt_model_path",
  "turn_detection",
  "tools",
] as const;

/** The pages of the settings window. */
export type SettingsPage = "pipeline" | "personas";

/** What ends a turn: Smart Turn hearing the user has finished, or a silence. */
export type TurnDetection = "smart_turn" | "silence";

/** A Whisper size as the recognition stage offers it. */
export interface RecognitionModel {
  id: string;
  size_mb: number | null;
  /** Downloaded already */
  cached: boolean;
  active: boolean;
}

export interface RecognitionModels {
  /** "mlx" or "faster-whisper" */
  engine: string;
  /** A model folder in use instead of a size; "" for none */
  folder: string;
  models: RecognitionModel[];
}

/** Id of a speech engine the core has: "kyutai", "kokoro", or one a
 *  plugin adds. */
export type SpeechEngine = string;

/** A speech engine as the persona editor offers it. */
export interface EngineInfo {
  id: SpeechEngine;
  name: string;
  /** Whether its voices take a speed */
  takes_speed: boolean;
}

/** A persona as edited in the persona editor. */
export interface PersonaSettings {
  /** File name; none for a persona not saved yet */
  id?: string;
  name: string;
  engine: SpeechEngine;
  voice: string;
  speed: number;
  /** "" for none */
  audio_filter: string;
  description: string;
  /** Shipped with Iris: it can be reset once edited, but not deleted */
  shipped?: boolean;
  edited?: boolean;
}

export interface VoiceInfo {
  id: string;
  name: string;
  language: string;
}

export interface VoiceOptions {
  engines: Record<SpeechEngine, VoiceInfo[]>;
  /** The engines in the core's order, with their names */
  engine_info?: EngineInfo[];
  filters: string[];
}

/** User choices remembered across launches. */
interface SavedSettings {
  llm_model?: string;
  persona?: string;
  input_device?: string;
  speak_replies?: boolean;
  /** Speaking interrupts the voice */
  barge_in?: boolean;
  tools?: Record<string, boolean>;
  ollama_host?: string;
  amnesia?: boolean;
  /** "" means the wake word is off */
  wake_word?: string;
  wake_window?: number;
  sleep_word?: string;
  /** Whisper size chosen in the pipeline; none for the best this computer runs */
  whisper_model?: string;
  turn_detection?: TurnDetection;
  /** Model servers kept to switch between, by URL; the default one is not
   *  listed */
  servers?: string[];
  /** The model last chosen on each server, by URL */
  server_models?: Record<string, string>;
  /** Named setups by id, and the one in use */
  setups?: Record<string, Setup>;
  active_setup?: string;
  /** Model folders chosen by the user; empty means download as usual */
  stt_model_path?: string;
  tts_models_dir?: string;
  /** The first-launch setup has been completed */
  setup_done?: boolean;
}

/** What the first-launch setup offers to download. */
export interface SetupInfo {
  stt: { engine: string; model: string; cached: boolean; size_mb: number | null };
  tts: { found: string; size_mb: number };
  ollama_url: string;
}

const SETTINGS_KEY = "iris.settings";
/** How long a voice preview may take to start (an engine's first load), in ms */
const PREVIEW_TIMEOUT = 60_000;
/** Turn timings kept, as many as the core keeps */
const TIMINGS_KEPT = 20;

/** The file's contents as base64, for the trip to the core. */
function readBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",", 2)[1] ?? "");
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

/** Free the composer's local picture URLs once nothing shows them. */
export function releasePreviews(attachments: AttachmentInfo[]) {
  for (const a of attachments) {
    if (a.preview?.startsWith("blob:")) URL.revokeObjectURL(a.preview);
  }
}

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
  panel = $state<"memory" | null>(null);
  /** The settings window's page, while it is open */
  settingsPage = $state<SettingsPage | null>(null);
  /** The pipeline stage open in the inspector */
  stage = $state<string | null>(null);
  /** The voice being previewed, from the click until it has been heard */
  previewing = $state<string | null>(null);
  private previewHeard = false;
  /** Whisper's sizes for the recognition stage, once asked for */
  recognitionModels = $state<RecognitionModels | null>(null);
  /** Model servers kept to switch between, by URL (the default one, "", is
   *  always there and not listed) */
  servers = $state<string[]>(loadSettings().servers ?? []);
  /** Whether each server answered when last checked, by URL */
  reachable = $state<Record<string, boolean>>({});
  /** The latest turns' timings, newest last */
  timings = $state<TurnTiming[]>([]);
  /** Named setups by id; the active one follows every change made */
  setups = $state<Record<string, Setup>>(loadSettings().setups ?? {});
  activeSetup = $state(loadSettings().active_setup ?? "");
  /** Presence: only the orb shows, in a round window of its own, for voice
   *  and vision */
  presence = $state(window.iris?.presence ?? false);
  /** The persona open in the persona editor; "" for a new one */
  editingPersona = $state(loadSettings().persona ?? "iris");
  /** Each speech engine's voices, for the persona editor */
  voices = $state<VoiceOptions | null>(null);
  memoryTopics = $state<MemoryTopic[]>([]);
  /** First launch: the setup screen is shown and nothing starts until done.
   *  Shown from the first frame, while the core is still booting. */
  needsSetup = $state(!loadSettings().setup_done);
  setupInfo = $state<SetupInfo | null>(null);
  /** Setup is done and Iris is fetching its models; the setup screen stays
   *  until they are in */
  installing = $state(false);
  /** How far the setup flower had opened, so the orb's bloom carries on */
  bloomFrom = 0;
  /** A model download at first start, shown under the orb */
  setup = $state<{ model: string; label: string; done: number; total: number } | null>(null);
  /** When Iris falls asleep (performance.now() ms), or null when it isn't awake */
  awakeUntil = $state<number | null>(null);

  /** Mic input level, driving the meter and spinner while listening. */
  level = $state(0);
  /** TTS playback level, driving the meter while the assistant speaks. */
  outputLevel = $state(0);

  private streamingIndex: number | null = null;
  /** Turns from before this launch shown at the top so far */
  private restoredCount = 0;
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

  /** What needs the user while only the orb shows: the latest message, when
   *  it is an error nothing has followed */
  get presenceError(): string | null {
    const last = this.messages[this.messages.length - 1];
    return last?.role === "error" ? last.text : null;
  }

  handleEvent(type: string, data: Record<string, unknown>) {
    switch (type) {
      case "setup.progress":
        this.setup = data.label
          ? {
              model: data.model as string,
              label: data.label as string,
              done: data.done as number,
              total: data.total as number,
            }
          : null;
        if (!data.label) this.leaveSetup();
        break;

      case "config": {
        if (this.recognitionModels) this.loadRecognitionModels().catch(() => {});
        const previous = this.config?.llm_model;
        const previousServer = this.config?.ollama_url;
        this.config = data as unknown as AssistantConfig;
        this.persona = this.config.persona_id ?? this.config.persona.toLowerCase();
        this.speakReplies = this.config.speak_replies ?? true;
        if (previous && previous !== this.config.llm_model) {
          this.messages = this.messages.filter((m) => m.kind !== "model_missing");
        }
        if (previousServer && previousServer !== this.config.ollama_url) {
          this.messages = this.messages.filter((m) => m.kind !== "ollama_unreachable");
        }
        this.keepServer(this.config.ollama_host ?? "");
        this.restoreMic();
        break;
      }

      case "state":
        this.state = (data.state as AssistantState) ?? "idle";
        this.running = this.state !== "idle";
        // Started, or failed before its models were fetched
        if (this.state !== "starting") this.leaveSetup();
        if (this.previewHeard && this.state !== "speaking") this.previewing = null;
        break;

      case "turn.timing":
        this.timings = [...this.timings, data as unknown as TurnTiming].slice(-TIMINGS_KEPT);
        break;

      case "audio.level":
        this.level = (data.rms as number) ?? 0;
        break;

      case "audio.output":
        this.outputLevel = (data.rms as number) ?? 0;
        if (this.previewing && this.outputLevel > 0) this.previewHeard = true;
        break;

      case "chat.user": {
        const message: ChatMessage = {
          role: "user",
          text: (data.text as string) ?? "",
          source: (data.source as "voice" | "text") ?? "voice",
          attachments: data.attachments as AttachmentInfo[] | undefined,
          streaming: false,
        };
        if (data.restored) {
          this.restore(message);
          break;
        }
        const id = data.id as string | undefined;
        if (id) {
          this.startedTurns.add(id);
          const waiting = this.pending.find((p) => p.id === id);
          if (waiting) releasePreviews(waiting.attachments);
          this.pending = this.pending.filter((p) => p.id !== id);
        }
        this.streamingIndex = null;
        this.messages.push(message);
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

      case "chat.continued": {
        // The user went on with the last spoken turn: it and the reply begun
        // to it give way to the whole turn, which follows as chat.user
        let last = this.messages.length - 1;
        while (last >= 0) {
          const message = this.messages[last];
          if (message.role === "user" && message.source === "voice") break;
          last--;
        }
        if (last >= 0) this.messages.splice(last);
        this.streamingIndex = null;
        break;
      }

      case "chat.done": {
        if (data.restored) {
          this.restore({ role: "assistant", text: (data.text as string) ?? "", streaming: false });
          break;
        }
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

      case "core.restarted":
        this.coreRestarted();
        break;

      case "tool.use":
        this.messages.push({
          role: "tool",
          text: (data.detail as string) ?? "",
          tool: (data.tool as string) ?? "tool",
          label: (data.label as string) || undefined,
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

  /** A turn of the conversation before this launch, as memory kept it: it
   *  goes above everything said since, notices included, which stay where
   *  they are seen. */
  private restore(message: ChatMessage) {
    const at = this.restoredCount++;
    this.messages.splice(at, 0, message);
    if (this.streamingIndex !== null && this.streamingIndex >= at) this.streamingIndex++;
  }

  // ------------------------------------------------------------------
  // Commands
  // ------------------------------------------------------------------

  /** Send a typed message with any files. Returns false, and says why in
   *  the conversation, when it could not be sent. */
  async send(text: string, files: DraftFile[] = []): Promise<boolean> {
    const value = text.trim();
    if (!value && files.length === 0) return true;
    let result: { ok: boolean; id?: string; error?: string };
    try {
      const attachments = await Promise.all(
        files.map(async ({ file, info }) => ({
          name: info.name,
          mime: info.mime,
          data: await readBase64(file),
        }))
      );
      result = await rpc("chat.send_text", { text: value, attachments });
    } catch (err) {
      result = { ok: false, error: `The message could not be sent: ${err}` };
    }
    if (!result.ok) {
      this.messages.push({
        role: "error",
        text: result.error ?? "The message could not be sent",
        streaming: false,
      });
      return false;
    }
    const shown = files.map((f) => f.info);
    if (result.id && !this.startedTurns.has(result.id)) {
      this.pending.push({ id: result.id, text: value, attachments: shown });
    } else {
      releasePreviews(shown);
    }
    return true;
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

  /** Open the persona editor on a persona, or on a new one (no id). */
  editPersona(id?: string) {
    this.editingPersona = id ?? "";
    this.settingsPage = "personas";
    this.panel = null;
  }

  async loadPersona(id: string): Promise<PersonaSettings | null> {
    const result = await rpc<PersonaSettings & { ok: boolean }>("persona.get", { id });
    if (!result.ok) return null;
    const { ok: _, ...settings } = result;
    return settings;
  }

  async loadVoices() {
    this.voices = await rpc<VoiceOptions>("persona.voices");
  }

  /** Save a persona; a new one becomes the current persona. Returns an error
   *  message, or null once saved. */
  async savePersona(settings: PersonaSettings): Promise<string | null> {
    const result = await rpc<{
      ok: boolean;
      error?: string;
      id: string;
      personas: string[];
      names: Record<string, string>;
    }>("persona.save", { ...settings });
    if (!result.ok) return result.error ?? "Could not save the persona";
    this.personas = result.personas;
    this.personaNames = result.names;
    if (!settings.id) await this.setPersona(result.id);
    return null;
  }

  /** Delete a persona, or reset an edited one that ships with Iris. */
  async deletePersona(id: string) {
    const result = await rpc<{
      persona: string | null;
      personas: string[];
      names: Record<string, string>;
    }>("persona.delete", { id });
    this.personas = result.personas;
    this.personaNames = result.names;
    if (result.persona && result.persona !== loadSettings().persona) {
      this.persona = result.persona;
      saveSettings({ persona: result.persona });
    }
  }

  /** Say a line in a voice as it is being edited. Returns an error message,
   *  or null once the line is on its way. */
  async previewVoice(settings: PersonaSettings): Promise<string | null> {
    const voice = `${settings.engine}:${settings.voice}`;
    this.previewing = voice;
    this.previewHeard = false;
    // A voice that never comes does not stay marked
    setTimeout(() => {
      if (this.previewing === voice && !this.previewHeard) this.previewing = null;
    }, PREVIEW_TIMEOUT);
    let error: string | null = null;
    try {
      const result = await rpc<{ ok: boolean; error?: string }>("persona.preview", { ...settings });
      if (!result.ok) error = result.error ?? "The voice could not be played";
    } catch (err) {
      error = `The voice could not be played: ${err}`;
    }
    if (error) this.previewing = null;
    return error;
  }

  /** Show Iris as the orb alone, or bring the conversation back. Presence is
   *  a window of its own; this one waits hidden, as it was left. */
  async setPresence(on: boolean) {
    if (on === this.presence || (on && this.needsSetup)) return;
    this.presence = on;
    await window.iris.setPresence(on);
  }

  /** Open the settings window on a page, or close it (null). */
  openSettings(page: SettingsPage | null) {
    this.settingsPage = page;
    if (page) {
      this.panel = null;
      this.ensureSetup();
    }
  }

  // ------------------------------------------------------------------
  // Setups
  // ------------------------------------------------------------------

  /** The setup's settings as they are now. */
  private currentSetup(name: string): Setup {
    const saved = loadSettings();
    const tools = Object.fromEntries(
      Object.entries(this.config?.tools ?? {}).map(([id, state]) => [id, !!state.enabled])
    );
    return {
      name,
      ollama_host: saved.ollama_host ?? this.config?.ollama_host ?? "",
      llm_model: saved.llm_model ?? this.config?.llm_model,
      whisper_model: saved.whisper_model ?? this.config?.whisper_model,
      stt_model_path: saved.stt_model_path ?? this.config?.stt_model_path ?? "",
      turn_detection: saved.turn_detection ?? this.config?.turn_detection,
      tools: { ...tools, ...saved.tools },
    };
  }

  private saveSetups() {
    saveSettings({ setups: $state.snapshot(this.setups), active_setup: this.activeSetup });
  }

  /** There is always a setup to follow the choices made. */
  private ensureSetup() {
    if (this.setups[this.activeSetup]) return;
    const first = Object.keys(this.setups)[0];
    if (first) {
      this.activeSetup = first;
    } else {
      this.setups = { default: this.currentSetup("My setup") };
      this.activeSetup = "default";
    }
    this.saveSetups();
  }

  /** Remember a choice, and keep it in the active setup. */
  private remember(patch: SavedSettings) {
    saveSettings(patch);
    const setup = this.setups[this.activeSetup];
    if (!setup) return;
    const choices: Partial<Setup> = {};
    for (const key of SETUP_KEYS) {
      if (patch[key] !== undefined) Object.assign(choices, { [key]: patch[key] });
    }
    Object.assign(setup, choices);
    this.saveSetups();
  }

  /** Keep the current choices as a new setup, and make it the active one. */
  newSetup(name: string) {
    const id = `setup-${Date.now().toString(36)}`;
    this.setups[id] = this.currentSetup(name.trim() || "New setup");
    this.activeSetup = id;
    this.saveSetups();
  }

  renameSetup(name: string) {
    const setup = this.setups[this.activeSetup];
    if (!setup || !name.trim()) return;
    setup.name = name.trim();
    this.saveSetups();
  }

  /** Delete the active setup and switch to another; the last one stays. */
  async deleteSetup() {
    const others = Object.keys(this.setups).filter((id) => id !== this.activeSetup);
    if (others.length === 0) return;
    delete this.setups[this.activeSetup];
    await this.useSetup(others[0]);
  }

  /** Switch to a setup: its choices are applied one after the other. */
  async useSetup(id: string) {
    // As the setup holds them now: each choice applied is remembered in the
    // setup in use, which this one is from here on
    const target = $state.snapshot(this.setups[id]);
    if (!target) return;
    this.activeSetup = id;
    this.saveSetups();
    const config = this.config;
    let onItsServer = true;
    if ((target.ollama_host ?? "") !== (config?.ollama_host ?? "")) {
      // With the setup's model, not the one last used on that server
      const error = await this.setOllamaHost(target.ollama_host ?? "", target.llm_model);
      if (error) {
        // The server in use stays, and so does its model: the setup's model
        // is one of the other server's
        onItsServer = false;
        this.messages.push({ role: "error", text: error, streaming: false });
      }
    }
    if (onItsServer && target.llm_model && target.llm_model !== this.config?.llm_model) {
      await this.setModel(target.llm_model);
    }
    if (target.stt_model_path && target.stt_model_path !== config?.stt_model_path) {
      await this.setModelPaths({ stt_path: target.stt_model_path });
    } else if (
      !target.stt_model_path &&
      target.whisper_model &&
      (target.whisper_model !== config?.whisper_model || config?.stt_model_path)
    ) {
      await this.setRecognitionModel(target.whisper_model);
    }
    if (target.turn_detection && target.turn_detection !== config?.turn_detection) {
      await this.setTurnDetection(target.turn_detection);
    }
    for (const [name, enabled] of Object.entries(target.tools ?? {})) {
      const state = this.config?.tools?.[name];
      if (state && state.available && !!state.enabled !== enabled) await this.setTool(name, enabled);
    }
  }

  toggleSettings() {
    this.openSettings(this.settingsPage ? null : "pipeline");
  }

  togglePanel(name: "memory") {
    this.panel = this.panel === name ? null : name;
    if (this.panel) this.settingsPage = null;
    if (this.panel === "memory") this.loadMemory();
  }

  async loadRecognitionModels() {
    this.recognitionModels = await rpc<RecognitionModels>("stt.models");
  }

  /** Recognize speech with another Whisper size; it is downloaded if needed
   *  and switched to once loaded. Returns an error message, or null. */
  async setRecognitionModel(model: string): Promise<string | null> {
    const result = await rpc<{ ok: boolean; error?: string }>("stt.set", { model });
    if (!result.ok) return result.error ?? "Could not switch the model";
    this.remember({ whisper_model: model, stt_model_path: "" });
    return null;
  }

  async setTurnDetection(mode: TurnDetection) {
    this.remember({ turn_detection: mode });
    await rpc("turn.set", { mode });
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

  /** Point Iris at model folders it should use as they are ("" = download).
   *  Returns an error message, or null when the folder was accepted. */
  async setModelPaths(paths: { stt_path?: string; tts_dir?: string }): Promise<string | null> {
    const method = this.needsSetup ? "models.check" : "models.set";
    const result = await rpc<{ ok: boolean; error?: string }>(method, paths);
    if (!result.ok) return result.error ?? "Not a model folder";
    if (!this.needsSetup) {
      this.remember({
        ...(paths.stt_path !== undefined ? { stt_model_path: paths.stt_path } : {}),
        ...(paths.tts_dir !== undefined ? { tts_models_dir: paths.tts_dir } : {}),
      });
    }
    return null;
  }

  async setBargeIn(enabled: boolean) {
    saveSettings({ barge_in: enabled });
    await rpc("voice.set_barge_in", { enabled });
  }

  async setTool(name: string, enabled: boolean) {
    this.remember({ tools: { ...loadSettings().tools, [name]: enabled } });
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
   *  model list is refreshed, since it differs per server, and a model of
   *  that server is chosen: this one if it has it. */
  async setOllamaHost(url: string, model?: string): Promise<string | null> {
    const result = await rpc<{ ok: boolean; error?: string; host?: string }>(
      "llm.set_host",
      { url }
    );
    if (!result.ok) return result.error ?? "Could not connect";
    // Kept under the address the core settled on, which is the one it reports
    const host = result.host ?? "";
    if (url.trim() !== host) {
      this.servers = this.servers.filter((server) => server !== url.trim());
    }
    this.keepServer(host);
    this.reachable[host] = true;
    this.remember({ ollama_host: host });
    this.models = [];
    const llms = await rpc<{ models: { name: string; tools: boolean; vision: boolean }[] }>(
      "llm.list"
    );
    this.models = llms.models;
    await this.modelForServer(host, model);
    return null;
  }

  /** The server in use is always among those kept, so it can be switched
   *  back to. */
  private keepServer(host: string) {
    if (!host || this.servers.includes(host)) return;
    this.servers = [...this.servers, host];
    saveSettings({ servers: $state.snapshot(this.servers) });
  }

  /** After switching servers: the model asked for if the server has it,
   *  else the model last used on this one, or, when the model in use is not
   *  on it, the first it has (one that takes tools if there is one). */
  private async modelForServer(host: string, wanted?: string) {
    const names = this.models.map((m) => m.name);
    if (names.length === 0) return;
    const remembered = loadSettings().server_models?.[host];
    const current = this.config?.llm_model ?? "";
    let model: string | undefined;
    if (wanted && names.includes(wanted)) model = wanted;
    else if (remembered && names.includes(remembered)) model = remembered;
    else if (!names.includes(current)) {
      model = (this.models.find((m) => m.tools) ?? this.models[0]).name;
    }
    if (model && model !== current) await this.setModel(model);
  }

  /** Keep a server to switch to later. */
  addServer(url: string) {
    const server = url.trim().replace(/\/+$/, "");
    if (!server || this.servers.includes(server)) return;
    this.servers = [...this.servers, server];
    saveSettings({ servers: $state.snapshot(this.servers) });
    this.checkServers().catch(() => {});
  }

  /** Forget a server; the one in use stays. */
  removeServer(url: string) {
    if (url === (this.config?.ollama_host ?? "")) return;
    this.servers = this.servers.filter((server) => server !== url);
    saveSettings({ servers: $state.snapshot(this.servers) });
  }

  /** Ask which of the servers answer, the default one included. */
  async checkServers() {
    const urls = [...new Set(["", ...this.servers, this.config?.ollama_host ?? ""])];
    const result = await rpc<{ servers: { url: string; reachable: boolean }[] }>(
      "llm.check_hosts",
      { urls }
    );
    for (const server of result.servers) this.reachable[server.url] = server.reachable;
  }

  async setModel(name: string) {
    const host = this.config?.ollama_host ?? "";
    this.remember({
      llm_model: name,
      server_models: { ...loadSettings().server_models, [host]: name },
    });
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
    let micNotice: ChatMessage | null = null;
    if (micStatus !== "granted") {
      const packaged = await window.iris.isPackaged();
      micNotice = {
        role: "error",
        text: packaged
          ? `Microphone access ${micStatus}. Enable it for Iris under System ` +
            `Settings > Privacy & Security > Microphone, then reopen Iris.`
          : `Microphone access ${micStatus}. Enable it under System Settings > ` +
            `Privacy & Security > Microphone for the app that launched this dev ` +
            `server (Cursor, VS Code, Terminal or iTerm), then relaunch it.`,
        streaming: false,
      };
    }
    // Said below the conversation rebuilt after a reload, where it is seen
    const sayMicNotice = () => {
      if (micNotice) this.messages.push(micNotice);
      micNotice = null;
    };

    try {
      const info = await rpc<AppInfo>("app.info");
      this.personas = info.personas;
      this.personaNames = info.persona_names ?? {};
      this.state = info.state as AssistantState;
      this.running = info.running;
      if (info.config) this.config = info.config;
      this.timings = info.timings ?? [];

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
      sayMicNotice();

      // First launch: nothing starts (or downloads) until the user has said
      // which server, model and speech models to use
      if (info.running) this.needsSetup = false;
      if (this.needsSetup) this.setPresence(false);
      if (this.needsSetup) {
        this.setupInfo = await rpc<SetupInfo>("setup.info");
        return;
      }

      // Start before listing models: listing asks Ollama about each one,
      // which is slow against a remote server
      if (!info.running) await this.startAssistant();
      this.loadLists();
    } catch (err) {
      console.error("[store] init failed:", err);
      this.messages.push({ role: "error", text: `Iris could not be reached: ${err}`, streaming: false });
    }
    sayMicNotice();
  }

  /** The core stopped and was started again (the error saying so came
   *  before): it holds nothing of the conversation, which stays shown, and
   *  nothing runs in it yet. Messages waiting in the old one are gone. */
  private async coreRestarted() {
    for (const waiting of this.pending) releasePreviews(waiting.attachments);
    this.pending = [];
    if (this.streamingIndex !== null && this.messages[this.streamingIndex]) {
      this.messages[this.streamingIndex].streaming = false;
    }
    this.streamingIndex = null;
    // The microphone chosen is chosen again in the new core
    this.micRestored = false;
    try {
      if (this.needsSetup && !this.installing) {
        this.setupInfo = await rpc<SetupInfo>("setup.info");
        return;
      }
      const info = await rpc<AppInfo>("app.info");
      if (!info.running) await this.startAssistant(false);
    } catch (err) {
      this.messages.push({ role: "error", text: `Iris could not be started again: ${err}`, streaming: false });
    }
  }

  /** Save the first-launch choices and start Iris with them. The setup
   *  screen stays while the speech models download. */
  async finishSetup(choices: SavedSettings) {
    saveSettings({ ...choices, setup_done: true });
    this.installing = true;
    await this.startAssistant();
    this.loadLists();
  }

  private leaveSetup() {
    if (!this.installing) return;
    this.installing = false;
    this.needsSetup = false;
  }

  /** Start the assistant with the saved settings; restore shows the recent
   *  part of the conversation the core remembers (not when it is shown
   *  already). */
  private async startAssistant(restore = true) {
    const saved = loadSettings();
    const options: Record<string, unknown> = {};
    if (!restore) options.restore_thread = false;
    if (saved.llm_model) options.llm_model = saved.llm_model;
    if (saved.persona) options.persona = saved.persona;
    if (saved.speak_replies !== undefined) options.speak_replies = saved.speak_replies;
    if (saved.barge_in !== undefined) options.barge_in = saved.barge_in;
    if (saved.tools) options.tools = saved.tools;
    if (saved.ollama_host) options.ollama_host = saved.ollama_host;
    if (saved.amnesia) options.amnesia = true;
    if (saved.wake_word) options.wake_word = saved.wake_word;
    if (saved.wake_window) options.wake_window = saved.wake_window;
    if (saved.sleep_word !== undefined) options.sleep_word = saved.sleep_word;
    if (saved.stt_model_path) options.stt_model_path = saved.stt_model_path;
    else if (saved.whisper_model) options.whisper_model = saved.whisper_model;
    if (saved.turn_detection) options.turn_detection = saved.turn_detection;
    if (saved.tts_models_dir) options.tts_models_dir = saved.tts_models_dir;
    this.speakReplies = saved.speak_replies ?? true;
    await rpc("assistant.start", options);
  }

  private loadLists() {
    try {
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
