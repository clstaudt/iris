/** Central reactive store, fed by Python events over the RPC bridge. */

import { onEvent, rpc, type AppInfo, type TurnTiming } from "./bridge";
import { written } from "./keep";
import { Exposures } from "./shutter";
import type { WalkthroughChoices } from "./walkthrough.svelte";

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
  /** A voice being made ready that the conversation waits on, by job id */
  job?: string;
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

/** A thread: a conversation with one persona, with its own memory. */
export interface ThreadInfo {
  id: number;
  /** The persona's id, Iris for one whose persona is gone */
  persona: string;
  persona_name: string;
  /** Its role's id and name, if it has one */
  role: string | null;
  role_name: string;
  /** Its earliest topic's, once memory has filed one; "" until then */
  title: string;
  created: number;
  /** When it was last talked in (seconds since the epoch) */
  updated: number;
}

/** A role: a focus a thread can be given. */
export interface RoleInfo {
  /** File name; none for a role not saved yet */
  id?: string;
  name: string;
  focus: string;
  /** Shipped with Iris: it can be reset once edited, but not deleted */
  shipped?: boolean;
  edited?: boolean;
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
  /** The word listened for; "" while the wake word is off */
  wake_word?: string;
  /** Seconds Iris stays awake after a turn; 0 when it sleeps only when told */
  wake_window?: number;
  /** The persona's sleep word; "" for none */
  sleep_word?: string;
  /** The persona's wake word, also while the wake word is off */
  persona_wake_word?: string;
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
  /** Memory switched on, for every thread */
  remember?: boolean;
  /** The thread Iris is in; null for a new one, stored with its first turn */
  thread?: number | null;
  /** The thread's role, if it has one, and its name */
  role?: string | null;
  role_name?: string;
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
export type SettingsPage = "pipeline" | "personas" | "voices" | "roles";

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
  /** Its name unless it is given another */
  wake_word?: string;
  /** "" for none */
  sleep_word?: string;
  /** Shipped with Iris: it can be reset once edited, but not deleted */
  shipped?: boolean;
  edited?: boolean;
}

export interface VoiceInfo {
  id: string;
  name: string;
  /** The language it was made for, as shown */
  language: string;
  /** Whisper codes of the languages it speaks properly */
  languages?: string[];
  /** Of those, the ones it is ready to speak now: nothing to download or
   *  make first */
  ready?: string[];
  /** The personas that speak with it */
  personas?: { id: string; name: string }[];
  /** A voice cloned here: where its files are, and the languages made of
   *  its sample so far */
  cloned?: { folder: string; made: string[] };
}

/** Making a voice ready to speak a language: a download, then for a cloned
 *  voice its making, shown wherever it is waited on. */
export interface VoiceJob {
  /** engine|voice|language */
  id: string;
  engine: SpeechEngine;
  voice: string;
  /** "" for a download that serves every language (Kokoro's) */
  language: string;
  name: string;
  personas: string[];
  /** What started it: "conversation", "languages", "voices", "preview", "clone" */
  origin: string;
  state: "queued" | "downloading" | "making" | "ready" | "failed";
  stage: "download" | "make";
  /** What downloads: "model", "voice" or "cloning" */
  what: string;
  done: number;
  total: number;
  /** Nothing has come for a while */
  stalled: boolean;
  error: string;
  /** What goes on, e.g. "Downloading the German voice model" */
  title: string;
  /** What it is for, e.g. "Iris will speak German once it's done" */
  purpose: string;
}

/** A language Iris talks in. */
export interface LanguageInfo {
  /** Whisper code */
  code: string;
  name: string;
}

export interface VoiceOptions {
  engines: Record<SpeechEngine, VoiceInfo[]>;
  /** The engines in the core's order, with their names */
  engine_info?: EngineInfo[];
  /** The languages Iris talks in, in order */
  languages?: LanguageInfo[];
  filters: string[];
  /** Iris's languages, chosen in the app: the voices are made ready for them */
  chosen?: string[];
  /** The jobs making voices ready */
  jobs?: VoiceJob[];
  /** The voice a persona speaks with once its cloned voice is deleted */
  fallback?: string;
}

/** User choices remembered across launches. */
export interface SavedSettings {
  llm_model?: string;
  /** The persona of the thread Iris is in, or of a new one */
  persona?: string;
  /** The thread Iris is in, to open again; null for a new one */
  thread?: number | null;
  /** Whether the threads are shown beside the conversation, and whether
   *  as a narrow strip of rings */
  sidebar?: boolean;
  sidebar_strip?: boolean;
  input_device?: string;
  speak_replies?: boolean;
  /** Speaking interrupts the voice */
  barge_in?: boolean;
  tools?: Record<string, boolean>;
  ollama_host?: string;
  /** Memory switched on, for every thread; on unless switched off */
  remember?: boolean;
  /** Before amnesia was each thread's own: one switch for all, which goes
   *  to the thread Iris starts in */
  amnesia?: boolean;
  /** The wake word switch; the words are each persona's own */
  wake_on?: boolean;
  /** Seconds Iris stays awake after a turn, kept while sleeping when idle
   *  is off */
  wake_window?: number;
  /** Whether Iris falls asleep after wake_window without a turn */
  idle_sleep?: boolean;
  /** Before the words were each persona's: one wake word ("" for off) and
   *  one sleep word for all, which go to the persona in use */
  wake_word?: string;
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
  /** The first-launch walkthrough has been completed */
  setup_done?: boolean;
  /** Iris's languages: every persona's voice is made ready for them ahead.
   *  Conversations are not held to them */
  languages?: string[];
}

const SETTINGS_KEY = "iris.settings";
/** Seconds Iris stays awake after a turn unless another time is chosen */
export const DEFAULT_WAKE_WINDOW = 30;
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

export function loadSettings(): SavedSettings {
  try {
    return JSON.parse(localStorage.getItem(SETTINGS_KEY) ?? "{}") as SavedSettings;
  } catch {
    return {};
  }
}

export function saveSettings(patch: SavedSettings) {
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
  /** Each speech engine's voices, for the persona editor and the Voices page */
  voices = $state<VoiceOptions | null>(null);
  /** Iris's languages, as chosen; English always */
  languages = $state<string[]>(loadSettings().languages ?? ["en"]);
  /** The jobs making voices ready, by id */
  jobs = $state<Record<string, VoiceJob>>({});
  private voicesStale: ReturnType<typeof setTimeout> | null = null;
  memoryTopics = $state<MemoryTopic[]>([]);
  /** The threads, the one talked in last first */
  threads = $state<ThreadInfo[]>([]);
  /** The thread Iris is in; null for a new one */
  openThread = $state<number | null>(null);
  /** The threads are shown beside the conversation */
  sidebar = $state(loadSettings().sidebar ?? false);
  /** Shown, the threads are a narrow strip of their rings */
  sidebarStrip = $state(loadSettings().sidebar_strip ?? false);
  /** A new thread was asked for: its persona and role are chosen until
   *  something is said */
  choosing = $state(false);
  /** A switch of thread asked for while Iris replies, to be confirmed */
  switchAsked = $state<{
    open?: number;
    persona?: string;
    role?: string | null;
    choose?: boolean;
  } | null>(null);
  /** The roles, by name */
  roles = $state<RoleInfo[]>([]);
  /** The role open in the role editor; "" for a new one, null for the
   *  first in the list */
  editingRole = $state<string | null>(null);
  /** Turns the core keeps, filed under a topic or not yet */
  memoryTurns = $state(0);
  /** The walkthrough is shown instead of the conversation: on the first
   *  launch, from the first frame while the core is still booting (nothing
   *  starts until it is done), or run again from Settings */
  needsSetup = $state(!loadSettings().setup_done);
  /** How far the walkthrough's flower had opened, so the orb's bloom carries on */
  bloomFrom = 0;
  /** Iris has been started in this window (not yet on a first launch) */
  private started = false;
  /** A model download at first start, shown under the orb */
  setup = $state<{ model: string; label: string; done: number; total: number } | null>(null);
  /** When Iris falls asleep (performance.now() ms), or null when it isn't awake */
  awakeUntil = $state<number | null>(null);

  /** Mic input level, driving the meter and spinner while listening. */
  level = $state(0);
  /** TTS playback level, driving the meter while the assistant speaks. */
  outputLevel = $state(0);
  /** The looks Iris has begun through the camera or at the screen: every
   *  orb fires its shutter as the count goes up. Counted in the
   *  conversation window, and sent to presence */
  shots = $state(0);
  private exposures = new Exposures();
  /** The memories Iris has written: every orb plays its keep as the count
   *  goes up. Counted in the conversation window, and sent to presence */
  kept = $state(0);

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

  /** The open thread is in amnesia mode: nothing of it is written */
  get amnesia() {
    return this.config?.amnesia ?? false;
  }

  /** Memory is switched on, for every thread */
  get memoryOn() {
    return this.config?.remember ?? true;
  }

  /** Whether there is anything to forget: topics, turns kept but not filed
   *  yet, or the conversation shown */
  get remembersAnything() {
    return (
      this.memoryTopics.length > 0 ||
      this.memoryTurns > 0 ||
      this.pending.length > 0 ||
      this.messages.some((m) => m.role === "user" || m.role === "assistant")
    );
  }

  /** Whether what is remembered is shown: in the memory panel, or in the
   *  settings' Memory stage */
  private get showsMemory() {
    return this.panel === "memory" || (!!this.settingsPage && this.stage === "memory");
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
        break;

      case "config": {
        if (this.recognitionModels) this.loadRecognitionModels().catch(() => {});
        const previous = this.config?.llm_model;
        const previousServer = this.config?.ollama_url;
        this.config = data as unknown as AssistantConfig;
        this.persona = this.config.persona_id ?? this.config.persona.toLowerCase();
        // Iris goes on with this persona, in this thread, when it starts again
        saveSettings({ persona: this.persona });
        if (this.config.thread !== undefined) this.settleThread(this.config.thread);
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

      // A tool's activity shows the moment it starts, also while the state
      // shows Iris speaking
      case "flags":
        if (this.exposures.flags((data.flags as Record<string, boolean>) ?? {})) this.shots++;
        break;

      case "state":
        this.state = (data.state as AssistantState) ?? "idle";
        this.running = this.state !== "idle";
        if (this.exposures.state(this.state)) this.shots++;
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

      case "voice.job":
        this.takeJob(data as unknown as VoiceJob);
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
        // Something was said: the new thread's persona is chosen
        this.choosing = false;
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
        if (written(data, this)) this.kept++;
        if (this.showsMemory) this.loadMemory();
        // A topic filed may give a thread its title
        this.loadThreads();
        break;

      case "threads.changed":
        this.loadThreads();
        break;

      // Iris went on in another thread: its recent turns follow, restored
      case "thread.opened":
        for (const waiting of this.pending) releasePreviews(waiting.attachments);
        this.pending = [];
        this.messages = [];
        this.restoredCount = 0;
        this.streamingIndex = null;
        this.settleThread((data.id as number | null) ?? null);
        if (this.showsMemory) this.loadMemory();
        break;

      case "memory.forgotten":
        this.forgotten();
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

      // Something to know that is not a fault, such as replies shown and
      // not spoken, or the speaker or microphone Iris switched to. It goes
      // above the reply streaming in, where it stays when the transcript is
      // rebuilt (the reply comes whole after it)
      case "notice": {
        const notice: ChatMessage = {
          role: "system",
          text: (data.message as string) ?? "",
          kind: data.kind as string | undefined,
          streaming: false,
        };
        if (this.streamingIndex !== null) {
          this.messages.splice(this.streamingIndex, 0, notice);
          this.streamingIndex++;
        } else {
          this.messages.push(notice);
        }
        break;
      }
    }
  }

  /** Everything was forgotten: the conversation shown goes with it, as do
   *  the messages waiting to be answered, which the core dropped. */
  private forgotten() {
    for (const waiting of this.pending) releasePreviews(waiting.attachments);
    this.pending = [];
    this.messages = [];
    this.streamingIndex = null;
    this.restoredCount = 0;
    this.memoryTopics = [];
    this.memoryTurns = 0;
    // Every thread is gone: Iris goes on in a new one
    this.choosing = false;
    this.settleThread(null);
    this.threads = [];
    if (this.showsMemory) this.loadMemory();
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
    this.choosing = false;
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

  /** While on, nothing of the open thread is written to memory; it stays
   *  with the thread. */
  async toggleAmnesia() {
    await rpc("memory.set_amnesia", { on: !this.amnesia });
  }

  /** Switch memory on or off, for every thread. */
  async setRemember(on: boolean) {
    saveSettings({ remember: on });
    await rpc("memory.set_remember", { on });
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

  // ------------------------------------------------------------------
  // Threads
  // ------------------------------------------------------------------

  async loadThreads() {
    try {
      const result = await rpc<{ threads: ThreadInfo[]; open?: number | null }>("threads.list");
      this.threads = result.threads;
      // Told while Iris runs: a thread is stored with its first turn
      if (result.open !== undefined) this.settleThread(result.open);
    } catch (err) {
      console.error("[store] threads.list failed:", err);
    }
  }

  /** The thread Iris is in, kept to open again when it starts again. */
  private settleThread(id: number | null) {
    this.openThread = id;
    saveSettings({ thread: id });
  }

  /** The open thread, as listed; none for a new one. */
  get thread(): ThreadInfo | null {
    return this.threads.find((t) => t.id === this.openThread) ?? null;
  }

  /** Go on in another thread: its persona and its memory. While Iris
   *  replies, which that stops, it is asked first (switchAsked). */
  async openThreadById(id: number, { confirmed = false } = {}) {
    if (this.settingsPage) this.settingsPage = null;
    if (id === this.openThread) {
      this.choosing = false;
      return;
    }
    if (this.busy && !confirmed) {
      this.switchAsked = { open: id };
      return;
    }
    this.choosing = false;
    await rpc("threads.open", { id });
  }

  /** A new thread with a persona, and a role if one is given: Iris and no
   *  role unless others are chosen. Choosing, they can be picked until
   *  something is said. While Iris replies, which that stops, it is asked
   *  first (switchAsked). */
  async newThread(
    persona = "iris",
    { choose = true, role = null as string | null, confirmed = false } = {}
  ) {
    this.settingsPage = null;
    if (this.busy && !confirmed) {
      this.switchAsked = { persona, role, choose };
      return;
    }
    this.choosing = choose;
    await rpc("threads.new", { persona, ...(role ? { role } : {}) });
  }

  /** Go on with the switch asked for while Iris replied, or not. */
  async answerSwitch(confirmed: boolean) {
    const asked = this.switchAsked;
    this.switchAsked = null;
    if (!asked || !confirmed) return;
    if (asked.open !== undefined) await this.openThreadById(asked.open, { confirmed: true });
    else {
      await this.newThread(asked.persona, { choose: asked.choose, role: asked.role, confirmed: true });
    }
  }

  /** Forget a thread with its memory; the open one leaves a new thread. */
  async forgetThread(id: number) {
    await rpc("threads.forget", { id });
    await this.loadThreads();
  }

  // ------------------------------------------------------------------
  // Roles
  // ------------------------------------------------------------------

  async loadRoles() {
    const result = await rpc<{ roles: RoleInfo[] }>("roles.list");
    this.roles = result.roles;
  }

  /** Open the role editor on a role, or on a new one (no id). */
  editRole(id?: string) {
    this.editingRole = id ?? "";
    this.settingsPage = "roles";
    this.panel = null;
  }

  /** Save a role; its threads have the new focus from their next turn.
   *  Returns an error message, or null once saved. */
  async saveRole(settings: RoleInfo): Promise<string | null> {
    const result = await rpc<{ ok: boolean; error?: string; id: string; roles: RoleInfo[] }>(
      "roles.save",
      { ...settings }
    );
    if (!result.ok) return result.error ?? "Could not save the role";
    this.roles = result.roles;
    this.editingRole = result.id;
    return null;
  }

  /** Delete the user's role, or reset an edited one that ships with Iris. */
  async deleteRole(id: string) {
    const result = await rpc<{ roles: RoleInfo[] }>("roles.delete", { id });
    this.roles = result.roles;
  }

  toggleSidebar() {
    this.sidebar = !this.sidebar;
    saveSettings({ sidebar: this.sidebar });
  }

  /** The threads as a narrow strip of rings, or listed in full. */
  toggleSidebarStrip() {
    this.sidebarStrip = !this.sidebarStrip;
    saveSettings({ sidebar_strip: this.sidebarStrip });
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
    const voices = await rpc<VoiceOptions>("voice.list");
    this.voices = voices;
    for (const job of voices.jobs ?? []) this.takeJob(job, false);
  }

  /** A job making a voice ready, as it changes. The conversation shows the
   *  ones it waits on; once a voice is ready the voices are listed again. */
  private takeJob(job: VoiceJob, live = true) {
    this.jobs = { ...this.jobs, [job.id]: job };
    if (live && job.origin === "conversation") this.showJob(job);
    if (live && (job.state === "ready" || job.state === "failed") && this.voices) {
      if (this.voicesStale) clearTimeout(this.voicesStale);
      this.voicesStale = setTimeout(() => this.loadVoices().catch(() => {}), 300);
    }
  }

  /** A job the conversation waits on, as a line of its own that follows it
   *  in place: its progress, then that it is done (kept as it was said), or
   *  that it failed, to retry there. Asked for again once done, a new line. */
  private showJob(job: VoiceJob) {
    let line: ChatMessage | undefined;
    for (let i = this.messages.length - 1; i >= 0 && !line; i--) {
      const message = this.messages[i];
      if (message.job === job.id && message.kind === "voice.job") line = message;
    }
    if (!line) {
      line = { role: "system", text: job.title, kind: "voice.job", job: job.id, streaming: false };
      this.messages.push(line);
      line = this.messages[this.messages.length - 1];
    }
    if (job.state === "ready") {
      line.kind = "voice.job.done";
      line.text = job.title;
    }
  }

  /** The job making a voice ready for a language, if there is one. */
  jobFor(engine: SpeechEngine, voice: string, language: string): VoiceJob | undefined {
    return this.jobs[`${engine}|${voice}|${language}`] ?? this.jobs[`${engine}|${voice}|`];
  }

  /** The jobs running, the ones the conversation waits on first. */
  get runningJobs(): VoiceJob[] {
    const running = Object.values(this.jobs).filter((j) => j.state !== "ready" && j.state !== "failed");
    return running.sort((a, b) => Number(b.origin === "conversation") - Number(a.origin === "conversation"));
  }

  /** Choose Iris's languages: every persona's voice is made ready for them. */
  async setLanguages(codes: string[]) {
    const chosen = [...new Set(["en", ...codes])];
    this.languages = chosen;
    this.remember({ languages: chosen });
    const result = await rpc<{ languages: string[]; jobs: VoiceJob[] }>("languages.set", {
      languages: chosen,
    });
    for (const job of result.jobs ?? []) this.takeJob(job, false);
    if (this.voices) this.voices.chosen = result.languages;
  }

  /** Make a voice ready for languages (Iris's own without any). */
  async makeReady(engine: SpeechEngine, voice: string, languages?: string[]) {
    const result = await rpc<{ jobs: VoiceJob[] }>("voice.ready", {
      engine,
      voice,
      languages: languages ? [...languages] : undefined,
    });
    for (const job of result.jobs ?? []) this.takeJob(job, false);
  }

  async retryJob(id: string) {
    const result = await rpc<{ job: VoiceJob | null }>("voice.retry", { id });
    if (result.job) this.takeJob(result.job, false);
  }

  /** Rename a cloned voice. Returns an error message, or null once renamed. */
  async renameVoice(voice: string, name: string): Promise<string | null> {
    const result = await rpc<{ ok: boolean; error?: string }>("voice.rename", { voice, name });
    if (!result.ok) return result.error ?? "The voice could not be renamed";
    await this.loadVoices();
    return null;
  }

  /** Delete a cloned voice; reassign gives the personas that speak with it
   *  Kyutai's default voice. Returns an error message, or null once deleted. */
  async deleteVoice(voice: string, reassign = false): Promise<string | null> {
    const result = await rpc<{ ok: boolean; error?: string }>("voice.delete", { voice, reassign });
    await this.loadVoices();
    if (!result.ok) return result.error ?? "The voice could not be deleted";
    return null;
  }

  /** Save a persona; a new one is edited from here on. Returns an error
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
    if (!settings.id) this.editingPersona = result.id;
    // The voices as they are now, with the personas that speak with each
    if (this.voices) await this.loadVoices();
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

  /** Write a persona, as saved, to a file the user chooses, with its cloned
   *  voice. Returns an error message, or null once written (or cancelled). */
  async exportPersona(id: string, name: string): Promise<string | null> {
    const path = await window.iris.savePersonaFile(name);
    if (!path) return null;
    const result = await rpc<{ ok: boolean; error?: string }>("persona.export", { id, path });
    return result.ok ? null : (result.error ?? "The persona could not be exported");
  }

  /** Add a persona from a file the user chooses, with its cloned voice; it
   *  opens in the editor. Returns an error message, or null once added (or
   *  cancelled). */
  async importPersona(): Promise<string | null> {
    const path = await window.iris.openPersonaFile();
    if (!path) return null;
    const result = await rpc<{
      ok: boolean;
      error?: string;
      id: string;
      personas: string[];
      names: Record<string, string>;
    }>("persona.import", { path });
    if (!result.ok) return result.error ?? "The persona could not be imported";
    this.personas = result.personas;
    this.personaNames = result.names;
    this.editingPersona = result.id;
    // Its cloned voice, in the library now
    if (this.voices) await this.loadVoices();
    return null;
  }

  /** Say a line in a voice as it is being edited. Returns an error message,
   *  or null once the line is on its way. */
  async previewVoice(settings: PersonaSettings): Promise<string | null> {
    const voice = `${settings.engine}:${settings.voice}`;
    this.previewing = voice;
    this.previewHeard = false;
    // A voice that never comes does not stay marked, unless it downloads first
    setTimeout(() => {
      const downloading = this.runningJobs.some((j) => `${j.engine}:${j.voice}` === voice);
      if (this.previewing === voice && !this.previewHeard && !downloading) this.previewing = null;
    }, PREVIEW_TIMEOUT);
    let error: string | null = null;
    try {
      const result = await rpc<{ ok: boolean; error?: string; job?: VoiceJob }>(
        "persona.preview",
        { ...settings }
      );
      if (!result.ok) error = result.error ?? "The voice could not be played";
      // The voice downloads first, shown on its tile, and plays once it is here
      if (result.job) this.waitForPreview(result.job, voice);
    } catch (err) {
      error = `The voice could not be played: ${err}`;
    }
    if (error) this.previewing = null;
    return error;
  }

  /** Play a sample prepared for cloning, as a preview plays: marked
   *  ("sample:<id>") from the click until it has been heard. Returns an
   *  error message, or null once it is on its way. */
  async playSample(sample: string): Promise<string | null> {
    const playing = `sample:${sample}`;
    this.previewing = playing;
    this.previewHeard = false;
    setTimeout(() => {
      if (this.previewing === playing && !this.previewHeard) this.previewing = null;
    }, PREVIEW_TIMEOUT);
    let error: string | null = null;
    try {
      const result = await rpc<{ ok: boolean; error?: string }>("voice.sample_play", { sample });
      if (!result.ok) error = result.error ?? "The sample could not be played";
    } catch (err) {
      error = `The sample could not be played: ${err}`;
    }
    if (error && this.previewing === playing) this.previewing = null;
    return error;
  }

  /** Stop a sample playing, if one is. */
  async stopSample() {
    if (!this.previewing?.startsWith("sample:")) return;
    this.previewing = null;
    await rpc("voice.sample_play", { on: false }).catch(() => {});
  }

  /** A preview whose voice downloads first stays marked until it is heard,
   *  or its download fails. */
  private waitForPreview(job: VoiceJob, voice: string) {
    this.takeJob(job, false);
    const check = () => {
      if (this.previewing !== voice) return;
      const now = this.jobs[job.id];
      if (now?.state === "failed") this.previewing = null;
      else if (now?.state === "ready") {
        setTimeout(() => {
          if (this.previewing === voice && !this.previewHeard) this.previewing = null;
        }, PREVIEW_TIMEOUT);
      } else setTimeout(check, 500);
    };
    this.previewHeard = false;
    setTimeout(check, 500);
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
    const result = await rpc<{ topics: MemoryTopic[]; turns?: number }>("memory.topics");
    this.memoryTopics = result.topics;
    this.memoryTurns = result.turns ?? 0;
  }

  async memoryTopic(id: number): Promise<MemoryTopicDetail | null> {
    const result = await rpc<MemoryTopicDetail & { ok: boolean }>("memory.topic", { id });
    return result.ok ? result : null;
  }

  async forgetTopic(id: number) {
    await rpc("memory.forget", { id });
    await this.loadMemory();
  }

  /** Forget everything Iris remembers, and the conversation with it (the
   *  core says so to every window, memory.forgotten). */
  async forgetEverything() {
    let result: { ok: boolean; error?: string };
    try {
      result = await rpc("memory.forget_all");
    } catch (err) {
      result = { ok: false, error: `${err}` };
    }
    if (!result.ok) {
      this.messages.push({
        role: "error",
        text: `Forgetting failed: ${result.error ?? "Iris did not answer"}`,
        streaming: false,
      });
    }
  }

  /** Listen only for the persona's wake word until addressed, or (off) to
   *  everything. */
  async setWake(on: boolean) {
    saveSettings({ wake_on: on });
    this.awakeUntil = null;
    await rpc("wake.set", { on });
  }

  /** A persona's wake word or sleep word (the current persona's without an
   *  id). An empty wake word is its name again, an empty sleep word none. */
  async setPersonaWords(words: { wake_word?: string; sleep_word?: string }, id?: string) {
    const result = await rpc<{ ok: boolean; personas?: string[]; names?: Record<string, string> }>(
      "persona.words",
      { ...(id ? { id } : {}), ...words }
    );
    if (result.ok && result.personas) {
      this.personas = result.personas;
      this.personaNames = result.names ?? {};
    }
  }

  /** Whether Iris falls asleep when nothing was said for a while, and after
   *  how long; the time is kept while it is off. */
  async setIdleSleep(on: boolean, window = loadSettings().wake_window ?? DEFAULT_WAKE_WINDOW) {
    saveSettings({ idle_sleep: on, wake_window: window });
    if (!on) this.awakeUntil = null;
    await rpc("wake.options", { window: on ? window : 0 });
  }

  /** Point Iris at model folders it should use as they are ("" = download).
   *  Returns an error message, or null when the folder was accepted. */
  async setModelPaths(paths: { stt_path?: string; tts_dir?: string }): Promise<string | null> {
    const result = await rpc<{ ok: boolean; error?: string }>("models.set", paths);
    if (!result.ok) return result.error ?? "Not a model folder";
    this.remember({
      ...(paths.stt_path !== undefined ? { stt_model_path: paths.stt_path } : {}),
      ...(paths.tts_dir !== undefined ? { tts_models_dir: paths.tts_dir } : {}),
    });
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
    // One listener for the core's events however often the store starts: in
    // development a hot reload runs this again in a fresh module, and a second
    // listener would show every message twice
    window.__irisOffEvent?.();
    window.__irisOffEvent = onEvent((type, data) => this.handleEvent(type, data));

    let micStatus = await window.iris.micStatus();
    // The walkthrough asks for the microphone at its step; a launch after
    // it asks now, should it never have been asked
    if (micStatus === "not-determined" && !this.needsSetup) {
      micStatus = await window.iris.askMicrophone();
    }
    let micNotice: ChatMessage | null = null;
    if (micStatus !== "granted" && !this.needsSetup) {
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
      if (info.config) {
        this.config = info.config;
        this.openThread = info.config.thread ?? null;
      }
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

      // First launch: nothing starts until the walkthrough is done; it
      // fetches the speech models while the user makes the other choices
      if (info.running) {
        this.needsSetup = false;
        this.started = true;
      }
      if (this.needsSetup) {
        this.setPresence(false);
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
      // The walkthrough of a first launch goes on with the new core
      if (this.needsSetup && !this.started) return;
      const info = await rpc<AppInfo>("app.info");
      if (!info.running) await this.startAssistant({ restore: false });
    } catch (err) {
      this.messages.push({ role: "error", text: `Iris could not be started again: ${err}`, streaming: false });
    }
  }

  /** The walkthrough's end on the first launch: keep its choices and start
   *  Iris with them. The conversation shows at once; the orb blooms while
   *  the models load, with the progress of any still downloading, and Iris
   *  greets the user once it listens. */
  async finishFirstRun(choices: WalkthroughChoices, greet: boolean) {
    const host = choices.ollama_host;
    const saved = loadSettings();
    saveSettings({
      persona: choices.persona,
      input_device: choices.input_device,
      wake_on: choices.wake_on,
      idle_sleep: choices.idle_sleep,
      tools: choices.tools,
      stt_model_path: choices.stt_model_path,
      tts_models_dir: choices.tts_models_dir,
      ...(host !== undefined ? { ollama_host: host } : {}),
      ...(choices.llm_model ? { llm_model: choices.llm_model } : {}),
      ...(host !== undefined && choices.llm_model
        ? { server_models: { ...saved.server_models, [host]: choices.llm_model } }
        : {}),
      setup_done: true,
    });
    if (host) this.keepServer(host);
    this.persona = choices.persona;
    this.needsSetup = false;
    // The words are the persona's, which Iris starts with
    if (choices.wake_word !== undefined) {
      await this.setPersonaWords(
        { wake_word: choices.wake_word, sleep_word: choices.sleep_word },
        choices.persona
      ).catch((err) => console.error("[store] persona.words failed:", err));
    }
    await this.startAssistant({ greet });
    this.loadLists();
  }

  /** The walkthrough's end when run again: what changed is applied to Iris
   *  as it runs, the microphone opens as it was, and Iris says hello. */
  async finishWalkthrough(
    choices: WalkthroughChoices,
    { greet, micOn }: { greet: boolean; micOn: boolean }
  ) {
    this.needsSetup = false;
    const config = this.config;
    const host = choices.ollama_host;
    try {
      if (host !== undefined && host !== (config?.ollama_host ?? "")) {
        await this.setOllamaHost(host);
      }
      if (choices.llm_model && choices.llm_model !== this.config?.llm_model) {
        await this.setModel(choices.llm_model);
      }
      // Another persona is talked to in a thread of its own
      if (choices.persona !== this.persona) await this.newThread(choices.persona, { choose: false });
      const device = this.inputDevices.find((d) => d.name === choices.input_device);
      if (device && device.name !== config?.input_device) await this.setMic(device.index);
      if (choices.wake_word !== undefined) {
        await this.setPersonaWords(
          { wake_word: choices.wake_word, sleep_word: choices.sleep_word },
          choices.persona
        );
      }
      if (choices.wake_on !== !!this.config?.wake_word) await this.setWake(choices.wake_on);
      if (choices.idle_sleep !== (this.config?.wake_window ?? DEFAULT_WAKE_WINDOW) > 0) {
        await this.setIdleSleep(choices.idle_sleep);
      }
      for (const [id, on] of Object.entries(choices.tools)) {
        const tool = this.config?.tools?.[id];
        if (tool?.available && !!tool.enabled !== on) await this.setTool(id, on);
      }
      if (choices.stt_model_path !== (config?.stt_model_path ?? "")) {
        await this.setModelPaths({ stt_path: choices.stt_model_path });
      }
      if (choices.tts_models_dir !== (config?.tts_models_dir ?? "")) {
        await this.setModelPaths({ tts_dir: choices.tts_models_dir });
      }
    } finally {
      await rpc("mic.set", { on: micOn });
    }
    if (greet) await rpc("assistant.greet");
  }

  /** Start the assistant with the saved settings; restore shows the recent
   *  part of the conversation the core remembers (not when it is shown
   *  already), and greet has Iris say hello once it listens (the end of the
   *  walkthrough). */
  private async startAssistant({ restore = true, greet = false } = {}) {
    this.started = true;
    await this.adoptWakeWords();
    const saved = loadSettings();
    const options: Record<string, unknown> = greet ? { greet: true } : {};
    if (!restore) options.restore_thread = false;
    if (saved.llm_model) options.llm_model = saved.llm_model;
    if (saved.persona) options.persona = saved.persona;
    // Without one, Iris opens the thread talked in last
    if (saved.thread !== undefined) options.thread = saved.thread;
    if (saved.speak_replies !== undefined) options.speak_replies = saved.speak_replies;
    if (saved.barge_in !== undefined) options.barge_in = saved.barge_in;
    if (saved.tools) options.tools = saved.tools;
    if (saved.ollama_host) options.ollama_host = saved.ollama_host;
    if (saved.remember === false) options.remember = false;
    if (saved.amnesia) options.amnesia = true;
    if (saved.wake_on) options.wake_on = true;
    if (saved.idle_sleep === false) options.wake_window = 0;
    else if (saved.wake_window) options.wake_window = saved.wake_window;
    if (saved.stt_model_path) options.stt_model_path = saved.stt_model_path;
    else if (saved.whisper_model) options.whisper_model = saved.whisper_model;
    if (saved.turn_detection) options.turn_detection = saved.turn_detection;
    if (saved.tts_models_dir) options.tts_models_dir = saved.tts_models_dir;
    if (saved.languages) options.languages = saved.languages;
    this.speakReplies = saved.speak_replies ?? true;
    await rpc("assistant.start", options);
    // The old switch went to the thread Iris started in, once
    if (saved.amnesia !== undefined) saveSettings({ amnesia: undefined });
  }

  /** The wake word and sleep word kept for all personas before each had
   *  its own become those of the persona in use, once, so they go on
   *  working; the switch is whether there was a wake word. */
  private async adoptWakeWords() {
    const saved = loadSettings();
    if (saved.wake_word === undefined && saved.sleep_word === undefined) return;
    // The switch holds whatever becomes of the words
    saveSettings({ wake_on: saved.wake_on ?? !!saved.wake_word?.trim() });
    try {
      const result = await rpc<{ ok: boolean }>("wake.adopt", {
        persona: saved.persona ?? "iris",
        wake_word: saved.wake_word ?? "",
        sleep_word: saved.sleep_word ?? "",
      });
      // Tried again at the next start, should it fail
      if (result.ok) saveSettings({ wake_word: undefined, sleep_word: undefined });
    } catch (err) {
      console.error("[store] wake.adopt failed:", err);
    }
  }

  private loadLists() {
    this.loadThreads();
    this.loadRoles().catch((err) => console.error("[store] roles.list failed:", err));
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

declare global {
  interface Window {
    /** Ends the store's listener for the core's events (see IrisStore.init) */
    __irisOffEvent?: () => void;
    /** Ends the walkthrough's listener for the core's events */
    __irisOffWalkthroughEvent?: () => void;
  }
}
