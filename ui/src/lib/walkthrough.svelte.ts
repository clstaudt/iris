/** The walkthrough: from opening Iris for the first time to talking with it,
 *  a step at a time. Run again from Settings, it goes over the same choices
 *  with Iris running.
 *
 *  Its choices are drafts until the last step, which keeps them and starts
 *  Iris, or applies them to Iris as it runs. Skipping a step leaves what it
 *  shows as it was when the step opened. The speech models download in the
 *  background from their step on, while the user goes on with the rest. */

import { onEvent, rpc } from "./bridge";
import { DEFAULT_WAKE_WINDOW, loadSettings, store, type PersonaSettings } from "./state.svelte";

export const STEPS = ["welcome", "server", "models", "microphone", "voice", "extras", "ready"] as const;
export type Step = (typeof STEPS)[number];

/** The model setup offers first, when the server has it: small, and it calls tools */
export const PREFERRED_MODEL = "qwen3.5:9b";
/** Plugins the walkthrough does not offer: memory is always on, and going to
 *  sleep comes with the wake word */
const UNOFFERED_TOOLS = new Set(["memory", "sleep"]);

/** A model server in the list; "" is the default one on this computer. */
export interface ServerEntry {
  url: string;
  /** Whether it answered when last asked; unknown until then */
  reachable?: boolean;
}

export interface LlmModel {
  name: string;
  tools: boolean;
  vision: boolean;
}

/** A speech model as the core fetches it. */
export interface SpeechDownload {
  id: string;
  label: string;
  detail: string;
  size_mb: number | null;
  state: "waiting" | "downloading" | "done" | "failed";
  done: number;
  total: number;
  error: string;
}

export interface PersonaOption {
  id: string;
  name: string;
  engine: string;
  voice: string;
  speed: number;
  audio_filter: string;
  description: string;
  wake_word: string;
  sleep_word: string;
}

export interface ToolOption {
  id: string;
  name: string;
  available: boolean;
  on: boolean;
}

/** What the last step keeps or applies. */
export interface WalkthroughChoices {
  /** The server as the core keeps its address; undefined leaves it */
  ollama_host?: string;
  llm_model?: string;
  persona: string;
  /** The microphone by name; undefined for the system's */
  input_device?: string;
  wake_on: boolean;
  /** The persona's words; undefined leaves them (they were not shown) */
  wake_word?: string;
  /** "" for none */
  sleep_word?: string;
  /** Whether Iris falls asleep when nothing was said for a while */
  idle_sleep: boolean;
  tools: Record<string, boolean>;
  stt_model_path: string;
  tts_models_dir: string;
}

/** The model to offer first: the preferred one, else one that calls tools. */
export function preferredModel(models: LlmModel[]): string {
  const model =
    models.find((m) => m.name === PREFERRED_MODEL) ?? models.find((m) => m.tools) ?? models[0];
  return model?.name ?? "";
}

/** A server by its address without the scheme; the default one by where it is. */
export function serverName(url: string): string {
  if (!url) return "This computer";
  try {
    const parsed = new URL(url.includes("://") ? url : `http://${url}`);
    return parsed.host + (parsed.pathname === "/" ? "" : parsed.pathname);
  } catch {
    return url;
  }
}

/** A voice as people say it: "bm_lewis" is Lewis, "alba" is Alba. */
export function voiceName(voice: string): string {
  const name = voice.replace(/^[a-z][fm]_/, "").replace(/_/g, " ");
  return name.replace(/\b\w/g, (c) => c.toUpperCase());
}

const ENGINE_NAMES: Record<string, string> = { kyutai: "Kyutai", kokoro: "Kokoro" };
export const engineName = (engine: string) => ENGINE_NAMES[engine] ?? engine;

class Walkthrough {
  /** Shown now: the first launch, or run again from Settings */
  active = $state(false);
  /** Before Iris has ever run: the last step starts it */
  firstRun = $state(true);
  step = $state<Step>("welcome");
  /** The furthest step reached, which the dots can go back to */
  reached = $state(0);
  starting = $state(false);

  // -- the language model ----------------------------------------------
  servers = $state<ServerEntry[]>([{ url: "" }]);
  /** The server chosen, under the address the core settled on (.../v1 for
   *  an OpenAI-compatible server, the root for Ollama); null for none */
  server = $state<string | null>(null);
  serverModels = $state<LlmModel[]>([]);
  model = $state("");
  connecting = $state<string | null>(null);
  serverError = $state("");
  searching = $state(false);
  /** Servers have been looked for once */
  searched = $state(false);
  /** Where the default server is, e.g. http://localhost:11434 */
  defaultUrl = $state("");

  // -- the speech models -----------------------------------------------
  downloads = $state<SpeechDownload[]>([]);
  downloading = $state(false);
  sttPath = $state("");
  ttsDir = $state("");
  folderError = $state("");

  // -- the microphone --------------------------------------------------
  /** "granted", "denied", "restricted", "not-determined", or "unknown"
   *  until asked */
  micAccess = $state("unknown");
  inputs = $state<{ index: number; name: string }[]>([]);
  /** The microphone chosen; null for the system's */
  micDevice = $state<number | null>(null);
  level = $state(0);
  heard = $state(false);
  silent = $state(false);
  micError = $state("");
  listening = $state(false);

  // -- the voice -------------------------------------------------------
  personas = $state<PersonaOption[]>([]);
  persona = $state("iris");
  previewError = $state("");

  // -- wake word and tools ---------------------------------------------
  wakeOn = $state(false);
  /** The persona's words, as edited here */
  wakeWord = $state("");
  sleepWord = $state("");
  idleSleep = $state(true);
  tools = $state<ToolOption[]>([]);
  /** The persona whose words the fields hold */
  private wordsOf: string | null = null;

  /** What Skip puts back on the step shown */
  private undo: (() => void) | null = null;
  /** Counts the servers asked for their models, so only the latest counts */
  private asking = 0;
  private subscribed = false;
  /** Whether the microphone was on before the walkthrough ran again */
  private micWasOn = true;

  get index() {
    return STEPS.indexOf(this.step);
  }

  /** Bytes in, of all the speech models (estimated where a download has
   *  not said its size yet), from 0 to 1. */
  get downloaded(): number {
    let done = 0;
    let total = 0;
    for (const model of this.downloads) {
      const size = model.total || (model.size_mb ?? 0) * 1e6;
      total += size;
      if (model.state === "done") done += size;
      else if (model.state === "downloading" && model.total) done += model.done;
    }
    return total ? Math.min(1, done / total) : 1;
  }

  /** The name of the persona chosen, its wake word unless given another */
  get personaName(): string {
    return this.personas.find((p) => p.id === this.persona)?.name ?? "Iris";
  }

  get downloadsFailed(): boolean {
    return this.downloads.some((m) => m.state === "failed");
  }

  get downloadsDone(): boolean {
    return this.downloads.length > 0 && this.downloads.every((m) => m.state === "done");
  }

  /** Whether a persona's voice can be heard yet: its engine's model is in. */
  voiceReady(engine: string): boolean {
    if (!this.firstRun) return true;
    const model = this.downloads.find((m) => m.id === engine);
    return !model || model.state === "done";
  }

  /** Open the walkthrough on its first step, with the choices in use (or the
   *  defaults, the first time). */
  async open(firstRun: boolean) {
    this.subscribe();
    const saved = loadSettings();
    const config = store.config;
    this.firstRun = firstRun;
    this.step = "welcome";
    this.reached = 0;
    this.starting = false;
    this.undo = null;

    this.server = firstRun ? null : (config?.ollama_host ?? saved.ollama_host ?? "");
    this.model = firstRun ? "" : (config?.llm_model ?? "");
    this.serverModels = firstRun ? [] : store.models;
    const urls = ["", ...(saved.servers ?? []), ...(this.server ? [this.server] : [])];
    this.servers = [...new Set(urls)].map((url) => ({ url }));
    this.serverError = "";
    this.searched = false;

    this.sttPath = config?.stt_model_path ?? saved.stt_model_path ?? "";
    this.ttsDir = config?.tts_models_dir ?? saved.tts_models_dir ?? "";
    this.folderError = "";

    this.micDevice = null;
    this.micAccess = "unknown";
    this.heard = false;

    this.persona = store.persona || saved.persona || "iris";
    this.wakeOn = config ? !!config.wake_word : !!(saved.wake_on ?? saved.wake_word);
    this.idleSleep = config
      ? (config.wake_window ?? DEFAULT_WAKE_WINDOW) > 0
      : saved.idle_sleep !== false;
    this.wordsOf = null;
    this.tools = [];

    if (!firstRun) {
      // Iris does not listen while its choices are gone over
      this.micWasOn = store.micOn;
      rpc("mic.set", { on: false }).catch(() => {});
      store.openSettings(null);
      store.panel = null;
    }
    this.active = true;
    store.needsSetup = true;

    const info = await rpc<{ ollama_url: string; models: SpeechDownload[] }>("setup.info");
    this.defaultUrl = info.ollama_url;
    if (!this.downloads.length) this.downloads = info.models ?? [];
  }

  private subscribe() {
    if (this.subscribed) return;
    this.subscribed = true;
    // One listener, also after a hot reload in development (see IrisStore.init)
    window.__irisOffWalkthroughEvent?.();
    window.__irisOffWalkthroughEvent = onEvent((type, data) => {
      if (type === "setup.models") {
        this.downloads = data.models as SpeechDownload[];
        this.downloading = !!data.busy;
      } else if (type === "mic.check" && this.listening) {
        this.level = (data.level as number) ?? 0;
        if (data.heard) this.heard = true;
        this.silent = !!data.silent;
      }
    });
  }

  // ---------------------------------------------------------------------
  // Moving between the steps
  // ---------------------------------------------------------------------

  goto(step: Step) {
    if (step === this.step) return;
    this.leave(this.step);
    this.step = step;
    this.reached = Math.max(this.reached, STEPS.indexOf(step));
    this.enter(step);
  }

  next() {
    const next = STEPS[this.index + 1];
    if (next) this.goto(next);
  }

  back() {
    const previous = STEPS[this.index - 1];
    if (previous) this.goto(previous);
  }

  /** On to the next step, leaving this one's choices as they were. */
  skip() {
    this.undo?.();
    this.next();
  }

  private enter(step: Step) {
    this.undo = null;
    if (step === "server") {
      const { server, model, serverModels } = this;
      this.undo = () => {
        this.server = server;
        this.model = model;
        this.serverModels = serverModels;
      };
      if (!this.searched) this.findServers();
    } else if (step === "models") {
      const { sttPath, ttsDir } = this;
      this.undo = () => {
        if (this.sttPath === sttPath && this.ttsDir === ttsDir) return;
        this.sttPath = sttPath;
        this.ttsDir = ttsDir;
        this.download();
      };
      this.download();
    } else if (step === "microphone") {
      const device = this.micDevice;
      this.undo = () => (this.micDevice = device);
      this.openMicrophone();
    } else if (step === "voice") {
      const persona = this.persona;
      this.undo = () => (this.persona = persona);
      this.loadPersonas();
    } else if (step === "extras") {
      this.takePersonaWords();
      const { wakeOn, idleSleep } = this;
      const tools = this.tools.map((tool) => ({ ...tool }));
      this.undo = () => {
        this.wakeOn = wakeOn;
        this.idleSleep = idleSleep;
        this.wordsOf = null;
        this.takePersonaWords();
        if (tools.length) this.tools = tools;
      };
      this.loadTools();
    }
  }

  private leave(step: Step) {
    if (step === "microphone") this.closeMicrophone();
    // The persona chosen brings its words, unless they are its already
    if (step === "voice") this.takePersonaWords();
    // A wake word left empty is the persona's name
    if (step === "extras" && !this.wakeWord.trim()) this.wakeWord = this.personaName;
    // A wake word needs a small model of its own, fetched with the others
    if (step === "extras" && this.firstRun && (this.wakeOn || this.sleepWord.trim())) {
      this.download();
    }
  }

  // ---------------------------------------------------------------------
  // The language model
  // ---------------------------------------------------------------------

  /** Ask which servers answer, look for others on this computer, and choose
   *  the first that answers unless one is chosen. */
  async findServers() {
    if (this.searching) return;
    this.searching = true;
    try {
      const [found] = await Promise.all([
        rpc<{ servers: string[] }>("llm.find_servers").catch(() => ({ servers: [] as string[] })),
        this.checkServers(),
      ]);
      for (const url of found.servers) {
        if (!this.servers.some((s) => s.url === url)) this.servers.push({ url, reachable: true });
      }
    } finally {
      this.searching = false;
      this.searched = true;
    }
    // Unless the user has chosen one meanwhile
    if (this.server === null && this.connecting === null) {
      const first = this.servers.find((s) => s.reachable);
      if (first) await this.connect(first.url);
    }
  }

  async checkServers() {
    const urls = this.servers.map((s) => s.url);
    const result = await rpc<{ servers: { url: string; reachable: boolean }[] }>(
      "llm.check_hosts",
      { urls }
    );
    for (const { url, reachable } of result.servers) {
      const entry = this.servers.find((s) => s.url === url);
      if (entry) entry.reachable = reachable;
    }
  }

  /** Ask a server for its models and choose it. It is kept under the
   *  address the core settled on, not as typed: an OpenAI-compatible server
   *  typed without /v1 is kept with it, or Iris would take it for Ollama.
   *  The latest choice wins: a server asked before it answers too late. */
  async connect(url: string) {
    const asked = ++this.asking;
    this.connecting = url;
    this.serverError = "";
    try {
      const result = await rpc<{ ok: boolean; host?: string; models?: LlmModel[]; error?: string }>(
        "llm.probe",
        { url }
      );
      if (asked !== this.asking) return;
      const entry = this.servers.find((s) => s.url === url);
      if (!result.ok) {
        if (entry) entry.reachable = false;
        this.serverError = `No server answers at ${serverName(url)}`;
        return;
      }
      const host = result.host ?? "";
      if (host !== url) {
        this.servers = this.servers.filter((s) => s.url !== url && s.url !== host);
        this.servers.push({ url: host, reachable: true });
      } else if (entry) {
        entry.reachable = true;
      }
      this.server = host;
      this.serverModels = result.models ?? [];
      if (!this.serverModels.some((m) => m.name === this.model)) {
        this.model = preferredModel(this.serverModels);
      }
    } catch (err) {
      if (asked === this.asking) this.serverError = String(err);
    } finally {
      if (asked === this.asking) this.connecting = null;
    }
  }

  /** A server typed by the user: listed, and asked for its models. */
  async addServer(url: string) {
    const typed = url.trim().replace(/\/+$/, "");
    if (!typed) return;
    if (!this.servers.some((s) => s.url === typed)) this.servers.push({ url: typed });
    await this.connect(typed);
  }

  /** Look again, after the user started a server. */
  async searchAgain() {
    this.searched = false;
    await this.findServers();
  }

  // ---------------------------------------------------------------------
  // The speech models
  // ---------------------------------------------------------------------

  /** Have the core fetch what is missing (again, after a failure). */
  async download() {
    const result = await rpc<{ models: SpeechDownload[] }>("setup.download", {
      whisper_model: loadSettings().whisper_model,
      stt_path: this.sttPath || undefined,
      tts_dir: this.ttsDir || undefined,
      // The sleep word wakes Iris by the wake word, on or off
      wake_word:
        this.wakeOn || this.sleepWord.trim() ? this.wakeWord.trim() || this.personaName : undefined,
    });
    this.downloads = result.models;
  }

  /** Use a model folder the user has instead of a download. */
  async chooseFolder(kind: "stt" | "tts") {
    const title = kind === "stt" ? "Speech recognition model folder" : "Voice model folder";
    const folder = await window.iris.chooseFolder(title);
    if (!folder) return;
    const result = await rpc<{ ok: boolean; error?: string }>(
      "models.check",
      kind === "stt" ? { stt_path: folder } : { tts_dir: folder }
    );
    this.folderError = result.ok ? "" : (result.error ?? "Not a model folder");
    if (!result.ok) return;
    if (kind === "stt") this.sttPath = folder;
    else this.ttsDir = folder;
    await this.download();
  }

  async useDownload(kind: "stt" | "tts") {
    if (kind === "stt") this.sttPath = "";
    else this.ttsDir = "";
    await this.download();
  }

  // ---------------------------------------------------------------------
  // The microphone
  // ---------------------------------------------------------------------

  async openMicrophone() {
    this.micError = "";
    this.silent = false;
    this.level = 0;
    if (!this.inputs.length) {
      rpc<{ devices: { index: number; name: string }[] }>("audio.list_inputs")
        .then((result) => {
          this.inputs = result.devices;
          // The microphone kept from before, if it is still there
          const saved = loadSettings().input_device;
          const device = this.inputs.find((d) => d.name === saved);
          if (device && this.micDevice === null) this.micDevice = device.index;
        })
        .catch(() => {});
    }
    this.micAccess = await window.iris.micStatus();
    if (this.micAccess === "granted" && this.step === "microphone") await this.listen();
  }

  /** Ask the system for the microphone (macOS asks the user once). */
  async allowMicrophone() {
    this.micAccess = await window.iris.askMicrophone();
    if (this.micAccess === "granted" && this.step === "microphone") await this.listen();
  }

  private async listen() {
    this.listening = true;
    const result = await rpc<{ ok: boolean; error?: string }>("mic.check", {
      on: true,
      index: this.micDevice,
    });
    if (!result.ok) {
      this.listening = false;
      this.micError = "The microphone could not be opened";
    }
  }

  async chooseMicrophone(index: number | null) {
    this.micDevice = index;
    this.heard = false;
    this.silent = false;
    if (this.listening) await this.listen();
  }

  closeMicrophone() {
    if (!this.listening) return;
    this.listening = false;
    this.level = 0;
    rpc("mic.check", { on: false }).catch(() => {});
  }

  // ---------------------------------------------------------------------
  // The voice
  // ---------------------------------------------------------------------

  async loadPersonas() {
    const list = await rpc<{ personas: string[] }>("persona.list");
    const loaded = await Promise.all(list.personas.map((id) => store.loadPersona(id)));
    this.personas = loaded
      .filter((p): p is PersonaSettings => !!p && !!p.id)
      .map((p) => ({
        id: p.id!,
        name: p.name,
        engine: p.engine,
        voice: p.voice,
        speed: p.speed,
        audio_filter: p.audio_filter,
        description: p.description,
        wake_word: p.wake_word || p.name,
        sleep_word: p.sleep_word ?? "",
      }));
    if (!this.personas.some((p) => p.id === this.persona) && this.personas.length) {
      this.persona = this.personas[0].id;
    }
  }

  async preview(persona: PersonaOption) {
    this.previewError = (await store.previewVoice({ ...persona })) ?? "";
  }

  // ---------------------------------------------------------------------
  // Wake word and tools
  // ---------------------------------------------------------------------

  /** Show the words of the persona chosen, unless they are shown already:
   *  another persona chosen since brings its own. */
  private async takePersonaWords() {
    if (!this.personas.length) await this.loadPersonas().catch(() => {});
    if (this.wordsOf === this.persona) return;
    const persona = this.personas.find((p) => p.id === this.persona);
    this.wakeWord = persona?.wake_word || this.personaName;
    this.sleepWord = persona?.sleep_word ?? "";
    this.wordsOf = this.persona;
  }

  async loadTools() {
    if (this.tools.length) return;
    const result = await rpc<{
      plugins: {
        id: string;
        name: string;
        available: boolean;
        switchable: boolean;
        enabled_by_default?: boolean;
      }[];
    }>("plugins.list");
    const saved = loadSettings().tools ?? {};
    const live = store.config?.tools;
    this.tools = result.plugins
      .filter((p) => p.switchable && !UNOFFERED_TOOLS.has(p.id))
      .map((p) => ({
        id: p.id,
        name: p.name,
        available: p.available,
        on: p.available && (live?.[p.id]?.enabled ?? saved[p.id] ?? p.enabled_by_default ?? true),
      }));
  }

  // ---------------------------------------------------------------------
  // The end
  // ---------------------------------------------------------------------

  choices(): WalkthroughChoices {
    const device = this.inputs.find((d) => d.index === this.micDevice);
    return {
      ...(this.server !== null ? { ollama_host: this.server } : {}),
      ...(this.server !== null && this.model ? { llm_model: this.model } : {}),
      persona: this.persona,
      ...(device ? { input_device: device.name } : {}),
      wake_on: this.wakeOn,
      ...(this.wordsOf === this.persona
        ? { wake_word: this.wakeWord.trim() || this.personaName, sleep_word: this.sleepWord.trim() }
        : {}),
      idle_sleep: this.idleSleep,
      tools: Object.fromEntries(this.tools.map((tool) => [tool.id, tool.on])),
      stt_model_path: this.sttPath,
      tts_models_dir: this.ttsDir,
    };
  }

  /** Keep the choices and start talking: Iris starts (or takes them on, run
   *  again) and greets the user, when it has a model to greet with. */
  async finish() {
    if (this.starting) return;
    this.starting = true;
    this.closeMicrophone();
    const choices = this.choices();
    const greet = !!choices.llm_model || (!this.firstRun && !!store.config?.llm_model);
    this.active = false;
    if (this.firstRun) await store.finishFirstRun(choices, greet);
    else await store.finishWalkthrough(choices, { greet, micOn: this.micWasOn });
  }

  /** Leave a walkthrough run again from Settings, changing nothing. */
  cancel() {
    if (this.firstRun) return;
    this.closeMicrophone();
    this.active = false;
    store.needsSetup = false;
    rpc("mic.set", { on: this.micWasOn }).catch(() => {});
  }
}

export const walkthrough = new Walkthrough();

