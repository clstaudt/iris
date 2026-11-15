/** Typed wrapper over the preload-exposed window.iris API. */

export interface IrisEventPayload {
  type: string;
  data: Record<string, unknown>;
}

/** Where the orb's animation stands, handed from the window that hides to
 *  the one that shows, so the orb carries on without a jump. */
export interface OrbMotion {
  col: [number, number, number];
  level: number;
  eased: number;
  shutter: number;
  wasLooking: boolean;
  spinner: number;
  spin: number;
  blooming: boolean;
  unfold: number;
  merge: number;
}

/** What the orb shows, as the conversation window's store has it. */
export interface OrbInputs {
  state: string;
  level: number;
  outputLevel: number;
  /** When Iris falls asleep, in ms since the epoch: each window counts
   *  performance.now() from its own start */
  awakeUntil: number | null;
  /** The latest message, when it is an error that needs the user */
  error: string | null;
}

export interface OrbHandoff {
  motion: OrbMotion | null;
  /** Sent by the conversation window, which holds the store */
  inputs?: OrbInputs;
}

/** A window's part in a switch between the full window and presence. */
export interface OrbHandlers {
  /** The orb's motion, as the window is about to hide */
  release: () => OrbHandoff;
  /** Take over the motion and draw; resolves once a frame is drawn */
  adopt: (handoff: OrbHandoff | null) => Promise<void>;
  /** The window is hidden: the orb stops drawing */
  rest: () => void;
}

/** What the presence window asks of the conversation window. */
export type PresenceCommand = "leave" | "pause";

export interface IrisApi {
  rpc: (method: string, params?: Record<string, unknown>) => Promise<unknown>;
  onEvent: (callback: (type: string, data: IrisEventPayload["data"]) => void) => () => void;
  /** Whether the app may use the microphone: "granted", "denied",
   *  "restricted" or "not-determined" (never asked) */
  micStatus: () => Promise<string>;
  /** Ask for the microphone (macOS asks the user once); resolves to the
   *  status afterwards */
  askMicrophone: () => Promise<string>;
  isPackaged: () => Promise<boolean>;
  chooseFolder: (title: string) => Promise<string | null>;
  /** Whether Iris opened as presence (only the orb) */
  presence: boolean;
  /** Show presence in place of the full window, or the full window again */
  setPresence: (on: boolean) => Promise<void>;
  /** Take part in the switches between the windows */
  onOrb: (handlers: OrbHandlers) => () => void;
  /** Conversation window: what the orb in presence shows */
  sendOrbInputs: (inputs: OrbInputs) => void;
  /** Presence window: what its orb shows */
  onOrbInputs: (callback: (inputs: OrbInputs) => void) => () => void;
  /** Presence window: ask the conversation window to leave presence or pause */
  presenceCommand: (command: PresenceCommand) => void;
  /** Conversation window: what the presence window asks */
  onPresenceCommand: (callback: (command: PresenceCommand) => void) => () => void;
  /** Presence window: whether the pointer is over the orb; everywhere else,
   *  clicks go to the windows below */
  presenceHover: (over: boolean) => void;
  /** Presence window: a press on the orb moves the window with the pointer.
   *  Resolves to whether it moved, so a press that did not is a click. */
  presenceDrag: (phase: "start" | "move" | "end") => Promise<boolean>;
  platform: string;
}

declare global {
  interface Window {
    iris: IrisApi;
  }
}

/** Where one turn's time went, in ms, from the end of speech to the first
 *  word heard; null for a stage the turn did not pass. */
export interface TurnTiming {
  source: "voice" | "text";
  stages: {
    turn_end: number | null;
    recognition: number | null;
    reply: number | null;
    voice: number | null;
  };
  total: number;
  /** What ran each stage, to match a measurement to the setup */
  components: Record<string, string>;
}

export interface AppInfo {
  /** The latest turns' timings */
  timings?: TurnTiming[];
  version: string;
  personas: string[];
  persona_names?: Record<string, string>;
  state: string;
  running: boolean;
  config?: {
    persona: string;
    llm_model: string;
    whisper_model: string;
    whisper_device: string;
    tts_voice: string;
    input_device: string;
    speak_replies?: boolean;
  };
}

export async function rpc<T = unknown>(
  method: string,
  params: Record<string, unknown> = {}
): Promise<T> {
  return (await window.iris.rpc(method, params)) as T;
}

export function onEvent(
  callback: (type: string, data: IrisEventPayload["data"]) => void
): () => void {
  return window.iris.onEvent(callback);
}
