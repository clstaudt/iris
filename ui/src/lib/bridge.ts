/** Typed wrapper over the preload-exposed window.iris API. */

export interface IrisEventPayload {
  type: string;
  data: Record<string, unknown>;
}

export interface IrisApi {
  rpc: (method: string, params?: Record<string, unknown>) => Promise<unknown>;
  onEvent: (callback: (type: string, data: IrisEventPayload["data"]) => void) => () => void;
  micStatus: () => Promise<string>;
  isPackaged: () => Promise<boolean>;
  chooseFolder: (title: string) => Promise<string | null>;
  /** Whether the window opened as presence (only the orb) */
  presence: boolean;
  setPresence: (on: boolean) => Promise<void>;
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
