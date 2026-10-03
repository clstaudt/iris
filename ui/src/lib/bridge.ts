/** Typed wrapper over the preload-exposed window.iris API. */

export interface IrisEventPayload {
  type: string;
  data: Record<string, unknown>;
}

export interface IrisApi {
  rpc: (method: string, params?: Record<string, unknown>) => Promise<unknown>;
  onEvent: (callback: (type: string, data: IrisEventPayload["data"]) => void) => () => void;
  micStatus: () => Promise<string>;
  platform: string;
}

declare global {
  interface Window {
    iris: IrisApi;
  }
}

export interface AppInfo {
  version: string;
  personas: string[];
  state: string;
  running: boolean;
  config?: {
    persona: string;
    llm_model: string;
    whisper_model: string;
    whisper_device: string;
    tts_voice: string;
    input_device: string;
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
