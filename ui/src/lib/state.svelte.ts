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
  | "paused";

export interface ChatMessage {
  role: "user" | "assistant" | "tool" | "error" | "system";
  text: string;
  streaming: boolean;
  tool?: string;
}

/** State colors mirror the Rich TUI palette. */
export const STATE_COLORS: Record<string, string> = {
  // At rest the orb is plain white; every other state tints it
  idle: "#e8e8ea",
  starting: "#94a3b8",
  listening: "#ef4444",
  understanding: "#a855f7",
  thinking: "#22d3ee",
  speaking: "#eab308",
  looking: "#e879f9",
  capturing_screen: "#e879f9",
  searching: "#67e8f9",
  browsing: "#67e8f9",
  summarizing: "#60a5fa",
  paused: "#a8a29e",
};

export interface AssistantConfig {
  persona: string;
  llm_model: string;
  whisper_model: string;
  whisper_device: string;
  tts_voice: string;
  input_device: string;
}

class IrisStore {
  state = $state<AssistantState>("idle");
  running = $state(false);
  personas = $state<string[]>([]);
  persona = $state("iris");
  config = $state<AssistantConfig | null>(null);
  inputDevices = $state<{ index: number; name: string }[]>([]);
  models = $state<{ name: string; tools: boolean; vision: boolean }[]>([]);
  messages = $state<ChatMessage[]>([]);

  /** Mic input level, driving the meter and spinner while listening. */
  level = $state(0);
  /** TTS playback level, driving the meter while the assistant speaks. */
  outputLevel = $state(0);

  private streamingIndex: number | null = null;

  handleEvent(type: string, data: Record<string, unknown>) {
    switch (type) {
      case "config":
        this.config = data as unknown as AssistantConfig;
        this.persona = this.config.persona.toLowerCase();
        break;

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

      case "chat.user":
        this.streamingIndex = null;
        this.messages.push({
          role: "user",
          text: (data.text as string) ?? "",
          streaming: false,
        });
        break;

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
          streaming: false,
        });
        break;
    }
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
      this.state = info.state as AssistantState;
      this.running = info.running;
      if (info.config) this.config = info.config;
      const inputs = await rpc<{ devices: { index: number; name: string }[] }>(
        "audio.list_inputs"
      );
      this.inputDevices = inputs.devices;

      const llms = await rpc<{
        models: { name: string; tools: boolean; vision: boolean }[];
      }>("llm.list");
      this.models = llms.models;
      // Mirror the terminal flow: the assistant starts listening on launch
      if (!info.running) {
        this.messages.push({
          role: "system",
          text: "starting assistant — loading models, this can take a moment…",
          streaming: false,
        });
        await rpc("assistant.start");
      }
    } catch (err) {
      console.error("[store] app.info failed:", err);
    }
  }
}

export const store = new IrisStore();
