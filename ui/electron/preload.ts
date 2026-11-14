import { contextBridge, ipcRenderer, IpcRendererEvent } from "electron";

export type IrisEventCallback = (type: string, data: unknown) => void;

/** A window's part in a switch between the full window and presence; the
 *  payloads are typed in src/lib/bridge.ts. */
interface OrbHandlers {
  release: () => unknown;
  adopt: (handoff: unknown) => Promise<void>;
  rest: () => void;
}

contextBridge.exposeInMainWorld("iris", {
  rpc: (method: string, params: Record<string, unknown> = {}) =>
    ipcRenderer.invoke("rpc", method, params),

  onEvent: (callback: IrisEventCallback) => {
    const listener = (
      _event: IpcRendererEvent,
      payload: { type: string; data: unknown }
    ) => callback(payload.type, payload.data);
    ipcRenderer.on("iris-event", listener);
    return () => ipcRenderer.removeListener("iris-event", listener);
  },

  micStatus: () => ipcRenderer.invoke("mic-status"),

  isPackaged: () => ipcRenderer.invoke("is-packaged"),

  /** The system's folder picker; resolves to the chosen path, or null */
  chooseFolder: (title: string) => ipcRenderer.invoke("choose-folder", title),

  /** Whether Iris opened as presence (only the orb) */
  presence: ipcRenderer.sendSync("get-presence") as boolean,

  /** Show presence in place of the full window, or the full window again */
  setPresence: (on: boolean) => ipcRenderer.invoke("set-presence", on),

  /** Take part in the switches between the windows: the window that hides
   *  releases the orb's motion, the one that shows adopts it and draws */
  onOrb: (handlers: OrbHandlers) => {
    const release = () => ipcRenderer.send("orb-release-reply", handlers.release());
    const adopt = async (_event: IpcRendererEvent, handoff: unknown) => {
      await handlers.adopt(handoff);
      ipcRenderer.send("orb-adopt-reply", true);
    };
    const rest = () => handlers.rest();
    ipcRenderer.on("orb-release", release);
    ipcRenderer.on("orb-adopt", adopt);
    ipcRenderer.on("orb-rest", rest);
    return () => {
      ipcRenderer.removeListener("orb-release", release);
      ipcRenderer.removeListener("orb-adopt", adopt);
      ipcRenderer.removeListener("orb-rest", rest);
    };
  },

  /** Conversation window: what the orb in presence shows */
  sendOrbInputs: (inputs: unknown) => ipcRenderer.send("orb-inputs", inputs),

  /** Presence window: what its orb shows */
  onOrbInputs: (callback: (inputs: unknown) => void) => {
    const listener = (_event: IpcRendererEvent, inputs: unknown) => callback(inputs);
    ipcRenderer.on("orb-inputs", listener);
    return () => ipcRenderer.removeListener("orb-inputs", listener);
  },

  /** Presence window: ask the conversation window to leave presence or pause */
  presenceCommand: (command: string) => ipcRenderer.send("presence-command", command),

  /** Conversation window: what the presence window asks */
  onPresenceCommand: (callback: (command: string) => void) => {
    const listener = (_event: IpcRendererEvent, command: string) => callback(command);
    ipcRenderer.on("presence-command", listener);
    return () => ipcRenderer.removeListener("presence-command", listener);
  },

  /** Presence window: whether the pointer is over the orb */
  presenceHover: (over: boolean) => ipcRenderer.send("presence-hover", over),

  /** Presence window: a press on the orb moves the window; resolves to
   *  whether it moved */
  presenceDrag: (phase: string) => ipcRenderer.invoke("presence-drag", phase),

  platform: process.platform,
});
