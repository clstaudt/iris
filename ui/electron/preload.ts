import { contextBridge, ipcRenderer, IpcRendererEvent } from "electron";

export type IrisEventCallback = (type: string, data: unknown) => void;

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

  /** Whether the window opened as presence (only the orb) */
  presence: ipcRenderer.sendSync("get-presence") as boolean,

  /** Shrink the window to presence, or bring the conversation back */
  setPresence: (on: boolean) => ipcRenderer.invoke("set-presence", on),

  platform: process.platform,
});
