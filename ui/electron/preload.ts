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

  platform: process.platform,
});
