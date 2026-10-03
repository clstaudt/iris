import { app, BrowserWindow, ipcMain, systemPreferences } from "electron";
import path from "node:path";
import { PythonBridge } from "./python-bridge";

// Bundled to CJS by esbuild, so the CommonJS __dirname global is available.
declare const __dirname: string;

let mainWindow: BrowserWindow | null = null;
let pythonBridge: PythonBridge | null = null;

function createWindow() {
  mainWindow = new BrowserWindow({
    title: "iris",
    width: 1280,
    height: 800,
    minWidth: 720,
    minHeight: 480,
    show: false,
    titleBarStyle: "hiddenInset",
    trafficLightPosition: { x: 16, y: 16 },
    backgroundColor: "#050505",
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  mainWindow.once("ready-to-show", () => {
    mainWindow?.show();
  });

  if (process.env.VITE_DEV_SERVER_URL) {
    mainWindow.loadURL(process.env.VITE_DEV_SERVER_URL);
    mainWindow.webContents.openDevTools({ mode: "detach" });
  } else {
    mainWindow.loadFile(path.join(__dirname, "../dist/index.html"));
  }

  mainWindow.on("closed", () => {
    mainWindow = null;
  });

  // Route Python events to the renderer
  if (pythonBridge) {
    pythonBridge.onEvent = (type, data) => {
      mainWindow?.webContents.send("iris-event", { type, data });
    };
  }
}

app.name = "iris";

/**
 * The Python child process opens the microphone, but macOS attributes that
 * access to this app. Nothing in Electron itself touches the mic, so the
 * system would never prompt — and an app that never asks does not appear in
 * System Settings > Privacy & Security > Microphone. Asking here registers
 * the app with TCC and triggers the prompt.
 */
async function ensureMicrophoneAccess(): Promise<string> {
  if (process.platform !== "darwin") return "granted";

  const status = systemPreferences.getMediaAccessStatus("microphone");
  console.log(`[main] microphone access status: ${status}`);
  if (status === "granted") return status;

  const granted = await systemPreferences.askForMediaAccess("microphone");
  const result = granted ? "granted" : "denied";
  console.log(`[main] microphone access after prompt: ${result}`);
  return result;
}

app.whenReady().then(async () => {
  const micStatus = await ensureMicrophoneAccess();
  const isPackaged = app.isPackaged;
  const projectRoot = isPackaged
    ? app.getAppPath()
    : path.resolve(__dirname, "../..");
  const resourcesPath = isPackaged ? process.resourcesPath : "";

  pythonBridge = new PythonBridge(projectRoot, isPackaged, resourcesPath);

  ipcMain.handle(
    "rpc",
    async (_event, method: string, params: Record<string, unknown>) => {
      if (!pythonBridge) throw new Error("Python bridge not initialised");
      try {
        return await pythonBridge.call(method, params);
      } catch (err) {
        console.error(`[main] rpc error in ${method}:`, err);
        throw err;
      }
    }
  );

  ipcMain.handle("mic-status", () => micStatus);

  createWindow();
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") app.quit();
});

app.on("activate", () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});

app.on("before-quit", () => {
  pythonBridge?.kill();
});
