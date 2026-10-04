import {
  app,
  BrowserWindow,
  dialog,
  ipcMain,
  nativeImage,
  shell,
  systemPreferences,
} from "electron";
import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { PythonBridge } from "./python-bridge";

// Bundled to CJS by esbuild, so the CommonJS __dirname global is available.
declare const __dirname: string;

let mainWindow: BrowserWindow | null = null;

// The iris icon, rendered by `npm run icon` (scripts/icon) into ui/build
const ICON_PATH = path.join(__dirname, "../build/icon.png");
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
    backgroundColor: "#121316",
    icon: ICON_PATH, // Windows and Linux; macOS takes the Dock icon below
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  mainWindow.once("ready-to-show", () => {
    mainWindow?.show();
  });

  // Links in replies open in the browser; the app window never navigates away
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    if (/^https?:/.test(url)) shell.openExternal(url);
    return { action: "deny" };
  });
  mainWindow.webContents.on("will-navigate", (event, url) => {
    if (url !== mainWindow?.webContents.getURL()) event.preventDefault();
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
 * Developer switch for the first-launch setup, which otherwise shows only
 * once per settings folder. IRIS_SETUP=1 starts like a first launch, with
 * settings and memory of its own and the speech models already on this Mac.
 * IRIS_SETUP=download also gives the core a home folder of its own, so the
 * models download as on a new Mac. It all lives in one temporary folder,
 * emptied when a test starts (a crashed run leaves nothing behind for long)
 * and removed on quit.
 */
const SETUP_TEST = process.env.IRIS_SETUP ?? "";
const coreEnv: NodeJS.ProcessEnv = {};
if (SETUP_TEST && SETUP_TEST !== "0") {
  const scratch = path.join(os.tmpdir(), "iris-setup-test");
  fs.rmSync(scratch, { recursive: true, force: true });
  fs.mkdirSync(scratch);
  app.setPath("userData", path.join(scratch, "userdata"));
  coreEnv.IRIS_MEMORY_DB = path.join(scratch, "memory.db");
  if (SETUP_TEST === "download") {
    const home = path.join(scratch, "home");
    fs.mkdirSync(home);
    coreEnv.HOME = home;
    coreEnv.HF_HOME = path.join(home, ".cache", "huggingface");
    coreEnv.IRIS_MODELS_DIR = "";
  }
  console.log(`[main] setup test (${SETUP_TEST}) in ${scratch}`);
  // Chromium still writes a few files after will-quit, so a small process
  // waits for this one to exit and then removes the folder
  app.on("will-quit", () => {
    if (process.platform === "win32") {
      fs.rmSync(scratch, { recursive: true, force: true });
      return;
    }
    const sweep = 'while kill -0 "$0" 2>/dev/null; do sleep 0.2; done; rm -rf "$1"';
    spawn("/bin/sh", ["-c", sweep, String(process.pid), scratch], {
      detached: true,
      stdio: "ignore",
    }).unref();
  });
}

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
  // A packaged app gets its icon from the bundle; in development the Dock
  // would otherwise show Electron's default
  const icon = nativeImage.createFromPath(ICON_PATH);
  if (process.platform === "darwin" && !icon.isEmpty()) app.dock?.setIcon(icon);

  const micStatus = await ensureMicrophoneAccess();
  const isPackaged = app.isPackaged;
  const projectRoot = isPackaged
    ? app.getAppPath()
    : path.resolve(__dirname, "../..");
  const resourcesPath = isPackaged ? process.resourcesPath : "";

  pythonBridge = new PythonBridge(projectRoot, isPackaged, resourcesPath, coreEnv);

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
  ipcMain.handle("is-packaged", () => app.isPackaged);
  ipcMain.handle("choose-folder", async (_event, title: string) => {
    const options = { title, properties: ["openDirectory" as const] };
    const result = mainWindow
      ? await dialog.showOpenDialog(mainWindow, options)
      : await dialog.showOpenDialog(options);
    return result.canceled ? null : result.filePaths[0];
  });

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
