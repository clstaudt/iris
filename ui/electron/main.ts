import {
  app,
  BrowserWindow,
  dialog,
  ipcMain,
  nativeImage,
  screen,
  shell,
  systemPreferences,
  type Rectangle,
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

// The full window, and presence: a small window with only the orb, always
// on top, for talking to Iris by voice while doing something else
const FULL_SIZE = { width: 1280, height: 800 };
const FULL_MIN = { width: 720, height: 480 };
const PRESENCE_SIZE = { width: 220, height: 220 };
const PRESENCE_MARGIN = 24;

/** The window's mode and where it was, remembered between launches. */
interface WindowState {
  presence?: boolean;
  full?: Rectangle;
  presenceAt?: { x: number; y: number };
}

let windowState: WindowState = {};

function windowStateFile() {
  return path.join(app.getPath("userData"), "window-state.json");
}

function loadWindowState(): WindowState {
  try {
    return JSON.parse(fs.readFileSync(windowStateFile(), "utf8")) as WindowState;
  } catch {
    return {};
  }
}

function saveWindowState() {
  try {
    fs.writeFileSync(windowStateFile(), JSON.stringify(windowState));
  } catch (err) {
    console.error("[main] could not save the window state:", err);
  }
}

/** The bounds moved and shrunk onto the screen nearest to them, so a window
 *  last seen on a monitor since unplugged comes back into view. */
function onScreen(bounds: Rectangle): Rectangle {
  const area = screen.getDisplayMatching(bounds).workArea;
  const width = Math.min(bounds.width, area.width);
  const height = Math.min(bounds.height, area.height);
  return {
    x: Math.min(Math.max(bounds.x, area.x), area.x + area.width - width),
    y: Math.min(Math.max(bounds.y, area.y), area.y + area.height - height),
    width,
    height,
  };
}

/** Where the presence window goes: where it was last, or the bottom right
 *  corner of the screen the window is on. */
function presenceBounds(): Rectangle {
  if (windowState.presenceAt) return onScreen({ ...windowState.presenceAt, ...PRESENCE_SIZE });
  const display = mainWindow
    ? screen.getDisplayMatching(mainWindow.getBounds())
    : screen.getPrimaryDisplay();
  const area = display.workArea;
  return {
    x: area.x + area.width - PRESENCE_SIZE.width - PRESENCE_MARGIN,
    y: area.y + area.height - PRESENCE_SIZE.height - PRESENCE_MARGIN,
    ...PRESENCE_SIZE,
  };
}

/** Where the full window goes: where it was last, or centered. */
function fullBounds(): Rectangle {
  if (windowState.full) return onScreen(windowState.full);
  const display = mainWindow
    ? screen.getDisplayMatching(mainWindow.getBounds())
    : screen.getPrimaryDisplay();
  const area = display.workArea;
  const width = Math.min(FULL_SIZE.width, area.width);
  const height = Math.min(FULL_SIZE.height, area.height);
  return {
    x: area.x + Math.round((area.width - width) / 2),
    y: area.y + Math.round((area.height - height) / 2),
    width,
    height,
  };
}

/** The window's behavior in each mode: presence floats above other windows
 *  at a fixed size, without window buttons. */
function styleWindow(win: BrowserWindow, presence: boolean) {
  win.setAlwaysOnTop(presence, "floating");
  win.setResizable(!presence);
  win.setMaximizable(!presence);
  win.setFullScreenable(!presence);
  if (process.platform === "darwin") win.setWindowButtonVisibility(!presence);
}

function setPresence(on: boolean) {
  const win = mainWindow;
  if (!win || on === !!windowState.presence) return;
  if (on) {
    // Out of full screen first; presence follows once the window is back
    if (win.isFullScreen()) {
      win.once("leave-full-screen", () => setPresence(true));
      win.setFullScreen(false);
      return;
    }
    windowState.full = win.getNormalBounds();
    if (win.isMaximized()) win.unmaximize();
    windowState.presence = true;
    win.setMinimumSize(PRESENCE_SIZE.width, PRESENCE_SIZE.height);
    styleWindow(win, true);
    win.setBounds(presenceBounds(), true);
  } else {
    const [x, y] = win.getPosition();
    windowState.presenceAt = { x, y };
    windowState.presence = false;
    styleWindow(win, false);
    win.setBounds(fullBounds(), true);
    win.setMinimumSize(FULL_MIN.width, FULL_MIN.height);
  }
  saveWindowState();
}

function createWindow() {
  windowState = loadWindowState();
  const presence = !!windowState.presence;
  const min = presence ? PRESENCE_SIZE : FULL_MIN;
  mainWindow = new BrowserWindow({
    title: "iris",
    ...(presence ? presenceBounds() : fullBounds()),
    minWidth: min.width,
    minHeight: min.height,
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
  if (presence) styleWindow(mainWindow, true);

  mainWindow.once("ready-to-show", () => {
    mainWindow?.show();
  });

  // Remember where the window is, in either mode
  mainWindow.on("moved", () => {
    if (!mainWindow || !windowState.presence) return;
    const [x, y] = mainWindow.getPosition();
    windowState.presenceAt = { x, y };
    saveWindowState();
  });
  mainWindow.on("close", () => {
    if (!mainWindow) return;
    if (windowState.presence) {
      const [x, y] = mainWindow.getPosition();
      windowState.presenceAt = { x, y };
    } else {
      windowState.full = mainWindow.getNormalBounds();
    }
    saveWindowState();
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
  // Read once by the preload, so the first frame already has the right layout
  ipcMain.on("get-presence", (event) => {
    event.returnValue = !!windowState.presence;
  });
  ipcMain.handle("set-presence", (_event, on: boolean) => setPresence(!!on));
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
