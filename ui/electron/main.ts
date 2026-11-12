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
let presenceWindow: BrowserWindow | null = null;
let quitting = false;

// The iris icon, rendered by `npm run icon` (scripts/icon) into ui/build
const ICON_PATH = path.join(__dirname, "../build/icon.png");
let pythonBridge: PythonBridge | null = null;

// The full window, and presence: Iris as only its orb, a round window of
// its own above other windows, for talking to Iris by voice while doing
// something else
const FULL_SIZE = { width: 1280, height: 800 };
const FULL_MIN = { width: 720, height: 480 };
// The orb's canvas: its ring and the glow that fades out before the edge
const PRESENCE_SIZE = { width: 112, height: 112 };
const PRESENCE_MARGIN = 24;
// A press on the orb that moves less than this is a click, not a drag (px)
const DRAG_THRESHOLD = 4;
// How long a window may take to hand the orb over (ms); the switch goes on
// without it after that
const HANDOVER_TIMEOUT = 500;

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

/** The presence window's outline as rows of pixels, for Linux, where the
 *  window is cut to the circle: clicks outside it go to the windows below,
 *  and the circle still shows without a compositor for transparency. */
function circleShape(size: number): Rectangle[] {
  const r = size / 2;
  const rows: Rectangle[] = [];
  for (let y = 0; y < size; y++) {
    const dy = y + 0.5 - r;
    const x = Math.round(r - Math.sqrt(Math.max(0, r * r - dy * dy)));
    rows.push({ x, y, width: size - 2 * x, height: 1 });
  }
  return rows;
}

function loadPage(win: BrowserWindow, view?: "presence") {
  if (process.env.VITE_DEV_SERVER_URL) {
    const url = new URL(process.env.VITE_DEV_SERVER_URL);
    if (view) url.searchParams.set("window", view);
    win.loadURL(url.toString());
  } else {
    const query: Record<string, string> = view ? { window: view } : {};
    win.loadFile(path.join(__dirname, "../dist/index.html"), { query });
  }
}

/** Presence: a transparent, frameless window with only the round orb,
 *  floating above other windows. It is made once and kept, hidden while the
 *  full window shows, so a switch never waits for a page to load. */
function createPresenceWindow(): BrowserWindow {
  const win = new BrowserWindow({
    title: "iris",
    ...presenceBounds(),
    show: false,
    frame: false,
    transparent: true,
    backgroundColor: "#00000000",
    hasShadow: false,
    roundedCorners: false,
    thickFrame: false,
    resizable: false,
    maximizable: false,
    minimizable: false,
    fullscreenable: false,
    alwaysOnTop: true,
    // The first click on the orb counts, even while another app is active
    acceptFirstMouse: true,
    icon: ICON_PATH,
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      // It draws the orb while still hidden or fully transparent at a switch
      backgroundThrottling: false,
    },
  });
  win.setAlwaysOnTop(true, "floating");
  // Only the orb takes the pointer. Linux cuts the window to the circle;
  // elsewhere the page tells whether the pointer is over the orb, and until
  // it is, clicks go through to the windows below.
  if (process.platform === "linux") win.setShape(circleShape(PRESENCE_SIZE.width));
  else win.setIgnoreMouseEvents(true, { forward: true });
  // Closing presence closes Iris's window, as closing the full window does
  win.on("close", (event) => {
    if (quitting || !mainWindow) return;
    event.preventDefault();
    mainWindow.close();
  });
  loadPage(win, "presence");
  return win;
}

function loaded(win: BrowserWindow): Promise<void> {
  if (!win.webContents.isLoading()) return Promise.resolve();
  return new Promise((resolve) => win.webContents.once("did-finish-load", () => resolve()));
}

/** Send a window's page a message and wait for its reply, or for the
 *  timeout (null). */
function ask<T>(win: BrowserWindow, channel: string, payload?: unknown): Promise<T | null> {
  return new Promise((resolve) => {
    const reply = `${channel}-reply`;
    const finish = (value: T | null) => {
      clearTimeout(timer);
      ipcMain.removeListener(reply, listener);
      resolve(value);
    };
    const listener = (event: Electron.IpcMainEvent, value: T) => {
      if (event.sender === win.webContents) finish(value ?? null);
    };
    const timer = setTimeout(() => finish(null), HANDOVER_TIMEOUT);
    ipcMain.on(reply, listener);
    win.webContents.send(channel, payload);
  });
}

/**
 * Put one window on screen in place of the other, without a moment where the
 * orb is missing or stale: the window going away hands over the orb's
 * motion, the window coming in draws with it and reports back, and only then
 * does it show and the other hide.
 */
async function handOver(from: BrowserWindow, to: BrowserWindow) {
  await Promise.all([loaded(from), loaded(to)]);
  const handoff = await ask(from, "orb-release");
  if (process.platform === "darwin") {
    // Shown fully transparent first, so the first frame anyone sees is one
    // drawn with the handed-over motion
    to.setOpacity(0);
    to.show();
    await ask(to, "orb-adopt", handoff);
    to.setOpacity(1);
  } else {
    // Drawn while still hidden (the page is not throttled), then shown
    await ask(to, "orb-adopt", handoff);
    to.show();
  }
  from.hide();
  from.webContents.send("orb-rest");
}

let switching: Promise<void> = Promise.resolve();

/** Switch between the full window and presence, one switch at a time. */
function setPresence(on: boolean): Promise<void> {
  switching = switching
    .then(() => switchPresence(on))
    .catch((err) => console.error("[main] could not switch presence:", err));
  return switching;
}

async function switchPresence(on: boolean) {
  const full = mainWindow;
  const orb = presenceWindow;
  if (!full || !orb) return;
  // Which one is on screen; neither while Iris is still opening
  const shown = orb.isVisible() ? true : full.isVisible() ? false : null;
  if (on === shown) return;
  if (on) {
    // Out of full screen first, or the hidden window would leave an empty Space
    if (full.isFullScreen()) {
      await new Promise<void>((resolve) => {
        full.once("leave-full-screen", () => resolve());
        full.setFullScreen(false);
      });
    }
    if (shown === false) windowState.full = full.getNormalBounds();
    windowState.presence = true;
    // Hidden, the full window's page still runs the conversation
    full.webContents.setBackgroundThrottling(false);
    orb.setBounds(presenceBounds());
    await handOver(full, orb);
  } else {
    if (shown) {
      const [x, y] = orb.getPosition();
      windowState.presenceAt = { x, y };
    }
    windowState.presence = false;
    await handOver(orb, full);
    full.webContents.setBackgroundThrottling(true);
    if (process.platform !== "linux") orb.setIgnoreMouseEvents(true, { forward: true });
  }
  saveWindowState();
}

function createWindow() {
  windowState = loadWindowState();
  const presence = !!windowState.presence;
  mainWindow = new BrowserWindow({
    title: "iris",
    ...fullBounds(),
    minWidth: FULL_MIN.width,
    minHeight: FULL_MIN.height,
    show: false,
    titleBarStyle: "hiddenInset",
    trafficLightPosition: { x: 16, y: 16 },
    backgroundColor: "#121316",
    icon: ICON_PATH, // Windows and Linux; macOS takes the Dock icon below
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      // Hidden while presence is up, the page still runs the conversation
      backgroundThrottling: !presence,
    },
  });
  presenceWindow = createPresenceWindow();

  // Opened as presence, the orb shows and the full window waits hidden
  mainWindow.once("ready-to-show", () => {
    if (windowState.presence) setPresence(true);
    else mainWindow?.show();
  });

  // Remember where the windows are
  mainWindow.on("close", () => {
    if (!mainWindow) return;
    if (presenceWindow?.isVisible()) {
      const [x, y] = presenceWindow.getPosition();
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

  loadPage(mainWindow);
  if (process.env.VITE_DEV_SERVER_URL) mainWindow.webContents.openDevTools({ mode: "detach" });

  mainWindow.on("closed", () => {
    mainWindow = null;
    presenceWindow?.destroy();
    presenceWindow = null;
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

/** Whether the app may use the microphone; only macOS asks. */
function microphoneStatus(): string {
  if (process.platform !== "darwin") return "granted";
  return systemPreferences.getMediaAccessStatus("microphone");
}

/**
 * The Python child process opens the microphone, but macOS attributes that
 * access to this app. Nothing in Electron itself touches the mic, so the
 * system would never prompt — and an app that never asks does not appear in
 * System Settings > Privacy & Security > Microphone. Asking here registers
 * the app with TCC and triggers the prompt. The window asks: the
 * walkthrough at its microphone step, a later launch as it starts.
 */
async function askMicrophoneAccess(): Promise<string> {
  const status = microphoneStatus();
  console.log(`[main] microphone access status: ${status}`);
  if (status !== "not-determined") return status;

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

  ipcMain.handle("mic-status", () => microphoneStatus());
  ipcMain.handle("ask-microphone", () => askMicrophoneAccess());
  // Read once by the preload, so the first frame already has the right layout
  ipcMain.on("get-presence", (event) => {
    event.returnValue = !!windowState.presence;
  });
  ipcMain.handle("set-presence", (_event, on: boolean) => setPresence(!!on));
  // What the orb in presence shows comes from the conversation window
  ipcMain.on("orb-inputs", (event, inputs) => {
    if (event.sender !== mainWindow?.webContents) return;
    presenceWindow?.webContents.send("orb-inputs", inputs);
  });
  ipcMain.on("presence-command", (event, command: string) => {
    if (event.sender !== presenceWindow?.webContents) return;
    mainWindow?.webContents.send("presence-command", command);
  });
  ipcMain.on("presence-hover", (event, over: boolean) => {
    const win = presenceWindow;
    if (!win || event.sender !== win.webContents || process.platform === "linux") return;
    win.setIgnoreMouseEvents(!over, { forward: true });
  });
  // Dragging the orb: the window follows the pointer as the system reports
  // it, so it never lags behind its own movement
  let drag: { cursor: Electron.Point; x: number; y: number; moved: boolean } | null = null;
  ipcMain.handle("presence-drag", (event, phase: "start" | "move" | "end") => {
    const win = presenceWindow;
    if (!win || event.sender !== win.webContents) return false;
    const cursor = screen.getCursorScreenPoint();
    if (phase === "start") {
      const [x, y] = win.getPosition();
      drag = { cursor, x, y, moved: false };
      return false;
    }
    if (!drag) return false;
    const dx = cursor.x - drag.cursor.x;
    const dy = cursor.y - drag.cursor.y;
    if (!drag.moved && Math.hypot(dx, dy) >= DRAG_THRESHOLD) drag.moved = true;
    if (drag.moved) win.setPosition(drag.x + dx, drag.y + dy);
    const moved = drag.moved;
    if (phase === "end") {
      drag = null;
      if (moved) {
        const [x, y] = win.getPosition();
        windowState.presenceAt = { x, y };
        saveWindowState();
      }
    }
    return moved;
  });
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
  quitting = true;
  pythonBridge?.kill();
});
