// End-to-end test of the app's UI against the real Python core.
//
// The built UI (ui/dist) runs in headless Chrome; window.iris, which Electron's
// preload provides in the app, is bridged to a core started for the test,
// through the app's own bridge (electron/python-bridge.ts), which reports a
// core that stops and starts it again. The core gets a home folder of its
// own, so the tester's personas, memory and settings are left alone; the
// speech models already on this computer are used. The app opens as on a
// first launch, with empty settings: the test goes through the walkthrough,
// which starts Iris. Presence, the orb's round window of its own, is a second
// page, with the orb relayed between the two as the app's main process does.
// Its microphone and speaker are a virtual room (irisdev.e2e_core), which
// holds only Iris's own voice coming back: the real room is never heard, and
// nothing sounds on the speakers. Iris sleeps until a wake word nobody says.
// The app is restarted with a new core near the end. Beside the model server
// it is given, the test runs a small one of its own, with fixed replies, to
// switch servers to.
//
//   npm run build && npm run e2e
//
// Environment:
//   E2E_MODEL   language model to talk to (default: the first the server lists
//               with tools)
//   E2E_HOST    model server URL (default: the core's default)
//   E2E_PYTHON  Python to run the core with (default: <repo>/.venv/bin/python)
//   E2E_ONLY    run only the steps whose name contains this text, or one of
//               several split by "|"; the walkthrough's steps, which start
//               Iris and choose the model, always run
//   E2E_HEADED  set to watch the browser

import { execFileSync } from "node:child_process";
import fs from "node:fs";
import http from "node:http";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright-core";

import { PythonBridge } from "../electron/python-bridge.ts";

const UI = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const REPO = path.resolve(UI, "..");
const DIST = path.join(UI, "dist");
const OUT = path.join(UI, "e2e", "output");
const WAKE_WORD = "Zanzibar";

if (!fs.existsSync(path.join(DIST, "index.html"))) {
  console.error("ui/dist is missing: run `npm run build` first");
  process.exit(2);
}
fs.rmSync(OUT, { recursive: true, force: true });
fs.mkdirSync(OUT, { recursive: true });

// ---------------------------------------------------------------- the core

const home = fs.mkdtempSync(path.join(os.tmpdir(), "iris-e2e-"));
// Where the test breaks the room's speaker, as the core's stand-in reads it
// (irisdev/e2e_core.py)
const AUDIO_FAULTS = path.join(home, ".e2e-audio.json");
// The steps keep setups' homes under that name; the core's is this one
const coreHome = home;
const realHome = os.homedir();
const python =
  process.env.E2E_PYTHON ??
  path.join(REPO, ".venv", process.platform === "win32" ? "Scripts/python.exe" : "bin/python");

/** Every event the core pushed, and the bridge told, in order. */
const events = [];
let pushEvent = () => {};
let bridge;

/** Start a core through the app's own bridge, which starts it again when it
 *  stops; a restart of the app starts another with the same home and memory. */
function startCore(log) {
  const coreLog = fs.createWriteStream(path.join(OUT, log));
  bridge = new PythonBridge(
    REPO,
    false,
    "",
    {
      HOME: home,
      USERPROFILE: home,
      HF_HOME: process.env.HF_HOME ?? path.join(realHome, ".cache", "huggingface"),
      IRIS_MODELS_DIR: process.env.IRIS_MODELS_DIR ?? path.join(realHome, ".iris", "models"),
      IRIS_MEMORY_DB: path.join(home, "memory.db"),
    },
    { command: python, args: ["-m", "irisdev.e2e_core"] }
  );
  bridge.onLog = (line) => coreLog.write(line + "\n");
  bridge.onEvent = (type, data) => {
    events.push({ type, data, at: Date.now() });
    pushEvent(type, data);
  };
}

/** Stop the core as quitting the app does, and start a new one. */
async function restartCore() {
  await Promise.race([rpc("assistant.stop").catch(() => {}), quiet(5000)]);
  const pid = bridge.pid;
  bridge.kill();
  await until("the core to end", () => {
    try {
      process.kill(pid, 0);
      return false;
    } catch {
      return true;
    }
  });
  startCore("core-2.log");
}

function rpc(method, params = {}) {
  return bridge.call(method, params);
}

startCore("core.log");

// ------------------------------------------------- a model server of its own

// A server the test runs itself, with the OpenAI-compatible API, so that
// switching servers is tested whatever E2E_HOST is, and what a model is sent
// can be seen. Its models give one fixed reply: markdown, with an emoji and
// HTML a page could carry. One of them says it sees images, unless the test
// wants a server where none can see.
const STUB_MODELS = [
  { id: "stub-large" },
  { id: "stub-small" },
  { id: "stub-vision", capabilities: ["vision"] },
];
let stubSees = true;
const STUB_REPLY =
  "Here is **the stub's answer** \u{1F642}\n\n" +
  "- first point\n- second point\n\n" +
  "See [the example](https://example.com/page) for more.\n\n" +
  '<img src="x" onerror="window.__injected = true"> <a href="javascript:window.__injected = true">a trap</a>';
/** The chat requests the stub server was sent */
const stubRequests = [];
const stub = http.createServer((request, response) => {
  let body = "";
  request.on("data", (part) => (body += part));
  request.on("end", () => {
    const route = request.url.split("?")[0];
    if (request.method === "GET" && route === "/v1/models") {
      response.writeHead(200, { "Content-Type": "application/json" });
      const models = STUB_MODELS.filter((m) => stubSees || !m.capabilities?.includes("vision"));
      response.end(JSON.stringify({ object: "list", data: models.map((m) => ({ ...m, object: "model" })) }));
    } else if (request.method === "POST" && route === "/v1/chat/completions") {
      const payload = JSON.parse(body || "{}");
      if (!payload.stream) {
        // Filing memory away: it is tried again on the real server
        response.writeHead(503).end();
        return;
      }
      stubRequests.push(payload);
      response.writeHead(200, { "Content-Type": "text/event-stream" });
      for (const piece of STUB_REPLY.match(/[\s\S]{1,12}/g)) {
        response.write(`data: ${JSON.stringify({ choices: [{ index: 0, delta: { content: piece } }] })}\n\n`);
      }
      response.end("data: [DONE]\n\n");
    } else {
      response.writeHead(404).end();
    }
  });
});
await new Promise((resolve) => stub.listen(0, "127.0.0.1", resolve));
const stubUrl = `http://127.0.0.1:${stub.address().port}/v1`;

// ------------------------------------------------------------- the browser

const MIME = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css" };
const server = http.createServer((request, response) => {
  // Presence's window is the same page, as /?window=presence
  const route = request.url.split("?")[0];
  const file = path.join(DIST, route === "/" ? "index.html" : route);
  if (route === "/favicon.ico") {
    response.writeHead(204).end();
    return;
  }
  if (!file.startsWith(DIST) || !fs.existsSync(file) || !fs.statSync(file).isFile()) {
    response.writeHead(404).end();
    return;
  }
  response.writeHead(200, { "Content-Type": MIME[path.extname(file)] ?? "application/octet-stream" });
  fs.createReadStream(file).pipe(response);
});
await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
const url = `http://127.0.0.1:${server.address().port}/`;

const browser = await chromium.launch({ channel: "chrome", headless: !process.env.E2E_HEADED });
const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
const pageErrors = [];
page.on("pageerror", (error) => pageErrors.push(String(error)));
page.on("console", (message) => {
  if (message.type() === "error") pageErrors.push(message.text());
});

/** Every request the page made of the core, with what it answered. */
const calls = [];
await page.exposeFunction("__irisRpc", async (method, params) => {
  const call = { method, params, at: Date.now() };
  calls.push(call);
  call.result = await rpc(method, params);
  return call.result;
});
/** What the page asked of the window: presence on or off, in order. */
const presence = [];
/** Presence's window: a page of its own, opened when first needed */
let presencePage = null;

/** At each switch the orb passes from the window that hides to the one that
 *  shows, as main.ts hands it over. */
async function handOver(on) {
  if (!presencePage) return;
  const [from, to] = on ? [page, presencePage] : [presencePage, page];
  const handoff = await from.evaluate(() => window.__irisOrb.release());
  await to.evaluate((h) => void window.__irisOrb.adopt(h), handoff);
  await from.evaluate(() => void window.__irisOrb.rest());
}

await page.exposeFunction("__irisPresence", async (on) => {
  presence.push(on);
  await handOver(on);
});
// What the orb shows goes from the conversation to presence
await page.exposeFunction("__irisOrbInputs", (inputs) =>
  presencePage?.evaluate((i) => window.__irisOrb.inputs(i), inputs).catch(() => {})
);

/** window.iris as Electron's preload gives it, for both windows. */
function stubIris({ settings }) {
    // Seeded once; a reload finds what the app saved
    if (localStorage.getItem("iris.settings") === null) {
      localStorage.setItem("iris.settings", JSON.stringify(settings));
    }
    const listeners = [];
    window.__irisEvent = (type, data) => listeners.forEach((listener) => listener(type, data));
    // The orb between this window and the other, relayed by the test
    const orb = { handlers: null, inputs: [], commands: [] };
    window.__irisOrb = {
      release: () => orb.handlers?.release() ?? null,
      adopt: (handoff) => orb.handlers?.adopt(handoff),
      rest: () => orb.handlers?.rest(),
      inputs: (inputs) => orb.inputs.forEach((listener) => listener(inputs)),
      command: (command) => orb.commands.forEach((listener) => listener(command)),
    };
    // Links go to the system browser, as main.ts routes window.open there
    window.__opened = [];
    window.open = (target) => {
      window.__opened.push(String(target));
      return null;
    };
    window.iris = {
      rpc: (method, params) => window.__irisRpc(method, params ?? {}),
      onEvent: (callback) => {
        listeners.push(callback);
        return () => listeners.splice(listeners.indexOf(callback), 1);
      },
      micStatus: async () => sessionStorage.getItem("e2e.mic") ?? "granted",
      askMicrophone: async () => sessionStorage.getItem("e2e.mic") ?? "granted",
      isPackaged: async () => false,
      chooseFolder: async () => null,
      presence: false,
      setPresence: (on) => window.__irisPresence(on),
      onOrb: (handlers) => {
        orb.handlers = handlers;
        return () => {
          if (orb.handlers === handlers) orb.handlers = null;
        };
      },
      sendOrbInputs: (inputs) => window.__irisOrbInputs?.(inputs),
      onOrbInputs: (listener) => {
        orb.inputs.push(listener);
        return () => orb.inputs.splice(orb.inputs.indexOf(listener), 1);
      },
      presenceCommand: (command) => window.__irisPresenceCommand?.(command),
      onPresenceCommand: (listener) => {
        orb.commands.push(listener);
        return () => orb.commands.splice(orb.commands.indexOf(listener), 1);
      },
      presenceHover: () => {},
      // Never a drag: a press and release is a click
      presenceDrag: async () => false,
      platform: "test",
    };
}
// A first launch: the walkthrough sets the server, the model and the wake word
await page.addInitScript(stubIris, { settings: {} });
pushEvent = (type, data) => {
  page.evaluate(([t, d]) => window.__irisEvent?.(t, d), [type, data]).catch(() => {});
};

// ------------------------------------------------------------------ helpers

const results = [];

async function step(name, body) {
  const only = process.env.E2E_ONLY?.toLowerCase().split("|");
  const always = ["first run", "the app starts", "language model: the server's models"];
  if (only && !only.some((part) => name.toLowerCase().includes(part)) && !always.some((s) => name.startsWith(s))) {
    return;
  }
  const started = Date.now();
  try {
    await body();
    results.push({ name, ok: true });
    console.log(`  ok    ${name} (${((Date.now() - started) / 1000).toFixed(1)} s)`);
  } catch (error) {
    const shot = path.join(OUT, `${results.length + 1}-${name.replace(/[^a-z0-9]+/gi, "-")}.png`);
    await page.screenshot({ path: shot }).catch(() => {});
    const states = events.filter((e) => e.at >= started && e.type === "state").map((e) => e.data.state);
    const errors = events.filter((e) => e.at >= started && e.type === "error").map((e) => e.data.message);
    const detail =
      ` [states: ${states.join(" > ") || "none"}]` +
      (errors.length ? ` [errors: ${errors.join(" | ")}]` : "") +
      (pageErrors.length ? ` [page: ${pageErrors.slice(-3).join(" | ")}]` : "");
    results.push({ name, ok: false, error: String(error.message ?? error).split("\n")[0] + detail, shot });
    console.log(`  FAIL  ${name}\n        ${results.at(-1).error}\n        ${shot}`);
  }
}

function check(condition, message) {
  if (!condition) throw new Error(message);
}

const quiet = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/** A one-page PDF holding a line of text, made here so the test needs no
 *  files. */
function pdf(line) {
  const text = `BT /F1 20 Tf 72 700 Td (${line}) Tj ET`;
  const objects = [
    "<< /Type /Catalog /Pages 2 0 R >>",
    "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
    "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R " +
      "/Resources << /Font << /F1 5 0 R >> >> >>",
    `<< /Length ${text.length} >>\nstream\n${text}\nendstream`,
    "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
  ];
  let out = "%PDF-1.4\n";
  const offsets = objects.map((body, i) => {
    const at = out.length;
    out += `${i + 1} 0 obj\n${body}\nendobj\n`;
    return at;
  });
  const xref = out.length;
  out += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n`;
  out += offsets.map((at) => `${String(at).padStart(10, "0")} 00000 n \n`).join("");
  out += `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF\n`;
  return Buffer.from(out, "latin1");
}

/** A picture of a word in large white letters on a colour, drawn by the
 *  browser, as a PNG or a JPEG. */
async function picture(word, color, type) {
  const data = await page.evaluate(
    async ([word, color, type]) => {
      const canvas = document.createElement("canvas");
      canvas.width = 640;
      canvas.height = 360;
      const c = canvas.getContext("2d");
      c.fillStyle = color;
      c.fillRect(0, 0, canvas.width, canvas.height);
      c.fillStyle = "white";
      c.font = "bold 120px sans-serif";
      c.textAlign = "center";
      c.textBaseline = "middle";
      c.fillText(word, canvas.width / 2, canvas.height / 2);
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, type, 0.9));
      const bytes = new Uint8Array(await blob.arrayBuffer());
      let binary = "";
      for (const byte of bytes) binary += String.fromCharCode(byte);
      return btoa(binary);
    },
    [word, color, type]
  );
  return Buffer.from(data, "base64");
}

/** Wait until the page function returns something truthy. */
async function until(what, fn, timeout = 10_000) {
  const deadline = Date.now() + timeout;
  let last;
  while (Date.now() < deadline) {
    last = await fn();
    if (last) return last;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error(`Timed out waiting for ${what}`);
}

/** Wait for an event of this type pushed after `since` that passes `test`. */
function eventAfter(type, since, test = () => true, timeout = 10_000) {
  return until(
    `the ${type} event`,
    () => events.find((e) => e.at >= since && e.type === type && test(e.data)),
    timeout
  );
}

const stage = (label) =>
  page.locator("button.stage").filter({ has: page.locator(".label", { hasText: new RegExp(`^${label}$`) }) });
const stageValue = (label) => stage(label).locator(".value").innerText();
const inspector = page.locator("aside");
const option = (text) => inspector.locator(".s-option", { hasText: text });

async function openStage(label, title = label) {
  if (!(await stage(label).count())) await openSettings();
  if ((await stage(label).getAttribute("aria-expanded")) !== "true") await stage(label).click();
  await inspector.locator("h2", { hasText: title }).waitFor({ timeout: 5000 });
}

/** Out of presence, if a step left the window there. */
/** Presence's window, the orb alone: index.html?window=presence. */
async function openPresence() {
  if (presencePage) return presencePage;
  presencePage = await browser.newPage({ viewport: { width: 220, height: 220 } });
  presencePage.on("pageerror", (error) => pageErrors.push(`presence: ${error}`));
  await presencePage.exposeFunction("__irisRpc", (method, params) => rpc(method, params));
  await presencePage.exposeFunction("__irisPresence", () => {});
  // What presence asks (leave, pause) goes to the conversation window
  await presencePage.exposeFunction("__irisPresenceCommand", (command) =>
    page.evaluate((c) => window.__irisOrb.command(c), command)
  );
  await presencePage.addInitScript(stubIris, { settings: {} });
  await presencePage.goto(`${url}?window=presence`);
  await presencePage.locator('[aria-label="iris"]').waitFor({ timeout: 5000 });
  return presencePage;
}

async function leavePresence() {
  if (presence.at(-1) !== true) return;
  await presencePage.keyboard.press("Escape");
  await until("the conversation", () => presence.at(-1) === false, 5000);
}

async function openSettings(pageName = "Pipeline") {
  await leavePresence();
  if (!(await page.locator('nav[aria-label="Settings"]').count())) {
    await page.locator('button[title="Settings"]').click();
  }
  await page.locator('nav[aria-label="Settings"] button', { hasText: pageName }).click();
}

const config = () => events.findLast((e) => e.type === "config")?.data;
const exactly = (text) => new RegExp(`^${text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}$`);
const walkthroughStep = (name) => page.locator(`section[data-step="${name}"]`);
const footerButton = (text) => page.locator("footer button", { hasText: text });
const settings = () => page.evaluate(() => JSON.parse(localStorage.getItem("iris.settings") ?? "{}"));

/** Back to the conversation from the settings or the memory panel. */
async function toConversation() {
  await leavePresence();
  const back = page.locator('nav[aria-label="Settings"] button', { hasText: "Conversation" });
  if (await back.count()) await back.click();
  const close = page.locator('button[title="Close memory"]');
  if (await close.count()) await close.click();
  await page.locator("textarea").waitFor({ timeout: 5000 });
}

/** What was typed into the message box this run, to find it again. */
const asked = [];

async function ask(text) {
  await toConversation().catch(() => {});
  await page.locator("textarea").fill(text);
  await page.locator("textarea").press("Enter");
  asked.push(text);
  return Date.now();
}

const voiceDone = (timeout = 60_000) =>
  until("the voice to finish", () => rpc("app.info").then((info) => info.state !== "speaking"), timeout);

/** Seconds of sound the virtual speaker played from a moment on (until
 *  another, or now): what was heard, not what the core says it played. */
const heard = async (since, until) => (await rpc("e2e.speaker", { since, until })).seconds;

// Sound enough to be a voice: a preview's sentence gives seconds of it, and
// so does a short reply; a click or a breath, a few hundredths
const SPOKEN = 0.5;

/** Wait for a voice to reach the speaker from a moment on: SPOKEN seconds of
 *  sound at least. */
async function checkHeard(since, what, timeout = 60_000) {
  let seconds = 0;
  await until(`${what} to be heard`, async () => (seconds = await heard(since)) >= SPOKEN, timeout).catch(() => {
    throw new Error(`${what} reached the speaker as ${seconds.toFixed(2)} s of sound`);
  });
}

/** That the speaker played no sound from a moment on. */
async function checkSilent(since) {
  const seconds = await heard(since);
  check(seconds === 0, `the speaker played ${seconds.toFixed(2)} s of sound after Stop`);
}

/** The model's streamed replies begun since a moment, its reasoning included,
 *  which the app is never shown: when their first and last lines came. */
async function modelOutput(since) {
  const streams = (await rpc("e2e.model")).streams.filter((s) => s.first >= since);
  if (!streams.length) return null;
  return {
    first: Math.min(...streams.map((s) => s.first)),
    last: Math.max(...streams.map((s) => s.last)),
    lines: streams.reduce((sum, s) => sum + s.lines, 0),
  };
}

// The longest the model may stream nothing before the reply is given up on,
// as the core gives up on a server silent for longer
const MODEL_SILENCE = 60_000;

/** Wait for something a reply asked for at `since` brings (its first words,
 *  a tool), however long the model thinks before: a reasoning model streams
 *  minutes of reasoning first against a slow server. It is waited for while
 *  the model streams; a model that says nothing at all, or goes silent
 *  before it, fails. */
async function afterThinking(what, since, found) {
  await until("the model's first output", () => modelOutput(since), 180_000);
  return until(what, async () => {
    const got = await found();
    if (got) return got;
    const output = await modelOutput(since);
    const silent = Date.now() - output.last;
    check(silent < MODEL_SILENCE, `the model streamed ${output.lines} lines, then nothing for ${Math.round(silent / 1000)} s`);
    return null;
  }, 900_000);
}

/** An event of a reply asked for at `since` (its end, its timing), waited for
 *  as long as the model streams (see afterThinking). */
const replyEvent = (type, since) =>
  afterThinking(`the ${type} event`, since, () => events.find((e) => e.at >= since && e.type === type));

// Hugging Face as the core's stand-in has it (irisdev/e2e_core.py): the
// token it takes, and what the account did on the website
const CLONE_TOKEN = "hf_e2e_test_token";
function huggingFace(changes) {
  const file = path.join(home, ".e2e-huggingface.json");
  const state = fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, "utf8")) : {};
  fs.writeFileSync(file, JSON.stringify({ ...state, ...changes }));
}

/** Speech to clone a voice from: one of Pocket TTS's own voices reading a
 *  few lines, written to a WAV file by the core's Python (nothing plays). */
function speechSample() {
  const file = path.join(home, "sample.wav");
  const script = [
    "import sys",
    "import soundfile as sf",
    "import torch",
    "from pocket_tts import TTSModel",
    "model = TTSModel.load_model(language='english')",
    "state = model.get_state_for_audio_prompt('alba')",
    "lines = ['Have you ever noticed how different a familiar street looks early in the morning?',",
    "         'The shops are still closed, and the only sounds are a few birds.',",
    "         'I like to walk down to the river then, and watch the light move across the water.']",
    "audio = torch.cat([model.generate_audio(state, line) for line in lines])",
    "sf.write(sys.argv[1], audio.numpy(), model.sample_rate)",
  ].join("\n");
  execFileSync(python, ["-c", script, file], { stdio: "ignore" });
  return fs.readFileSync(file);
}

// -------------------------------------------------------------------- steps

console.log(`Iris UI against the core (${python})`);
let model = process.env.E2E_MODEL ?? "";

try {
  await page.goto(url);

  // The address the core keeps for the test's server: .../v1 for one with
  // the OpenAI-compatible API, the root for Ollama
  let serverHost = "";

  await step("first run: the walkthrough welcomes a new user, and nothing starts yet", async () => {
    await page.getByText("Welcome to Iris").waitFor({ timeout: 30_000 });
    check(!(await page.locator("textarea").count()), "the conversation shows before the walkthrough");
    await footerButton("Get started").click();
    await walkthroughStep("server").waitFor({ timeout: 5000 });
    const started = events.find((e) => e.type === "state" && e.data.state === "starting");
    check(!started, "Iris started before the walkthrough was done");
  });

  await step("first run: a server typed without /v1 is kept as the core resolves it, and a model chosen", async () => {
    const models = page.locator('[aria-label="Model"] .s-option');
    if (process.env.E2E_HOST) {
      // As a user might type it: an OpenAI-compatible server without /v1
      serverHost = (await rpc("llm.probe", { url: process.env.E2E_HOST })).host;
      const typed = process.env.E2E_HOST.replace(/\/v1\/?$/, "");
      await page.locator("button.s-btn", { hasText: "Add server" }).click();
      await page.locator('input[aria-label="Server address"]').fill(typed);
      await page.locator('input[aria-label="Server address"]').press("Enter");
      await until(
        "the server to be chosen",
        async () =>
          (await page.locator(".server.active .name").getAttribute("title").catch(() => null)) === serverHost,
        30_000
      );
    } else {
      // This computer's server, chosen as the first that answers
      await page.locator(".server.active").waitFor({ timeout: 30_000 });
    }
    await until("the server's models", () => models.count(), 60_000);
    if (!model) {
      model = (await page.locator('[aria-label="Model"] .s-option.active .name').innerText()).trim();
    }
    await models.filter({ has: page.locator(".name", { hasText: exactly(model) }) }).click();
    await page.locator('[aria-label="Model"] .s-option.active', { hasText: model }).waitFor({ timeout: 5000 });
    await footerButton("Continue").click();
  });

  await step("first run: the speech models download in the background, the voices first", async () => {
    await walkthroughStep("models").waitFor({ timeout: 5000 });
    await until("the speech models", () => page.locator("[data-model]").count(), 10_000);
    const ids = await page.locator("[data-model]").evaluateAll((rows) => rows.map((row) => row.dataset.model));
    check(ids[0] === "kyutai" && ids[1] === "kokoro", `the voices do not come first: ${ids.join(", ")}`);
    check(ids.includes("turn") && ids.includes("stt"), `models missing: ${ids.join(", ")}`);
    // On this computer they are there already, or arrive with their bytes
    await until(
      "every model to be in",
      async () => (await page.locator("[data-model]:not([data-state=done])").count()) === 0,
      900_000
    );
    for (const update of events.filter((e) => e.type === "setup.models")) {
      for (const m of update.data.models) {
        check(m.state !== "downloading" || !m.total || m.done <= m.total, `${m.id}: ${m.done} of ${m.total} bytes`);
      }
    }
    await footerButton("Continue").click();
  });

  await step("first run: the microphone's sound check shows its level, and closes", async () => {
    await walkthroughStep("microphone").waitFor({ timeout: 5000 });
    await page.locator('select[aria-label="Microphone"]').waitFor({ timeout: 10_000 });
    const since = Date.now();
    // A computer without a microphone says so instead
    await until(
      "the microphone's level",
      async () =>
        events.some((e) => e.at >= since && e.type === "mic.check") ||
        (await page.getByText("The microphone could not be opened").count()),
      10_000
    );
    check(await page.locator(".s-meter i").count(), "no level meter");
    await footerButton("Continue").click();
    await walkthroughStep("voice").waitFor({ timeout: 5000 });
    await quiet(800);
    const late = events.filter((e) => e.type === "mic.check" && e.at > Date.now() - 400);
    check(late.length === 0, "the sound check goes on after its step");
  });

  await step("first run: a persona's voice plays before Iris runs", async () => {
    await until("the personas", async () => (await page.locator(".persona").count()) >= 3);
    const since = Date.now();
    await page.locator('button[aria-label="Play HAL"]').click();
    await eventAfter("audio.output", since, (data) => data.rms > 0, 90_000);
    await eventAfter("state", since, (data) => data.state === "idle", 60_000);
    check(!events.some((e) => e.at >= since && e.type === "error"), "the preview failed");
    await checkHeard(since, "HAL's voice");
    await page.locator(".persona", { hasText: "Iris" }).locator("button.pick").click();
    await footerButton("Continue").click();
  });

  await step("first run: the wake word is set, and the tools are offered", async () => {
    await walkthroughStep("extras").waitFor({ timeout: 5000 });
    for (const name of ["Camera", "Screen", "Web search"]) {
      await page.getByRole("switch", { name }).waitFor({ timeout: 10_000 });
    }
    // The persona's words: its name, and no sleep word
    const word = page.locator('input[aria-label="Wake word"]');
    await until("the persona's name as its wake word", async () => (await word.inputValue()) === "Iris");
    check((await page.locator("#walkthrough-sleep-word").inputValue()) === "", "a sleep word is set");
    check(await page.getByRole("switch", { name: "Sleep when idle" }).isDisabled(), "sleeping when idle is offered without a wake word");
    await page.getByRole("switch", { name: "Wake word" }).click();
    await word.fill(WAKE_WORD);
    await footerButton("Continue").click();
  });

  await step("first run: Iris starts with the choices, and greets the user", async () => {
    await walkthroughStep("ready").waitFor({ timeout: 5000 });
    const stage = (id) => walkthroughStep("ready").locator(`[data-stage="${id}"]`);
    check((await stage("llm").innerText()).includes(model), "the pipeline does not show the model");
    check((await stage("wake").innerText()).includes(WAKE_WORD), "the pipeline does not show the wake word");
    // A stage leads back to the step that sets it
    await stage("wake").click();
    await walkthroughStep("extras").waitFor({ timeout: 5000 });
    await page.locator('nav[aria-label="Steps"] button[aria-label="Ready"]').click();
    await walkthroughStep("ready").waitFor({ timeout: 5000 });
    const since = Date.now();
    await footerButton("Start talking").click();
    await until("the conversation", () => page.locator("textarea").count(), 10_000);
    const started = await eventAfter("config", since, () => true, 600_000);
    // Typed turns need no microphone; the greeting keeps Iris awake meanwhile
    await rpc("mic.set", { on: false });
    check(started.data.wake_word === WAKE_WORD, `the wake word is ${started.data.wake_word}`);
    check(started.data.wake_window === 30, `Iris falls asleep after ${started.data.wake_window} s`);
    check((await rpc("persona.get", { id: "iris" })).wake_word === WAKE_WORD, "the wake word is not Iris's own");
    check(started.data.llm_model === model, `the model is ${started.data.llm_model}`);
    check(started.data.persona === "Iris", `the persona is ${started.data.persona}`);
    if (serverHost) check(started.data.ollama_host === serverHost, `the server is ${started.data.ollama_host}`);
    const greeting = await replyEvent("chat.done", since);
    const errors = events.filter((e) => e.at >= since && e.type === "error").map((e) => e.data.message);
    check(!errors.length, `the greeting failed: ${errors.join(" | ")}`);
    check(greeting.data.text.trim().length > 0, "the greeting is empty");
    check(!events.some((e) => e.at >= since && e.type === "chat.user"), "a turn of the user's came before the greeting");
    await until("the greeting in the conversation", () => page.getByText(greeting.data.text.trim().slice(0, 12)).count());
    await until("the voice to finish", () => rpc("app.info").then((info) => info.state !== "speaking"), 60_000);
    const saved = await settings();
    check(saved.setup_done === true, "the walkthrough is not marked done");
    if (serverHost) check(saved.ollama_host === serverHost, `the server is kept as ${saved.ollama_host}`);
  });

  await step("the app starts and the core loads its models", async () => {
    await eventAfter("config", 0, () => true, 180_000);
    await until("the conversation", () => page.locator("textarea").count());
    check(config().wake_word === WAKE_WORD, "the test core must sleep until its wake word");
    // Typed turns need no microphone; the steps that listen switch it on
    await rpc("mic.set", { on: false });
  });

  await step("settings open on the pipeline, with every stage", async () => {
    await openSettings();
    const labels = await page.locator("button.stage .label").allInnerTexts();
    const expected = [
      "Microphone", "Wake word", "Turn end", "Recognition", "Language model",
      "Persona", "Tools", "Memory", "Voice", "Output",
    ];
    check(JSON.stringify(labels) === JSON.stringify(expected), `stages: ${labels.join(", ")}`);
    for (const label of expected) {
      check((await stageValue(label)).trim(), `${label} shows no value`);
    }
    check((await stageValue("Wake word")).includes(WAKE_WORD), "the wake word is not shown");
  });

  await step("every stage opens its inspector, and it closes", async () => {
    const titles = { "Turn end": "Turn end", Output: "Output" };
    for (const label of await page.locator("button.stage .label").allInnerTexts()) {
      await openStage(label, titles[label] ?? label);
      check((await inspector.locator("section, p, .s-option").count()) > 0, `${label} is empty`);
    }
    await inspector.locator('button[title="Close"]').click();
    await until("the inspector to close", async () => !(await inspector.count()));
  });

  await step("language model: the server's models are listed and one is chosen", async () => {
    await openStage("Language model");
    await until("the model list", () => option("").count(), 60_000);
    if (!model) {
      model = (await inspector.locator(".s-option:has(.s-chip:text('tools')) .name").first().innerText()).trim();
    }
    await option(model).first().click();
    await until("the chain to show the model", async () => (await stageValue("Language model")).includes(model));
    check(config().llm_model === model, `the core runs ${config().llm_model}`);
    check(await option(model).first().evaluate((el) => el.classList.contains("active")), "not marked");
  });

  const servers = inspector.locator(".s-option.server");

  await step("servers: one is added, shown as unreachable, refused, and removed", async () => {
    await openStage("Language model");
    const before = await servers.count();
    await inspector.locator("button.s-btn", { hasText: "Add server" }).click();
    await inspector.locator('input[aria-label="Server address"]').fill("http://127.0.0.1:9");
    await inspector.locator('input[aria-label="Server address"]').press("Enter");
    const row = servers.filter({ hasText: "127.0.0.1:9" });
    await row.waitFor({ timeout: 5000 });
    await row.locator(".meta", { hasText: "Unreachable" }).waitFor({ timeout: 15_000 });
    const host = config().ollama_host;
    await row.locator("button.pick").click();
    await inspector.locator(".s-status.error").waitFor({ timeout: 30_000 });
    check(config().ollama_host === host, "switched to a server that does not answer");
    await row.locator('button[title="Remove"]').click();
    await until("the server to be gone", async () => (await servers.count()) === before);
  });

  await step("servers: the one in use is marked reachable and cannot be removed", async () => {
    await openStage("Language model");
    const active = servers.filter({ has: page.locator('button.pick[aria-checked="true"]') });
    check((await active.count()) === 1, "exactly one server is in use");
    await active.locator(".dot.ok").waitFor({ timeout: 15_000 });
    check(!(await active.locator('button[title="Remove"]').count()), "the server in use can be removed");
  });

  await step("servers: switching to another one and back", async () => {
    await openStage("Language model");
    const local = servers.filter({ hasText: "This computer" });
    // Needs two servers that answer: the one given and this computer's
    if (!process.env.E2E_HOST || !(await local.locator(".dot.ok").count())) return;
    const host = config().ollama_host;
    await local.locator("button.pick").click();
    await until("this computer's server to be in use", () => config().ollama_host === "", 30_000);
    await local.locator('button.pick[aria-checked="true"]').waitFor({ timeout: 5000 });
    // The model in use is one this server has, not the other server's
    await until("a model of this server", async () => {
      const names = await inspector.locator(".s-option .name").allInnerTexts();
      return config().llm_model !== model && names.some((name) => name.trim() === config().llm_model);
    }, 30_000);
    // The server switched away from is still listed, and switching back
    // brings back the model it had
    const other = servers.filter({ hasText: new URL(host).host });
    check(await other.count(), "the server switched away from is no longer listed");
    await other.locator("button.pick").click();
    await until("the other server to be in use again", () => config().ollama_host === host, 30_000);
    await until(`${model} to be in use again`, () => config().llm_model === model, 30_000);
    await inspector.locator(".s-option.active", { hasText: model }).waitFor({ timeout: 5000 });
  });

  await step("turn end: silence, and back to Smart Turn", async () => {
    await openStage("Turn end");
    await option("Silence").click();
    await until("the chain to show the silence", async () => (await stageValue("Turn end")).startsWith("Silence"));
    check(config().turn_detection === "silence", "the core still uses Smart Turn");
    await option("Smart Turn").click();
    await until("the chain to show Smart Turn", async () => (await stageValue("Turn end")) === "Smart Turn");
    check(config().turn_detection === "smart_turn", "the core still ends turns after a silence");
  });

  await step("recognition: the sizes are listed, and a downloaded one is switched to", async () => {
    await openStage("Recognition");
    await until("the sizes", async () => (await option("Whisper").count()) >= 6);
    const active = (await inspector.locator(".s-option.active .name").innerText()).trim();
    const other = inspector.locator(".s-option:not(.active)", { hasText: "Downloaded" }).first();
    if (!(await other.count())) return; // nothing else is on this computer
    const name = (await other.locator(".name").innerText()).trim();
    await other.click();
    await until(`the chain to show ${name}`, async () => (await stageValue("Recognition")).includes(name), 120_000);
    await option(active).click();
    await until(`the chain to show ${active} again`, async () => (await stageValue("Recognition")).includes(active), 120_000);
  });

  await step("tools: a switch turns a tool off and on", async () => {
    await openStage("Tools");
    const toggle = inspector.locator('button[role="switch"][aria-label="Web search"]');
    await toggle.click();
    await until("the chain to drop Web search", async () => !(await stageValue("Tools")).includes("Web search"));
    check(config().tools.web.enabled === false, "the core still has web search on");
    await toggle.click();
    await until("the chain to show Web search", async () => (await stageValue("Tools")).includes("Web search"));
  });

  await step("wake word: off and on again", async () => {
    await openStage("Wake word");
    const toggle = inspector.locator('button[role="switch"][aria-label="Wake word"]');
    await toggle.click();
    await until("the chain to show Off", async () => (await stageValue("Wake word")) === "Off");
    await toggle.click();
    await until("the chain to show the word", async () => (await stageValue("Wake word")) === WAKE_WORD);
    check((await settings()).wake_on === true, "the switch is not kept");
  });

  await step("voice: the persona's voice plays", async () => {
    await openStage("Voice");
    const since = Date.now();
    await inspector.locator('button[aria-label="Play"]').click();
    await eventAfter("audio.output", since, (data) => data.rms > 0, 60_000);
    await until("the voice to finish", () => rpc("app.info").then((info) => info.state !== "speaking"), 30_000);
    await checkHeard(since, "the persona's voice");
  });

  await step("personas: each one opens, and a voice tile plays that voice", async () => {
    await openSettings("Personas");
    await until("the persona list", async () => (await page.locator("button.card").count()) >= 3);
    await page.locator("button.card", { hasText: "HAL" }).click();
    await until("HAL's form", async () => (await page.locator("input.title").inputValue()) === "HAL");
    check((await page.locator(".voice").count()) > 5, "no voice tiles");
    const since = Date.now();
    await page.locator(".voice .play").first().click();
    await page.locator(".voice .play.playing").waitFor({ timeout: 2000 });
    await eventAfter("audio.output", since, (data) => data.rms > 0, 60_000);
    await until("the voice to finish", () => rpc("app.info").then((info) => info.state !== "speaking"), 30_000);
    await page.locator(".voice .play.playing").waitFor({ state: "detached", timeout: 5000 });
    await checkHeard(since, "the voice tile's voice");
  });

  await step("personas: a voice of the other engine plays too", async () => {
    const engines = page.locator('[role="radiogroup"][aria-label="Speech engine"] button');
    const other = engines.locator("xpath=self::*[not(contains(@class,'active'))]").first();
    await other.click();
    await until("that engine's voices", () => page.locator(".voice").count());
    const since = Date.now();
    await page.locator(".voice .play").first().click();
    await eventAfter("audio.output", since, (data) => data.rms > 0, 90_000);
    await until("the voice to finish", () => rpc("app.info").then((info) => info.state !== "speaking"), 30_000);
    await checkHeard(since, "the other engine's voice");
    await page.locator("button.s-btn", { hasText: "Cancel" }).click();
  });

  await step("personas: an edit is saved, and Cancel reverts one", async () => {
    await page.locator("button.card", { hasText: "Computer" }).click();
    const text = page.locator("#persona-personality");
    await until("the form", async () => (await page.locator("input.title").inputValue()) === "Computer");
    const original = await text.inputValue();
    await text.fill(original + "\nSpeak like a test.");
    await page.locator("button.s-btn.primary", { hasText: "Save" }).click();
    await until("the save", async () => (await rpc("persona.get", { id: "computer" })).description.includes("Speak like a test."));
    await until("Save to rest", () => page.locator("button.s-btn.primary").isDisabled());
    await text.fill("Something else entirely");
    await page.locator("button.s-btn", { hasText: "Cancel" }).click();
    await until("the edit to be reverted", async () => (await text.inputValue()).includes("Speak like a test."));
    await page.locator("button.s-btn.danger", { hasText: "Reset" }).click();
    await page.locator("button.s-btn.danger", { hasText: "Confirm reset" }).click();
    await until("the reset", async () => !(await rpc("persona.get", { id: "computer" })).description.includes("Speak like a test."));
  });

  await step("personas: a new one is created, becomes active, and is deleted", async () => {
    await page.locator("button.s-btn", { hasText: "New persona" }).click();
    await page.locator("input.title").fill("Testa");
    // Its wake word is its name
    check((await page.locator("#persona-wake-word").inputValue()) === "Testa", "the wake word does not follow the name");
    await page.locator("#persona-personality").fill("You are Testa, made by a test.");
    await page.locator("button.s-btn.primary", { hasText: "Save" }).click();
    await until("Testa in the list", () => page.locator("button.card", { hasText: "Testa" }).count());
    await until("Testa to be active", () => config().persona === "Testa", 60_000);
    check(config().wake_word === "Testa", `Testa is woken by ${config().wake_word}`);
    check(await page.locator("button.card", { hasText: "Testa" }).locator(".s-chip").count(), "not marked active");
    await page.locator("button.s-btn.danger", { hasText: "Delete" }).click();
    await page.locator("button.s-btn.danger", { hasText: "Confirm delete" }).click();
    await until("Testa to be gone", async () => !(await page.locator("button.card", { hasText: "Testa" }).count()));
    await until("another persona to be active", () => config().persona !== "Testa", 60_000);
  });

  await step("personas: Talk to makes another persona the active one", async () => {
    const other = page.locator("button.card:not(:has(.s-chip))").first();
    const name = (await other.innerText()).trim().split("\n").pop();
    await other.click();
    await page.locator("button.s-btn", { hasText: "Talk to" }).click();
    await until(`${name} to be active`, () => config().persona === name, 60_000);
    await page.locator("button.card", { hasText: "Iris" }).first().click();
    await until("Iris's form", async () => (await page.locator("input.title").inputValue()) === "Iris");
    if (config().persona !== "Iris") await page.locator("button.s-btn", { hasText: "Talk to Iris" }).click();
    await until("Iris to be active again", () => config().persona === "Iris", 60_000);
  });

  await step("personas: a voice plays while the microphone listens, and is not taken for the user", async () => {
    await rpc("wake.set", { on: false });
    await rpc("mic.set", { on: true });
    await until("Iris to listen", () => rpc("app.info").then((info) => info.state === "listening"), 15_000);
    const since = Date.now();
    await page.locator(".voice .play").nth(1).click();
    await eventAfter("audio.output", since, (data) => data.rms > 0, 60_000);
    await until("the voice to finish", () => rpc("app.info").then((info) => info.state !== "speaking"), 30_000);
    await checkHeard(since, "the voice played while the microphone listens");
    await new Promise((resolve) => setTimeout(resolve, 6000));
    await rpc("mic.set", { on: false });
    await rpc("wake.set", { on: true });
    const heard = events.find((e) => e.at >= since && e.type === "chat.user");
    check(!heard, `the preview was heard as the user saying: ${heard?.data.text}`);
    // And once more: the voice still plays after the first preview
    const again = Date.now();
    await page.locator(".voice .play").nth(2).click();
    await eventAfter("audio.output", again, (data) => data.rms > 0, 60_000);
    await until("the voice to finish", () => rpc("app.info").then((info) => info.state !== "speaking"), 30_000);
    await checkHeard(again, "the voice played next");
  });

  await step("personas: saving the current persona applies it at once", async () => {
    await openSettings("Personas");
    await page.locator("button.card", { hasText: "Iris" }).first().click();
    await until("Iris's form", async () => (await page.locator("input.title").inputValue()) === "Iris");
    check(config().persona === "Iris", "Iris is not the current persona");
    const shipped = config().tts_voice;
    // Another voice of her language
    const other = page.locator(".voices").first().locator(".voice:not(.active) button.pick").first();
    await other.click();
    await page.locator("button.s-btn.primary", { hasText: "Save" }).click();
    await until("the core to take the new voice", () => config().tts_voice !== shipped, 10_000);
    const voice = `${config().tts_engine}:${config().tts_voice}`;
    // The next reply is spoken in it
    const since = await ask("Reply with one word: yes.");
    const timing = await replyEvent("turn.timing", since);
    check(timing.data.components.voice === voice, `the reply was spoken by ${timing.data.components.voice}, not ${voice}`);
    await voiceDone();
    // Reset brings her shipped voice back, at once too
    await openSettings("Personas");
    await page.locator("button.card", { hasText: "Iris" }).first().click();
    await until("Iris's form", async () => (await page.locator("input.title").inputValue()) === "Iris");
    await page.locator("button.s-btn.danger", { hasText: "Reset" }).click();
    await page.locator("button.s-btn.danger", { hasText: "Confirm reset" }).click();
    await until("her shipped voice", () => config().tts_voice === shipped, 10_000);
    // And her name as the wake word; the test's own is set in the editor again
    await until("her name as the wake word", () => config().wake_word === "Iris", 10_000);
    await until("Iris's form", async () => (await page.locator("#persona-wake-word").inputValue()) === "Iris");
    await page.locator("#persona-wake-word").fill(WAKE_WORD);
    await page.locator("button.s-btn.primary", { hasText: "Save" }).click();
    await until("the test's wake word", () => config().wake_word === WAKE_WORD, 10_000);
  });

  // Iris's voice before she is given a cloned one, to give back at the end
  let shippedVoice = "anna";

  await step("voice cloning: Hugging Face is connected, a sample is cloned, and Iris speaks in the voice", async () => {
    await openSettings("Personas");
    await page.locator("button.card", { hasText: "Iris" }).first().click();
    await until("Iris's form", async () => (await page.locator("input.title").inputValue()) === "Iris");
    shippedVoice = (await rpc("persona.get", { id: "iris" })).voice;
    await page.getByRole("button", { name: "Clone a voice" }).click();
    // No login yet: the model page opens in the browser, and a token is asked for
    await page.locator('.clone[data-step="connect"]').waitFor({ timeout: 10_000 });
    await page.getByRole("button", { name: "Open the model page" }).click();
    check((await page.evaluate(() => window.__opened)).includes("https://huggingface.co/kyutai/pocket-tts"),
      "the model page did not go to the browser");
    const token = page.locator('input[aria-label="Hugging Face access token"]');
    check((await token.getAttribute("type")) === "password", "the token shows");
    await token.fill("hf_not_the_one");
    await page.getByRole("button", { name: "Check", exact: true }).click();
    await page.getByText("Hugging Face did not accept the token").waitFor({ timeout: 10_000 });
    await token.fill(CLONE_TOKEN);
    await page.getByRole("button", { name: "Check", exact: true }).click();
    // Logged in, the terms not accepted yet
    await page.getByText("then check again").waitFor({ timeout: 10_000 });
    const answers = JSON.stringify(calls.filter((c) => c.method.startsWith("voice.")).map((c) => c.result));
    check(!answers.includes(CLONE_TOKEN), "the token came back from the core");
    // Accepted on the website: the cloning model downloads, with progress
    huggingFace({ terms: true });
    const since = Date.now();
    await page.getByRole("button", { name: "Check", exact: true }).click();
    await page.locator('.clone[data-step="sample"]').waitFor({ timeout: 30_000 });
    check(events.some((e) => e.at >= since && e.type === "clone.progress" && e.data.done === e.data.total),
      "the download showed no progress");
    await page.getByText("Only clone a voice you have permission to use.").waitFor();
    // A sample from a file: the voice is made, chosen, and says the preview
    const sample = speechSample();
    const made = Date.now();
    await page.locator('input[aria-label="Audio file"]').setInputFiles({ name: "sample.wav", mimeType: "audio/wav", buffer: sample });
    await page.getByRole("button", { name: "Clone again" }).waitFor({ timeout: 120_000 });
    check(await page.locator(".voice.active", { hasText: "Iris" }).count(), "the cloned voice is not chosen");
    await eventAfter("audio.output", made, (data) => data.rms > 0, 90_000);
    await voiceDone();
    await checkHeard(made, "the cloned voice's first words");
    // Its tile plays it as well
    const played = Date.now();
    await page.locator(".voice.active .play").click();
    await eventAfter("audio.output", played, (data) => data.rms > 0, 90_000);
    await voiceDone();
    await checkHeard(played, "the cloned voice's tile");
    // Saved, Iris speaks in it at once
    await page.locator("button.s-btn.primary", { hasText: "Save" }).click();
    await until("the cloned voice", () => config().tts_voice === "clone:iris", 10_000);
    check((await rpc("persona.get", { id: "iris" })).voice === "clone:iris", "Iris was not saved with the cloned voice");
    const sent = await ask("Reply with one word: yes.");
    const timing = await replyEvent("turn.timing", sent);
    check(timing.data.components.voice === "kyutai:clone:iris", `the reply was spoken by ${timing.data.components.voice}`);
    await voiceDone();
  });

  await step("voice cloning: a recording of the empty room is refused as too short", async () => {
    await openSettings("Personas");
    await page.locator("button.card", { hasText: "Iris" }).first().click();
    await page.getByRole("button", { name: "Clone again" }).click();
    await page.getByRole("button", { name: "Record", exact: true }).click();
    await page.locator('.clone[data-step="recording"]').waitFor({ timeout: 10_000 });
    const since = Date.now();
    await eventAfter("clone.level", since, (data) => data.seconds > 1.5, 10_000);
    await page.getByRole("button", { name: "Stop", exact: true }).click();
    await page.getByText("Too short").waitFor({ timeout: 15_000 });
    await page.getByRole("button", { name: "Close", exact: true }).click();
    await quiet(500);
    const late = events.filter((e) => e.type === "clone.level" && e.at > Date.now() - 300);
    check(late.length === 0, "the microphone still records");
    check((await rpc("persona.get", { id: "iris" })).voice === "clone:iris", "the voice changed");
  });

  await step("voice cloning: Save is offered for what changed, a voice cloned again too, and Cancel takes it back", async () => {
    const save = page.locator("button.s-btn.primary", { hasText: "Save" });
    const effect = (name) => page.locator("button.toggle", { hasText: exactly(name) });
    const pick = (name) => page.locator(".voice button.pick", { hasText: exactly(name) });
    const offered = async (yes, after) =>
      check((await save.isEnabled()) === yes, `Save is ${yes ? "not " : ""}offered after ${after}`);
    await openSettings("Personas");
    await page.locator("button.card", { hasText: "Iris" }).first().click();
    await until("Iris's form", async () => (await page.locator("input.title").inputValue()) === "Iris");
    await offered(false, "opening her");
    await effect("Phone").click();
    await offered(true, "choosing an effect");
    await effect("None").click();
    await offered(false, "choosing no effect again");
    await pick("Anna").click();
    await offered(true, "choosing another voice");
    await pick("Iris").click();
    await offered(false, "choosing her cloned voice again");
    // A voice cloned again is chosen, and is hers once saved
    await page.getByRole("button", { name: "Clone again" }).click();
    await page.locator('.clone[data-step="sample"]').waitFor({ timeout: 30_000 });
    const made = Date.now();
    await page.locator('input[aria-label="Audio file"]').setInputFiles({ name: "sample.wav", mimeType: "audio/wav", buffer: speechSample() });
    await until("the voice cloned again", () => page.locator(".voice.active .pick", { hasText: exactly("Iris 2") }).count(), 120_000);
    await eventAfter("audio.output", made, (data) => data.rms > 0, 90_000);
    await voiceDone();
    await offered(true, "cloning a voice");
    await effect("Phone").click();
    await effect("None").click();
    await offered(true, "an effect and back, with the voice cloned again");
    await pick("Iris").click();
    await offered(false, "choosing her saved voice");
    await pick("Iris 2").click();
    await offered(true, "choosing the voice cloned again");
    // Cancel goes back to what is saved
    await page.locator("button.s-btn", { hasText: "Cancel" }).click();
    await until("her saved voice", () => page.locator(".voice.active .pick", { hasText: exactly("Iris") }).count());
    await offered(false, "Cancel");
  });

  await step("voice cloning: a voice cloned and not kept is offered no more, at once or when the editor closes", async () => {
    const pick = (name) => page.locator(".voice button.pick", { hasText: exactly(name) });
    const listed = async () => (await page.locator(".voice button.pick").allInnerTexts()).filter((v) => v.startsWith("Iris"));
    const kept = async () => (await rpc("persona.voices")).engines.kyutai.filter((v) => v.id.startsWith("clone:")).map((v) => v.name);
    // Cloned again in place of the voice made before, that one goes at once
    await pick("Iris 2").click();
    await page.getByRole("button", { name: "Clone again" }).click();
    await page.locator('.clone[data-step="sample"]').waitFor({ timeout: 30_000 });
    const made = Date.now();
    await page.locator('input[aria-label="Audio file"]').setInputFiles({ name: "sample.wav", mimeType: "audio/wav", buffer: speechSample() });
    await until("the voice cloned once more", () => page.locator(".voice.active .pick", { hasText: exactly("Iris 3") }).count(), 120_000);
    await eventAfter("audio.output", made, (data) => data.rms > 0, 90_000);
    await voiceDone();
    await until("the voice it replaces to go", async () => !(await listed()).includes("Iris 2"));
    check(!(await kept()).includes("Iris 2"), "the voice replaced is kept");
    // Not saved, the last one goes when the editor closes
    await page.locator("button.s-btn", { hasText: "Cancel" }).click();
    await openSettings();
    await until("the voice not kept to go", async () => !(await kept()).includes("Iris 3"));
    check((await rpc("persona.get", { id: "iris" })).voice === "clone:iris", "her voice changed");
    // And is not offered when the editor opens again
    await openSettings("Personas");
    await page.locator("button.card", { hasText: "Iris" }).first().click();
    await until("Iris's form", async () => (await page.locator("input.title").inputValue()) === "Iris");
    await until("her cloned voice alone to be offered", async () => JSON.stringify(await listed()) === '["Iris"]', 5000)
      .catch(async () => check(false, `the cloned voices offered are ${JSON.stringify(await listed())}`));
    await openSettings();
  });

  await step("setups: a new one, renamed, switched and deleted", async () => {
    await openSettings();
    const current = page.locator(".switcher .current");
    const menu = page.locator(".switcher .menu");
    await current.click();
    await menu.locator("button", { hasText: "New setup" }).click();
    await menu.locator("input").fill("Quiet");
    await menu.locator("input").press("Enter");
    await until("the new setup to be in use", async () => (await current.innerText()).includes("Quiet"));
    // A choice made now belongs to Quiet only
    await openStage("Turn end");
    await option("Silence").click();
    await until("silence", () => config().turn_detection === "silence");
    await current.click();
    await menu.locator('button[role="menuitemradio"]', { hasText: "My setup" }).click();
    await until("the first setup's turn end", () => config().turn_detection === "smart_turn");
    await current.click();
    await menu.locator('button[role="menuitemradio"]', { hasText: "Quiet" }).click();
    await until("Quiet's turn end", () => config().turn_detection === "silence");
    await current.click();
    await menu.locator("button", { hasText: "Rename" }).click();
    await menu.locator("input").fill("Hushed");
    await menu.locator("input").press("Enter");
    await until("the new name", async () => (await current.innerText()).includes("Hushed"));
    await current.click();
    await menu.locator("button", { hasText: "Delete" }).click();
    await until("the first setup again", async () => (await current.innerText()).includes("My setup"));
    await until("its turn end", () => config().turn_detection === "smart_turn");
  });

  const current = page.locator(".switcher .current");
  const menu = page.locator(".switcher .menu");

  async function newSetup(name) {
    await openSettings();
    await current.click();
    await menu.locator("button", { hasText: "New setup" }).click();
    await menu.locator("input").fill(name);
    await menu.locator("input").press("Enter");
    await until(`${name} to be in use`, async () => (await current.innerText()).includes(name));
  }

  async function useSetup(name) {
    await openSettings();
    await current.click();
    await menu.locator('button[role="menuitemradio"]', { hasText: name }).click();
    await until(`${name} to be in use`, async () => (await current.innerText()).includes(name));
  }

  const stubHost = new URL(stubUrl).host;
  const home = {};
  /** Whether the core runs on this server, with this model and web search. */
  const runs = (server, llm, web) => () => {
    const c = config();
    return c.ollama_host === server && c.llm_model === llm && c.tools.web.enabled === web;
  };

  await step("setups: switching applies their server, model and tools", async () => {
    Object.assign(home, { host: config().ollama_host, model: config().llm_model });
    check(config().tools.web.enabled, "web search is off to begin with");
    // A setup on the test's own server with its small model and no web search
    await newSetup("Stub small");
    await openStage("Language model");
    await inspector.locator("button.s-btn", { hasText: "Add server" }).click();
    await inspector.locator('input[aria-label="Server address"]').fill(stubUrl);
    await inspector.locator('input[aria-label="Server address"]').press("Enter");
    await servers.filter({ hasText: stubHost }).locator("button.pick").click();
    await until("the core on that server", () => config().ollama_host === stubUrl, 30_000);
    await until("its models", () => option("stub-small").count(), 30_000);
    await option("stub-small").click();
    await until("its small model", () => config().llm_model === "stub-small");
    await openStage("Tools");
    await inspector.locator('button[role="switch"][aria-label="Web search"]').click();
    await until("web search off", () => !config().tools.web.enabled);
    // Another on the same server, with its large model
    await newSetup("Stub large");
    await openStage("Language model");
    await until("the models", () => option("stub-large").count(), 30_000);
    await option("stub-large").click();
    await until("its large model", () => config().llm_model === "stub-large");

    await useSetup("My setup");
    await until("the first setup's server, model and web search", runs(home.host, home.model, true), 30_000);
    // The warning that the model is not on the server goes once it is chosen
    await quiet(1000);
    check(!(await page.locator(".bubble.error", { hasText: /has no model|is not installed/ }).count()),
      "the conversation still says the model is not on the server");
    // Back to a server: the setup's own model, not the one last used there
    await useSetup("Stub small");
    await until("Stub small's server, model and web search", runs(stubUrl, "stub-small", false), 30_000);
    await quiet(1000);
    check(runs(stubUrl, "stub-small", false)(), `Stub small runs ${config().llm_model}`);
    // Its replies come from that server
    const since = await ask("Hello there.");
    const done = await eventAfter("chat.done", since, () => true, 60_000);
    check(done.data.text.includes("the stub's answer"), `the reply came from elsewhere: ${done.data.text}`);
    check(stubRequests.at(-1)?.model === "stub-small", `the server was asked for ${stubRequests.at(-1)?.model}`);
    await voiceDone();
    await useSetup("Stub large");
    await until("Stub large's model", runs(stubUrl, "stub-large", false), 30_000);
  });

  await step("conversation: a reply shows its markdown, without emoji or unsafe HTML, and links leave the window", async () => {
    await toConversation();
    const reply = page.locator(".bubble.assistant", { hasText: "the stub's answer" }).last();
    await reply.waitFor({ timeout: 5000 });
    check((await reply.locator("strong").innerText()) === "the stub's answer", "the bold text is not bold");
    check((await reply.locator("li").count()) === 2, "the list is not a list");
    check(!/\u{1F642}/u.test(await reply.innerText()), "the emoji is shown");
    check(!(await reply.locator("img").count()), "an image from the reply is shown");
    check(!(await reply.locator('a[href^="javascript"]').count()), "a script link is kept");
    check(!(await page.evaluate(() => window.__injected)), "the reply ran a script");
    const before = page.url();
    await reply.locator("a", { hasText: "the example" }).click();
    check(page.url() === before, "the window left the app");
    check((await page.evaluate(() => window.__opened)).includes("https://example.com/page"),
      "the link did not go to the browser");
  });

  await step("setups: one whose server is away leaves the server and model in use, and says so", async () => {
    await useSetup("My setup");
    await until("the first setup's server, model and web search", runs(home.host, home.model, true), 30_000);
    const port = stub.address().port;
    stub.close();
    stub.closeAllConnections();
    try {
      await useSetup("Stub small");
      // In the conversation, behind the settings
      await page.locator(".bubble.error", { hasText: stubHost }).waitFor({ state: "attached", timeout: 30_000 });
      await quiet(1000);
      check(config().ollama_host === home.host, `the core went to ${config().ollama_host}`);
      check(config().llm_model === home.model, `the core runs ${config().llm_model}, a model of the server away`);
    } finally {
      await new Promise((resolve) => stub.listen(port, "127.0.0.1", resolve));
      // The test's server is left: the setups on it go
      for (const name of ["Stub small", "Stub large"]) {
        await useSetup(name);
        await current.click();
        await menu.locator("button", { hasText: "Delete" }).click();
        await until("the first setup again", async () => (await current.innerText()).includes("My setup"));
      }
      await until("the first setup's server, model and web search", runs(home.host, home.model, true), 30_000);
    }
  });

  await step("conversation: a typed message is answered, after a note in the conversation", async () => {
    await page.locator("nav button", { hasText: "Conversation" }).click();
    const since = Date.now();
    await page.locator("textarea").fill("Say hello in five words.");
    await page.locator("textarea").press("Enter");
    const done = await replyEvent("chat.done", since);
    const error = events.find((e) => e.at >= since && e.type === "error");
    check(!error, `the reply failed: ${error?.data.message}`);
    check(done.data.text.trim().length > 0, "the reply is empty");
    await until("the reply in the conversation", () => page.getByText(done.data.text.trim().slice(0, 12)).count());
    await checkHeard(since, "the reply");
    await until("the voice to finish", () => rpc("app.info").then((info) => info.state !== "speaking"), 60_000);
  });

  await step("pipeline: the turn's time shows per stage", async () => {
    await openSettings();
    await until("the model's time", async () => /\d+ ms/.test(await stage("Language model").locator(".ms").innerText()));
    check(/\d+ ms/.test(await stage("Voice").locator(".ms").innerText()), "the voice shows no time");
    check(/\d\.\d\d s/.test(await page.locator(".total .ms").innerText()), "no time to the first word");
    await page.locator("nav button", { hasText: "Conversation" }).click();
  });

  const box = page.locator(".composer");
  /** A message box button by what it does now */
  const boxButton = (title) => box.locator(`button[title="${title}"]`);
  const state = () => rpc("app.info").then((info) => info.state);

  await step("message box: Pause holds a typed message until Resume, and so does the orb", async () => {
    await toConversation();
    await boxButton("Pause the conversation").click();
    await until("the core to pause", () => config().paused === true);
    await boxButton("Resume the conversation").waitFor({ timeout: 5000 });
    check((await boxButton("Resume the conversation").innerText()).trim() === "Resume", "the button does not say Resume");
    await until("the paused state", async () => (await state()) === "paused");
    // A message sent now waits, shown as waiting
    const since = await ask("Reply with one word: ready.");
    await page.locator(".bubble.pending", { hasText: "Reply with one word: ready." }).waitFor({ timeout: 5000 });
    await quiet(3000);
    check(!events.find((e) => e.at >= since && e.type === "chat.user"), "a message was taken while paused");
    await boxButton("Resume the conversation").click();
    await until("the core to resume", () => config().paused === false);
    const done = await replyEvent("chat.done", since);
    check(done.data.text.trim(), "no reply after Resume");
    await page.locator(".bubble.pending").waitFor({ state: "detached", timeout: 5000 });
    const sent = page.locator(".row", { has: page.locator(".bubble.user", { hasText: "Reply with one word: ready." }) });
    check(await sent.locator('.gutter[title="Typed"]').count(), "the message is not marked as typed");
    await voiceDone();
    // A click on the orb pauses and resumes too
    const orb = page.locator("button", { has: page.locator('[aria-label="assistant status"]') });
    await orb.click();
    await until("the core to pause", () => config().paused === true);
    await orb.click();
    await until("the core to resume", () => config().paused === false);
    await until("Iris at rest", async () => (await state()) === "muted");
  });

  await step("message box: Mic switches the microphone, and a draft holds it", async () => {
    await boxButton("Turn microphone on").click();
    await until("the mic on", () => config().mic_on === true);
    check((await boxButton("Turn microphone off").innerText()).trim() === "Mic", "the button does not say Mic");
    // Awake after the last turn, or asleep until its word
    const listening = async () => ["listening", "dormant"].includes(await state());
    await until("Iris to listen", listening, 15_000);
    // While a draft is being typed, the mic waits
    await page.locator("textarea").fill("A draft");
    await until("the mic to wait for the draft", async () => (await state()) === "typing", 5000);
    await page.locator("textarea").fill("");
    await until("the mic to listen again", listening, 5000);
    await boxButton("Turn microphone off").click();
    await until("the mic off", () => config().mic_on === false);
    check((await boxButton("Turn microphone on").innerText()).trim() === "Mic off", "the button does not say Mic off");
    await until("Iris muted", async () => (await state()) === "muted");
  });

  await step("message box: Voice off gives the reply as text, with no sound", async () => {
    await boxButton("Reply in text only").click();
    await until("the voice off", () => config().speak_replies === false);
    check((await boxButton("Speak replies").innerText()).trim() === "Voice off", "the button does not say Voice off");
    const since = await ask("Reply with exactly: text only today.");
    const done = await replyEvent("chat.done", since);
    check(done.data.text.trim(), "the reply is empty");
    await until("the reply in the conversation", () => page.getByText(done.data.text.trim().slice(0, 12)).count());
    const timing = await eventAfter("turn.timing", since, () => true, 10_000);
    check(timing.data.stages.voice === null, "a reply in text only was timed as spoken");
    await quiet(2000);
    const sound = events.filter((e) => e.at >= since && e.type === "audio.output" && e.data.rms > 0.001).length;
    check(sound === 0, `${sound} audible moments in a reply in text only`);
    check(!events.some((e) => e.at >= since && e.type === "state" && e.data.state === "speaking"), "Iris spoke");
    await boxButton("Speak replies").click();
    await until("the voice on", () => config().speak_replies === true);
    await boxButton("Reply in text only").waitFor({ timeout: 5000 });
  });

  await step("message box: Amnesia stops remembering, and starts again", async () => {
    check(config().memory, "the core runs without memory");
    const amnesia = box.locator("button.toggle", { hasText: "Amnesia" });
    await amnesia.click();
    await until("amnesia on", () => config().amnesia === true);
    await boxButton("Amnesia mode is on: nothing is remembered").waitFor({ timeout: 5000 });
    await amnesia.click();
    await until("amnesia off", () => config().amnesia === false);
    await boxButton("Amnesia mode: remember nothing of this conversation").waitFor({ timeout: 5000 });
  });

  await step("settings: the Output and Memory switches follow the core", async () => {
    const switchOf = (label) => inspector.locator(`button[role="switch"][aria-label="${label}"]`);
    const checked = async (label) => (await switchOf(label).getAttribute("aria-checked")) === "true";
    await openStage("Output");
    check(await checked("Speak replies"), "Speak replies is off");
    await switchOf("Speak replies").click();
    await until("the voice off", () => config().speak_replies === false);
    await until("the switch off", async () => !(await checked("Speak replies")));
    check((await stageValue("Output")) === "Text only", `Output shows ${await stageValue("Output")}`);
    await toConversation();
    await boxButton("Speak replies").waitFor({ timeout: 5000 });
    await openStage("Output");
    if (config().barge_in_available) {
      check(await switchOf("Interrupt by voice").isDisabled(), "Interrupt by voice can be switched with the voice off");
    }
    await switchOf("Speak replies").click();
    await until("the voice on", () => config().speak_replies === true);
    if (config().barge_in_available) {
      check(await checked("Interrupt by voice"), "Interrupt by voice is off");
      await switchOf("Interrupt by voice").click();
      await until("interrupting by voice off", () => config().barge_in === false);
      await until("the switch off", async () => !(await checked("Interrupt by voice")));
      await switchOf("Interrupt by voice").click();
      await until("interrupting by voice on", () => config().barge_in === true);
    }
    await openStage("Memory");
    check(await checked("Remember conversations"), "Remember conversations is off");
    await switchOf("Remember conversations").click();
    await until("amnesia on", () => config().amnesia === true);
    await until("the switch off", async () => !(await checked("Remember conversations")));
    check((await stageValue("Memory")) === "Paused", `Memory shows ${await stageValue("Memory")}`);
    await switchOf("Remember conversations").click();
    await until("amnesia off", () => config().amnesia === false);
    check((await stageValue("Memory")) === "On", `Memory shows ${await stageValue("Memory")}`);
    // Open memory shows the memory panel in place of the settings
    await inspector.locator("button.s-btn", { hasText: "Open memory" }).click();
    await page.locator("aside h2", { hasText: "Memory" }).waitFor({ timeout: 5000 });
    check(!(await page.locator('nav[aria-label="Settings"]').count()), "the settings are still open");
    await page.locator('button[title="Close memory"]').click();
    await page.locator('button[title="Close memory"]').waitFor({ state: "detached", timeout: 5000 });
  });

  await step("message box: Enter sends, Shift+Enter breaks the line, Escape leaves, typing anywhere lands in it", async () => {
    await toConversation();
    const field = page.locator("textarea");
    const focused = () => field.evaluate((el) => document.activeElement === el);
    await field.evaluate((el) => el.blur());
    const since = Date.now();
    await page.keyboard.press("H");
    await until("the box to take the typing", focused, 2000);
    await page.keyboard.press("Shift+Enter");
    await page.keyboard.type("i");
    check((await field.inputValue()) === "H\ni", `the box holds ${JSON.stringify(await field.inputValue())}`);
    // Enter that confirms an input method's candidate does not send
    await field.evaluate((el) =>
      el.dispatchEvent(new KeyboardEvent("keydown", { key: "Enter", isComposing: true, bubbles: true, cancelable: true })));
    await quiet(300);
    check(!calls.some((c) => c.at >= since && c.method === "chat.send_text"), "the message was sent");
    await page.keyboard.press("Escape");
    check(!(await focused()), "Escape left the focus in the box");
    await field.fill("");
  });

  const attach = async (files) => {
    const chooser = page.waitForEvent("filechooser");
    await page.locator('button[title="Attach files"]').click();
    await (await chooser).setFiles(files);
  };
  /** Files given to the message box as a paste, or dropped on the window. */
  const handOver = (how, files) =>
    page.evaluate(
      ([how, files]) => {
        const data = new DataTransfer();
        for (const file of files) {
          const bytes = Uint8Array.from(atob(file.data), (c) => c.charCodeAt(0));
          data.items.add(new File([bytes], file.name, { type: file.type }));
        }
        if (how === "paste") {
          const field = document.querySelector("textarea");
          field.focus();
          field.dispatchEvent(new ClipboardEvent("paste", { clipboardData: data, bubbles: true, cancelable: true }));
        } else {
          const target = document.querySelector(".bubble") ?? document.body;
          for (const type of ["dragenter", "dragover", "drop"]) {
            target.dispatchEvent(new DragEvent(type, { dataTransfer: data, bubbles: true, cancelable: true }));
          }
        }
      },
      [how, files.map((f) => ({ name: f.name, type: f.mimeType, data: f.buffer.toString("base64") }))]
    );
  const drafts = page.locator(".draft");
  async function clearDraft() {
    while (await drafts.count()) await drafts.first().locator(".remove").click();
    await page.locator("textarea").fill("");
  }
  /** The files of a chat.user event, as name:kind */
  const described = (data) => (data.attachments ?? []).map((f) => `${f.name}:${f.kind}`).join();
  /** The core's record of the latest typed turn, as a reloaded app gets it */
  const lastTyped = async () =>
    (await rpc("chat.history")).events.findLast((e) => e.type === "chat.user" && e.data.source === "text")?.data;
  /** The user's message in the latest request to the test's server, as text */
  const sentToStub = () => {
    const message = stubRequests.at(-1).messages.findLast((m) => m.role === "user");
    const parts = Array.isArray(message.content) ? message.content : [{ type: "text", text: message.content }];
    return {
      text: parts.filter((p) => p.type === "text").map((p) => p.text).join("\n"),
      images: parts.filter((p) => p.type === "image_url").map((p) => p.image_url.url),
    };
  };
  /** Switch the core to a server and model through the Language model stage. */
  async function runOn(server, llm) {
    await openStage("Language model");
    let row = servers.filter({ has: page.locator(`.name[title="${server}"]`) });
    if (server && !(await row.count())) {
      await inspector.locator("button.s-btn", { hasText: "Add server" }).click();
      await inspector.locator('input[aria-label="Server address"]').fill(server);
      await inspector.locator('input[aria-label="Server address"]').press("Enter");
      row = servers.filter({ has: page.locator(`.name[title="${server}"]`) });
    }
    if (config().ollama_host !== server) await row.locator("button.pick").click();
    await until(`the core on ${server}`, () => config().ollama_host === server, 30_000);
    await until(`${llm} on offer`, () => option(llm).count(), 30_000);
    if (config().llm_model !== llm) await option(llm).first().click();
    await until(`${llm} in use`, () => config().llm_model === llm, 30_000);
  }

  const NOTES = "The lighthouse keeper's cat is called Saffron Quill.";
  const LEAFLET = "The harbour bell is called Marigold.";
  const notes = { name: "notes.txt", mimeType: "text/plain", buffer: Buffer.from(NOTES + "\n") };
  const leaflet = { name: "leaflet.pdf", mimeType: "application/pdf", buffer: pdf(LEAFLET) };
  const NO_VISION = "None of the installed models can see images";
  let lemon;
  let orbit;

  await step("attachments: a text file and a PDF from the paperclip reach the model, which answers from them", async () => {
    await toConversation();
    await attach([notes, leaflet]);
    await page.locator(".draft.document", { hasText: "notes.txt" }).waitFor({ timeout: 5000 });
    await page.locator(".draft.document", { hasText: "leaflet.pdf" }).waitFor({ timeout: 5000 });
    const since = await ask(
      "What is the cat called in my notes, and what is the bell called in the leaflet? Answer with the two names."
    );
    const user = await eventAfter("chat.user", since, (data) => data.source === "text", 30_000);
    check(described(user.data) === "notes.txt:document,leaflet.pdf:document", `the files arrived as ${described(user.data)}`);
    check(!(await drafts.count()), "the files are still in the message box");
    const done = await replyEvent("chat.done", since);
    check(/saffron/i.test(done.data.text) && /marigold/i.test(done.data.text),
      `the reply does not answer from the files: ${done.data.text}`);
    // The core keeps the files with the message, for an app that reloads
    const kept = await lastTyped();
    check(described(kept) === described(user.data), `the core keeps ${described(kept)}`);
    // Shown with the message
    const sent = page.locator(".sent").filter({ hasText: "What is the cat called" }).last();
    await sent.locator(".document", { hasText: "notes.txt" }).waitFor({ timeout: 5000 });
    await sent.locator(".document", { hasText: "leaflet.pdf" }).waitFor({ timeout: 5000 });
    await voiceDone();
  });

  await step("attachments: a pasted PNG and a dropped JPEG show and open large, and a model that cannot see them is said to", async () => {
    await toConversation();
    // Their names do not give their words away
    lemon = { name: "first.png", mimeType: "image/png", buffer: await picture("LEMON", "#c8323c", "image/png") };
    orbit = { name: "second.jpg", mimeType: "image/jpeg", buffer: await picture("ORBIT", "#2f4fb4", "image/jpeg") };
    await handOver("paste", [lemon]);
    await page.locator('.draft.image img[alt="first.png"]').waitFor({ timeout: 5000 });
    await handOver("drop", [orbit]);
    await page.locator('.draft.image img[alt="second.jpg"]').waitFor({ timeout: 5000 });
    const sees = (await rpc("llm.list")).models.find((m) => m.name === config().llm_model)?.vision;
    const since = await ask("Which words are written in these two pictures?");
    const user = await eventAfter("chat.user", since, (data) => data.source === "text", 30_000);
    check(described(user.data) === "first.png:image,second.jpg:image", `the files arrived as ${described(user.data)}`);
    check(user.data.attachments.every((f) => f.preview?.startsWith("data:image/jpeg;base64,")), "a picture has no preview");
    const done = await replyEvent("chat.done", since);
    if (sees) {
      check(/lemon/i.test(done.data.text) && /orbit/i.test(done.data.text), `the model did not read the pictures: ${done.data.text}`);
    } else {
      // Said plainly, with the message, and the model is told too
      const notice = events.find((e) => e.at >= since && e.type === "error" && e.data.kind === "no_vision");
      check(notice, "nothing says the model cannot see the pictures");
      await page.locator(".bubble.error", { hasText: NO_VISION }).last().waitFor({ timeout: 5000 });
      check(!/lemon|orbit/i.test(done.data.text), `the model claims to see the pictures: ${done.data.text}`);
    }
    // Each opens large, as the core made it, and Escape closes it
    const sent = page.locator(".sent").filter({ has: page.locator('button.image[title="first.png"]') }).last();
    for (const name of ["first.png", "second.jpg"]) {
      await sent.locator(`button.image[title="${name}"]`).click();
      const viewer = page.locator(".viewer img");
      await viewer.waitFor({ timeout: 5000 });
      check((await viewer.getAttribute("alt")) === name, `the viewer shows ${await viewer.getAttribute("alt")}`);
      check((await viewer.getAttribute("src")).startsWith("data:image/jpeg"), "the viewer shows no picture");
      const size = await viewer.evaluate((img) => `${img.naturalWidth}x${img.naturalHeight}`);
      check(size === "640x360", `the picture is ${size}`);
      await page.keyboard.press("Escape");
      await page.locator(".viewer").waitFor({ state: "detached", timeout: 5000 });
    }
    await voiceDone();
  });

  /** The server and model the conversation runs on, to come back to */
  const usual = {};

  await step("attachments: a model that cannot see is told so with the pictures, and not again after", async () => {
    Object.assign(usual, { host: config().ollama_host, model: config().llm_model });
    // A server where no model sees, or its vision model would take the picture
    stubSees = false;
    await runOn(stubUrl, "stub-large");
    try {
      await toConversation();
      await handOver("paste", [lemon]);
      await page.locator('.draft.image img[alt="first.png"]').waitFor({ timeout: 5000 });
      const since = await ask("What is in this picture?");
      await eventAfter("chat.done", since, () => true, 60_000);
      check(events.some((e) => e.at >= since && e.type === "error" && e.data.kind === "no_vision"),
        "nothing says the model cannot see the picture");
      let sent = sentToStub();
      check(!sent.images.length, "pictures were sent to a model that cannot see");
      check(/cannot be shown/.test(sent.text), "the model is not told it cannot see the picture");
      const later = await ask("And something else entirely.");
      await eventAfter("chat.done", later, () => true, 60_000);
      sent = sentToStub();
      check(!/cannot be shown/.test(sent.text), "a message without pictures tells the model again that it cannot see them");
      await voiceDone();
    } finally {
      stubSees = true;
      await runOn(usual.host, usual.model);
    }
  });

  await step("attachments: ten files reach a model that sees: the documents as text, the pictures as images", async () => {
    await runOn(stubUrl, "stub-vision");
    try {
      await toConversation();
      const parts = Array.from({ length: 6 }, (_, i) => ({
        name: `part-${i + 1}.txt`,
        mimeType: "text/plain",
        buffer: Buffer.from(`Part ${i + 1} of the plan.\n`),
      }));
      await attach([notes, leaflet, ...parts]);
      await handOver("paste", [lemon]);
      await handOver("drop", [orbit]);
      await until("ten files in the message box", async () => (await drafts.count()) === 10);
      const since = await ask("Here are ten files.");
      const user = await eventAfter("chat.user", since, (data) => data.source === "text", 30_000);
      check(user.data.attachments?.length === 10, `${user.data.attachments?.length} files arrived`);
      await eventAfter("chat.done", since, () => true, 60_000);
      check(stubRequests.at(-1).model === "stub-vision", `the server was asked for ${stubRequests.at(-1).model}`);
      const sent = sentToStub();
      const shared = (name, text) => `<file name="${name}" shared="with this message">\n${text}`;
      for (const text of [
        shared("notes.txt", NOTES),
        shared("leaflet.pdf", `[Page 1]\n${LEAFLET}`),
        shared("part-6.txt", "Part 6 of the plan."),
        "first.png, second.jpg",
      ]) {
        check(sent.text.includes(text), `the model was not sent ${JSON.stringify(text)}`);
      }
      check(sent.images.length >= 2 && sent.images.every((url) => url.startsWith("data:image/jpeg;base64,")),
        `the model was sent ${sent.images.length} pictures`);
      await voiceDone();
    } finally {
      await runOn(usual.host, usual.model);
    }
  });

  await step("attachments: more than ten files, one over 25 MB, or one Iris cannot read are refused, and the draft comes back", async () => {
    await toConversation();
    const refused = async (files, text, error) => {
      await attach(files);
      await until(`${files.length} files in the message box`, async () => (await drafts.count()) === files.length);
      const since = await ask(text);
      await page.locator(".bubble.error", { hasText: error }).last().waitFor({ timeout: 60_000 });
      const took = Date.now() - since;
      check((await page.locator("textarea").inputValue()) === text, "the text did not come back");
      check((await drafts.count()) === files.length, "the files did not come back");
      check(!events.some((e) => e.at >= since && e.type === "chat.user"), "the message was taken");
      await clearDraft();
      return took;
    };
    const eleven = Array.from({ length: 11 }, (_, i) => ({
      name: `note-${i + 1}.txt`,
      mimeType: "text/plain",
      buffer: Buffer.from(`Note ${i + 1}.\n`),
    }));
    await refused(eleven, "Eleven notes.", "At most 10 files can go with one message");
    const big = { name: "big.txt", mimeType: "text/plain", buffer: Buffer.alloc(25 * 1024 * 1024 + 1, "a") };
    const took = await refused([big], "A big file.", "big.txt is larger than 25 MB");
    check(took < 10_000, `the file over 25 MB was refused only after ${(took / 1000).toFixed(1)} s`);
    const binary = { name: "data.bin", mimeType: "application/octet-stream", buffer: Buffer.from([0, 1, 2, 3]) };
    await refused([binary], "Here is a file.", "cannot read data.bin");
  });

  await step("conversation: Stop ends a reply at once", async () => {
    const since = Date.now();
    await page.locator("textarea").fill("Write a very long story about a lighthouse, at least a thousand words.");
    await page.locator("textarea").press("Enter");
    const stop = page.locator('button[title="Stop the current reply"]');
    await stop.waitFor({ timeout: 30_000 });
    await new Promise((resolve) => setTimeout(resolve, 1500));
    const pressed = Date.now();
    await stop.click();
    const done = await eventAfter("chat.done", since, () => true, 5000);
    check(done.data.interrupted, "the reply was not marked as stopped");
    check(done.at - pressed < 2000, `Stop took ${done.at - pressed} ms`);
    await until("Iris to be at rest", () => rpc("app.info").then((info) => !["thinking", "speaking"].includes(info.state)), 5000);
    await stop.waitFor({ state: "detached", timeout: 5000 });
  });

  await step("conversation: Stop works before the model has said anything", async () => {
    const since = Date.now();
    await page.locator("textarea").fill("Explain what a haiku is.");
    await page.locator("textarea").press("Enter");
    const stop = page.locator('button[title="Stop the current reply"]');
    await stop.waitFor({ timeout: 30_000 });
    const pressed = Date.now();
    await stop.click();
    const done = await eventAfter("chat.done", since, () => true, 5000);
    check(done.at - pressed < 2000, `Stop took ${done.at - pressed} ms`);
    await stop.waitFor({ state: "detached", timeout: 5000 });
  });

  /** Busy states and audible speech between two moments. */
  const activity = (from, to) => ({
    busy: events
      .filter((e) => e.at >= from && e.at <= to && e.type === "state")
      .map((e) => e.data.state)
      .filter((state) => ["thinking", "speaking", "searching", "browsing", "looking"].includes(state)),
    sound: events.filter((e) => e.at >= from && e.at <= to && e.type === "audio.output" && e.data.rms > 0.001).length,
  });
  const stopButton = page.locator('button[title="Stop the current reply"]');

  await step("Stop silences a reply being spoken, and Iris stays silent", async () => {
    const since = await ask("Count from one to forty in words, one number per sentence.");
    await afterThinking("the voice", since, () =>
      events.find((e) => e.at >= since && e.type === "audio.output" && e.data.rms > 0.001));
    await quiet(2500);
    const pressed = Date.now();
    await stopButton.click();
    await quiet(8000);
    const after = activity(pressed + 600, Date.now());
    check(after.sound === 0, `the voice went on: ${after.sound} audible moments after Stop`);
    await checkSilent(pressed + 600);
    check(after.busy.length === 0, `Iris went on: ${after.busy.join(" > ")}`);
    check(!(await stopButton.count()), "the Stop button is still there");
    const info = await rpc("app.info");
    check(!["thinking", "speaking"].includes(info.state), `Iris is ${info.state}`);
  });

  await step("after Stop the next message is answered and spoken", async () => {
    const since = await ask("Reply with exactly: ready when you are.");
    const done = await afterThinking("the reply", since, () =>
      events.find((e) => e.at >= since && e.type === "chat.done"));
    check(!done.data.interrupted && done.data.text.trim(), "the reply did not come");
    await eventAfter("audio.output", since, (data) => data.rms > 0.001, 60_000);
    await until("the voice to finish", () => rpc("app.info").then((info) => info.state !== "speaking"), 60_000);
    await checkHeard(since, "the reply after Stop");
  });

  await step("Stop while the reply is still being written silences it too", async () => {
    const since = await ask("Tell a story about a lighthouse keeper in about three hundred words.");
    await afterThinking("the reply's first words", since, () =>
      events.find((e) => e.at >= since && e.type === "chat.chunk"));
    await quiet(1200);
    const pressed = Date.now();
    await stopButton.click();
    await quiet(8000);
    const after = activity(pressed + 600, Date.now());
    const chunks = events.filter((e) => e.at >= pressed + 600 && e.type === "chat.chunk").length;
    check(chunks === 0, `the reply went on: ${chunks} more chunks`);
    check(after.sound === 0, `the voice went on: ${after.sound} audible moments after Stop`);
    await checkSilent(pressed + 600);
    check(after.busy.length === 0, `Iris went on: ${after.busy.join(" > ")}`);
  });

  await step("Stop during a tool ends the turn", async () => {
    const since = await ask("Search the web for the population of Reykjavik and tell me.");
    await afterThinking("a tool to run", since, () => events.find((e) => e.at >= since && e.type === "tool.use"));
    const pressed = Date.now();
    await stopButton.click();
    await quiet(10_000);
    const after = activity(pressed + 1500, Date.now());
    check(after.sound === 0, `the voice went on: ${after.sound} audible moments after Stop`);
    await checkSilent(pressed + 1500);
    check(after.busy.length === 0, `Iris went on: ${after.busy.join(" > ")}`);
    const info = await rpc("app.info");
    check(!["thinking", "speaking", "searching", "browsing"].includes(info.state), `Iris is ${info.state}`);
  });

  await step("a speaker that cannot be opened is told once, and Iris goes on: Pause, Resume and a preview work after", async () => {
    // As when AirPods disconnect under PortAudio
    fs.writeFileSync(AUDIO_FAULTS, JSON.stringify({ speaker: "broken" }));
    let since;
    try {
      await toConversation();
      since = await ask("Reply with exactly: the speaker is quiet.");
      const done = await eventAfter("chat.done", since, () => true, 180_000);
      check(done.data.text.trim(), "the reply did not come");
      const told = await eventAfter(
        "error",
        since,
        (data) => data.message.startsWith("The speaker could not be opened:"),
        60_000
      );
      check(told.data.message.includes("PaErrorCode -9986"), `told: ${told.data.message}`);
      await page.locator(".bubble.error", { hasText: "The speaker could not be opened" }).waitFor({ timeout: 5000 });
      // The reply is shown all the same
      await until("the reply in the conversation", async () =>
        (await page.locator(".bubble.assistant").last().innerText()).trim()
      );
      // The next reply is not spoken either, and not told again
      const again = await ask("Reply with exactly: still quiet.");
      await eventAfter("chat.done", again, () => true, 180_000);
      await voiceDone();
    } finally {
      fs.rmSync(AUDIO_FAULTS, { force: true });
    }
    const errors = events.filter((e) => e.at >= since && e.type === "error").map((e) => e.data.message);
    const speaker = errors.filter((message) => message.startsWith("The speaker could not be opened"));
    check(speaker.length === 1, `told ${speaker.length} times`);
    check(!errors.some((message) => message.startsWith("Assistant loop failed")), "the loop failed");
    check((await rpc("app.info")).running, "Iris no longer runs");
    // Pause and Resume still steer Iris
    await boxButton("Pause the conversation").click();
    await until("the paused state", async () => (await state()) === "paused");
    await boxButton("Resume the conversation").click();
    await until("the core to resume", () => config().paused === false);
    await until("Iris no longer paused", async () => (await state()) !== "paused", 5000);
    // The speaker works again: a preview plays, and a reply is spoken
    await openStage("Voice");
    const played = Date.now();
    await inspector.locator('button[aria-label="Play"]').click();
    await eventAfter("audio.output", played, (data) => data.rms > 0, 60_000);
    await voiceDone();
    const spoken = await ask("Reply with exactly: I can be heard again.");
    await eventAfter("chat.done", spoken, () => true, 180_000);
    await eventAfter("audio.output", spoken, (data) => data.rms > 0.001, 60_000);
    await voiceDone();
  });

  /** Whether a notice is in sight, below the conversation (not above it, where
   *  a long one hides it). */
  const inSight = async (notice) => {
    const shown = await notice.evaluate((el) => {
      const view = el.closest(".overflow-y-auto").getBoundingClientRect();
      const box = el.getBoundingClientRect();
      return box.bottom > view.top && box.top < view.bottom;
    });
    const said = await page.locator(".bubble.user").last().boundingBox();
    const box = await notice.boundingBox();
    return shown && (!said || box.y >= said.y + said.height);
  };

  await step("a reload rebuilds the conversation and the timings, and says the microphone is denied", async () => {
    await toConversation();
    // What was said, and the files sent with it
    const shown = () =>
      page.evaluate(() =>
        [...document.querySelectorAll(".sent")].map((sent) =>
          [...sent.querySelectorAll(".bubble.user, button.image img, .document")]
            .map((el) => el.getAttribute("alt") ?? el.textContent.trim())
            .join("+")
        ).join("|")
      );
    const before = await shown();
    const reloaded = Date.now();
    await page.evaluate(() => sessionStorage.setItem("e2e.mic", "denied"));
    await page.reload();
    await page.locator("textarea").waitFor({ timeout: 30_000 });
    const denied = page.locator(".bubble.error", { hasText: "Microphone access denied" });
    await denied.waitFor({ timeout: 10_000 });
    await until("the conversation to be rebuilt, with its files", async () => (await shown()) === before, 10_000);
    await quiet(500);
    check(await inSight(denied), "the microphone warning is out of sight, above the conversation");
    check(!calls.some((c) => c.at >= reloaded && c.method === "assistant.start"), "the reload started Iris again");
    await openSettings();
    await until("the model's time", async () => /\d+ ms/.test(await stage("Language model").locator(".ms").innerText()));
    await page.evaluate(() => sessionStorage.removeItem("e2e.mic"));
    await toConversation();
  });

  await step("presence shows only the orb, a click pauses, the mark and Escape lead back, and a silent microphone tints it", async () => {
    await toConversation();
    await rpc("mic.set", { on: false });
    // A warning left by an earlier step (the denied microphone) would be the
    // orb's to show, and the expand mark gives way to it: a reply after it,
    // so nothing needs the user
    await rpc("debug.emit", { type: "chat.done", data: { text: "Nothing needs you now.", interrupted: false } });
    await until("the reply", () => page.getByText("Nothing needs you now.").count(), 5000);
    const shown = presence.length;
    const orb = (await openPresence()).locator('[aria-label="iris"]');
    const mark = presencePage.locator(".mark");
    const at = (x, y) => presencePage.mouse.click(x, y);
    // Presence is a window of its own: the conversation window asks for it,
    // and presence holds nothing but the orb
    await page.locator('button[title="Presence"]').click();
    await until("presence", () => presence.at(-1) === true, 5000);
    check(await orb.locator("canvas").count(), "presence shows no orb");
    check(!(await presencePage.locator("textarea, header, nav, input").count()), "presence shows more than the orb");
    check(!(await orb.getAttribute("title")), "the orb shows a warning nothing calls for");
    // The orb says what Iris does, as the conversation window has it
    const state = (await rpc("app.info")).state;
    await until(`the orb to show ${state}`, async () => (await orb.getAttribute("data-state")) === state, 5000);
    // On the orb the expand mark shows; a click beside it pauses, and again resumes
    await presencePage.mouse.move(110, 110);
    await until("the expand mark", () => mark.evaluate((m) => m.classList.contains("shown")), 5000);
    let since = Date.now();
    await at(110, 110);
    await eventAfter("state", since, (data) => data.state === "paused", 5000);
    await until("the orb to rest", async () => (await orb.getAttribute("data-state")) === "paused", 5000);
    since = Date.now();
    await at(110, 110);
    await eventAfter("state", since, (data) => data.state !== "paused", 5000);
    check(presence.at(-1) === true, "a click on the orb left presence");
    // The mark leads back to the conversation
    await presencePage.mouse.move(146, 74);
    await at(146, 74);
    await until("the conversation", () => presence.at(-1) === false, 5000);
    // A microphone that delivers exact zeros (the core's own is off; its
    // levels are given to it as the microphone would)
    await page.locator('button[title="Presence"]').click();
    await until("presence", () => presence.at(-1) === true, 5000);
    since = Date.now();
    while (Date.now() - since < 8000) {
      await rpc("debug.emit", { type: "audio.level", data: { rms: 0 } });
      await quiet(100);
    }
    const warnings = events.filter((e) => e.at >= since && e.type === "error");
    check(warnings.length === 1, `${warnings.length} warnings for one silence`);
    const message = warnings[0].data.message;
    // It needs the user: the orb takes the danger tint, says it as its
    // title, and a click on it opens the conversation, where it is
    await until("the warning on the orb", async () => (await orb.getAttribute("title")) === message, 5000);
    await presencePage.mouse.move(110, 110);
    await quiet(300);
    check(!(await mark.evaluate((m) => m.classList.contains("shown"))), "the expand mark shows over the warning");
    await at(110, 110);
    await until("the conversation", () => presence.at(-1) === false, 5000);
    check((await page.locator(".bubble.error", { hasText: message.slice(0, 40) }).count()) === 1,
      "the warning is not in the conversation once");
    // Presence again, and Escape brings the conversation back
    await page.locator('button[title="Presence"]').click();
    await until("presence", () => presence.at(-1) === true, 5000);
    await presencePage.keyboard.press("Escape");
    await until("the conversation", () => presence.at(-1) === false, 5000);
    const went = presence.slice(shown).join();
    check(went === "true,false,true,false,true,false", `presence went ${went}`);
  });

  await step("the core stops: the app says so, starts it again, and goes on", async () => {
    await toConversation();
    const messages = page.locator(".bubble.user, .bubble.assistant, .bubble.error");
    const shown = await messages.count();
    const since = Date.now();
    process.kill(bridge.pid, "SIGKILL");
    await page.getByText("Iris stopped unexpectedly (stopped by SIGKILL). Starting it again.").waitFor({ timeout: 10_000 });
    await eventAfter("core.restarted", since, () => true, 30_000);
    // The app starts the assistant in the new core, with its settings
    await eventAfter("config", since, () => true, 180_000);
    check(config().wake_word === WAKE_WORD, "the new core was not started with the settings");
    check(config().llm_model === model, `the new core runs ${config().llm_model}`);
    await rpc("mic.set", { on: false });
    // The conversation stays shown, once
    check((await messages.count()) === shown + 1, `${shown} messages before, ${await messages.count()} after`);
    const typed = await ask("Reply with exactly: back again.");
    const done = await replyEvent("chat.done", typed);
    check(!done.data.interrupted && done.data.text.trim(), "the reply did not come");
  });

  /** Set a switch of the open stage, and wait for the core to apply it. */
  async function setSwitch(label, on, applied) {
    const control = inspector.locator(`button[role="switch"][aria-label="${label}"]`);
    if (((await control.getAttribute("aria-checked")) === "true") !== on) await control.click();
    await until(`${label} ${on ? "on" : "off"}`, applied);
  }

  await step("settings come back after a restart with a new core", async () => {
    // Choices to keep, made where people make them
    await openStage("Wake word");
    const sleepWord = inspector.locator("#sleep-word");
    await sleepWord.fill("Goodnight");
    await sleepWord.press("Enter");
    await until("the sleep word", () => config().sleep_word === "Goodnight");
    check((await rpc("persona.get", { id: "iris" })).sleep_word === "Goodnight", "the sleep word is not Iris's own");
    await inspector.locator("#stay-awake").selectOption("60");
    await until("a minute awake", () => config().wake_window === 60);
    await setSwitch("Sleep when idle", false, () => config().wake_window === 0);
    check(await inspector.locator("#stay-awake").isDisabled(), "the time can be chosen while Iris does not sleep when idle");
    await setSwitch("Sleep when idle", true, () => config().wake_window === 60);
    await openStage("Turn end");
    await option("Silence").click();
    await until("silence", () => config().turn_detection === "silence");
    await openStage("Tools");
    await setSwitch("Web search", false, () => !config().tools.web.enabled);
    await openStage("Output");
    const bargeIn = config().barge_in_available;
    if (bargeIn) await setSwitch("Interrupt by voice", false, () => config().barge_in === false);
    await setSwitch("Speak replies", false, () => config().speak_replies === false);
    await openStage("Persona");
    await option("HAL").click();
    await until("HAL", () => config().persona === "HAL", 60_000);
    // HAL's words: his name, and no sleep word
    check(config().wake_word === "HAL", `HAL is woken by ${config().wake_word}`);
    check(config().sleep_word === "", `HAL sleeps by ${config().sleep_word}`);
    // Remembered before amnesia: what was typed
    await openStage("Memory");
    await setSwitch("Remember conversations", false, () => config().amnesia === true);
    const kept = config();
    const keptServers = (await settings()).servers ?? [];

    const restarted = Date.now();
    await restartCore();
    // Launched once more without the microphone, and with the wake word kept
    // as it was before each persona had its own: it goes to the persona in use
    await page.evaluate(() => sessionStorage.setItem("e2e.mic", "denied"));
    await page.evaluate(() => {
      const kept = JSON.parse(localStorage.getItem("iris.settings") ?? "{}");
      delete kept.wake_on;
      localStorage.setItem("iris.settings", JSON.stringify({ ...kept, wake_word: "HAL" }));
    });
    await page.reload();
    const started = await until("the app to start Iris", () =>
      calls.find((c) => c.at >= restarted && c.method === "assistant.start"), 30_000);
    const options = started.params;
    const expected = {
      persona: "hal",
      llm_model: kept.llm_model,
      ollama_host: kept.ollama_host || undefined,
      speak_replies: false,
      ...(bargeIn ? { barge_in: false } : {}),
      amnesia: true,
      wake_on: true,
      wake_window: 60,
      wake_word: undefined,
      sleep_word: undefined,
      turn_detection: "silence",
    };
    for (const [key, value] of Object.entries(expected)) {
      check(options[key] === value, `Iris was started with ${key} ${JSON.stringify(options[key])}, not ${JSON.stringify(value)}`);
    }
    check(options.tools?.web === false, `Iris was started with the tools ${JSON.stringify(options.tools)}`);
    const migrated = await settings();
    check(migrated.wake_on === true && !("wake_word" in migrated), `the wake word was kept as ${JSON.stringify(migrated)}`);
    await eventAfter("config", restarted, () => true, 180_000);
    await rpc("mic.set", { on: false });
    const now = config();
    for (const key of ["persona", "llm_model", "ollama_host", "speak_replies", "barge_in", "amnesia",
      "wake_word", "wake_window", "sleep_word", "turn_detection"]) {
      check(now[key] === kept[key], `the core runs with ${key} ${JSON.stringify(now[key])}, not ${JSON.stringify(kept[key])}`);
    }
    check(now.tools.web.enabled === false, "web search is on again");
    // The app shows them
    await boxButton("Speak replies").waitFor({ timeout: 10_000 });
    await boxButton("Amnesia mode is on: nothing is remembered").waitFor({ timeout: 5000 });
    // The conversation goes on: its latest turns are shown again
    await until("the earlier conversation", async () => {
      const shown = await page.locator(".bubble.user").allInnerTexts();
      return shown.some((text) => asked.some((typed) => text.startsWith(typed)));
    }, 30_000);
    // Below it, in sight, that the microphone is denied
    await quiet(500);
    check(await inSight(page.locator(".bubble.error", { hasText: "Microphone access denied" })),
      "the microphone warning is out of sight, above the earlier conversation");
    await page.evaluate(() => sessionStorage.removeItem("e2e.mic"));
    // The servers kept, and the setup in use
    await openSettings();
    check((await current.innerText()).includes("My setup"), `the setup in use is ${await current.innerText()}`);
    await openStage("Language model");
    for (const url of keptServers) {
      await servers.filter({ has: page.locator(`.name[title="${url}"]`) }).waitFor({ timeout: 5000 });
    }

    // Back to how it was
    const stubRow = servers.filter({ hasText: stubHost });
    if (await stubRow.count()) await stubRow.locator('button[title="Remove"]').click();
    await openStage("Memory");
    await setSwitch("Remember conversations", true, () => config().amnesia === false);
    await openStage("Persona");
    await option("Iris").click();
    await until("Iris", () => config().persona === "Iris", 60_000);
    check(config().wake_word === WAKE_WORD, `Iris is woken by ${config().wake_word}`);
    check(config().sleep_word === "Goodnight", `Iris sleeps by ${config().sleep_word}`);
    await openStage("Output");
    await setSwitch("Speak replies", true, () => config().speak_replies === true);
    if (bargeIn) await setSwitch("Interrupt by voice", true, () => config().barge_in === true);
    await openStage("Tools");
    await setSwitch("Web search", true, () => config().tools.web.enabled);
    await openStage("Turn end");
    await option("Smart Turn").click();
    await until("Smart Turn", () => config().turn_detection === "smart_turn");
    await toConversation();
  });

  await step("voice cloning: Iris keeps her cloned voice after a restart, and gets her own back", async () => {
    check((await rpc("persona.get", { id: "iris" })).voice === "clone:iris", "the cloned voice was lost");
    await openSettings("Personas");
    await page.locator("button.card", { hasText: "Iris" }).first().click();
    await until("Iris's cloned voice", () => page.locator(".voice.active", { hasText: "Iris" }).count());
    const since = Date.now();
    await page.locator(".voice.active .play").click();
    await eventAfter("audio.output", since, (data) => data.rms > 0, 90_000);
    await voiceDone();
    await checkHeard(since, "the cloned voice's tile after a restart");
    // Back to a voice of the catalogue: the clone stays, to switch back to
    // By its name alone: the tile shows the languages it speaks beside it
    const shippedName = exactly(shippedVoice.replace(/^./, (c) => c.toUpperCase()));
    await page.locator(".voice button.pick", { has: page.locator(".name", { hasText: shippedName }) }).click();
    await page.locator("button.s-btn.primary", { hasText: "Save" }).click();
    await until("her own voice", () => config().tts_voice === shippedVoice, 10_000);
    const voices = await rpc("persona.voices");
    check(voices.engines.kyutai.some((v) => v.id === "clone:iris" && v.name === "Iris"), "the cloned voice is gone");
    await toConversation();
  });

  await step("the walkthrough runs again from Settings, skips and keeps, and Iris says hello", async () => {
    const before = config();
    await openSettings();
    await page.locator('nav[aria-label="Settings"] button', { hasText: "Walkthrough" }).click();
    await page.getByText("Welcome to Iris").waitFor({ timeout: 5000 });
    await footerButton("Get started").click();
    const moves = [["server", "Skip"], ["models", "Continue"], ["microphone", "Skip"], ["voice", "Continue"], ["extras", "Continue"]];
    for (const [name, move] of moves) {
      await walkthroughStep(name).waitFor({ timeout: 10_000 });
      await quiet(500);
      await footerButton(move).click();
    }
    await walkthroughStep("ready").waitFor({ timeout: 5000 });
    const since = Date.now();
    await footerButton("Done").click();
    await until("the conversation", () => page.locator("textarea").count(), 10_000);
    const greeting = await replyEvent("chat.done", since);
    check(greeting.data.text.trim().length > 0, "the greeting is empty");
    const after = config();
    for (const key of ["llm_model", "ollama_host", "wake_word", "sleep_word", "wake_window", "persona"]) {
      check(after[key] === before[key], `${key} changed from ${before[key]} to ${after[key]}`);
    }
    check(after.mic_on === before.mic_on, "the microphone is not as it was");
    await until("the voice to finish", () => rpc("app.info").then((info) => info.state !== "speaking"), 60_000);
  });

  // ------------------------------------------------------------- the Voices page

  const voicesPage = async () => {
    await openSettings("Voices");
    await page.locator('section[aria-label="Kyutai"]').waitFor({ timeout: 15_000 });
  };
  const tile = (voice) => page.locator(`.tile[data-voice="${voice}"]`);
  const card = (voice) => page.locator(`article.cloned[data-voice="${voice}"]`);
  const library = async () => (await rpc("voice.list")).engines;
  const cloneOf = async (name) => (await library()).kyutai.find((v) => v.cloned && v.name === name);
  const voiceFilter = page.locator('[role="group"][aria-label="Voices that speak"]');
  /** A voice was heard: the core reported its level, and the virtual
   *  speaker played it where the core tells (e2e.speaker) */
  const heard = async (since, timeout = 90_000) => {
    await eventAfter("audio.output", since, (data) => data.rms > 0, timeout);
    await voiceDone();
    const played = await rpc("e2e.speaker", { since }).catch(() => null);
    if (played) check(played.seconds > 0, "the speaker played nothing");
  };
  let sample = null;

  await step("voices: every voice is listed by engine, with its languages and the personas that speak with it", async () => {
    await voicesPage();
    const engines = await library();
    for (const [engine, label] of [["kyutai", "Kyutai"], ["kokoro", "Kokoro"]]) {
      const catalogue = (engines[engine] ?? []).filter((v) => !v.cloned);
      if (!catalogue.length) continue;
      const shown = await page.locator(`section[aria-label="${label}"] .tile`).count();
      check(shown === catalogue.length, `${label}: ${shown} tiles for ${catalogue.length} voices`);
    }
    // Each with the personas that speak with it, and the languages it speaks
    const iris = await rpc("persona.get", { id: "iris" });
    check(iris.engine === "kyutai", `Iris speaks with ${iris.engine}`);
    const irisVoice = page.locator(`[data-voice="${iris.voice}"]`);
    check((await irisVoice.locator(".user").allInnerTexts()).includes("Iris"), "Iris is not shown with her voice");
    const alba = engines.kyutai.find((v) => v.id === "alba");
    const codes = (await tile("alba").locator(".codes").innerText()).toLowerCase().split(/\s+/);
    check(codes.join() === alba.languages.join(), `Alba shows ${codes} for ${alba.languages}`);
    // A persona opens in the persona editor, which leads back here
    await irisVoice.locator(".user", { hasText: "Iris" }).click();
    await until("Iris's form", async () => (await page.locator("input.title").inputValue()) === "Iris");
    await page.locator("button.manage", { hasText: "Manage voices" }).click();
    await page.locator('section[aria-label="Kyutai"]').waitFor({ timeout: 5000 });
  });

  await step("voices: the language filter shows the voices that speak every language picked", async () => {
    await voicesPage();
    const all = Object.values(await library()).flat();
    const speaking = (...codes) => all.filter((v) => codes.every((c) => v.languages.includes(c)));
    const visible = () => page.locator(".tile, article.cloned").count();
    await voiceFilter.locator("button", { hasText: "French" }).click();
    await until("the French voices", async () => (await visible()) === speaking("fr").length);
    check(!(await tile("alba").count()), "Alba, who does not speak French, is shown");
    check(await tile("anna").count(), "Anna, who speaks French, is hidden");
    // A language no voice that speaks French speaks cannot be added
    check(await voiceFilter.locator("button", { hasText: "Japanese" }).isDisabled(), "Japanese can be added");
    await voiceFilter.locator("button", { hasText: "Hindi" }).click();
    await until("the voices that speak French and Hindi", async () => (await visible()) === speaking("fr", "hi").length);
    check(!(await page.locator('section[aria-label="Kyutai"]').count()), "Kyutai is shown without a voice");
    await voiceFilter.locator("button", { hasText: "Hindi" }).click();
    await voiceFilter.locator("button", { hasText: "French" }).click();
    await until("every voice", async () => (await visible()) === all.length);
  });

  await step("voices: a voice plays from its tile, through the editor's preview, and reaches the speaker", async () => {
    await voicesPage();
    const voice = (await library()).kyutai.find((v) => !v.cloned && v.id !== "anna" && v.ready.includes(v.languages[0]));
    check(voice, "no voice of Kyutai is ready");
    const since = Date.now();
    await tile(voice.id).locator(".s-play").click();
    await tile(voice.id).locator(".s-play.playing").waitFor({ timeout: 2000 });
    await heard(since);
    const asked = calls.find((c) => c.at >= since && c.method === "persona.preview");
    check(asked?.params.voice === voice.id && asked.params.name === voice.name, `the preview asked for ${JSON.stringify(asked?.params)}`);
    await tile(voice.id).locator(".s-play.playing").waitFor({ state: "detached", timeout: 5000 });
  });

  await step("voices: a voice is cloned here for the languages picked, plays, and is renamed", async () => {
    await voicesPage();
    huggingFace({ token: CLONE_TOKEN, terms: true });
    sample = sample ?? speechSample();
    await page.getByRole("button", { name: "Clone a voice" }).click();
    await page.locator('.clone[data-step="sample"]').waitFor({ timeout: 30_000 });
    // English, and French picked besides
    const languagesFor = page.locator('[aria-label="Languages to make the voice for"]');
    check((await languagesFor.locator("button", { hasText: "English" }).getAttribute("aria-pressed")) === "true", "English is not picked");
    await languagesFor.locator("button", { hasText: "French" }).click();
    const since = Date.now();
    await page.locator('input[aria-label="Audio file"]').setInputFiles({ name: "sample.wav", mimeType: "audio/wav", buffer: sample });
    // Cloned once it is made for every language picked
    await page.getByRole("button", { name: "Clone a voice" }).waitFor({ timeout: 180_000 });
    const made = await until("the voice", () => cloneOf("Cloned voice"), 10_000);
    check(made.cloned.made.includes("en") && made.cloned.made.includes("fr"), `the voice is made for ${made.cloned.made}`);
    check(events.some((e) => e.at >= since && e.type === "voice.job" && e.data.voice === made.id && e.data.language === "fr"), "making it for French was not shown");
    // It says the preview, then its name is to be given
    await heard(since);
    const name = card(made.id).locator('input[aria-label="Name of the voice"]');
    await until("the name to be asked for", () => name.evaluate((input) => document.activeElement === input));
    await name.fill("Testa");
    await name.press("Enter");
    await until("the new name", async () => (await cloneOf("Testa"))?.id === made.id);
    const again = Date.now();
    await card(made.id).locator(".s-play").click();
    await heard(again);
  });

  await step("voices: a cloned voice is cloned again from a new sample, and keeps its name", async () => {
    await voicesPage();
    const before = await cloneOf("Testa");
    check(before, "no voice named Testa");
    const folder = path.join(coreHome, ".iris", "voices", before.id.split(":")[1]);
    const old = fs.statSync(path.join(folder, "sample.wav")).mtimeMs;
    await card(before.id).getByRole("button", { name: "Clone again" }).click();
    await card(before.id).locator('.clone[data-step="sample"]').waitFor({ timeout: 30_000 });
    const since = Date.now();
    await card(before.id).locator('input[aria-label="Audio file"]').setInputFiles({ name: "again.wav", mimeType: "audio/wav", buffer: sample });
    await card(before.id).locator(".clone").waitFor({ state: "detached", timeout: 180_000 });
    await heard(since);
    const after = await cloneOf("Testa");
    check(after?.id === before.id, `cloned again, the voice is ${after?.id}`);
    check(fs.statSync(path.join(folder, "sample.wav")).mtimeMs > old, "the sample was not replaced");
    check(after.cloned.made.includes("en"), `made for ${after.cloned.made}`);
  });

  await step("voices: a cloned voice no persona speaks with is deleted", async () => {
    await voicesPage();
    const voice = await cloneOf("Testa");
    const folder = path.join(coreHome, ".iris", "voices", voice.id.split(":")[1]);
    await card(voice.id).getByRole("button", { name: "Delete" }).click();
    await card(voice.id).getByRole("button", { name: "Confirm delete" }).click();
    await card(voice.id).waitFor({ state: "detached", timeout: 10_000 });
    check(!(await cloneOf("Testa")) && !fs.existsSync(folder), "the voice is still there");
  });

  await step("voices: a voice a persona speaks with asks before it is deleted, and the persona speaks with Anna", async () => {
    // A voice made of the sample as the cloning steps make one, and Iris
    // speaking with it
    const data = sample.toString("base64");
    check((await rpc("voice.sample", { data, name: "sample.wav" })).ok, "the sample was refused");
    const made = await rpc("voice.clone", { name: "Guarded" });
    check(made.ok, made.error);
    const voice = made.voice.id;
    const iris = await rpc("persona.get", { id: "iris" });
    await rpc("persona.save", { ...iris, voice });
    await until("Iris on the voice", () => config().tts_voice === voice, 15_000);
    await openSettings("Pipeline");
    await voicesPage();
    await card(voice).locator(".user", { hasText: "Iris" }).waitFor({ timeout: 5000 });
    await card(voice).getByRole("button", { name: "Delete" }).click();
    const guard = card(voice).locator(".guard");
    await guard.getByText("Iris speaks with this voice, and will speak with Anna.").waitFor({ timeout: 5000 });
    await guard.getByRole("button", { name: "Cancel" }).click();
    check(await card(voice).count(), "the voice went without being deleted");
    await card(voice).getByRole("button", { name: "Delete" }).click();
    await card(voice).locator(".guard").getByRole("button", { name: "Delete" }).click();
    await card(voice).waitFor({ state: "detached", timeout: 10_000 });
    await until("Iris on Anna", () => config().tts_voice === "anna", 15_000);
    check((await rpc("persona.get", { id: "iris" })).voice === "anna", "Iris was not saved with Anna");
  });

  await step("voices: a reply its voice is not ready for is shown with its download, and spoken once the voice is ready", async () => {
    // Italian's model is not here: the stand-in downloads it slowly
    huggingFace({ hidden: ["italian"] });
    check(config().tts_voice === "anna", `Iris speaks with ${config().tts_voice}`);
    const question = "Reply only in Italian, in one short sentence: what is the capital of Italy?";
    const since = await ask(question);
    const done = await eventAfter("chat.done", since, () => true, 180_000);
    const job = await eventAfter("voice.job", since, (data) => data.voice === "anna" && data.language === "it", 30_000);
    check(job.data.origin === "conversation", `the download was started by ${job.data.origin}`);
    // The reply is shown at once, and the download in the conversation and
    // under the orb, with its progress, while it goes on
    await page.locator(".bubble.assistant").last().waitFor();
    const line = page.locator(`.job[data-job="${job.data.id}"]`);
    await line.waitFor({ timeout: 5000 });
    await until("its progress", async () => / MB, \d+ %/.test(await line.innerText()), 10_000);
    check((await line.innerText()).includes("Iris will speak Italian once it's done"), `the line says: ${await line.innerText()}`);
    check(await page.locator(`[data-job="${job.data.id}"]`).count() >= 2, "the download is not shown under the orb");
    const ready = await eventAfter("voice.job", since, (data) => data.id === job.data.id && data.state === "ready", 60_000);
    // Nothing was spoken of the reply meanwhile
    const sounded = events.filter((e) => e.type === "audio.output" && e.data.rms > 0 && e.at >= done.at && e.at <= ready.at);
    check(!sounded.length, `${sounded.length} audible moments while the voice was not ready`);
    await page.getByText("Anna's voice is ready for Italian").waitFor({ timeout: 5000 });
    // The next reply in Italian is spoken, and reaches the speaker
    const next = await ask(question);
    const timing = await eventAfter("turn.timing", next, () => true, 180_000);
    check(timing.data.components.voice === "kyutai:anna", `spoken by ${timing.data.components.voice}`);
    await heard(next);
  });

  await step("nothing went wrong in the page", async () => {
    check(pageErrors.length === 0, pageErrors.slice(0, 3).join(" | "));
  });
} finally {
  await browser.close().catch(() => {});
  server.close();
  stub.close();
  await rpc("assistant.stop").catch(() => {});
  bridge.kill();
  fs.rmSync(home, { recursive: true, force: true });
}

const failed = results.filter((result) => !result.ok);
console.log(`\n${results.length - failed.length} passed, ${failed.length} failed`);
process.exit(failed.length ? 1 : 0);
