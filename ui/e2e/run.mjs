// End-to-end test of the app's UI against the real Python core.
//
// The built UI (ui/dist) runs in headless Chrome; window.iris, which Electron's
// preload provides in the app, is bridged to a core started for the test. The
// core gets a home folder of its own, so the tester's personas, memory and
// settings are left alone; the speech models already on this computer are used.
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
//               several split by "|"; the steps that start Iris and choose
//               the model always run
//   E2E_HEADED  set to watch the browser

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
const realHome = os.homedir();
const python =
  process.env.E2E_PYTHON ??
  path.join(REPO, ".venv", process.platform === "win32" ? "Scripts/python.exe" : "bin/python");

/** Every event the core pushed, in order. */
const events = [];
const pending = new Map();
let nextId = 1;
let pushEvent = () => {};
let core;

/** Start a core; a restart starts another with the same home and memory. */
function startCore(log) {
  core = spawn(python, ["-m", "irisdev.e2e_core"], {
    cwd: REPO,
    env: {
      ...process.env,
      HOME: home,
      USERPROFILE: home,
      HF_HOME: process.env.HF_HOME ?? path.join(realHome, ".cache", "huggingface"),
      IRIS_MODELS_DIR: process.env.IRIS_MODELS_DIR ?? path.join(realHome, ".iris", "models"),
      IRIS_MEMORY_DB: path.join(home, "memory.db"),
      PYTHONPATH: path.join(REPO, "src"),
      PYTHONUNBUFFERED: "1",
    },
  });
  const coreLog = fs.createWriteStream(path.join(OUT, log));
  core.stderr.pipe(coreLog);
  core.stdin.on("error", () => {}); // a request to a core on its way out
  core.on("exit", () => {
    for (const { reject } of pending.values()) reject(new Error("the core exited"));
    pending.clear();
  });
  readline.createInterface({ input: core.stdout }).on("line", (line) => {
    let frame;
    try {
      frame = JSON.parse(line);
    } catch {
      coreLog.write(line + "\n");
      return;
    }
    if (frame.id !== undefined && pending.has(frame.id)) {
      const { resolve, reject } = pending.get(frame.id);
      pending.delete(frame.id);
      if (frame.error) reject(new Error(frame.error.message));
      else resolve(frame.result);
    } else if (frame.method === "event") {
      events.push({ ...frame.params, at: Date.now() });
      pushEvent(frame.params.type, frame.params.data);
    }
  });
}

/** Stop the core as quitting the app does, and start a new one. */
async function restartCore() {
  await Promise.race([rpc("assistant.stop").catch(() => {}), quiet(5000)]);
  const exited = new Promise((resolve) => core.once("exit", resolve));
  core.kill();
  await exited;
  startCore("core-2.log");
}

function rpc(method, params = {}) {
  return new Promise((resolve, reject) => {
    if (core.exitCode !== null || core.signalCode !== null || !core.stdin.writable) {
      reject(new Error("the core is not running"));
      return;
    }
    const id = nextId++;
    pending.set(id, { resolve, reject });
    core.stdin.write(JSON.stringify({ jsonrpc: "2.0", id, method, params }) + "\n");
  });
}

startCore("core.log");

// ------------------------------------------------- a model server of its own

// A server the test runs itself, with the OpenAI-compatible API, so that
// switching servers is tested whatever E2E_HOST is, and what a model is sent
// can be seen. Its models give one fixed reply: markdown, with an emoji and
// HTML a page could carry. One of them says it sees images.
const STUB_MODELS = [
  { id: "stub-large" },
  { id: "stub-small" },
  { id: "stub-vision", capabilities: ["vision"] },
];
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
      response.end(JSON.stringify({ object: "list", data: STUB_MODELS.map((m) => ({ ...m, object: "model" })) }));
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
  const file = path.join(DIST, request.url === "/" ? "index.html" : request.url.split("?")[0]);
  if (request.url === "/favicon.ico") {
    response.writeHead(204).end();
    return;
  }
  if (!file.startsWith(DIST) || !fs.existsSync(file)) {
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
await page.exposeFunction("__irisPresence", (on) => presence.push(on));
await page.addInitScript(
  ({ settings }) => {
    // Seeded once; a reload finds what the app saved
    if (localStorage.getItem("iris.settings") === null) {
      localStorage.setItem("iris.settings", JSON.stringify(settings));
    }
    const listeners = [];
    window.__irisEvent = (type, data) => listeners.forEach((listener) => listener(type, data));
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
      isPackaged: async () => false,
      chooseFolder: async () => null,
      presence: false,
      setPresence: (on) => window.__irisPresence(on),
      platform: "test",
    };
  },
  {
    settings: {
      setup_done: true,
      // Iris sleeps until a word nobody says
      wake_word: WAKE_WORD,
      ...(process.env.E2E_HOST ? { ollama_host: process.env.E2E_HOST } : {}),
    },
  }
);
pushEvent = (type, data) => {
  page.evaluate(([t, d]) => window.__irisEvent?.(t, d), [type, data]).catch(() => {});
};

// ------------------------------------------------------------------ helpers

const results = [];

async function step(name, body) {
  const only = process.env.E2E_ONLY?.toLowerCase().split("|");
  const always = ["the app starts", "language model: the server's models"];
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
async function leavePresence() {
  const leave = page.locator('main.presence button[title="Conversation"]');
  if (await leave.count()) await leave.click();
}

async function openSettings(pageName = "Pipeline") {
  await leavePresence();
  if (!(await page.locator('nav[aria-label="Settings"]').count())) {
    await page.locator('button[title="Settings"]').click();
  }
  await page.locator('nav[aria-label="Settings"] button', { hasText: pageName }).click();
}

const config = () => events.findLast((e) => e.type === "config")?.data;
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

// -------------------------------------------------------------------- steps

console.log(`Iris UI against the core (${python})`);
let model = process.env.E2E_MODEL ?? "";

try {
  await page.goto(url);

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
  });

  await step("voice: the persona's voice plays", async () => {
    await openStage("Voice");
    const since = Date.now();
    await inspector.locator('button[aria-label="Play"]').click();
    await eventAfter("audio.output", since, (data) => data.rms > 0, 60_000);
    await until("the voice to finish", () => rpc("app.info").then((info) => info.state !== "speaking"), 30_000);
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
    await page.locator("#persona-personality").fill("You are Testa, made by a test.");
    await page.locator("button.s-btn.primary", { hasText: "Save" }).click();
    await until("Testa in the list", () => page.locator("button.card", { hasText: "Testa" }).count());
    await until("Testa to be active", () => config().persona === "Testa", 60_000);
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
    await rpc("wake.set", { word: "" });
    await rpc("mic.set", { on: true });
    await until("Iris to listen", () => rpc("app.info").then((info) => info.state === "listening"), 15_000);
    const since = Date.now();
    await page.locator(".voice .play").nth(1).click();
    await eventAfter("audio.output", since, (data) => data.rms > 0, 60_000);
    await until("the voice to finish", () => rpc("app.info").then((info) => info.state !== "speaking"), 30_000);
    await new Promise((resolve) => setTimeout(resolve, 6000));
    await rpc("mic.set", { on: false });
    await rpc("wake.set", { word: WAKE_WORD });
    const heard = events.find((e) => e.at >= since && e.type === "chat.user");
    check(!heard, `the preview was heard as the user saying: ${heard?.data.text}`);
    // And once more: the voice still plays after the first preview
    const again = Date.now();
    await page.locator(".voice .play").nth(2).click();
    await eventAfter("audio.output", again, (data) => data.rms > 0, 60_000);
    await until("the voice to finish", () => rpc("app.info").then((info) => info.state !== "speaking"), 30_000);
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
    const timing = await eventAfter("turn.timing", since, () => true, 180_000);
    check(timing.data.components.voice === voice, `the reply was spoken by ${timing.data.components.voice}, not ${voice}`);
    await voiceDone();
    // Reset brings her shipped voice back, at once too
    await openSettings("Personas");
    await page.locator("button.card", { hasText: "Iris" }).first().click();
    await until("Iris's form", async () => (await page.locator("input.title").inputValue()) === "Iris");
    await page.locator("button.s-btn.danger", { hasText: "Reset" }).click();
    await page.locator("button.s-btn.danger", { hasText: "Confirm reset" }).click();
    await until("her shipped voice", () => config().tts_voice === shipped, 10_000);
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
    const done = await eventAfter("chat.done", since, () => true, 180_000);
    const error = events.find((e) => e.at >= since && e.type === "error");
    check(!error, `the reply failed: ${error?.data.message}`);
    check(done.data.text.trim().length > 0, "the reply is empty");
    await until("the reply in the conversation", () => page.getByText(done.data.text.trim().slice(0, 12)).count());
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
    const done = await eventAfter("chat.done", since, () => true, 180_000);
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
    const done = await eventAfter("chat.done", since, () => true, 180_000);
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
    const done = await eventAfter("chat.done", since, () => true, 180_000);
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
    const done = await eventAfter("chat.done", since, () => true, 180_000);
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
    await eventAfter("audio.output", since, (data) => data.rms > 0.001, 180_000);
    await quiet(2500);
    const pressed = Date.now();
    await stopButton.click();
    await quiet(8000);
    const after = activity(pressed + 600, Date.now());
    check(after.sound === 0, `the voice went on: ${after.sound} audible moments after Stop`);
    check(after.busy.length === 0, `Iris went on: ${after.busy.join(" > ")}`);
    check(!(await stopButton.count()), "the Stop button is still there");
    const info = await rpc("app.info");
    check(!["thinking", "speaking"].includes(info.state), `Iris is ${info.state}`);
  });

  await step("after Stop the next message is answered and spoken", async () => {
    const since = await ask("Reply with exactly: ready when you are.");
    const done = await eventAfter("chat.done", since, () => true, 180_000);
    check(!done.data.interrupted && done.data.text.trim(), "the reply did not come");
    await eventAfter("audio.output", since, (data) => data.rms > 0.001, 60_000);
    await until("the voice to finish", () => rpc("app.info").then((info) => info.state !== "speaking"), 60_000);
  });

  await step("Stop while the reply is still being written silences it too", async () => {
    const since = await ask("Tell a story about a lighthouse keeper in about three hundred words.");
    await eventAfter("chat.chunk", since, () => true, 180_000);
    await quiet(1200);
    const pressed = Date.now();
    await stopButton.click();
    await quiet(8000);
    const after = activity(pressed + 600, Date.now());
    const chunks = events.filter((e) => e.at >= pressed + 600 && e.type === "chat.chunk").length;
    check(chunks === 0, `the reply went on: ${chunks} more chunks`);
    check(after.sound === 0, `the voice went on: ${after.sound} audible moments after Stop`);
    check(after.busy.length === 0, `Iris went on: ${after.busy.join(" > ")}`);
  });

  await step("Stop during a tool ends the turn", async () => {
    const since = await ask("Search the web for the population of Reykjavik and tell me.");
    await until("a tool to run", () => events.find((e) => e.at >= since && e.type === "tool.use"), 180_000);
    const pressed = Date.now();
    await stopButton.click();
    await quiet(10_000);
    const after = activity(pressed + 1500, Date.now());
    check(after.sound === 0, `the voice went on: ${after.sound} audible moments after Stop`);
    check(after.busy.length === 0, `Iris went on: ${after.busy.join(" > ")}`);
    const info = await rpc("app.info");
    check(!["thinking", "speaking", "searching", "browsing"].includes(info.state), `Iris is ${info.state}`);
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

  await step("a silent microphone is warned about once, also under the orb, and presence comes and goes", async () => {
    await toConversation();
    await rpc("mic.set", { on: false });
    // Only the orb shows
    await page.locator('button[title="Presence"]').click();
    await until("only the orb", async () => !(await page.locator("textarea").count()));
    check(presence.at(-1) === true, "the window was not asked to shrink");
    check(!(await page.locator('button[title="Settings"]').isVisible()), "the top bar still shows");
    check(await page.locator('[aria-label="assistant status"]').isVisible(), "the orb is gone");
    // A microphone that delivers exact zeros (the core's own is off; its
    // levels are given to it as the microphone would)
    const since = Date.now();
    while (Date.now() - since < 8000) {
      await rpc("debug.emit", { type: "audio.level", data: { rms: 0 } });
      await quiet(100);
    }
    const warnings = events.filter((e) => e.at >= since && e.type === "error");
    check(warnings.length === 1, `${warnings.length} warnings for one silence`);
    const message = warnings[0].data.message;
    // It needs the user: it shows under the orb, and opens the conversation
    const notice = page.locator("button.presence-error", { hasText: message.slice(0, 20) });
    await notice.waitFor({ timeout: 5000 });
    await notice.click();
    await page.locator("textarea").waitFor({ timeout: 5000 });
    check(presence.at(-1) === false, "the window was not asked to grow");
    check((await page.locator(".bubble.error", { hasText: message.slice(0, 40) }).count()) === 1,
      "the warning is not in the conversation once");
    // Presence again, and Escape brings the conversation back
    await page.locator('button[title="Presence"]').click();
    await until("only the orb", async () => !(await page.locator("textarea").count()));
    await page.keyboard.press("Escape");
    await page.locator("textarea").waitFor({ timeout: 5000 });
    check(presence.join() === "true,false,true,false", `presence went ${presence.join()}`);
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
    await inspector.locator("#stay-awake").selectOption("60");
    await until("a minute awake", () => config().wake_window === 60);
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
    // Remembered before amnesia: what was typed
    await openStage("Memory");
    await setSwitch("Remember conversations", false, () => config().amnesia === true);
    const kept = config();
    const keptServers = (await settings()).servers ?? [];

    const restarted = Date.now();
    await restartCore();
    // Launched once more without the microphone
    await page.evaluate(() => sessionStorage.setItem("e2e.mic", "denied"));
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
      wake_word: WAKE_WORD,
      wake_window: 60,
      sleep_word: "Goodnight",
      turn_detection: "silence",
    };
    for (const [key, value] of Object.entries(expected)) {
      check(options[key] === value, `Iris was started with ${key} ${JSON.stringify(options[key])}, not ${JSON.stringify(value)}`);
    }
    check(options.tools?.web === false, `Iris was started with the tools ${JSON.stringify(options.tools)}`);
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
