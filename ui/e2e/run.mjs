// End-to-end test of the app's UI against the real Python core.
//
// The built UI (ui/dist) runs in headless Chrome; window.iris, which Electron's
// preload provides in the app, is bridged to a core started for the test. The
// core gets a home folder of its own, so the tester's personas, memory and
// settings are left alone; the speech models already on this computer are used.
// Its microphone and speaker are a virtual room (irisdev.e2e_core), which
// holds only Iris's own voice coming back: the real room is never heard, and
// nothing sounds on the speakers. Iris sleeps until a wake word nobody says.
//
//   npm run build && npm run e2e
//
// Environment:
//   E2E_MODEL   language model to talk to (default: the first the server lists
//               with tools)
//   E2E_HOST    model server URL (default: the core's default)
//   E2E_PYTHON  Python to run the core with (default: <repo>/.venv/bin/python)
//   E2E_ONLY    run only the steps whose name contains this text
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
const core = spawn(python, ["-m", "irisdev.e2e_core"], {
  cwd: REPO,
  env: {
    ...process.env,
    HOME: home,
    USERPROFILE: home,
    HF_HOME: process.env.HF_HOME ?? path.join(realHome, ".cache", "huggingface"),
    IRIS_MODELS_DIR: process.env.IRIS_MODELS_DIR ?? path.join(realHome, ".iris", "models"),
    IRIS_MEMORY_DB: path.join(home, "memory.db"),
  },
  { command: python, args: ["-m", "iris.rpc_server"] }
);
bridge.onLog = (line) => coreLog.write(line + "\n");
bridge.onEvent = (type, data) => {
  events.push({ type, data, at: Date.now() });
  pushEvent(type, data);
};

function rpc(method, params = {}) {
  return bridge.call(method, params);
}

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

await page.exposeFunction("__irisRpc", (method, params) => rpc(method, params));
await page.addInitScript(
  ({ settings }) => {
    localStorage.setItem("iris.settings", JSON.stringify(settings));
    const listeners = [];
    window.__irisEvent = (type, data) => listeners.forEach((listener) => listener(type, data));
    window.iris = {
      rpc: (method, params) => window.__irisRpc(method, params ?? {}),
      onEvent: (callback) => {
        listeners.push(callback);
        return () => listeners.splice(listeners.indexOf(callback), 1);
      },
      micStatus: async () => "granted",
      isPackaged: async () => false,
      chooseFolder: async () => null,
      presence: false,
      setPresence: async () => {},
      onOrb: () => () => {},
      sendOrbInputs: () => {},
      onOrbInputs: () => () => {},
      presenceCommand: () => {},
      onPresenceCommand: () => () => {},
      presenceHover: () => {},
      presenceDrag: async () => false,
      platform: "test",
    };
  },
  {
    settings: {
      setup_done: true,
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
  const only = process.env.E2E_ONLY?.toLowerCase();
  if (only && !name.toLowerCase().includes(only) && !name.startsWith("the app starts")) return;
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
    const detail = ` [states: ${states.join(" > ") || "none"}]` + (errors.length ? ` [errors: ${errors.join(" | ")}]` : "");
    results.push({ name, ok: false, error: String(error.message ?? error).split("\n")[0] + detail, shot });
    console.log(`  FAIL  ${name}\n        ${results.at(-1).error}\n        ${shot}`);
  }
}

function check(condition, message) {
  if (!condition) throw new Error(message);
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

async function openSettings(pageName = "Pipeline") {
  if (!(await page.locator('nav[aria-label="Settings"]').count())) {
    await page.locator('button[title="Settings"]').click();
  }
  await page.locator('nav[aria-label="Settings"] button', { hasText: pageName }).click();
}

const config = () => events.findLast((e) => e.type === "config")?.data;

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
  const quiet = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  async function ask(text) {
    await page.locator("nav button", { hasText: "Conversation" }).click().catch(() => {});
    await page.locator("textarea").fill(text);
    await page.locator("textarea").press("Enter");
    return Date.now();
  }

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

  await step("the core stops: the app says so, starts it again, and goes on", async () => {
    await page.locator("nav button", { hasText: "Conversation" }).click().catch(() => {});
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
    const asked = await ask("Reply with exactly: back again.");
    const done = await eventAfter("chat.done", asked, () => true, 180_000);
    check(!done.data.interrupted && done.data.text.trim(), "the reply did not come");
  });

  await step("nothing went wrong in the page", async () => {
    check(pageErrors.length === 0, pageErrors.slice(0, 3).join(" | "));
  });
} finally {
  await browser.close().catch(() => {});
  server.close();
  await rpc("assistant.stop").catch(() => {});
  bridge.kill();
  fs.rmSync(home, { recursive: true, force: true });
}

const failed = results.filter((result) => !result.ok);
console.log(`\n${results.length - failed.length} passed, ${failed.length} failed`);
process.exit(failed.length ? 1 : 0);
