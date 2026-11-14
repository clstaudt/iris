// The bridge to Iris's core, as the app runs it, with a stand-in core
// (fake-core.mjs): a core that stops unexpectedly is reported and started
// again, up to a limit; one that cannot be started is reported once; quitting
// stops it, also when it does not listen.
//
//   npm run test:bridge

import assert from "node:assert/strict";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

import { MAX_RESTARTS, PythonBridge } from "../electron/python-bridge.ts";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FAKE_CORE = path.join(HERE, "fake-core.mjs");

/** A bridge to the stand-in core in this mode, and the events it tells the app. */
function bridged(mode, command = { command: process.execPath, args: [FAKE_CORE, mode] }) {
  const events = [];
  const bridge = new PythonBridge(HERE, false, "", {}, command);
  bridge.onLog = () => {};
  bridge.onEvent = (type, data) => events.push({ type, data });
  return { bridge, events };
}

async function until(what, condition, timeout = 10_000) {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    if (condition()) return;
    await new Promise((resolve) => setTimeout(resolve, 20));
  }
  throw new Error(`Timed out waiting for ${what}`);
}

const errors = (events) => events.filter((e) => e.type === "error").map((e) => e.data);
const count = (events, type) => events.filter((e) => e.type === type).length;
const alive = (pid) => {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
};

test("a core that stops unexpectedly is reported and started again", async () => {
  const { bridge, events } = bridged("answer");
  try {
    const { pid } = await bridge.call("app.info");
    // A request in flight when it stops is answered with the failure
    await assert.rejects(bridge.call("die", { code: 1, error: "RuntimeError: out of memory" }), /exited/);
    await until("the core to be started again", () => count(events, "core.restarted") === 1);
    assert.deepEqual(errors(events), [
      {
        message: "Iris stopped unexpectedly (exit code 1: RuntimeError: out of memory). Starting it again.",
        kind: "core",
      },
    ]);
    assert.ok(events.some((e) => e.type === "state" && e.data.state === "starting"));
    const again = await bridge.call("app.info");
    assert.notEqual(again.pid, pid);
  } finally {
    bridge.kill();
  }
});

test("a core killed by a signal says so", async () => {
  const { bridge, events } = bridged("answer");
  try {
    await bridge.call("app.info");
    process.kill(bridge.pid, "SIGKILL");
    await until("the core to be started again", () => count(events, "core.restarted") === 1);
    assert.equal(errors(events)[0].message, "Iris stopped unexpectedly (stopped by SIGKILL). Starting it again.");
  } finally {
    bridge.kill();
  }
});

test("a core that keeps stopping is left stopped after a few tries", async () => {
  const { bridge, events } = bridged("no-start");
  try {
    await until("the bridge to give up", () => errors(events).length === MAX_RESTARTS + 1, 20_000);
    // Give it the time of another restart: none comes
    await new Promise((resolve) => setTimeout(resolve, 1500));
    assert.equal(count(events, "core.restarted"), MAX_RESTARTS);
    const last = errors(events).at(-1).message;
    assert.match(last, /^Iris stopped unexpectedly again \(exit code 1: ModuleNotFoundError: No module named 'iris'\)/);
    assert.match(last, /Quit and reopen Iris\.$/);
    assert.equal(events.at(-1).type, "state");
    assert.equal(events.at(-1).data.state, "idle");
    await assert.rejects(bridge.call("app.info"), /not running/);
  } finally {
    bridge.kill();
  }
});

test("a core that cannot be started is reported once and not tried again", async () => {
  const missing = path.join(HERE, "no-such-python");
  const { bridge, events } = bridged(null, { command: missing, args: [] });
  try {
    await until("the failure to be told", () => errors(events).length === 1);
    await new Promise((resolve) => setTimeout(resolve, 1500));
    assert.equal(errors(events).length, 1);
    assert.match(errors(events)[0].message, /^Iris could not start: spawn .*no-such-python ENOENT$/);
    assert.equal(count(events, "core.restarted"), 0);
    await assert.rejects(bridge.call("app.info"), /not running/);
  } finally {
    bridge.kill();
  }
});

test("a request written to a core that stopped reading does not bring the app down", async () => {
  const { bridge, events } = bridged("deaf");
  try {
    await new Promise((resolve) => setTimeout(resolve, 200));
    // The pipe is closed: the write fails, and the request fails with the core
    await assert.rejects(bridge.call("app.info"), /exited \(code 3\)/);
    await until("the core to be started again", () => count(events, "core.restarted") === 1);
  } finally {
    bridge.kill();
  }
});

test("quitting stops the core without calling it a crash, even one that ignores SIGTERM", async () => {
  const { bridge, events } = bridged("stubborn");
  await bridge.call("app.info");
  const pid = bridge.pid;
  bridge.kill();
  // SIGTERM is ignored; SIGKILL follows after three seconds
  await new Promise((resolve) => setTimeout(resolve, 1000));
  assert.ok(alive(pid), "the core went at SIGTERM, which it ignores");
  await until("the core to be gone", () => !alive(pid), 5000);
  assert.deepEqual(errors(events), []);
  assert.equal(count(events, "core.restarted"), 0);
});
