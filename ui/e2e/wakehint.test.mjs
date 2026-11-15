// The wake word hint's timing, with short intervals: shown as Iris falls
// asleep, hidden, shown again each interval, and gone once stopped.
//
//   npm run test:wake-hint

import assert from "node:assert/strict";
import { test } from "node:test";

import { wakeHint } from "../src/lib/wakehint.ts";

const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

test("the hint shows at once, then again every interval, until stopped", async () => {
  const start = Date.now();
  const changes = [];
  const stop = wakeHint((on) => changes.push({ on, at: Date.now() - start }), { first: 100, every: 300, length: 60 });
  assert.deepEqual(changes.map((c) => c.on), [true], "the hint is not shown at once");
  await wait(750);
  stop();
  // Shown at 0 for 100 ms, at 300 and 600 for 60 ms each, then stopped
  assert.deepEqual(changes.map((c) => c.on), [true, false, true, false, true, false, false]);
  const [, firstOff, second, secondOff, third] = changes.map((c) => c.at);
  assert.ok(firstOff >= 95 && firstOff < 200, `the first showing ended at ${firstOff} ms`);
  assert.ok(second >= 295 && second < 400, `the second showing began at ${second} ms`);
  assert.ok(secondOff - second >= 55 && secondOff - second < 150, `the second showing lasted ${secondOff - second} ms`);
  assert.ok(third >= 595 && third < 700, `the third showing began at ${third} ms`);
  await wait(400);
  assert.equal(changes.length, 7, "the hint came back after it was stopped");
});

test("stopped while hidden, the hint stays hidden", async () => {
  const changes = [];
  const stop = wakeHint((on) => changes.push(on), { first: 20, every: 200, length: 20 });
  await wait(80);
  stop();
  await wait(250);
  assert.deepEqual(changes, [true, false, false]);
});
