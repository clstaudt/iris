// The orb's keep: the conversation window counts each memory Iris writes,
// as the core reports it, and an orb plays its keep once per write it is
// told of. A forget, or a write while memory is off or the thread is in
// amnesia, plays nothing; a keep begins and ends at rest, so one that
// follows another shows no jump.
//
//   npm run test:keep

import assert from "node:assert/strict";
import { test } from "node:test";

import { Keep, KEEP_DURATION, keepLight, written } from "../src/lib/keep.ts";

// One frame at 60 Hz, in seconds
const FRAME = 1 / 60;

/** The conversation window's store, as far as the keep goes: the core's
 *  memory.updated in, the writes counted. */
function conversation({ memoryOn = true, amnesia = false } = {}) {
  const store = {
    memoryOn,
    amnesia,
    kept: 0,
    /** The core's memory.updated */
    updated: (data) => {
      if (written(data, store)) store.kept++;
    },
  };
  return store;
}

/** Frames for this many seconds. */
function run(orb, store, seconds) {
  for (let t = 0; t < seconds - 1e-9; t += FRAME) orb.frame(store, FRAME);
}

test("a memory written plays the keep once, and it ends after its time", () => {
  const store = conversation();
  const orb = new Keep(store.kept);
  orb.frame(store, FRAME);
  store.updated({ written: true });
  orb.frame(store, FRAME);
  assert.equal(orb.played, 1, "the keep did not play");
  run(orb, store, KEEP_DURATION / 2);
  assert.ok(orb.progress > 0.4 && orb.progress < 0.6, `halfway, the keep is at ${orb.progress}`);
  run(orb, store, KEEP_DURATION / 2 + FRAME);
  assert.equal(orb.progress, 1, "the keep did not end");
  run(orb, store, 2);
  assert.equal(orb.played, 1, `the keep played ${orb.played} times for one write`);
});

test("a forget, a thread opened, or memory off or in amnesia plays nothing", () => {
  const store = conversation();
  const orb = new Keep(store.kept);
  // Forgetting a topic or a thread, and opening one, update memory too
  store.updated({});
  orb.frame(store, FRAME);
  // Memory is switched off: nothing is written; a filing that ends as it
  // is switched off is not shown either
  store.memoryOn = false;
  store.updated({ written: true });
  orb.frame(store, FRAME);
  store.memoryOn = true;
  store.amnesia = true;
  store.updated({ written: true });
  orb.frame(store, FRAME);
  run(orb, store, 2);
  assert.equal(orb.played, 0, `the keep played ${orb.played} times`);
  assert.equal(store.kept, 0, `${store.kept} writes counted`);
});

test("a write during a keep plays after it, and the light follows on without a jump", () => {
  const store = conversation();
  const orb = new Keep(store.kept);
  store.updated({ written: true });
  orb.frame(store, FRAME);
  run(orb, store, 0.3);
  // Filed again while the first keep is drawn in
  store.updated({ written: true });
  let last = keepLight(orb.progress);
  let jump = 0;
  for (let t = 0; t < 2 * KEEP_DURATION + 1; t += FRAME) {
    orb.frame(store, FRAME);
    const light = keepLight(orb.progress);
    for (const key of ["glow", "heart", "give"]) jump = Math.max(jump, Math.abs(light[key] - last[key]));
    last = light;
  }
  assert.equal(orb.played, 2, `the keep played ${orb.played} times for two writes`);
  assert.equal(orb.progress, 1, "the second keep did not end");
  // At 60 Hz no light changes by more than a fraction a frame
  assert.ok(jump < 0.1, `the light jumped by ${jump} in a frame`);
});

test("a keep starts and ends at rest", () => {
  for (const progress of [0, 1]) {
    const light = keepLight(progress);
    assert.ok(light.glow < 1e-9, `the light glows at ${progress}`);
    assert.ok(light.heart < 1e-9, `the heart glows at ${progress}`);
    assert.ok(light.give < 1e-9, `the ring gives at ${progress}`);
  }
  assert.equal(keepLight(0).reach, 1, "the light does not leave from the ring");
});

test("an orb at rest goes on from the count it finds, and one that starts over from the new count", () => {
  const store = conversation();
  // Presence is hidden while the conversation window shows two writes
  const presence = new Keep(null);
  store.updated({ written: true });
  store.updated({ written: true });
  // Shown without the other window's motion, it does not play them
  presence.frame(store, FRAME);
  run(presence, store, 2);
  assert.equal(presence.played, 0, "the orb played writes the other window showed");
  // Handed the motion, it plays a write the other window had not begun
  const handed = new Keep(null);
  handed.shown = store.kept - 1;
  handed.frame(store, FRAME);
  assert.equal(handed.played, 1, "the orb missed the write it was handed");
  // The conversation window reloads: its count starts over
  const reloaded = conversation();
  presence.frame(reloaded, FRAME);
  reloaded.updated({ written: true });
  presence.frame(reloaded, FRAME);
  assert.equal(presence.played, 1, "the orb missed the first write after the count started over");
});
