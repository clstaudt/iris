// The orb's shutter: the conversation window counts each look through the
// camera or at the screen from the core's events, and an orb fires its
// shutter once per look it is told of, however briefly the state showed the
// look, or whether it showed it at all.
//
//   npm run test:shutter

import assert from "node:assert/strict";
import { test } from "node:test";

import { Exposures, SHUTTER_DURATION, Shutter } from "../src/lib/shutter.ts";

// One frame at 60 Hz, in seconds
const FRAME = 1 / 60;

/** The conversation window's store, as far as the orb goes: the core's
 *  state and status flags in, the looks counted. */
function conversation() {
  const exposures = new Exposures();
  const store = {
    state: "idle",
    shots: 0,
    /** The core's state changed */
    becomes: (state) => {
      store.state = state;
      if (exposures.state(state)) store.shots++;
    },
    /** The core's status flags changed: these are on */
    flags: (...on) => {
      if (exposures.flags(Object.fromEntries(on.map((name) => [name, true])))) store.shots++;
    },
  };
  return store;
}

test("a look the state leaves before the orb's next frame still fires the shutter", () => {
  const store = conversation();
  store.becomes("thinking");
  const orb = new Shutter(store.shots);
  orb.frame(store, FRAME);
  // The camera was quick: looking and thinking again, both between two frames
  store.becomes("looking");
  store.becomes("thinking");
  orb.frame(store, FRAME);
  assert.equal(orb.fired, 1, "the shutter did not fire");
  assert.ok(orb.progress > 0 && orb.progress < 1, `the shutter is at ${orb.progress}`);
});

test("a look while Iris speaks, which the state never shows, fires the shutter", () => {
  const store = conversation();
  store.becomes("speaking");
  const orb = new Shutter(store.shots);
  orb.frame(store, FRAME);
  // What it said before the tool is still heard while the camera takes the picture
  store.flags("thinking", "looking");
  orb.frame(store, FRAME);
  store.flags("thinking");
  orb.frame(store, FRAME);
  assert.equal(orb.fired, 1, "the shutter did not fire");
});

test("each look fires the shutter once, and it opens and closes once", () => {
  const store = conversation();
  const orb = new Shutter(store.shots);
  // The flags show the look first, the state a tenth of a second later
  store.flags("thinking", "looking");
  orb.frame(store, FRAME);
  store.becomes("looking");
  orb.frame(store, FRAME);
  // Taking longer than one open-and-close, the shutter stays shut after it
  for (let t = 0; t < SHUTTER_DURATION + 1; t += FRAME) orb.frame(store, FRAME);
  store.flags("thinking");
  store.becomes("thinking");
  orb.frame(store, FRAME);
  assert.equal(orb.fired, 1, `the shutter fired ${orb.fired} times for one look`);
  assert.equal(orb.progress, 1, "the shutter is still open");
  // The next look fires it again: the screen this time
  store.flags("thinking", "capturing_screen");
  orb.frame(store, FRAME);
  assert.equal(orb.fired, 2, "the second look did not fire the shutter");
});

test("an orb at rest goes on from the count it finds, and one that starts over from the new count", () => {
  const store = conversation();
  // Presence is hidden while the conversation window shows two looks
  const presence = new Shutter(null);
  store.flags("looking");
  store.flags();
  store.flags("capturing_screen");
  // Shown without the other window's motion, it does not fire for them
  presence.frame(store, FRAME);
  assert.equal(presence.fired, 0, "the orb fired for looks the other window showed");
  // Handed the motion, it fires for a look the other window had not drawn
  const handed = new Shutter(null);
  handed.shown = store.shots - 1;
  handed.frame(store, FRAME);
  assert.equal(handed.fired, 1, "the orb missed the look it was handed");
  // The conversation window reloads: its count starts over
  const reloaded = conversation();
  presence.frame(reloaded, FRAME);
  reloaded.flags("looking");
  presence.frame(reloaded, FRAME);
  assert.equal(presence.fired, 1, "the orb missed the first look after the count started over");
});
