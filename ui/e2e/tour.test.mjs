// The feature tour's steps, what it opens and leaves as it found it, and the
// setting that keeps whether it was taken, with a stand-in for the app.
//
//   npm run test:tour

import assert from "node:assert/strict";
import { test } from "node:test";

import { TOUR, TourGuide, after, calloutAt } from "../src/lib/tour.ts";

const CONVERSATION = { settingsPage: null, sidebar: false, panel: null };

/** The app as the tour sees it: what it shows, each view it was asked to
 *  show, and its saved settings. */
function app(shown = CONVERSATION, saved = { setup_done: true }) {
  const fake = {
    shown: { ...shown },
    views: [],
    settings: { ...saved },
    view: () => ({ ...fake.shown }),
    show(view) {
      fake.shown = { ...view };
      fake.views.push({ ...view });
    },
    saved: () => ({ ...fake.settings }),
    save(patch) {
      fake.settings = { ...fake.settings, ...patch };
    },
  };
  const guide = new TourGuide(fake, { position: null });
  return { fake, guide };
}

/** Every place from the first, in order. */
function places() {
  const all = [];
  for (let at = { step: 0, place: 0 }; at; at = after(at)) all.push({ ...at, ...TOUR[at.step].places[at.place] });
  return all;
}

test("six steps at most, each one sentence", () => {
  assert.ok(TOUR.length <= 6, `${TOUR.length} steps`);
  assert.deepEqual(
    TOUR.map((s) => s.id),
    ["talking", "pause", "threads", "personas", "voices", "memory"]
  );
  for (const step of TOUR) {
    assert.match(step.text, /^[A-Z][^.!?]*\.$/, `${step.id}: "${step.text}" is not one sentence`);
  }
});

test("each step stands at its controls, and a click goes on only where it opens what comes next", () => {
  assert.deepEqual(
    places().map((p) => [TOUR[p.step].id, p.targets.join("+"), !!p.click]),
    [
      ["talking", "composer+orb", false],
      ["pause", "pause+mic", false],
      ["threads", "threads", true],
      ["threads", "new-thread", false],
      ["personas", "settings", true],
      ["personas", "settings-personas+personas", false],
      ["voices", "settings-voices+voices", false],
      ["memory", "memory", false],
    ]
  );
});

test("taken through, it opens the threads and the settings pages, counts the steps, and leaves the app as it was", () => {
  const { fake, guide } = app();
  assert.equal(guide.active, false);
  guide.start();
  const seen = [];
  while (guide.active) {
    seen.push({ number: guide.number, last: guide.last, view: fake.shown });
    guide.next();
  }
  assert.deepEqual(
    seen.map((s) => s.number),
    [1, 2, 3, 3, 4, 4, 5, 6]
  );
  assert.deepEqual(
    seen.map((s) => s.last),
    [false, false, false, false, false, false, false, true]
  );
  assert.deepEqual(
    seen.map((s) => [s.view.settingsPage, s.view.sidebar]),
    [
      [null, false],
      [null, false],
      [null, false],
      [null, true],
      [null, false],
      ["personas", false],
      ["voices", false],
      [null, false],
    ]
  );
  assert.deepEqual(fake.shown, CONVERSATION, "the app is not as it was");
  assert.equal(fake.settings.tour, "done");
});

test("run from Settings, it shows the conversation, and a skip midway goes back to the page it was run from", () => {
  const found = { settingsPage: "pipeline", sidebar: false, panel: null };
  const { fake, guide } = app(found, { setup_done: true, tour: "done" });
  guide.start();
  assert.deepEqual(fake.shown, CONVERSATION, "the conversation does not show");
  for (let i = 0; i < 3; i++) guide.next();
  assert.equal(fake.shown.sidebar, true, "the threads are not shown");
  guide.skip();
  assert.equal(guide.active, false);
  assert.deepEqual(fake.shown, found, "the app is not as it was");
  assert.equal(fake.settings.tour, "done");
});

test("a skip on a settings page closes it, and opens the memory panel and threads left open", () => {
  const found = { settingsPage: null, sidebar: true, panel: "memory" };
  const { fake, guide } = app(found);
  guide.start();
  assert.deepEqual(fake.shown, found, "the conversation is not shown as it was");
  while (guide.step?.id !== "voices") guide.next();
  assert.deepEqual(fake.shown, { settingsPage: "voices", sidebar: true, panel: null });
  guide.skip();
  assert.deepEqual(fake.shown, found);
});

test("taken at the end of the walkthrough, it starts by itself until finished or skipped, then never again", () => {
  // Not taken: never by itself
  assert.equal(app(CONVERSATION, { setup_done: true }).guide.startsByItself(), false);

  for (const end of ["next", "skip"]) {
    const { fake, guide } = app(CONVERSATION, {});
    guide.take();
    assert.equal(fake.settings.tour, "taking");
    // Not before the walkthrough is done
    assert.equal(guide.startsByItself(), false);
    fake.save({ setup_done: true });
    assert.equal(guide.startsByItself(), true);
    // A launch midway starts it again
    guide.start();
    guide.next();
    assert.equal(guide.startsByItself(), true);
    if (end === "next") while (guide.active) guide.next();
    else guide.skip();
    assert.equal(fake.settings.tour, "done");
    assert.equal(guide.startsByItself(), false, `it starts again after a ${end}`);
  }
});

test("started twice it goes on where it is, and Next or Skip with no tour change nothing", () => {
  const { fake, guide } = app();
  guide.next();
  guide.skip();
  assert.equal(fake.views.length, 0);
  assert.equal(fake.settings.tour, undefined);
  guide.start();
  guide.next();
  guide.start();
  assert.equal(guide.step?.id, "pause");
});

test("the sentence stands beside its controls, in the window and clear of them", () => {
  const space = { width: 1280, height: 800 };
  const size = { width: 290, height: 100 };
  // A control at the top right: below it, in the window
  const corner = { left: 1230, top: 6, width: 40, height: 40 };
  assert.deepEqual(calloutAt([corner], size, space), { left: 1280 - 290 - 12, top: 46 + 14 });
  // The message box and the orb, at the bottom: above them
  const box = { left: 400, top: 600, width: 720, height: 100 };
  const orb = { left: 700, top: 700, width: 120, height: 90 };
  const above = calloutAt([box, orb], size, space);
  assert.equal(above.top, 600 - 14 - 100);
  // Lit almost everywhere: in the largest one's lower right corner
  const page = { left: 200, top: 50, width: 1070, height: 740 };
  const nav = { left: 10, top: 150, width: 170, height: 36 };
  assert.deepEqual(calloutAt([nav, page], size, space), { left: 1270 - 290 - 14, top: 790 - 100 - 14 });
});
