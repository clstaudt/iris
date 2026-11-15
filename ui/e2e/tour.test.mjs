// The feature tour's steps, what its lit controls do to it, what it opens and
// puts back, and the setting that keeps whether it was taken, with a
// stand-in for the app.
//
//   npm run test:tour

import assert from "node:assert/strict";
import { test } from "node:test";

import { TOUR, TourGuide, after, calloutAt } from "../src/lib/tour.ts";

const CONVERSATION = { settingsPage: null, sidebar: false, panel: null };

/** The app as the tour sees it: what it shows, and its saved settings. */
function app(shown = CONVERSATION, saved = { setup_done: true }) {
  const fake = {
    shown: { ...shown },
    settings: { ...saved },
    view: () => ({ ...fake.shown }),
    show(view) {
      fake.shown = { ...view };
    },
    saved: () => ({ ...fake.settings }),
    save(patch) {
      fake.settings = { ...fake.settings, ...patch };
    },
  };
  const guide = new TourGuide(fake, { position: null, away: false });
  /** A click on a lit control: what the real one does, then the tour told */
  const click = (name) => {
    const before = fake.view();
    const view = fake.shown;
    if (name === "settings") view.settingsPage = view.settingsPage ? null : "pipeline";
    if (name.startsWith("settings-")) view.settingsPage = name.slice("settings-".length);
    if (name === "settings" || name.startsWith("settings-")) view.panel = null;
    if (name === "threads") view.sidebar = !view.sidebar;
    if (name === "memory") Object.assign(view, { panel: view.panel ? null : "memory", settingsPage: null });
    if (name === "stage-persona") view.settingsPage = null;
    guide.clicked(name, before);
  };
  /** Where the tour is: its step and the controls lit */
  const where = () => (guide.active ? `${guide.step.id}:${guide.place.targets.join("+")}` : "ended");
  return { fake, guide, click, where };
}

test("ten steps in four chapters, in order", () => {
  assert.deepEqual(
    TOUR.map((s) => `${s.chapter}: ${s.id}`),
    [
      "Talking: conversation",
      "Talking: pause",
      "Talking: compact",
      "Your assistant: persona",
      "Your assistant: voices",
      "Your assistant: role",
      "Conversations: thread",
      "Conversations: memory",
      "How Iris works: pipeline",
      "How Iris works: stages",
    ]
  );
});

test("each place lights its controls, and names those that go on", () => {
  const all = [];
  for (let at = { step: 0, place: 0 }; at; at = after(at)) {
    const place = TOUR[at.step].places[at.place];
    all.push(`${TOUR[at.step].id}: ${place.targets.join("+")} > ${(place.go ?? []).join("+") || "Next"}`);
  }
  assert.deepEqual(all, [
    "conversation: composer+orb > orb+send",
    "pause: pause+mic > pause+mic",
    "compact: presence > presence",
    "persona: settings > settings",
    "persona: settings-personas > settings-personas",
    "persona: settings-personas+personas > new-persona",
    "voices: settings-voices > settings-voices",
    "voices: settings-voices+voices > Next",
    "role: settings-roles > settings-roles",
    "role: settings-roles+roles > Next",
    "thread: threads > threads",
    "thread: new-thread > new-thread",
    "memory: memory > memory",
    "pipeline: settings > settings",
    "pipeline: settings-pipeline > settings-pipeline",
    "pipeline: settings-pipeline+pipeline > Next",
    "stages: stage-llm > stage-llm",
  ]);
});

test("read with Next, it opens what each step shows, counts the steps, and leaves the app as it was", () => {
  const { fake, guide, where } = app();
  guide.start();
  const seen = [];
  while (guide.active) {
    const { settingsPage, sidebar } = fake.shown;
    seen.push(`${guide.number} ${where()} [${settingsPage ?? "conversation"}${sidebar ? ", threads" : ""}]${guide.last ? " Done" : ""}`);
    guide.next();
  }
  assert.deepEqual(seen, [
    "1 conversation:composer+orb [conversation]",
    "2 pause:pause+mic [conversation]",
    "3 compact:presence [conversation]",
    "4 persona:settings [conversation]",
    // Next does what Settings' button would, on the tab the step is about
    "4 persona:settings-personas+personas [personas]",
    "5 voices:settings-voices [personas]",
    "5 voices:settings-voices+voices [voices]",
    "6 role:settings-roles [voices]",
    "6 role:settings-roles+roles [roles]",
    "7 thread:threads [conversation]",
    "7 thread:new-thread [conversation, threads]",
    "8 memory:memory [conversation, threads]",
    "9 pipeline:settings [conversation, threads]",
    "9 pipeline:settings-pipeline+pipeline [pipeline, threads]",
    "10 stages:stage-llm [pipeline, threads] Done",
  ]);
  assert.deepEqual(fake.shown, CONVERSATION, "the app is not as it was");
  assert.equal(fake.settings.tour, "done");
});

test("used, each lit control goes on, compact mode once the user is back, and what the user opened stays", () => {
  const { fake, guide, click, where } = app();
  guide.start();
  const path = [];
  const use = (name) => {
    click(name);
    path.push(where());
  };
  use("orb");
  use("pause");
  // Compact mode: the tour waits for the user to come back
  click("presence");
  assert.equal(guide.away, true);
  guide.next();
  assert.equal(where(), "compact:presence", "Next went on while the user was away");
  guide.cameBack();
  path.push(where());
  use("settings");
  use("settings-personas");
  use("new-persona");
  use("settings-voices");
  guide.next();
  use("settings-roles");
  guide.next();
  path.push(where());
  use("threads");
  use("new-thread");
  use("memory");
  use("settings");
  guide.next();
  use("stage-llm");
  assert.deepEqual(path, [
    "pause:pause+mic",
    "compact:presence",
    "persona:settings",
    "persona:settings-personas",
    "persona:settings-personas+personas",
    "voices:settings-voices",
    "voices:settings-voices+voices",
    "role:settings-roles+roles",
    // The thread needs the conversation: the tour closes the settings
    "thread:threads",
    "thread:new-thread",
    "memory:memory",
    "pipeline:settings",
    "pipeline:settings-pipeline+pipeline",
    "ended",
  ]);
  // Opened by the user: the threads, and the pipeline with its stage
  assert.deepEqual(fake.shown, { settingsPage: "pipeline", sidebar: true, panel: null });
  assert.equal(fake.settings.tour, "done");
});

test("run from Settings, a skip midway puts back what the tour changed, and keeps what the user did", () => {
  const found = { settingsPage: "pipeline", sidebar: false, panel: null };
  const { fake, guide, click, where } = app(found, { setup_done: true, tour: "done" });
  guide.start();
  assert.deepEqual(fake.shown, CONVERSATION, "the conversation does not show");
  guide.next();
  click("mic");
  guide.next();
  guide.next();
  assert.equal(where(), "persona:settings-personas+personas");
  click("new-persona");
  while (guide.step.id !== "thread") guide.next();
  assert.equal(fake.shown.settingsPage, null);
  click("threads");
  assert.equal(where(), "thread:new-thread");
  guide.skip();
  assert.equal(guide.active, false);
  assert.deepEqual(fake.shown, { settingsPage: "pipeline", sidebar: true, panel: null });
});

test("a skip closes the threads and settings the tour opened, and brings back the memory panel they hid", () => {
  const found = { settingsPage: null, sidebar: false, panel: "memory" };
  const { fake, guide, where } = app(found);
  guide.start();
  while (where() !== "thread:new-thread") guide.next();
  assert.equal(fake.shown.sidebar, true);
  guide.skip();
  assert.deepEqual(fake.shown, found);

  const second = app(found);
  second.guide.start();
  while (second.guide.step.id !== "voices") second.guide.next();
  assert.deepEqual(second.fake.shown, { settingsPage: "personas", sidebar: false, panel: null });
  second.guide.skip();
  assert.deepEqual(second.fake.shown, found);
});

test("what the user changes with a lit control that does not go on is theirs, and the tour stays", () => {
  const { fake, guide, click, where } = app();
  guide.start();
  while (where() !== "pipeline:settings-pipeline+pipeline") guide.next();
  // The persona's stage leads to the conversation
  click("stage-persona");
  assert.equal(where(), "pipeline:settings-pipeline+pipeline");
  assert.equal(fake.shown.settingsPage, null);
  guide.skip();
  assert.equal(fake.shown.settingsPage, null, "the tour opened the settings again");
  assert.equal(fake.shown.sidebar, false, "the threads the tour opened stay open");
});

test("taken at the end of the walkthrough, it starts by itself until finished or skipped, then never again", () => {
  assert.equal(app(CONVERSATION, { setup_done: true }).guide.startsByItself(), false);
  for (const end of ["next", "skip"]) {
    const { fake, guide } = app(CONVERSATION, {});
    guide.take();
    assert.equal(fake.settings.tour, "taking");
    assert.equal(guide.startsByItself(), false, "before the walkthrough is done");
    fake.save({ setup_done: true });
    assert.equal(guide.startsByItself(), true);
    guide.start();
    guide.next();
    assert.equal(guide.startsByItself(), true, "a launch midway does not start it again");
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
  assert.equal(fake.settings.tour, undefined);
  guide.start();
  guide.next();
  guide.start();
  assert.equal(guide.step.id, "pause");
});

const overlaps = (a, b) =>
  a.left < b.left + b.width && b.left < a.left + a.width && a.top < b.top + b.height && b.top < a.top + a.height;

test("the sentence stands beside its controls, in the window, never over them, and narrower where it must", () => {
  const space = { width: 1100, height: 750 };
  const sizes = [
    { width: 300, height: 130 },
    { width: 240, height: 160 },
    { width: 180, height: 230 },
  ];
  const inside = (box) =>
    box.left >= 12 && box.top >= 12 && box.left + box.width <= 1088 && box.top + box.height <= 738;
  // A control at the top right: below it
  const corner = { left: 1050, top: 6, width: 40, height: 40 };
  assert.deepEqual(calloutAt([corner], sizes, space), { left: 1100 - 300 - 12, top: 60, width: 300, height: 130 });
  // The message box and the orb, at the bottom: above them
  const box = { left: 300, top: 560, width: 600, height: 100 };
  const orb = { left: 540, top: 660, width: 120, height: 80 };
  assert.equal(calloutAt([box, orb], sizes, space).top, 560 - 14 - 130);
  // A tab and a page lit almost to the window's edges: the narrow column
  // beside them
  const entry = { left: 11, top: 108, width: 182, height: 40 };
  const page = { left: 219, top: 98, width: 870, height: 646 };
  const narrow = calloutAt([entry, page], sizes, space);
  assert.equal(narrow.width, 180);
  assert.ok(inside(narrow), JSON.stringify(narrow));
  for (const hole of [entry, page]) assert.ok(!overlaps(narrow, hole), JSON.stringify(narrow));
  // Wherever the controls are, never over them
  for (let i = 0; i < 200; i++) {
    const holes = Array.from({ length: 1 + (i % 3) }, (_, j) => ({
      left: (i * 97 + j * 311) % 900,
      top: (i * 53 + j * 173) % 600,
      width: 40 + ((i * 7 + j * 13) % 200),
      height: 30 + ((i * 11 + j * 5) % 120),
    }));
    const at = calloutAt(holes, sizes, space);
    assert.ok(inside(at), JSON.stringify(at));
    for (const hole of holes) assert.ok(!overlaps(at, hole), `${JSON.stringify(at)} over ${JSON.stringify(hole)}`);
  }
});
