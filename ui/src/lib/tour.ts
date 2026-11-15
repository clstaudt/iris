// The feature tour: a few steps, each one sentence at the real control it is
// about, with everything else dimmed. Taken at the end of the first
// walkthrough, or from Settings. It opens what a step needs (the threads,
// a settings page) and leaves the app as it found it, finished or skipped.
// It changes nothing in the conversation: a click on a control it lights
// goes on with the tour, and never reaches the control.

import type { SettingsPage } from "./state.svelte";

/** What the tour opens and closes. */
export interface TourView {
  settingsPage: SettingsPage | null;
  sidebar: boolean;
  panel: "memory" | null;
}

/** Where a step's sentence stands: the controls lit, by their data-tour
 *  names, and what the app shows beside the conversation. */
export interface TourPlace {
  targets: string[];
  /** A click on the lit control goes on, as it opens what comes next */
  click?: boolean;
  /** The threads shown */
  sidebar?: boolean;
  /** A settings page, instead of the conversation */
  settings?: SettingsPage;
}

export interface TourStep {
  id: string;
  text: string;
  /** One place, or a control and then what it opens */
  places: TourPlace[];
}

export const TOUR: TourStep[] = [
  {
    id: "talking",
    text: "Speak or type to talk; the orb shows what Iris is doing.",
    places: [{ targets: ["composer", "orb"] }],
  },
  {
    id: "pause",
    text: "Pause holds the conversation, and Mic turns the microphone off and on.",
    places: [{ targets: ["pause", "mic"] }],
  },
  {
    id: "threads",
    text: "Start a new thread to talk with another persona, or in a role.",
    places: [{ targets: ["threads"], click: true }, { targets: ["new-thread"], sidebar: true }],
  },
  {
    id: "personas",
    text: "A persona shapes your assistant's character and style: its personality, voice and wake word. Make your own here.",
    places: [
      { targets: ["settings"], click: true },
      { targets: ["settings-personas", "personas"], settings: "personas" },
    ],
  },
  {
    id: "voices",
    text: "Every voice and the languages it speaks; clone your own here.",
    places: [{ targets: ["settings-voices", "voices"], settings: "voices" }],
  },
  {
    id: "memory",
    text: "See what each thread remembers.",
    places: [{ targets: ["memory"] }],
  },
];

export interface TourPosition {
  step: number;
  place: number;
}

/** The place after this one; null after the last. */
export function after({ step, place }: TourPosition): TourPosition | null {
  if (place + 1 < TOUR[step].places.length) return { step, place: place + 1 };
  if (step + 1 < TOUR.length) return { step: step + 1, place: 0 };
  return null;
}

/** The app as a place shows it: as the tour found it, with the
 *  conversation shown unless the place is a settings page. */
export function viewAt(place: TourPlace, found: TourView): TourView {
  if (place.settings) return { settingsPage: place.settings, sidebar: found.sidebar, panel: null };
  return { settingsPage: null, sidebar: place.sidebar ?? found.sidebar, panel: found.panel };
}

/** Taken at the end of the first walkthrough, until finished or skipped;
 *  then done, and it starts only from Settings. */
export type TourProgress = "taking" | "done";

/** The app, as the tour sees it. */
export interface TourApp {
  view(): TourView;
  show(view: TourView): void;
  /** The settings kept across launches */
  saved(): { setup_done?: boolean; tour?: TourProgress };
  save(patch: { tour: TourProgress }): void;
}

export class TourGuide {
  private app: TourApp;
  /** Where the tour is, null when it is not shown; a reactive cell in the app */
  private now: { position: TourPosition | null };
  /** The app as the tour found it, to leave it so */
  private found: TourView | null = null;

  constructor(app: TourApp, now: { position: TourPosition | null }) {
    this.app = app;
    this.now = now;
  }

  get active(): boolean {
    return this.now.position !== null;
  }

  get step(): TourStep | null {
    return this.now.position ? TOUR[this.now.position.step] : null;
  }

  get place(): TourPlace | null {
    const position = this.now.position;
    return position ? TOUR[position.step].places[position.place] : null;
  }

  /** The step shown, counted from 1 */
  get number(): number {
    return (this.now.position?.step ?? -1) + 1;
  }

  get last(): boolean {
    return !!this.now.position && after(this.now.position) === null;
  }

  /** Whether the tour starts as Iris opens: it was taken at the end of the
   *  first walkthrough, and neither finished nor skipped since. */
  startsByItself(): boolean {
    const saved = this.app.saved();
    return !!saved.setup_done && saved.tour === "taking";
  }

  /** Taken at the end of the first walkthrough: it starts once the
   *  conversation shows. */
  take() {
    this.app.save({ tour: "taking" });
  }

  start() {
    if (this.active) return;
    this.found = this.app.view();
    this.go({ step: 0, place: 0 });
  }

  /** On to the next place, or the end after the last. */
  next() {
    if (!this.now.position) return;
    const next = after(this.now.position);
    if (next) this.go(next);
    else this.end();
  }

  /** The end before the last step: the same as the end after it. */
  skip() {
    this.end();
  }

  private go(position: TourPosition) {
    this.now.position = position;
    this.app.show(viewAt(TOUR[position.step].places[position.place], this.found!));
  }

  private end() {
    if (!this.active) return;
    this.app.show(this.found!);
    this.now.position = null;
    this.found = null;
    this.app.save({ tour: "done" });
  }
}

export interface Box {
  left: number;
  top: number;
  width: number;
  height: number;
}

const overlaps = (a: Box, b: Box) =>
  a.left < b.left + b.width && b.left < a.left + a.width && a.top < b.top + b.height && b.top < a.top + a.height;

/** Where the sentence stands in the window (space): beside the lit
 *  controls, below, above, right or left of them, wherever it fits clear of
 *  them; else in the largest one's lower right corner. */
export function calloutAt(
  holes: Box[],
  size: { width: number; height: number },
  space: { width: number; height: number },
  margin = 12,
  gap = 14
): { left: number; top: number } {
  if (!holes.length) {
    return { left: (space.width - size.width) / 2, top: (space.height - size.height) / 2 };
  }
  const left = Math.min(...holes.map((h) => h.left));
  const top = Math.min(...holes.map((h) => h.top));
  const right = Math.max(...holes.map((h) => h.left + h.width));
  const bottom = Math.max(...holes.map((h) => h.top + h.height));
  const clamp = (value: number, max: number) => Math.max(margin, Math.min(value, max - margin));
  const across = clamp((left + right - size.width) / 2, space.width - size.width);
  const down = clamp((top + bottom - size.height) / 2, space.height - size.height);
  const candidates = [
    { left: across, top: bottom + gap },
    { left: across, top: top - gap - size.height },
    { left: right + gap, top: down },
    { left: left - gap - size.width, top: down },
  ];
  for (const at of candidates) {
    const box = { ...at, ...size };
    const inside =
      box.left >= margin &&
      box.top >= margin &&
      box.left + box.width <= space.width - margin &&
      box.top + box.height <= space.height - margin;
    if (inside && !holes.some((hole) => overlaps(box, hole))) return at;
  }
  const largest = holes.reduce((a, b) => (b.width * b.height > a.width * a.height ? b : a));
  return {
    left: clamp(largest.left + largest.width - size.width - gap, space.width - size.width),
    top: clamp(largest.top + largest.height - size.height - gap, space.height - size.height),
  };
}
