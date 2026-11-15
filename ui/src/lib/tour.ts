// The feature tour: ten steps in four chapters, each a sentence at the real
// controls it is about, with everything else dimmed. Taken at the end of the
// first walkthrough, or from Settings.
//
// The lit controls work as ever, and using one goes on with the tour; Next
// goes on for those who only read, doing what the control would have done.
// What the tour opens or closes so that a step can show its controls
// (Settings, the threads), it puts back at the end, unless the user has
// changed it since; what the user does with the controls stays. The tour
// itself never sends a message, plays a sound or downloads anything.

import type { SettingsPage } from "./state.svelte";

/** What the tour opens and closes. */
export interface TourView {
  settingsPage: SettingsPage | null;
  sidebar: boolean;
  panel: "memory" | null;
}

/** What a place needs shown, or what using its control opens: a settings
 *  page, "open" for any, or null for the conversation; the threads. */
export interface TourWant {
  settings?: SettingsPage | "open" | null;
  sidebar?: boolean;
}

/** Where a step's sentence stands: the controls lit, by their data-tour
 *  names, and which of them go on when used. */
export interface TourPlace {
  targets: string[];
  go?: string[];
  needs?: TourWant;
  /** What its control opens: Next opens it too, and the place is passed
   *  over while it shows */
  opens?: TourWant;
  /** Its control takes the user away (presence): the tour goes on once
   *  they are back */
  away?: boolean;
  /** What the sentence speaks of besides, by CSS selector, which the
   *  callout leaves in sight where there is room */
  avoid?: string[];
}

export interface TourStep {
  id: string;
  chapter: string;
  text: string;
  places: TourPlace[];
}

const CONVERSATION: TourWant = { settings: null };

/** A step in Settings: the tab's entry, which opens it, then the tab. */
function tab(page: SettingsPage, content: string, go: string[] = [], avoid?: string[]): TourPlace[] {
  const entry = `settings-${page}`;
  return [
    { targets: [entry], go: [entry], needs: { settings: "open" }, opens: { settings: page } },
    { targets: [entry, content], go, needs: { settings: page }, avoid },
  ];
}

/** Settings' own button, for a step whose tab is opened from there. */
const SETTINGS: TourPlace = { targets: ["settings"], go: ["settings"], opens: { settings: "open" } };

export const TOUR: TourStep[] = [
  {
    id: "conversation",
    chapter: "Talking",
    text:
      "This is your conversation with Iris. Speak or type, and Iris answers aloud; the orb shows " +
      "what it is doing: listening, thinking, looking or speaking.",
    places: [{ targets: ["composer", "orb"], go: ["orb", "send"], needs: CONVERSATION, avoid: [".bubble"] }],
  },
  {
    id: "pause",
    chapter: "Talking",
    text:
      "Try Pause: Iris stops listening and speaking until you resume. Mic turns the microphone " +
      "off and on.",
    places: [{ targets: ["pause", "mic"], go: ["pause", "mic"], needs: CONVERSATION }],
  },
  {
    id: "compact",
    chapter: "Talking",
    text:
      "Try compact mode: Iris shrinks to its orb, floating above your other windows. The small " +
      "arrows on the orb bring you back, and the tour goes on.",
    places: [{ targets: ["presence"], go: ["presence"], away: true }],
  },
  {
    id: "persona",
    chapter: "Your assistant",
    text:
      "A persona sets your assistant's personality, voice and wake word, the name you say to " +
      "wake it. Iris is the default persona; the Personas tab is where you create others.",
    places: [SETTINGS, ...tab("personas", "personas", ["new-persona"])],
  },
  {
    id: "voices",
    chapter: "Your assistant",
    text:
      "The Voices tab lists every voice and the languages it speaks. Each persona speaks with " +
      "one of these voices, and you can clone your own from a recording.",
    places: tab("voices", "voices"),
  },
  {
    id: "role",
    chapter: "Your assistant",
    text:
      "A role is an optional focus for a conversation, such as Research or Language partner. " +
      "Any persona can take on any role; the Roles tab is where you edit them.",
    places: tab("roles", "roles"),
  },
  {
    id: "thread",
    chapter: "Conversations",
    text:
      "A thread is one conversation with its own memory. You start it with a persona and, if you " +
      "like, a role; both stay with that thread. Switch between threads here.",
    places: [
      { targets: ["threads"], go: ["threads"], needs: CONVERSATION, opens: { sidebar: true } },
      { targets: ["thread-list"], go: ["new-thread"], needs: { settings: null, sidebar: true } },
    ],
  },
  {
    id: "memory",
    chapter: "Conversations",
    text:
      "Iris has a memory: it remembers the topics of each thread and brings them back later in " +
      "that thread. The Memory panel shows what it remembers.",
    places: [{ targets: ["memory"], go: ["memory"] }],
  },
  {
    id: "pipeline",
    chapter: "How Iris works",
    text:
      "The Pipeline tab shows how Iris works, from your voice to its answer: it hears you, thinks " +
      "with a language model, and speaks. The times show how long each stage took in your last turn.",
    // The other tabs' entries stay in sight
    places: [SETTINGS, ...tab("pipeline", "pipeline", [], ['nav[aria-label="Settings"] button'])],
  },
  {
    id: "stages",
    chapter: "How Iris works",
    text:
      "Click a stage to change it, such as the microphone or the language model. A value tagged " +
      "with a persona or a thread is set there instead.",
    places: [
      {
        targets: ["stage-llm"],
        go: ["stage-llm"],
        needs: { settings: "pipeline" },
        // The stages' names, and the values set by a persona or a thread,
        // tagged so
        avoid: [
          "[data-stage] .label",
          '[data-stage="persona"]',
          '[data-stage="tools"]',
          '[data-stage="memory"]',
          "[data-stage] .tag",
        ],
      },
    ],
  },
];

export interface TourPosition {
  step: number;
  place: number;
}

const placeAt = ({ step, place }: TourPosition) => TOUR[step].places[place];

/** The settings page wanted, if one is named. */
const pageOf = (want?: TourWant): SettingsPage | undefined =>
  want?.settings && want.settings !== "open" ? want.settings : undefined;

/** The place after this one; null after the last. */
export function after({ step, place }: TourPosition): TourPosition | null {
  if (place + 1 < TOUR[step].places.length) return { step, place: place + 1 };
  if (step + 1 < TOUR.length) return { step: step + 1, place: 0 };
  return null;
}

/** Whether the app shows what is wanted. */
export function holds(want: TourWant | undefined, view: TourView): boolean {
  if (!want) return true;
  const { settings, sidebar } = want;
  if (settings === "open" && view.settingsPage === null) return false;
  if (settings !== undefined && settings !== "open" && view.settingsPage !== settings) return false;
  return sidebar === undefined || view.sidebar === sidebar;
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

type Aspect = keyof TourView;

/** A change the tour made: the value before it, and the one it left. */
interface Change {
  from: TourView[Aspect];
  to: TourView[Aspect];
}

export class TourGuide {
  private app: TourApp;
  /** Where the tour is, null when it is not shown, and whether the user is
   *  away in presence; a reactive cell in the app */
  private now: { position: TourPosition | null; away: boolean };
  /** What the tour changed, to put back at the end */
  private changes = new Map<Aspect, Change>();

  constructor(app: TourApp, now: { position: TourPosition | null; away: boolean }) {
    this.app = app;
    this.now = now;
  }

  get active(): boolean {
    return this.now.position !== null;
  }

  get away(): boolean {
    return this.now.away;
  }

  get step(): TourStep | null {
    return this.now.position ? TOUR[this.now.position.step] : null;
  }

  get place(): TourPlace | null {
    return this.now.position ? placeAt(this.now.position) : null;
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
    this.changes.clear();
    this.now.away = false;
    this.arrive({ step: 0, place: 0 });
  }

  /** On, doing what the lit control would have done. */
  next() {
    if (!this.now.position || this.now.away) return;
    this.change(this.place!.opens, this.ahead());
    this.arrive(after(this.now.position));
  }

  /** A lit control was clicked, and did what it does (`before` is the app as
   *  it was): what that changed is the user's, and a control that goes on
   *  goes on, or waits for the user to come back. */
  clicked(name: string, before: TourView) {
    const place = this.place;
    if (!place || this.now.away) return;
    const view = this.app.view();
    for (const aspect of Object.keys(view) as Aspect[]) {
      if (view[aspect] !== before[aspect]) this.changes.delete(aspect);
    }
    if (!place.go?.includes(name)) return;
    if (place.away) {
      this.now.away = true;
      return;
    }
    const next = after(this.now.position!);
    // The last control's use ends the tour where it leaves the app
    if (!next) this.keep(place.needs);
    this.arrive(next);
  }

  /** Back from presence, however the user came: the tour goes on. */
  cameBack() {
    if (!this.now.away) return;
    this.now.away = false;
    this.arrive(after(this.now.position!));
  }

  /** The end before the last step: the same as the end after it. */
  skip() {
    this.end();
  }

  /** The page the steps ahead open, for Settings opened by Next. */
  private ahead(): SettingsPage | undefined {
    const step = this.now.position!.step;
    for (let at = after(this.now.position!); at && at.step === step; at = after(at)) {
      const page = pageOf(placeAt(at).opens) ?? pageOf(placeAt(at).needs);
      if (page) return page;
    }
    return undefined;
  }

  /** To a place, showing what it needs, and past those whose control's
   *  effect shows already; the end after the last. */
  private arrive(position: TourPosition | null) {
    while (position) {
      const place = placeAt(position);
      this.change(place.needs, pageOf(place.opens));
      if (!place.opens || !holds(place.opens, this.app.view())) break;
      position = after(position);
    }
    if (!position) return this.end();
    this.now.position = position;
  }

  /** Have the app show what is wanted, keeping what the tour changed. */
  private change(want: TourWant | undefined, page?: SettingsPage) {
    if (!want || holds(want, this.app.view())) return;
    const view = this.app.view();
    const next = { ...view };
    if (want.settings === "open") next.settingsPage = page ?? "pipeline";
    else if (want.settings !== undefined) next.settingsPage = want.settings;
    if (next.settingsPage && !view.settingsPage) next.panel = null;
    if (want.sidebar !== undefined) next.sidebar = want.sidebar;
    for (const aspect of Object.keys(next) as Aspect[]) {
      if (next[aspect] === view[aspect]) continue;
      const change = this.changes.get(aspect);
      this.changes.set(aspect, { from: change ? change.from : view[aspect], to: next[aspect] });
    }
    this.app.show(next);
  }

  /** What the tour opened for these needs becomes the user's. */
  private keep(needs: TourWant | undefined) {
    if (needs?.settings) this.changes.delete("settingsPage");
    if (needs?.sidebar) this.changes.delete("sidebar");
  }

  private end() {
    if (!this.active) return;
    const view = this.app.view();
    const back = { ...view };
    for (const aspect of ["settingsPage", "sidebar", "panel"] as Aspect[]) {
      const change = this.changes.get(aspect);
      // Put back unless the user changed it since; the memory panel only
      // beside the conversation
      if (!change || view[aspect] !== change.to) continue;
      if (aspect === "panel" && back.settingsPage) continue;
      Object.assign(back, { [aspect]: change.from });
    }
    this.changes.clear();
    this.now.position = null;
    this.now.away = false;
    this.app.show(back);
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

/** Where the sentence stands in the window (space), never over a lit
 *  control (holes), and clear of what else it speaks of (avoid) where there
 *  is room: beside the controls, below, above, right or left of them all or
 *  of one, else wherever there is room, nearest them; narrower when there is
 *  no room at its width. `sizes` are the callout's, widest first. */
export function calloutAt(
  holes: Box[],
  sizes: { width: number; height: number }[],
  space: { width: number; height: number },
  avoid: Box[] = []
): Box {
  if (avoid.length) {
    const clear = place(holes, [...holes, ...avoid], sizes, space);
    if (clear) return clear;
  }
  const at = place(holes, holes, sizes, space);
  if (at) return at;
  const narrowest = sizes[sizes.length - 1];
  return {
    left: space.width - MARGIN - narrowest.width,
    top: space.height - MARGIN - narrowest.height,
    ...narrowest,
  };
}

const MARGIN = 12;
const GAP = 14;

/** The first place for the callout beside the holes, clear of the
 *  obstacles, at the widest size that has one; null if none. */
function place(
  holes: Box[],
  obstacles: Box[],
  sizes: { width: number; height: number }[],
  space: { width: number; height: number }
): Box | null {
  const widest = sizes[0];
  if (!holes.length) {
    return { left: (space.width - widest.width) / 2, top: (space.height - widest.height) / 2, ...widest };
  }
  const union = {
    left: Math.min(...holes.map((h) => h.left)),
    top: Math.min(...holes.map((h) => h.top)),
    right: Math.max(...holes.map((h) => h.left + h.width)),
    bottom: Math.max(...holes.map((h) => h.top + h.height)),
  };
  const fits = (box: Box) =>
    box.left >= MARGIN &&
    box.top >= MARGIN &&
    box.left + box.width <= space.width - MARGIN &&
    box.top + box.height <= space.height - MARGIN &&
    !obstacles.some((obstacle) => overlaps(box, obstacle));
  for (const size of sizes) {
    const clampX = (x: number) => Math.max(MARGIN, Math.min(x, space.width - MARGIN - size.width));
    const clampY = (y: number) => Math.max(MARGIN, Math.min(y, space.height - MARGIN - size.height));
    // Below or above, centred; right or left, level with the top
    const around = (r: { left: number; top: number; right: number; bottom: number }) => {
      const x = clampX((r.left + r.right - size.width) / 2);
      return [
        { left: x, top: r.bottom + GAP },
        { left: x, top: r.top - GAP - size.height },
        { left: r.right + GAP, top: clampY(r.top) },
        { left: r.left - GAP - size.width, top: clampY(r.top) },
      ];
    };
    const sides = [union, ...holes.map((h) => ({ ...h, right: h.left + h.width, bottom: h.top + h.height }))];
    for (const at of sides.flatMap(around)) {
      if (fits({ ...at, ...size })) return { ...at, ...size };
    }
    // Wherever there is room, nearest the controls
    let best: Box | null = null;
    let nearest = Infinity;
    for (let top = MARGIN; top + size.height <= space.height - MARGIN; top += 8) {
      for (let left = MARGIN; left + size.width <= space.width - MARGIN; left += 8) {
        const box = { left, top, ...size };
        if (!fits(box)) continue;
        const dx = Math.max(union.left - (left + size.width), 0, left - union.right);
        const dy = Math.max(union.top - (top + size.height), 0, top - union.bottom);
        if (dx + dy < nearest) [best, nearest] = [box, dx + dy];
      }
    }
    if (best) return best;
  }
  return null;
}
