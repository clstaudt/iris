/**
 * The orb's shutter: each time Iris looks through the camera or at the
 * screen, a camera's aperture opens and closes once inside the ring.
 *
 * A look is brief, and the state may never show it: the core gives the
 * state ten times a second, and while Iris speaks the state is "speaking"
 * whatever it does meanwhile. So the conversation window counts the looks
 * as the core reports them (Exposures), and every orb fires its shutter as
 * the count goes up (Shutter), the one in presence too, which is sent the
 * count with the rest of what the orb shows.
 */

/** States in which Iris looks through the camera or at the screen. */
export const APERTURE_STATES = new Set(["looking", "capturing_screen"]);

/** Seconds for one open-and-close. */
export const SHUTTER_DURATION = 1.2;

/** Whether the core's status flags show the camera or the screen in use. */
const lookingIn = (flags: Record<string, boolean>) =>
  [...APERTURE_STATES].some((name) => flags[name]);

/**
 * The looks Iris starts, from the core's events: its status flags, which
 * show a tool's activity the moment it starts, and its state, which shows
 * it later or not at all. Each look counts once.
 */
export class Exposures {
  /** A tool's activity shows the camera or the screen */
  private capturing = false;
  /** The state shows the camera or the screen */
  private looking = false;
  /** A look counted from the flags that the state has not shown yet */
  private unshown = false;

  /** The core's status flags changed. Returns whether a look started. */
  flags(flags: Record<string, boolean>): boolean {
    const on = lookingIn(flags);
    const started = on && !this.capturing;
    this.capturing = on;
    if (started) this.unshown = true;
    return started;
  }

  /** The core's state changed. Returns whether a look started that the
   *  flags did not show. */
  state(state: string): boolean {
    const looking = APERTURE_STATES.has(state);
    const entered = looking && !this.looking;
    this.looking = looking;
    if (!entered) return false;
    if (this.unshown) {
      this.unshown = false;
      return false;
    }
    return true;
  }
}

/**
 * The shutter as one orb draws it: a look counted since the orb's last
 * frame starts one open-and-close, however long ago the state left it.
 */
export class Shutter {
  /** Progress through one open-and-close; 1 means finished */
  progress = 1;
  /** The looks this orb has shown; null while it rests, for it misses
   *  what the other window shows: it goes on from the count it is handed,
   *  or else from the one it finds when it draws again */
  shown: number | null;
  /** Open-and-closes it began */
  fired = 0;

  constructor(shots: number | null) {
    this.shown = shots;
  }

  /** One frame, dt seconds after the last, with the looks counted so far
   *  as the orb's store has them. Returns whether the shutter fired. */
  frame({ shots }: { shots: number }, dt: number): boolean {
    // A count that starts over, as in a conversation window that reloads,
    // is gone on from
    const fires = this.shown !== null && shots > this.shown;
    this.shown = shots;
    if (fires) {
      this.progress = 0;
      this.fired++;
    }
    if (this.progress < 1) this.progress = Math.min(1, this.progress + dt / SHUTTER_DURATION);
    return fires;
  }
}
