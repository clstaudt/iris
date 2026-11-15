/**
 * The orb's keep: each time Iris writes to memory, a ring of light leaves
 * the orb's ring and is drawn in toward its heart, fading as it goes, and
 * the heart glows softly as it takes the light in: something kept. The
 * ring gives a little as the light leaves it. One keep lasts a second.
 *
 * Memory is written in the background, in a quiet spell after a turn: an
 * exchange filed under a topic, a topic made or its summary written. The
 * core says so (memory.updated, written). Forgetting updates memory too,
 * and is not written, nor is opening another thread; and nothing is
 * written while memory is off or the thread is in amnesia. So the
 * conversation window counts the writes as the core reports them, and
 * every orb plays its keep as the count goes up, the one in presence too,
 * which is sent the count with the rest of what the orb shows.
 */

/** Seconds for one keep. */
export const KEEP_DURATION = 1;

/** Whether the core's memory.updated tells of memory written, to show:
 *  not of a forget, and not while nothing is to be written. */
export function written(
  data: Record<string, unknown>,
  memory: { memoryOn: boolean; amnesia: boolean }
): boolean {
  return data.written === true && memory.memoryOn && !memory.amnesia;
}

/** The light of a keep, progress from 0 to 1: how far in the ring of
 *  light is (its radius as a share of the orb's ring), how bright it is,
 *  how bright the heart glows as it arrives, and how much the ring gives.
 *  Each starts and ends at rest, so a keep begins and ends without a jump,
 *  and one may follow another. */
export function keepLight(progress: number) {
  const p = Math.min(1, Math.max(0, progress));
  // Drawn in over the first three quarters, eased in and out
  const t = Math.min(1, p / 0.75);
  const drawn = t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2;
  // Eased up quickly and out slowly: brightest a third of the way in
  const glow = Math.sin(Math.PI * t ** 0.7) ** 2;
  // The heart takes the light in as it arrives, and lets it settle
  const h = Math.min(1, Math.max(0, (p - 0.35) / 0.65));
  return {
    reach: 1 - 0.85 * drawn,
    glow,
    heart: Math.sin(Math.PI * h) ** 2,
    // The ring draws in as the light leaves it, and comes back
    give: Math.sin(Math.PI * p) ** 2,
  };
}

/**
 * The keep as one orb draws it: each memory written that the orb was told
 * of starts one keep. One written while another plays follows it.
 */
export class Keep {
  /** Progress through one keep; 1 means finished */
  progress = 1;
  /** The writes this orb has begun to show; null while it rests, for it
   *  misses what the other window shows: it goes on from the count it is
   *  handed, or else from the one it finds when it draws again */
  shown: number | null;
  /** Keeps it began */
  played = 0;

  constructor(kept: number | null) {
    this.shown = kept;
  }

  /** One frame, dt seconds after the last, with the writes counted so far
   *  as the orb's store has them. Returns whether a keep began. */
  frame({ kept }: { kept: number }, dt: number): boolean {
    // A count that starts over, as in a conversation window that reloads,
    // is gone on from
    if (this.shown === null || kept < this.shown) this.shown = kept;
    if (this.progress < 1) this.progress = Math.min(1, this.progress + dt / KEEP_DURATION);
    if (this.progress < 1 || kept <= this.shown) return false;
    this.shown++;
    this.progress = 0;
    this.played++;
    return true;
  }
}
