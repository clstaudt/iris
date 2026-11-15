// While Iris sleeps, the status line now and then names the wake word: for a
// few seconds as it falls asleep, then again every half minute or so.

export interface WakeHintTiming {
  /** How long the hint shows as Iris falls asleep, in ms */
  first: number;
  /** From one showing to the next, in ms */
  every: number;
  /** How long it shows each time after that, in ms */
  length: number;
}

export const WAKE_HINT: WakeHintTiming = { first: 5000, every: 30_000, length: 5000 };

/** Show the hint at once, then again and again, until the returned stop is
 * called; `show` is told each time it comes or goes. */
export function wakeHint(show: (on: boolean) => void, timing: WakeHintTiming = WAKE_HINT): () => void {
  let timer: ReturnType<typeof setTimeout>;
  const showFor = (length: number) => {
    show(true);
    timer = setTimeout(() => {
      show(false);
      timer = setTimeout(() => showFor(timing.length), Math.max(0, timing.every - length));
    }, length);
  };
  showFor(timing.first);
  return () => {
    clearTimeout(timer);
    show(false);
  };
}
