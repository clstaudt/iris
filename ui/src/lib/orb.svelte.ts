/**
 * The orb between the conversation window and presence.
 *
 * Presence is a window of its own, transparent and round, while the store
 * lives in the conversation window. The conversation window sends presence
 * what the orb shows, and at each switch the orb's motion passes from the
 * window that hides to the one that shows: the orb carries on with the same
 * color, swell and turn of the ring, and only the window on screen draws.
 */

import type { OrbHandoff, OrbInputs, OrbMotion, PresenceCommand } from "./bridge";
import { store, type AssistantState } from "./state.svelte";

/** The orb in presence turns this color while an error needs the user
 *  (the theme's danger color) */
export const ERROR_TINT = "#df8a93";

/** The orb drawn in this window, once it is mounted. */
export interface OrbHandle {
  motion: () => OrbMotion;
  adopt: (motion: OrbMotion) => void;
  rest: () => void;
  wake: () => void;
}

let orb: OrbHandle | null = null;
// An orb in a window that is not on screen draws nothing until it is handed
// the motion
let resting = false;

/** Whether the orb starts at rest, before it is mounted. */
export function setOrbResting(on: boolean) {
  resting = on;
}

export function registerOrb(handle: OrbHandle): () => void {
  orb = handle;
  if (resting) handle.rest();
  else handle.wake();
  return () => {
    if (orb === handle) orb = null;
  };
}

const nextFrame = () => new Promise<number>((resolve) => requestAnimationFrame(resolve));

/** Take over the motion and draw. Resolves once a frame with it is drawn. */
async function adoptOrb(motion: OrbMotion | null) {
  resting = false;
  if (motion) orb?.adopt(motion);
  orb?.wake();
  // The first frame draws the motion; by the second it is on the screen
  await nextFrame();
  await nextFrame();
}

function restOrb() {
  resting = true;
  orb?.rest();
}

/** The orb's inputs as the store has them. */
function orbInputs(): OrbInputs {
  return {
    state: store.state,
    level: store.level,
    outputLevel: store.outputLevel,
    awakeUntil: store.awakeUntil === null ? null : performance.timeOrigin + store.awakeUntil,
    error: store.presenceError,
    shots: store.shots,
    kept: store.kept,
  };
}

/** What the presence window shows beside the store's own fields. */
export const presence = $state({ error: null as string | null });

function applyOrbInputs(inputs: OrbInputs) {
  store.state = inputs.state as AssistantState;
  store.level = inputs.level;
  store.outputLevel = inputs.outputLevel;
  store.awakeUntil = inputs.awakeUntil === null ? null : inputs.awakeUntil - performance.timeOrigin;
  store.shots = inputs.shots;
  store.kept = inputs.kept;
  presence.error = inputs.error;
}

/**
 * The conversation window's part: it hands the orb over at each switch,
 * sends presence what to show while presence is up, and does what presence
 * asks.
 */
export function connectConversation(): () => void {
  const offOrb = window.iris.onOrb({
    release: () => ({ motion: orb?.motion() ?? null, inputs: orbInputs() }),
    adopt: (handoff) => adoptOrb(handoff?.motion ?? null),
    rest: restOrb,
  });
  const offCommand = window.iris.onPresenceCommand((command: PresenceCommand) => {
    if (command === "leave") store.setPresence(false);
    else if (command === "pause") store.togglePause();
  });
  let sent = "";
  const stopMirror = $effect.root(() => {
    $effect(() => {
      if (!store.presence) {
        sent = "";
        return;
      }
      const inputs = orbInputs();
      // Streaming replies touch the messages many times a second; only a
      // change the orb would show is sent
      const key = JSON.stringify(inputs);
      if (key === sent) return;
      sent = key;
      window.iris.sendOrbInputs(inputs);
    });
  });
  return () => {
    offOrb();
    offCommand();
    stopMirror();
  };
}

/** The presence window's part: it shows what the conversation window sends,
 *  and takes the orb over at each switch. */
export function connectPresence(onRest: () => void): () => void {
  const offOrb = window.iris.onOrb({
    release: (): OrbHandoff => ({ motion: orb?.motion() ?? null }),
    adopt: (handoff) => {
      if (handoff?.inputs) applyOrbInputs(handoff.inputs);
      return adoptOrb(handoff?.motion ?? null);
    },
    rest: () => {
      restOrb();
      onRest();
    },
  });
  const offInputs = window.iris.onOrbInputs(applyOrbInputs);
  return () => {
    offOrb();
    offInputs();
  };
}
