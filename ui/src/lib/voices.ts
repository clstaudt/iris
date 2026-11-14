// Voices as the persona editor and the Voices page show them: filtered by
// the languages they speak, grouped by the language they were made for, and
// the jobs that make them ready to speak a language.
import type { VoiceInfo, VoiceJob } from "./state.svelte";

/** The languages a voice speaks properly, as Whisper codes, its own first. */
export const spokenBy = (voice: VoiceInfo) => voice.languages ?? [];

/** Voices filtered to those that speak every language wanted: of those
 *  languages, the ones some voice speaks (asked); the voices that speak all
 *  of them (shown); and the languages one of those also speaks, which can be
 *  added to the filter (addable). */
export function byLanguages(voices: VoiceInfo[], wanted: string[]) {
  const asked = wanted.filter((code) => voices.some((v) => spokenBy(v).includes(code)));
  const shown = voices.filter((v) => asked.every((code) => spokenBy(v).includes(code)));
  return { asked, shown, addable: new Set(shown.flatMap(spokenBy)) };
}

/** The languages asked for, with one more or one fewer. */
export const toggled = (asked: string[], code: string) =>
  asked.includes(code) ? asked.filter((c) => c !== code) : [...asked, code];

/** Whether the voices all speak the same languages: they need no codes of
 *  their own then, nor groups. */
export const speakAlike = (voices: VoiceInfo[]) =>
  new Set(voices.map((v) => [...spokenBy(v)].sort().join())).size <= 1;

/** The voices shown, by the language they were made for, in the order they
 *  are listed; one group when they all speak the same. */
export function byOrigin(shown: VoiceInfo[], alike: boolean): [string, VoiceInfo[]][] {
  if (alike) return [["", shown]];
  const groups = new Map<string, VoiceInfo[]>();
  for (const voice of shown) groups.set(voice.language, [...(groups.get(voice.language) ?? []), voice]);
  return [...groups];
}

const mb = (bytes: number) => Math.round(bytes / 1e6);

/** How far a download has come, in percent; null before its size is known. */
export const percent = (job: VoiceJob) =>
  job.total ? Math.min(100, Math.floor((100 * job.done) / job.total)) : null;

/** A job's progress in words: "120 of 219 MB, 55 %", or what it waits for. */
export function progressOf(job: VoiceJob): string {
  if (job.state === "queued") return "Waiting for another download";
  if (job.state === "making") return "Making the voice";
  if (job.stalled) return job.total ? `Stalled at ${mb(job.done)} of ${mb(job.total)} MB` : "Stalled";
  const done = percent(job);
  return done === null ? "Starting" : `${mb(job.done)} of ${mb(job.total)} MB, ${done} %`;
}

export const running = (job?: VoiceJob) => !!job && job.state !== "ready" && job.state !== "failed";
