<script lang="ts">
  import { store, type SpeechEngine, type VoiceInfo } from "../../lib/state.svelte";
  import { percent, running } from "../../lib/voices";

  // Of Iris's languages, those a voice speaks now; those it speaks and is
  // not ready for yet (its files to download, or a cloned voice to make for
  // them), with how far that has come; and those it does not speak, whose
  // replies are shown, not spoken.

  let { engine, voice, label }: { engine: SpeechEngine; voice: VoiceInfo; label: string } =
    $props();

  const languages = $derived(store.voices?.languages ?? []);
  const speaks = (code: string) => (voice.languages ?? []).includes(code);
  const ready = (code: string) => (voice.ready ?? voice.languages ?? []).includes(code);
  const spoken = $derived(languages.filter((l) => speaks(l.code) && ready(l.code)));
  const waiting = $derived(languages.filter((l) => speaks(l.code) && !ready(l.code)));
  const unspoken = $derived(languages.filter((l) => !speaks(l.code)));

  function pending(code: string, name: string) {
    const job = store.jobFor(engine, voice.id, code);
    const done = job && running(job) ? percent(job) : null;
    return done === null ? name : `${name} ${done} %`;
  }
</script>

<dl class="spoken" aria-label={label}>
  <dt>Spoken</dt>
  <dd>{spoken.map((l) => l.name).join(", ") || "None"}</dd>
  {#if waiting.length}
    <dt>Not ready</dt>
    <dd class="tabular-nums">{waiting.map((l) => pending(l.code, l.name)).join(", ")}</dd>
  {/if}
  {#if unspoken.length}
    <dt>Not spoken</dt>
    <dd class="text-ink-3">{unspoken.map((l) => l.name).join(", ")}</dd>
  {/if}
</dl>

<style>
  .spoken {
    display: grid;
    grid-template-columns: max-content minmax(0, 1fr);
    gap: 0.2rem 0.9rem;
    font-size: 12px;
    color: var(--color-ink-2);
  }
  .spoken dt {
    color: var(--color-ink-3);
  }
</style>
