<script lang="ts">
  import { onMount } from "svelte";
  import { store, type VoiceJob } from "../../lib/state.svelte";
  import { percent, progressOf, running } from "../../lib/voices";
  import Icon from "../Icon.svelte";
  import Progress from "../Progress.svelte";

  // Iris's languages: for each one chosen, every persona's voice is made
  // ready ahead, downloads and all, with their progress here. They never
  // hold a conversation to them: Iris follows the user into any of its
  // languages, and a voice not ready for one is made ready then.

  onMount(() => {
    if (!store.voices) store.loadVoices().catch(() => {});
  });

  const languages = $derived(store.voices?.languages ?? []);
  /** The voices personas speak with, by engine */
  const used = $derived(
    Object.entries(store.voices?.engines ?? {}).flatMap(([engine, voices]) =>
      voices.filter((v) => v.personas?.length).map((voice) => ({ engine, voice }))
    )
  );

  function readiness(code: string) {
    const speakers = used.filter(({ voice }) => voice.languages?.includes(code));
    const jobs = speakers
      .map(({ engine, voice }) => store.jobFor(engine, voice.id, code))
      .filter((job): job is VoiceJob => !!job);
    const ready = speakers.filter(({ voice }) => voice.ready?.includes(code));
    return {
      speakers,
      busy: jobs.find(running),
      failed: jobs.find((job) => job.state === "failed"),
      ready: speakers.length > 0 && ready.length === speakers.length,
    };
  }

  function toggle(code: string) {
    const chosen = store.languages.includes(code);
    store.setLanguages(chosen ? store.languages.filter((c) => c !== code) : [...store.languages, code]);
  }
</script>

<div class="choice" role="group" aria-label="Iris's languages">
  {#each languages as language (language.code)}
    {@const chosen = store.languages.includes(language.code)}
    {@const now = readiness(language.code)}
    <div class="s-option row {chosen ? 'active' : ''}" data-language={language.code}>
      <button
        class="pick"
        role="checkbox"
        aria-checked={chosen}
        disabled={language.code === "en"}
        onclick={() => toggle(language.code)}
      >
        <span class="box">
          {#if chosen}<Icon name="check" size={11} />{/if}
        </span>
        <span class="name truncate">{language.name}</span>
      </button>
      <span class="state tabular-nums" title={now.busy?.title ?? now.failed?.error ?? ""}>
        {#if now.busy}
          {now.busy.state === "downloading" && percent(now.busy) !== null && !now.busy.stalled
            ? `${percent(now.busy)} %`
            : progressOf(now.busy)}
        {:else if now.failed && chosen}
          <button class="retry" onclick={() => store.retryJob(now.failed!.id)}>Retry</button>
        {:else if !now.speakers.length}
          {chosen ? "Shown, not spoken" : ""}
        {:else if now.ready}
          Ready
        {:else if chosen}
          Not ready
        {/if}
      </span>
      {#if now.busy}
        <Progress
          value={now.busy.total ? now.busy.done / now.busy.total : 0}
          waiting={now.busy.state === "making"}
          class="bar"
        />
      {/if}
    </div>
  {/each}
</div>

<style>
  .choice {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 6px;
  }
  .row {
    position: relative;
    min-height: 2.3rem;
    padding: 0 0.7rem 0 0;
    gap: 0.4rem;
    overflow: hidden;
  }
  .pick {
    display: flex;
    min-width: 0;
    flex: 1;
    align-items: center;
    gap: 0.55rem;
    align-self: stretch;
    padding-left: 0.7rem;
    text-align: left;
  }
  .pick:disabled {
    cursor: default;
  }
  .box {
    display: flex;
    height: 14px;
    width: 14px;
    flex-shrink: 0;
    align-items: center;
    justify-content: center;
    border-radius: 4px;
    border: 1.5px solid var(--color-ink-3);
    color: var(--color-frame);
  }
  .active .box {
    border-color: var(--color-accent);
    background: var(--color-accent);
  }
  .name {
    font-size: 12.5px;
    color: var(--color-ink);
  }
  .state {
    flex-shrink: 0;
    font-size: 11px;
    color: var(--color-ink-3);
  }
  .retry {
    font-size: 11px;
    color: var(--color-danger);
  }
  .retry:hover {
    text-decoration: underline;
  }
  .row :global(.bar) {
    position: absolute;
    right: 0;
    bottom: 0;
    left: 0;
  }
</style>
