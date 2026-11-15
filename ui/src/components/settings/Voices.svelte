<script lang="ts">
  import { onMount } from "svelte";
  import {
    store,
    type EngineInfo,
    type PersonaSettings,
    type SpeechEngine,
    type VoiceInfo,
  } from "../../lib/state.svelte";
  import { byLanguages, byOrigin, running, speakAlike, spokenBy, toggled } from "../../lib/voices";
  import Icon from "../Icon.svelte";
  import Progress from "../Progress.svelte";
  import ClonedVoice from "./ClonedVoice.svelte";
  import LanguageChoice from "./LanguageChoice.svelte";
  import LanguageFilter from "./LanguageFilter.svelte";
  import VoiceClone from "./VoiceClone.svelte";

  // Every voice Iris has: the ones cloned here, managed here, and each
  // engine's own, with the languages each speaks and is ready for, and the
  // personas that speak with it. Iris's languages, which the voices are made
  // ready for, come first. Choosing a persona's voice is the persona's.

  onMount(() => {
    store.loadVoices().catch(() => {});
    store.refreshPersonas().catch(() => {});
  });

  /** What a voice is played as: itself, saying its own name */
  const AS_ITSELF: PersonaSettings = {
    name: "",
    engine: "kyutai",
    voice: "",
    speed: 1,
    audio_filter: "",
    description: "",
  };

  const languages = $derived(store.voices?.languages ?? []);
  const engines = $derived<EngineInfo[]>(store.voices?.engine_info ?? []);
  const all = $derived(engines.flatMap((engine) => store.voices?.engines[engine.id] ?? []));
  /** The languages the voices shown must speak; none shows them all */
  let wanted = $state<string[]>([]);
  const filtered = $derived(byLanguages(all, wanted));
  const cloned = $derived(all.filter((voice) => voice.cloned && filtered.shown.includes(voice)));
  const catalogue = $derived(
    engines
      .map((engine) => {
        const voices = (store.voices?.engines[engine.id] ?? []).filter((v) => !v.cloned);
        const shown = voices.filter((v) => filtered.shown.includes(v));
        return { engine, groups: byOrigin(shown, speakAlike(voices)), count: shown.length };
      })
      .filter((entry) => entry.count)
  );
  const nameOf = (code: string) => languages.find((l) => l.code === code)?.name ?? code;
  /** A voice cloned here just now, to be named */
  let fresh = $state("");
  let error = $state("");

  async function play(engine: SpeechEngine, voice: VoiceInfo) {
    error =
      (await store.previewVoice({ ...AS_ITSELF, name: voice.name, engine, voice: voice.id })) ?? "";
  }

  /** The job downloading a voice, for any language */
  const jobOf = (engine: SpeechEngine, voice: VoiceInfo) =>
    Object.values(store.jobs).find((j) => j.engine === engine && j.voice === voice.id && running(j));

  function readiness(voice: VoiceInfo) {
    const ready = spokenBy(voice).filter((code) => voice.ready?.includes(code));
    const waiting = spokenBy(voice).filter((code) => !voice.ready?.includes(code));
    return [
      ready.length ? `Ready: ${ready.map(nameOf).join(", ")}` : "",
      waiting.length ? `Not ready: ${waiting.map(nameOf).join(", ")}` : "",
    ]
      .filter(Boolean)
      .join("\n");
  }
</script>

<div class="flex max-w-[980px] flex-col gap-8">
  <section class="flex flex-col gap-3" aria-label="Languages">
    <h3 class="s-section">Languages</h3>
    <LanguageChoice />
  </section>

  <section class="flex flex-col gap-5" aria-label="Library" data-tour="voices">
    <div class="flex flex-col gap-3">
      <h3 class="s-section">Voices</h3>
      {#if languages.length}
        <LanguageFilter
          {languages}
          asked={filtered.asked}
          addable={filtered.addable}
          onwant={(code) => (wanted = toggled(filtered.asked, code))}
        />
      {/if}
      {#if !filtered.shown.length && filtered.asked.length}
        <p class="text-[12px] text-ink-3">
          No voice speaks {filtered.asked.map(nameOf).join(" and ")}
        </p>
      {/if}
      {#if error}
        <p class="s-status error">{error}</p>
      {/if}
    </div>

    <section class="engine" aria-label="Cloned">
      <h4 class="engine-name">Cloned</h4>
      {#each cloned as voice (voice.id)}
        <ClonedVoice {voice} focus={voice.id === fresh} onplay={() => play("kyutai", voice)} />
      {/each}
      <VoiceClone draft={AS_ITSELF} keep onchoose={(voice) => (fresh = voice)} />
    </section>

    {#each catalogue as { engine, groups } (engine.id)}
      <section class="engine" aria-label={engine.name}>
        <h4 class="engine-name">{engine.name}</h4>
        {#each groups as [origin, list]}
          <div class="flex flex-col gap-1.5">
            {#if origin}
              <span class="origin">{origin}</span>
            {/if}
            <div class="tiles">
              {#each list as voice (voice.id)}
                {@const job = jobOf(engine.id, voice)}
                <div
                  class="tile"
                  data-engine={engine.id}
                  data-voice={voice.id}
                  title={readiness(voice)}
                >
                  <button
                    class="s-play {store.previewing === `${engine.id}:${voice.id}` ? 'playing' : ''}"
                    onclick={() => play(engine.id, voice)}
                    aria-label="Play {voice.name}"
                  >
                    <Icon name="play" size={10} />
                  </button>
                  <span class="body">
                    <span class="top">
                      <span class="name">{voice.name}</span>
                      {#each voice.personas ?? [] as persona (persona.id)}
                        <button
                          class="user"
                          onclick={() => store.editPersona(persona.id)}
                          title="Open {persona.name}"
                        >
                          {persona.name}
                        </button>
                      {/each}
                    </span>
                    <span class="codes">
                      {#each spokenBy(voice) as code}
                        <span class={voice.ready?.includes(code) ? "" : "unready"}>{code}</span>
                      {/each}
                    </span>
                  </span>
                  {#if job}
                    <Progress
                      value={job.total ? job.done / job.total : 0}
                      waiting={job.state === "making"}
                      class="tile-bar"
                    />
                  {/if}
                </div>
              {/each}
            </div>
          </div>
        {/each}
      </section>
    {/each}
  </section>
</div>

<style>
  .engine {
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
  }
  .engine-name {
    font-size: 12.5px;
    font-weight: 500;
    color: var(--color-ink);
  }
  .origin {
    font-size: 11.5px;
    color: var(--color-ink-3);
  }
  .tiles {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(176px, 1fr));
    gap: 6px;
  }
  .tile {
    position: relative;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.35rem 0.45rem 0.35rem 0.35rem;
    overflow: hidden;
    border-radius: 0.55rem;
    border: 1px solid var(--color-line);
    background: var(--color-raised);
    transition: border-color 150ms;
  }
  .tile:hover {
    border-color: var(--color-line-strong);
  }
  .body {
    display: flex;
    min-width: 0;
    flex: 1;
    flex-direction: column;
  }
  .top {
    display: flex;
    min-width: 0;
    align-items: center;
    gap: 0.35rem;
  }
  .name {
    min-width: 0;
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 12.5px;
    color: var(--color-ink);
  }
  /* A persona that speaks with the voice: it opens in the persona editor */
  .user {
    flex-shrink: 0;
    max-width: 6rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    border-radius: 0.375rem;
    padding: 0 0.4rem;
    font-size: 10.5px;
    line-height: 1.15rem;
    color: var(--color-accent);
    background: color-mix(in srgb, var(--color-accent) 12%, transparent);
  }
  .user:hover {
    background: color-mix(in srgb, var(--color-accent) 20%, transparent);
  }
  /* The languages it speaks, its own first; those not ready yet fainter */
  .codes {
    display: flex;
    gap: 0.3rem;
    font-size: 10px;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    color: var(--color-ink-2);
  }
  .codes .unready {
    color: var(--color-ink-3);
    opacity: 0.6;
  }
  .tile :global(.tile-bar) {
    position: absolute;
    right: 0;
    bottom: 0;
    left: 0;
  }
</style>
