<script lang="ts">
  import { onMount } from "svelte";
  import { store, type PersonaSettings } from "../lib/state.svelte";
  import Icon from "./Icon.svelte";
  import PersonaRing from "./PersonaRing.svelte";

  // A new thread: who it is with, chosen here until something is said or
  // typed, which starts it

  let personas = $state<PersonaSettings[]>([]);
  let error = $state("");

  onMount(async () => {
    if (!store.voices) store.loadVoices().catch(() => {});
    await store.refreshPersonas().catch(() => {});
    const loaded = await Promise.all(store.personas.map((id) => store.loadPersona(id)));
    // Iris, the one a new thread is with unless another is chosen, first
    personas = loaded
      .filter((p): p is PersonaSettings => p !== null)
      .sort((a, b) => Number(b.id === "iris") - Number(a.id === "iris"));
  });

  function voiceOf(persona: PersonaSettings): string {
    const engine =
      store.voices?.engine_info?.find((e) => e.id === persona.engine)?.name ?? persona.engine;
    const voice =
      store.voices?.engines[persona.engine]?.find((v) => v.id === persona.voice)?.name ??
      persona.voice;
    return `${engine.charAt(0).toUpperCase()}${engine.slice(1)} · ${voice}`;
  }

  async function choose(id: string) {
    if (id === store.persona) return;
    await store.newThread(id).catch((err) => console.error("[threads] new failed:", err));
  }

  async function preview(event: MouseEvent, persona: PersonaSettings) {
    event.stopPropagation();
    error = (await store.previewVoice({ ...persona })) ?? "";
  }
</script>

<div class="flex min-h-0 flex-1 flex-col items-center justify-center gap-3 overflow-y-auto px-8 pb-4">
  <h3 class="s-section">Persona</h3>
  <div class="tiles" role="radiogroup" aria-label="Persona">
    {#each personas as persona (persona.id)}
      <div class="tile {persona.id === store.persona ? 'active' : ''}">
        <button
          class="pick"
          role="radio"
          aria-checked={persona.id === store.persona}
          onclick={() => choose(persona.id!)}
        >
          <PersonaRing name={persona.name} size="lg" />
          <span class="name">{persona.name}</span>
          <span class="voice">{voiceOf(persona)}</span>
        </button>
        <button
          class="play {store.previewing === `${persona.engine}:${persona.voice}` ? 'playing' : ''}"
          onclick={(e) => preview(e, persona)}
          aria-label="Play {persona.name}"
          title="Play"
        >
          <Icon name="play" size={10} />
        </button>
      </div>
    {/each}
  </div>
  {#if error}
    <p class="s-status error">{error}</p>
  {/if}
</div>

<style>
  .tiles {
    display: flex;
    max-width: 640px;
    flex-wrap: wrap;
    justify-content: center;
    gap: 10px;
  }
  .tile {
    position: relative;
    width: 132px;
    border-radius: 14px;
    border: 1px solid var(--color-line);
    background: var(--color-raised);
    transition:
      border-color 150ms,
      background-color 150ms;
  }
  .tile:hover {
    border-color: var(--color-line-strong);
  }
  .tile.active {
    border-color: color-mix(in srgb, var(--color-accent) 55%, transparent);
    background: color-mix(in srgb, var(--color-accent) 7%, var(--color-raised));
  }
  .pick {
    display: flex;
    width: 100%;
    flex-direction: column;
    align-items: center;
    gap: 10px;
    padding: 18px 10px 14px;
  }
  .name {
    font-size: 13.5px;
    color: var(--color-ink);
  }
  .voice {
    max-width: 100%;
    overflow: hidden;
    font-size: 11.5px;
    color: var(--color-ink-3);
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .play {
    position: absolute;
    top: 8px;
    right: 8px;
    display: flex;
    height: 22px;
    width: 22px;
    align-items: center;
    justify-content: center;
    border-radius: 9999px;
    color: var(--color-ink-3);
    opacity: 0;
    transition:
      opacity 150ms,
      background-color 150ms,
      color 150ms;
  }
  .tile:hover .play,
  .play.playing {
    opacity: 1;
  }
  .play:hover {
    background: var(--color-hover);
    color: var(--color-ink);
  }
  .play.playing {
    background: color-mix(in srgb, var(--color-petal-warm) 22%, transparent);
    color: var(--color-petal-warm);
  }
</style>
