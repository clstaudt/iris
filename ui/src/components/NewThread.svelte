<script lang="ts">
  import { onMount } from "svelte";
  import { store, type PersonaSettings } from "../lib/state.svelte";
  import Icon from "./Icon.svelte";
  import PersonaRing from "./PersonaRing.svelte";

  // A new thread: who it is with and its focus, chosen here until
  // something is said or typed, which starts it

  let personas = $state<PersonaSettings[]>([]);
  let error = $state("");

  onMount(async () => {
    if (!store.voices) store.loadVoices().catch(() => {});
    store.loadRoles().catch(() => {});
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

  const role = $derived(store.config?.role ?? null);

  /** Another persona, or role, for the new thread; the other choice stays. */
  async function choose(persona: string, chosen: string | null = role) {
    if (persona === store.persona && chosen === role) return;
    await store
      .newThread(persona, { role: chosen })
      .catch((err) => console.error("[threads] new failed:", err));
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
          onclick={() => choose(persona.id!, role)}
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
  <h3 class="s-section mt-5">Role</h3>
  <div class="chips" role="radiogroup" aria-label="Role">
    <button
      class="chip {role === null ? 'active' : ''}"
      role="radio"
      aria-checked={role === null}
      onclick={() => choose(store.persona, null)}
    >
      No role
    </button>
    {#each store.roles as item (item.id)}
      <button
        class="chip {role === item.id ? 'active' : ''}"
        role="radio"
        aria-checked={role === item.id}
        onclick={() => choose(store.persona, item.id!)}
      >
        {item.name}
      </button>
    {/each}
    <button class="chip add" onclick={() => store.editRole()}>
      <Icon name="plus" size={13} />
      New role
    </button>
  </div>
</div>

<style>
  .tiles {
    display: flex;
    max-width: 640px;
    flex-wrap: wrap;
    justify-content: center;
    gap: 10px;
  }
  .chips {
    display: flex;
    max-width: 640px;
    flex-wrap: wrap;
    justify-content: center;
    gap: 8px;
  }
  .chip {
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 7px 14px;
    border-radius: 9999px;
    border: 1px solid var(--color-line);
    font-size: 13px;
    color: var(--color-ink-2);
    transition:
      border-color 150ms,
      color 150ms,
      background-color 150ms;
  }
  .chip:hover {
    border-color: var(--color-line-strong);
    color: var(--color-ink);
  }
  .chip.active {
    border-color: color-mix(in srgb, var(--color-accent) 55%, transparent);
    background: color-mix(in srgb, var(--color-accent) 7%, var(--color-raised));
    color: var(--color-ink);
  }
  .chip.add {
    border-style: dashed;
    color: var(--color-ink-3);
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
