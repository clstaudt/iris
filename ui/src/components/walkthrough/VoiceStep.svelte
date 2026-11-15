<script lang="ts">
  import { onMount } from "svelte";
  import { store } from "../../lib/state.svelte";
  import { engineName, voiceName, walkthrough as w } from "../../lib/walkthrough.svelte";
  import Icon from "../Icon.svelte";
  import VoiceLanguages from "../settings/VoiceLanguages.svelte";
  import Step from "./Step.svelte";

  onMount(() => {
    store.loadVoices().catch(() => {});
  });

  /** The chosen persona's voice, with the languages it speaks and is ready for */
  const chosen = $derived(w.personas.find((p) => p.id === w.persona));
  const voice = $derived(
    chosen ? store.voices?.engines[chosen.engine]?.find((v) => v.id === chosen.voice) : undefined
  );
</script>

<Step
  title="Persona"
  line={w.personas.length > 1
    ? "Choose who you talk to: each has a personality and a voice of its own."
    : "Who you talk to, with a personality and a voice of its own."}
>
  <div class="flex flex-col gap-1.5" role="radiogroup" aria-label="Persona">
    {#each w.personas as persona (persona.id)}
      {@const ready = w.voiceReady(persona.engine)}
      <div class="s-option persona {persona.id === w.persona ? 'active' : ''}">
        <button
          class="pick"
          role="radio"
          aria-checked={persona.id === w.persona}
          onclick={() => (w.persona = persona.id)}
        >
          <span class="radio"></span>
          <span class="ring">{persona.name.charAt(0)}</span>
          <span class="flex min-w-0 flex-1 flex-col">
            <span class="name truncate">{persona.name}</span>
            <span class="meta truncate">
              {engineName(persona.engine)} · {voiceName(persona.voice)}
            </span>
          </span>
        </button>
        {#if ready}
          <button
            class="s-play {store.previewing === `${persona.engine}:${persona.voice}` ? 'playing' : ''}"
            onclick={() => w.preview(persona)}
            aria-label="Play {persona.name}"
            title="Play"
          >
            <Icon name="play" size={11} />
          </button>
        {:else}
          <span class="meta shrink-0">Downloading</span>
        {/if}
      </div>
    {/each}
    {#if w.previewError}
      <p class="s-status error">{w.previewError}</p>
    {/if}
    {#if chosen && voice}
      <div class="mt-2 px-1">
        <VoiceLanguages engine={chosen.engine} {voice} label="Languages of {chosen.name}" />
      </div>
    {/if}
  </div>
</Step>

<style>
  .persona {
    gap: 0.5rem;
    padding: 0 0.75rem 0 0;
  }
  .pick {
    display: flex;
    min-width: 0;
    flex: 1;
    align-items: center;
    gap: 0.65rem;
    padding: 0.5rem 0 0.5rem 0.75rem;
    text-align: left;
  }
  /* The persona's initial in a ring, as the persona list shows it */
  .ring {
    display: flex;
    height: 1.75rem;
    width: 1.75rem;
    flex-shrink: 0;
    align-items: center;
    justify-content: center;
    border-radius: 9999px;
    border: 1.5px solid var(--color-ink-2);
    font-size: 11.5px;
    color: var(--color-ink-2);
  }
  .persona.active .ring {
    border-color: var(--color-accent);
    color: var(--color-ink);
  }
</style>
