<script lang="ts">
  import { store } from "../../lib/state.svelte";
  import { engineName, voiceName, walkthrough as w } from "../../lib/walkthrough.svelte";
  import Icon from "../Icon.svelte";
  import Step from "./Step.svelte";
</script>

<Step title="Voice" line="Choose who you talk to. Each has a voice of its own.">
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
