<script lang="ts">
  import { onMount } from "svelte";
  import { store } from "../../lib/state.svelte";
  import Icon from "../Icon.svelte";
  import PersonaForm from "./PersonaForm.svelte";

  // Pick up personas added since launch
  onMount(() => {
    store.refreshPersonas().catch(() => {});
    if (!store.voices) store.loadVoices().catch(() => {});
  });

  const name = (id: string) => store.personaNames[id] ?? id;
</script>

<div class="grid max-w-[980px] grid-cols-[200px_minmax(0,1fr)] items-start gap-6">
  <div class="flex flex-col gap-1.5">
    {#each store.personas as id}
      <button
        onclick={() => (store.editingPersona = id)}
        aria-current={store.editingPersona === id ? "true" : undefined}
        class="card {store.editingPersona === id ? 'active' : ''}"
      >
        <span class="ring">{name(id).charAt(0)}</span>
        <span class="min-w-0 flex-1 truncate">{name(id)}</span>
        {#if id === store.persona}
          <span class="s-chip accent">Active</span>
        {/if}
      </button>
    {/each}
    <button
      onclick={() => (store.editingPersona = "")}
      class="s-btn mt-1 {store.editingPersona === '' ? 'text-ink' : ''}"
    >
      <Icon name="plus" size={13} />
      New persona
    </button>
  </div>

  {#key store.editingPersona}
    <PersonaForm />
  {/key}
</div>

<style>
  .card {
    display: flex;
    width: 100%;
    align-items: center;
    gap: 0.65rem;
    padding: 0.5rem 0.6rem;
    border-radius: 0.65rem;
    border: 1px solid var(--color-line);
    background: var(--color-raised);
    font-size: 13px;
    color: var(--color-ink);
    text-align: left;
    transition: border-color 150ms;
  }
  .card:hover {
    border-color: var(--color-line-strong);
  }
  .card.active {
    border-color: color-mix(in srgb, var(--color-accent) 55%, transparent);
  }
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
</style>
