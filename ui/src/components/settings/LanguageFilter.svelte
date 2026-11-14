<script lang="ts">
  import type { LanguageInfo } from "../../lib/state.svelte";

  // Iris's languages as a row of pills: the voices shown speak every one
  // picked. A language none of them also speaks cannot be added.

  let {
    languages,
    asked,
    addable,
    onwant,
    label = "Voices that speak",
  }: {
    languages: LanguageInfo[];
    asked: string[];
    addable: Set<string>;
    onwant: (code: string) => void;
    label?: string;
  } = $props();
</script>

<div class="langs" role="group" aria-label={label}>
  {#each languages as language}
    <button
      class="lang {asked.includes(language.code) ? 'active' : ''}"
      aria-pressed={asked.includes(language.code)}
      disabled={!asked.includes(language.code) && !addable.has(language.code)}
      onclick={() => onwant(language.code)}
    >
      {language.name}
    </button>
  {/each}
</div>

<style>
  .langs {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
  }
  .lang {
    height: 1.5rem;
    padding: 0 0.55rem;
    border-radius: 9999px;
    border: 1px solid var(--color-line);
    font-size: 11.5px;
    color: var(--color-ink-2);
    transition:
      border-color 150ms,
      color 150ms;
  }
  .lang:hover:not(:disabled) {
    border-color: var(--color-line-strong);
    color: var(--color-ink);
  }
  .lang.active {
    border-color: color-mix(in srgb, var(--color-accent) 45%, transparent);
    background: color-mix(in srgb, var(--color-accent) 10%, transparent);
    color: var(--color-ink);
  }
  .lang:disabled {
    opacity: 0.35;
  }
</style>
