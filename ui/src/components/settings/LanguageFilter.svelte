<script lang="ts">
  import type { LanguageInfo } from "../../lib/state.svelte";
  import Icon from "../Icon.svelte";

  // Iris's languages as a row of pills: the voices shown speak every one
  // picked. A language none of them also speaks cannot be added. Some may
  // be picked always, and some marked as downloading something first.

  let {
    languages,
    asked,
    addable,
    onwant,
    label = "Voices that speak",
    fixed = [],
    downloads = {},
  }: {
    languages: LanguageInfo[];
    asked: string[];
    addable: Set<string>;
    onwant: (code: string) => void;
    label?: string;
    /** Picked whatever else is: they cannot be taken out */
    fixed?: string[];
    /** The languages that download something first, with what, by code */
    downloads?: Record<string, string>;
  } = $props();
</script>

<div class="langs" role="group" aria-label={label}>
  {#each languages as language}
    {@const download = downloads[language.code]}
    <button
      class="lang {asked.includes(language.code) ? 'active' : ''}"
      class:always={fixed.includes(language.code)}
      aria-pressed={asked.includes(language.code)}
      disabled={fixed.includes(language.code) ||
        (!asked.includes(language.code) && !addable.has(language.code))}
      title={download}
      data-language={language.code}
      data-download={download ? "" : undefined}
      onclick={() => onwant(language.code)}
    >
      {language.name}
      {#if download}
        <Icon name="download" size={10} class="mark" />
      {/if}
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
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
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
  .lang.always:disabled {
    opacity: 1;
    cursor: default;
  }
  .lang :global(.mark) {
    color: var(--color-ink-3);
  }
</style>
