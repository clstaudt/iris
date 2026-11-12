<script lang="ts">
  import { walkthrough as w } from "../../lib/walkthrough.svelte";
  import Icon from "../Icon.svelte";
  import Switch from "../Switch.svelte";
  import Step from "./Step.svelte";

  const ICONS: Record<string, string> = { camera: "camera", screen: "monitor", web: "globe" };
</script>

<Step title="Wake word and tools" line="Both optional, and easy to change later.">
  <section class="flex flex-col gap-2" aria-label="Wake word">
    <h3 class="s-section">Wake word</h3>
    <div class="s-option gap-3">
      <Icon name="ear" size={15} class="shrink-0 text-ink-3" />
      <input
        bind:value={w.wakeWord}
        disabled={!w.wakeOn}
        spellcheck="false"
        aria-label="Wake word"
        class="word min-w-0 flex-1"
      />
      <Switch label="Wake word" checked={w.wakeOn} onchange={(on) => (w.wakeOn = on)} />
    </div>
  </section>

  <section class="flex flex-col gap-2" aria-label="Tools">
    <h3 class="s-section">Tools</h3>
    {#each w.tools as tool (tool.id)}
      <div class="s-option gap-3">
        <Icon name={ICONS[tool.id] ?? "wrench"} size={15} class="shrink-0 text-ink-3" />
        <span class="name flex-1">{tool.name}</span>
        {#if !tool.available}<span class="meta">Not installed</span>{/if}
        <Switch
          label={tool.name}
          checked={tool.on}
          disabled={!tool.available}
          onchange={(on) => (tool.on = on)}
        />
      </div>
    {/each}
  </section>
</Step>

<style>
  .word {
    height: 1.6rem;
    background: transparent;
    font-size: 12.5px;
    color: var(--color-ink);
    outline: none;
  }
  .word:disabled {
    color: var(--color-ink-3);
  }
</style>
