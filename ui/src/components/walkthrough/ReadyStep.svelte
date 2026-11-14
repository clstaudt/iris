<script lang="ts">
  import { serverName, walkthrough as w, type Step as StepId } from "../../lib/walkthrough.svelte";
  import Icon from "../Icon.svelte";
  import Step from "./Step.svelte";

  const waiting = $derived(w.firstRun && w.downloads.length > 0 && !w.downloadsDone);
  const title = $derived(!w.firstRun ? "All set" : waiting ? "Nearly there" : "Iris is ready");
  const line = $derived(
    !w.firstRun
      ? "Iris takes on what you changed."
      : waiting
        ? "Iris starts as soon as its speech models are in."
        : "Say hello, or type a message."
  );

  const rows = $derived.by(() => {
    const persona = w.personas.find((p) => p.id === w.persona);
    const microphone = w.inputs.find((d) => d.index === w.micDevice)?.name ?? "System default";
    const tools = w.tools.filter((t) => t.on).map((t) => t.name);
    const models = w.downloadsFailed
      ? "A download failed"
      : w.downloadsDone || !w.downloads.length
        ? "On this computer"
        : `Downloading, ${Math.round(w.downloaded * 100)} %`;
    return [
      {
        step: "server" as StepId,
        label: "Language model",
        value: w.server !== null && w.model ? `${w.model} · ${serverName(w.server)}` : "None",
        warn: w.server === null || !w.model,
      },
      { step: "models" as StepId, label: "Speech models", value: models, warn: w.downloadsFailed },
      { step: "microphone" as StepId, label: "Microphone", value: microphone, warn: false },
      { step: "voice" as StepId, label: "Persona", value: persona?.name ?? w.persona, warn: false },
      {
        step: "extras" as StepId,
        label: "Wake word",
        value: w.wakeOn ? w.wakeWord.trim() || "Iris" : "Off",
        warn: false,
      },
      {
        step: "extras" as StepId,
        label: "Tools",
        value: w.tools.length ? tools.join(", ") || "None" : "As they are",
        warn: false,
      },
    ];
  });
</script>

<Step {title} {line}>
  <section class="flex flex-col gap-1" aria-label="Summary">
    {#each rows as row}
      <button class="row" onclick={() => w.goto(row.step)} disabled={w.starting}>
        <span class="w-[8.5rem] shrink-0 text-ink-3">{row.label}</span>
        <span class="min-w-0 flex-1 truncate {row.warn ? 'text-danger' : 'text-ink'}">{row.value}</span>
        <Icon name="pencil" size={11} class="edit shrink-0" />
      </button>
    {/each}
  </section>
</Step>

<style>
  .row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    min-height: 2.2rem;
    padding: 0 0.75rem;
    border-radius: 0.55rem;
    font-size: 12.5px;
    text-align: left;
    transition: background-color 150ms;
  }
  .row:hover:not(:disabled) {
    background: var(--color-raised);
  }
  .row :global(.edit) {
    color: var(--color-ink-3);
    opacity: 0;
    transition: opacity 150ms;
  }
  .row:hover :global(.edit) {
    opacity: 1;
  }
</style>
