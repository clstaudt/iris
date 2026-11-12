<script lang="ts">
  import { walkthrough as w, type SpeechDownload } from "../../lib/walkthrough.svelte";
  import Icon from "../Icon.svelte";
  import Progress from "../Progress.svelte";
  import Step from "./Step.svelte";

  const size = (mb: number) => (mb >= 1000 ? `${(mb / 1000).toFixed(1)} GB` : `${Math.round(mb)} MB`);

  /** What is left to download, roughly */
  const missing = $derived(
    w.downloads.filter((m) => m.state !== "done").reduce((sum, m) => sum + (m.size_mb ?? 0), 0)
  );
  const line = $derived(
    w.downloads.length && missing === 0
      ? "Iris hears and speaks with these, on this computer."
      : `Iris hears and speaks with these. They download once${missing ? `, about ${size(missing)}` : ""}.`
  );

  function meta(model: SpeechDownload) {
    if (model.state === "downloading" && model.total) {
      return `${model.detail} · ${size(model.done / 1e6)} of ${size(model.total / 1e6)}`;
    }
    return model.size_mb ? `${model.detail} · ${size(model.size_mb)}` : model.detail;
  }

  const percent = (model: SpeechDownload) =>
    model.total ? Math.min(100, Math.floor((100 * model.done) / model.total)) : 0;

  const folderName = (path: string) => path.split(/[\\/]/).filter(Boolean).pop() ?? path;
</script>

<Step title="Speech models" {line}>
  <section class="flex flex-col gap-1.5" aria-label="Speech models">
    {#each w.downloads as model (model.id)}
      <div class="s-option model {model.state}" data-model={model.id} data-state={model.state}>
        <span class="state" aria-hidden="true">
          {#if model.state === "done"}
            <Icon name="check" size={13} />
          {:else if model.state === "failed"}
            <Icon name="alert" size={13} />
          {:else}
            <span class="ring {model.state === 'downloading' ? 'turning' : ''}"></span>
          {/if}
        </span>
        <span class="flex min-w-0 flex-1 flex-col gap-1">
          <span class="flex items-baseline justify-between gap-2">
            <span class="name truncate">{model.label}</span>
            <span class="status tabular-nums">
              {#if model.state === "downloading"}
                {percent(model)} %
              {:else if model.state === "waiting"}
                Waiting
              {:else if model.state === "done"}
                Downloaded
              {/if}
            </span>
          </span>
          {#if model.state === "failed"}
            <span class="meta error truncate" title={model.error}>{model.error || "Failed"}</span>
          {:else}
            <span class="meta truncate">{meta(model)}</span>
          {/if}
          {#if model.state === "downloading"}
            <Progress value={model.total ? model.done / model.total : 0} class="mt-0.5" />
          {/if}
        </span>
        {#if model.state === "failed"}
          <button class="s-btn" onclick={() => w.download()}>Retry</button>
        {/if}
      </div>
    {/each}
  </section>

  <details class="s-more" open={!!(w.sttPath || w.ttsDir)}>
    <summary>Use model folders instead</summary>
    <div class="mt-3 flex flex-col gap-2">
      {#each [{ kind: "stt" as const, label: "Speech recognition", path: w.sttPath }, { kind: "tts" as const, label: "Kokoro voices", path: w.ttsDir }] as row}
        <div class="flex items-center gap-2">
          <span class="w-[8.5rem] shrink-0 text-[12.5px] text-ink-2">{row.label}</span>
          <span
            class="min-w-0 flex-1 truncate text-[12px] {row.path ? 'text-ink' : 'text-ink-3'}"
            title={row.path}
          >
            {row.path ? folderName(row.path) : "Download"}
          </span>
          {#if row.path}
            <button class="s-btn" onclick={() => w.useDownload(row.kind)}>Download</button>
          {/if}
          <button class="s-btn" onclick={() => w.chooseFolder(row.kind)}>Choose…</button>
        </div>
      {/each}
      {#if w.folderError}
        <p class="s-status error">{w.folderError}</p>
      {/if}
    </div>
  </details>
</Step>

<style>
  .model {
    align-items: flex-start;
    padding-top: 0.6rem;
    padding-bottom: 0.6rem;
  }
  .model .s-btn {
    align-self: center;
  }
  .state {
    display: flex;
    height: 1.1rem;
    width: 1rem;
    flex-shrink: 0;
    align-items: center;
    justify-content: center;
    color: var(--color-ink-3);
  }
  .done .state {
    color: var(--color-ok);
  }
  .failed .state {
    color: var(--color-danger);
  }
  /* Waiting: a faint ring; downloading: the ring with a gap, turning slowly
     like the orb's spinner */
  .ring {
    height: 11px;
    width: 11px;
    border-radius: 9999px;
    border: 1.5px solid var(--color-line-strong);
  }
  .ring.turning {
    border-color: var(--color-accent);
    border-right-color: transparent;
    animation: turn 1.6s linear infinite;
  }
  @keyframes turn {
    to {
      transform: rotate(360deg);
    }
  }
  .status {
    flex-shrink: 0;
    font-size: 11px;
    color: var(--color-ink-3);
  }
  .error {
    color: var(--color-danger);
  }
</style>
