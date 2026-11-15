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

  /** Recognition and Kokoro's voices can come from a folder the user has
   *  instead of the download */
  const kindOf = (model: SpeechDownload) =>
    model.id === "stt" ? ("stt" as const) : model.id === "kokoro" ? ("tts" as const) : null;
  const folderOf = (model: SpeechDownload) => {
    const kind = kindOf(model);
    return kind === "stt" ? w.sttPath : kind === "tts" ? w.ttsDir : "";
  };
</script>

<Step title="Speech models" {line}>
  <section class="flex flex-col gap-1.5" aria-label="Speech models">
    {#each w.downloads as model (model.id)}
      {@const kind = kindOf(model)}
      {@const folder = folderOf(model)}
      <div
        class="s-option model {folder ? 'done' : model.state}"
        data-model={model.id}
        data-state={folder ? "folder" : model.state}
      >
        <span class="state" aria-hidden="true">
          {#if folder || model.state === "done"}
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
              {#if folder}
                From folder
              {:else if model.state === "downloading"}
                {percent(model)} %
              {:else if model.state === "waiting"}
                Waiting
              {:else if model.state === "done"}
                Downloaded
              {/if}
            </span>
          </span>
          <span class="flex items-baseline justify-between gap-2">
            {#if folder}
              <span class="meta truncate" title={folder}>{folderName(folder)}</span>
            {:else if model.state === "failed"}
              <span class="meta error truncate" title={model.error}>{model.error || "Failed"}</span>
            {:else}
              <span class="meta truncate">{meta(model)}</span>
            {/if}
            {#if kind}
              {#if folder}
                <button class="link" onclick={() => w.useDownload(kind)}>Download instead</button>
              {:else}
                <button class="link" onclick={() => w.chooseFolder(kind)}>Use a folder…</button>
              {/if}
            {/if}
          </span>
          {#if model.state === "downloading" && !folder}
            <Progress value={model.total ? model.done / model.total : 0} class="mt-0.5" />
          {/if}
        </span>
        {#if model.state === "failed" && !folder}
          <button class="s-btn" onclick={() => w.download()}>Retry</button>
        {/if}
      </div>
    {/each}
  </section>

  {#if w.folderError}
    <p class="s-status error mt-2">{w.folderError}</p>
  {/if}
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
  /* Choosing a folder instead: quiet until pointed at */
  .link {
    flex-shrink: 0;
    font-size: 11px;
    color: var(--color-ink-3);
    background: none;
    border: none;
    padding: 0;
    cursor: pointer;
  }
  .link:hover {
    color: var(--color-ink);
  }
</style>
