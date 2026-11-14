<script lang="ts">
  import { store, type VoiceJob } from "../lib/state.svelte";
  import { progressOf, running } from "../lib/voices";
  import Progress from "./Progress.svelte";

  // A voice being made ready, wherever it is waited on: what downloads or is
  // made, how far it has come, and what for; a stalled or failed one can be
  // tried again.

  let { job, quiet = false }: { job: VoiceJob; quiet?: boolean } = $props();
</script>

<div class="job {quiet ? 'quiet' : ''}" data-job={job.id} data-state={job.state}>
  <div class="head">
    <span class="title">{job.title}</span>
    {#if running(job)}
      <span class="status tabular-nums">{progressOf(job)}</span>
    {/if}
  </div>
  {#if running(job)}
    <Progress value={job.total ? job.done / job.total : 0} waiting={job.state === "making"} />
  {/if}
  {#if job.state === "failed" || job.stalled}
    <div class="head">
      <span class="error" title={job.error}>{job.stalled ? "Nothing has come for a while" : job.error}</span>
      <button class="s-btn" onclick={() => store.retryJob(job.id)}>Retry</button>
    </div>
  {:else if running(job) && job.purpose && !quiet}
    <span class="purpose">{job.purpose}</span>
  {/if}
</div>

<style>
  .job {
    display: flex;
    flex-direction: column;
    gap: 0.4rem;
    width: 100%;
    max-width: 30rem;
    padding: 0.55rem 0.75rem 0.6rem;
    border-radius: 0.6rem;
    border: 1px solid var(--color-line);
    background: var(--color-raised);
    text-align: left;
  }
  .job.quiet {
    max-width: none;
    padding: 0;
    border: none;
    background: none;
  }
  .head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.75rem;
  }
  .title {
    min-width: 0;
    font-size: 12px;
    color: var(--color-ink-2);
  }
  .status,
  .purpose {
    flex-shrink: 0;
    font-size: 11px;
    color: var(--color-ink-3);
  }
  .error {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 11.5px;
    color: var(--color-danger);
  }
</style>
