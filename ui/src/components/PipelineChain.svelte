<script lang="ts">
  import type { PipelineSection, PipelineStage, StageTag } from "../lib/pipeline";
  import Icon from "./Icon.svelte";
  import PersonaRing from "./PersonaRing.svelte";

  /** The pipeline as one line of stages, in sections. With timings, each
   *  timed stage shows its slice of the time to the first word, and a total
   *  follows; without, a stage that can be opened shows a pencil on hover.
   *  Compact, it fits the walkthrough's narrow column. */
  let {
    sections,
    timings = false,
    compact = false,
    active,
    disabled = false,
    label = "Pipeline",
    onselect,
  }: {
    sections: PipelineSection[];
    timings?: boolean;
    compact?: boolean;
    /** The stage open now, for stages that open in place; left out for
     *  stages that lead elsewhere */
    active?: string | null;
    disabled?: boolean;
    label?: string;
    onselect?: (id: string) => void;
  } = $props();

  interface Row {
    key: string;
    color: string;
    head?: string;
    stage?: PipelineStage;
    /** The last of a run of branches */
    end?: boolean;
    /** Where its bar starts and how long it is (%) */
    from?: number;
    width?: number;
  }

  const total = $derived(
    sections
      .flatMap((section) => section.stages)
      .reduce((sum, stage) => sum + (stage.ms ?? 0), 0)
  );

  const rows = $derived.by<Row[]>(() => {
    let start = 0;
    return sections.flatMap((section) => [
      { key: section.id, color: section.color, head: section.label },
      ...section.stages.map((stage, i, list) => {
        const row: Row = {
          key: stage.id,
          color: section.color,
          stage,
          end: stage.branch && !list[i + 1]?.branch,
        };
        if (timings && stage.ms != null && total > 0) {
          row.from = (start / total) * 100;
          row.width = Math.max(0.8, (stage.ms / total) * 100);
          start += stage.ms;
        }
        return row;
      }),
    ]);
  });

  /** The bars of the total, one per measured stage. */
  const totalBars = $derived(
    rows
      .filter((row) => row.width !== undefined)
      .map((row) => ({ key: row.key, color: row.color, from: row.from!, width: row.width! }))
  );

  const seconds = (ms: number) => `${(ms / 1000).toFixed(2)} s`;

  /** A tag opens where its value is set, not the stage's inspector. */
  function openTag(event: MouseEvent, tag: StageTag) {
    event.stopPropagation();
    tag.open();
  }
  const opens = (stage: PipelineStage) => !!onselect && !stage.fixed;
</script>

{#snippet body(stage: PipelineStage, row: Row)}
  <Icon name={stage.icon} size={stage.branch ? 14 : compact ? 15 : 16} />
  <span class="label">{stage.label}</span>
  <span class="cell">
    <span class="value">
      {stage.value}{#if stage.meta}<span class="meta">&nbsp;· {stage.meta}</span>{/if}
    </span>
    {#if stage.tag}
      {@const tag = stage.tag}
      <!-- svelte-ignore a11y_click_events_have_key_events, a11y_no_static_element_interactions -->
      <span class="tag" onclick={(e) => openTag(e, tag)} title="Open">
        {#if tag.persona}
          <PersonaRing name={tag.persona} size="xs" />{tag.persona}
        {:else}
          <Icon name="thread" size={11} />{tag.thread}
        {/if}
      </span>
    {/if}
  </span>
  {#if timings}
    {#if stage.ms !== undefined}
      <span class="ms">{stage.ms !== null ? `${stage.ms} ms` : "–"}</span>
      <span class="track">
        {#if row.width !== undefined}
          <i style="left: {row.from}%; width: {row.width}%; background: {row.color}"></i>
        {/if}
      </span>
    {/if}
  {:else if opens(stage)}
    <Icon name="pencil" size={11} class="edit" />
  {/if}
{/snippet}

<div class="chain {timings ? 'timed' : ''} {compact ? 'compact' : ''}" role="group" aria-label={label}>
  {#each rows as row, i (row.key)}
    <div class="row">
      <span
        class="rail {i === 0 ? 'first' : ''} {i === rows.length - 1 ? 'last' : ''}"
        style="--c: {row.color}"
      >
        {#if row.head}
          <span class="dot sect"></span>
        {:else if !row.stage?.branch}
          <span class="dot"></span>
        {/if}
      </span>
      {#if row.head}
        <h3 class="s-section head">{row.head}</h3>
      {:else if row.stage}
        {@const stage = row.stage}
        {@const classes = `stage ${stage.branch ? "branch" : ""} ${row.end ? "end" : ""} ${stage.off ? "off" : ""} ${stage.warn ? "warn" : ""} ${active === stage.id ? "active" : ""}`}
        {#if opens(stage)}
          <button
            onclick={() => onselect?.(stage.id)}
            {disabled}
            aria-expanded={active !== undefined ? active === stage.id : undefined}
            data-stage={stage.id}
            class="{classes} opens"
          >
            {@render body(stage, row)}
          </button>
        {:else}
          <div data-stage={stage.id} class={classes}>
            {@render body(stage, row)}
          </div>
        {/if}
      {/if}
    </div>
  {/each}
  {#if timings}
    <div class="row">
      <span></span>
      <div class="total">
        <Icon name="clock" size={16} />
        <span class="label">First word</span>
        <span class="value"></span>
        <span class="ms">{total ? seconds(total) : "–"}</span>
        <span class="track">
          {#each totalBars as bar (bar.key)}
            <i style="left: {bar.from}%; width: {bar.width}%; background: {bar.color}"></i>
          {/each}
        </span>
      </div>
    </div>
  {/if}
</div>

<style>
  .chain {
    --cols: 18px 112px minmax(0, 1fr) 12px;
    --branch-cols: 16px 94px minmax(0, 1fr) 12px;
    --stage-h: 3.1rem;
    --branch-h: 2.4rem;
    --head-h: 2.5rem;
    display: flex;
    max-width: 860px;
    flex-direction: column;
  }
  .chain.timed {
    --cols: 18px 112px minmax(0, 1fr) 64px 160px;
    --branch-cols: 16px 94px minmax(0, 1fr) 64px 160px;
  }
  .chain.compact {
    --cols: 16px 104px minmax(0, 1fr) 12px;
    --branch-cols: 14px 86px minmax(0, 1fr) 12px;
    --stage-h: 2rem;
    --branch-h: 1.75rem;
    --head-h: 1.6rem;
  }
  .row {
    display: grid;
    grid-template-columns: 30px minmax(0, 1fr);
  }
  .rail {
    position: relative;
  }
  .rail::before {
    content: "";
    position: absolute;
    left: 14px;
    top: 0;
    bottom: 0;
    width: 2px;
    background: var(--c);
    opacity: 0.4;
  }
  .rail.first::before {
    top: 50%;
  }
  .rail.last::before {
    bottom: 50%;
  }
  .dot {
    position: absolute;
    left: 9px;
    top: 50%;
    width: 12px;
    height: 12px;
    margin-top: -6px;
    border-radius: 9999px;
    border: 2px solid var(--c);
    background: var(--color-panel);
  }
  .compact .dot {
    left: 10px;
    width: 10px;
    height: 10px;
    margin-top: -5px;
  }
  .dot.sect,
  .compact .dot.sect {
    left: 10px;
    width: 10px;
    height: 10px;
    margin-top: -5px;
    border: 0;
    background: var(--c);
  }
  .head {
    display: flex;
    min-height: var(--head-h);
    align-items: center;
    padding-left: 0.75rem;
  }
  .stage {
    position: relative;
    display: grid;
    width: 100%;
    min-height: var(--stage-h);
    grid-template-columns: var(--cols);
    align-items: center;
    gap: 0.75rem;
    margin: 1px 0;
    padding: 0 0.75rem;
    border-radius: 0.6rem;
    border: 1px solid transparent;
    color: var(--color-ink-2);
    text-align: left;
    transition:
      background-color 150ms,
      border-color 150ms;
  }
  .stage.opens:hover:not(:disabled) {
    background: var(--color-raised);
  }
  .stage.active {
    border-color: color-mix(in srgb, var(--color-accent) 50%, transparent);
    background: var(--color-raised);
  }
  .stage :global(.edit) {
    color: var(--color-ink-3);
    opacity: 0;
    transition: opacity 150ms;
  }
  .stage.opens:hover:not(:disabled) :global(.edit) {
    opacity: 1;
  }
  .label {
    font-size: 12.5px;
    color: var(--color-ink-2);
    white-space: nowrap;
  }
  .cell {
    display: flex;
    min-width: 0;
    align-items: center;
  }
  .value {
    overflow: hidden;
    font-size: 13px;
    color: var(--color-ink);
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  /* A value set elsewhere: where, as a tag that opens it */
  .tag {
    display: inline-flex;
    flex-shrink: 0;
    align-items: center;
    gap: 6px;
    margin-left: 10px;
    padding: 2px 9px 2px 6px;
    border-radius: 9999px;
    border: 1px solid var(--color-line-strong);
    font-size: 11.5px;
    color: var(--color-ink-2);
    white-space: nowrap;
    cursor: pointer;
    transition:
      background-color 150ms,
      color 150ms;
  }
  .tag:hover {
    background: var(--color-hover);
    color: var(--color-ink);
  }
  .compact .value {
    font-size: 12.5px;
  }
  .meta {
    font-size: 12px;
    color: var(--color-ink-3);
  }
  .stage.off .value {
    color: var(--color-ink-3);
  }
  .stage.warn .value,
  .stage.warn .meta {
    color: var(--color-danger);
  }
  /* Each timed stage's slice of the time to the first word, on one scale */
  .ms {
    font-size: 12px;
    color: var(--color-ink-2);
    text-align: right;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
  }
  .track {
    position: relative;
    height: 6px;
    border-radius: 3px;
    background: rgb(255 255 255 / 0.04);
  }
  .track i {
    position: absolute;
    top: 0;
    bottom: 0;
    border-radius: 3px;
    transition:
      left 300ms ease,
      width 300ms ease;
  }
  .total {
    display: grid;
    min-height: 3.1rem;
    grid-template-columns: var(--cols);
    align-items: center;
    gap: 0.75rem;
    margin-top: 0.5rem;
    padding: 0 0.75rem;
    border-top: 1px solid var(--color-line);
    color: var(--color-ink-2);
  }
  .total .ms {
    font-size: 13px;
    font-weight: 600;
    color: var(--color-ink);
  }
  /* What the language model draws on, branching off the line */
  .stage.branch {
    min-height: var(--branch-h);
    grid-template-columns: var(--branch-cols);
    padding-left: 2.5rem;
  }
  .stage.branch .value {
    font-size: 12.5px;
    color: var(--color-ink-2);
  }
  .stage.branch::before {
    content: "";
    position: absolute;
    left: 1.25rem;
    top: -2px;
    height: 50%;
    width: 0.75rem;
    border-left: 1.5px solid var(--color-line-strong);
    border-bottom: 1.5px solid var(--color-line-strong);
    border-bottom-left-radius: 7px;
  }
  .stage.branch:not(.end)::after {
    content: "";
    position: absolute;
    left: 1.25rem;
    top: 0;
    bottom: 0;
    border-left: 1.5px solid var(--color-line-strong);
  }
</style>
