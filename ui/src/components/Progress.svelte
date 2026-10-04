<script lang="ts">
  /** A thin bar for model downloads; value from 0 to 1. */
  let { value, class: className = "" }: { value: number; class?: string } = $props();

  const percent = $derived(Math.round(Math.min(1, Math.max(0, value)) * 100));
</script>

<div
  class="track {className}"
  role="progressbar"
  aria-valuemin={0}
  aria-valuemax={100}
  aria-valuenow={percent}
>
  <div class="fill" style="width: {percent}%"></div>
</div>

<style>
  .track {
    height: 2px;
    overflow: hidden;
    border-radius: 9999px;
    background: var(--color-line-strong);
  }
  .fill {
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg, var(--color-accent), var(--color-petal-cyan));
    transition: width 300ms ease-out;
  }
</style>
