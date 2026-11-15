<script lang="ts">
  /** A thin bar for model downloads; value from 0 to 1. Waiting, for work
   *  whose length is not known (a voice being made), a light runs along it. */
  let {
    value,
    waiting = false,
    class: className = "",
  }: { value: number; waiting?: boolean; class?: string } = $props();

  const percent = $derived(Math.round(Math.min(1, Math.max(0, value)) * 100));
</script>

<div
  class="track {className}"
  role="progressbar"
  aria-valuemin={0}
  aria-valuemax={100}
  aria-valuenow={waiting ? undefined : percent}
>
  {#if waiting}
    <div class="light"></div>
  {:else}
    <div class="fill" style="width: {percent}%"></div>
  {/if}
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
  .light {
    height: 100%;
    width: 40%;
    border-radius: inherit;
    background: linear-gradient(
      90deg,
      transparent,
      var(--color-accent),
      var(--color-petal-cyan),
      transparent
    );
    animation: run 1.6s ease-in-out infinite;
  }
  @keyframes run {
    from {
      transform: translateX(-100%);
    }
    to {
      transform: translateX(250%);
    }
  }
</style>
