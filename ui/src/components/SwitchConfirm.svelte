<script lang="ts">
  import { store } from "../lib/state.svelte";

  // Opening another thread stops a reply underway: asked first, while one is
</script>

{#if store.switchAsked}
  <div class="confirm" role="alertdialog" aria-label="Switch threads">
    <p>Switch threads? {store.config?.persona ?? "Iris"}'s reply will stop.</p>
    <div class="flex justify-end gap-1.5">
      <button class="s-btn" onclick={() => store.answerSwitch(false)}>Cancel</button>
      <button class="s-btn primary" onclick={() => store.answerSwitch(true)}>Switch</button>
    </div>
  </div>
{/if}

<style>
  .confirm {
    position: absolute;
    top: 0.75rem;
    left: 50%;
    z-index: 20;
    display: flex;
    width: 320px;
    flex-direction: column;
    gap: 0.75rem;
    padding: 0.875rem 1rem;
    border-radius: 12px;
    border: 1px solid var(--color-line-strong);
    background: var(--color-raised);
    box-shadow: 0 16px 40px rgb(0 0 0 / 0.45);
    transform: translateX(-50%);
  }
  .confirm p {
    font-size: 13px;
    line-height: 1.5;
    color: var(--color-ink);
  }
</style>
