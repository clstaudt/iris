<script lang="ts">
  /** A persona's mark: a quiet ring holding its initial, with a soft glow.
   *  Neutral for every persona: the orb's hues tell its state, never who. */
  let { name, size = "sm" }: { name: string; size?: "xs" | "sm" | "lg" } = $props();
</script>

<span class="ring {size}" aria-hidden="true">{name.trim().charAt(0).toUpperCase()}</span>

<style>
  .ring {
    display: inline-flex;
    flex-shrink: 0;
    align-items: center;
    justify-content: center;
    border-radius: 9999px;
    border: 1.5px solid var(--color-ink-3);
    color: var(--color-ink-2);
    font-weight: 500;
    line-height: 1;
    box-shadow:
      0 0 6px rgb(228 229 233 / 0.14),
      inset 0 0 4px rgb(228 229 233 / 0.08);
  }
  .xs {
    width: 14px;
    height: 14px;
    border-width: 1.2px;
    font-size: 7.5px;
  }
  .sm {
    width: 20px;
    height: 20px;
    font-size: 10px;
  }
  .lg {
    width: 40px;
    height: 40px;
    font-size: 16px;
    box-shadow:
      0 0 14px rgb(228 229 233 / 0.14),
      inset 0 0 9px rgb(228 229 233 / 0.08);
  }
</style>
