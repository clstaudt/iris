<script lang="ts">
  import { walkthrough as w } from "../../lib/walkthrough.svelte";
  import Icon from "../Icon.svelte";
  import Step from "./Step.svelte";

  const BARS = 32;
  const mac = window.iris.platform === "darwin";
  const PRIVACY = "System Settings > Privacy & Security > Microphone";

  /** The level on a decibel scale, so the room shows a little and a voice
   *  most of the meter: -60 dB lights nothing, -10 dB everything. */
  const lit = $derived.by(() => {
    const rms = w.level * 0.15;
    if (rms <= 0) return 0;
    return Math.max(0, Math.min(1, (20 * Math.log10(rms) + 60) / 50));
  });

  function onDevice(event: Event) {
    const index = Number((event.target as HTMLSelectElement).value);
    w.chooseMicrophone(index < 0 ? null : index);
  }
</script>

<Step title="Microphone" line="Say something, and Iris shows that it hears you.">
  {#if w.micAccess === "not-determined"}
    <div class="s-option">
      <Icon name="mic" size={15} class="text-ink-3" />
      <span class="name flex-1">Iris needs the microphone</span>
      <button class="s-btn primary" onclick={() => w.allowMicrophone()}>Allow</button>
    </div>
  {:else if w.micAccess === "denied" || w.micAccess === "restricted"}
    <p class="s-status error">
      Iris may not use the microphone. Allow it {mac ? `in ${PRIVACY}` : "in the system's settings"},
      then open Iris again.
    </p>
  {:else if w.micAccess === "granted"}
    <section class="flex flex-col gap-3">
      <select
        class="s-field"
        aria-label="Microphone"
        value={w.micDevice ?? -1}
        onchange={onDevice}
      >
        <option value={-1}>System default</option>
        {#each w.inputs as device}
          <option value={device.index}>{device.name}</option>
        {/each}
      </select>
      <div class="s-option flex-col !items-stretch gap-3 py-4">
        <div class="s-meter wide" aria-hidden="true">
          {#each Array(BARS) as _, i}
            <i class:lit={i < Math.round(lit * BARS)}></i>
          {/each}
        </div>
        <p class="flex items-center justify-center gap-1.5 text-[12.5px]" aria-live="polite">
          {#if w.micError}
            <span class="text-danger">{w.micError}</span>
          {:else if w.heard}
            <span class="flex items-center gap-1.5 text-ok" data-heard>
              <Icon name="check" size={13} />
              Iris hears you
            </span>
          {:else if w.silent}
            <span class="text-center text-danger">
              Only silence comes through.
              {mac ? `Allow the microphone for Iris in ${PRIVACY}.` : "Check that the microphone is on."}
            </span>
          {:else}
            <span class="text-ink-3">Listening</span>
          {/if}
        </p>
      </div>
    </section>
  {/if}
</Step>

<style>
  .wide {
    justify-content: center;
    height: 26px;
  }
  .wide i {
    width: 5px;
  }
  .wide i.lit {
    height: 22px;
  }
</style>
