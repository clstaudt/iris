<script lang="ts">
  import { onMount } from "svelte";
  import { fade } from "svelte/transition";
  import { STEPS, walkthrough as w, type Step as StepId } from "../../lib/walkthrough.svelte";
  import Icon from "../Icon.svelte";
  import Progress from "../Progress.svelte";
  import ExtrasStep from "./ExtrasStep.svelte";
  import Flower from "./Flower.svelte";
  import MicrophoneStep from "./MicrophoneStep.svelte";
  import ModelsStep from "./ModelsStep.svelte";
  import ReadyStep from "./ReadyStep.svelte";
  import ServerStep from "./ServerStep.svelte";
  import Step from "./Step.svelte";
  import VoiceStep from "./VoiceStep.svelte";

  // The first launch opens it here; Settings opens it to run it again
  onMount(() => {
    if (!w.active) w.open(true);
  });

  const LABELS: Record<StepId, string> = {
    welcome: "Welcome",
    server: "Language model",
    models: "Speech models",
    microphone: "Microphone",
    voice: "Voice",
    extras: "Wake word and tools",
    ready: "Ready",
  };

  /** The flower opens with each step reached and each model downloaded, to
   *  about two thirds; the orb's bloom takes it the rest of the way. */
  const open = $derived(
    0.35 * (w.reached / (STEPS.length - 1)) + (w.downloads.length ? 0.35 * w.downloaded : 0)
  );

  const between = $derived(w.step !== "welcome" && w.step !== "ready");
  /** Downloads go on while the later steps are done: a quiet line says so */
  const showDownloads = $derived(
    w.index > STEPS.indexOf("models") && w.step !== "ready" && !w.downloadsDone && w.downloads.length > 0
  );
</script>

<div class="flex min-h-0 flex-1 flex-col items-center px-6" aria-label="Walkthrough">
  {#if !w.firstRun}
    <button onclick={() => w.cancel()} class="s-icon-btn absolute right-3 top-12" title="Close">
      <Icon name="close" size={15} />
    </button>
  {/if}

  <div class="flex w-full max-w-[460px] shrink-0 flex-col items-center pt-2">
    <Flower {open} />
    <nav class="dots" aria-label="Steps">
      {#each STEPS as step, i}
        <button
          class="dot {step === w.step ? 'current' : i <= w.reached ? 'reached' : ''}"
          disabled={i > w.reached || w.starting}
          onclick={() => w.goto(step)}
          aria-label={LABELS[step]}
          aria-current={step === w.step ? "step" : undefined}
          title={LABELS[step]}
        ></button>
      {/each}
    </nav>
  </div>

  {#key w.step}
    <section
      class="flex min-h-0 w-full max-w-[460px] flex-1 flex-col pt-5"
      in:fade={{ duration: 260 }}
      data-step={w.step}
    >
      {#if w.step === "welcome"}
        <div class="pt-6">
          <Step title="Welcome to Iris" line="A few choices, and you can talk." />
        </div>
      {:else if w.step === "server"}
        <ServerStep />
      {:else if w.step === "models"}
        <ModelsStep />
      {:else if w.step === "microphone"}
        <MicrophoneStep />
      {:else if w.step === "voice"}
        <VoiceStep />
      {:else if w.step === "extras"}
        <ExtrasStep />
      {:else}
        <ReadyStep />
      {/if}
    </section>
  {/key}

  <footer class="flex w-full max-w-[460px] shrink-0 items-center gap-3 pb-7 pt-4">
    {#if w.step === "welcome"}
      <div class="flex-1"></div>
      <button class="primary" onclick={() => w.next()}>Get started</button>
      <div class="flex-1"></div>
    {:else}
      <button class="quiet" onclick={() => w.back()} disabled={w.starting}>Back</button>
      <div class="flex min-w-0 flex-1 flex-col items-center gap-1.5">
        {#if showDownloads}
          {#if w.downloadsFailed}
            <button class="quiet !text-danger" onclick={() => w.goto("models")}>
              A download failed
            </button>
          {:else}
            <span class="text-[11.5px] tabular-nums text-ink-3">
              Speech models {Math.round(w.downloaded * 100)} %
            </span>
            <Progress value={w.downloaded} class="w-[120px]" />
          {/if}
        {/if}
      </div>
      {#if between}
        <button class="quiet" onclick={() => w.skip()}>Skip</button>
        <button class="primary" onclick={() => w.next()}>Continue</button>
      {:else}
        <button class="primary" onclick={() => w.finish()} disabled={w.starting}>
          {w.firstRun ? "Start talking" : "Done"}
        </button>
      {/if}
    {/if}
  </footer>
</div>

<style>
  .dots {
    display: flex;
    align-items: center;
    gap: 7px;
    margin-top: 0.4rem;
  }
  .dot {
    height: 6px;
    width: 6px;
    border-radius: 9999px;
    background: var(--color-line-strong);
    transition:
      width 300ms ease,
      background-color 300ms ease;
  }
  .dot.reached {
    background: var(--color-ink-3);
  }
  .dot.reached:hover {
    background: var(--color-ink-2);
  }
  .dot.current {
    width: 18px;
    background: var(--color-accent);
  }
  .primary {
    height: 2.4rem;
    flex-shrink: 0;
    border-radius: 9999px;
    background: var(--color-accent);
    padding: 0 1.4rem;
    font-size: 13.5px;
    font-weight: 500;
    color: var(--color-frame);
    transition: opacity 150ms;
  }
  .primary:hover:not(:disabled) {
    opacity: 0.9;
  }
  .primary:disabled {
    opacity: 0.5;
  }
  .quiet {
    height: 2rem;
    flex-shrink: 0;
    border-radius: 0.45rem;
    padding: 0 0.6rem;
    font-size: 12.5px;
    color: var(--color-ink-3);
    transition: color 150ms;
  }
  .quiet:hover:not(:disabled) {
    color: var(--color-ink);
  }
  .quiet:disabled {
    opacity: 0.4;
  }
</style>
