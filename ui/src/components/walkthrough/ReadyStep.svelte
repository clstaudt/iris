<script lang="ts">
  import { pipelineSection, type PipelineStage } from "../../lib/pipeline";
  import { store } from "../../lib/state.svelte";
  import {
    DEFAULT_WAKE_WORD,
    engineName,
    serverName,
    voiceName,
    walkthrough as w,
    type Step as StepId,
  } from "../../lib/walkthrough.svelte";
  import PipelineChain from "../PipelineChain.svelte";
  import Step from "./Step.svelte";

  const waiting = $derived(w.firstRun && w.downloads.length > 0 && !w.downloadsDone);
  const title = $derived(!w.firstRun ? "All set" : waiting ? "Nearly there" : "Iris is ready");
  const line = $derived(
    !w.firstRun
      ? "Iris takes on what you changed."
      : waiting
        ? "Iris starts as soon as its speech models are in."
        : "Say hello, or type a message."
  );

  /** A stage, and the step of the walkthrough that sets it */
  type Stage = PipelineStage & { step?: StepId };

  const folderName = (path: string) => path.split(/[\\/]/).filter(Boolean).pop() ?? path;

  /** Where a speech model's download is, while it is not in yet */
  function fetching(id: string): { meta?: string; warn?: boolean } {
    const model = w.downloads.find((m) => m.id === id);
    if (!model || model.state === "done") return {};
    if (model.state === "failed") return { meta: "download failed", warn: true };
    if (model.state === "waiting") return { meta: "waiting" };
    const percent = model.total ? Math.min(100, Math.floor((100 * model.done) / model.total)) : 0;
    return { meta: `${percent} %` };
  }

  /** The pipeline as the walkthrough's choices make it: no core config yet,
   *  and nothing measured. */
  const sections = $derived.by(() => {
    const persona = w.personas.find((p) => p.id === w.persona);
    const microphone = w.inputs.find((d) => d.index === w.micDevice)?.name ?? "System default";
    const tools = w.tools.filter((t) => t.on).map((t) => t.name);
    const recognition = w.downloads.find((m) => m.id === "stt");
    const turn = w.downloads.find((m) => m.id === "turn");
    const wakeWord = w.wakeWord.trim() || DEFAULT_WAKE_WORD;
    const config = store.config;
    const noModel = w.server === null || !w.model;
    return [
      pipelineSection<Stage>("hear", [
        { id: "mic", icon: "mic", label: "Microphone", value: microphone, step: "microphone" },
        {
          id: "wake",
          icon: "wave",
          label: "Wake word",
          value: w.wakeOn ? wakeWord : "Off",
          off: !w.wakeOn,
          step: "extras",
          ...(w.wakeOn ? fetching("wake") : {}),
        },
        {
          id: "turn",
          icon: "timer",
          label: "Turn end",
          value: config?.turn_detection === "silence" ? "Silence" : "Smart Turn",
          step: turn ? "models" : undefined,
          ...fetching("turn"),
        },
        {
          id: "stt",
          icon: "ear",
          label: "Recognition",
          value: w.sttPath ? folderName(w.sttPath) : (recognition?.detail ?? "Whisper"),
          step: "models",
          ...(w.sttPath ? {} : fetching("stt")),
        },
      ]),
      pipelineSection<Stage>("think", [
        {
          id: "llm",
          icon: "cpu",
          label: "Language model",
          value: noModel ? "None" : w.model,
          meta: noModel ? undefined : serverName(w.server!),
          warn: noModel,
          step: "server",
        },
        {
          id: "persona",
          icon: "user",
          label: "Persona",
          value: persona?.name ?? w.persona,
          branch: true,
          step: "voice",
        },
        {
          id: "tools",
          icon: "wrench",
          label: "Tools",
          value: w.tools.length ? tools.join(", ") || "None" : "As they are",
          off: w.tools.length > 0 && tools.length === 0,
          branch: true,
          step: "extras",
        },
        {
          id: "memory",
          icon: "brain",
          label: "Memory",
          value: config && !config.memory ? "Off" : config?.amnesia ? "Paused" : "On",
          off: !!config && (!config.memory || config.amnesia),
          branch: true,
        },
      ]),
      pipelineSection<Stage>("speak", [
        {
          id: "voice",
          icon: "message",
          label: "Voice",
          value: persona ? `${engineName(persona.engine)} · ${voiceName(persona.voice)}` : "–",
          step: "voice",
          ...(persona && !w.ttsDir ? fetching(persona.engine) : {}),
        },
        {
          id: "output",
          icon: "volume",
          label: "Output",
          value: store.speakReplies ? "Spoken" : "Text only",
          off: !store.speakReplies,
        },
      ]),
    ].map((section) => ({
      ...section,
      stages: section.stages.map((stage) => ({ ...stage, fixed: !stage.step })),
    }));
  });

  /** The step that sets each stage */
  const steps = $derived(
    new Map(sections.flatMap((s) => s.stages).map((stage) => [stage.id, stage.step]))
  );

  function select(id: string) {
    const step = steps.get(id);
    if (step) w.goto(step);
  }
</script>

<Step {title} {line}>
  <PipelineChain {sections} compact disabled={w.starting} label="Summary" onselect={select} />
</Step>
