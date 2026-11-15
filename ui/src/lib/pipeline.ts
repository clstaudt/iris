/** How a turn travels through Iris, as one line of stages: what it hears,
 *  how it thinks and how it speaks. Settings shows it with what runs Iris
 *  and how long each stage took; the walkthrough's last step shows it with
 *  the choices just made. */

/** One stage of the line, or something the language model draws on (a
 *  branch). */
export interface PipelineStage {
  id: string;
  icon: string;
  label: string;
  value: string;
  meta?: string;
  /** Not in use, e.g. no wake word */
  off?: boolean;
  /** Something is wrong with it, e.g. a download failed */
  warn?: boolean;
  branch?: boolean;
  /** A stage that takes part of a turn's time: its latest measurement, or
   *  null until it is measured. Left out for the others. */
  ms?: number | null;
  /** Nothing to open for it here */
  fixed?: boolean;
}

export interface PipelineSection<S extends PipelineStage = PipelineStage> {
  id: string;
  label: string;
  color: string;
  stages: S[];
}

const SECTIONS = {
  hear: { label: "Hear", color: "var(--color-petal-cyan)" },
  think: { label: "Think", color: "var(--color-accent)" },
  speak: { label: "Speak", color: "var(--color-petal-warm)" },
};

/** A section of the line in its own colour. */
export function pipelineSection<S extends PipelineStage = PipelineStage>(
  id: keyof typeof SECTIONS,
  stages: S[]
): PipelineSection<S> {
  return { id, ...SECTIONS[id], stages };
}
