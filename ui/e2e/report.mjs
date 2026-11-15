// What the end-to-end test's steps did, as its runner (run.mjs) tells it:
// plain lines, or, with E2E_EVENTS, JSON lines on stdout for the reporter
// that `npm run e2e` starts (src/irisdev/e2e_report.py), which shows them.
//
// Each event is a line of "@@e2e " and an object, by its type:
//   run    the steps in the order they are written, each with its area
//          ("group"), whether E2E_ONLY lets it run, and its typical duration
//          in seconds from earlier runs (null when none passed yet)
//   start  a step begins: name, group
//   step   a step ended: name, group, ok, seconds; a failed one also with
//          reason, states, errors, page and screenshot
//   skip   a step E2E_ONLY leaves out: name, group
//   log    a line the core ("core"), the app's bridge to it ("app") or the
//          runner ("runner") wrote: source, line
//   done   the run ended: passed, failed, skipped, seconds

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { format } from "node:util";

const PREFIX = "@@e2e ";
/** The steps' last durations, kept between runs for the reporter's ETA */
const TIMINGS = path.join(path.dirname(fileURLToPath(import.meta.url)), "timings.json");
/** How many of a step's last durations its typical one is the median of */
const KEPT = 5;
/** The steps that run whatever E2E_ONLY picks: they start Iris and choose the model */
const ALWAYS = ["first run", "the app starts", "language model: the server's models"];

/** A step's area, by the first of these its name matches; else what its
 *  name says before a colon ("threads: …"), or "other" */
const AREAS = [
  [/^first run:|^the app starts/, "first run"],
  [/\bStop\b/, "Stop"],
  [/after a restart|^the core stops|^settings come back/, "restarts"],
  [/^settings open|^every stage|^(language model|servers|turn end|recognition|tools|wake word):/, "settings"],
  [/^voice:/, "personas"],
  [/^(message box|pipeline):/, "conversation"],
  [/^the walkthrough runs again/, "walkthrough"],
  [/^a speaker /, "speaker"],
  [/^a reload /, "reload"],
  [/^presence /, "presence"],
  [/^nothing went wrong in the page/, "page"],
];

export function area(name) {
  for (const [pattern, group] of AREAS) if (pattern.test(name)) return group;
  return name.match(/^([^:,]{1,24}):\s/)?.[1] ?? "other";
}

/** The steps a runner's source holds, in order: every step("…") */
function plan(source) {
  const text = fs.readFileSync(source, "utf8");
  return [...text.matchAll(/\bstep\(\s*("(?:[^"\\\n]|\\.)*")/g)].map((m) => JSON.parse(m[1]));
}

function readTimings() {
  try {
    return JSON.parse(fs.readFileSync(TIMINGS, "utf8"));
  } catch {
    return {};
  }
}

const median = (values) => {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
};

export class Report {
  /** @param source the runner's own file, whose steps are announced */
  constructor(source) {
    this.events = Boolean(process.env.E2E_EVENTS);
    this.only = process.env.E2E_ONLY?.toLowerCase().split("|");
    this.results = [];
    this.skipped = 0;
    this.began = Date.now();
    this.started = 0;
    this.timings = readTimings();
    if (!this.events) return;
    // Whatever else is written, the bridge's lines among it, goes as a log
    // line, so that stdout holds only events
    for (const method of ["log", "info", "warn", "error", "debug"]) {
      console[method] = (...args) => {
        const line = format(...args);
        this.log(/^\[python(-bridge)?\]/.test(line) ? "app" : "runner", line);
      };
    }
    const steps = plan(source).map((name) => {
      const last = this.timings[name];
      return { name, group: area(name), run: this.selected(name), typical: last?.length ? median(last) : null };
    });
    this.emit({ type: "run", steps, only: process.env.E2E_ONLY ?? null });
  }

  emit(event) {
    process.stdout.write(PREFIX + JSON.stringify(event) + "\n");
  }

  /** Whether E2E_ONLY lets a step run */
  selected(name) {
    const lower = name.toLowerCase();
    return !this.only || this.only.some((part) => lower.includes(part)) || ALWAYS.some((s) => name.startsWith(s));
  }

  skip(name) {
    this.skipped++;
    if (this.events) this.emit({ type: "skip", name, group: area(name) });
  }

  /** A step begins; returns when */
  start(name) {
    this.started = Date.now();
    if (this.events) this.emit({ type: "start", name, group: area(name) });
    return this.started;
  }

  get seconds() {
    return Math.round((Date.now() - this.started) / 10) / 100;
  }

  pass(name) {
    const seconds = this.seconds;
    this.results.push({ name, ok: true, seconds });
    this.timings[name] = [...(this.timings[name] ?? []), seconds].slice(-KEPT);
    try {
      const lines = Object.entries(this.timings).map(([step, last]) => `  ${JSON.stringify(step)}: ${JSON.stringify(last)}`);
      fs.writeFileSync(TIMINGS, `{\n${lines.join(",\n")}\n}\n`);
    } catch {
      // Without them the reporter only guesses how long the rest takes
    }
    if (this.events) this.emit({ type: "step", name, group: area(name), ok: true, seconds });
    else console.log(`  ok    ${name} (${seconds.toFixed(1)} s)`);
  }

  fail(name, { reason, states, errors, page, shot }) {
    const seconds = this.seconds;
    const error =
      reason +
      ` [states: ${states.join(" > ") || "none"}]` +
      (errors.length ? ` [errors: ${errors.join(" | ")}]` : "") +
      (page.length ? ` [page: ${page.join(" | ")}]` : "");
    this.results.push({ name, ok: false, seconds, error, shot });
    if (this.events) {
      this.emit({ type: "step", name, group: area(name), ok: false, seconds, reason, states, errors, page, screenshot: shot });
    } else {
      console.log(`  FAIL  ${name}\n        ${error}\n        ${shot}`);
    }
  }

  /** A log line, for the reporter; without it the core's go only to its log file */
  log(source, line) {
    if (this.events) this.emit({ type: "log", source, line });
  }

  /** Tells how the run went, and ends it: failed when a step did */
  finish() {
    const failed = this.results.filter((result) => !result.ok).length;
    const passed = this.results.length - failed;
    if (this.events) {
      const seconds = Math.round((Date.now() - this.began) / 100) / 10;
      this.emit({ type: "done", passed, failed, skipped: this.skipped, seconds });
    } else {
      console.log(`\n${passed} passed, ${failed} failed`);
    }
    // Once stdout is written out: to a pipe it is written asynchronously
    process.stdout.write("", () => process.exit(failed ? 1 : 0));
  }
}
