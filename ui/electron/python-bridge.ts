import { spawn, ChildProcess } from "node:child_process";
import path from "node:path";
import readline from "node:readline";

type PendingRequest = {
  resolve: (value: unknown) => void;
  reject: (reason: unknown) => void;
};

export type IrisEventHandler = (type: string, data: unknown) => void;

/** The program that runs the core, instead of the one found by itself */
export type CoreCommand = { command: string; args: string[] };

/** Times the core may stop unexpectedly within RESTART_WINDOW_MS and be
 *  started again; one more, and it is left stopped */
export const MAX_RESTARTS = 3;
export const RESTART_WINDOW_MS = 60_000;
/** A moment before starting it again, so a core that cannot start does not
 *  spin */
export const RESTART_DELAY_MS = 1_000;
/** Lines of the core's stderr kept, to tell why it stopped */
const STDERR_KEPT = 20;
/** A line of a Python traceback that says what went wrong */
const PYTHON_ERROR = /^[A-Za-z_][\w.]*(Error|Exception|Exit|Interrupt)\b.*$/;

/**
 * Manages the Python core process.
 * Communicates via newline-delimited JSON-RPC 2.0 over stdio.
 *
 * Unlike a pure request/response bridge, the Python side also pushes
 * notifications ({"jsonrpc":"2.0","method":"event","params":{type,data}})
 * for state changes, streaming chat chunks, and mic levels. Frames without
 * a matching request id are routed to `onEvent`.
 *
 * In dev mode, spawns <repo>/.venv/bin/python -m iris.rpc_server.
 * In production (packaged app), spawns the PyInstaller-bundled binary
 * from the app's resources directory.
 *
 * A core that stops without being asked to (it crashed, or was killed) is
 * reported to the app as an error and started again, up to MAX_RESTARTS
 * times a minute; the app hears "core.restarted" and starts the assistant
 * in it. A core that cannot be started at all is reported, and not tried
 * again.
 */
export class PythonBridge {
  private process: ChildProcess | null = null;
  private rl: readline.Interface | null = null;
  private pending = new Map<number, PendingRequest>();
  private nextId = 1;
  private projectRoot: string;
  private isPackaged: boolean;
  private resourcesPath: string;
  /** Added to the core's environment (the setup test's own folders) */
  private extraEnv: NodeJS.ProcessEnv;
  private command: CoreCommand | null;
  /** Set by kill(): the core's end is no surprise */
  private stopping = false;
  /** When the core stopped unexpectedly, within the last RESTART_WINDOW_MS */
  private crashes: number[] = [];
  private restartTimer: ReturnType<typeof setTimeout> | null = null;
  private stderrTail: string[] = [];

  /** Called for every server-push event; set once the window exists. */
  onEvent: IrisEventHandler | null = null;

  /** Where the core's stderr goes, line by line */
  onLog: (line: string) => void = (line) => console.error(`[python] ${line}`);

  constructor(
    projectRoot: string,
    isPackaged: boolean,
    resourcesPath: string,
    extraEnv: NodeJS.ProcessEnv = {},
    command: CoreCommand | null = null
  ) {
    this.projectRoot = projectRoot;
    this.isPackaged = isPackaged;
    this.resourcesPath = resourcesPath;
    this.extraEnv = extraEnv;
    this.command = command;
    this.spawn();
  }

  /** The core's process id, while it runs */
  get pid(): number | undefined {
    return this.process?.pid;
  }

  private spawn() {
    let cmd: string;
    let args: string[];
    let cwd: string;
    let env: NodeJS.ProcessEnv;

    if (this.isPackaged && !this.command) {
      const exeName =
        process.platform === "win32" ? "iris-rpc.exe" : "iris-rpc";
      cmd = path.join(this.resourcesPath, "iris-rpc", exeName);
      args = [];
      cwd = path.join(this.resourcesPath, "iris-rpc");
      env = { ...process.env, ...this.extraEnv, PYTHONUNBUFFERED: "1" };
      console.log(`[python-bridge] production mode, spawning: ${cmd}`);
    } else {
      const venvBin = process.platform === "win32" ? "Scripts" : "bin";
      const pyExe = process.platform === "win32" ? "python.exe" : "python";
      cmd = this.command?.command ?? path.join(this.projectRoot, ".venv", venvBin, pyExe);
      args = this.command?.args ?? ["-m", "iris.rpc_server"];
      cwd = this.projectRoot;
      env = {
        ...process.env,
        ...this.extraEnv,
        PYTHONPATH: path.join(this.projectRoot, "src"),
        PYTHONUNBUFFERED: "1",
      };
      console.log(
        `[python-bridge] dev mode, spawning: ${cmd} ${args.join(" ")}`
      );
    }

    this.stderrTail = [];
    const proc = spawn(cmd, args, {
      cwd,
      stdio: ["pipe", "pipe", "pipe"],
      env,
    });
    this.process = proc;

    readline.createInterface({ input: proc.stderr! }).on("line", (line: string) => {
      this.stderrTail.push(line);
      if (this.stderrTail.length > STDERR_KEPT) this.stderrTail.shift();
      this.onLog(line);
    });

    // The program could not be started at all (missing, not executable):
    // starting it again would fail the same way
    proc.on("error", (err) => {
      if (this.process !== proc) return;
      this.process = null;
      this.rejectPending(`Python process could not start: ${err.message}`);
      if (this.stopping) return;
      this.stopping = true;
      this.notify("error", { message: `Iris could not start: ${err.message}`, kind: "core" });
      this.notify("state", { state: "idle" });
    });

    proc.on("exit", (code, signal) => {
      console.error(`[python] process exited with code ${code}${signal ? ` (${signal})` : ""}`);
      if (this.process !== proc) return;
      this.process = null;
      this.rejectPending(`Python process exited (code ${code})`);
      if (!this.stopping) this.stoppedUnexpectedly(code, signal);
    });

    this.rl = readline.createInterface({ input: proc.stdout! });
    this.rl.on("line", (line: string) => {
      let frame: {
        id?: number;
        result?: unknown;
        error?: { message?: string };
        method?: string;
        params?: { type?: string; data?: unknown };
      };
      try {
        frame = JSON.parse(line);
      } catch (err) {
        console.error("[python] failed to parse frame:", line, err);
        return;
      }

      // Server-push notification (no request id)
      if (frame.method === "event" && frame.params) {
        this.onEvent?.(frame.params.type ?? "unknown", frame.params.data ?? {});
        return;
      }

      if (frame.id === undefined) return;
      const req = this.pending.get(frame.id);
      if (req) {
        this.pending.delete(frame.id);
        if (frame.error) {
          req.reject(new Error(frame.error.message || "RPC error"));
        } else {
          req.resolve(frame.result);
        }
      }
    });
  }

  private notify(type: string, data: unknown) {
    this.onEvent?.(type, data);
  }

  private rejectPending(reason: string) {
    for (const [, req] of this.pending) req.reject(new Error(reason));
    this.pending.clear();
  }

  /** Why the core stopped, as well as it can be told: the signal or exit
   *  code, and the error its traceback ended with. */
  private why(code: number | null, signal: NodeJS.Signals | null): string {
    const how = signal ? `stopped by ${signal}` : `exit code ${code}`;
    const error = [...this.stderrTail].reverse().find((line) => PYTHON_ERROR.test(line.trim()));
    return error ? `${how}: ${error.trim()}` : how;
  }

  private stoppedUnexpectedly(code: number | null, signal: NodeJS.Signals | null) {
    const now = Date.now();
    this.crashes = this.crashes.filter((at) => now - at < RESTART_WINDOW_MS);
    this.crashes.push(now);
    const why = this.why(code, signal);
    if (this.crashes.length > MAX_RESTARTS) {
      this.notify("error", {
        message: `Iris stopped unexpectedly again (${why}) and was not started once more. Quit and reopen Iris.`,
        kind: "core",
      });
      this.notify("state", { state: "idle" });
      return;
    }
    this.notify("error", {
      message: `Iris stopped unexpectedly (${why}). Starting it again.`,
      kind: "core",
    });
    this.notify("state", { state: "starting" });
    this.restartTimer = setTimeout(() => {
      this.restartTimer = null;
      if (this.stopping) return;
      this.spawn();
      // The new core runs nothing yet: the app starts the assistant in it
      this.notify("core.restarted", {});
    }, RESTART_DELAY_MS);
  }

  call(method: string, params: Record<string, unknown> = {}): Promise<unknown> {
    return new Promise((resolve, reject) => {
      if (!this.process?.stdin?.writable) {
        reject(new Error("Python process not running"));
        return;
      }

      const id = this.nextId++;
      this.pending.set(id, { resolve, reject });

      const request = JSON.stringify({
        jsonrpc: "2.0",
        id,
        method,
        params,
      });

      this.process.stdin.write(request + "\n");
    });
  }

  kill() {
    this.stopping = true;
    if (this.restartTimer) clearTimeout(this.restartTimer);
    this.rl?.close();
    if (this.process) {
      this.process.kill("SIGTERM");
      const proc = this.process;
      setTimeout(() => {
        if (!proc.killed) proc.kill("SIGKILL");
      }, 3000);
    }
    this.process = null;
  }
}
