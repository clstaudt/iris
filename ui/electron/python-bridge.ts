import { spawn, ChildProcess } from "node:child_process";
import path from "node:path";
import readline from "node:readline";

type PendingRequest = {
  resolve: (value: unknown) => void;
  reject: (reason: unknown) => void;
};

export type IrisEventHandler = (type: string, data: unknown) => void;

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

  /** Called for every server-push event; set once the window exists. */
  onEvent: IrisEventHandler | null = null;

  constructor(
    projectRoot: string,
    isPackaged: boolean,
    resourcesPath: string,
    extraEnv: NodeJS.ProcessEnv = {}
  ) {
    this.projectRoot = projectRoot;
    this.isPackaged = isPackaged;
    this.resourcesPath = resourcesPath;
    this.extraEnv = extraEnv;
    this.spawn();
  }

  private spawn() {
    let cmd: string;
    let args: string[];
    let cwd: string;
    let env: NodeJS.ProcessEnv;

    if (this.isPackaged) {
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
      cmd = path.join(this.projectRoot, ".venv", venvBin, pyExe);
      args = ["-m", "iris.rpc_server"];
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

    this.process = spawn(cmd, args, {
      cwd,
      stdio: ["pipe", "pipe", "pipe"],
      env,
    });

    this.process.stderr?.on("data", (data: Buffer) => {
      console.error(`[python] ${data.toString().trimEnd()}`);
    });

    this.process.on("exit", (code) => {
      console.error(`[python] process exited with code ${code}`);
      for (const [, req] of this.pending) {
        req.reject(new Error(`Python process exited (code ${code})`));
      }
      this.pending.clear();
    });

    this.rl = readline.createInterface({ input: this.process.stdout! });
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
