// A stand-in for Iris's Python core, for the bridge's test: it answers
// JSON-RPC requests on stdin with its process id, and does what its mode says.
//
//   node fake-core.mjs [mode]
//
//   answer     answers; "die" {code, error} makes it write the error to stderr
//              as a traceback ends, and exit with the code
//   no-start   writes an import error to stderr and exits at once, as a core
//              that cannot import its modules does
//   stubborn   answers, and ignores SIGTERM, as a core stuck in native code
//   deaf       closes its input at once and exits a second later, so what is
//              written to it meanwhile finds the pipe closed

import fs from "node:fs";
import readline from "node:readline";

const mode = process.argv[2] ?? "answer";

if (mode === "no-start") {
  process.stderr.write("Traceback (most recent call last):\n");
  process.stderr.write("ModuleNotFoundError: No module named 'iris'\n");
  process.exit(1);
}
if (mode === "stubborn") process.on("SIGTERM", () => {});
if (mode === "deaf") {
  fs.closeSync(0);
  setTimeout(() => process.exit(3), 1000);
}

const send = (frame) => process.stdout.write(JSON.stringify(frame) + "\n");
send({ jsonrpc: "2.0", method: "event", params: { type: "state", data: { state: "idle" } } });

if (mode !== "deaf") readline.createInterface({ input: process.stdin }).on("line", (line) => {
  const request = JSON.parse(line);
  if (request.method === "die") {
    process.stderr.write("some log line\n");
    if (request.params.error) process.stderr.write(`${request.params.error}\n`);
    process.exit(request.params.code ?? 1);
  }
  send({ jsonrpc: "2.0", id: request.id, result: { pid: process.pid } });
});
