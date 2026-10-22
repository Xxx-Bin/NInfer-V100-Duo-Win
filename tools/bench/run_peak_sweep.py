"""Peak prefill / decode sweep through the real serving route.

Every rate here is `tokens / phase wall time` from the server's own `request_done` record, not a
sampled window: a one-second sample can only report a whole prefill chunk divided by its length, so
`/metrics` and the periodic throughput log are not usable as peak figures.

One server process per prefill chunk (execution graphs and the memory plan are startup-fixed), then
`--reps` requests per point, each with a fresh nonce so no prompt repeats and prefix reuse cannot
turn prefill work into a cache hit. The reported peak is the maximum over repetitions: this answers
"how fast can it go", not "what does it sustain".

Example:

    python tools/bench/run_peak_sweep.py \
      --artifact F:/models/ninfer/qwen3_8_27b_nvfp4.ninfer \
      --serve build-win/apps/ninfer-serve.exe \
      --suite both --chunks 1024,2048 --reps 3 \
      --output profiles/bench/peak-27b-nvfp4
"""

from __future__ import annotations

import argparse
import csv
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_MODEL = "qwen3.8-27b"


# --------------------------------------------------------------------------- HTTP


def post_json(url: str, body: dict, timeout: float | None = None) -> dict:
    data = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {detail}") from error


def get_json(url: str, timeout: float = 5.0) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_server(base_url: str, process: subprocess.Popen, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during startup with code {process.returncode}")
        try:
            return get_json(base_url + "/health", timeout=3.0)
        except Exception:
            time.sleep(2.0)
    raise RuntimeError(f"server did not answer /health within {timeout:.0f}s")


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def free_vram_mib() -> list[tuple[int, int]]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            text=True,
            timeout=10,
        )
    except Exception:
        return []
    rows = []
    for line in out.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2:
            rows.append((int(parts[0]), int(parts[1])))
    return rows


def free_ram_mib() -> int | None:
    """Available physical memory. Loading an artifact that does not fit aborts the server with a
    fail-fast code and no diagnostic, so this is checked before launching anything."""
    try:
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_phys", ctypes.c_ulonglong),
                ("avail_phys", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("avail_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("avail_virtual", ctypes.c_ulonglong),
                ("avail_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(status.avail_phys // (1024 * 1024))
    except Exception:
        return None


# ------------------------------------------------------------------ prompt sizing


def make_prompt(lines: int, nonce: str, long_output: bool = False, style: str = "archive") -> str:
    """Deterministic filler whose token count scales linearly with `lines`.

    A decode point only measures decode if the model keeps generating: a closing instruction to
    "reply briefly" ends the request after one MTP round, which reports a per-round rate rather
    than sustained decode. Decode points therefore ask for a long enumeration instead.

    `style="prose"` keeps the same filler but asks for original continuous prose. Combined with
    sampling instead of greedy, drafts from the MTP head get rejected at a realistic rate, so the
    measured decode rate is a production figure rather than the 100%-acceptance ceiling.
    `style="json"` asks for the same archive as structured records: the syntactic skeleton is
    predictable, so it brackets real traffic between the enumeration ceiling and prose."""
    body = [
        f"Long-context benchmark payload. Run nonce {nonce}. This synthetic archive has no "
        "external meaning; retain only its token structure."
    ]
    for index in range(lines):
        body.append(
            f"Archive segment {index:06d}. Signal alpha beta gamma delta epsilon zeta eta theta "
            f"iota kappa lambda mu; record id {nonce}-{index:06d}, checksum "
            f"{(index * 2654435761) & 0xFFFFFFFF:08x}. The purpose is deterministic tokenizer and "
            "attention workload generation, not semantic retrieval."
        )
    if style == "prose":
        body.append(
            "The archive is complete. Write an original piece of continuous prose of at least 600 "
            "words describing what such an archive would hold. Vary sentence length and wording; "
            "do not list segment ids and do not repeat sentences."
        )
    elif style == "json":
        body.append(
            "The archive is complete. Emit a JSON array of objects, one per archive segment, each "
            "with keys id, checksum and note. Write at least 400 objects and emit nothing outside "
            "the array."
        )
    elif long_output:
        body.append(
            "The archive is complete. List one archive segment id per line and keep going until "
            "you have written 600 lines. Do not summarise and do not stop early."
        )
    else:
        body.append("The archive is complete. Reply briefly.")
    return "\n".join(body)


def count_tokens(base_url: str, model: str, text: str) -> int:
    reply = post_json(base_url + "/v1/responses/input_tokens", {"model": model, "input": text})
    return int(reply["input_tokens"])


def calibrate(base_url: str, model: str, target: int, nonce: str,
              long_output: bool = False, style: str = "archive") -> tuple[str, int]:
    """Grow the filler until the checkpoint-native count lands within tolerance of `target`."""
    lines = max(1, int(target / 36))
    tolerance = max(8, int(target * 0.001))
    text, actual = "", 0
    for _ in range(8):
        text = make_prompt(lines, nonce, long_output, style)
        actual = count_tokens(base_url, model, text)
        if abs(actual - target) <= tolerance:
            return text, actual
        lines = max(1, round(lines * target / max(actual, 1)))
    return text, actual


# --------------------------------------------------------------- request records


def read_new_request_done(log_path: Path, offset: int) -> tuple[dict | None, int]:
    """Return the last `request_done` written past `offset`, plus the new offset."""
    record = None
    if not log_path.exists():
        return None, offset
    with open(log_path, "r", encoding="utf-8") as handle:
        handle.seek(offset)
        for line in handle:
            try:
                event = json.loads(line)
            except Exception:
                continue
            if event.get("event") == "request_done":
                record = event
        new_offset = handle.tell()
    return record, new_offset


def point_rates(record: dict) -> dict:
    """Exact phase rates: prefill counts only tokens actually evaluated, decode excludes the
    token prefill emitted."""
    result = record.get("result", {})
    timings = record.get("timings_seconds", {})
    speculative = record.get("speculative", {})
    prefill_tokens = float(result.get("computed_prefill_tokens", 0))
    decode_tokens = float(max(0, int(result.get("completion_tokens", 0)) - 1))
    prefill_seconds = float(timings.get("prefill", 0.0))
    decode_seconds = float(timings.get("decode", 0.0))
    return {
        "prompt_tokens": int(result.get("prompt_tokens", 0)),
        "computed_prefill_tokens": int(prefill_tokens),
        "cache_hit_tokens": int(result.get("prefix_cache_hit_tokens", 0)),
        "completion_tokens": int(result.get("completion_tokens", 0)),
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "ttft_seconds": float(timings.get("ttft", 0.0)),
        "prefill_tok_s": prefill_tokens / prefill_seconds if prefill_seconds > 0 else None,
        "decode_tok_s": decode_tokens / decode_seconds if decode_seconds > 0 else None,
        "spec_backend": speculative.get("backend", "none"),
        "draft_window": speculative.get("draft_window", 0),
        "drafted_tokens": speculative.get("drafted_tokens", 0),
        "accepted_tokens": speculative.get("accepted_tokens", 0),
    }


# ----------------------------------------------------------------------- server


class Server:
    def __init__(self, args: argparse.Namespace, chunk: int, run_dir: Path):
        self.args = args
        self.chunk = chunk
        self.run_dir = run_dir
        self.port = args.port
        self.base_url = f"http://127.0.0.1:{args.port}"
        # Absolute: the child runs with cwd=run_dir, so a relative path would resolve twice.
        self.log_path = (run_dir / f"requests-chunk{chunk}.jsonl").resolve()
        self.process: subprocess.Popen | None = None
        self.offset = 0

    def command(self) -> list[str]:
        args = self.args
        command = [
            str(args.serve),
            str(args.artifact),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--tp", str(args.tp),
            "--devices", args.devices,
            "--kv-dtype", args.kv_dtype,
            "--max-context", str(args.max_context),
            "--kv-capacity", str(args.kv_capacity if args.kv_capacity is not None
                                 else args.max_context),
            "--max-concurrency", str(args.max_concurrency),
            "--prefill-chunk", str(self.chunk),
            "--no-prefix-reuse",
            "--log-stats-interval-ms", "0",
            "--request-log-jsonl", str(self.log_path),
        ]
        if args.spec != "none":
            command += ["--spec", args.spec, "--draft-tokens", str(args.draft_tokens)]
            if args.lm_head_draft:
                command.append("--lm-head-draft")
        if args.greedy:
            command.append("--greedy")
        else:
            command += ["--temperature", str(args.temperature), "--seed", str(args.seed)]
        command += args.serve_arg
        return command

    def start(self) -> None:
        if self.log_path.exists():
            self.log_path.unlink()
        stdout = open(self.run_dir / f"server-chunk{self.chunk}.stdout.log", "w", encoding="utf-8")
        stderr = open(self.run_dir / f"server-chunk{self.chunk}.stderr.log", "w", encoding="utf-8")
        self.process = subprocess.Popen(
            self.command(), stdout=stdout, stderr=stderr, cwd=str(self.run_dir)
        )
        try:
            wait_for_server(self.base_url, self.process, self.args.startup_timeout)
        except Exception as error:
            self.stop()
            tail = ""
            stderr_path = self.run_dir / f"server-chunk{self.chunk}.stderr.log"
            if stderr_path.exists():
                tail = "\n".join(stderr_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-8:])
            raise RuntimeError(f"{error}\nserver stderr tail:\n{tail}") from error
        self.offset = 0

    def stop(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=15)
        self.process = None


# ------------------------------------------------------------------------- sweep


def measure(server: Server, prompt: str, max_output: int, timeout: float) -> dict:
    payload = {
        "model": server.args.model,
        "input": prompt,
        "max_output_tokens": max_output,
        "store": False,
        "reasoning": {"effort": server.args.reasoning_effort},
    }
    if not server.args.greedy:
        payload["temperature"] = server.args.temperature
    post_json(server.base_url + "/v1/responses", payload, timeout=timeout)

    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        record, offset = read_new_request_done(server.log_path, server.offset)
        if record is not None:
            server.offset = offset
            return point_rates(record)
        time.sleep(0.2)
    raise RuntimeError("request completed but no request_done record appeared")


def sweep(args: argparse.Namespace, run_dir: Path) -> list[dict]:
    points: list[dict] = []
    suites = ["prefill", "decode"] if args.suite == "both" else [args.suite]
    chunks = args.chunks if "prefill" in suites else [args.chunks[0]]

    for chunk in chunks:
        server = Server(args, chunk, run_dir)
        print(f"[chunk {chunk}] starting {args.serve.name} on port {args.port}", flush=True)
        server.start()
        try:
            now = time.strftime("%Y%m%dT%H%M%S")
            for suite in suites:
                targets = args.prefill_tokens if suite == "prefill" else args.decode_contexts
                output_tokens = (args.prefill_output_tokens if suite == "prefill"
                                 else args.decode_tokens)
                for target in targets:
                    for rep in range(args.reps):
                        nonce = f"peak-{chunk}-{suite}-{target}-{rep}-{now}"
                        try:
                            prompt, counted = calibrate(server.base_url, args.model, target, nonce,
                                                        long_output=suite == "decode",
                                                        style=args.prompt_style)
                            rates = measure(server, prompt, output_tokens, args.request_timeout)
                        except Exception as error:  # one failed point must not lose the sweep
                            print(f"  {suite:7s} target={target:6d} rep={rep + 1} FAILED: {error}",
                                  flush=True)
                            points.append({
                                "suite": suite, "chunk": chunk, "rep": rep,
                                "target_tokens": target, "counted_tokens": None,
                                "max_output_tokens": output_tokens, "tp": args.tp,
                                "devices": args.devices, "kv_dtype": args.kv_dtype,
                                "spec": args.spec, "error": str(error),
                                "prefill_tok_s": None, "decode_tok_s": None,
                            })
                            continue
                        point = {
                            "suite": suite,
                            "chunk": chunk,
                            "rep": rep,
                            "target_tokens": target,
                            "counted_tokens": counted,
                            "max_output_tokens": output_tokens,
                            "tp": args.tp,
                            "devices": args.devices,
                            "kv_dtype": args.kv_dtype,
                            "spec": args.spec,
                            "draft_tokens": args.draft_tokens if args.spec != "none" else 0,
                            "lm_head_draft": bool(args.lm_head_draft and args.spec != "none"),
                            "greedy": bool(args.greedy),
                            **rates,
                        }
                        points.append(point)
                        prefill = point["prefill_tok_s"]
                        decode = point["decode_tok_s"]
                        print(
                            f"  {suite:7s} target={target:6d} counted={counted:6d} "
                            f"rep={rep + 1}/{args.reps} "
                            f"prefill={prefill:8.1f} t/s decode={decode if decode is not None else float('nan'):7.1f} t/s "
                            f"accept={point['accepted_tokens']}/{point['drafted_tokens']}",
                            flush=True,
                        )
        finally:
            server.stop()
    return points


def summarize(points: list[dict], args: argparse.Namespace, run_dir: Path) -> None:
    """Peak per (chunk, suite, target): the maximum over repetitions."""
    valid = [point for point in points if not point.get("error")]
    if not valid:
        print("\nno usable points; every request failed", file=sys.stderr)
        return
    best: dict[tuple[int, str, int], dict] = {}
    for point in valid:
        key = (point["chunk"], point["suite"], point["target_tokens"])
        current = best.get(key)
        metric = "prefill_tok_s" if point["suite"] == "prefill" else "decode_tok_s"
        if current is None or (point[metric] or 0.0) > (current[metric] or 0.0):
            best[key] = point

    rows = [best[key] for key in sorted(best)]
    with open(run_dir / "peak.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    headline = {
        "prefill": max((r["prefill_tok_s"] or 0.0) for r in rows if r["suite"] == "prefill")
        if any(r["suite"] == "prefill" for r in rows)
        else None,
        "decode": max((r["decode_tok_s"] or 0.0) for r in rows if r["suite"] == "decode")
        if any(r["suite"] == "decode" for r in rows)
        else None,
    }
    summary = {
        "artifact": str(args.artifact),
        "serve": str(args.serve),
        "tp": args.tp,
        "devices": args.devices,
        "kv_dtype": args.kv_dtype,
        "spec": args.spec,
        "draft_tokens": 0 if args.spec == "none" else args.draft_tokens,
        "lm_head_draft": bool(args.lm_head_draft and args.spec != "none"),
        "greedy": bool(args.greedy),
        "temperature": None if args.greedy else args.temperature,
        "prompt_style": args.prompt_style,
        "reps": args.reps,
        "max_context": args.max_context,
        "kv_capacity": args.kv_capacity,
        "peak": headline,
        "points": rows,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\npeak over repetitions")
    print(f"{'suite':8s} {'chunk':>6s} {'target':>7s} {'counted':>8s} {'rate t/s':>10s}")
    for row in rows:
        metric = "prefill_tok_s" if row["suite"] == "prefill" else "decode_tok_s"
        print(
            f"{row['suite']:8s} {row['chunk']:6d} {row['target_tokens']:7d} "
            f"{row['counted_tokens']:8d} {row[metric]:10.1f}"
        )
    if headline["prefill"] is not None:
        print(f"\nprefill peak {headline['prefill']:.1f} t/s")
    if headline["decode"] is not None:
        print(f"decode  peak {headline['decode']:.1f} t/s (single request)")
    print(f"report: {run_dir / 'summary.json'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--serve", type=Path, default=Path("build-win/apps/ninfer-serve.exe"))
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=("prefill", "decode", "both"), default="both")
    parser.add_argument("--chunks", type=lambda s: [int(v) for v in s.split(",")], default=[2048])
    parser.add_argument("--prefill-tokens", type=lambda s: [int(v) for v in s.split(",")],
                        default=[2048, 4096, 8192, 16384, 32768])
    parser.add_argument("--decode-contexts", type=lambda s: [int(v) for v in s.split(",")],
                        default=[512, 8192, 32768])
    parser.add_argument("--decode-tokens", type=int, default=512)
    parser.add_argument("--prefill-output-tokens", type=int, default=16,
                        help="the server's minimum; prefill points only need the phase to end")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--kv-dtype", choices=("bf16", "int8"), default="int8")
    # 262144 tokens of INT8 KV does not fit beside the weights on a 16 GB card and aborts the load
    # with a bare fail-fast code; this is the window the two-card profile is known to admit.
    parser.add_argument("--max-context", type=int, default=184320)
    # Defaults to --max-context. Do not use `auto` for a peak run: it sizes the pool to the card
    # rather than to the window, which over-allocates (2880 of 5760 page groups at 184320) and can
    # abort the load once the MTP proposal head is added.
    parser.add_argument("--kv-capacity", default=None)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--spec", choices=("mtp", "none"), default="mtp")
    parser.add_argument("--draft-tokens", type=int, default=3)
    parser.add_argument("--lm-head-draft", action="store_true")
    parser.add_argument("--greedy", action=argparse.BooleanOptionalAction, default=True,
                        help="default on; pass --no-greedy to sample, which is what a realistic "
                             "MTP acceptance rate needs")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--prompt-style", choices=("archive", "prose", "json"), default="archive",
                        help="archive = long enumeration (predictable, 100%% draft acceptance); "
                             "prose = original continuous prose (pessimistic acceptance); "
                             "json = structured records (between the two)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--serve-arg", action="append", default=[], help="extra server flag; repeatable")
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--request-timeout", type=float, default=3600.0)
    parser.add_argument("--min-free-mib", type=int, default=12000,
                        help="refuse to start when a device has less free VRAM than this")
    parser.add_argument("--min-free-ram-mib", type=int, default=6144,
                        help="refuse to start when available host memory is below this")
    parser.add_argument("--force", action="store_true", help="run despite a busy port or low VRAM")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and every server command")
    args = parser.parse_args()

    run_dir = args.output
    plan_servers = args.chunks if args.suite in ("prefill", "both") else args.chunks[:1]

    if args.dry_run:
        print(f"output: {run_dir}")
        print(f"servers: {len(plan_servers)} (one per prefill chunk)")
        for chunk in plan_servers:
            server = Server(args, chunk, run_dir)
            print(" ".join(server.command()))
        return 0

    if not args.serve.exists():
        print(f"missing server binary: {args.serve}", file=sys.stderr)
        return 2
    if not args.artifact.exists():
        print(f"missing artifact: {args.artifact}", file=sys.stderr)
        return 2
    artifact_mib = args.artifact.stat().st_size // (1024 * 1024)
    print(f"artifact {artifact_mib} MiB, port {args.port}", flush=True)
    if not args.force:
        if not port_is_free(args.port):
            print(f"port {args.port} is already in use; stop that server or pass --force", file=sys.stderr)
            return 2
        needed = {int(part) for part in args.devices.split(",")}
        snapshot = free_vram_mib()
        if not snapshot:
            print(
                "cannot read GPU free memory (nvidia-smi unavailable): a second server could "
                "collide with one already holding the model; pass --force to run anyway",
                file=sys.stderr,
            )
            return 2
        for device, free in snapshot:
            if device in needed and free < args.min_free_mib:
                print(
                    f"GPU {device} has {free} MiB free, below --min-free-mib {args.min_free_mib}: "
                    "another process (a serving instance or a desktop model runner) must release it",
                    file=sys.stderr,
                )
                return 2
        ram = free_ram_mib()
        print(f"host RAM free {ram} MiB", flush=True)
        # The artifact is streamed, not resident: a 22.6 GiB artifact loads with 16 GiB free. Only a
        # hard floor is enforced; being below the artifact size is reported, not refused.
        if ram is not None:
            if ram < args.min_free_ram_mib:
                print(
                    f"host RAM free is {ram} MiB, below --min-free-ram-mib "
                    f"{args.min_free_ram_mib}: the load would abort with a bare fail-fast code",
                    file=sys.stderr,
                )
                return 2
            if ram < artifact_mib:
                print(
                    f"note: free RAM {ram} MiB is below the artifact size {artifact_mib} MiB; this "
                    "loads only because the artifact is streamed, so expect slower first-token times",
                    flush=True,
                )

    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        points = sweep(args, run_dir)
    except RuntimeError as error:
        print(f"\nsweep aborted: {error}", file=sys.stderr)
        return 3
    (run_dir / "points.json").write_text(json.dumps(points, indent=2), encoding="utf-8")
    summarize(points, args, run_dir)
    print(f"elapsed {(time.time() - started) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
