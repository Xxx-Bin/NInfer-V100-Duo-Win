"""TP2 stability soak on Windows: long context, Chinese artifact path, concurrency 1.

Starts `ninfer-serve.exe` as a child process, drives OpenAI chat completions over
HTTP with 8k/16k/24k/32k-token prompts that each end with a retrieval question, and
samples VRAM at 1 Hz. The Chinese-named artifact path exercises the Windows
mapping/read path; the request cycle exercises KV allocation and release at four
context sizes. Python is used instead of PowerShell because Windows PowerShell 5.1
decodes BOM-less UTF-8 scripts as ANSI and its HTTP stack trips the server parser.

Example:
    python tools\\tp2\\win\\soak.py --minutes 10
    python tools\\tp2\\win\\soak.py --minutes 1 --out _scratch\\soak-smoke
"""

import argparse
import csv
import json
import os
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from make_prompt import DEFAULT_NEEDLE, build, write_messages

ROOT = Path(__file__).resolve().parents[3]
EXE = ROOT / "build-win" / "apps" / "ninfer-serve.exe"
DEFAULT_MODEL_DIR = Path(r"F:\models\ninfer")

# (label, repetitions, measured prompt tokens). 20.1 tokens per repetition measured
# on qwen3.8-27b/nvfp4; the CLI reports the true count on every run.
SIZES = [("8k", 380, 7623), ("16k", 763, 15310), ("24k", 1140, 22870), ("32k", 1556, 31170)]

ZH_PROMPT = "用一句中文回答：保险库的访问码是 918273，请只重复这六个数字。"

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def find_chinese_model(model_dir: Path) -> Path:
    for path in model_dir.glob("*.ninfer"):
        if any("\u4e00" <= ch <= "\u9fff" for ch in path.name):
            return path
    raise SystemExit(f"no Chinese-named .ninfer artifact under {model_dir}")


def gpu_used() -> tuple[int, int]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    return int(out[0]), int(out[1])


def preflight() -> None:
    """Refuse to start on top of a leftover engine process: the weight upload needs the GPUs empty."""
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
        capture_output=True, text=True, check=False,
    ).stdout.strip()
    if apps:
        raise SystemExit(f"GPU is not idle; stop it first:\n{apps}")


def bind_lifetime_to_parent(pid: int) -> bool:
    """Put the server in a kill-on-close job object so it cannot outlive this process.

    A killed soak must not leave a 10 GiB server behind: the orphan keeps both GPUs busy and
    the next run's preflight refuses to start.
    """
    if os.name != "nt":
        return False
    try:
        import ctypes
        from ctypes import wintypes

        class BasicLimits(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
                ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
                ("LimitFlags", wintypes.DWORD),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", wintypes.DWORD),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", wintypes.DWORD),
                ("SchedulingClass", wintypes.DWORD),
            ]

        class IoCounters(ctypes.Structure):
            _fields_ = [
                (name, ctypes.c_ulonglong)
                for name in (
                    "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
                    "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
                )
            ]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", BasicLimits),
                ("IoInfo", IoCounters),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return False
        info = ExtendedLimits()
        info.BasicLimitInformation.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
            kernel32.CloseHandle(job)
            return False
        handle = kernel32.OpenProcess(0x0101, False, pid)  # PROCESS_SET_QUOTA | PROCESS_TERMINATE
        if not handle:
            kernel32.CloseHandle(job)
            return False
        assigned = bool(kernel32.AssignProcessToJobObject(job, handle))
        kernel32.CloseHandle(handle)
        if not assigned:
            kernel32.CloseHandle(job)
            return False
        return True
    except Exception:  # noqa: BLE001 - lifetime binding is best-effort
        return False


def health(base: str) -> dict:
    with opener.open(base + "/health", timeout=5) as resp:
        return json.load(resp)


def chat(base: str, messages: list, max_tokens: int = 16, timeout: float = 600.0) -> dict:
    payload = {
        "model": "qwen3.8-27b",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": False,
        "enable_thinking": False,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        base + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with opener.open(request, timeout=timeout) as resp:
            reply = json.load(resp)
        choice = reply["choices"][0]
        message = choice["message"]
        text = f"{message.get('content') or ''} {message.get('reasoning_content') or ''}"
        return {
            "status": "ok",
            "seconds": time.perf_counter() - started,
            "needle": DEFAULT_NEEDLE in text,
            "finish": choice.get("finish_reason"),
            "chars": len(text),
        }
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:300]
        return {"status": f"http {error.code}: {detail}", "seconds": time.perf_counter() - started,
                "needle": False, "finish": "", "chars": 0}
    except Exception as error:  # noqa: BLE001 - the soak records any transport failure
        return {"status": f"error: {type(error).__name__}: {error}",
                "seconds": time.perf_counter() - started, "needle": False, "finish": "", "chars": 0}


class VramSampler:
    """1 Hz per-GPU used-memory samples, kept in memory and on disk."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.stop = threading.Event()
        self.samples: list[tuple[int, int]] = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _read(self) -> tuple[int, int]:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        ).stdout.split()
        return int(out[0]), int(out[1])

    def _run(self) -> None:
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sampled_at", "gpu0_mib", "gpu1_mib"])
            while not self.stop.is_set():
                try:
                    g0, g1 = self._read()
                    self.samples.append((g0, g1))
                    writer.writerow([now(), g0, g1])
                    handle.flush()
                except Exception:  # noqa: BLE001 - nvidia-smi can transiently fail
                    pass
                self.stop.wait(1.0)

    def start(self) -> None:
        self.thread.start()

    def halt(self) -> None:
        self.stop.set()
        self.thread.join(timeout=10)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=float, default=10.0)
    parser.add_argument("--port", type=int, default=18081)
    parser.add_argument("--max-context", type=int, default=32768)
    parser.add_argument("--model", type=Path, default=None,
                        help="artifact path; defaults to the Chinese-named file in --model-dir")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--exe", type=Path, default=EXE)
    parser.add_argument("--out", type=Path, default=ROOT / "_scratch" / "soak")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    prompt_dir = args.out / "prompts"
    prompt_dir.mkdir(exist_ok=True)
    prompts = {}
    for label, reps, estimate in SIZES:
        path = prompt_dir / f"p{reps}.json"
        if not path.exists():
            write_messages(path, build(reps))
        prompts[label] = (path, estimate)

    preflight()
    base = f"http://127.0.0.1:{args.port}"
    model = args.model or find_chinese_model(args.model_dir)
    print(f"model: {model}")

    sampler = VramSampler(args.out / "vram.csv")
    sampler.start()

    server_out = (args.out / "server.out.log").open("wb")
    server_err = (args.out / "server.err.log").open("wb")
    proc = subprocess.Popen(
        [
            str(args.exe), str(model),
            "--host", "127.0.0.1", "--port", str(args.port),
            "--max-context", str(args.max_context),
            "--kv-capacity", str(args.max_context),
            "--tp", "2", "--devices", "0,1",
            "--max-concurrency", "1",
        ],
        stdout=server_out,
        stderr=server_err,
    )
    if not bind_lifetime_to_parent(proc.pid):
        print("warning: server is not bound to this process; kill it manually if this run dies",
              flush=True)

    load_started = time.perf_counter()
    healthy = None
    while time.perf_counter() - load_started < 420:
        time.sleep(2)
        if proc.poll() is not None:
            break
        try:
            healthy = health(base)
            break
        except Exception:  # noqa: BLE001 - not listening yet
            pass
    load_seconds = time.perf_counter() - load_started
    if healthy is None:
        print(f"RESULT load_failed exit={proc.poll()}")
        sampler.halt()
        proc.kill()
        proc.wait(timeout=20)
        return

    print(f"health ok after {load_seconds:.1f} s; model={healthy.get('model')}")
    v0 = sampler.samples[-1] if sampler.samples else (0, 0)

    rows: list[list] = []
    cycles = 0
    ok = 0
    failed = 0

    def run(label: str, path: Path, estimate: int) -> None:
        nonlocal ok, failed
        result = chat(base, json.loads(path.read_text(encoding="utf-8")))
        good = result["status"] == "ok" and result["needle"]
        ok += 1 if good else 0
        failed += 0 if good else 1
        rows.append([now(), label, estimate, result["status"], round(result["seconds"] * 1000),
                     result["needle"], result["finish"], result["chars"]])
        print(f"[{datetime.now():%H:%M:%S}] {label:<5} {result['seconds']:7.1f} s "
              f"needle={result['needle']} {result['status'][:60]}", flush=True)

    # Short Chinese prompt: UTF-8 through HTTP, JSON, tokenizer and chat template.
    write_messages(prompt_dir / "zh.json", ZH_PROMPT)
    run("zh", prompt_dir / "zh.json", 30)

    deadline = time.monotonic() + args.minutes * 60
    while time.monotonic() < deadline and proc.poll() is None:
        for label, (path, estimate) in prompts.items():
            if time.monotonic() >= deadline or proc.poll() is not None:
                break
            run(label, path, estimate)
        cycles += 1
        if proc.poll() is None:
            try:
                if health(base).get("status") != "ok":
                    failed += 1
            except Exception:  # noqa: BLE001
                failed += 1

    # Concurrency 1: two simultaneous 8k requests, both must complete.
    results: list[dict] = []
    lock = threading.Lock()

    def parallel() -> None:
        result = chat(base, json.loads(prompts["8k"][0].read_text(encoding="utf-8")))
        with lock:
            results.append(result)

    threads = [threading.Thread(target=parallel) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=900)
    concurrent_ok = len(results) == 2 and all(r["status"] == "ok" and r["needle"] for r in results)

    v1 = sampler.samples[-1] if sampler.samples else (0, 0)
    alive = proc.poll() is None
    if alive:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=20)
    released = False
    for _ in range(60):
        if all(used < 1024 for used in gpu_used()):
            released = True
            break
        time.sleep(0.5)
    sampler.halt()
    server_out.close()
    server_err.close()

    with (args.out / "requests.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ts", "label", "estimate", "status", "latency_ms", "needle", "finish", "chars"])
        writer.writerows(rows)

    peak = max(sampler.samples, key=lambda s: s[0] + s[1]) if sampler.samples else (0, 0)
    g0 = [s[0] for s in sampler.samples] or [0]
    g1 = [s[1] for s in sampler.samples] or [0]
    print(f"RESULT alive={alive} cycles={cycles} ok={ok} fail={failed} concurrent_ok={concurrent_ok} "
          f"gpu_released={released}")
    print(f"VRAM after_load={v0} after_soak={v1} peak={peak} MiB")
    print(f"VRAM gpu0 min/max={min(g0)}/{max(g0)} gpu1 min/max={min(g1)}/{max(g1)} MiB")
    print(f"LOAD health_seconds={load_seconds:.1f}")


if __name__ == "__main__":
    main()
