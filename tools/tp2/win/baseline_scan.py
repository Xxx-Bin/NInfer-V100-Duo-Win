"""Phase 0 baseline scan for the tuxKOH/ninfer-V100X2 port assessment.

Runs the same long-context prompt through one artifact under several speculative
settings and records the numbers the CLI prints on stderr. It stops at the first
configuration that fails, so a leaked CUDA context cannot poison the rest.

Usage:
    python tools/tp2/win/baseline_scan.py --model F:/models/ninfer/qwen3_8_27b_nvfp4.ninfer \
        --reps 2800 --max-new 512 --drafts 0,3,5

Writes <out>/<tag>.json plus <out>/summary.csv. Read docs/notes/v100x2-port-assessment.md
for why these four columns matter.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

FILLER = (
    "The archive room holds shelf after shelf of routine maintenance notes, "
    "and none of them mentions anything unusual. "
)
NEEDLE = "918273"

METRICS = {
    "prompt tokens": "prompt_tokens",
    "generated tokens": "generated_tokens",
    "prefill speed": "prefill_tok_s",
    "decode speed": "decode_tok_s",
    "throughput (overall)": "overall_tok_s",
    "model elapsed": "model_seconds",
}


def build_prompt(reps: int) -> str:
    return (
        f"Reference note. The access code for the vault is {NEEDLE}.\n\n"
        f"{FILLER * reps}\n"
        "Question: what is the access code for the vault? "
        "Reply with the six digits only."
    )


def parse_metrics(stderr: str) -> dict:
    found = {}
    for label, key in METRICS.items():
        m = re.search(re.escape(label) + r":\s*([0-9.]+)", stderr)
        if m:
            found[key] = float(m.group(1))
    accepted = re.search(r"accepted\s+(\d+)\s*/\s*(\d+)", stderr)
    if accepted:
        found["drafts_accepted"] = int(accepted.group(1))
        found["drafts_total"] = int(accepted.group(2))
    return found


def run_case(exe: Path, model: Path, prompt_file: Path, out_dir: Path, tag: str,
             drafts: int, args) -> dict:
    cmd = [
        str(exe), str(model),
        "--messages", str(prompt_file),
        "--max-context", str(args.max_context),
        "--kv-capacity", str(args.max_context),
        "--prefill-chunk", str(args.prefill_chunk),
        "--max-new", str(args.max_new),
        "--kv-dtype", args.kv_dtype,
        "--tp", "2", "--devices", "0,1",
        "--no-thinking", "--greedy", "--ignore-eos",
    ]
    if drafts > 0:
        cmd += ["--spec", "mtp", "--draft-tokens", str(drafts), "--lm-head-draft"]
    if args.no_cuda_graph:
        cmd += ["--no-cuda-graph"]

    print(f"[{tag}] {' '.join(cmd)}", flush=True)
    started = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace")
    elapsed = time.time() - started
    (out_dir / f"{tag}.stderr.log").write_text(proc.stderr, encoding="utf-8")
    (out_dir / f"{tag}.stdout.log").write_text(proc.stdout, encoding="utf-8")

    record = {
        "tag": tag,
        "drafts": drafts,
        "exit_code": proc.returncode,
        "wall_seconds": round(elapsed, 3),
        "max_context": args.max_context,
        "prefill_chunk": args.prefill_chunk,
        "max_new": args.max_new,
    }
    record.update(parse_metrics(proc.stderr))
    (out_dir / f"{tag}.json").write_text(json.dumps(record, indent=2, ensure_ascii=False),
                                        encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False), flush=True)
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--exe", default="build-win-media/apps/ninfer.exe")
    parser.add_argument("--out", default="_scratch/baseline")
    parser.add_argument("--reps", type=int, default=2800)
    parser.add_argument("--drafts", default="0,3,5")
    parser.add_argument("--max-new", type=int, default=512)
    parser.add_argument("--max-context", type=int, default=86016)
    parser.add_argument("--prefill-chunk", type=int, default=4096)
    parser.add_argument("--kv-dtype", default="int8")
    parser.add_argument("--no-cuda-graph", action="store_true")
    args = parser.parse_args()

    exe = Path(args.exe).resolve()
    model = Path(args.model).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if not exe.exists():
        print(f"missing executable: {exe}", file=sys.stderr)
        return 2
    if not model.exists():
        print(f"missing model: {model}", file=sys.stderr)
        return 2

    prompt_file = out_dir / "prompt_85k.json"
    prompt_file.write_text(
        json.dumps([{"role": "user", "content": build_prompt(args.reps)}], ensure_ascii=False),
        encoding="utf-8",
    )

    rows = []
    for drafts in [int(x) for x in args.drafts.split(",") if x.strip()]:
        tag = f"mtp{drafts}"
        row = run_case(exe, model, prompt_file, out_dir, tag, drafts, args)
        rows.append(row)
        if row["exit_code"] != 0:
            print(f"[{tag}] failed with exit {row['exit_code']}; stopping", file=sys.stderr)
            break

    keys = ["tag", "drafts", "prompt_tokens", "generated_tokens", "prefill_tok_s",
            "decode_tok_s", "overall_tok_s", "wall_seconds", "exit_code"]
    lines = [",".join(keys)]
    for row in rows:
        lines.append(",".join(str(row.get(k, "")) for k in keys))
    (out_dir / "summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
