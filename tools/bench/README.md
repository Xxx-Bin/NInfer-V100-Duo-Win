# tools/bench

Offline helper for the `ninfer_bench` throughput tool. Correctness/parity tooling lives separately
under [`tools/parity`](../parity).

The upstream `tools/bench/ttft/` streaming TTFT client is **not ported to this fork**, and the serve
benchmarks here request `stream: false`. Every TTFT number produced locally is the server-side phase
sum `server_ttft_ms` (`prepare + vision + prefill`), which excludes ingress queueing and HTTP
transport. Do not compare it with upstream streaming TTFT figures.

## Corpus baker

`ninfer_bench` benchmarks prefill at an exact length by slicing the first `P` token ids of a
committed corpus, so the corpus must be real, in-distribution text (not random tokens) and at
least as long as the largest prefill you want to run. `make_bench_corpus.py` bakes that corpus
offline with a local Hugging Face Qwen3.6 tokenizer.

Outputs (committed):

```text
bench/fixtures/bench_corpus.ids            whitespace-separated decimal token ids (exactly --tokens)
bench/fixtures/bench_corpus.manifest.json  tokenizer id, token count, and source description
```

Content sources:

- Built-in curated multi-domain prose (Chinese / English / code / math) — the default. It is
  encoded WITHOUT the chat template or special tokens, then tiled (paragraphs rotated each cycle)
  and truncated to exactly `--tokens`. Repetition only fills length; because prefill/decode
  throughput is token-count / bandwidth bound, it does not bias the numbers.
- `--source-text <file>` (repeatable) — tokenize your own long meaningful text instead, e.g. a
  downloaded public-domain book or a concatenated document set, for genuinely diverse very long
  content. The committed default is `~64k` tokens; raise `--tokens` and/or pass `--source-text`
  for more.

The binary slices `[0:P]`; the manifest is provenance only.

## Requirements

Install the tokenizer dependencies into the active Python environment:

```bash
pip install -r tools/bench/requirements.txt
```

The tokenizer is loaded locally only; the tool never downloads from the network. Pass
`--tokenizer-path` or set `NINFER_TOKENIZER_PATH`.

## Regenerate / check

```bash
# Regenerate the committed corpus from the built-in bank (default 65536 tokens).
python3 tools/bench/make_bench_corpus.py \
  --tokenizer-path /path/to/local/Qwen3.6-27B/tokenizer \
  --tokens 65536

# Bake from your own downloaded/assembled text instead (kept local; not committed).
python3 tools/bench/make_bench_corpus.py \
  --tokenizer-path /path/to/local/Qwen3.6-27B/tokenizer \
  --tokens 131072 --source-text /path/to/book.txt

# Check that the committed .ids and its descriptive manifest agree; no tokenizer or source needed.
python3 tools/bench/make_bench_corpus.py --check
```

`--tokens` is the exact committed corpus size and the ceiling on prefill length; increase it (and
optionally use `--source-text`) to benchmark longer prefills, memory permitting.

## NInfer performance matrix

`run_ninfer_bench_matrix.py` runs the layered public-Engine `ninfer_bench` matrix against the native
`.ninfer` artifact and stores its local reports under `profiles/bench/`. Its defaults are:

```text
artifact: out/qwen3_6_27b.ninfer
binary:   build/bench/ninfer_bench
corpus:   bench/fixtures/bench_corpus.ids
```

The matrix treats MTP `k=3` with the optimized proposal head as the primary path, keeps `k=0` and
`k=5` as controls, and sweeps `k=0..5` on representative context-decode cases. Decode-bearing cases
cover CUDA Graph and eager execution; prefill-only cases vary prompt length and prefill chunk.

```bash
# Configure the benchmark targets once; they are off in the default public build.
cmake -S . -B build -DNINFER_BUILD_BENCHMARKS=ON

# Inspect commands without running the model.
python3 tools/bench/run_ninfer_bench_matrix.py --preset core --dry-run

# Main run. Builds build/bench/ninfer_bench first, then writes JSON and summary.csv.
python3 tools/bench/run_ninfer_bench_matrix.py --preset core

# Longer run that adds 32k/64k prompt and context-decode points.
python3 tools/bench/run_ninfer_bench_matrix.py --preset full

# Run only the MTP draft-window sweep.
python3 tools/bench/run_ninfer_bench_matrix.py --preset full --suite mtp_sweep
```

Default outputs:

```text
profiles/bench/ninfer-<preset>-<timestamp>/
  commands.sh
  manifest.json
  json/<suite>/<case>.json
  logs/<suite>.<case>.stderr.txt
  summary.csv
  summary.json
```

Use `--resume` to skip completed JSON reports in an existing `--output-dir`, and `--preset smoke`
for a minimal script/runner check. `--no-build` uses the binary supplied by `--bench` without
building it.

Each raw report must be `ninfer_bench_report` schema v12. The flattened summary and schema-v3 matrix
manifest carry native names from the report: selected target, canonical `weights_id`, artifact,
load/read/upload/staging values, Engine memory arenas including request transient and CUDA Graph
allowance, per-test planned logical and allocator-observed workspace peaks, KV capacity and
payload, configured proposal head and graph mode, phase timings and throughput, and speculative
rounds/drafts/acceptance/fallbacks. The matrix manifest is descriptive and records the commands and
selected local inputs; it does not make repository state part of report validity.

`run_serve_corpus.py` runs both registered targets and both published MTP0/MTP3 suites when both
artifacts are supplied. Pass one `--artifact` to select a single target and `--mode mtp0` or
`--mode mtp3` to run only that suite. The 35B-A3B-only `--mode dflash7` route runs the same
decode corpus with DFlash block=8 (`k=7`) and the optimized proposal head. Add
`--sampling greedy` to force exact argmax while retaining the same fixtures and repetition count.
Its schema-v5 result and flattened summaries retain the canonical `weights_id` received from the
schema-v9 serving startup record. The stochastic route pins its complete
temperature/top-p/top-k/min-p/presence/frequency profile explicitly, so model-default changes do
not alter the measurement method.

## Concurrent serving benchmark

`run_serve_concurrency.py` measures two separate concurrency properties through real loopback
Chat Completions requests:

- `decode-saturation` submits one long-decode wave and uses only complete one-second intervals in
  which every decode round has exactly the configured batch size. Ramp-up, prefill, and drain
  intervals are excluded.
- `corpus-makespan` shuffles the existing mode-specific corpus once with the fixed seed `20260811`,
  then runs that same order with exactly `N` persistent client workers. A worker submits the next
  request only after its current response completes, and makespan ends when the final response has
  been read. Request bodies are sent in shuffled-order sequence while response waits remain fully
  concurrent, removing client-thread arrival races without serializing inference.

Each concurrency point starts a fresh server because its execution graphs and memory plan are
startup-fixed. Prefix reuse is disabled, startup and warmup are outside both measurements, and the
runner writes per-point JSON, raw serving JSONL, and combined JSON/CSV/Markdown summaries.
The point report records the shuffle seed, dispatch method, shuffled position, and canonical corpus
position for every request.

```bash
python3 tools/bench/run_serve_concurrency.py \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ninfer \
  --mode mtp3 --suite decode-saturation \
  --concurrency 1 --concurrency 2 --concurrency 4 \
  --decode-tokens 8192 \
  --output profiles/bench/concurrent-decode

python3 tools/bench/run_serve_concurrency.py \
  --artifact qwen3_6_27b=out/qwen3_6_27b_nvfp4.ninfer \
  --mode mtp3 --suite corpus-makespan \
  --concurrency 1 --concurrency 2 \
  --output profiles/bench/concurrent-corpus
```

Use `--kv-capacity auto` when the fixed corpus needs more shared KV than the default 262,144-token
pool. A point is intentionally not resumable: combining fragments from separate server processes
would not preserve either a steady interval or one continuous makespan.

## Peak prefill / decode sweep

`run_peak_sweep.py` answers "how fast can this configuration go", which the panel and the periodic
throughput log cannot: those divide a monotonically increasing counter by a one-second interval,
and a prefill counter only moves when a whole chunk lands, so a sampled second can only report one
chunk divided by its length. This tool reads `request_done.timings_seconds` and divides the
tokens that phase actually evaluated by that phase's own wall time.

It starts one server per prefill chunk (execution graphs and the memory plan are startup-fixed),
disables prefix reuse, sends `--reps` requests per point with a fresh nonce so no prompt can hit
the cache, and reports the maximum over repetitions per `(chunk, suite, prompt length)`.

```bash
python tools/bench/run_peak_sweep.py \
  --artifact F:/models/ninfer/qwen3_8_27b_nvfp4.ninfer \
  --serve build-win/apps/ninfer-serve.exe \
  --suite both --chunks 1024,2048 --reps 3 \
  --prefill-tokens 2048,4096,8192,16384,32768 \
  --decode-contexts 512,8192,32768 --decode-tokens 512 \
  --output profiles/bench/peak-27b-nvfp4
```

Outputs `points.json` (every repetition), `peak.csv` (best per point), `summary.json` (the full
configuration plus the headline peaks), and one server stdout/stderr/request-log triplet per chunk.

The point of a peak number is that it is only comparable to another run with the same
`tp / devices / kv-dtype / chunk / spec / draft window / greedy` setting; the summary records all
of them. Repetition count is a peak-vs-sustain choice: a V100 that runs for minutes drops clocks,
so `--reps 3` reports the peak while a large `--reps` reports something closer to sustainable.

### Decode depends on the draft acceptance rate

A decode point is only meaningful together with its acceptance rate, so the tool measures both and
the summary records `accepted_tokens / drafted_tokens` per point. The default prompt asks for a
long ordered enumeration, which the model can predict almost perfectly: acceptance is 100% and the
reported rate is an upper bound, not a production figure. Pass `--prompt-style prose --no-greedy
--temperature 0.7` to sample original prose instead, which is the pessimistic end of real traffic
(36% acceptance at a three-token draft window).

Accepted drafts per round drive the rate, so the draft window has to be measured rather than
assumed: at 100% acceptance `--draft-tokens 5` beats 3 by 24%, while on sampled prose it loses 6%
because the fourth and fifth drafts are rejected almost always and still cost a verify slot. The
window optimum follows the per-position acceptance rate, which the request log records as
`speculative.accepted_per_position`: sampled prose accepts 67/30/12% at positions 1-3, so its best
window is 2, while structured output (`--prompt-style json`) accepts 100/94/89% and gains from 3-5.

Before launching anything it refuses to run when the port is taken, a target GPU has less than
`--min-free-mib` free, or available host memory is below `--min-free-ram-mib` — an artifact that
does not fit aborts the server with a bare fail-fast code and no diagnostic. `--force` overrides.

`--kv-capacity` defaults to `--max-context`, which is what a peak run wants. Do not pass `auto`:
it sizes the pool to the card rather than to the window (2,880 of 5,760 page groups stay live at a
184,320-token window), and the extra pages are what makes an otherwise loadable long-context
profile abort once an MTP proposal head is resident too.
