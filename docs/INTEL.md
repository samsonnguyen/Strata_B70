# Strata on an Intel Arc

Optional: `sycl/tools/Dockerfile.serve` builds a serving image (ahead-of-time for an A770, CPU baseline AVX2) and
`.github/workflows/sycl-image.yml` publishes it under the repository owner on pushes to the `a770-dg2` branch (manual runs take a `tag`).
`--build-arg BASE=<image>` replaces the Dockerfile's base image. Neither is needed to build or run the port.

Strata's engine is CUDA (and HIP for AMD). On an Intel Arc it runs as **Strata's own engine, ported to SYCL** (`sycl/`, the
section "The engine itself on Intel" below). It sits behind the same Strata server, so the OpenAI and Anthropic
APIs, streaming, tool calls, MCP and the web app are all unchanged. llama.cpp's SYCL backend is the comparison
point: it runs the same GGUF, several times slower.

**Every measured speed is in [INTEL_PERFORMANCE.md](INTEL_PERFORMANCE.md).** That covers this port, llama.cpp on
the same card, the history of each change, and results other people posted for the B50, the B580 and two B70s.
This file explains how the port works and how to run it.

Everything Intel-specific lives in `sycl/`. No shared file of upstream's is changed, so upstream merges stay clean.
`sycl/setup_intel.py` and `sycl/serve/server_intel.py` wrap upstream's `setup.py` and `serve/server.py` from
outside.

Written for and tested on an **Arc Pro B70 (32 GB)** running the Coder (IQ1_M), on Ubuntu 24.04 (in the dev image)
and on Fedora Silverblue 44 (in a Debian 13 distrobox); an Arc Pro B65 (32 GB) runs the same build.

## What you get

| | NVIDIA (Strata engine) | Intel Arc, Strata SYCL port | Intel Arc, llama.cpp |
|---|---|---|---|
| models | all four | the Coder IQ1_M; the original IQ2_XS and Swift 1.5 too (experts beyond VRAM in the host mirror) | the Coder (fits VRAM) |
| model files | the same GGUFs | the same GGUFs + a native pack | the same GGUFs |
| where the model lives | experts in RAM, hot ones on the card | every expert in VRAM (`--stream-experts`: no host copy); shard 2's lookup table read from the SSD by row | all of shard 1 on the card, shard 2 paged from disk |
| RAM needed | 32-64 GB | little (23 GB is fine) | little (23 GB is fine) |
| speculative decoding (MTP) | yes | yes (the base checkpoint's draft layer) | no |
| several cards | yes | layer split (`--layer-split`), tested by a user on 2x B70; `--peer-device` not yet | - |
| logprobs | - | `/v1/chat/completions` | - |
| images | yes | not yet | not yet |
| context | up to 262K | 256K (`--kv-resident`: the KV in pinned host memory, the attended window in VRAM) | 131K |
| speed | | [INTEL_PERFORMANCE.md](INTEL_PERFORMANCE.md) | [INTEL_PERFORMANCE.md](INTEL_PERFORMANCE.md) |

## Setup

On Ubuntu 24.04 or newer, from nothing. The Intel GPU driver (the `xe` or `i915` kernel driver and the compute runtime;
[INTEL_ARC.md](INTEL_ARC.md), "What you need") is assumed: `ls /dev/dri` shows a `renderD128`.

1. **Docker, once.** The engine runs in a container, so the PC needs Docker and no oneAPI:

        sudo apt install docker.io git
        sudo usermod -aG docker,render $USER

   Log out and in again (or reboot) so the groups apply. `docker ps` must work without `sudo`; the `render` group lets
   the container open the GPU.
2. **Get Strata and build the engine** ("How to build it" below, about 10 minutes the first time).
3. **Run setup:**

        ./setup.sh --backend sycl [setup.py's options, e.g. --model IQ3_S --context 32768 --port 8085]

   `setup.sh` makes its own Python environment in `.venv`, so Ubuntu's "externally managed" system Python is not
   touched. **Do not run `python3 sycl/setup_intel.py` yourself:** it installs setup's packages into whichever Python runs
   it, and Ubuntu's refuses (`externally-managed-environment`). `./setup.sh --backend sycl` is the one command, for the
   first install and every later start.

Setup is upstream's `setup.py` with the Intel steps swapped in (`sycl/setup_intel.py` imports it and replaces those steps;
setup.py itself is unchanged). The model choice, download, pack, tokenizer, MTP draft layer and the context and KV
questions are setup's own. What changes:

- **GPU check:** the Arc is found in sysfs (vendor 8086 under `xe` or `i915`) and offered through setup's AMD
  path. That is the path that builds locally and has no images.
- **Engine step:** it uses the SYCL build (`build-sycl-aot/strata` or `build-sycl/strata`, run in the `strata-sycl-dev`
  image by `sycl/serve/strata-sycl.sh`) instead of compiling CUDA or HIP.
- **RAM rule:** this does not apply on an `xe` card. The CUDA engine keeps every expert in RAM; the port streams them from
  the GGUF into VRAM (`--stream-experts`), so RAM only decides the KV streaming. `--check` lists what fits by VRAM.
- **The config:** it uses the container's paths and `"backend": "sycl"`. The VRAM reserve is 1,024 MiB up to
  32K and 2,048 MiB with 4,096-token prompt chunks above that, 300 MiB on a card under 12 GB; a
  `--vram-reserve-mib N` you give is kept as given. KV streaming (`--kv-resident 32768`) is on from
  64K up when the RAM holds the KV. A `model_switcher` or `sampling` block from an earlier config is kept.
  `run-<model>.sh` starts `sycl/serve/server_intel.py`.
- **Arc A-series (`i915`, e.g. the A750 with 8 GB):** a different config, see "Arc A750 and the other Alchemist cards" below.
- **Engine settings:** the config's `"env"` block reaches the engine. `strata-sycl.sh` forwards every variable starting with
  `STRATA_`, `ONEAPI_`, `UR_`, `IGC_`, `SYCL_` or `ZES_` into the container and sets `STRATA_VERIFY_DEVICE_PLAN=1`,
  `STRATA_VERIFY_NO_HOST=1` and `STRATA_STAGER_THREADS=12` unless the environment says otherwise (for example
  `"env": {"STRATA_VERIFY_NO_HOST": "0"}`).

If a future `setup.py` drops a step this relies on, it stops with a message instead of writing a wrong config.
The models, packs and checkout must sit under the folder `strata-sycl.sh` mounts at `/work` (the one above the
checkout, or `STRATA_SYCL_ROOT`). An earlier Strata install on the same PC keeps its data folder in
`~/.config/strata/settings.json`; pass `--data-dir <folder above the checkout>/Strata-data` to put this one under the
mount. A symlink to a folder outside the mount does not work in the container; a hard link does.

Then `run-<model>.sh` (or `./setup.sh --backend sycl` again) starts the model. The first start of a 30 GB model
takes about two minutes. `--port N` and `--host 0.0.0.0` work as in upstream's setup.

**Several cards.** The image pins `ONEAPI_DEVICE_SELECTOR=level_zero:0`. `sycl/serve/strata-sycl.sh` forwards the
host's value when it is set. Without one, a `--layer-split` defaults to `level_zero:gpu`, so the engine sees every
card.

## How to build it

The engine is built in Docker, in the `strata-sycl-dev` image (oneAPI 2026.1 compiler, oneMKL and Level Zero, from a
community llama.cpp SYCL image), so the PC needs no oneAPI. From the checkout (its folder is called `Strata` here; the
folder above it is what the container mounts at `/work`):

    cd Strata
    docker build -t strata-sycl-dev sycl/tools          # once: about 6 minutes, a 13 GB image
    H=$(basename "$PWD")

**Arc B-series** (Pro B70: `bmg-g31`; B580, B570 and Pro B60: `bmg-g21`; `ocloc ids bmg-g21` in the image lists the
device ids). Ahead-of-time code, no compile at the first start:

    docker run --rm -u $(id -u):$(id -g) -v "$PWD/..:/work" -e AOT=bmg-g31 -e REPO=/work/$H \
        -e BUILD_DIR=/work/$H/build-sycl-aot strata-sycl-dev "cd /work/$H && bash sycl/tools/build.sh strata"

**Arc A-series** (Alchemist: A750, A770, A580, A380): no `AOT=`. The code is compiled from SPIR-V at the first start
(JIT, about a minute). Nobody has run an AOT build for these cards.

    docker run --rm -u $(id -u):$(id -g) -v "$PWD/..:/work" -e REPO=/work/$H \
        -e BUILD_DIR=/work/$H/build-sycl strata-sycl-dev "cd /work/$H && bash sycl/tools/build.sh strata"

The build takes about 3 minutes and ends with `BUILD EXIT 0` and `errors: 0`. The binary is `build-sycl-aot/strata`
(B-series) or `build-sycl/strata` (A-series); setup finds either one. Run the `docker build` again after pulling a
Strata update that changes `sycl/tools/Dockerfile`, and the `docker run` after every update of the engine (setup
says when the engine is older than it needs).

## Things that matter on this GPU

The first three apply to running llama.cpp's server on the card; the thinking switch applies to both engines.

- **`SYCL_CACHE_PERSISTENT` must be 0 for llama.cpp.** Its persistent JIT cache segfaulted on Xe2 during the first
  compile. The llama.cpp start script sets it; if you run llama-server by hand, do too.
- **The whole model goes on the card** (`--n-gpu-layers 999`), except `per_layer_token_embd.weight`, the single
  28.8 GB tensor of shard 2.
  - `--override-tensor per_layer_token_embd=CPU` keeps that tensor in host memory, mmapped and paged from the SSD
    by row, which is how the model's authors serve it.
  - Host RSS stays around 2 GB.
- **Thinking.** The model reasons before it answers. Strata's web app has the setting; for an API client, either:
  - put `{"reasoning_effort": "none"}` in `strata-<model>.shared-settings.json` next to the config, or
  - send `chat_template_kwargs: {"enable_thinking": false}` per request.

  Without one of those, a short `max_tokens` is spent entirely inside the think block and the answer looks empty.
- **llama-server's `/health` says 503 while loading**; poll `/props` instead.

## How much context fits

This table is for llama.cpp. The architecture keeps the KV small: only every fourth layer is full attention (12 of
48). The other 36 are gated-delta-net layers, with a fixed-size recurrent state that does not grow with context.

- **What each token keeps:** the 12 attention layers keep 2 KV heads x 256 x (K + V) at q8_0, plus the
  sparse-attention indexer's keys.
- **Measured: about 20 KiB per token.** VRAM grows 0.6 GB per 32K of context with everything else unchanged.

| context | VRAM in use, model loaded | headroom on 32 GB | status |
|---|---|---|---|
| 32,768 | 28.4 GB | 3.5 GB | measured, the default |
| 65,536 | 29.0 GB | 2.9 GB | measured, loads |
| 98,304 | 29.6 GB | 2.3 GB | measured, loads |
| 131,072 | 30.3 GB | 1.5 GB | **measured, the practical ceiling**: a 104,798-token prompt read and answered |
| 163,840 | ~31.0 GB | ~0.9 GB | not attempted: under the 1.2 GB safety margin |
| 262,144 | ~32.6 GB | none | **does not fit - asking for it took the host down** |

**Do not ask for more than fits.**

- **What goes wrong:** on this driver, a GPU allocation past VRAM does not fail. The xe driver evicts buffers into
  host RAM, the kernel runs out of memory, and the machine livelocks until its hardware watchdog resets it.
- **It has happened:** a 262K request did exactly that on 2026-09-29. llama.cpp's own "failed to fit params" check
  fired too late to prevent it.
- **What to do:** compute the KV size first and leave 1.5 GB free.

## The engine itself on Intel: the SYCL port (`sycl/`)

This is Strata's own engine built for the Arc with oneAPI: the CUDA kernels (about 270, in 53 `.cu` files) and the
host code that drives them
(streams, events, graph capture, pinned memory). It is a migration of the tree, not a new backend: the engine has
no backend seam to slot into.

**Layout.** `sycl/src` and `sycl/include` mirror the tree and hold only the files the port changes. The SYCL build
(`sycl/CMakeLists.txt`) takes every other source from the original location, and no upstream file is edited.
`sycl/include/dpct/` is the vendored SYCLomatic helper library, so the port builds without the migration tool.

**How it was made, so it can be redone.**

1. **`sycl/tools/Dockerfile`:** the dev image. It is the llama.cpp SYCL image plus SYCLomatic (`dpct` 2025.3) and
   ninja. dpct parses CUDA, so it needs the CUDA 12.8 headers, not the toolkit: `sycl/tools/get-cuda-headers.sh`
   pulls them out of NVIDIA's pip wheels, and the migration mounts them at `/cuda-headers`. Building the port does
   not need them.
2. **`sycl/tools/migrate.sh`:** writes a compilation database for the 86 CUDA-touching translation units and runs
   dpct over them. 85 migrate; dpct reports no line it could not migrate, and about 1,400 advisory notes.
3. **`sycl/tools/fixups.py`:** what dpct got wrong or could not do, as an idempotent script with a reason per item.
   The ones that mattered:
   - CUDA's null stream means the default stream; dpct turned it into a null `sycl::queue*`. Every stream cast now
     goes through `strata::q_of()` (`sycl/include/strata/sycl_queue.hpp`).
   - `__ldg((const float*) p)` came out as `*p`, reading one byte of a float scale (two sites, s_gemv).
   - `__fadd_rn(a, b ? c : d)` lost its parentheses.
   - The ggml lookup tables were threaded through kernel parameters with the wrong table per template. They are
     plain `static const` arrays read from device code now, as ggml-sycl does.
   - A free-memory query written as an `if` with an initializer was dropped entirely. That one is the layer
     split's `stage_room()`, and it made every later card report 0 bytes free.
   - Smaller items:
     - helper headers renamed since dpct 2025.3 (`entangle`, `chunked_partition`);
     - graph introspection and `cudaGraphUpload`, which have no SYCL equivalents;
     - `%globaltimer` (the stage profiler reads zeros).
4. **`sycl/tools/build.sh`:** configure and build with icpx inside the image. It defaults to `$REPO/build-sycl-aot`;
   `BUILD_DIR` overrides the directory and `AOT` selects the target when configuring a new build. Two compiler flags are load-bearing:
   - `-fp-model=precise`: icpx defaults to a fast FP model.
   - `-cl-fp32-correctly-rounded-divide-sqrt` for the device compiler. The Arc's fp32 divide is not correctly
     rounded by default (OpenCL allows 2.5 ulp), and Strata's quantizers are byte-exact against ggml through
     `amax / 127`. Without the flag `quantize_act_parity` has 303k mismatches; with it, none.

**Parity tests.** The whole tree builds and links: the `strata` binary plus the kernel parity tests.

| parity test | result |
|---|---|
| bf16_gemv, cvec, dequant_s2, elementwise, gdn, gr, kv_q4, kv_q8, kv_stream, qsa, quantize_act, rope, router_top10, s2_gemv, s2_gemv_q8, s_gemv, s_gemv_q8k, sampler, shared_expert, iq_multi (IQ2_XS included) | pass |
| kv_hybrid_parity, qsa_prompt_attn_parity (int8 and fp16 XMX cases) | pass |
| qsa_prompt_attn_parity, Q4_0 tensor-core cases (#452) | "refused": the port does not have that mode |
| iq_parity, ple_parity, native_expert_parity | need fixtures or model files |
| s2_expert_grouped_parity | fails (the s2 path, unused here) |

**How to build it.** The checkout must sit inside a data root that holds the models too (the container mounts it
at `/work`; `REPO` is the checkout's path inside it):

```
docker build -t strata-sycl-dev -f sycl/tools/Dockerfile sycl/tools
docker run --rm -e AOT=bmg-g31 -e BUILD_DIR=/work/<checkout>/build-sycl-aot -e REPO=/work/<checkout> \
    -v <data root>:/work strata-sycl-dev "bash /work/<checkout>/sycl/tools/build.sh"
```

**Without Docker: a toolbox with oneAPI** (Fedora Silverblue, where tools live in a distrobox). Measured with a
Debian 13 distrobox, Intel's compute runtime 26.35 and oneAPI 2026.1.1:

- Install `intel-oneapi-compiler-dpcpp-cpp-2026.1` and `intel-oneapi-mkl-devel-2026.1` from Intel's apt repository
  (the dpct helpers are vendored, so SYCLomatic is not needed).
- **The Level Zero loader must be 1.21 or newer.** Debian 13's `libze1` is 1.20.6; with it the engine stops at its
  first KV allocation with `UR_RESULT_ERROR_UNSUPPORTED_FEATURE` (`layer.cpp`). The `libze1` deb from
  github.com/oneapi-src/level-zero releases (1.34.0, the Ubuntu 24.04 build) fixes it.
- Link the data root to `/work` inside the toolbox (`sudo ln -s <data root> /work`), so configs keep the image's
  paths, then build as above without the container:
  `AOT=bmg-g31 REPO=/work/<checkout> BUILD_DIR=/work/<checkout>/build-sycl-aot bash sycl/tools/build.sh strata`.
- Running: `STRATA_SYCL_RUNNER=native` makes `sycl/serve/strata-sycl.sh` start the engine in place (the server runs
  in the toolbox too); `STRATA_SYCL_RUNNER=distrobox:<name>` starts it in that distrobox from the host. benchy takes
  the same variable (`STRATA_SYCL_RUNNER=distrobox:<name> sycl/benchy.sh`), and pins `level_zero:0` as the image
  does unless `ONEAPI_DEVICE_SELECTOR` is set.

- `AOT` is the card's device target: `bmg-g31` for the B70 (what everything here was measured on) and the B65
  (the same G31 die). The B580 report
  in INTEL_PERFORMANCE.md used `bmg-g21`. `ocloc compile --help` in the image lists the targets (`-device`).
  Cards of different dies in one layer split need every die's code: a comma list, e.g. `AOT=bmg-g21,bmg-g31`.
- `JOBS` (default 12) caps the parallel compiles: 23 GB of RAM builds with `JOBS=8`, 61 GB with `JOBS=16`.
- `sycl/tools/build.sh <target>` builds one target (`strata`, a parity test, a bench).

**How to run it by hand.** This is a greedy test run, the way the engine numbers are measured. Run it inside the
`strata-sycl-dev` image, with the AOT build in `build-sycl-aot/`:

```
STRATA_VERIFY_DEVICE_PLAN=1 STRATA_VERIFY_NO_HOST=1 \
build-sycl-aot/strata --pack <iq pack> --native <shard1> --ple-gguf <shard2> \
    --expert-profile data/expert-profile-coder.bin --expert-cache auto --stream-experts \
    --prefill auto --spec 4 --spec-min-p 0.5 --mtp <mtp rt dir> --max-context 8192 --tokens <ids> --max-new 64 --greedy
```

- **`--stream-experts`** (this port): no resident host copy of the experts. Every one of the 12,288 goes from the
  GGUF into the VRAM cache through a small staging ring (`GgufExpertSource`). Upstream needs 32 GB of RAM for this
  model; with the flag the engine runs on 23 GiB.
- **`STRATA_VERIFY_DEVICE_PLAN=1`:** the GPU plans each layer itself (upstream's E-6, off by default there).
- **`STRATA_VERIFY_NO_HOST=1`** (this port): the host waits for the whole window graph instead of per-layer
  rings. Only valid with every expert resident or in the pinned host mirror.
- **`--mtp`:** the base Qwen3.8-Flash-Next checkpoint's MTP draft layer (`tools/mtp_fetch.py fetch`,
  `mtp_pack.py --experts q2_0`, `mtp_rt.py`; 4.9 GB downloaded, about 860 MiB of VRAM). It drafts for the Coder
  fine-tune with the same greedy tokens. The suffix drafter alone is rarely accepted on this card, which makes the
  draft layer the lever for decode.
- **IDs over 128 KB:** 80K-token ids exceed Linux's 128 KB single-argument limit, so use `--tokens-file`.

AOT device code is what runs ("How to build it" above). Without AOT, the runtime JIT-compiles every kernel on
first use, which is slow the first time a process runs. `SYCL_CACHE_PERSISTENT=1 SYCL_CACHE_DIR=<dir>` kept the
port's JIT result across runs on the B70 (the segfault above was llama.cpp's); the AOT build needs neither.

**What the port had to get right beyond compiling.** Each item is an entry in `sycl/tools/fixups.py` or a flag.

- **Host-mapped flags.** Strata's decode is a GPU/CPU handshake through host-mapped memory.
  - `volatile` device loads do not bypass the caches on Intel. System-scope atomics do, and they are the only thing
    measured to work (`sycl/probe/doorbell.cpp`, six variants).
  - Host-to-device visibility *during* a kernel stays unreliable on this platform. That is why the all-resident
    path waits for the window instead.
- **Bounded spins.** A device spin that never sees its flag is not a hang of one process.
  - The xe driver times the queue out and resets the GT node by node; a window graph has 2,400 nodes.
  - The card then stays wedged until a reboot (twice).
  - Every spin is capped (`kSpinMax`).
- **32-lane sub-groups.** The kernels are written for warps. dpct pinned 134 of 289 launches; the rest would run
  at Xe2's default 16 (`-fsycl-default-sub-group-size=32`).
- **Synchronous copies.** `cudaMemcpy` blocks, but dpct's default-queue `memcpy` did not wait. With the streaming
  ring, that filled the expert cache from overwritten buffers: non-deterministic residuals, NaN by layer 5. It was
  found with the per-layer residual ladder (`STRATA_VERIFY_DEBUG=1` prints R after every layer).
- **No host thread waits on a queue's event that another queue's barrier uses.**
  - Under the Level Zero v2 adapter, such an event no longer releases the other barrier, and the GPU waits forever.
  - The stager and the PLE upload mark their copies with a sequence number the copy queue writes into page-locked
    memory. The host polls that number instead.
  - See "Prompt-slot borrowing: the hang" below.
- **No non-char type punning in device code.** A `uint16_t*` read through an `int2` is not honoured by the SYCL
  device compiler; extract with shifts (see "Two model-specific bugs").
- **`native_expert_parity` was hand-ported.** The GPU native expert kernel matches ggml's float reference on real
  IQ1_M rows (rel 1.1e-2, the same class as the CPU path).

**Profiling.**

- `sycl/benchy.sh` (benchy v1) runs the standard bench (`sycl/tools/perf_matrix.py`): every model x the v1 prompt
  sizes, with each model's serve config, from a cold page cache. Its report is what INTEL_PERFORMANCE.md asks
  submitters to post.
- `sycl/tools/Dockerfile.unitrace` builds the dev image with Intel's unitrace (`strata-sycl-dev:unitrace`). Run
  `unitrace -d` around the engine, then `sycl/tools/rank_kernels.py <log>`, for device time per kernel.
- Hardware counters need Intel's metrics libraries in the image and `dev.xe.observation_paranoid=0`; that image is
  not in the repo.
- `sycl/probe/` holds the small standalone programs behind the platform findings: `doorbell.cpp` (host<->device
  flags), `bw.cpp` (read bandwidth), `hostread.cpp`, `graphbench.cpp`, `nodecost.cpp`, `xmx.cpp`. Build each with
  `icpx -fsycl` in the dev image.
- `mmvq_bench`, `mmvq_sg_bench`, `q6k_align_bench`, `xmx_gemm_bench`, `xmx_int8_bench`, `sel_scores_bench` (QSA block
  scores), `attn_bench` (the prompt attention and its variants) and `native_expert_parity NATIVE_BENCH=1` time
  kernels in isolation. Warm the clocks first: a 5 ms run measures the
  ramp, not the kernel.
- `STRATA_PLE_TRACE=1` traces each PLE gather.
- `STRATA_DBG_NAN=1` reports the first non-finite values per layer, including the experts' fp16 GEMM inputs.

**Two traps worth knowing.**

- **`--prefill-until N` with a native pack** does not feed the rest of the prompt through the token loop (that
  loop is skipped for native packs). The tokens after N are dropped and the model free-runs. Compare output tokens
  between paths, never only timings.
- **Identical greedy runs can decode at two speeds.** One PLE read stall lands either in the prompt's PLE wait or
  in the first decode round. It is a once-per-process cost, not lost throughput, so compare runs on time to first
  token plus decode.

### How the prompt path uses VRAM

- **The VRAM plan.** `--expert-cache auto` fills the card down to `--vram-reserve-mib` *before* two things exist:
  the KV state and the prompt chunk buffers. With too small a reserve at long contexts, the driver starts
  migrating buffers and the run never finishes. Setup's reserves (1,024 MiB to 32K; 2,048 MiB with
  `--prefill 4096` above) leave room.
- **Streamed experts.** Experts the cache does not hold are copied for every chunk, in one of two walks:
  - **Stream-all:** every non-resident expert, layer by layer ahead of the compute. This is the default for chunks of
    1,024 tokens or more (`STRATA_PREFILL_STREAM_MIN`) when the VRAM holds more than 90% of the (layer, expert) pairs.
  - **Routed-only:** only the experts the chunk routes to (`STRATA_PREFILL_RING=8`). This is the default otherwise:
    for smaller chunks, and past 90%, where stream-all would copy several times the routed experts.

  `STRATA_PREFILL_STREAM_ALL=1` / `=0` force a walk. The stager threads read the blobs themselves; an early version
  held `GgufExpertSource::blob()` pointers across reads of the ring, and those blobs were overwritten before they
  were copied.
- **Prompt-slot borrowing.**
  - **Without it:** the reserve evicts experts from VRAM for good, and decode after a long prompt is slow.
  - **With it:** the prompt path borrows cache slots for its buffers and refills them in about a second afterwards.
  - **The cost:** a refill after every prompt. So the port borrows by default only above a 32K context;
    `--prefill-borrow` / `--no-prefill-borrow` decide explicitly.
  - **The lend mirror.** With `--stream-experts`, the experts in the slots a prompt may lend (the cache's last ones)
    are also kept in pinned host memory, read once at start. The prompt path DMAs them for each chunk and the
    refill copies them from RAM, instead of reading the GGUF both times. The same bytes go into the same slots, so
    outputs are identical. It costs about 2 GB of RAM and about 2 s at start (taken only beyond 8 GiB of free RAM);
    `STRATA_LEND_MIRROR=0` turns it off.
- **KV streaming** (`--kv-resident`) keeps the whole KV in pinned host memory and only the attended window in VRAM,
  so the KV pushes no experts out. Setup turns it on from 64K up and keeps INT8, the faster KV format at every size
  measured.
  - `--kv k8v4` does not support streaming yet. Setup keeps its KV in VRAM, which pushes more experts out to the
    host mirror.
  - k8v4 reads prompts fastest, because it needs no streaming copies.
- **The pinned host mirror.** At start-up, every expert without a VRAM slot is read into pinned host memory
  (`STRATA_MIRROR_MIB`; by default free RAM less 4 GiB). The device-built verify plan points the expert kernels
  straight at it over PCIe. That is how a model bigger than VRAM runs: the original IQ2_XS keeps about a quarter of
  its experts there.

### The prompt path's QSA kernels

Two kernels of the prompt path differ from the CUDA build's on this card. Both are FP32 in another summation order,
so outputs are not bitwise those of the older kernels (neither is the CUDA build's tensor-core path): a greedy
continuation parts at a near-tie after tens of tokens, with the same text. Decode keeps the older kernels.

- **QSA block scores as GEMM tiles.** Each batch of 256 queries' indexer heads is multiplied against the pooled block
  keys with oneMKL (fp32, tiles of 8,192 blocks, 32 MB of scratch), and a small kernel sums the per-head relus and
  scores the incomplete tail block. The older kernel scored every (query, block) pair with one sub-group: its cost
  grows with the square of the context. `STRATA_SELECT_GEMM=0`: the older kernel.
- **Attention scores one work-item per cell.** The batched attention scores each 128-cell chunk with one work-item
  per cell holding its key row and all 12 heads' dot products, the query heads read from local memory. The older
  kernel split each cell over a sub-group and added the parts with 60 shuffles a cell. The values pass and the
  merge are unchanged; all four KV formats. `STRATA_ATTN_PERCELL=0`: the older kernel.
- **What limits attention now** (`attn_bench`): fetching the selected K/V rows alone takes a fifth of the kernel's
  time (neighbouring queries share cells, so they come from cache). The rest is arithmetic on the vector units,
  most of it the values pass.

The attempts behind these, and the ones that did not help, are in [sycl/TODO.md](../sycl/TODO.md).

### XMX (Intel's matrix engine)

oneMKL's FP16 GEMMs already run on the XMX units, so the prompt path is bound by the dequant that feeds them, not by
the products. Every hand-written joint_matrix kernel so far is correct but loses to the existing paths on this card
(numbers in [INTEL_PERFORMANCE.md](INTEL_PERFORMANCE.md), "XMX experiments").

- **`xmx_gemm_iq`:** a fused dequant + FP16 GEMM straight from the quantized rows.
- **`qsa_prompt_attn_xmx` v1 and v2:** the port of the mma.sync prompt attention, opt-in
  (`STRATA_PROMPT_ATTN_XMX=1` for 64-cell chunks with 120 KB of local memory, or `=32`).
  - 120 KB of local memory leaves room for one work-group per core, and each product runs twice (fp16 hi + lo).
  - Only 12 of the 16 matrix rows are real heads.
  - A lean fp16 version of the values pass (2026-10-04, ~50 KB of local memory) lost too. The 16-lane sub-group it
    needs costs little, so the joint_matrix path itself is what loses here.
  - Grouping neighbouring positions' cells was not built: their selections overlap little, so it would multiply the
    arithmetic, and the K/V fetch is not the bottleneck anyway.
- **Expert dot products on int8 DPAS for decode:** three versions were tried and are not in the tree. At 1-6 rows,
  the grid decode and the packed-B layout cost more than the DPAS saves, and one version hung the GPU.
- **An int8 DPAS GEMM straight from IQ4_NL for prompts** (`xmx_int8_bench`, standalone).
  - One 32-element block per DPAS, rescaled by d_x * d_w after each.
  - The per-block rescale keeps the matrix engine waiting.
  - The dequant it would save is small at expert size.
  - Not ported.

### Serving the port

`serve/server.py --engine strata` runs the SYCL engine unchanged through `sycl/serve/strata-sycl.sh`. That script
is an `exe` that starts the binary inside the oneAPI runtime image, with the serve pipes attached. Paths in the
config's `args` are the container's, with the data root mounted at `/work`. A config:

```json
{"engine": "strata", "exe": "<repo>/sycl/serve/strata-sycl.sh",
 "args": ["--pack", "/work/pack", "--native", "<shard 1>", "--ple-gguf", "<shard 2>",
          "--expert-profile", "data/expert-profile-coder.bin", "--expert-cache", "auto", "--stream-experts",
          "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5", "--mtp", "/work/mtp/rt",
          "--max-context", "32768", "--kv", "int8", "--vram-reserve-mib", "1024"],
 "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "repetition_penalty": 1.05}, ...}
```

**The reserve matters.** With `--stream-experts` there is no host copy of the experts beyond the pinned mirror. Any
expert left out of both is read from the SSD and computed on the CPU whenever it is routed, and decode collapses.
1,024 MiB fits all 12,288 Coder experts at 32K.

**Start time.** Starting a model is mostly the expert cache's fill from the GGUF. The fill is a pipeline:

- the slots are admitted in profile order first (the same placement);
- the reads go in file order, by up to 8 threads, into page-locked batches of 64;
- a batch's copies run while the next batch is read.

`STRATA_FILL_SERIAL=1` is the old fill: three slices read per expert, then a copy, then a wait.

**Logprobs.** `/v1/chat/completions` takes OpenAI's `"logprobs": true` and `"top_logprobs": 0..20`, streamed or
not, through `sycl/serve/server_intel.py`.

- **The reply** carries `choices[0].logprobs.content[]` (`token`, `logprob`, `bytes`, `top_logprobs`). With thinking
  on, the thinking tokens go under `reasoning_content`.
- **The engine** (the port only) gets `logprobs=K` on its GEN line. It writes an `LP logprob id:logprob ...` line
  after each `T` line, from the verify window's head logits. That is the distribution the token was taken from,
  before temperature, penalties and sampling.
- **Compatibility:** upstream's server ignores the lines, and an engine that is never asked writes none.
- **Cost:** each token costs one 1 MB logits row copy when asked.
- **Not yet:** `/v1/completions` and the Anthropic endpoint.

**The Monitor tab on Intel.** `sycl/serve/server_intel.py` is `serve/server.py` with two additions made at run
time:

- **GPU readings.** When NVML has no card, it plugs `sycl/serve/xe_telemetry.py` into `serve/telemetry.py`'s
  `gpu_reader`.
  - From sysfs: temperature, power and its cap, and the PCIe link the card trained at.
  - Load and VRAM come from `/run/gpustat.json`. xe reports them per client in `/proc/*/fdinfo`, which only root
    can read, so a small root sampler writes them (`sycl/tools/gpustat.py`; install it with its `gpustat.service`
    as its header says).
- **A Model menu.** A run config may name a `model_switcher`, an RPC taking `{"mode": m}`, for a host that swaps
  models on one card. The web app's header then gets a Model menu (`sycl/serve/web/switcher.js`, injected into the
  page).

**The engine exits directly** once its requests are done. It used to abort at exit in serve mode: a queue wait in a
destructor ran after the runtime's teardown began.

### Status of the planned work

1. **Expert dot products on XMX in integer mode** for decode: parked (see XMX above). Today it is scalar dp4a,
   ALU-bound.
2. **Experts missing from VRAM read from pinned host memory over PCIe instead of the SSD:** done (the host mirror).
3. **KV streaming from 64K up:** done.
4. **QSA block selection:** done another way - oneMKL fp32 GEMM tiles (above), not XMX.
5. **The hot decode kernels re-tuned for Xe2's native 16-wide sub-groups:** tried, no gain in the engine
   (`STRATA_MMVQ_SG`: 16 all, 1 IQ4_XS only, 2 short outputs only; default 32). The real headroom was misaligned
   loads, since fixed:
   - A Q6_K block is 210 bytes, so every block's `ql`/`qh` runs start only 2-byte aligned. The B70 splits a
     misaligned 16-byte load into pieces.
   - The wide Q6_K kernel now does two aligned 16-byte loads and a shift (`load16_a2`; both stay inside the block),
     and takes every Q6_K shape and window width. `STRATA_MMVQ_A2=0` restores the old path.
   - The same treatment followed for Q4_K, Q5_K, IQ4_XS, Q8_0 and IQ4_NL ("Decode round 2" in
     INTEL_PERFORMANCE.md). The aligned-load helper only loads its second chunk when the address is unaligned, so it
     never reads a 16-byte chunk without a needed byte and cannot cross a page at the end of an allocation.
   - Switches back to the old paths:

     | switch | restores |
     |---|---|
     | `STRATA_MMVQ_WIDE_32=0` | the old Q8_0/IQ4_NL kernels |
     | `STRATA_GR_DOWN_SLICED=0` | the direct GR down kernel |
     | `STRATA_PLAN_PARALLEL=0` | the serial resident plan |
     | `STRATA_GR_DOWN_DIRECT=0` | GR down staging its activations in local memory |
     | `STRATA_MMVQ_WIDE_K=0` | the old K-quant kernels |

   - Some of these change the summation order. A long greedy continuation can then flip at a near-tie.
6. **INT8 prompt GEMMs:** experts dequantized to INT8, run as oneMKL/oneDNN INT8 on XMX. Open. A fused int8 kernel
   was tried (`xmx_int8_bench`) and lost.

The current speedup list, with what was measured for each, is [sycl/TODO.md](../sycl/TODO.md).
7. **Fewer graph nodes per decode round** (~2,500): norm+rope, scores+top-k, gate+quantize fused. Open.
8. **Wider speculation** (two draft branches per verify window): the kernels are latency-bound, so it is nearly
   free. Open.
9. **A model bigger than VRAM:** done.
   - The original Qwen3.8-Flash-Next IQ2_XS runs on the B70 with 23 GB of RAM.
     - Shard 2 is byte-identical to the Coder's, so it is a hard link.
     - A native pack (`tools/iq_pack.py`).
     - The original model's expert profile and the same MTP draft layer.
   - Swift 1.5 runs the same way.
   - IQ3_XXS/IQ3_S (43-50 GB of experts) would need 19-26 GB mirrored, more than 23 GB of RAM allows on one card.
     Two cards hold them (see INTEL_PERFORMANCE.md, "Other people's cards").

## Keeping up with upstream

A merge of upstream `main` into `b70` leaves the copies in `sycl/` behind wherever upstream touched a file they
mirror. They are refreshed by re-migration, not by hand:

1. **Merge upstream into `b70`.** No shared file should conflict: the Intel code is all in `sycl/`. Then check that
   `sycl/setup_intel.py --check` and `sycl/serve/server_intel.py --engine mock` still run against the new setup.py
   and server.py.
2. **Migrate both trees.** The *old* upstream tree (a `git archive` of the pre-merge commit) goes through
   `migrate.sh` into `sycl-base`, and the merged tree into `sycl-new`. Each takes about 3 minutes in the dev image.
3. **Normalize both:** `sycl/tools/normalize.sh <dir> <commit>`. It covers dpct's file names, the unchanged files
   and the message serials, then runs `fixups.py`. Two runs of dpct on the same source now differ only in the
   kernel name hashes it generates.
4. **Merge:** `sycl/tools/merge_upstream.py BASE_OUT NEW_OUT OLD_REV NEW_REV` 3-way merges only the files upstream
   changed.
   - It canonicalizes dpct's kernel-name hashes to the port's first, and resolves hash-only hunks.
   - Four files take upstream's diff by hand (the script's `HAND` list): `verify.cpp`, `mtp.cpp` and
     `ple_reader.cpp`, which dpct never produced (they include no CUDA header directly), and
     `native_expert_parity.cpp`, which was hand-ported.
   - New parity tests are copied in and listed in `sycl/CMakeLists.txt`.
   - `git merge-file --diff-algorithm=histogram` aligns big restructures better.
   - To port open upstream PRs ahead of upstream: migrate main plus the PRs, and merge into only the files they
     touch. When upstream later merges them, that "main + PRs" output is the merge base.
5. **Fix what the fixups missed.** A fixup whose pattern upstream changed shows up as a compile error; extend the
   fixup, re-run it, rebuild.
6. **Audit symbols.** Count the port's feature identifiers before and after. It has caught a dropped mirror hook and
   doorbell waits that had lost their spin bound.
7. **Check outputs.** Compare greedy output tokens against the previous build: Coder 20 / 2,185-token prompts,
   IQ2_XS, and a 40K prompt.

What each merge needed:

- **0.1.25-0.1.27 (2026-09-30):** the first re-migration. The draft layer's prompt pass became upstream's batched
  one.
- **0.1.31 (2026-10-01, 132 commits):** 45 migrated files changed for real, 38 conflicts.
  - **The expert kernels.** Upstream sends the i-quant experts to new multi kernels (one warp per row, 8 rows a
    group). The port's grid is sized for its own kernels (8 lanes a row, 32 rows a group).
    - With upstream's kernels on the port's grid, only a quarter of each expert's rows were written, and the output
      was end-of-text tokens.
    - The port's kernels stay the default; they also serve upstream's new Q4_K/Q5_K/Q5_1/Q8_0 experts.
    - `STRATA_EXPERT_SPLIT=1` runs upstream's kernels with their grid; `STRATA_GR_V3=1` runs upstream's GR read.
      Both are slower here and stay opt-in.
  - **Lanes per row** of the port's expert kernels are tunable at run time (`STRATA_GU_LANES` /
    `STRATA_DOWN_LANES`, 4/8/16/32). The default (8 / 8) stays.
  - **Three dpct mistranslations from the first migration,** found by upstream's new tests. dpct writes
    `__fadd_rn(a, b)` as `a + b` without parentheses, so `__fadd_rn(sum, c ? x : y)` became `sum + c ? x : y`.
    The three places:
    - the native QSA indexer's per-token pooled key: blocks completed while generating got wrong pooled keys;
    - the native QSA scores, an opt-in path;
    - the s2 activation rounding: every value was 0, in the Q2_0 s2 pack, which is unused here.

    `fixups.py` parenthesises now.
- **0.1.32 (2026-10-01, 89 commits):** hash-only differences are canonicalized before the merge, which left 10 real
  conflicts.
  - **Async commit:** upstream's `set_commit_async` / `wait_commit` maps onto the port's deferred commit
    (`commit(n, err, false)` + `commit_finish`).
  - **New hyper-connection read variants:** upstream's are not used. The port keeps its sliced down / split norm
    read, renamed `gr_norm_split_port_kernel`.
  - **The read-variant self-test** segfaulted on the B70, so on SYCL it runs only with `STRATA_HC_CHECK=1`.
  - **Kernel names:** two collided after hash canonicalization (renamed).
  - **fused_gr's** per-block shared-memory query is a fixup now.
- **0.1.33 (2026-10-01, 17 commits):** two conflicts.
  - `--resident-cpu-experts` (upstream's new `resident_cpu_explicit`) beside the port's `--stream-experts`.
  - The prompt attention's compute-capability check: the port keeps its XMX dispatch.
- **0.1.35 + seven open PRs (2026-10-02):** ported ahead of upstream.
  - **#374 needed two port fixes:**
    - Its next-chunk read assumed every chunk is the full chunk length. The port's first chunk is 256 tokens, so the
      second chunk's rows were read from the wrong place.
    - Its host wait on the PLE upload's event deadlocked past ~4K tokens (the Level Zero v2 bug above).
  - **#413's gate** is an NVIDIA SM-count rule, and its parity test an NVIDIA bench (not built).
  - **The PRs:**

    | PR | what | in the port |
    |---|---|---|
    | #385 (sergqwer) | stager: a buffer's first job of a generation waits for the previous DMA from it | race fix |
    | #463 (constantindjonkam) | decode waits for an adaptive swap before reading the residency table | determinism fix |
    | #453 (architectds) | the drafter's batched K/V on the KV-streaming ring | on |
    | #374 (sergqwer) | the first chunk's PLE rows read beside layer 0 | on, with two port fixes |
    | #363 (BlueKingMuch) | the PCIe expert call: group stride, fused SwiGLU + q8_1 | re-done on the port's lane kernels |
    | #407 (sergqwer) | `--adapt-tuned` | opt-in |
    | #413 (BlueKingMuch) | DeltaNet recurrence per key head | opt-in (`STRATA_GDN_KEYHEAD=1`): slower here |
- **0.1.38 (2026-10-03, 83 commits):** upstream merged the seven PRs, so their header forks in `sycl/include` are
  gone.
  - **New in the port with it:**
    - the DeltaNet output norm without its dead FP32 store;
    - two prompt-path fixes: the fused layout's buffers, and the streamed walk's resident lookup;
    - `STRATA_GR_DOWN_MAX4=1`, opt-in and neutral here.
  - **Stubbed:**
    - `--peer-device`, a second GPU as an expert-cache tier. Upstream's code is CUDA calls; `open` refuses with a
      message.
    - The fused int8 prompt kernels (#136), part of the MMQ library the port does not build.
  - **Off on SYCL:** the sm_90 thread-block-cluster greedy sampler and QSA top-k. They have no SYCL counterpart,
    and the callers take the plain kernels.

## Bugs worth remembering

**Prompt-slot borrowing: the hang (fixed 2026-10-01).** From 0.1.31 on, a prompt that borrowed cache slots stopped
in its first full chunk, with the GPU at 100% and the host waiting on the compute queue.

- **Not a borrowing bug.** Borrowing is just the only way the Coder streams experts during a prompt (every expert is
  resident otherwise), and the streamed path's stager deadlocked.
- **How the stager deadlocked:**
  - Its threads read experts into a ring of 16 page-locked buffers.
  - To reuse a buffer, a thread waited on the copy queue's event for the DMA that last read it. That is the CUDA
    form, `cudaEventSynchronize`.
  - Under the Level Zero v2 adapter, once a host thread had waited on that event, it no longer released the other
    queue's barrier that also listed it. The GPU waited forever.
  - It only happened once a layer streamed more than 16 experts, when the ring wrapped.
- **What ran fine:** the v1 adapter (`SYCL_UR_USE_LEVEL_ZERO_V2=0`), a 256-buffer ring, and a sync after every phase
  (`STRATA_PREFILL_SYNC=1`, kept as a debug switch).
- **The fix:** the copy queue writes a sequence number into page-locked host memory after each DMA, and the stager
  polls it. Output is bit-identical to the run without borrowing.

**Two model-specific bugs found with Swift 1.5 (fixed 2026-10-01).** UkisAI's Swift 1.5 IQ2_XS (setup's `swift`
family, the same GSQ-RCO layout) decoded token 0 forever: NaN logits from layer 13 on.

- **`--stream-experts` read the wrong shard.**
  - Since upstream 0.1.31, `ExpertLayout::gguf_file` is per layer AND role (`3 * layer + role`), because a shard
    boundary can fall inside a layer.
  - The port's `GgufExpertSource` still indexed it by layer. So a layer in shard 2 was read from shard 1 at
    shard 2's offsets: garbage IQ1_M scales, then infinities after the fp16 dequantization.
  - The original model keeps every expert in shard 1 and never noticed; Swift's layers 13-47 are in shard 2.
- **IQ2_XS products were garbage.**
  - ggml's `vec_dot_iq2_xs` reads its four 16-bit codes through a `uint16_t*` to an `int2`. The SYCL device
    compiler does not honour that type punning.
  - The dequantizer, which reads the codes directly, was exact.
  - The codes are extracted by shifts now (the dot and the expert kernels' `Split<17>`).
  - `iq_multi_parity` checks against a CPU decode from ggml's tables too, and passes.

**The layer split's free memory (fixed 2026-10-03).** A layer split across two B70s crashed at the second card's
expert cache ("no room").

- Upstream reads free memory in an `if` with an initializer, and dpct dropped the call itself. So every later stage
  saw 0 bytes free.
- The call is restored, and `tools/fixups.py` re-applies it after a re-migration.
- Found and tested on 2x B70 by tmking01 in the upstream PR review.

### Long prompts decode to token 0 with a borrowed cache over 4 GiB

Found and fixed on the A770: oneMKL's bf16/f16 GEMM returns zeros for inputs 4 GiB or more into an allocation, which broke prompt
borrowing with a cache over 4 GiB. The write-up, the fix (a reversed cache layout) and the reproducer are in
[INTEL_A770.md](INTEL_A770.md#long-prompts-decode-to-token-0-with-a-borrowed-cache-over-4-gib-root-cause-found-2026-10-04).

## Arc A750 and the other Alchemist cards (`i915`, 2026-10-07)

Measured on an Arc A750 (8 GB, `i915`, PCIe 4.0) with the Flash-Next IQ3_XXS in a PC with 64 GB of RAM, a Ryzen 5 5600X
(AVX2, no AVX-512). `./setup.sh --backend sycl` writes a different config for an `i915` card (and a 300 MiB VRAM reserve and
`--draft-vocab en` for any card under 12 GB):

| | Arc Pro B70 (`xe`, 32 GB) | Arc A750 (`i915`, 8 GB) |
|---|---|---|
| experts | `--stream-experts`: from the GGUF into VRAM, the rest in a pinned RAM mirror | no `--stream-experts`: all of them in a RAM arena (39.97 GiB for IQ3_XXS), 406 of 24,576 in VRAM, the CPU computes the others |
| `STRATA_VERIFY_NO_HOST` | 1 (the wrapper's default) | 0, written to the config's `"env"` |
| `--vram-reserve-mib` | 1024 / 2048 | 300 |
| `--ple-io` | `ram` (rotational-disk rule of setup) | `ram` |
| FP64 | native | emulated: `IGC_EnableDPEmulation=1`, `OverrideDefaultFP64Settings=1`, `NEOReadDebugKeys=1` in `"env"` |
| build | AOT `bmg-g31` | JIT (no `AOT=`) |

- **Why no `--stream-experts`.** The mirror is one pinned host allocation for everything the card does not hold (40 GB here). A
  single `sycl::malloc_host` above a few GB fails on the A750 (`STRATA_MIRROR_MIB` 3,000 worked, 8,192 and 24,000 did not, also
  with `memlock` unlimited and the relaxed-allocation-limit variables), and the engine then refuses to start with
  `24170 experts are neither in VRAM nor mirrored`. Without the flag the engine loads the GGUF's experts into an ordinary RAM
  arena and the CPU computes the experts that are not in the 0.67 GiB cache. That needs the RAM: about 43 GB for IQ3_XXS plus
  the 29 GB n-gram table with `--ple-io ram` (the table is not locked in the container, so it is reclaimable; do not give the
  container `--ulimit memlock=-1`, the locked table and the arena then do not fit in 64 GB and the PC swaps). Setup warns when the RAM is less than the model's.
- **Why `STRATA_VERIFY_NO_HOST=0`.** With it set the GPU plans each layer itself and the CPU's experts are never asked for:
  the window hangs (the i915 log says `Fence expiration time out`) or, if the pool is refused, the engine says
  `REFUSED: ... neither in VRAM nor mirrored`. It is the switch of the B70's case, where every expert the card lacks is in the mirror.
  The engine takes a value of `0` (or empty) as "off", and `strata-sycl.sh` does not pass such a value on.
- **FP64.** An Alchemist has no FP64 hardware. One kernel on the sampled path declares `double`, and the SYCL runtime refuses it at
  its first launch (`'double' is not supported in 'Intel(R) Arc(TM) A750 Graphics' device`, the engine dies on the first request
  that has a temperature). With the three variables above the driver emulates it; the cost is the sampler's tail
  (see "FP64." in the B70 section).
- **The first request after a start used to return `!!!!!` (token 0) or crash the engine.** The GPU waits for the CPU's experts at
  every layer in a spin that is bounded (`strata::kSpinMax`, `sycl/include/strata/sycl_doorbell.hpp`): a spin that never ends
  hangs the driver. The bound was 20,000 reads, a few tens of milliseconds, which is enough when nothing is waited for (the B70)
  and is not for a CPU layer in the first request (cold pages, 13 s for a 26-token prompt). The GPU gave up, went on with the
  experts' outputs missing, the next layers' routing was garbage (one expert chosen ten times for a token: `nt=10`) and the
  CPU pool wrote past its token arrays and died with a segmentation fault (exit code 139); when it did not die the logits were NaN
  and the sampler's answer was token 0. Later requests ran with warm pages and were right. The bound is now a build option,
  `STRATA_SYCL_SPIN_MAX` (CMake; `SPIN_MAX=` for `sycl/tools/build.sh`): 2,000,000 reads (a few seconds) unless the build is an
  AOT build for a `bmg` card, which keeps 20,000. Before the change 11 of 16 runs of the engine (three requests each, a 26-token prompt) died or gave
  token 0 in the first request; with 2,000,000 reads 4 of 4 were right in all three requests.
- **Speed.** About 10 to 15 tok/s decode (76 to 92% of the drafts accepted), a 26-token first prompt in 13 s and later short prompts in 0.2 to
  1.3 s; the A750's PCIe link probes at 10.6 GB/s.
- **A750 and the xe error counters.** The engine segfaults in a worker thread of the CPU pool when it exits (dmesg only, the server
  has already printed "stopped"); it is not a GPU event.

## Not done

- **Images,** on both Intel engines. Strata's vision path encodes with `strata-vision` into embeddings the CUDA
  engine reads; neither the SYCL port nor the llama.cpp config wires it yet.
- Speculative decoding on the llama.cpp path (the GGUF carries no draft layer llama.cpp can use). The SYCL
  port has it (MTP draft layer, `--mtp`).

## Arc A770 (DG2, 16 GB)

The port also runs on an Arc A770 (an older, 16 GB Alchemist card on the `xe` driver). What differs from the B70, how it was brought up,
the measurements and the bugs found are in [INTEL_A770.md](INTEL_A770.md); the driver, runtime and compiler issues are catalogued in
[INTEL_A770_ISSUES.md](INTEL_A770_ISSUES.md).

## Arc Pro B70 with a model that does not fit: Flash-Next IQ3_S (2026-10-07, branch f-141-intel)

The earlier B70 rows are for models whose experts all fit in VRAM. IQ3_S has 24,576 experts of 1.97 MB (46 GB): the 32 GB
card holds 11.5k of them (22 GiB of cache) and the rest sit in a pinned host mirror that the GPU reads over PCIe. Tested on
the B70 VM (xe driver, kernel 7.0, oneAPI 2026.1, PCIe measured at 13.3 GB/s host to device), engine 0.1.40-sycl.

**Two things made it produce garbage, both fixed or documented here.**

- **Aliased pages in big allocations (fixed in the engine).** The xe driver returned the 22 GiB expert-cache arena with two
  of its 2 MiB pages mapped onto the same memory: a write at +2 MiB showed at +1022 MiB. Cache slots there held other
  experts' bytes, and the prompt came out NaN at the first layer that routed to one (layer 4, expert 249); the same run
  with another `--max-context`, or with an MTP window of 4 instead of 6, was clean. `STRATA_VERIFY_ALL_SLOTS=1`
  (`STRATA_VERIFY_FIND=1`, `STRATA_VERIFY_DUMP=dir`) reads every filled slot back and compares it with the GGUF; that is how
  the aliasing was found (the wrong bytes were slot 0's blob). Every allocation of 32 MiB or more now goes through
  `strata::malloc_device_guarded`: it tags every 64 KiB, reads the tags back in a second kernel, and allocates again behind a
  growing spacer until the range is clean (a warning names each retry; the check costs under 0.5 s at startup;
  `STRATA_ARENA_ALIAS_CHECK=0` skips it). `sycl/probe/arena_rw.cpp` and `alias_scan.cpp` do not reproduce it on their own: it
  needs the engine's allocation history.
- **`STRATA_VERIFY_NO_HOST=1` is required whenever part of the experts is not in VRAM.** `sycl/serve/strata-sycl.sh` sets
  it; a by-hand run must too. Without it the per-layer host/GPU handshake is not visible across the bus on xe and the logits
  turn NaN a few tokens into the answer, differently on every run. With it (and the mirror covering every miss) the run is
  deterministic: two runs gave identical tokens.

**Numbers** (greedy, `--spec 4 --spec-min-p 0.5 --mtp`, INT8 KV, 32K context, `--stream-experts --vram-reserve-mib 1024
--ple-io ram`, 11.5k experts resident; the 10-prompt `mg_norepeat` gate passes 10 of 10, the xe error counters did not move):

| | before | after |
|---|---|---|
| 4,095-token prompt, `--prefill auto` (2,048-token chunks) | 618 tok/s | 730 |
| 4,095-token prompt, `--prefill 4096` | 784 | **980-1,002** |
| 8,169-token prompt: auto / 4096 / 8192 lending cache slots / 8192 own buffers | 676 | 953 / 1,037 / 1,117 (9.4k slots) |
| decode, 200-token story (66% of drafts accepted, 1.98 tokens per round) | 31.4 tok/s | 30.5 with 4096-token chunks (700 fewer cache slots) |
| decode, 200-token code answer (90% accepted, 3.65 tokens per round) | not measured | 40.8 with 4096-token chunks |

(Interleaved A/B pairs, medians of 3-5. "Before" is the same build with the old short first chunk; the decode figures are
the old default and `--prefill 4096`.)

**Why the prompt is slow here, and the two changes.** Every prompt chunk streams the experts the cache does not hold over
PCIe, and a 4K prompt touches nearly all 512 experts of every layer: 13.5k missing experts x 1.97 MB = 26 GB, 2 s of the
bus per chunk. So the prompt reads faster in fewer, bigger chunks, and the short first chunk (`STRATA_PREFILL_FIRST`, 256
tokens, there to start the GPU while the PLE rows are read) cost a whole extra pass of that stream. It is now 256 only when
the cache holds every expert, else 0. `setup_intel.py` writes `--prefill 4096` for a card with 24 GB or more (the chunk's
buffers take ~700 cache slots; `--prefill 8192 --prefill-borrow` lends cache slots instead and keeps all of them for decode,
at ~1 s of refill per prompt). Where the 4K prompt's GPU time goes now (GPU timeline, timing on): expert dequant 20%, the two
expert GEMMs 34%, QSA attention 14%, host grouping 6.5%, GDN 9%, hyper-connection reads 4.6%.

**Draft length** (200 tokens, medians of 5 interleaved pairs, `--spec-min-p 0.5`): `--spec 6` against `--spec 4` is +6.5% on
code (43.4 against 40.8 tok/s, 84% accepted against 90%) and -1.5% on prose (29.9 against 30.4). A 2-run sweep of the rest
(`--spec 2/3/5`, min-p 0.3/0.7) was within the run-to-run noise of ~1 tok/s; the setup default (4, 0.5) stays.

**FP64.** Arc Alchemist emulates it (`IGC_EnableDPEmulation=1`); on the A750 one row of the sampler's top_p/temperature tail
costs 451 us in double and 20 us in float (`sycl/probe/fp64_cost.cpp`), and a sampled decode ran at 7.6 tok/s against 12 for
a greedy one. On the B70 the same kernels cost 3.4 against 2.1 us, so FP64 is not a B70 problem. Of the kernels that use
double, the native-pack path (the packs setup builds) runs only the sampler (a 16-token greedy run created 173 kernels, the
double ones among them: `sample_tokens` only; the native router, combine, gate and norm kernels are float). The sampler's tail
now accumulates in `strata::samp_acc_t`, float by default (`-DSTRATA_SYCL_SAMPLER_FP64=1` keeps double): A750 sampled decode
7.62 -> 8.10 tok/s (medians of 5 interleaved pairs, runs 7.1-9.2), `sampler_parity` 0 failures on both cards, and the B70 gate's
sampled cases picked the same tokens.

**XMX.** oneMKL's FP16 GEMMs already run on the matrix engines; per expert the dequant (42 us for a gate/up matrix) costs
twice the GEMM (20 us), and the fused dequant+XMX kernel (`xmx_gemm_bench`) is 0.22x of the pair on this card. The speed left
on this card for this model is the dequant kernels (at ~200 GB/s against a 600 GB/s card), the host grouping, and the QSA
prompt attention (14%); the multi-column int8 DPAS decode kernels of llama.cpp PR 29864 target K-quant and Q8_0 weights, not
the IQ-quant experts, and were not ported.

## Measured, 2026-10-01, Arc Pro B70, Coder IQ1_M, 32K context, INT8 KV: the SYCL port (engine 0.1.31-sycl)

| | |
|---|---|
| decode | 78.2 tok/s on a 19-token prompt, 75.7 after a 2,184-token prompt (256 greedy tokens, MTP + suffix drafts) |
| prompt | 780 tok/s on 2,184 tokens; 934 tok/s at 128K, 784 at 256K |
| through the API | 48-72 tok/s on a ~190-token chat answer, prompt included (the first request after a start is the slower one) |
| VRAM | all 12,288 experts resident, ~1.9 GB free with everything loaded (`--vram-reserve-mib 1024`) |
| RAM | no host copy of the experts (`--stream-experts`) |

## Measured, 2026-09-29, Arc Pro B70, Coder IQ1_M, 32K context, q8_0 KV: llama.cpp

| | |
|---|---|
| load | ~100 s |
| VRAM | 28.4 of 32 GB |
| decode | 23-25 tok/s; GPU 92% busy at 165 W, CPU idle |
| prefill | 149 tok/s on a 2,701-token prompt; 424 tok/s on a 104,798-token prompt at 131k context |
| quality | correct code on every test; matches the NVIDIA path token for token in spirit, not measured |
