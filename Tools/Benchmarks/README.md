# Benchmarks

These are the T0 spikes, kept so the numbers can be measured again. Two kinds:
- **Standalone scripts** (this folder): run with PythonCLI's bundled interpreter and a site-packages that has torch.
- **In Unreal**: `PyTorch.Benchmarks.*` automation tests, with scripts in `Content/Scripts/benchmarks/`.

```
<PythonCLI>/ThirdParty/python/Win64/python.exe -I -X utf8 bench_compile.py <site-packages> <cache-dir> [modes...]
<PythonCLI>/ThirdParty/python/Win64/python.exe -I -X utf8 bench_upload.py <site-packages>
<PythonCLI>/ThirdParty/python/Win64/python.exe -I -X utf8 bench_export.py <site-packages> <out-dir>
UnrealEditor-Cmd.exe <Project>.uproject -ExecCmds="Automation RunTests PyTorch.Benchmarks; Quit" -unattended -nullrhi -nopause
```

The Unreal tests log `BENCH ...` lines. They're in the Perf filter, so they don't run with the product tests.

## Results (2026-10-13)

**Setup**
- RTX 5090 (sm_120), driver 581.80 (CUDA 13.0); Windows 10.
- torch 2.14.1+cu132, triton-windows 3.8.0.post29, onnxruntime 1.31 (Python). Unreal's NNE ships ORT 1.24.
- UE 5.8, editor with `-nullrhi`. Results reach the game thread in the next frame, about 8 ms.

### (f) torch.compile with triton-windows

It works in PythonCLI's portable interpreter (`-I`, a site dir added at runtime). This was tested with no MSVC, no CUDA toolkit, and `PATH` reduced to `System32`.

**One fix is needed: `sysconfig`.** triton-windows finds its bundled TinyCC and CUDA headers through `sysconfig` platlib. Under `-I` that path points at the base interpreter, so without the fix:
- `Failed to find C compiler`;
- or, on machines with a CUDA 12.x toolkit, the wrong CUDA files.

PythonCLI 0.5's runtime makes platlib/purelib return the site dir, as a venv would.

Times are ms per call; the cold column is compile seconds on a cold cache.

| Model, batch | Eager | `default` (cold s) | `reduce-overhead` | `max-autotune-no-cudagraphs` (cold s) |
|---|---|---|---|---|
| CNN 224², 1 | 1.68 | 1.16 (13.0) | **0.30** (5.5×) | 1.22 (269) |
| CNN 224², 32 | 1.84 | 1.22 (8.7) | 0.95 | 1.54 (315) |
| MLP policy, 1 | 0.18 | 0.30 | 0.20 | 0.22 |
| MLP policy, 256 | 0.22 | 0.33 | 0.21 | 0.29 |
| MLP policy, 4096 | 0.20 | 0.32 | 0.20 | |

**Warm cache (a new process, caches in the project):** compile takes 1.3–3.0 s for the CNN and 0.15–0.35 s for the MLP. In that run eager was noisier (other GPU load) and `reduce-overhead` was 2.7× at batch 1.

**Decision:** compile is opt-in per model (`compile=True` or `compile="reduce-overhead"`), not on by default.
- Cold compiles take about 10 s per shape for a conv net.
- Small MLPs get slower: Python overhead (~0.2 ms) dominates.
- `reduce-overhead` is the mode for small-batch conv nets.
- `max-autotune` isn't worth minutes of tuning here.

Compiled models are shape-specialized, so the batcher pads to bucket sizes instead of recompiling for every batch size.

### (b) Shared memory → CUDA at 1080p

The frame is BGRA8, 8.3 MB.

| Path | ms | GB/s |
|---|---|---|
| `torch.from_numpy(shm).cuda()` (pageable) | 1.04 | 8.0 |
| shm → pinned staging (memcpy) → async H2D | 0.99 | 8.4 |
| **`cudaHostRegister` on the stream's shared memory → async H2D** | **0.63** | **13.3** |
| + BGRA→RGB, float, normalize on the GPU (staging / registered) | 2.32 / 1.66 | |
| 224² frame: pinned staging / pageable | 0.048 / 0.077 | |
| 28² frame: pinned staging / pageable | 0.030 / 0.103 | |

**Decision:** register each stream's shared memory with CUDA once, then upload each frame straight from its slot (no staging copy) and convert on the GPU. Small frames use a pinned staging buffer.

### (c) ONNX export → NNE (ORT CPU, ORT DirectML)

**Export:** dynamo exporter, opset 20, dynamic batch. It takes 1–5 s; the first export includes the exporter's import.

**Parity** with torch: max abs diff ≤ 3e-7 on every runtime.

**One DirectML fix:** the exporter writes `Reshape` with `allowzero=1`, which DirectML rejects. When the shape is a constant without a 0, `allowzero` changes nothing, so the export drops it.

Times are ms per `RunSync`; inputs and outputs are in CPU memory.

| Model | Batch | ORT CPU | ORT DML, dynamic batch | ORT DML, static batch | torch CUDA (eager, Python) |
|---|---|---|---|---|---|
| digits CNN 28² | 1 | 0.11 | 0.34 | | 0.62 |
| digits CNN 28² | 16 | 0.24 | 3.05 | | 0.71 |
| MLP policy | 1 | 0.014 | 0.27 | | 0.24 |
| MLP policy | 256 | 0.26 | 1.12 | **0.25** | 0.30 |
| vision CNN 224² | 1 | 1.69 | **0.62** | | 1.82 |
| vision CNN 224² | 8 | 11.2 | 9.98 | **1.51** | 1.96 |
| stylize 256² (image → image) | 1 | 20.4 | **1.60** | | 1.87 |

- **Model creation:** ORT CPU 2–270 ms (the first includes ORT's startup); DirectML 260–660 ms.
- With `-nullrhi`, DirectML runs on its own D3D12 device. The RDG interface (render graph, D3D12 RHI) wasn't measured.

**Decisions for shipping (T6):**
- DirectML is fast only with **static shapes**. With a dynamic batch it's fast at batch 1 only. So the NNE backend exports at a fixed batch size and pads, and the dynamic export is kept for ORT CPU.
- The runtime can't be "DirectML first" for everything: small MLPs are 10–20× faster on ORT CPU, and image models are 3–13× faster on DirectML. The default is **Auto**, which times both on load and keeps the faster.

### (a) Many callers, one model (from Unreal)

The model is the digits CNN on CUDA. N requests are made in one tick; the times are until the last reply reached the game thread.

| N | One call each | Python micro-batching (`ue.defer`) | **Stacked by Unreal into one call** |
|---|---|---|---|
| 1 | 6.6 ms | 15 ms | 7.3 ms |
| 16 | 15 ms | 15 ms | 7.3 ms |
| 64 | 56 ms (1,150/s) | 23 ms (2,750/s) | 7.0 ms (9,200/s) |
| 256 | 194–219 ms (1,170–1,320/s) | 61 ms (4,200/s, ~19 rows per pass) | **6.5 ms (39,400/s)** |

Per-call overhead dominates: framing, JSON, the worker-thread hop, and game-thread delivery. GPU time doesn't.

**Decision:** two levels of batching.
- The Torch subsystem **coalesces requests per model and method within a tick into one stacked call**: one frame each way, then splits the results.
- In Python, micro-batching with deferred replies merges calls from different callers and ticks.

### (d) Lockstep environment step (from Unreal)

The observations are (agents, 64) float32 through shared memory; the policy is a 64→256→256→8 MLP; one step per Lockstep frame.

| Policy | 1 agent | 64 | 256 | 1024 |
|---|---|---|---|---|
| none (transport only) | 0.33 ms (2,960 steps/s) | 0.34 | 0.34 (739k agent-steps/s) | 0.37 (2.7M) |
| CPU | 0.65 | 0.73 | 0.87 (287k) | 1.06 (959k) |
| CUDA | 0.72 | 0.78 | 0.75 (334k) | 0.76 (1.28M) |

**Decision:** Python in the loop is fast enough for RL training. At 256 agents with a CUDA policy that's 1,300 environment steps/s, and the game thread pays 0.75 ms per step.

### (e) GPU interop (D3D12 shared buffer → CUDA)

Not spiked. **Go, in T7 (Windows first).**
- Uploading from registered shared memory costs 0.63 ms at 1080p.
- Render-target readback is asynchronous (about 2 frames late).

So interop would save latency more than throughput, and T4's targets don't need it.
