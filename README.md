# PyTorch-Unreal
PyTorch ML support via [PythonCLI](https://github.com/getnamo/PythonCLI-Unreal) plugin for the Unreal Engine.

PyTorch runs **out of process** in an embedded Python (CPython 3.12) with a per-project venv. A crash in a native library never takes the editor down, and the GIL never touches the game thread.

- **Models:** each model is loaded once and shared by every caller.
- **Batching:** requests made in the same tick run as one batched forward pass on the GPU.
- **Hot reload:** a reloaded script keeps the model's weights.

Status: **in development** (UE 5.8, Win64; Linux later). Plan: `Notes/PyTorchUnreal-Plan.md` in the development project.

| | |
|---|---|
| Python | `import unreal_torch as ut`: `@ut.model` classes with `@ut.export` methods, typed inputs (`ut.Image`, `ut.Tensor`, `ut.Audio`) |
| Unreal | a **Torch Model** asset (script + model name) and the **Run Model** node; `UTorchSubsystem` in C++ |
| Under it | [PythonCLI](https://github.com/getnamo/PythonCLI-Unreal) 0.5+ (services, deferred replies, jobs, shared-memory streams) on [CLISystem](https://github.com/getnamo/CLISystem-Unreal) 1.4+ |

## Setup

1. Put `PyTorch-Unreal`, [PythonCLI-Unreal](https://github.com/getnamo/PythonCLI-Unreal) and [CLISystem-Unreal](https://github.com/getnamo/CLISystem-Unreal) in your project's `Plugins/`. PyTorch also enables Unreal's NNERuntimeORT plugin.
2. Start the editor. PythonCLI installs the profile (Project Settings > Plugins > PyTorch) into `Intermediate/PythonCLI/venv` in the background.
   - The first time, the CUDA build of torch is about 3 GB.
   - Later installs come from uv's cache.

## Quick start

`Content/Scripts/digits.py` (or use the bundled `examples/digits.py`):

```python
import torch.nn as nn
import unreal_torch as ut

@ut.model("digits")                                    # the asset's Model Name
class Digits(ut.Model):
    def setup(self):                                   # once, on the model's device (CUDA if available)
        self.net = DigitNet()
        self.load_weights(missing_ok=True)             # Saved/Torch/Models/digits/digits.safetensors

    @ut.export(batch=True)                             # concurrent requests: one forward pass
    def classify(self, image: ut.Image(size=(28, 28), gray=True)) -> dict:
        p = self.net(image).softmax(-1)                # image: float (N, 1, 28, 28) on the device
        return {"digit": p.argmax(-1), "confidence": p.amax(-1)}
```

In Unreal:
1. Create a **Torch Model** asset (Miscellaneous > Data Asset > Torch Model). Set Script to `digits.py` and Model Name to `digits`.
   - **Refresh Signature** lists its methods and inputs.
2. In a Blueprint, call **Run Model** (Model, Method `classify`, Inputs `{"image": tensor}`) and read the result in **On Success**:
   - **Get Output Int** (`digit`);
   - **Get Output Float** (`confidence`).
3. Make the inputs with **Make Image Tensor From Texture / Render Target**, **Make Tensor From Floats** or **Make Tensor From Vector**. Non-tensor inputs go in **Json Inputs** (`{"top_k": 3}`).

## How requests run

- **One instance per process:** every Run Model call goes to the same model. Models stay loaded across PIE sessions; Unload Model frees them.
- **Stacking:** requests made in the same tick, for the same model and method with inputs of the same shape, are stacked on dim 0. They are sent as one call at the end of the frame, and the results are split back per request.
  - 64 Blueprint callers in one tick → one forward pass of 64.
  - The results arrive in the next frame: about 7 ms on an RTX 5090.
- **Micro-batching:** in Python, a `batch=True` method also merges calls that arrive within `max_wait_ms` (default 1 ms) of each other.
- **Unbatched methods** (`@ut.export` without `batch=True`) get one request at a time, without a batch dimension.
- **Typed inputs:**

  | Annotation | What the method gets |
  |---|---|
  | `torch.Tensor` | a tensor on the model's device, keeping its dtype |
  | `ut.Tensor(dtype=..., shape=(None, 64))` | checked per row (`None` is any size) and cast |
  | `ut.Image(size=(H, W), gray=False, normalize="unit"\|"symmetric"\|"imagenet", layout="NCHW")` | any image Unreal sends: (H, W, C) uint8 RGBA/BGRA or float, or (C, H, W). It's converted on the device: channel order, resize (antialiased), gray, normalization. |
  | `ut.Audio(rate=16000, mono=True)` | float audio resampled to `rate` |
  | `int`, `float`, `str`, `bool`, `dict`, `list` | JSON values (Json Inputs), shared by a stacked batch |

- **Outputs:**
  - A tensor is the output `output`.
  - A dict gives named outputs; tuples give `output0`, `output1`, ….
  - Tensors become `FPyTensor`s; numbers and strings go to the result's JSON.
  - With `batch=True`, outputs whose first dimension is the batch are split per request.
- **Runtime defaults:**
  - `torch.inference_mode()`, TF32 matmuls, and a CUDA stream per model.
  - Optional autocast (`precision="bf16"` / `"fp16"`).
  - Optional `torch.compile` (`compile=True` or `"reduce-overhead"`). triton-windows is installed with the CUDA profiles, and no MSVC or CUDA toolkit is needed. Batches are padded to power-of-two sizes so a compiled model compiles a few times rather than once per batch size.
- **Hot reload:** **Reload Script** (or editing and reloading) re-runs the script. Its models keep their weights wherever the shapes still match; `@ut.model(reload="fresh")` opts out.
- **Weights:** `load_weights()` / `save_weights()` use safetensors (`.pt` on request).
  - Lookup order: `Saved/Torch/Models/<model>/`, then `Content/Models/`, then `hf://org/repo/file` with huggingface_hub.
  - `publish(module_or_state)` swaps new weights into the live model.

## Settings (Project Settings > Plugins > PyTorch)

| Setting | |
|---|---|
| Profile | **CUDA (Auto)**: the build matching the driver, plus triton-windows on Windows. Also CUDA 13.0, CUDA 13.2, CPU, ROCm (Linux), XPU, or None (bring your own). torch is pinned to 2.14.* |
| Extras | torchvision (on), torchaudio, ONNX export (on), Gymnasium + PettingZoo, TensorBoard, Hugging Face Hub |
| Packaged Profile | **CPU** (CPU torch replaces the editor's CUDA build in packaged builds that ship Python) or Same As Editor |
| Compile Cache Directory | `Intermediate/Torch/CompileCache`: Inductor and Triton caches, so compiled kernels survive restarts |

The Torch Model asset overrides the script's Device, Precision and Compile settings. It also sets Load (On First Use, On Startup, Manual) and Warmup.

## C++

```cpp
UTorchSubsystem* Torch = UTorchSubsystem::Get();
Torch->Run(DigitsModel, TEXT("classify"), { { TEXT("image"), ImageTensor } }, TEXT(""),
    [](const FTorchResult& R) { if (R.bSuccess) { const int32 Digit = UTorchBlueprintLibrary::GetOutputInt(R, TEXT("digit")); } });
```

**Options** (`FTorchRunOptions`):
- `bBatched`: inputs already have a batch dimension.
- `Layouts`: e.g. `{"image": "BGRA"}` for raw render-target pixels.
- `bImmediate`: send now, outside the tick's stack.

`LoadModel`, `UnloadModel`, `ReloadScript`, `FetchSignature` and `GetStats` cover the rest.

## Tests

- **Engine-free** (unreal_torch over PythonCLI's protocol, 21 checks):
  `<PythonCLI>/ThirdParty/python/Win64/python.exe -I Content/Scripts/tests/torch_harness.py --site <venv>/Lib/site-packages`
- **In Unreal** (`PyTorch.Model.*`, `PyTorch.Settings.*`):
  `UnrealEditor-Cmd.exe <Project>.uproject -ExecCmds="Automation RunTests PyTorch; Quit" -unattended -nullrhi -nopause`
- **Benchmarks** (`PyTorch.Benchmarks.*`, the T0 measurements): see [Tools/Benchmarks](Tools/Benchmarks/README.md).

## Progress notes

- **T0, spikes:** done. Measurements and decisions are in [Tools/Benchmarks](Tools/Benchmarks/README.md).
- **T1, PythonCLI 0.5** (services, deferred replies, jobs, image helpers): done.
- **T2, plugin core:** done.
  - Settings and profiles, the Torch Model asset, `UTorchSubsystem` with per-tick stacking, Run Model / Load Model.
  - unreal_torch: models, typed inputs, batching, weights, signatures, hot reload with weights.
