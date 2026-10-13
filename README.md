# PyTorch-Unreal
PyTorch ML support via [PythonCLI](https://github.com/getnamo/PythonCLI-Unreal) plugin for the Unreal Engine.


## Progress notes

In development for UE 5.8; not usable yet.

- **T0, spikes:** done. Measurements and the decisions they led to are in [Tools/Benchmarks](Tools/Benchmarks/README.md):
  - micro-batching;
  - shared memory → CUDA;
  - ONNX → NNE;
  - Lockstep RL stepping;
  - `torch.compile` with triton-windows.
