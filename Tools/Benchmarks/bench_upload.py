"""T0(b): a 1080p BGRA frame in shared memory -> CUDA tensor, ready for a model (float NCHW RGB, normalized).

usage: python -I bench_upload.py <site-packages>
"""
import json
import mmap
import os
import sys
import time

import site
site.addsitedir(sys.argv[1])
import numpy as np
import torch


def out(**kw):
    print(json.dumps(kw), flush=True)


H, W = 1080, 1920
nbytes = H * W * 4
name = f"pue_bench_{os.getpid()}"
mm = mmap.mmap(-1, nbytes * 3, tagname=name)  # what Unreal's stream region looks like: 3 slots
buf = memoryview(mm)
frame = np.frombuffer(buf[:nbytes], dtype=np.uint8).reshape(H, W, 4)
frame[:] = np.random.default_rng(0).integers(0, 255, size=(H, W, 4), dtype=np.uint8)
dev = torch.device("cuda")
mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
stream = torch.cuda.Stream()


def to_model_input(u8):  # (H, W, 4) uint8 BGRA on the GPU -> (1, 3, H, W) float RGB normalized
    x = u8[..., [2, 1, 0]].permute(2, 0, 1).unsqueeze(0).float().div_(255)
    return (x - mean) / std


def bench(label, fn, iters=200):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t) / iters * 1000
    out(event="upload", path=label, ms=round(ms, 3), gb_s=round(nbytes / ms / 1e6, 2), fps=round(1000 / ms))


# 1. pageable: torch.from_numpy(view).cuda() - the simple way
bench("pageable copy (from_numpy(shm).cuda())", lambda: torch.from_numpy(frame).to(dev))

# 2. pinned staging: shm -> pinned host buffer (memcpy) -> GPU (non_blocking)
pinned = torch.empty((H, W, 4), dtype=torch.uint8).pin_memory()
pinned_np = pinned.numpy()


def pinned_path():
    np.copyto(pinned_np, frame)
    return pinned.to(dev, non_blocking=True)


bench("pinned staging (memcpy + async H2D)", pinned_path)

# 3. register the shared memory itself as pinned (cudaHostRegister): no staging copy at all
cudart = torch.cuda.cudart()
shm_tensor = torch.frombuffer(buf, dtype=torch.uint8)  # whole region
ptr = shm_tensor.data_ptr()
err = cudart.cudaHostRegister(ptr, nbytes * 3, 0)
out(event="host_register", result=str(err))
registered = str(err).endswith("cudaSuccess") or int(err) == 0 if not isinstance(err, tuple) else False
if registered:
    view = shm_tensor[:nbytes].view(H, W, 4)
    bench("cudaHostRegister(shm) + async H2D", lambda: view.to(dev, non_blocking=True))

# 4. end to end with conversion to a normalized NCHW float tensor
bench("pinned staging + BGRA->RGB float normalize", lambda: to_model_input(pinned_path()))
if registered:
    bench("registered shm + BGRA->RGB float normalize", lambda: to_model_input(view.to(dev, non_blocking=True)))

# 5. small frames (224x224 crop-size input, 28x28 MNIST)
for h, w in ((224, 224), (28, 28)):
    small = np.ascontiguousarray(frame[:h, :w])
    p = torch.empty((h, w, 4), dtype=torch.uint8).pin_memory()
    pn = p.numpy()

    def small_path():
        np.copyto(pn, small)
        return p.to(dev, non_blocking=True)
    bench(f"pinned staging {w}x{h}", small_path, 1000)
    bench(f"pageable {w}x{h}", lambda: torch.from_numpy(small).to(dev), 1000)

if registered:
    cudart.cudaHostUnregister(ptr)
