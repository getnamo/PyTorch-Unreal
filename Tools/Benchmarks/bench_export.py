"""T0(c): dynamo ONNX export (opset 20, dynamic batch) of a CNN and an MLP policy; parity and latency in
onnxruntime (CPU, CUDA) against torch. Writes <out>/cnn.onnx, <out>/mlp.onnx and reference inputs/outputs
(<name>_in.bin / <name>_out.bin, float32) for the Unreal NNE test.

usage: python -I bench_export.py <site-packages> <out-dir>
"""
import json
import os
import sys
import time

import site
site.addsitedir(sys.argv[1])
out_dir = sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort


def out(**kw):
    print(json.dumps(kw), flush=True)


class CNN(nn.Module):  # MNIST-sized classifier (what the example map uses) with BatchNorm
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
                               nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
                               nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, image):
        return self.f(image).softmax(-1)


class Policy(nn.Module):
    def __init__(self, obs=64, act=8, hidden=256):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.mean = nn.Linear(hidden, act)

    def forward(self, obs):
        return torch.tanh(self.mean(self.body(obs)))


def bench(fn, iters=200):
    for _ in range(10):
        fn()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t) / iters * 1000


torch.manual_seed(0)
cases = [("cnn", CNN().eval(), (1, 1, 28, 28), "image", [1, 16]),
         ("mlp", Policy().eval(), (1, 64), "obs", [1, 256])]
for name, model, shape, input_name, batches in cases:
    x = torch.randn(*shape)
    path = os.path.join(out_dir, f"{name}.onnx")
    t = time.perf_counter()
    batch = torch.export.Dim("batch", min=1, max=4096)
    prog = torch.onnx.export(model, (x,), path, dynamo=True, opset_version=20, input_names=[input_name],
                             output_names=["output"], dynamic_shapes={input_name: {0: batch}}, external_data=False)
    export_s = time.perf_counter() - t
    out(event="export", model=name, path=path, seconds=round(export_s, 2), bytes=os.path.getsize(path),
        opset=prog.model.opset_imports.get("", None) if prog is not None else None)
    for b in batches:
        xb = torch.randn(b, *shape[1:])
        with torch.inference_mode():
            ref = model(xb).numpy()
        np.ascontiguousarray(xb.numpy(), dtype=np.float32).tofile(os.path.join(out_dir, f"{name}_b{b}_in.bin"))
        ref.astype(np.float32).tofile(os.path.join(out_dir, f"{name}_b{b}_out.bin"))
        torch_cpu_ms = bench(lambda: model(xb))
        mg = model.cuda()
        xg = xb.cuda()
        def torch_gpu():
            with torch.inference_mode():
                mg(xg)
            torch.cuda.synchronize()
        torch_gpu_ms = bench(torch_gpu)
        model.cpu()
        for providers in (["CPUExecutionProvider"], ["CUDAExecutionProvider", "CPUExecutionProvider"]):
            try:
                sess = ort.InferenceSession(path, providers=providers)
            except Exception as e:  # noqa: BLE001
                out(event="ort_error", model=name, providers=providers, error=str(e)[:500])
                continue
            feed = {input_name: xb.numpy()}
            y = sess.run(None, feed)[0]
            ms = bench(lambda: sess.run(None, feed))
            out(event="ort", model=name, batch=b, provider=sess.get_providers()[0], ms=round(ms, 4),
                torch_cpu_ms=round(torch_cpu_ms, 4), torch_cuda_ms=round(torch_gpu_ms, 4),
                max_abs_diff=float(np.abs(y - ref).max()))
