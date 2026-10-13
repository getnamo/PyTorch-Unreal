"""PyTorch.Benchmarks.NNE: export a CNN and an MLP policy to ONNX (dynamo exporter, opset 20, dynamic batch)
with reference inputs and outputs, for Unreal to run through NNE and compare.

    export(out_dir) -> [{name, onnx, input, cases: [{batch, input_shape, output_shape, input_file,
                         output_file, torch_cpu_ms, torch_cuda_ms}]}]
"""
import os
import time

import numpy as np
import torch
import torch.nn as nn


class Digits(nn.Module):
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


class Vision(nn.Module):  # a small ResNet-style classifier at 224x224
    def __init__(self):
        super().__init__()
        def block(i, o, s):
            return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(),
                                 nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU())
        self.f = nn.Sequential(nn.Conv2d(3, 32, 7, 2, 3, bias=False), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(3, 2, 1),
                               block(32, 64, 1), block(64, 128, 2), block(128, 256, 2), block(256, 512, 2))
        self.head = nn.Linear(512, 1000)

    def forward(self, image):
        return self.head(self.f(image).mean((2, 3)))


class Stylize(nn.Module):  # image to image at 256x256 (an encoder-decoder like fast style transfer)
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(3, 32, 9, 1, 4), nn.ReLU(), nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
                                 nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU())
        self.res = nn.Sequential(*[nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU()) for _ in range(5)])
        self.dec = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(128, 64, 3, 1, 1), nn.ReLU(),
                                 nn.Upsample(scale_factor=2), nn.Conv2d(64, 32, 3, 1, 1), nn.ReLU(), nn.Conv2d(32, 3, 9, 1, 4))

    def forward(self, image):
        return torch.sigmoid(self.dec(self.res(self.enc(image))))


def _time(fn, iters=200, budget_s=1.0):
    for _ in range(5):
        fn()
    t = time.perf_counter()
    n = 0
    while n < iters and (n < 5 or time.perf_counter() - t < budget_s):
        fn()
        n += 1
    return (time.perf_counter() - t) / n * 1000


def directml_fixups(path):
    """The dynamo exporter writes Reshape with allowzero=1, which DirectML rejects. With a constant shape
    that has no 0, allowzero changes nothing: drop it."""
    import onnx
    from onnx import numpy_helper
    model = onnx.load(path)
    consts = {t.name: t for t in model.graph.initializer}
    for n in model.graph.node:
        if n.op_type == "Reshape" and len(n.input) > 1 and n.input[1] in consts:
            if 0 not in numpy_helper.to_array(consts[n.input[1]]).tolist():
                for a in list(n.attribute):
                    if a.name == "allowzero":
                        n.attribute.remove(a)
    onnx.save(model, path)


MODELS = {
    "digits": (Digits, (1, 1, 28, 28), "image", (1, 16)),
    "policy": (Policy, (1, 64), "obs", (1, 256)),
    "vision": (Vision, (1, 3, 224, 224), "image", (1, 8)),
    "stylize": (Stylize, (1, 3, 256, 256), "image", (1,)),
    # static batch (no dynamic dimension): DirectML runs batch > 1 much faster this way
    "vision_static8": (Vision, (8, 3, 224, 224), "image", (8,)),
    "policy_static256": (Policy, (256, 64), "obs", (256,)),
}


def export(out_dir, fixups=True, names=("digits", "policy", "vision", "stylize", "vision_static8", "policy_static256")):
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(0)
    results = []
    for name in names:
        cls, shape, input_name, batches = MODELS[name]
        model = cls()
        model.eval()
        # BatchNorm with non-trivial statistics, so the comparison means something
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean.uniform_(-0.5, 0.5)
                m.running_var.uniform_(0.5, 2.0)
        path = os.path.join(out_dir, f"{name}.onnx")
        t = time.perf_counter()
        torch.onnx.export(model, (torch.randn(*shape),), path, dynamo=True, opset_version=20, input_names=[input_name],
                          output_names=["output"], dynamic_shapes=None if "static" in name else {input_name: {0: torch.export.Dim("batch", min=1, max=4096)}},
                          external_data=False, verbose=False)
        if fixups:
            directml_fixups(path)
        entry = {"name": name, "onnx": path, "input": input_name, "export_s": round(time.perf_counter() - t, 2), "cases": []}
        for b in batches:
            x = torch.randn(b, *shape[1:])
            with torch.inference_mode():
                y = model(x)
            in_file, out_file = os.path.join(out_dir, f"{name}_b{b}_in.bin"), os.path.join(out_dir, f"{name}_b{b}_out.bin")
            x.numpy().astype(np.float32).tofile(in_file)
            y.numpy().astype(np.float32).tofile(out_file)
            with torch.inference_mode():
                cpu_ms = _time(lambda: model(x))
                cuda_ms = None
                if torch.cuda.is_available():
                    mg, xg = model.cuda(), x.cuda()

                    def run():
                        mg(xg)
                        torch.cuda.synchronize()
                    cuda_ms = round(_time(run), 4)
                    model.cpu()
            entry["cases"].append({"batch": b, "input_shape": list(x.shape), "output_shape": list(y.shape), "input_file": in_file,
                                   "output_file": out_file, "torch_cpu_ms": round(cpu_ms, 4), "torch_cuda_ms": cuda_ms})
        results.append(entry)
    return results
