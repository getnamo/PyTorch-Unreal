"""T0(f): torch.compile on Windows CUDA with triton-windows, in the PythonCLI interpreter (-I + addsitedir).

usage: python -I bench_compile.py <site-packages> <cache-dir> [mode...]
Prints JSON lines: eager vs compiled latency for a CNN and an MLP policy, and compile times.
"""
import json
import os
import sys
import time

site_dir, cache_dir = sys.argv[1], sys.argv[2]
modes = sys.argv[3:] or ["default", "reduce-overhead", "max-autotune-no-cudagraphs"]
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(cache_dir, "inductor"))
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(cache_dir, "triton"))
import site
site.addsitedir(site_dir)


def site_dir_as_platlib(site_dir):
    """What PythonCLI's runtime does: code that finds its own files via sysconfig's platlib/purelib (triton's
    bundled TinyCC and CUDA files) must see the site dir, as it would in a venv."""
    import sysconfig
    get_paths, get_path = sysconfig.get_paths, sysconfig.get_path

    def paths(scheme=None, vars=None, expand=True):
        p = get_paths(scheme, vars, expand) if scheme else get_paths(vars=vars, expand=expand)
        if scheme is None and vars is None and expand:
            p["platlib"] = p["purelib"] = site_dir
        return p

    def path(name, scheme=None, vars=None, expand=True):
        if name in ("platlib", "purelib") and scheme is None and vars is None and expand:
            return site_dir
        return get_path(name, scheme, vars, expand) if scheme else get_path(name, vars=vars, expand=expand)
    sysconfig.get_paths, sysconfig.get_path = paths, path


if os.environ.get("NO_PLATLIB_FIX") != "1":
    site_dir_as_platlib(site_dir)

t0 = time.perf_counter()
import torch
import torch.nn as nn
import_s = time.perf_counter() - t0
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
dev = "cuda"


def out(**kw):
    print(json.dumps(kw), flush=True)


out(event="env", torch=torch.__version__, cuda=torch.version.cuda, import_s=round(import_s, 2),
    gpu=torch.cuda.get_device_name(0), triton=__import__("triton").__version__)


class CNN(nn.Module):  # a small ResNet-ish image model at 224x224
    def __init__(self):
        super().__init__()
        def block(i, o, s):
            return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                                 nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))
        self.stem = nn.Sequential(nn.Conv2d(3, 32, 7, 2, 3, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1))
        self.body = nn.Sequential(block(32, 64, 1), block(64, 128, 2), block(128, 256, 2), block(256, 512, 2))
        self.head = nn.Linear(512, 1000)

    def forward(self, x):
        return self.head(self.body(self.stem(x)).mean((2, 3)))


class MLP(nn.Module):  # a PPO-sized policy
    def __init__(self, obs=64, act=8, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, act))

    def forward(self, x):
        return self.net(x)


def timeit(fn, x, iters):
    with torch.inference_mode():
        for _ in range(5):
            fn(x)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(iters):
            fn(x)
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1000


cases = [
    ("cnn224", CNN, [(1, 3, 224, 224), (32, 3, 224, 224)], 50),
    ("mlp64", MLP, [(1, 64), (256, 64), (4096, 64)], 500),
]
for name, cls, shapes, iters in cases:
    torch.manual_seed(0)
    model = cls().to(dev).eval()
    for shape in shapes:
        x = torch.randn(*shape, device=dev)
        eager = timeit(model, x, iters)
        out(event="eager", model=name, shape=list(shape), ms=round(eager, 4))
        for mode in modes:
            torch._dynamo.reset()
            compiled = torch.compile(model, mode=mode, dynamic=False)
            t = time.perf_counter()
            try:
                with torch.inference_mode():
                    ref = model(x)
                    y = compiled(x)
                    torch.cuda.synchronize()
            except Exception as e:  # noqa: BLE001
                out(event="error", model=name, shape=list(shape), mode=mode, error=f"{type(e).__name__}: {e}"[:2000])
                continue
            compile_s = time.perf_counter() - t
            diff = (y.float() - ref.float()).abs().max().item()
            ms = timeit(compiled, x, iters)
            out(event="compiled", model=name, shape=list(shape), mode=mode, compile_s=round(compile_s, 2), ms=round(ms, 4),
                speedup=round(eager / ms, 2), max_abs_diff=diff)
