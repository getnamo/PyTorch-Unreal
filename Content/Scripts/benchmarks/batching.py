"""PyTorch.Benchmarks.Batching: many concurrent calls from Unreal into one model, one forward pass per call
versus micro-batches built from deferred replies (ue.defer()).

    classify(image)           one forward pass per call, replied on the script's worker thread
    classify_batched(image)   queued; a batcher thread stacks whatever arrived within MAX_WAIT_MS (up to
                              MAX_BATCH) into one forward pass and resolves each call's reply
"""
import queue
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import unreal_bridge as ue

MAX_BATCH = 256
MAX_WAIT_MS = 1.0


class Digits(nn.Module):  # the MNIST CNN the examples use
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                               nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                               nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.f(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
net = Digits().to(device).eval()
stats = {"passes": 0, "rows": 0, "max_batch": 0}
lock = threading.Lock()


def _count(rows):
    with lock:
        stats["passes"] += 1
        stats["rows"] += rows
        stats["max_batch"] = max(stats["max_batch"], rows)


def classify(image):
    x = torch.from_numpy(np.asarray(image, dtype=np.float32).reshape(1, 1, 28, 28)).to(device)
    with torch.inference_mode():
        digit = int(net(x).argmax(-1))
    _count(1)
    return digit


def classify_stacked(images):
    """Unreal stacked N requests into one (N, 28, 28) tensor: one call, one forward pass."""
    x = torch.from_numpy(np.asarray(images, dtype=np.float32).reshape(-1, 1, 28, 28)).to(device, non_blocking=True)
    with torch.inference_mode():
        digits = net(x).argmax(-1).to(torch.int32).cpu().numpy()
    _count(digits.shape[0])
    return digits


requests = queue.SimpleQueue()


def classify_batched(image):
    requests.put((np.asarray(image, dtype=np.float32), ue.defer()))


def _batcher():
    while not ue.should_stop():
        try:
            items = [requests.get(timeout=0.1)]
        except queue.Empty:
            continue
        deadline = time.perf_counter() + MAX_WAIT_MS / 1000
        while len(items) < MAX_BATCH:
            remaining = deadline - time.perf_counter()
            try:
                items.append(requests.get(timeout=remaining) if remaining > 0 else requests.get_nowait())
            except queue.Empty:
                break
        try:
            x = torch.from_numpy(np.stack([i[0].reshape(1, 28, 28) for i in items])).to(device, non_blocking=True)
            with torch.inference_mode():
                digits = net(x).argmax(-1).tolist()
            _count(len(items))
            for (_, reply), d in zip(items, digits):
                reply.resolve(d)
        except Exception as e:  # noqa: BLE001
            for _, reply in items:
                reply.reject(e)


ue.run_in_background(_batcher)


def warmup():
    x = torch.zeros(MAX_BATCH, 1, 28, 28, device=device)
    with torch.inference_mode():
        for n in (1, 16, 64, 256):
            net(x[:n])
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {"device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "", "torch": torch.__version__}


def reset_stats():
    with lock:
        stats.update(passes=0, rows=0, max_batch=0)


def get_stats():
    with lock:
        return dict(stats)
