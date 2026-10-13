"""Streams bound to model methods (Torch Output Component): a camera, microphone or observation stream runs a
model's exported method on every frame, with no script of its own.

Unreal opens the stream on the model's service with function "_torch_stream" and info
{"model", "method", "input", "binding", "inputs"?}. Each frame becomes one request (batched with others for
batch=True methods); results go back as "_torch_stream_result" events {binding, seq, outputs, per_row, ...},
or as the step's result for Lockstep streams. Latest streams (cameras) keep one request in flight: frames that
arrive meanwhile are skipped, so results never lag behind; Queue streams run every frame. On CUDA, a stream's shared
memory is pinned (cudaHostRegister) once, so frames upload straight from Unreal's slots.
"""
import threading
import time
import traceback

import torch
import unreal_bridge as ue
from unreal_bridge import _host as _bridge

from . import _model

__all__ = ["handle_frame"]


MAX_IN_FLIGHT = 1  # more would batch a stream's frames together: new shapes (recompiles) and no gain at camera rates


class _InFlight:
    """How many of a Latest stream's frames are being processed."""

    def __init__(self):
        self.lock = threading.Lock()
        self.count = 0

    def try_acquire(self):
        with self.lock:
            if self.count >= MAX_IN_FLIGHT:
                return False
            self.count += 1
            return True

    def clear(self):
        with self.lock:
            self.count = max(0, self.count - 1)


class _StreamReply:
    """Answers one frame's request: an event to Unreal, or the Lockstep step's reply."""

    __slots__ = ("ctx", "binding", "frame", "seq", "busy", "age_ms")

    def __init__(self, ctx, binding, frame, seq, busy, age_ms):
        self.ctx, self.binding, self.frame, self.seq, self.busy, self.age_ms = ctx, binding, frame, seq, busy, age_ms

    def resolve(self, value):
        if self.busy is not None:
            self.busy.clear()
        if self.frame is not None:
            self.frame.reply(value)
            return
        self.ctx.host.emit(self.ctx, "_torch_stream_result", [{"binding": self.binding, "seq": self.seq, "frame_age_ms": self.age_ms, **value}])

    def reject(self, error, tb=None):
        if self.busy is not None:
            self.busy.clear()
        message = f"{type(error).__name__}: {error}" if isinstance(error, BaseException) else str(error)
        if self.frame is not None:
            self.frame.fail(message, tb)
            return
        self.ctx.host.emit(self.ctx, "_torch_stream_result", [{"binding": self.binding, "seq": self.seq, "error": message, "traceback": tb or ""}])


def _pin(stream, device):
    """Page-lock the stream's shared memory for CUDA once, so uploads are straight DMA from Unreal's slots."""
    state = stream.user
    if state.get("pinned") is not None or device.type != "cuda" or stream.region is None:
        return
    addr, size = stream.address
    cudart = torch.cuda.cudart()
    err = cudart.cudaHostRegister(addr, size, 0)
    ok = int(err) == 0
    state["pinned"] = ok
    if ok:
        def unpin(_stream, addr=addr):
            torch.cuda.synchronize()
            cudart.cudaHostUnregister(addr)
        stream.on_close.append(unpin)


_UE_TIME_OFFSET = 16777216.0  # FPlatformTime::Seconds() is QPC time plus 2^24 s (to make float truncation obvious)


def _frame_age_ms(frame):
    """How long ago Unreal captured the frame: its time is FPlatformTime::Seconds(), on the same clock as
    perf_counter (QueryPerformanceCounter on Windows, CLOCK_MONOTONIC on Linux) apart from UE's offset."""
    if not frame.time:
        return None
    age = time.perf_counter() - frame.time
    if age < -1e6:
        age += _UE_TIME_OFFSET
    return round(age * 1000, 2) if -1.0 < age < 60.0 else None


def handle_frame(frame):
    age_ms = _frame_age_ms(frame)
    info = frame.info
    h = _model.host(info["model"])
    method = h.methods.get(info.get("method", ""))
    if method is None:
        raise KeyError(f"model '{h.name}' has no exported method '{info.get('method')}'")
    h.ensure_loaded()
    ctx = _bridge.current_context()
    state = frame.stream.user
    lockstep = frame.lockstep
    busy = None
    if frame.stream.policy == "latest":
        busy = state.get("busy")
        if busy is None:
            busy = state["busy"] = _InFlight()
        if not busy.try_acquire():
            frame.stream.skipped += 1  # earlier frames are still running: this one would only add latency
            return None
    _pin(frame.stream, h.device)
    if busy is not None:
        try:
            return _submit(frame, h, method, info, ctx, lockstep, busy, age_ms)
        except BaseException:
            busy.clear()
            raise
    return _submit(frame, h, method, info, ctx, lockstep, busy, age_ms)


def _submit(frame, h, method, info, ctx, lockstep, busy, age_ms):
    name = info.get("input") or (method.params[0].name if method.params else "x")
    options = {"layouts": {name: frame.layout}} if frame.layout else {}
    rate = (frame.meta or {}).get("sample_rate") or (frame.info or {}).get("sample_rate")
    if rate:
        options["rates"] = {name: rate}
    inputs = dict(info.get("inputs") or {})
    array = frame.array
    inputs[name] = array.reshape((1,) + tuple(array.shape))
    tensors, values = h.convert(method, inputs, options)
    # The slot goes back to Unreal when this returns: uploads from it must be done (pinned memory copies
    # asynchronously), and what stays on the CPU must own its data
    if h.device.type == "cuda":
        torch.cuda.current_stream(h.device).synchronize()
    tensors = {k: (t.clone() if t.device.type == "cpu" else t) for k, t in tensors.items()}
    reply = _StreamReply(ctx, info.get("binding"), frame if lockstep else None, frame.seq, busy, age_ms)
    if lockstep:
        frame.defer()
    try:
        if method.batch:
            b = h.batchers.get(method.name)
            if b is None:
                b = h.batchers[method.name] = _model._Batcher(h, method)
            b.submit(_model._Request(tensors, values, [1], reply))
        else:
            outputs, per_row, ms = h.call(method, tensors, values, 1)
            reply.resolve({"outputs": outputs, "per_row": per_row, "rows": 1, "batch": 1, "ms": round(ms, 3)})
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        ue.error(f"[{h.name}] {method.name} on stream '{frame.stream.info.get('name') or frame.name}' failed:\n{tb}")
        reply.reject(e, tb)
    return None
