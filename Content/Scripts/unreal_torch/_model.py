"""Models: one instance per process for each @ut.model, typed exported methods, and batching.

Unreal's UTorchSubsystem talks to a model script (a PythonCLI service) with reserved calls:
    _torch_load(name, options) -> {signature, device, load_ms}
    _torch_run(name, method, inputs, rows, options) -> {outputs, per_row, rows, batch, ms}
    _torch_unload(name), _torch_signature(name), _torch_list(), _torch_stats(name)
Tensor inputs carry a leading batch dimension with sum(rows) rows: Unreal stacks the requests of a tick, one
entry of `rows` per request. Outputs listed in per_row are split back by rows in Unreal.
"""
import inspect
import queue
import threading
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
import unreal_bridge as ue
from unreal_bridge import _host as _bridge

from . import _types, _weights

__all__ = ["Model", "model", "export", "get_model", "signature", "host"]

_lock = threading.RLock()
_classes = {}  # model name -> (cls, options)
_hosts = {}  # model name -> ModelHost; process-wide, so a reloaded script keeps its models' weights
_defaults_applied = False


# ---------------------------------------------------------------------------------------------------- export

class _Method:
    def __init__(self, fn, batch, max_batch, max_wait_ms, name):
        self.fn = fn
        self.name = name or fn.__name__
        self.batch = batch
        self.max_batch = int(max_batch)
        self.max_wait = float(max_wait_ms) / 1000.0
        sig = inspect.signature(fn)
        self.params = []
        for p in list(sig.parameters.values())[1:]:  # after self
            if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            ann = p.annotation if p.annotation is not inspect.Parameter.empty else None
            self.params.append(_types.InputSpec(p.name, ann, p.default, p.default is not inspect.Parameter.empty))
        self.returns = sig.return_annotation if sig.return_annotation is not inspect.Signature.empty else None

    def describe(self):
        return {"name": self.name, "batch": self.batch, "max_batch": self.max_batch, "inputs": [p.describe() for p in self.params],
                "returns": _types.describe_annotation(self.returns)["type"] if self.returns is not None else "",
                "doc": inspect.getdoc(self.fn) or ""}


class _Export:
    """@ut.export(batch=True, max_batch=256, max_wait_ms=1.0): make a model method callable from Unreal.
    With batch=True, concurrent requests are stacked on dim 0 into one call (outputs are split back per
    request); otherwise the method gets one request's values at a time (no batch dimension)."""

    def __call__(self, fn=None, *, batch=False, max_batch=256, max_wait_ms=1.0, name=None):
        def mark(f):
            f.__ut_export__ = _Method(f, batch, max_batch, max_wait_ms, name)
            return f
        return mark(fn) if callable(fn) else mark


export = _Export()


# ----------------------------------------------------------------------------------------------------- Model

class Model:
    """Base class of @ut.model classes. Build networks in setup() (they are moved to self.device and put in
    eval mode afterwards); export methods with @ut.export."""

    name = ""
    device = torch.device("cpu")

    def setup(self):
        pass

    def warmup(self):
        """Called after loading (when warmup is on): by default each exported method runs once on example
        inputs built from its annotations (fixed-size ut.Tensor / ut.Image)."""
        self.host._auto_warmup()

    # -- modules and weights -------------------------------------------------------------------------
    def modules(self):
        """{attribute: nn.Module} of this model (before compiling)."""
        mods = getattr(self, "_ut_modules", None)
        if mods is not None:
            return dict(mods)
        return {k: v for k, v in vars(self).items() if isinstance(v, nn.Module)}

    def state_dict(self):
        out = {}
        for attr, m in self.modules().items():
            for k, v in m.state_dict().items():
                out[f"{attr}.{k}"] = v
        return out

    def load_state_dict(self, state, strict=True):
        """A combined state ("net.layer.weight", from state_dict/save_weights), or a plain module state when
        the model has one module (pretrained checkpoints)."""
        mods = self.modules()
        missing, unexpected = [], []
        prefixed = any(k.split(".", 1)[0] in mods for k in state)
        for attr, m in mods.items():
            sub = {k[len(attr) + 1:]: v for k, v in state.items() if k.startswith(attr + ".")} if prefixed else (state if len(mods) == 1 else {})
            if not sub:
                continue
            r = m.load_state_dict(sub, strict=strict)
            missing += [f"{attr}.{k}" for k in r.missing_keys]
            unexpected += [f"{attr}.{k}" for k in r.unexpected_keys]
        return missing, unexpected

    def load_weights(self, path=None, missing_ok=False, strict=True):
        """Load weights saved by save_weights, a checkpoint, or a hf:// file. Relative paths: Saved/Torch/
        Models/<model>/ first, then Content/Models/. True if loaded."""
        path = path or f"{self.name}.safetensors"
        file = _weights.resolve(path, self.name)
        if file is None:
            if missing_ok:
                return False
            raise FileNotFoundError(f"weights '{path}' not found in {_weights.saved_dir(self.name)} or Content/Models")
        with self.host.lock:
            self.load_state_dict(_weights.load_state(file, self.device), strict=strict)
        ue.log(f"[{self.name}] weights loaded from {file}")
        return True

    def save_weights(self, path=None):
        """Save all modules' weights (safetensors unless the name ends in .pt); returns the file."""
        file = _weights.resolve(path or f"{self.name}.safetensors", self.name, for_write=True)
        with self.host.lock:
            return _weights.save_state(self.state_dict(), file)

    def publish(self, source):
        """Swap new weights into this (live) model: from a module (or a model with the same modules), or a
        state dict. Requests already running finish with the old weights."""
        if isinstance(source, Model):
            state = source.state_dict()
        elif isinstance(source, nn.Module):
            mods = self.modules()
            if len(mods) != 1:
                raise ValueError("publish(module) needs a model with one module; pass a state dict or a model")
            state = {f"{next(iter(mods))}.{k}": v for k, v in source.state_dict().items()}
        else:
            state = source
        with torch.no_grad(), self.host.lock:
            self.load_state_dict({k: v.detach().to(self.device) for k, v in state.items()}, strict=False)
        self.host.publishes += 1


def model(name=None, **options):
    """@ut.model("name", device="auto", precision=None, compile=None, warmup=True, reload="keep", cudnn_benchmark=False): one
    instance per process, shared by every Unreal caller. name is the Torch Model asset's Model Name.

        device     "auto" (cuda, then xpu, then cpu), "cuda", "cuda:1", "cpu"
        precision  None (fp32), "bf16" or "fp16": autocast around exported calls
        compile    None, True / "default", or "reduce-overhead" (CUDA graphs: small per-step calls)
        reload     "keep": a reloaded script keeps the weights (where shapes match); "fresh": starts over
        cudnn_benchmark  tune convolutions per input shape (fixed shapes; ~100 ms once per new batch size)
        pad        with compile: pad batches to power-of-two sizes (default True)
    """
    def register(cls, model_name=None):
        if not (isinstance(cls, type) and issubclass(cls, Model)):
            raise TypeError("@ut.model goes on a subclass of ut.Model")
        model_name = model_name or cls.__name__
        cls.name = model_name
        _install_handlers()
        with _lock:
            _classes[model_name] = (cls, options)
            h = _hosts.get(model_name)
        if h is not None:
            h.rebind(cls, options)  # a reloaded script: same weights, new code
        return cls
    if isinstance(name, type):
        return register(name)
    return lambda cls: register(cls, name)


def host(name):
    with _lock:
        h = _hosts.get(name)
        if h is None:
            entry = _classes.get(name)
            if entry is None:
                raise KeyError(f"no @ut.model('{name}') is registered (models: {', '.join(_classes) or 'none'})")
            h = _hosts[name] = ModelHost(name, *entry)
        return h


def get_model(name, load=True):
    """The live instance of a model (loading it if needed): what jobs train and publish to."""
    h = host(name)
    if load:
        h.ensure_loaded()
    return h.instance


def signature(name):
    return host(name).signature()


# ------------------------------------------------------------------------------------------------- the host

class ModelHost:
    def __init__(self, name, cls, options):
        self.name = name
        self.cls = cls
        self.options = dict(options)
        self.overrides = {}
        self.instance = None
        self.device = None
        self.lock = threading.RLock()
        self.stream = None
        self.batchers = {}
        self.load_ms = 0.0
        self.load_error = None  # (time, message) of the last failed load
        self.publishes = 0
        self.stats = {"calls": 0, "rows": 0, "passes": 0, "max_batch": 0, "forward_ms": 0.0}
        self.stats_lock = threading.Lock()

    # -- options -------------------------------------------------------------------------------------
    def option(self, key, default=None):
        v = self.overrides.get(key)
        if v not in (None, ""):
            return v
        v = self.options.get(key)
        return default if v in (None, "") else v

    @property
    def methods(self):
        out = {}
        for klass in reversed(self.cls.__mro__):
            for attr, fn in vars(klass).items():
                m = getattr(fn, "__ut_export__", None)
                if m is not None:
                    out[m.name] = m
        return out

    def signature(self):
        return {"model": self.name, "class": self.cls.__name__, "doc": inspect.getdoc(self.cls) or "",
                "loaded": self.instance is not None, "device": str(self.device) if self.device else "",
                "precision": self.option("precision", "") or "", "compile": str(self.option("compile", "") or ""),
                "methods": [m.describe() for m in self.methods.values()]}

    # -- loading -------------------------------------------------------------------------------------
    def ensure_loaded(self):
        if self.instance is None:
            if self.load_error is not None and time.monotonic() - self.load_error[0] < 10.0:
                # Don't run a failing setup() for every request (or stream frame): once every 10 s at most
                raise RuntimeError(f"model '{self.name}' failed to load: {self.load_error[1]}")
            try:
                self.load()
            except Exception as e:
                self.load_error = (time.monotonic(), f"{type(e).__name__}: {e}")
                raise
        return self.instance

    def load(self, overrides=None, state=None):
        global _defaults_applied
        with self.lock:
            if overrides is not None:
                self.overrides = {k: v for k, v in overrides.items() if v not in (None, "")}
            t0 = time.perf_counter()
            if not _defaults_applied:
                _defaults_applied = True
                torch.set_float32_matmul_precision("high")  # TF32 matmuls on Ampere and newer
            if self.option("cudnn_benchmark"):
                # Tunes per input shape (a one-off cost of ~100 ms per new batch size): for fixed shapes
                torch.backends.cudnn.benchmark = True
            self.device = _resolve_device(self.option("device", "auto"))
            inst = self.cls()
            inst.name = self.name
            inst.device = self.device
            inst.host = self
            with torch.device(self.device):  # tensors made in setup() land on the device
                inst.setup()
            mods = {k: v for k, v in vars(inst).items() if isinstance(v, nn.Module)}
            for m in mods.values():
                m.to(self.device).eval()
            inst._ut_modules = mods
            if state:
                carried = _carry_state(inst, state)
                ue.log(f"[{self.name}] reloaded, kept {carried} weight tensors")
            mode = self.option("compile")
            if mode:
                for k, m in mods.items():
                    setattr(inst, k, torch.compile(m, mode=None if mode in (True, "default", "true", "1") else mode))
            self.stream = torch.cuda.Stream(self.device) if self.device.type == "cuda" else None
            old, self.instance = self.instance, inst
            if self.option("warmup", True) not in (False, "false", "0", 0):
                try:
                    inst.warmup()
                except Exception:  # noqa: BLE001 - a warmup failure shows up again on the first real call
                    ue.warning(f"[{self.name}] warmup failed:\n{traceback.format_exc()}")
            self.load_ms = (time.perf_counter() - t0) * 1000
            del old
            return inst

    def rebind(self, cls, options):
        """The script was reloaded: new class, same name. Reload it now (keeping weights) if it was loaded."""
        with self.lock:
            self.cls = cls
            self.options = dict(options)
            for b in self.batchers.values():
                b.stop()
            self.batchers = {}
            if self.instance is None:
                return
            state = None if self.option("reload", "keep") == "fresh" else self.instance.state_dict()
            try:
                self.load(state=state)
            except Exception:  # noqa: BLE001
                self.instance = None
                ue.error(f"[{self.name}] reload failed:\n{traceback.format_exc()}")

    def unload(self):
        with self.lock:
            for b in self.batchers.values():
                b.stop()
            self.batchers = {}
            self.instance = None
            self.stream = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -- running -------------------------------------------------------------------------------------
    def _context(self):
        """inference_mode + autocast (precision) + the model's CUDA stream."""
        stack = _Stack()
        stack.enter(torch.inference_mode())
        precision = (self.option("precision") or "").lower()
        if precision in ("bf16", "bfloat16", "fp16", "float16", "half"):
            dt = torch.bfloat16 if precision.startswith("b") else torch.float16
            stack.enter(torch.autocast(self.device.type, dtype=dt))
        if self.stream is not None:
            self.stream.wait_stream(torch.cuda.default_stream(self.device))
            stack.enter(torch.cuda.stream(self.stream))
        return stack

    def convert(self, method, inputs, options):
        """Unreal's inputs -> keyword arguments: tensors on the device (batched), JSON values as they are."""
        layouts = (options or {}).get("layouts") or {}
        rates = (options or {}).get("rates") or {}
        unknown = set(inputs) - {p.name for p in method.params}
        if unknown:
            raise TypeError(f"{self.name}.{method.name} has no input {', '.join(sorted(unknown))} (inputs: {', '.join(p.name for p in method.params) or 'none'})")
        tensors, values = {}, {}
        for p in method.params:
            if p.name in inputs and inputs[p.name] is not None:
                v = inputs[p.name]
                if p.is_tensor:
                    if isinstance(p.annotation, _types.Audio):
                        tensors[p.name] = p.annotation.convert(v, self.device, rate=rates.get(p.name))
                    else:
                        tensors[p.name] = p.annotation.convert(v, self.device, layout=layouts.get(p.name))
                elif p.annotation in (int, float, str, bool) and v is not None:
                    values[p.name] = p.annotation(v)
                else:
                    values[p.name] = v
            elif not p.has_default:
                raise TypeError(f"{self.name}.{method.name}: missing input '{p.name}'")
        return tensors, values

    def call(self, method, tensors, values, total):
        """Run the method once on all rows (batch) or once per row; returns (outputs, per_row, ms)."""
        inst = self.ensure_loaded()
        fn = method.fn
        t0 = time.perf_counter()
        with self.lock, self._context():
            if method.batch:
                padded = total
                if self.option("compile") and self.option("pad", True) and tensors:
                    # Compiled code is specialized to its input shapes: pad to power-of-two batch sizes, so a
                    # varying number of requests compiles a few times rather than once per size
                    padded = 1 << (total - 1).bit_length()
                    if padded != total:
                        tensors = {k: torch.cat([t, t[-1:].expand(padded - total, *t.shape[1:])]) for k, t in tensors.items()}
                result = fn(inst, **tensors, **values)
                outputs, per_row = _outputs(result, padded)
                if padded != total:
                    outputs = {k: (v[:total] if k in per_row else v) for k, v in outputs.items()}
            else:
                rows = [_outputs(fn(inst, **{k: v[i] for k, v in tensors.items()}, **values), None)[0] for i in range(total)]
                outputs, per_row = _stack_rows(rows)
            outputs = {k: _to_unreal(v) for k, v in outputs.items()}  # syncs the model's stream
        ms = (time.perf_counter() - t0) * 1000
        with self.stats_lock:
            s = self.stats
            s["passes"] += 1
            s["rows"] += total
            s["max_batch"] = max(s["max_batch"], total)
            s["forward_ms"] += ms
        return outputs, per_row, ms

    def run(self, method_name, inputs, rows, options):
        """A _torch_run call: answered now, or later by the method's batcher (a Deferred)."""
        method = self.methods.get(method_name)
        if method is None:
            raise KeyError(f"model '{self.name}' has no exported method '{method_name}' (methods: {', '.join(self.methods) or 'none'})")
        self.ensure_loaded()
        rows = [int(r) for r in (rows or [1])]
        total = sum(rows)
        with self.stats_lock:
            self.stats["calls"] += 1
        tensors, values = self.convert(method, inputs or {}, options)
        for k, t in tensors.items():
            if t.shape[0] != total:
                raise ValueError(f"{self.name}.{method_name}: input '{k}' has {t.shape[0]} rows, the requests {total}")
        if method.batch:
            b = self.batchers.get(method.name)
            if b is None:
                b = self.batchers[method.name] = _Batcher(self, method)
            reply = ue.defer()
            b.submit(_Request(tensors, values, rows, reply))
            return reply
        outputs, per_row, ms = self.call(method, tensors, values, total)
        return {"outputs": outputs, "per_row": per_row, "rows": total, "batch": total, "ms": round(ms, 3)}

    # -- warmup --------------------------------------------------------------------------------------
    def _auto_warmup(self):
        for m in self.methods.values():
            if not any(p.is_tensor for p in m.params):
                continue  # nothing to warm up (and it may have side effects)
            tensors = {}
            for p in m.params:
                if p.is_tensor:
                    ex = p.annotation.example(1, self.device)
                    if ex is None:
                        break
                    tensors[p.name] = ex
                elif not p.has_default:
                    break
            else:
                with self.lock, self._context():
                    if m.batch:
                        m.fn(self.instance, **tensors)
                    else:
                        m.fn(self.instance, **{k: v[0] for k, v in tensors.items()})
        if self.device is not None and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)


class _Request:
    __slots__ = ("tensors", "values", "rows", "reply", "key")

    def __init__(self, tensors, values, rows, reply):
        self.tensors, self.values, self.rows, self.reply = tensors, values, rows, reply
        self.key = (tuple(sorted((k, tuple(t.shape[1:]), str(t.dtype)) for k, t in tensors.items())), repr(sorted(values.items())))


class _Batcher:
    """Collects requests for one batched method and runs them as one call: whatever arrives within
    max_wait_ms of the first (up to max_batch rows), with the same input shapes and JSON values."""

    def __init__(self, host, method):
        self.host = host
        self.method = method
        self.queue = queue.SimpleQueue()
        self.stopped = threading.Event()
        self.held = []
        self.thread = threading.Thread(target=self._run, name=f"torch:{host.name}.{method.name}", daemon=True)
        self.thread.start()

    def submit(self, request):
        self.queue.put(request)

    def stop(self):
        self.stopped.set()
        self.queue.put(None)

    def _take(self, timeout):
        if self.held:
            return self.held.pop(0)
        return self.queue.get(timeout=timeout)

    def _take_until(self, deadline):
        """The next request before the deadline. Polls with short sleeps: a blocking get with a timeout waits
        in steps of the system timer on Windows (~15 ms), far longer than a batching window."""
        while True:
            if self.held:
                return self.held.pop(0)
            try:
                return self.queue.get_nowait()
            except queue.Empty:
                pass
            left = deadline - time.perf_counter()
            if left <= 0:
                raise queue.Empty
            time.sleep(min(left, 0.0002))

    def _run(self):
        while not self.stopped.is_set():
            try:
                first = self._take(0.5)
            except queue.Empty:
                continue
            if first is None:
                break
            batch, total = [first], sum(first.rows)
            deadline = time.perf_counter() + self.method.max_wait
            skipped = []
            while total < self.method.max_batch:
                try:
                    nxt = self._take_until(deadline)
                except queue.Empty:
                    break
                if nxt is None:
                    self.stopped.set()
                    break
                if nxt.key != first.key or total + sum(nxt.rows) > self.method.max_batch:
                    skipped.append(nxt)  # another shape / JSON values, or too many rows: next round
                    continue
                batch.append(nxt)
                total += sum(nxt.rows)
            self.held = skipped + self.held
            self._execute(batch, total)
        for r in self.held:
            r.reply.reject(f"model '{self.host.name}' was unloaded")
        while True:
            try:
                r = self.queue.get_nowait()
            except queue.Empty:
                break
            if r is not None:
                r.reply.reject(f"model '{self.host.name}' was unloaded")

    def _execute(self, batch, total):
        try:
            if len(batch) == 1:
                tensors = batch[0].tensors
            else:
                tensors = {k: torch.cat([r.tensors[k] for r in batch]) for k in batch[0].tensors}
            outputs, per_row, ms = self.host.call(self.method, tensors, batch[0].values, total)
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            ue.error(f"[{self.host.name}] {self.method.name} failed:\n{tb}")
            for r in batch:
                r.reply.reject(e, tb)
            return
        offset = 0
        for r in batch:
            n = sum(r.rows)
            part = {k: (v[offset:offset + n] if k in per_row else v) for k, v in outputs.items()}
            r.reply.resolve({"outputs": part, "per_row": per_row, "rows": n, "batch": total, "ms": round(ms, 3)})
            offset += n


# --------------------------------------------------------------------------------------------- the protocol

def _install_handlers():
    """The reserved calls Unreal uses, registered once per script (a model script is a PythonCLI service)."""
    ctx = _bridge.current_context()
    if ctx is None or "_torch_run" in ctx.handlers:
        return
    from . import _streams
    ctx.handlers["_torch_stream"] = [_streams.handle_frame]
    for name, fn in (("_torch_load", _torch_load), ("_torch_run", _torch_run), ("_torch_unload", _torch_unload),
                     ("_torch_signature", _torch_signature), ("_torch_list", _torch_list), ("_torch_stats", _torch_stats),
                     ("_torch_load_weights", _torch_load_weights)):
        ctx.handlers[name] = [fn]


def _torch_load(name, options=None):
    """Load with the asset's options. Already loaded with the same options: nothing to do (Unreal sends this
    again after a script restart); other options: reload, keeping the weights."""
    h = host(name)
    wanted = {k: v for k, v in (options or {}).items() if v not in (None, "")}
    h.load_error = None  # asked explicitly: try again
    if h.instance is None:
        try:
            h.load(wanted)
        except Exception as e:
            h.load_error = (time.monotonic(), f"{type(e).__name__}: {e}")
            raise
    elif wanted != h.overrides:
        h.load(wanted, state=h.instance.state_dict())
    return {"signature": h.signature(), "device": str(h.device), "load_ms": round(h.load_ms, 1)}


def _torch_run(name, method, inputs=None, rows=None, options=None):
    return host(name).run(method, inputs, rows, options)


def _torch_unload(name):
    with _lock:
        h = _hosts.get(name)
    if h is not None:
        h.unload()
    return True


def _torch_signature(name=None):
    if name:
        return host(name).signature()
    return [host(n).signature() for n in list(_classes)]


def _torch_load_weights(name, file):
    """Weights a job in another process published (Saved/Torch/Models/<model>/published.safetensors)."""
    h = host(name)
    if h.instance is None:
        return False
    h.instance.publish(_weights.load_state(file, h.device))
    return True


def _torch_list():
    ctx = _bridge.current_context()
    jobs = []
    for s in (ctx.job_specs.values() if ctx is not None else ()):
        d = s.describe()
        d["process"] = getattr(s.fn, "__ut_job_options__", {}).get("process", "shared")
        jobs.append(d)
    with _lock:
        names = list(_classes)
    return {"models": [host(n).signature() for n in names], "jobs": jobs}


def _torch_stats(name, reset=False):
    h = host(name)
    with h.stats_lock:
        out = dict(h.stats, publishes=h.publishes, load_ms=round(h.load_ms, 1), device=str(h.device) if h.device else "")
        if reset:
            h.stats.update(calls=0, rows=0, passes=0, max_batch=0, forward_ms=0.0)
    return out


# -------------------------------------------------------------------------------------------------- helpers

class _Stack:
    def __init__(self):
        self.items = []

    def enter(self, cm):
        cm.__enter__()
        self.items.append(cm)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        for cm in reversed(self.items):
            cm.__exit__(*exc)
        return False


def _resolve_device(name):
    name = (name or "auto").lower()
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return torch.device("xpu")
        return torch.device("cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        ue.warning(f"device '{name}' asked for, but CUDA isn't available: using the CPU")
        return torch.device("cpu")
    return torch.device(name)


def _carry_state(inst, state):
    """Load the old weights where names and shapes still match."""
    own = inst.state_dict()
    keep = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    inst.load_state_dict(keep, strict=False)
    return len(keep)


def _outputs(result, total):
    """A method's return value -> ({name: value}, [names split per row])."""
    if isinstance(result, dict):
        outputs = dict(result)
    elif isinstance(result, (tuple, list)) and result and all(isinstance(v, (torch.Tensor, np.ndarray)) for v in result):
        outputs = {f"output{i}": v for i, v in enumerate(result)}
    else:
        outputs = {"output": result}
    per_row = []
    if total is not None:
        for k, v in outputs.items():
            if isinstance(v, (torch.Tensor, np.ndarray)) and v.ndim >= 1 and v.shape[0] == total:
                per_row.append(k)
            elif isinstance(v, list) and len(v) == total:
                per_row.append(k)
    return outputs, per_row


def _stack_rows(rows):
    """Per-row results of an unbatched method -> stacked outputs (all split per row)."""
    if not rows:
        return {}, []
    out = {}
    for k in rows[0]:
        vals = [r.get(k) for r in rows]
        if all(isinstance(v, torch.Tensor) for v in vals) and len({tuple(v.shape) for v in vals}) == 1:
            out[k] = torch.stack(vals)
        elif all(isinstance(v, np.ndarray) for v in vals) and len({v.shape for v in vals}) == 1:
            out[k] = np.stack(vals)
        else:
            out[k] = [_to_unreal(v) for v in vals]
    return out, list(out)


def _to_unreal(v):
    """Tensors -> numpy (float16/bfloat16 become float32), 0-d values -> Python scalars."""
    if isinstance(v, torch.Tensor):
        v = v.detach()
        if v.dtype in (torch.bfloat16, torch.float16):
            v = v.float()
        if v.ndim == 0:
            return v.item()
        return v.cpu().numpy()
    if isinstance(v, np.ndarray):
        if v.dtype == np.float16:
            v = v.astype(np.float32)
        return v.item() if v.ndim == 0 else v
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {k: _to_unreal(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_unreal(x) for x in v]
    return v
