"""ONNX export for Unreal's NNE: shipping builds run a model's exported methods without Python.

    ut.export.onnx("reach_policy", "act")              # a live model's exported method (batch=True)
    ut.export.onnx(net, inputs={"obs": (4,)})          # any nn.Module: forward's tensor inputs, per-row shapes

In the editor, the Torch Model asset's Export To NNE does this for its batched methods and imports the files as
NNE model assets: Run Model then uses NNE where Python isn't there (or always, Backend: NNE).

What's written (to Saved/Torch/Export/<model>/ unless out_dir):
    <method>.onnx               dynamic batch (opset 20, dynamo exporter): ORT CPU
    <method>.b<N>.onnx          static batch N (static_batch=N): DirectML is fast only with static shapes
    <method>.json               the inputs (images: how Unreal converts them), outputs, files
    <method>.parity.safetensors inputs and torch's outputs for 3 rows: Unreal checks NNE against them

A method exports when what it does with its tensors traces (torch.export): Python-side logic on values (.item(),
.tolist(), strings) doesn't. Outputs that aren't per-row tensors (text, JSON values) are left out. ut.Image inputs
export as the converted image (e.g. float (N, 1, 28, 28)); Unreal converts textures and render targets the same
way before NNE runs.
"""
import copy
import json
import os
import threading
import time

import torch
import torch.nn as nn
import unreal_bridge as ue

from . import _model, _types

__all__ = ["onnx", "OPSET"]

OPSET = 20  # the highest ORT DirectML supports
_lock = threading.Lock()  # torch.export isn't thread-safe: one export at a time per process (scripts share it)


class _MethodModule(nn.Module):
    """An exported method as a module: positional tensors in, a tuple of tensors out."""

    def __init__(self, inst, method, names, outputs):
        super().__init__()
        for attr, m in inst.modules().items():
            self.add_module(attr, m)
        self._inst = inst
        self._fn = method.fn
        self._names = list(names)
        self._outputs = list(outputs)

    def forward(self, *xs):
        result = self._fn(self._inst, **dict(zip(self._names, xs)))
        outputs, _ = _model._outputs(result, None)
        return tuple(outputs[k] for k in self._outputs)


class _ModuleWrapper(nn.Module):
    def __init__(self, module, outputs):
        super().__init__()
        self.module = module
        self._outputs = outputs

    def forward(self, *xs):
        result = self.module(*xs)
        outputs, _ = _model._outputs(result, None)
        return tuple(outputs[k] for k in self._outputs)


def _cpu_instance(inst):
    """A shallow copy of a model instance with CPU copies of its modules (the live one keeps serving)."""
    clone = copy.copy(inst)
    mods = {}
    for attr, m in inst.modules().items():
        c = copy.deepcopy(m).cpu().eval()
        setattr(clone, attr, c)
        mods[attr] = c
    clone._ut_modules = mods
    clone.device = torch.device("cpu")
    return clone


def _directml_fixups(path):
    """The dynamo exporter writes Reshape with allowzero=1, which DirectML rejects. With a constant shape that has
    no 0, allowzero changes nothing: drop it."""
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


def _describe_input(name, ann, shape):
    d = {"name": name, "kind": "tensor", "dtype": "float32", "shape": list(shape)}
    if isinstance(ann, _types.Image):
        d.update(kind="image", size=list(ann.size), gray=bool(ann.gray), normalize=ann.normalize or "", layout=ann.layout, alpha=bool(ann.alpha))
    elif isinstance(ann, _types.Tensor) and ann.dtype is not None:
        d["dtype"] = _types._dtype_name(ann.dtype)
    return d


def _dtype_name(t):
    return str(t.dtype).replace("torch.", "")


def _export_one(module, examples, path, input_names, output_names, dynamic):
    with _lock:
        _export_locked(module, examples, path, input_names, output_names, dynamic)


def _export_locked(module, examples, path, input_names, output_names, dynamic):
    kwargs = {}
    if dynamic:
        batch = torch.export.Dim("batch", min=1, max=65536)
        kwargs["dynamic_shapes"] = (tuple({0: batch} for _ in input_names),)  # forward(*xs): one positional tuple
    torch.onnx.export(module, tuple(examples), path, dynamo=True, opset_version=OPSET, input_names=list(input_names),
                      output_names=list(output_names), external_data=False, verbose=False, **kwargs)
    _directml_fixups(path)


def onnx(model, method=None, name=None, out_dir=None, static_batch=0, inputs=None, seed=0):
    """Export a model's batched method (or an nn.Module) to ONNX for NNE; returns the description (also
    <name>.json). model: a @ut.model name, a ut.Model instance, or an nn.Module (then inputs={"name": per-row
    shape}). static_batch: also write a static-batch file for DirectML (0: none)."""
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    model_name = None
    if isinstance(model, str):
        model_name = model
        model = _model.get_model(model)
    if isinstance(model, _model.Model):
        model_name = model_name or getattr(model, "name", None) or type(model).__name__
        host = _model.host(model_name)
        methods = host.methods
        if method is None:
            batched = [m for m in methods.values() if m.batch]
            if len(batched) != 1:
                raise ValueError(f"{model_name}: say which method to export ({', '.join(m.name for m in batched) or 'no batched methods'})")
            method = batched[0].name
        spec = methods.get(method)
        if spec is None:
            raise KeyError(f"{model_name} has no exported method '{method}'")
        if not spec.batch:
            raise ValueError(f"{model_name}.{method}: only batch=True methods export (NNE runs stacked rows)")
        names, anns, row_shapes = [], [], []
        for p in spec.params:
            if not p.is_tensor:
                if not p.has_default:
                    raise ValueError(f"{model_name}.{method}: input '{p.name}' isn't a tensor (NNE takes tensors only)")
                continue
            ex = p.annotation.example(1, torch.device("cpu"))
            if ex is None:
                raise ValueError(f"{model_name}.{method}: input '{p.name}' needs a full shape to export "
                                 f"(ut.Tensor(shape=...) without None, or ut.Image(size=...))")
            names.append(p.name)
            anns.append(p.annotation)
            row_shapes.append(tuple(ex.shape[1:]))
        inst = _cpu_instance(model)
        make = lambda outs: _MethodModule(inst, spec, names, outs)  # noqa: E731
        call = lambda xs: spec.fn(inst, **dict(zip(names, xs)))  # noqa: E731
        label = f"{model_name}.{method}"
    elif isinstance(model, nn.Module):
        if not inputs:
            raise ValueError("an nn.Module needs inputs={'name': per-row shape}")
        method = method or "forward"
        model_name = model_name or name or type(model).__name__
        names = list(inputs)
        anns = [None] * len(names)
        row_shapes = [tuple(s) for s in inputs.values()]
        module = copy.deepcopy(model).cpu().eval()
        make = lambda outs: _ModuleWrapper(module, outs)  # noqa: E731
        call = lambda xs: module(*xs)  # noqa: E731
        label = model_name
    else:
        raise TypeError("export.onnx takes a model name, a ut.Model or an nn.Module")

    name = name or method
    out_dir = out_dir or os.path.join(ue.project_dir or ".", "Saved", "Torch", "Export", model_name)
    os.makedirs(out_dir, exist_ok=True)

    def example(n, random=False):
        out = []
        for ann, shape in zip(anns, row_shapes):
            dt = ann.dtype if isinstance(ann, _types.Tensor) and ann.dtype is not None else torch.float32
            if dt.is_floating_point:
                out.append(torch.rand((n, *shape), dtype=dt) if random else torch.zeros((n, *shape), dtype=dt))
            else:
                out.append(torch.zeros((n, *shape), dtype=dt))
        return out

    # Which outputs: the method's per-row tensors (probe with 2 rows)
    with torch.inference_mode():
        result = call(example(2))
    outputs, _ = _model._outputs(result, 2)
    keep = [k for k, v in outputs.items() if isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == 2]
    dropped = [k for k in outputs if k not in keep]
    if not keep:
        raise ValueError(f"{label}: no per-row tensor outputs to export (outputs: {', '.join(outputs) or 'none'})")
    module = make(keep).eval()

    files = {}
    dynamic_path = os.path.join(out_dir, f"{name}.onnx")
    _export_one(module, example(2), dynamic_path, names, keep, dynamic=True)
    files["dynamic"] = dynamic_path
    if static_batch and int(static_batch) > 0:
        static_path = os.path.join(out_dir, f"{name}.b{int(static_batch)}.onnx")
        _export_one(module, example(int(static_batch)), static_path, names, keep, dynamic=False)
        files["static"] = static_path

    # Reference inputs and outputs for the parity check in Unreal
    xs = example(3, random=True)
    with torch.inference_mode():
        ys = module(*xs)
    from safetensors.torch import save_file
    parity = os.path.join(out_dir, f"{name}.parity.safetensors")
    save_file({**{f"in.{n}": x.contiguous() for n, x in zip(names, xs)}, **{f"out.{k}": y.contiguous() for k, y in zip(keep, ys)}}, parity)
    files["parity"] = parity

    desc = {"format": 1, "model": model_name, "method": method, "name": name, "opset": OPSET, "torch": torch.__version__,
            "inputs": [_describe_input(n, a, s) for n, a, s in zip(names, anns, row_shapes)],
            "outputs": [{"name": k, "dtype": _dtype_name(y), "shape": list(y.shape[1:])} for k, y in zip(keep, ys)],
            "dropped_outputs": dropped, "static_batch": int(static_batch or 0), "files": files,
            "export_s": round(time.perf_counter() - t0, 2)}
    with open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(desc, f, indent=1)
    return desc


def _torch_export(name, method=None, options=None):
    """Unreal (Torch Model > Export To NNE): export one method of a loaded model."""
    o = dict(options or {})
    return onnx(name, method, name=o.get("name"), out_dir=o.get("out_dir") or None, static_batch=int(o.get("static_batch") or 0))
