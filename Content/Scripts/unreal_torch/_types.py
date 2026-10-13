"""Input annotations: what an exported method's parameters are, and how Unreal's values become them.

    def classify(self, image: ut.Image(size=(28, 28), gray=True), top_k: int = 3): ...

Tensors arrive from Unreal with a leading batch dimension (one row per request); conversion runs on the
model's device. Plain annotations (int, float, str, bool, dict, list) are JSON values shared by the batch.
"""
import torch
import torch.nn.functional as F

__all__ = ["Tensor", "Image", "Audio", "InputSpec", "describe_annotation"]

_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16, "float64": torch.float64,
           "int64": torch.int64, "int32": torch.int32, "int16": torch.int16, "int8": torch.int8, "uint8": torch.uint8, "bool": torch.bool}


def _dtype(d):
    if d is None or isinstance(d, torch.dtype):
        return d
    return _DTYPES[str(d).replace("torch.", "")]


def _dtype_name(d):
    return str(d).replace("torch.", "") if d is not None else None


class Tensor:
    """A tensor input, checked and cast: ut.Tensor(dtype="float32", shape=(None, 64)). Shape is per row (no
    batch dimension); None matches any size."""

    kind = "tensor"

    def __init__(self, dtype=None, shape=None):
        self.dtype = _dtype(dtype)
        self.shape = tuple(shape) if shape is not None else None

    def convert(self, x, device, layout=None):
        x = _to_torch(x, device)
        if self.shape is not None:
            got = tuple(x.shape[1:])
            if len(got) != len(self.shape) or any(s is not None and s != g for s, g in zip(self.shape, got)):
                raise ValueError(f"expected rows of shape {self.shape}, got {got}")
        return x.to(self.dtype) if self.dtype is not None else x

    def describe(self):
        return {"kind": self.kind, "dtype": _dtype_name(self.dtype) or "", "shape": [(-1 if s is None else s) for s in (self.shape or ())]}

    def example(self, n, device):
        if self.shape is None or None in self.shape:
            return None
        return torch.zeros((n, *self.shape), dtype=self.dtype or torch.float32, device=device)


class Image(Tensor):
    """An image input as float NCHW (or NHWC) on the model's device. Accepts what Unreal sends: (N, H, W, C) uint8
    (render target streams, BGRA or RGBA) or float 0..1 (Texture To Tensor), or (N, C, H, W) float.

        size       (H, W) to resize to (bilinear, antialiased), or None
        gray       one channel ((R + G + B) / 3)
        normalize  "unit" (0..1, default), "symmetric" (-1..1), "imagenet" ((x - mean) / std), or None (as is)
        layout     "NCHW" (default) or "NHWC"
        alpha      keep a 4th channel
    """

    kind = "image"
    _MEAN = (0.485, 0.456, 0.406)
    _STD = (0.229, 0.224, 0.225)

    def __init__(self, size=None, gray=False, normalize="unit", layout="NCHW", alpha=False, dtype=torch.float32):
        super().__init__(dtype=dtype)
        self.size = tuple(size) if size is not None else None
        self.gray = gray
        self.normalize = normalize
        self.layout = layout.upper()
        self.alpha = alpha

    def convert(self, x, device, layout=None):
        x = _to_torch(x, device)
        if x.dim() == 3:  # one image without a batch dimension
            x = x.unsqueeze(0)
        if x.dim() != 4:
            raise ValueError(f"expected images (N, H, W, C) or (N, C, H, W), got shape {tuple(x.shape)}")
        hint = (layout or "").upper()
        if hint in ("BGRA", "RGBA", "BGR", "RGB", "R", "NHWC"):
            channels_last = True
        elif hint == "NCHW":
            channels_last = False
        else:  # Unreal's images are (H, W, C); torch's (C, H, W)
            channels_last = x.shape[-1] in (1, 3, 4) and (x.shape[1] not in (1, 3, 4) or x.dtype == torch.uint8)
        if channels_last:
            x = x.permute(0, 3, 1, 2)
        scale = x.dtype == torch.uint8
        x = x.float()
        if scale:
            x = x / 255.0
        c = x.shape[1]
        if c == 4 or c == 3:
            order = [2, 1, 0] if (layout or "").upper().startswith("BGR") else [0, 1, 2]
            rgb = x[:, order]
            a = x[:, 3:4] if c == 4 else None
        else:
            rgb, a = x.expand(-1, 3, -1, -1) if not self.gray else x, None
        if self.gray:
            rgb = rgb.mean(1, keepdim=True) if rgb.shape[1] == 3 else rgb
        x = torch.cat([rgb, a], 1) if (self.alpha and a is not None) else rgb
        if self.size is not None and tuple(x.shape[-2:]) != self.size:
            x = F.interpolate(x, size=self.size, mode="bilinear", align_corners=False, antialias=True)
        if self.normalize == "symmetric":
            x = x * 2 - 1
        elif self.normalize == "imagenet":
            n = x.shape[1]
            mean = torch.tensor(self._MEAN if n >= 3 else (sum(self._MEAN) / 3,), device=x.device).view(1, -1, 1, 1)
            std = torch.tensor(self._STD if n >= 3 else (sum(self._STD) / 3,), device=x.device).view(1, -1, 1, 1)
            x = torch.cat([(x[:, :3] - mean[:, :3]) / std[:, :3], x[:, 3:]], 1) if n >= 3 else (x - mean) / std
        if self.layout == "NHWC":
            x = x.permute(0, 2, 3, 1)
        return x.to(self.dtype).contiguous()

    def describe(self):
        return {"kind": self.kind, "dtype": _dtype_name(self.dtype), "size": list(self.size) if self.size else [],
                "gray": self.gray, "normalize": self.normalize or "", "layout": self.layout}

    def example(self, n, device):
        if self.size is None:
            return None
        c = 1 if self.gray else (4 if self.alpha else 3)
        shape = (n, c, *self.size) if self.layout == "NCHW" else (n, *self.size, c)
        return torch.zeros(shape, dtype=self.dtype, device=device)


class Audio(Tensor):
    """Audio as float32 (N, samples) (mono) or (N, channels, samples), resampled to rate. Unreal sends
    (N, frames, channels) float32 chunks (audio streams) with their sample rate in the request's meta."""

    kind = "audio"

    def __init__(self, rate=16000, mono=True):
        super().__init__(dtype=torch.float32)
        self.rate = rate
        self.mono = mono

    def convert(self, x, device, layout=None, rate=None):
        x = _to_torch(x, device).float()
        if x.dim() == 2:  # (N, samples): mono already
            x = x.unsqueeze(-1)
        x = x.transpose(1, 2)  # (N, channels, frames)
        if self.mono:
            x = x.mean(1, keepdim=True)
        if rate and self.rate and int(rate) != int(self.rate):
            n = max(1, round(x.shape[-1] * self.rate / float(rate)))
            x = F.interpolate(x, size=n, mode="linear", align_corners=False)
        return x.squeeze(1) if self.mono else x

    def describe(self):
        return {"kind": self.kind, "dtype": "float32", "rate": self.rate, "mono": self.mono}


class InputSpec:
    """One parameter of an exported method."""

    def __init__(self, name, annotation, default, has_default):
        self.name = name
        self.default = default
        self.has_default = has_default
        if isinstance(annotation, type) and issubclass(annotation, Tensor):
            annotation = annotation()
        if annotation is torch.Tensor:
            annotation = Tensor()
        self.annotation = annotation
        self.is_tensor = isinstance(annotation, Tensor)

    def describe(self):
        d = {"name": self.name, "required": not self.has_default}
        d.update(describe_annotation(self.annotation))
        return d


def describe_annotation(a):
    if isinstance(a, Tensor):
        return a.describe()
    if a is None or a is type(None):
        return {"kind": "json", "type": ""}
    return {"kind": "json", "type": getattr(a, "__name__", str(a))}


def _to_torch(x, device):
    if isinstance(x, torch.Tensor):
        t = x
    else:
        import numpy as np
        a = np.asarray(x)
        if not a.flags.writeable:
            a = a.copy()  # frames from Unreal can be read-only views
        t = torch.from_numpy(a)
    if device is not None and t.device != device:
        nonblocking = t.device.type == "cpu" and t.is_pinned()
        t = t.to(device, non_blocking=nonblocking)
    return t
