"""Vision examples: pretrained torchvision models on what a camera sees.

    "imagenet"   classify(image) -> label, confidence, text ("tabby cat (87%)"), top5
                 MobileNetV3-Large, ImageNet-1k (22 MB, downloaded once)
    "segmenter"  overlay(image) -> image ((H, W, 4) uint8 RGBA: the frame with each region tinted by its class), classes
                 LR-ASPP MobileNetV3-Large, 21 Pascal VOC classes (13 MB): an image-to-image effect

Use them with a Torch Output Component on a Python Render Target Stream (no script of your own), or Run Model.
Weights download to TORCH_HOME (Saved/Torch/Hub by default). Both compile with CUDA graphs (torch.compile
mode="reduce-overhead", triton-windows on Windows): the first frames wait for a one-time compile.
"""
import torch
import torch.nn.functional as F
import unreal_torch as ut


@ut.model("imagenet", compile="reduce-overhead")  # CUDA graphs: ~1.6 ms instead of ~10 ms (launch-bound) per frame
class ImageNet(ut.Model):
    def setup(self):
        from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
        self.net = mobilenet_v3_large(weights=weights, progress=False)
        self.labels = weights.meta["categories"]

    @ut.export(batch=True)
    def classify(self, image: ut.Image(size=(224, 224), normalize="imagenet")) -> dict:
        p = self.net(image).softmax(-1)
        top = p.topk(5, -1)
        labels = [self.labels[i] for i in top.indices[:, 0].tolist()]
        conf = top.values[:, 0].tolist()
        return {"label": labels, "confidence": top.values[:, 0], "text": [f"{l} ({c:.0%})" for l, c in zip(labels, conf)],
                "top5": top.indices.to(torch.int32)}


VOC = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "dining table", "dog",
       "horse", "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv/monitor"]


@ut.model("segmenter", compile="reduce-overhead")  # ~2.8 ms instead of ~10 ms at 512x512
class Segmenter(ut.Model):
    def setup(self):
        from torchvision.models.segmentation import LRASPP_MobileNet_V3_Large_Weights, lraspp_mobilenet_v3_large
        self.net = lraspp_mobilenet_v3_large(weights=LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1, progress=False)
        g = torch.Generator().manual_seed(3)
        palette = torch.rand(len(VOC), 3, generator=g, device="cpu") * 0.8 + 0.2  # setup() runs with the model's device as the default
        palette[0] = 0  # background: no tint
        self.palette = palette

    @ut.export(batch=True)
    def overlay(self, image: ut.Image(normalize=None)) -> dict:
        """The frame (N, 3, H, W) in 0..1 -> (N, H, W, 4) uint8 RGBA with each region tinted by its class color
        (video-friendly: a quarter of float RGBA, written to render targets as is); and the classes found (more
        than 0.2% of the frame)."""
        mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)
        logits = self.net((image - mean) / std)["out"]
        classes = logits.argmax(1)  # (N, H, W)
        color = self.palette.to(image.device)[classes].permute(0, 3, 1, 2)  # (N, 3, H, W)
        tinted = (classes > 0).unsqueeze(1)
        out = torch.where(tinted, image * 0.45 + color * 0.55, image * 0.6)
        out = torch.cat([out, torch.ones_like(out[:, :1])], 1).mul(255).round().to(torch.uint8).permute(0, 2, 3, 1)
        found = []
        for c in classes:
            counts = torch.bincount(c.flatten(), minlength=len(VOC)).float() / c.numel()
            found.append(", ".join(VOC[i] for i in (counts > 0.002).nonzero().flatten().tolist() if i > 0) or "nothing")
        return {"image": out, "classes": found}
