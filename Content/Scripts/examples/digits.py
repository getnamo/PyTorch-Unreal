"""Digits: an MNIST classifier as a Torch Model (Model Name "digits").

    Run Model (Digits, "classify", {"image": <28x28 or larger image tensor>}) -> digit, confidence, probabilities

Images are light digits on a dark background, like MNIST; ut.Image converts any size and channel layout to
gray 28x28 on the model's device. The weights come from Saved/Torch/Models/digits/digits.safetensors (written
by the train_digits job) or Content/Models/digits.safetensors; without them the network is untrained.
"""
import torch
import torch.nn as nn
import unreal_torch as ut


class DigitNet(nn.Module):
    """Two conv blocks and a small classifier: ~99% on MNIST after one epoch."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.25), nn.Linear(128, 10))

    def forward(self, x):
        x = (x - 0.1307) / 0.3081  # MNIST's mean and std
        return self.classifier(self.features(x))


@ut.model("digits")
class Digits(ut.Model):
    def setup(self):
        self.net = DigitNet()
        self.trained = self.load_weights(missing_ok=True)

    @ut.export(batch=True)
    def classify(self, image: ut.Image(size=(28, 28), gray=True)) -> dict:
        """The digit in each image, with its probability and all ten probabilities."""
        p = self.net(image).softmax(-1)
        return {"digit": p.argmax(-1), "confidence": p.amax(-1), "probabilities": p}

    @ut.export
    def info(self) -> dict:
        return {"trained": bool(self.trained), "device": str(self.device), "parameters": sum(p.numel() for p in self.net.parameters())}
