"""Digits: an MNIST classifier as a Torch Model (Model Name "digits"), and the job that trains it.

    Run Model (Digits, "classify", {"image": <image tensor>})  -> digit, confidence, probabilities
    Run Model (Digits, "read", {"image": ...})                 -> digit, text ("Camera sees: 3 (98%)")
    Run Model (Digits, "sample")                               -> image (256x256 gray), label, text: an MNIST test digit
    Start Job (Digits, "train_digits", {"epochs": 2})          -> trains on the GPU, publishing weights as it goes

Images are light digits on a dark background, like MNIST; ut.Image converts any size and channel layout to gray
28x28 on the model's device. Weights: Saved/Torch/Models/digits/digits.safetensors (saved by the job when it
finishes) or Content/Models/digits.safetensors; without them the network is untrained. MNIST is downloaded to
Saved/Torch/Data/ on first use (torchvision).
"""
import copy
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import unreal_torch as ut


class DigitNet(nn.Module):
    """Two conv blocks and a small classifier: ~99% on MNIST after two epochs."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.25), nn.Linear(128, 10))

    def forward(self, x):
        x = (x - 0.1307) / 0.3081  # MNIST's mean and std
        return self.classifier(self.features(x))


def mnist(train):
    """MNIST as (images float N x 1 x 28 x 28 in 0..1, labels), downloaded once."""
    import torchvision
    ds = torchvision.datasets.MNIST(ut.data_dir(), train=train, download=True)
    return ds.data.unsqueeze(1).float().div_(255), ds.targets


@ut.model("digits")
class Digits(ut.Model):
    def setup(self):
        self.net = DigitNet()
        self.trained = self.load_weights(missing_ok=True)
        self.test = None

    @ut.export(batch=True)
    def classify(self, image: ut.Image(size=(28, 28), gray=True)) -> dict:
        """The digit in each image, with its probability and all ten probabilities."""
        p = self.net(image).softmax(-1)
        return {"digit": p.argmax(-1), "confidence": p.amax(-1), "probabilities": p}

    @ut.export(batch=True)
    def read(self, image: ut.Image(size=(28, 28), gray=True)) -> dict:
        """classify, with a display line per image ("Camera sees: 3 (98%)"; nothing drawn: "Camera sees: -")."""
        p = self.net(image).softmax(-1)
        digit, conf = p.argmax(-1).tolist(), p.amax(-1).tolist()
        blank = (image.amax((1, 2, 3)) < 0.05).tolist()
        text = ["Camera sees: -" if b else f"Camera sees: {d} ({c:.0%})" for d, c, b in zip(digit, conf, blank)]
        return {"digit": p.argmax(-1), "confidence": p.amax(-1), "text": text}

    @ut.export
    def sample(self, index: int = -1, size: int = 256) -> dict:
        """An MNIST test image (random for index -1), scaled to size x size for display."""
        if self.test is None:
            self.test = mnist(train=False)
        images, labels = self.test
        i = random.randrange(len(images)) if index < 0 else index % len(images)
        img = F.interpolate(images[i:i + 1], size=(size, size), mode="nearest")[0, 0]
        return {"image": img, "label": int(labels[i]), "text": f"Drawn: test image {i}, a {int(labels[i])}"}

    @ut.export
    def info(self) -> dict:
        return {"trained": bool(self.trained), "device": str(self.device), "parameters": sum(p.numel() for p in self.net.parameters())}


@ut.job("train_digits")
def train_digits(job: ut.Job, epochs: int = 2, lr: float = 2e-3, batch_size: int = 128, publish_every: int = 50, max_steps: int = 0,
                 fresh: bool = False):
    """Train the live model on MNIST (the GPU when there is one), publishing the weights every publish_every steps
    so the game's predictions improve while it trains. fresh: start from random weights instead of the live ones.
    max_steps > 0 caps the steps per epoch (tests)."""
    device = job.device
    job.progress(0, "Loading MNIST...", force=True)
    x, y = (t.to(device) for t in mnist(train=True))
    xt, yt = (t.to(device) for t in mnist(train=False))
    live = ut.get_model("digits")
    net = (DigitNet() if fresh else copy.deepcopy(live.net)).to(device).train()
    opt = torch.optim.AdamW(net.parameters(), lr)
    if job.restore(net=net, opt=opt) or fresh:
        job.publish("digits", net)
    steps = len(x) // batch_size
    if max_steps > 0:
        steps = min(steps, max_steps)
    accuracy = 0.0
    for epoch in job.range(epochs):
        perm = torch.randperm(len(x), device=device)
        net.train()
        for step in job.track(range(steps)):
            idx = perm[step * batch_size:(step + 1) * batch_size]
            loss = F.cross_entropy(net(x[idx]), y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % publish_every == 0:
                job.publish("digits", net)
                job.progress(message=f"Training: epoch {epoch + 1}/{epochs}, step {step}/{steps}, loss {loss.item():.3f}")
        accuracy = evaluate(net, xt, yt)
        job.metrics(step=epoch, epoch=epoch + 1, loss=loss.item(), accuracy=accuracy)
        job.checkpoint(net=net, opt=opt)
        job.publish("digits", net)
        job.progress(message=f"Epoch {epoch + 1}/{epochs} done: test accuracy {accuracy:.1%}", force=True)
    stopped = job.should_stop
    accuracy = evaluate(net, xt, yt)  # also when a resumed run had no epochs left
    if not stopped:
        live.trained = True
        live.save_weights()
    return {"accuracy": accuracy, "text": f"{'Stopped' if stopped else 'Trained'}: test accuracy {accuracy:.1%}"}


@torch.no_grad()
def evaluate(net, x, y):
    net.eval()
    correct = sum(int((net(x[i:i + 2000]).argmax(-1) == y[i:i + 2000]).sum()) for i in range(0, len(x), 2000))
    net.train()
    return correct / len(x)
