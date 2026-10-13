"""unreal_torch: PyTorch models and training jobs for Unreal (PyTorch-Unreal), on PythonCLI's unreal_bridge.

    import torch
    import unreal_torch as ut

    @ut.model("digits")                                   # the Torch Model asset's Model Name
    class Digits(ut.Model):
        def setup(self):                                  # once, on the model's device
            self.net = DigitNet()
            self.load_weights(missing_ok=True)            # Saved/Torch/Models/digits/digits.safetensors

        @ut.export(batch=True)                            # concurrent requests run as one forward pass
        def classify(self, image: ut.Image(size=(28, 28), gray=True)) -> dict:
            p = self.net(image).softmax(-1)
            return {"digit": p.argmax(-1), "confidence": p.amax(-1)}   # split back per request

    @ut.job("train_digits")                               # Unreal: Start Job on the Torch Model
    def train(job: ut.Job, epochs: int = 2):
        for epoch in job.range(epochs):                   # progress, stop, resume
            ...
            job.metrics(epoch=epoch, accuracy=acc)
            job.checkpoint(net=net, opt=opt)
            job.publish("digits", net)                    # into the live model

In Unreal: a Torch Model asset (Script, Model Name) and the Run Model node, or UTorchSubsystem::Run in C++.
One instance per process serves every caller; requests made in the same tick arrive stacked on dim 0.
"""
from ._jobs import TorchJob as Job, data_dir, job
from ._model import Model, export, get_model, model, signature
from ._types import Audio, Image, Tensor
from . import _weights as weights
from . import data, rl

__all__ = ["Model", "model", "export", "get_model", "signature", "Tensor", "Image", "Audio", "weights", "job", "Job", "data_dir",
           "data", "rl"]
__version__ = "0.1.0"
