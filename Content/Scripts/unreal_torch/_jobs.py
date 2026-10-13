"""Training jobs on torch: PythonCLI's jobs (progress, metrics, run folders, resume, stop) plus checkpoints,
TensorBoard and publishing weights into live models.

    @ut.job("train_digits")
    def train(job, epochs: int = 2, lr: float = 1e-3):
        live = ut.get_model("digits")                 # the instance the game is using
        net = copy.deepcopy(live.net).train()
        opt = torch.optim.AdamW(net.parameters(), lr)
        job.restore(net=net, opt=opt)                 # on resume: the last checkpoint
        for epoch in job.range(epochs):
            for x, y in job.track(loader):
                ...
            job.metrics(epoch=epoch, loss=loss, accuracy=acc)   # also to TensorBoard when installed
            job.checkpoint(net=net, opt=opt)          # <run>/checkpoint.pt
            job.publish("digits", net)                # the game sees the new weights from its next batch
        return {"accuracy": acc}
"""
import functools
import os

import torch
import unreal_bridge as ue
from unreal_bridge import _host as _bridge

from . import _model, _weights

__all__ = ["job", "TorchJob", "data_dir"]


def data_dir(name=""):
    """<Project>/Saved/Torch/Data[/<name>]: where jobs keep datasets."""
    path = os.path.join(ue.project_dir or ".", "Saved", "Torch", "Data", name)
    os.makedirs(path, exist_ok=True)
    return path


def _process_name():
    return (_bridge._host.config.get("process") if _bridge._host is not None else "") or "shared"


class TorchJob:
    """The job object of a @ut.job function: everything of unreal_bridge's Job, plus the torch extras."""

    def __init__(self, job, options):
        self._job = job
        self._options = options
        self._writer = None
        self._tb_checked = False

    def __getattr__(self, name):
        return getattr(self._job, name)

    @property
    def device(self):
        """CUDA if available, else the CPU."""
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -- checkpoints ---------------------------------------------------------------------------------
    @property
    def checkpoint_path(self):
        return os.path.join(self._job.run_dir, "checkpoint.pt")

    def checkpoint(self, **objects):
        """Save the state of modules, optimizers, schedulers, scalers (anything with state_dict) to the run's
        checkpoint.pt. restore() on a resumed run loads it back."""
        state = {k: (v.state_dict() if hasattr(v, "state_dict") else v) for k, v in objects.items()}
        tmp = self.checkpoint_path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, self.checkpoint_path)
        self._job.save_state(checkpoint=True)

    def restore(self, **objects):
        """On a resumed run with a checkpoint: load it into the given objects (by name). True if it did."""
        if not self._job.resumed or not os.path.exists(self.checkpoint_path):
            return False
        state = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        for k, v in objects.items():
            if k in state and hasattr(v, "load_state_dict"):
                v.load_state_dict(state[k])
        self._job.log(f"restored {', '.join(k for k in objects if k in state)} from {self.checkpoint_path}")
        return True

    # -- metrics -------------------------------------------------------------------------------------
    def metrics(self, step=None, **values):
        self._job.metrics(step=step, **values)
        writer = self._tensorboard()
        if writer is not None:
            for k, v in values.items():
                try:
                    writer.add_scalar(k, float(v.item() if hasattr(v, "item") else v), step if step is not None else 0)
                except (TypeError, ValueError):
                    pass
            writer.flush()

    def _tensorboard(self):
        if not self._tb_checked:
            self._tb_checked = True
            if self._options.get("tensorboard", True):
                try:
                    from torch.utils.tensorboard import SummaryWriter
                    self._writer = SummaryWriter(os.path.join(self._job.run_dir, "tensorboard"))
                except Exception:  # noqa: BLE001 - tensorboard isn't installed: metrics.jsonl only
                    self._writer = None
        return self._writer

    # -- publishing ----------------------------------------------------------------------------------
    def publish(self, model_name, source):
        """Swap new weights into the live model `model_name` (a module, a ut.Model or a state dict). In the shared
        process that's immediate; from a job in its own process, the weights go through
        Saved/Torch/Models/<model>/published.safetensors and Unreal has the model's process load them."""
        if _process_name() == "shared":
            _model.get_model(model_name).publish(source)
            return
        if isinstance(source, _model.Model):
            state = source.state_dict()
        elif isinstance(source, torch.nn.Module):
            h = _model.host(model_name)
            attr = next(iter(h.instance.modules())) if h.instance is not None else "net"
            state = {f"{attr}.{k}": v for k, v in source.state_dict().items()}
        else:
            state = source
        file = _weights.resolve("published.safetensors", model_name, for_write=True)
        _weights.save_state(state, file)
        ctx = _bridge.current_context()
        ctx.host.emit(ctx, "_torch_publish", [{"model": model_name, "file": file}])

    def close(self):
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def job(name=None, **options):
    """@ut.job("name", process="shared"|"dedicated", tensorboard=True): a training (or any) job Unreal starts with
    Start Job on a Torch Model. The function gets a TorchJob; typed parameters come from Unreal's JSON.
    process="dedicated" runs it in its own Python process (no shared GIL with inference); publish() then goes
    through a weights file."""
    def register(fn, job_name=None):
        @functools.wraps(fn)
        def run(job, *args, **kwargs):
            tj = TorchJob(job, options)
            try:
                return fn(tj, *args, **kwargs)
            finally:
                tj.close()
        run.__ut_job_options__ = dict(options)
        ue.job(job_name or fn.__name__)(run)
        return fn
    if callable(name):
        return register(name)
    return lambda fn: register(fn, name)
