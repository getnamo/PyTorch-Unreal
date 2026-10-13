"""Fixture models for the unreal_torch tests (tests/torch_harness.py and PyTorch.* automation tests)."""
import time

import torch
import torch.nn as nn
import unreal_bridge as ue
import unreal_torch as ut

VERSION = 1  # the reload test rewrites this


@ut.model("doubler")
class Doubler(ut.Model):
    """x * scale, batched; slow enough that concurrent calls meet in the batcher."""

    def setup(self):
        self.scale = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            self.scale.weight.fill_(2.0)
        self.delay = 0.0

    @ut.export(batch=True, max_wait_ms=5.0)
    def double(self, x: torch.Tensor):
        if self.delay:
            time.sleep(self.delay)
        return self.scale.weight[0, 0] * x

    @ut.export
    def set_delay(self, seconds: float):
        self.delay = seconds
        return True


@ut.model("rowwise")
class Rowwise(ut.Model):
    """An unbatched method: called once per request, with that request's values (no batch dimension)."""

    @ut.export
    def stats(self, x: ut.Tensor(dtype="float32", shape=(None,)), label: str = "") -> dict:
        return {"sum": x.sum(), "n": x.numel(), "label": label}


@ut.model("imager")
class Imager(ut.Model):
    @ut.export(batch=True)
    def gray(self, image: ut.Image(size=(4, 4), gray=True)):
        return {"mean": image.mean((1, 2, 3)), "shape": list(image.shape[1:])}

    @ut.export(batch=True)
    def imagenet(self, image: ut.Image(normalize="imagenet")):
        return image[:, :, 0, 0]

    @ut.export(batch=True)
    def invert(self, image: ut.Image(normalize=None, alpha=True)):
        """Image to image: (N, 4, H, W) RGBA, colors inverted."""
        return {"image": torch.cat([1 - image[:, :3], image[:, 3:]], 1), "mean": image[:, :3].mean((1, 2, 3))}


@ut.model("counter", device="cpu")
class Counter(ut.Model):
    """Weights survive a script reload (reload='keep')."""

    def setup(self):
        self.p = nn.Linear(2, 2)

    @ut.export
    def bump(self, amount: float = 1.0):
        with torch.no_grad():
            self.p.bias += amount
        return float(self.p.bias[0])

    @ut.export
    def version(self):
        return VERSION


@ut.model("sized")
class Sized(ut.Model):
    """Warmup runs exported methods on example inputs from fixed-size annotations."""

    def setup(self):
        self.warmed = 0

    @ut.export(batch=True)
    def f(self, x: ut.Tensor(shape=(3,))):
        self.warmed += 1
        return x.sum(-1)

    @ut.export
    def warm_count(self):
        return self.warmed


@ut.model("faulty")
class Faulty(ut.Model):
    @ut.export(batch=True)
    def boom(self, x: torch.Tensor):
        raise ValueError("model failed on purpose")


@ut.model("compiled")
class Compiled(ut.Model):
    """torch.compile (triton-windows on CUDA under Windows), with batches padded to powers of two."""

    def setup(self):
        torch.manual_seed(0)
        self.net = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 2))

    @ut.export(batch=True)
    def f(self, x: ut.Tensor(shape=(4,))):
        return self.net(x)

    @ut.export
    def reference(self, x: ut.Tensor(shape=(4,))):
        return self.net._orig_mod(x) if hasattr(self.net, "_orig_mod") else self.net(x)


@ut.job("rl_probe")
def rl_probe(job, steps: int = 2):
    """An UnrealVectorEnv("Probe") driven by the harness's frames: what the env saw, for the checks."""
    import numpy as np
    from unreal_torch import rl
    env = rl.UnrealVectorEnv("Probe", timeout=30, job=job)
    obs, _ = env.reset()
    seen = {"num_envs": env.num_envs, "obs_size": env.obs_size, "action_space": repr(env.single_action_space),
            "batched_action_shape": list(env.action_space.shape), "gymnasium": rl._gym is not None, "first_obs": obs.tolist()}
    rewards, finals, dones = [], [], []
    for _ in range(steps):
        obs, r, term, trunc, info = env.step(np.full((env.num_envs, env.action_size), 0.5, np.float32))
        rewards.append(r.tolist())
        dones.append([bool(t) for t in term | trunc])
        if "final_obs" in info:
            finals.append([None if x is None else x.tolist() for x in info["final_obs"]])
    obs, _ = env.reset(seed=5)
    seen.update(rewards=rewards, dones=dones, finals=finals, reset_obs=obs.tolist())
    env.close()
    return seen
