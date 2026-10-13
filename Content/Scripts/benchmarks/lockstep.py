"""PyTorch.Benchmarks.Lockstep: a Lockstep stream as an RL environment step. Each step carries every agent's
observations as one (agents, obs) float32 frame; the handler runs the policy and returns (agents, actions).

    args: [device]   "cuda", "cpu", or "none" (no policy: transport only, returns zeros)
"""
import numpy as np
import torch
import torch.nn as nn
import unreal_bridge as ue

mode = (ue.args[0] if ue.args else "cuda").lower()
if mode == "cuda" and not torch.cuda.is_available():
    mode = "cpu"
OBS, ACT = 64, 8


class Policy(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(OBS, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, ACT), nn.Tanh())

    def forward(self, x):
        return self.net(x)


torch.manual_seed(0)
policy = None if mode == "none" else Policy().to(mode).eval()
zeros = {}


@ue.on_stream("agents")
def step(frame):
    obs = frame.array  # (agents, OBS) float32, a view of the shared-memory slot
    if policy is None:
        z = zeros.get(obs.shape[0])
        if z is None:
            z = zeros[obs.shape[0]] = np.zeros((obs.shape[0], ACT), np.float32)
        return z
    with torch.inference_mode():
        x = torch.from_numpy(obs).to(mode, non_blocking=True)
        return policy(x).cpu().numpy()


def info():
    return {"mode": mode, "torch": torch.__version__}
