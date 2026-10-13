"""Datasets of what Unreal recorded.

    from unreal_torch.data import Recording

    demos = Recording("Reach")                       # every session in Saved/Torch/Recordings/Reach/
    loader = torch.utils.data.DataLoader(demos, batch_size=256, shuffle=True)
    for obs, actions in loader:                      # behavior cloning: imitate the recorded actions
        ...

An ATorchEnvironment in Record mode writes one row per agent and decision step, a transition: obs, actions,
rewards, terminated, truncated, next_obs (the episode's final observation where it ended) and agent (the agent's
index), in safetensors shards: Saved/Torch/Recordings/<Recording Name>/<session timestamp>/shard_00000.safetensors.
"""
import glob
import json
import os

import torch
import unreal_bridge as ue

__all__ = ["Recording", "recordings_dir"]

FIELDS = ("obs", "actions", "rewards", "terminated", "truncated", "next_obs", "agent")


def recordings_dir(name=""):
    """<Project>/Saved/Torch/Recordings[/<name>]."""
    return os.path.join(ue.project_dir or ".", "Saved", "Torch", "Recordings", name)


class Recording(torch.utils.data.Dataset):
    """The transitions of a recording: all its sessions, or the given ones (folder names, or "latest").

    recording[i] is (obs, actions); the tensors are attributes (obs, actions, rewards, terminated, truncated,
    next_obs, agent, session), on `device` (the CPU by default). meta: the environment's spec (env, obs_size,
    action_size, action_type, agents, step_seconds)."""

    def __init__(self, name, sessions=None, device=None, path=None):
        root = path or recordings_dir(name)
        if not os.path.isdir(root):
            raise FileNotFoundError(f"no recording '{name}' ({root}): record with an ATorchEnvironment in Record mode")
        available = sorted(d for d in os.listdir(root) if glob.glob(os.path.join(root, d, "shard_*.safetensors")))
        if sessions == "latest":
            sessions = available[-1:]
        elif isinstance(sessions, str):
            sessions = [sessions]
        self.sessions = [s for s in available if sessions is None or s in sessions]
        if not self.sessions:
            raise FileNotFoundError(f"recording '{name}' has no sessions{f' named {sessions}' if sessions else ''} ({root})")
        from safetensors import safe_open
        parts = {k: [] for k in FIELDS + ("session",)}
        self.meta = {}
        for si, session in enumerate(self.sessions):
            for file in sorted(glob.glob(os.path.join(root, session, "shard_*.safetensors"))):
                with safe_open(file, "pt") as f:
                    meta = f.metadata() or {}
                    keys = set(f.keys())
                    rows = None
                    for k in FIELDS:
                        if k in keys:
                            t = f.get_tensor(k)
                            parts[k].append(t)
                            rows = t.shape[0]
                    if rows:
                        parts["session"].append(torch.full((rows,), si, dtype=torch.int32))
                if not self.meta:
                    self.meta = {k: _parse(v) for k, v in meta.items()}
        for k, ts in parts.items():
            value = torch.cat(ts).to(device) if ts else None
            setattr(self, k, value)

    def __len__(self):
        return 0 if self.obs is None else self.obs.shape[0]

    def __getitem__(self, i):
        return self.obs[i], self.actions[i]

    def transitions(self):
        """Everything as a dict of tensors (offline RL)."""
        return {k: getattr(self, k) for k in FIELDS + ("session",) if getattr(self, k) is not None}

    def episodes(self):
        """The recording split into episodes: a list of dicts of tensors (one agent's steps, in order)."""
        out = []
        for s in range(len(self.sessions)):
            for a in torch.unique(self.agent[self.session == s]).tolist():
                idx = torch.nonzero((self.session == s) & (self.agent == a)).flatten()
                done = (self.terminated[idx] > 0.5) | (self.truncated[idx] > 0.5)
                start = 0
                for end in (torch.nonzero(done).flatten() + 1).tolist() + [len(idx)]:
                    if end > start:
                        rows = idx[start:end]
                        out.append({k: getattr(self, k)[rows] for k in FIELDS})
                    start = end
        return out

    def __repr__(self):
        return f"Recording({len(self)} steps, {len(self.sessions)} session(s), {self.meta})"


def _parse(value):
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return value
