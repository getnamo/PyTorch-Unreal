"""Weights on disk: safetensors (default) or .pt, found under Saved/Torch/Models/<model>/ first, then
Content/Models/, or downloaded from the Hugging Face Hub ("hf://org/repo/file.safetensors")."""
import os
import time

import torch
import unreal_bridge as ue

__all__ = ["saved_dir", "resolve", "load_state", "save_state"]


def saved_dir(model_name=None):
    """<Project>/Saved/Torch/Models[/<model>]: where save_weights writes and checkpoints go."""
    base = os.path.join(ue.project_dir or ".", "Saved", "Torch", "Models")
    return os.path.join(base, model_name) if model_name else base


def resolve(path, model_name=None, for_write=False):
    """A weights path: absolute paths as they are; hf:// ids downloaded; relative paths in
    Saved/Torch/Models/<model>/ (written there), else Content/Models/. None if reading and not found."""
    if path.startswith("hf://"):
        if for_write:
            raise ValueError("can't write to the Hugging Face Hub")
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError("hf:// weights need huggingface_hub (Project Settings > PyTorch > Hugging Face Hub)") from None
        org, repo, file = path[5:].split("/", 2)
        return hf_hub_download(f"{org}/{repo}", file)
    if os.path.isabs(path):
        return path
    saved = os.path.join(saved_dir(model_name), path)
    if for_write:
        os.makedirs(os.path.dirname(saved), exist_ok=True)
        return saved
    for candidate in (saved, os.path.join(ue.content_dir or ".", "Models", path)):
        if os.path.exists(candidate):
            return candidate
    return None


def load_state(file, device="cpu"):
    if file.endswith(".safetensors"):
        # Copies, not views of a memory-mapped file: on Windows a mapped file can't be replaced, and another
        # process (a training job) may be about to save over it
        from safetensors import safe_open
        cpu = torch.device(device).type == "cpu"
        with safe_open(file, framework="pt", device=str(device)) as f:
            return {k: (f.get_tensor(k).clone() if cpu else f.get_tensor(k)) for k in f.keys()}
    state = torch.load(file, map_location=device, weights_only=True)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    return state


def save_state(state, file):
    tmp = file + ".tmp"
    if file.endswith(".safetensors"):
        from safetensors.torch import save_file
        save_file({k: v.detach().contiguous().cpu() for k, v in state.items()}, tmp)
    else:
        torch.save({k: v.detach().cpu() for k, v in state.items()}, tmp)
    for attempt in range(40):  # Windows: a reader (another process) may have the file open for a moment
        try:
            os.replace(tmp, file)
            break
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.05)
    return file
