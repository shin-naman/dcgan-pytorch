"""Save and restore complete DCGAN training state at epoch boundaries."""

import os
import random
from pathlib import Path

import numpy as np
import torch


# These options must remain unchanged for a faithful continuation. 
# The total epoch target (--niter) may change
RESUME_LOCKED_OPTIONS = (
    "nz", "nc", "ngf", "ndf", "imageSize", "batchSize", "lr", "beta1"
)


def _get_rng_state():
    numpy_state = np.random.get_state()
    state = {
        "python": random.getstate(),
        "numpy": (
            numpy_state[0],
            torch.from_numpy(numpy_state[1].copy()),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        state["torch_mps"] = torch.mps.get_rng_state()
    return state


def _set_rng_state(state):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((
        numpy_state[0], numpy_state[1].cpu().numpy(),
        numpy_state[2], numpy_state[3], numpy_state[4],
    ))
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if state.get("torch_mps") is not None and hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.set_rng_state(state["torch_mps"])


def save_checkpoint(path, epoch_completed, global_step, netG, netD,
                    optimizerG, optimizerD, G_losses, D_losses,
                    fixed_noise, args, dataloader):
    """Atomically replace the rolling checkpoint after a fully completed epoch."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch_completed": epoch_completed,  # 1-based count; resume from this index
        "global_step": global_step,
        "netG": netG.state_dict(),
        "netD": netD.state_dict(),
        "optimizerG": optimizerG.state_dict(),
        "optimizerD": optimizerD.state_dict(),
        "G_losses": G_losses,
        "D_losses": D_losses,
        "fixed_noise": fixed_noise.cpu(),
        "config": vars(args).copy(),
        "dataset_size": len(dataloader.dataset),
        "rng_state": _get_rng_state(),
    }

    # Replacing latest.pth only after a successful write preserves the
    # previous checkpoint if training is interrupted mid-save.
    temp_path = path.with_name(path.name + ".tmp")
    try:
        torch.save(state, temp_path)
        os.replace(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def load_checkpoint(path, netG, netD, optimizerG, optimizerD,
                    args, dataloader, device):
    """Restore model, Adam, counters, histories and random states."""
    state = torch.load(path, map_location="cpu", weights_only=True)

    saved_config = state["config"]
    for option in RESUME_LOCKED_OPTIONS:
        if getattr(args, option) != saved_config[option]:
            raise ValueError(
                f"Cannot resume: --{option} is {getattr(args, option)!r}, "
                f"but checkpoint used {saved_config[option]!r}."
            )

    if len(dataloader.dataset) != state["dataset_size"]:
        raise ValueError("Cannot resume: number of training images changed.")
    if args.dataroot != saved_config["dataroot"]:
        print("Warning: dataset path changed; ensure images and their order are unchanged.")

    netG.load_state_dict(state["netG"])
    netD.load_state_dict(state["netD"])
    optimizerG.load_state_dict(state["optimizerG"])
    optimizerD.load_state_dict(state["optimizerD"])

    # Restore RNG *last*: loading and reconstructing models/optimizers may
    # otherwise consume randomness needed for the following training epoch.
    _set_rng_state(state["rng_state"])

    print(
        f"Resumed from {path}: {state['epoch_completed']} completed epoch(s), "
        f"{state['global_step']} update(s)."
    )
    return (
        state["epoch_completed"], state["global_step"],
        state["G_losses"], state["D_losses"],
        state["fixed_noise"].to(device),
    )
