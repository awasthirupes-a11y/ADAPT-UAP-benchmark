"""Shared helpers for the audit runs requested in the supervisor review.

Everything here reuses the existing pipeline modules (config, data_loader,
loader, generator, objectives, adapt_uap) unchanged, so the numbers come from
the same code that produced Table 4.3. The only addition is bookkeeping:
image counts, fooled counts, measured L-infinity norms and run identifiers.
"""
import hashlib
import io
import os
import platform
import uuid
from datetime import datetime, timezone

import torch
from tqdm import tqdm


def make_run_id(prefix: str, seed: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%y%m%d%H%M")
    return f"{prefix}-s{seed}-{stamp}-{uuid.uuid4().hex[:4]}"


def tensor_sha256(t: torch.Tensor) -> str:
    buf = io.BytesIO()
    torch.save(t.detach().cpu(), buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def environment_info(device: torch.device) -> dict:
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    return {
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "device": gpu,
    }


@torch.no_grad()
def evaluate_with_counts(target_models, dataloader, delta, device, use_amp=True):
    """Same fooling-rate definition as evaluator.py (clean prediction changes),
    but also returns the counts and the L-infinity norm actually applied.

    linf_effective is max |clamp(x + delta, 0, 1) - x| over every pixel of every
    evaluated image; clamping to [0, 1] can only shrink it below max |delta|.
    """
    device_type = "cuda" if device.type == "cuda" else "cpu"
    delta = delta.to(device)
    out = {}
    for name, model in target_models.items():
        n_images = 0
        n_fooled = 0
        n_clean_correct = 0
        linf_eff = 0.0
        for images, labels in tqdm(dataloader, desc=f"Evaluating {name}", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            adv = torch.clamp(images + delta, 0.0, 1.0)
            linf_eff = max(linf_eff, (adv - images).abs().max().item())
            with torch.autocast(device_type=device_type, enabled=use_amp):
                clean_pred = model(images).argmax(dim=1)
                adv_pred = model(adv).argmax(dim=1)
            n_images += images.size(0)
            n_fooled += (clean_pred != adv_pred).sum().item()
            n_clean_correct += (clean_pred == labels).sum().item()
        out[name] = {
            "n_images": n_images,
            "n_fooled": n_fooled,
            "fooling_rate": 100.0 * n_fooled / n_images if n_images else float("nan"),
            "n_clean_correct": n_clean_correct,
            "linf_effective_x255": round(linf_eff * 255.0, 4),
        }
    return out


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path
