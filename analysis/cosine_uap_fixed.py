"""Corrected Cosine-UAP variant (re-run script).

The benchmark's CosineUAPObjective (objectives.py) computes the cosine similarity of a
single feature vector with itself, which is constant, so it gives no useful gradient
(thesis Section 4.5.1). This script implements the objective as the published method
defines it: minimise the cosine similarity between the classifier's output logits for a
clean input x and for the perturbed input x + v, averaged over a batch.

Everything else follows the shared protocol (Table 3.2): VGG-16 surrogate, sign-gradient
steps of 1/255, 300 iterations, zero start, L-infinity projection to 10/255 in pixel
space, seeds 0-4. The only change besides the objective is that each seed draws a fixed
batch of uniform random inputs (default 32) instead of one, because a cosine between clean
and perturbed outputs needs more than one input to generalise. The published method uses
artificial jigsaw images instead; that difference is a remaining simplification.

Run from the repository root in the same Colab runtime as main.py:
    python analysis/cosine_uap_fixed.py            # batch of 32
    python analysis/cosine_uap_fixed.py --batch 16 # if memory is short
Output: results/cosine_fixed_seeds.csv, one row per seed, plus the measured L-infinity norm.
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BenchmarkConfig  # noqa: E402
from data_loader import DatasetManager  # noqa: E402
from evaluator import Evaluator  # noqa: E402
from loader import ModelLoader  # noqa: E402

ARCHS = ["VGG-16", "AlexNet", "ResNet-152", "Swin-V2", "DeiT-III"]


def generate(surrogate, config, seed, batch, device):
    torch.manual_seed(seed)
    shape = (1, 3, config.crop_size, config.crop_size)
    delta = torch.zeros(shape, device=device)
    x = torch.empty((batch, 3, config.crop_size, config.crop_size), device=device).uniform_(0.0, 1.0)
    with torch.no_grad():
        clean_logits = surrogate(x)
    for _ in range(config.iterations):
        delta.requires_grad_(True)
        adv_logits = surrogate(torch.clamp(x + delta, 0.0, 1.0))
        loss = F.cosine_similarity(adv_logits, clean_logits, dim=1).mean()  # minimise
        loss.backward()
        with torch.no_grad():
            delta = delta - config.step_size * torch.sign(delta.grad)
            delta = torch.clamp(delta, -config.epsilon, config.epsilon)
    return delta.detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = BenchmarkConfig()
    archs = ModelLoader(device).load_architecture_suite()
    surrogate = archs["VGG-16"]
    surrogate.eval()
    loader = DatasetManager(config).get_imagenet_loader()
    evaluator = Evaluator(device, config)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_path = os.path.join(root, "results", "cosine_fixed_seeds.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rows = []
    for seed in config.seeds:
        delta = generate(surrogate, config, seed, args.batch, device)
        linf = float(delta.abs().max()) * 255
        torch.save(delta.cpu(), os.path.join(os.path.dirname(out_path), f"cosine_fixed_seed{seed}.pt"))
        fr = evaluator.evaluate_perturbation(archs, loader, delta)
        row = {"method": "Cosine-UAP (corrected)", "seed": seed, "batch": args.batch, "linf_x255": round(linf, 4)}
        row.update({a: round(fr[a], 3) for a in ARCHS})
        rows.append(row)
        print(row)
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        torch.cuda.empty_cache()

    print("\nMean +- population SD over seeds:")
    print(" | ".join(f"{a} {np.mean([r[a] for r in rows]):.2f} +- {np.std([r[a] for r in rows]):.2f}" for a in ARCHS))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
