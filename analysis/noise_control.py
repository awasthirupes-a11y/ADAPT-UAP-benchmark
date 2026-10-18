"""Random-noise control for the benchmark (no optimisation, evaluation only).

Evaluates two unoptimised perturbations at the same budget as the attacks
(epsilon = 10/255, L-infinity, pixel space) on all five architectures and the
full ImageNet validation set, for seeds 0-4:

  * Random-sign noise: every pixel is +epsilon or -epsilon at random. This is the
    fair control for the attacks, whose perturbations sit at the boundary for
    78-93% of pixel values (thesis Section 4.5.1).
  * Uniform noise: every pixel is drawn uniformly from [-epsilon, +epsilon].

It uses the same model loader, data loader and evaluator as main.py, so the
fooling rate is computed exactly as in Table 4.3 (prediction change against the
model's own clean prediction).

Run from the repository root, in the same Colab runtime as main.py:
    python analysis/noise_control.py
Output: results/noise_control_seeds.csv (one row per noise type and seed) and a
printed mean +- population SD per architecture.
"""
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import BenchmarkConfig  # noqa: E402
from data_loader import DatasetManager  # noqa: E402
from evaluator import Evaluator  # noqa: E402
from loader import ModelLoader  # noqa: E402

ARCHS = ["VGG-16", "AlexNet", "ResNet-152", "Swin-V2", "DeiT-III"]
N_IMAGES = 50_000


def make_noise(kind: str, seed: int, eps: float, size: int, device) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    shape = (1, 3, size, size)
    if kind == "Random-sign noise":
        noise = (torch.randint(0, 2, shape, generator=g).float() * 2 - 1) * eps
    else:
        noise = (torch.rand(shape, generator=g) * 2 - 1) * eps
    return noise.to(device)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = BenchmarkConfig()
    eps = config.epsilon
    size = config.crop_size

    archs = ModelLoader(device).load_architecture_suite()
    loader = DatasetManager(config).get_imagenet_loader()
    evaluator = Evaluator(device, config)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "noise_control_seeds.csv")

    rows = []
    for kind in ("Random-sign noise", "Uniform noise"):
        for seed in config.seeds:
            delta = make_noise(kind, seed, eps, size, device)
            assert float(delta.abs().max()) <= eps + 1e-7
            fr = evaluator.evaluate_perturbation(archs, loader, delta)
            row = {"method": kind, "seed": seed, "linf_x255": round(float(delta.abs().max()) * 255, 4)}
            for a in ARCHS:
                row[a] = round(fr[a], 3)
            rows.append(row)
            print(kind, seed, {a: row[a] for a in ARCHS})
            with open(out_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            torch.cuda.empty_cache()

    print("\nMean +- population SD over seeds (fooling rate, %):")
    for kind in ("Random-sign noise", "Uniform noise"):
        sub = [r for r in rows if r["method"] == kind]
        cells = [f"{a} {np.mean([r[a] for r in sub]):.2f} +- {np.std([r[a] for r in sub]):.2f}" for a in ARCHS]
        print(f"  {kind}: " + " | ".join(cells))
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
