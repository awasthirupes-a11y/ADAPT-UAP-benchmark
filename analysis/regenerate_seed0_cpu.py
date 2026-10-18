"""Post-hoc check (thesis Section 4.5.1): regenerate the seed-0 perturbations of all six methods
with the benchmark code and measure their L-infinity norm, how many values sit at the budget,
and how similar the perturbations are to each other.

The original perturbation tensors were not kept. A CPU draws a different random input from a GPU
for the same seed, so these are NOT the tensors evaluated in Table 4.3, and no fooling rates are computed.

Run from the repository root:
    python analysis/regenerate_seed0_cpu.py --weights vgg16-397923af.pth --out results/diagnostic
(the weights file is torchvision's VGG-16 IMAGENET1K_V1 checkpoint; omit --weights to download it)
"""
import argparse
import itertools
import json
import os
import sys
import time

import torch
import torchvision.models as models

sys.path.insert(0, os.getcwd())
import config as C  # noqa: E402

C.BenchmarkConfig.__post_init__ = lambda self: None  # do not create the Colab output folder
from config import BenchmarkConfig  # noqa: E402
from loader import NormalizationWrapper  # noqa: E402
from generator import AttackGenerator  # noqa: E402
from adapt_uap import AdaptUAPGenerator  # noqa: E402
from objectives import (FFFObjective, GDUAPObjective, CosineUAPObjective,  # noqa: E402
                        TRMUAPObjective, PSPUAPObjective)

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default=None)
ap.add_argument("--out", default="results/diagnostic")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

dev = torch.device("cpu")
cfg = BenchmarkConfig()
if args.weights:
    vgg = models.vgg16()
    vgg.load_state_dict(torch.load(args.weights, map_location="cpu"))
else:
    vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
sur = NormalizationWrapper(vgg).eval()
for p in sur.parameters():
    p.requires_grad = False

objs = {"FFF": FFFObjective(), "GD-UAP": GDUAPObjective(), "Cosine-UAP": CosineUAPObjective(),
        "TRM-UAP": TRMUAPObjective(), "PSP-UAP": PSPUAPObjective()}
gen = AttackGenerator(sur, dev, cfg)
eps, tensors, summary = cfg.epsilon, {}, {}
for name, obj in list(objs.items()) + [("ADAPT-UAP", None)]:
    t0 = time.time()
    delta = gen.generate(obj, seed=args.seed) if obj else AdaptUAPGenerator(sur, dev, cfg).generate(seed=args.seed)
    a = delta.abs()
    tensors[name] = delta.flatten()
    summary[name] = {"linf_x255": round(float(a.max() * 255), 4),
                     "fraction_at_budget": round(float((a > eps - 1e-6).float().mean()), 4),
                     "mean_abs_x255": round(float(a.mean() * 255), 4),
                     "seconds": round(time.time() - t0, 1)}
    print(name, summary[name], flush=True)

pairs = []
for x, y in itertools.combinations(tensors, 2):
    u, v = tensors[x], tensors[y]
    pairs.append({"a": x, "b": y,
                  "sign_agreement": round(float((torch.sign(u) == torch.sign(v)).float().mean()), 4),
                  "cosine": round(float(torch.nn.functional.cosine_similarity(u, v, dim=0)), 4)})
json.dump(summary, open(f"{args.out}/seed{args.seed}_summary.json", "w"), indent=1)
json.dump(pairs, open(f"{args.out}/seed{args.seed}_pairwise.json", "w"), indent=1)
