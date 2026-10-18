"""Section 4.6 ablation: Component A (condition-number layer weighting) in isolation.

Two arms, identical except for the per-layer weights:
    uniform : w_l = 1 / L                       (use_component_a = False)
    kappa   : w_l = log(1+k_l) / sum log(1+k)   (use_component_a = True)
Both arms use the truncated-ratio objective (Component B on) and switch off
the multi-scale priors (C) and the budget schedule (D), so Component A is the
only factor that changes. Same seeds, budget, iterations and evaluation set.

Output: <out>/ablation_results.csv, one row per (arm, architecture, seed),
and <out>/ablation_cmfr.csv, one row per (arm, seed) with
CMFR = mean fooling rate over the four non-surrogate architectures.

    python analysis/audit_ablation.py                  # full 50,000 images, seeds 0-4
    python analysis/audit_ablation.py --subset 10000   # reduced scale (state N in the thesis)
"""
import argparse
import copy
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, os.getcwd())
from config import BenchmarkConfig, AblationConfig       # noqa: E402
from data_loader import DatasetManager                   # noqa: E402
from loader import ModelLoader                           # noqa: E402
from adapt_uap import AdaptUAPGenerator                  # noqa: E402

from audit_common import (make_run_id, tensor_sha256, environment_info,  # noqa: E402
                          evaluate_with_counts, ensure_dir)

ARMS = {
    "uniform": AblationConfig(use_component_a=False, use_component_b=True,
                              use_component_c=False, use_component_d=False),
    "kappa": AblationConfig(use_component_a=True, use_component_b=True,
                            use_component_c=False, use_component_d=False),
}
SURROGATE = "VGG-16"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=0, help="0 = full 50k validation set")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base = BenchmarkConfig()
    if args.subset:
        base.use_full_imagenet = False
        base.subset_size = args.subset
    out_dir = ensure_dir(args.out or os.path.join(base.output_dir, "audit"))
    csv_path = os.path.join(out_dir, "ablation_results.csv")

    rows, done = [], set()
    if os.path.exists(csv_path):
        rows = pd.read_csv(csv_path).to_dict("records")
        done = {(r["arm"], int(r["seed"])) for r in rows}

    architectures = ModelLoader(device).load_architecture_suite()
    loader = DatasetManager(base).get_imagenet_loader()
    if not loader:
        raise SystemExit("ImageNet folder not found - aborting.")
    n_eval = len(loader.dataset)
    print(f"[*] Ablation evaluation set: {n_eval} images "
          f"({'full set' if not args.subset else f'random subset, subset_seed={base.subset_seed}'})")
    env = environment_info(device)

    for seed in args.seeds:
        for arm, abl in ARMS.items():
            if (arm, seed) in done:
                continue
            cfg = copy.deepcopy(base)
            cfg.ablation = abl
            gen = AdaptUAPGenerator(architectures[SURROGATE], device, cfg)
            delta = gen.generate(seed=seed)
            run_id = make_run_id(f"ABL{arm}", seed)
            metrics = evaluate_with_counts(architectures, loader, delta, device,
                                           use_amp=cfg.use_mixed_precision)
            for arch, m in metrics.items():
                rows.append({"run_id": run_id, "arm": arm, "seed": seed, "architecture": arch,
                             "n_images": m["n_images"], "n_fooled": m["n_fooled"],
                             "fooling_rate": m["fooling_rate"],
                             "linf_delta_x255": round(delta.abs().max().item() * 255, 4),
                             "linf_effective_x255": m["linf_effective_x255"],
                             "perturbation_sha256": tensor_sha256(delta),
                             "subset_size": args.subset or n_eval, **env})
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            print(f"[+] {arm:8s} seed {seed}: " + ", ".join(
                f"{a} {m['fooling_rate']:.2f}" for a, m in metrics.items()))
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    targets = df[df.architecture != SURROGATE]
    cmfr = (targets.groupby(["arm", "seed"]).fooling_rate.mean()
            .rename("cmfr").reset_index())
    fr = df[df.architecture == SURROGATE][["arm", "seed", "fooling_rate"]].rename(
        columns={"fooling_rate": "white_box_fr"})
    summary = cmfr.merge(fr, on=["arm", "seed"])
    summary.to_csv(os.path.join(out_dir, "ablation_cmfr.csv"), index=False)
    print(summary.pivot(index="seed", columns="arm", values="cmfr"))


if __name__ == "__main__":
    main()
