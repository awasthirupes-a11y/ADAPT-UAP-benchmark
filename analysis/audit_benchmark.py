"""Re-run of the six-method benchmark (Table 4.3) with seed-level logging.

Writes one row per (method, architecture, seed) = 6 x 5 x 5 = 150 rows to
<output_dir>/seed_level_results.csv, plus each perturbation tensor as .pt.

Run from the repository root (so the main modules import):
    python analysis/audit_benchmark.py                 # full 50,000-image run
    python analysis/audit_benchmark.py --subset 2000   # quick smoke test only

The run resumes automatically: finished (method, seed) pairs are skipped.
"""
import argparse
import os
import sys
import time

import pandas as pd
import torch

sys.path.insert(0, os.getcwd())
from config import BenchmarkConfig                       # noqa: E402
from data_loader import DatasetManager                   # noqa: E402
from loader import ModelLoader                           # noqa: E402
from generator import AttackGenerator                    # noqa: E402
from objectives import (FFFObjective, GDUAPObjective,    # noqa: E402
                        CosineUAPObjective, TRMUAPObjective, PSPUAPObjective)
from adapt_uap import AdaptUAPGenerator                  # noqa: E402

from audit_common import (make_run_id, tensor_sha256, environment_info,  # noqa: E402
                          evaluate_with_counts, ensure_dir)

METHODS = ["FFF", "GD-UAP", "Cosine-UAP", "TRM-UAP", "PSP-UAP", "ADAPT-UAP"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=0,
                    help="evaluate on a random subset of N images (0 = full 50k)")
    ap.add_argument("--out", default=None, help="output folder (default: config.output_dir/audit)")
    args = ap.parse_args()

    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = BenchmarkConfig()
    if args.subset:
        config.use_full_imagenet = False
        config.subset_size = args.subset

    out_dir = ensure_dir(args.out or os.path.join(config.output_dir, "audit"))
    pert_dir = ensure_dir(os.path.join(out_dir, "perturbations"))
    csv_path = os.path.join(out_dir, "seed_level_results.csv")

    done = set()
    rows = []
    if os.path.exists(csv_path):
        prev = pd.read_csv(csv_path)
        rows = prev.to_dict("records")
        done = {(r["method"], int(r["seed"])) for r in rows}
        print(f"[*] Resuming: {len(done)} (method, seed) pairs already logged.")

    architectures = ModelLoader(device).load_architecture_suite()
    loader = DatasetManager(config).get_imagenet_loader()
    if not loader:
        raise SystemExit("ImageNet folder not found at config.imagenet_val_dir - aborting "
                         "(no mock fallback is used for audit runs).")
    n_dataset = len(loader.dataset)
    print(f"[*] Evaluation set: {n_dataset} images from {config.imagenet_val_dir}")

    surrogate = architectures["VGG-16"]
    objectives = {"FFF": FFFObjective(), "GD-UAP": GDUAPObjective(),
                  "Cosine-UAP": CosineUAPObjective(), "TRM-UAP": TRMUAPObjective(),
                  "PSP-UAP": PSPUAPObjective()}
    baseline_gen = AttackGenerator(surrogate, device, config)
    env = environment_info(device)

    for seed in config.seeds:
        print(f"\n[>>>] Seed {seed}")
        for method in METHODS:
            if (method, seed) in done:
                print(f"      [!] {method} seed {seed} already logged - skipping")
                continue
            run_id = make_run_id(method.replace("-", ""), seed)
            t0 = time.time()
            if method == "ADAPT-UAP":
                delta = AdaptUAPGenerator(surrogate, device, config).generate(seed=seed)
            else:
                delta = baseline_gen.generate(objectives[method], seed=seed)
            gen_seconds = time.time() - t0
            linf_delta = delta.abs().max().item() * 255.0
            torch.save(delta.cpu(), os.path.join(pert_dir, f"{run_id}.pt"))
            sha = tensor_sha256(delta)

            metrics = evaluate_with_counts(architectures, loader, delta, device,
                                           use_amp=config.use_mixed_precision)
            for arch, m in metrics.items():
                rows.append({
                    "run_id": run_id, "method": method, "architecture": arch, "seed": seed,
                    "n_images": m["n_images"], "n_fooled": m["n_fooled"],
                    "fooling_rate": m["fooling_rate"],
                    "n_clean_correct": m["n_clean_correct"],
                    "linf_delta_x255": round(linf_delta, 4),
                    "linf_effective_x255": m["linf_effective_x255"],
                    "epsilon_x255": round(config.epsilon * 255.0, 4),
                    "perturbation_sha256": sha,
                    "generation_seconds": round(gen_seconds, 1),
                    "iterations": config.iterations, **env,
                })
            pd.DataFrame(rows).to_csv(csv_path, index=False)   # save after every run
            print(f"      [+] {method} seed {seed} logged as {run_id}  "
                  f"(VGG-16 FR {metrics['VGG-16']['fooling_rate']:.2f}%, "
                  f"L_inf {linf_delta:.3f}/255)")
            torch.cuda.empty_cache()

    baseline_gen.cleanup()
    df = pd.DataFrame(rows)
    print(f"\n[*] {len(df)} seed-level rows written to {csv_path}")
    print("[*] Run summarise.py on this file to rebuild Table 4.3 and the paired tests.")


if __name__ == "__main__":
    main()
