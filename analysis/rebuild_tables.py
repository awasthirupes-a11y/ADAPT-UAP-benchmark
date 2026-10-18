"""Rebuild the thesis result tables from results/benchmark_raw_seeds.csv.

    python analysis/rebuild_tables.py

Prints and writes to results/rebuilt/:
  * Table 4.3  - mean and population SD (ddof = 0) over seeds 0-4;
  * Section 4.7.2 primary test - TRM-UAP minus GD-UAP on ResNet-152, exact one-sided Wilcoxon;
  * Table 4.5  - ADAPT-UAP minus each baseline on AlexNet, paired t, Holm over 5, exact Wilcoxon;
  * Table 4.6  - ADAPT-UAP minus each working baseline on every architecture, paired t, Holm over 20.
Only the stored fooling rates are used, so every number in those tables can be checked from the CSV.
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHS = ["VGG-16", "AlexNet", "ResNet-152", "Swin-V2", "DeiT-III"]
METHODS = ["FFF", "GD-UAP", "Cosine-UAP", "TRM-UAP", "PSP-UAP", "ADAPT-UAP"]
WORKING = ["FFF", "GD-UAP", "TRM-UAP", "PSP-UAP"]


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - k) * p[i]))
        adj[i] = running
    return adj


def paired_t(a, b):
    d = np.asarray(a) - np.asarray(b)
    t = stats.ttest_rel(a, b)
    half = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    try:
        w = stats.wilcoxon(a, b, alternative="two-sided", method="exact").pvalue
    except ValueError:
        w = float("nan")
    return dict(mean_diff_pp=d.mean(), ci95_lo=d.mean() - half, ci95_hi=d.mean() + half,
                t_df4=t.statistic, p_two_sided=t.pvalue, wilcoxon_exact_p=w, seeds_ahead=int((d > 0).sum()))


def main():
    raw = pd.read_csv(os.path.join(ROOT, "results", "benchmark_raw_seeds.csv"))
    out = os.path.join(ROOT, "results", "rebuilt")
    os.makedirs(out, exist_ok=True)

    def series(method, arch):
        return raw[raw.Method == method].sort_values("Seed")[arch].to_numpy()

    # Table 4.3
    t43 = pd.DataFrame({a: [f"{series(m, a).mean():.2f} ± {series(m, a).std(ddof=0):.2f}" for m in METHODS]
                        for a in ARCHS}, index=METHODS)
    t43.to_csv(os.path.join(out, "table_4_3.csv"))
    print("Table 4.3\n", t43, "\n")

    # Primary test
    trm, gd = series("TRM-UAP", "ResNet-152"), series("GD-UAP", "ResNet-152")
    w = stats.wilcoxon(trm, gd, alternative="greater", method="exact")
    d = trm - gd
    print(f"Primary: TRM-UAP - GD-UAP on ResNet-152: mean {d.mean():+.4f} pp, "
          f"one-sided exact Wilcoxon p = {w.pvalue:.4f}, seeds ahead {int((d > 0).sum())}/5\n")

    # Table 4.5
    rows = []
    for b in ["FFF", "GD-UAP", "Cosine-UAP", "TRM-UAP", "PSP-UAP"]:
        rows.append(dict(baseline=b, **paired_t(series("ADAPT-UAP", "AlexNet"), series(b, "AlexNet"))))
    t45 = pd.DataFrame(rows)
    t45["holm_p_5"] = holm(t45.p_two_sided)
    t45.to_csv(os.path.join(out, "table_4_5.csv"), index=False)
    print("Table 4.5\n", t45.round(4), "\n")

    # Table 4.6
    rows = []
    for a in ARCHS:
        for b in WORKING:
            rows.append(dict(architecture=a, baseline=b, **paired_t(series("ADAPT-UAP", a), series(b, a))))
    t46 = pd.DataFrame(rows)
    t46["holm_p_20"] = holm(t46.p_two_sided)
    t46.to_csv(os.path.join(out, "table_4_6.csv"), index=False)
    print("Table 4.6\n", t46.drop(columns="wilcoxon_exact_p").round(4))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
