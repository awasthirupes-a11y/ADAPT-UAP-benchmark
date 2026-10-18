"""Rebuild Table 4.3 and the paired tests from seed_level_results.csv
(and, if present, the ablation from ablation_results.csv).

    python analysis/summarise.py /content/drive/MyDrive/thesis/results/audit
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

ARCH_ORDER = ["VGG-16", "AlexNet", "ResNet-152", "Swin-V2", "DeiT-III"]
METHOD_ORDER = ["FFF", "GD-UAP", "Cosine-UAP", "TRM-UAP", "PSP-UAP", "ADAPT-UAP"]


def paired(a: pd.Series, b: pd.Series, label: str, alternative="two-sided"):
    d = (a - b).values
    n = len(d)
    try:
        w = stats.wilcoxon(a.values, b.values, alternative=alternative, method="exact")
        p_w, stat = w.pvalue, w.statistic
    except ValueError:
        p_w, stat = float("nan"), float("nan")
    # matched-pairs rank-biserial correlation
    ranks = stats.rankdata(np.abs(d))
    rbc = (ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum() if ranks.sum() else 0.0
    t = stats.ttest_rel(a.values, b.values)
    ci = stats.t.interval(0.95, n - 1, loc=d.mean(), scale=stats.sem(d)) if n > 1 else (np.nan, np.nan)
    return {"comparison": label, "n_pairs": n, "mean_diff_pp": d.mean(), "sd_diff": d.std(ddof=1),
            "ci95_low": ci[0], "ci95_high": ci[1], "positive_pairs": int((d > 0).sum()),
            "wilcoxon_W": stat, "wilcoxon_p": p_w, "alternative": alternative,
            "rank_biserial": rbc, "paired_t_p": t.pvalue}


def main(folder):
    df = pd.read_csv(os.path.join(folder, "seed_level_results.csv"))
    assert len(df) == 150, f"expected 150 rows, found {len(df)}"
    # integrity checks
    recomputed = 100.0 * df.n_fooled / df.n_images
    assert np.allclose(recomputed, df.fooling_rate), "FR does not equal fooled/images"
    assert (df.linf_effective_x255 <= df.epsilon_x255 + 1e-3).all(), "budget exceeded"
    print("Images per row:", sorted(df.n_images.unique()))
    print("Max measured L_inf (x255):", df.linf_delta_x255.max(), "/ budget", df.epsilon_x255.max())

    g = df.groupby(["method", "architecture"]).fooling_rate
    tab = pd.DataFrame({"mean": g.mean(), "std_pop": g.std(ddof=0), "std_sample": g.std(ddof=1)})
    tab.to_csv(os.path.join(folder, "table_4_3_rebuilt.csv"))
    cell = tab.apply(lambda r: f"{r['mean']:.2f} ± {r['std_pop']:.2f}", axis=1).unstack()
    print("\nTable 4.3 (mean ± population std, as printed by reporter.py)")
    print(cell.reindex(index=METHOD_ORDER, columns=ARCH_ORDER).to_string())

    wide = df.pivot_table(index="seed", columns=["method", "architecture"], values="fooling_rate")
    tests = [paired(wide[("TRM-UAP", "ResNet-152")], wide[("GD-UAP", "ResNet-152")],
                    "PRIMARY (pre-specified, one-sided TRM > GD): TRM-UAP vs GD-UAP, ResNet-152",
                    alternative="greater")]
    for m in METHOD_ORDER[:-1]:
        tests.append(paired(wide[("ADAPT-UAP", "AlexNet")], wide[(m, "AlexNet")],
                            f"AlexNet: ADAPT-UAP vs {m}"))
    tdf = pd.DataFrame(tests)
    tdf.to_csv(os.path.join(folder, "paired_tests.csv"), index=False)
    with pd.option_context("display.width", 250, "display.float_format", "{:.4f}".format):
        print("\n", tdf.to_string(index=False))

    abl_path = os.path.join(folder, "ablation_results.csv")
    if os.path.exists(abl_path):
        a = pd.read_csv(abl_path)
        print("\nAblation fooling rate per arm x architecture x seed")
        print(a.pivot_table(index=["architecture", "seed"], columns="arm",
                            values="fooling_rate").to_string())
        tgt = a[a.architecture != "VGG-16"].groupby(["arm", "seed"]).fooling_rate.mean().unstack(0)
        print("\nCMFR per seed\n", tgt.to_string())
        res = paired(tgt["kappa"], tgt["uniform"], "Ablation CMFR: kappa vs uniform")
        print(pd.Series(res).to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
