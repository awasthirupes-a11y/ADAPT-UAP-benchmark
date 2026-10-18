"""Per-layer condition numbers for VGG-16 (torchvision IMAGENET1K_V1).

Uses exactly the convention in adapt_uap.py (Component A):
  * layers: every nn.Conv2d in VGG-16 (13 layers); the three Linear layers are
    not hooked by the generator and so are not part of the weighting;
  * unfolding: the kernel W of shape (C_out, C_in, k_h, k_w) is reshaped to a
    C_out x (C_in * k_h * k_w) matrix, one row per output filter;
  * bias terms are excluded;
  * BatchNorm: torchvision's vgg16 has no BatchNorm layers, so there is
    nothing to fold (the separate vgg16_bn model is not used);
  * kappa = sigma_max / sigma_min of that matrix, and
    w_l = log(1 + kappa_l) / sum_k log(1 + kappa_k).
Singular values are computed in float64 for the table; the column
kappa_code reproduces the float32 value the generator itself uses.

    python analysis/kappa_table.py            # writes kappa_table_vgg16.csv
"""
import math
import sys

import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models


def kappa_rows(model: nn.Module):
    rows = []
    for name, m in model.named_modules():
        if not isinstance(m, nn.Conv2d):
            continue
        W = m.weight.detach()
        mat64 = W.reshape(W.size(0), -1).double()
        s = torch.linalg.svdvals(mat64)
        s32 = torch.linalg.svdvals(W.reshape(W.size(0), -1).float())
        rows.append({
            "layer": name,
            "kernel_shape": "x".join(map(str, W.shape)),
            "unfolded_shape": f"{mat64.shape[0]}x{mat64.shape[1]}",
            "sigma_max": s[0].item(),
            "sigma_min": s[-1].item(),
            "kappa": (s[0] / s[-1]).item(),
            "kappa_code": (s32[0] / (s32[-1] + 1e-6)).item(),
        })
    total = sum(math.log1p(r["kappa_code"]) for r in rows)
    for r in rows:
        r["log1p_kappa"] = math.log1p(r["kappa_code"])
        r["w_l"] = r["log1p_kappa"] / total
        r["w_uniform"] = 1.0 / len(rows)
    return rows


def main(random_init: bool = False):
    weights = None if random_init else models.VGG16_Weights.IMAGENET1K_V1
    model = models.vgg16(weights=weights).eval()
    assert not any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    df = pd.DataFrame(kappa_rows(model))
    df.index = range(1, len(df) + 1)
    out = "kappa_table_vgg16.csv" if not random_init else "kappa_table_RANDOM_INIT_TEST.csv"
    df.to_csv(out, index_label="conv_index")
    with pd.option_context("display.width", 200, "display.float_format", "{:.4g}".format):
        print(df)
    print(f"\nkappa range: {df.kappa.min():.4g} - {df.kappa.max():.4g}   ->  {out}")


if __name__ == "__main__":
    main(random_init="--random-init-test" in sys.argv)
