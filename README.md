# ADAPT-UAP: a unified benchmark for data-free universal adversarial perturbations

Code and results for the Master of ICT Research thesis *Designing a Novel Data-Free Universal Adversarial Perturbation Framework for AI Security Systems* (Rupesh Awasthi, Melbourne Institute of Technology, 2026).

The pipeline re-implements five published data-free UAP methods (FFF, GD-UAP, Cosine-UAP, TRM-UAP, PSP-UAP) and the four-component ADAPT-UAP framework in one PyTorch codebase. Every method is run under the same protocol:

- VGG-16 as the surrogate model;
- an L∞ budget of ε = 10/255;
- 300 iterations;
- seeds 0–4;
- evaluation on the full 50,000-image ImageNet validation set across VGG-16, AlexNet, ResNet-152, Swin-V2 and DeiT-III.

## Results

Fooling rate (%), mean ± population standard deviation over five seeds:

| Method | VGG-16 | AlexNet | ResNet-152 | Swin-V2 | DeiT-III |
|---|---|---|---|---|---|
| FFF | 30.42 ± 0.05 | 37.45 ± 0.16 | 17.42 ± 0.10 | 7.73 ± 0.08 | 7.68 ± 0.06 |
| GD-UAP | 30.93 ± 0.22 | 37.41 ± 0.24 | 17.39 ± 0.07 | 7.76 ± 0.06 | 7.72 ± 0.05 |
| Cosine-UAP | 18.97 ± 1.08 | 22.49 ± 1.38 | 11.88 ± 0.64 | 5.99 ± 0.22 | 5.72 ± 0.19 |
| TRM-UAP | 30.49 ± 0.11 | 37.45 ± 0.21 | 17.41 ± 0.09 | 7.76 ± 0.07 | 7.63 ± 0.04 |
| PSP-UAP | 30.86 ± 0.08 | 37.17 ± 0.20 | 17.16 ± 0.14 | 7.50 ± 0.02 | 7.49 ± 0.06 |
| ADAPT-UAP | 29.30 ± 0.14 | 37.99 ± 0.34 | 16.89 ± 0.07 | 7.60 ± 0.13 | 7.76 ± 0.07 |

The raw per-seed values are in `results/benchmark_raw_seeds.csv`. The other files in `results/` are:

- `table_C1_seed_level_150_rows.csv`: all 150 method × architecture × seed values, with fooled-image counts.
- `paired_tests.csv`: the primary test and the AlexNet comparisons (thesis Table 4.5).
- `paired_tests_all_archs.csv`: ADAPT-UAP against each working baseline on every architecture, Holm-corrected across 20 tests (thesis Table 4.6).
- `kappa_table_vgg16.csv`: the per-layer condition numbers and Component A weights for VGG-16.

Please read the numbers with these limits in mind:

- The baselines are re-implementations inside one shared generator and do not reach their published fooling rates, so the table ranks the objectives within this pipeline, not the published systems.
- The Cosine-UAP objective, as implemented here, is computed on a single input and is effectively constant, so its row does not reflect the original method.
- ADAPT-UAP has the highest mean on AlexNet (about half a point) and on DeiT-III (0.04 points), and neither lead is significant after Holm correction. It is significantly worse than the working baselines on VGG-16 and ResNet-152 (`results/paired_tests_all_archs.csv`).
- The isolated test of Component A (condition-number weighting against uniform weighting) has not been run.
- The original runs did not save the perturbation tensors or the measured L∞ norms. The budget is enforced by projection after every step, and a CPU regeneration of the seed-0 perturbations reaches exactly 10/255 for every method.

## Repository layout

| Path | What it is |
|---|---|
| `main.py` | Runs the full six-method benchmark |
| `config.py` | Settings: batch size, seeds, ε, full or subset evaluation |
| `loader.py` | Loads the five pretrained models (torchvision and timm) |
| `data_loader.py` | ImageNet validation loader and pre-processing |
| `generator.py`, `objectives.py` | Shared perturbation generator and the five baseline objectives |
| `adapt_uap.py` | The ADAPT-UAP framework (Components A–D) |
| `evaluator.py`, `reporter.py` | Fooling-rate evaluation and result tables |
| `analysis/` | Scripts for seed-level logging, the Component A ablation, the κ table and summaries |
| `results/` | CSV results used in the thesis |

## How to run (Google Colab)

1. Put the ImageNet validation images (`archive.zip`, 50,000 images) in your Google Drive. ImageNet is not included here; download it from image-net.org under its terms of use.
2. In a GPU runtime, unzip it:
   ```bash
   !unzip -q /content/drive/MyDrive/archive.zip -d /content/imagenet_val
   ```
3. Install the dependencies (the original runs used the packages preinstalled on Colab and did not record their versions, so `requirements.txt` is unpinned):
   ```bash
   pip install -r requirements.txt
   ```
4. In `config.py`, set `use_full_imagenet = True` for the full 50,000 images, or `False` with a subset size for a quick check, and adjust the batch size to your GPU memory.
5. Run:
   ```bash
   python main.py
   ```

To regenerate the seed-0 perturbations on a CPU and measure their norms and similarity (thesis Section 4.5.1), run `python analysis/regenerate_seed0_cpu.py`; the outputs used in the thesis are in `results/diagnostic/`. To recompute the condition-number table, run `python analysis/kappa_table.py`. To rebuild Table 4.3, the primary test and Tables 4.5 and 4.6 from `results/benchmark_raw_seeds.csv`, run `python analysis/rebuild_tables.py`. To run a corrected Cosine-UAP variant (cosine between clean and perturbed logits over a batch of inputs, instead of the constant self-similarity in `objectives.py`), run `python analysis/cosine_uap_fixed.py`; it saves the perturbation tensors and measured norms. To measure the random-noise control (random-sign and uniform noise at the same 10/255 budget, no optimisation), run `python analysis/noise_control.py`; it writes `results/noise_control_seeds.csv`. To re-run the benchmark with per-run logging (run IDs, measured L∞, saved tensors), run `python analysis/audit_benchmark.py`.

## Citation

If you use this code, please cite the thesis:

> R. Awasthi, "Designing a novel data-free universal adversarial perturbation framework for AI security systems," Master of ICT Research thesis, Melbourne Institute of Technology, Melbourne, Australia, 2026.
