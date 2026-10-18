# In evaluator.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict
from tqdm import tqdm
from config import BenchmarkConfig


class Evaluator:

    def __init__(self, device: torch.device, config: BenchmarkConfig):
        self.device = device
        self.config = config

    @torch.no_grad()
    def evaluate_perturbation(self, target_models: Dict[str, nn.Module],
                              dataloader: DataLoader,
                              delta: torch.Tensor) -> Dict[str, float]:
        results = {}

        # Determine device type for autocast (cuda or cpu)
        device_type = 'cuda' if 'cuda' in str(self.device) else 'cpu'

        for name, model in target_models.items():
            fooled_samples = 0
            total_samples = 0

            for images, _ in tqdm(dataloader,
                                  desc=f"Evaluating {name}",
                                  leave=False):
                images = images.to(self.device)
                total_samples += images.size(0)

                # Use Automatic Mixed Precision (AMP) for safe 2x speedup
                with torch.autocast(device_type=device_type,
                                    enabled=self.config.use_mixed_precision):
                    clean_preds = model(images).argmax(dim=1)

                    adv_images = torch.clamp(images + delta, 0.0, 1.0)
                    adv_preds = model(adv_images).argmax(dim=1)

                fooled_samples += (clean_preds != adv_preds).sum().item()

            fooling_rate = (fooled_samples / total_samples
                            ) * 100.0 if total_samples > 0 else 0.0
            results[name] = fooling_rate

        return results
