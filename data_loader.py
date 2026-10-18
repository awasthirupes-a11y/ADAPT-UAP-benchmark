# In data_loader.py
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, TensorDataset
from config import BenchmarkConfig


class DatasetManager:

    def __init__(self, config: BenchmarkConfig):
        self.config = config

        self.base_transform = transforms.Compose([
            transforms.Resize(self.config.resize_shorter_edge),
            transforms.CenterCrop(self.config.crop_size),
            transforms.ToTensor()
        ])

    def get_imagenet_loader(self) -> DataLoader:
        if not os.path.exists(self.config.imagenet_val_dir):
            print(
                "[Warning] ImageNet path not found. Generating mock dataset for dry-run testing."
            )
            return 0

        dataset = datasets.ImageFolder(self.config.imagenet_val_dir,
                                       transform=self.base_transform)

        # Thesis Optimization: Random Subset for Rapid Testing
        if not self.config.use_full_imagenet:
            total_size = 50000
            if total_size > self.config.subset_size:
                print(
                    f"[*] Subsetting Mode: Randomly selecting {self.config.subset_size} out of {total_size} images (Subset Seed: {self.config.subset_seed})..."
                )

                # Use a dedicated generator so random selection is perfectly reproducible
                generator = torch.Generator().manual_seed(
                    self.config.subset_seed)
                indices = torch.randperm(
                    total_size,
                    generator=generator)[:self.config.subset_size].tolist()

                dataset = Subset(dataset, indices)
            else:
                print(
                    f"[*] Warning: Dataset size ({total_size}) is smaller than requested subset ({self.config.subset_size})."
                )

        return DataLoader(dataset,
                          batch_size=self.config.batch_size,
                          shuffle=False,
                          num_workers=self.config.num_workers,
                          pin_memory=True)

    def _get_mock_loader(self) -> DataLoader:
        torch.manual_seed(0)
        mock_data = torch.rand(
            (128, 3, self.config.crop_size, self.config.crop_size))
        mock_labels = torch.randint(0, 1000, (128, ))
        return DataLoader(TensorDataset(mock_data, mock_labels),
                          batch_size=self.config.batch_size)
