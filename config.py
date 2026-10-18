import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class AblationConfig:
    """Toggles for ADAPT-UAP systematic component-level ablations."""
    use_component_a: bool = True  # Condition-number-weighted layer targeting
    use_component_b: bool = True  # Truncated Ratio Maximization (TRM)
    use_component_c: bool = True  # Multi-Scale Pseudo-Semantic Priors
    use_component_d: bool = True  # Progressive Budget Scheduling


@dataclass
class BenchmarkConfig:
    # 1. Perturbation Budget (Strictly Pixel Space)
    epsilon: float = 10.0 / 255.0  # Standard L_inf budget
    step_size: float = 1.0 / 255.0
    iterations: int = 300

    # 2. Preprocessing Pipeline
    resize_shorter_edge: int = 256
    crop_size: int = 224

    # 3. Directories
    #extract_path: str = "/content/drive/MyDrive/thesis/data/archive"
    extract_path: str = "/content/imagenet_val"
    imagenet_val_dir: str = extract_path
    output_dir: str = '/content/drive/MyDrive/thesis/results'

    # 4. Repetition (Five seeds for Mean/Std)
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    num_workers: int = 16
    batch_size: int = 2048

    use_full_imagenet: bool = True  # Set to True only for final thesis production runs
    subset_size: int = 50000  # Number of random images to select
    subset_seed: int = 42
    use_mixed_precision: bool = True
    
    # 5. ADAPT-UAP Ablation Settings
    ablation: AblationConfig = field(default_factory=AblationConfig)

    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)