import torch
from config import BenchmarkConfig
from data_loader import DatasetManager
from loader import ModelLoader
from generator import AttackGenerator
from evaluator import Evaluator
from reporter import BenchmarkReporter
from objectives import FFFObjective, GDUAPObjective, CosineUAPObjective, TRMUAPObjective, PSPUAPObjective
from adapt_uap import AdaptUAPGenerator

import os
# If a model download ever needs a Hugging Face token, set HF_TOKEN in your environment;
# never write the token into this file.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def main():
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = BenchmarkConfig()

    model_loader = ModelLoader(device)
    evaluator = Evaluator(device, config)
    reporter = BenchmarkReporter(config.output_dir)

    architectures = model_loader.load_architecture_suite()
    data_manager = DatasetManager(config)
    eval_loader = data_manager.get_imagenet_loader()
    dataset_name = "ImageNet ILSVRC 2012 Val (or Mock Fallback)"

    surrogate_name = "VGG-16"
    surrogate_model = architectures[surrogate_name]

    objectives = {
        "FFF": FFFObjective(),
        "GD-UAP": GDUAPObjective(),
        "Cosine-UAP": CosineUAPObjective(),
        "TRM-UAP": TRMUAPObjective(),
        "PSP-UAP": PSPUAPObjective()
    }

    generator = AttackGenerator(surrogate_model, device, config)

    for seed in config.seeds:
        print(f"\n[>>>] Initiating Experimental Run with Seed: {seed}")

        # 1. Run all 5 baseline methods (Skip if already completed)
        for attack_name, obj_fn in objectives.items():
            if reporter.is_completed(seed, attack_name):
                print(f"      [!] Skipping {attack_name} (Seed {seed}) - Already completed.")
                continue
                
            print(f"      Crafting {attack_name}...")
            perturbation_tensor = generator.generate(obj_fn, seed=seed)
            metrics = evaluator.evaluate_perturbation(architectures,
                                                      eval_loader,
                                                      perturbation_tensor)

            reporter.collect_run(seed, attack_name, metrics)
            torch.cuda.empty_cache()
            
        # 2. Run the newly integrated ADAPT-UAP method (Skip if already completed)
        if reporter.is_completed(seed, "ADAPT-UAP"):
            print(f"      [!] Skipping ADAPT-UAP (Seed {seed}) - Already completed.")
        else:
            print(f"      Crafting ADAPT-UAP...")
            adapt_generator = AdaptUAPGenerator(surrogate_model, device, config)
            adapt_perturbation = adapt_generator.generate(seed=seed)
            
            adapt_metrics = evaluator.evaluate_perturbation(architectures,
                                                            eval_loader,
                                                            adapt_perturbation)
            reporter.collect_run(seed, "ADAPT-UAP", adapt_metrics)
            torch.cuda.empty_cache()

    generator.cleanup()
    
    # 3. Print final report (Will only print properly once ALL seeds and methods are done)
    reporter.write_reports(dataset_name=dataset_name,
                           surrogate_name=surrogate_name)

if __name__ == "__main__":
    main()