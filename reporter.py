import os
import pandas as pd
import numpy as np
from typing import List, Dict

class BenchmarkReporter:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.raw_csv_path = os.path.join(self.output_dir, "benchmark_raw_seeds.csv")
        self.summary_csv_path = os.path.join(self.output_dir, "benchmark_summary.csv")
        self.raw_records: List[Dict] = []
        
        # 1. State Detection: Load existing progress to resume from timeouts
        if os.path.exists(self.raw_csv_path):
            print(f"[*] Found existing partial results at {self.raw_csv_path}. Loading for resumption...")
            df = pd.read_csv(self.raw_csv_path)
            self.raw_records = df.to_dict('records')

    def is_completed(self, seed: int, method_name: str) -> bool:
        """Checks if a specific method and seed combination is already completed."""
        for record in self.raw_records:
            if record.get("Seed") == seed and record.get("Method") == method_name:
                return True
        return False

    def collect_run(self, seed: int, method_name: str, metrics: Dict[str, float]):
        row = {"Seed": seed, "Method": method_name}
        row.update({arch: score for arch, score in metrics.items()})
        self.raw_records.append(row)
        
        # 2. Atomic Saving: Save immediately after every run to prevent data loss on crash
        df_raw = pd.DataFrame(self.raw_records)
        df_raw.to_csv(self.raw_csv_path, index=False)
        print(f"      [+] Saved results for {method_name} (Seed {seed}) to Drive.")

    def write_reports(self, dataset_name: str, surrogate_name: str):
        if not self.raw_records:
            print("[!] No records to summarize.")
            return
            
        df_raw = pd.DataFrame(self.raw_records)
        
        df_summary = df_raw.groupby("Method").agg(
            lambda x: f"{np.mean(x):.2f} ± {np.std(x):.2f}" if np.issubdtype(x.dtype, np.number) else x
        ).drop(columns=["Seed"]).reset_index()

        df_summary.to_csv(self.summary_csv_path, index=False)
        
        print("\n==========================================================")
        print("          THESIS 5-SEED BENCHMARK SUMMARY REPORT            ")
        print("==========================================================")
        print(f"Dataset      : {dataset_name}")
        print(f"Surrogate    : {surrogate_name} (IMAGENET1K_V1)")
        print(f"Budget       : L_inf = 10/255 (Projected in Pixel Space)")
        print("----------------------------------------------------------")
        print(df_summary.to_string(index=False))
        print("==========================================================\n")