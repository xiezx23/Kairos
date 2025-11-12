import argparse
import pandas as pd
from pathlib import Path
import numpy as np
import random
import tqdm


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    

def sample_cmmlu(input_dir: str, output_dir: str, nums: float):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    assert 0 < nums , "num_samples should be a positive integer!"

    sub_nums = len(list(input_dir.glob("*.csv")))
    for i, file in enumerate(input_dir.glob("*.csv")):
    # for file in tqdm.tqdm(list(input_dir.glob("*.csv")), desc="Sampling CMMLU subjects"):
        df = pd.read_csv(file, index_col=0)
        subject = file.name.split(".")[0]
        # 确定采样策略
        if nums >= 1:
            num_samples = int(nums)
        else:
            num_samples = int(len(df) * nums)
            
        print(f"[{i+1}/{sub_nums}] Sampling Subject {subject} with {num_samples} samples on {len(df)} available rows...")
        
        if len(df) <= num_samples:
            sampled_df = df
            print(f"\t Subject {subject}: only {len(df)} samples available, no sampling needed.")
        else:
            sampled_df = df.sample(n=num_samples)

        # 按照原始顺序排序
        sampled_df = sampled_df.sort_index()

        out_file = output_dir / file.name
        # head名维持不变
        sampled_df.to_csv(out_file, index=True, encoding="utf-8")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample CMMLU dataset by subject")
    parser.add_argument("--dataset_path", type=str, help="Input data directory", default="/data/share/datasets/cmmlu/test_all")
    parser.add_argument("--output_dir", type=str, help="Output data directory", default="/data/share/datasets/cmmlu/sampled")
    parser.add_argument("--num_samples", type=float, default=0.1, help="Number of samples to extract, positive integer for absolute value, 0<float<1 for ratio")
    parser.add_argument("--seed", type=int, default=32, help="Random seed")

    args = parser.parse_args()
    set_seed(args.seed)
    sample_cmmlu(args.dataset_path, args.output_dir, args.num_samples)