import argparse
from pathlib import Path
import shutil
import pandas as pd
import numpy as np


def set_seed(seed: int):
    np.random.seed(seed)


def sample_parquet_file(df: pd.DataFrame, nums: float, min_if_ratio=True) -> pd.DataFrame:
    """
    对单个 DataFrame 按 category 分组采样并返回 concat 后的 DataFrame（并按原索引排序）。
    nums >= 1 : 每个 category 固定采样 nums 条（int）
    0 < nums < 1 : 每个 category 采样 int(len_group * nums) 条（若为0且 min_if_ratio=True 则至少1条）
    """
    assert 'category' in df.columns, "DataFrame must contain 'category' column!"
    assert 0 < nums , "num_samples should be a positive integer!"
    
    # 获取所有 category
    cat_nums = len(df['category'].unique())
    parts = []
    for iter, (category, group) in enumerate(df.groupby('category', sort=False)):
        group_len = len(group)
        if nums >= 1:
            n = int(nums)
        else:
            n = int(group_len * nums)
            if min_if_ratio and group_len > 0 and n == 0:
                n = 1  # 确保至少采样1条
                
        print(f"\t[{iter+1}/{cat_nums}] Sampling category {category} with {n} samples on {group_len} available rows...")

        if group_len <= n:
            sampled = group
            print(f"\t\tcategory {category}: only {len(group)} available, no sampling.")
        else:
            # 从 group.index 中随机选择 n 个索引
            chosen_idx = np.random.choice(group.index.values, size=n, replace=False)
            sampled = group.loc[chosen_idx]
        
        parts.append(sampled)

    if len(parts) == 0:
        return df.iloc[0:0]  # 空 DataFrame，保留列结构

    sampled_df = pd.concat(parts, axis=0)
    # 按照原始索引排序以尽量保留原始顺序
    sampled_df = sampled_df.sort_index()
    # 保证列顺序不变
    sampled_df = sampled_df[df.columns.tolist()]
    return sampled_df


def sample_mmlu_pro_parquet(input_dir: str, output_dir: str, nums: float, seed: int, test_only: bool = False):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 复制非 parquet 文件（例如 dataset_info.json 等），以保留原始目录的元数据
    for f in input_dir.iterdir():
        if f.is_file() and f.suffix != ".parquet":
            shutil.copy(f, output_dir / f.name)

    parquet_files = sorted([p for p in input_dir.glob("*.parquet")])
    if len(parquet_files) == 0:
        raise FileNotFoundError(f"No .parquet files found in {input_dir}")

    for i, pfile in enumerate(parquet_files, 1):
        print(f"[{i}/{len(parquet_files)}] Loading {pfile.name} ...")
        if test_only and "test" not in pfile.name:
            print(f"\tSkip sampling for file {pfile.name}.")
            continue
        
        df = pd.read_parquet(pfile)

        print(f"\ttotal rows: {len(df)}, columns: {list(df.columns)}")
        sampled_df = sample_parquet_file(df, nums)

        out_path = output_dir / pfile.name
        # 保存为单文件 parquet，使用 pyarrow 引擎（pandas 默认通常是 pyarrow）
        sampled_df.to_parquet(out_path, index=False)
        print(f"\tsaved sampled parquet -> {out_path} (rows: {len(sampled_df)})")

    print("Sampled parquet files are in:", output_dir, "\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample MMLU-Pro dataset by category")
    parser.add_argument("--dataset_path", type=str, default="/data/share/datasets/MMLU-Pro/data",
                        help="Path to original MMLU-Pro dataset (parquet directory)")
    parser.add_argument("--output_dir", type=str, default="/data/share/datasets/MMLU-Pro/sampled_data",
                        help="Path to save sampled dataset")
    parser.add_argument("--num_samples", type=float, default=0.01,
                        help="Number of samples: positive integer for absolute, 0<float<1 for ratio")
    parser.add_argument("--seed", type=int, default=32, help="Random seed")
    parser.add_argument("--no_test_only", action="store_true", help="Whether not to only sample the test split")
    
    args = parser.parse_args()

    set_seed(args.seed)
    sample_mmlu_pro_parquet(args.dataset_path, args.output_dir, args.num_samples, args.seed, not args.no_test_only)
