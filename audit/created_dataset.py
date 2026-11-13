"""Xuất train/dev/test cho hai task từ master split dùng chung."""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from preprocessing_data.preprocessing import SPLIT_DIR, load_master_split

parser = argparse.ArgumentParser(description="Export train/dev/test from the shared master split.")
parser.add_argument("--create-master", action="store_true",
                    help="Create a new partition only if both master files are absent.")
args = parser.parse_args()

features, binary, multiclass, master = load_master_split(create=args.create_master)
dataset = features.assign(
    row_index=master["row_index"],
    Class=binary,
    multiclass=multiclass,
)

for split, filename in [
    ("train", "train.csv"),
    ("validation", "dev.csv"),
    ("test", "test.csv"),
]:
    path = SPLIT_DIR / filename
    temp_path = path.with_suffix(".csv.tmp")
    dataset.loc[master["split"].eq(split)].to_csv(temp_path, index=False)
    temp_path.replace(path)
    print(f"Saved -> {path}")

for task, labels in [("Task A", binary), ("Task B", multiclass)]:
    counts = pd.crosstab(labels, master["split"])[["train", "validation", "test"]]
    counts = counts.rename(columns={"validation": "dev"})
    print(f"\n{task} - row counts\n{counts}")
    print(f"\n{task} - percent per class\n{counts.div(counts.sum(axis=1), axis=0).mul(100).round(2)}")
