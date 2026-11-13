"""Thống kê dấu hiệu mẫu liên quan trên CSV gốc trước khi chia tập."""
import pandas as pd
from sklearn.model_selection import train_test_split

from load import load_data, feature_columns, derive_sample_token

TEST_SIZE = 0.2
RANDOM_STATE = 42

df = load_data()
features = df[feature_columns(df)]
feature_groups = features.groupby(list(features.columns), dropna=False, sort=False)
vector_sizes = feature_groups.size()
repeated_vectors = vector_sizes[vector_sizes > 1]

malware = df["Class"].eq("Malware")
tokens = derive_sample_token(df).loc[malware]
token_sizes = tokens.value_counts()
repeated_tokens = token_sizes[token_sizes > 1]

# Thử chia ngẫu nhiên theo dòng để đo nguy cơ trước khi chọn cách chia chính thức.
train_rows, validation_rows = train_test_split(
    df.index,
    test_size=TEST_SIZE,
    stratify=df["Class"],
    random_state=RANDOM_STATE,
)
split = pd.Series("train", index=df.index)
split.loc[validation_rows] = "validation"

def count_crossing_groups(keys):
    rows = pd.DataFrame({"group": keys, "split": split}).dropna(subset=["group"])
    split_counts = rows.groupby("group")["split"].nunique()
    crossing_groups = split_counts[split_counts > 1].index
    return len(crossing_groups), int(rows["group"].isin(crossing_groups).sum())


vector_groups_crossing, vector_rows_crossing = count_crossing_groups(feature_groups.ngroup())
token_groups_crossing, token_rows_crossing = count_crossing_groups(tokens)

report = pd.DataFrame([
    ("feature_vector", "repeated_groups", len(repeated_vectors)),
    ("feature_vector", "rows_in_repeated_groups", repeated_vectors.sum()),
    ("sample_token", "unique_tokens", len(token_sizes)),
    ("sample_token", "repeated_groups", len(repeated_tokens)),
    ("sample_token", "rows_in_repeated_groups", repeated_tokens.sum()),
    ("sample_token", "malware_rows_with_token", tokens.notna().sum()),
    ("sample_token", "malware_rows_without_token", tokens.isna().sum()),
    ("sample_token", "benign_rows_without_sample_id", df["Category"].eq("Benign").sum()),
    ("random_80_20", "random_state", RANDOM_STATE),
    ("random_80_20", "train_rows", len(train_rows)),
    ("random_80_20", "validation_rows", len(validation_rows)),
    ("random_80_20_feature_vector", "groups_crossing_split", vector_groups_crossing),
    ("random_80_20_feature_vector", "rows_in_crossing_groups", vector_rows_crossing),
    ("random_80_20_sample_token", "groups_crossing_split", token_groups_crossing),
    ("random_80_20_sample_token", "rows_in_crossing_groups", token_rows_crossing),
], columns=["check", "metric", "value"])

print("Chia thử 80/20 theo dòng, stratify theo Class, seed 42.")
print(report.to_string(index=False))
print("Category của Benign chỉ ghi Benign. Các dòng này vẫn được kiểm tra trùng features.")
report.to_csv("audit/outputs/leakage_risk.csv", index=False)
