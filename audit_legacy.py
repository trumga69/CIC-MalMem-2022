"""Audit dữ liệu thô (không sửa file CSV). Kết quả lưu trong reports/audit/."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data/raw/Obfuscated-MalMem2022.csv"
SPLIT_FILE = ROOT / "data/splits/master_split.csv"
REPORT_DIR = ROOT / "reports/audit"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

BINARY_LABEL_OF = {"Benign": "Benign", "Ransomware": "Malware",
                   "Spyware": "Malware", "Trojan": "Malware"}


def count_with_percent(labels):
    counts = labels.value_counts()
    return pd.DataFrame({"count": counts, "percent": counts / len(labels) * 100})


# ---------------------------------------------------------------- 1. Đọc dữ liệu
table = pd.read_csv(DATA_PATH)
if not {"Category", "Class"}.issubset(table.columns):
    raise ValueError("Dataset thiếu cột Category hoặc Class")

features = table.drop(columns=["Category", "Class"])  # Category/Class KHÔNG được làm features
if not features.select_dtypes(exclude="number").empty:
    raise ValueError("Có features không phải kiểu số")

category = table["Category"]
binary_labels = table["Class"]
multiclass_labels = category.str.split("-").str[0]
print(f"Số dòng: {len(table)} | Số features: {features.shape[1]}")


# ---------------------------------------------------------------- 2. Nhãn và cân bằng lớp
expected_binary = multiclass_labels.map(BINARY_LABEL_OF)
invalid_category_rows = int(expected_binary.isna().sum())
mismatched_label_rows = int(binary_labels.ne(expected_binary).sum())

binary_balance = count_with_percent(binary_labels)
multiclass_balance = count_with_percent(multiclass_labels)
print("\nBinary:\n", binary_balance)
print("\nMulticlass:\n", multiclass_balance)
print("\nCategory không hợp lệ:", invalid_category_rows,
      "| Class không khớp Category:", mismatched_label_rows)


# ---------------------------------------------------------------- 3. Chất lượng dữ liệu
missing_cells = int(table.isna().sum().sum())
infinite_cells = int(np.isinf(features).sum().sum())
unique_values = features.nunique(dropna=False)
constant_features = unique_values[unique_values == 1].index.tolist()
print("\nÔ thiếu:", missing_cells, "| Ô vô cực:", infinite_cells)
print("Features hằng:", constant_features)


# ---------------------------------------------------------------- 4. Dòng trùng và xung đột nhãn
# Các dòng có cùng toàn bộ features nhận chung một mã nhóm.
feature_group = features.groupby(list(features.columns), dropna=False, sort=False).ngroup()

duplicate_rows = int(features.duplicated().sum())
rows_in_duplicate_groups = int(features.duplicated(keep=False).sum())

labels_per_group = pd.DataFrame({"group": feature_group, "binary": binary_labels,
                                 "multiclass": multiclass_labels}).groupby("group").nunique()
binary_conflict_groups = int((labels_per_group["binary"] > 1).sum())
multiclass_conflict_groups = int((labels_per_group["multiclass"] > 1).sum())

conflict_group_ids = labels_per_group.index[(labels_per_group > 1).any(axis=1)]
conflict_rows = pd.DataFrame({"Category": category, "Class": binary_labels,
                              "multiclass": multiclass_labels,
                              "feature_group": feature_group})[feature_group.isin(conflict_group_ids)]
print(f"\nDòng dư trùng features: {duplicate_rows} | "
      f"Dòng thuộc nhóm trùng: {rows_in_duplicate_groups}")
print(f"Nhóm xung đột nhãn - binary: {binary_conflict_groups}, "
      f"multiclass: {multiclass_conflict_groups}")


# ---------------------------------------------------------------- 5. Phân phối features
# Cờ IQR chỉ để mô tả giá trị cực đoan, không dùng để xóa dòng.
q1, q3 = features.quantile(0.25), features.quantile(0.75)
iqr = q3 - q1
is_extreme = features.lt(q1 - 1.5 * iqr) | features.gt(q3 + 1.5 * iqr)

distribution = features.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
distribution["unique_values"] = unique_values
distribution["zero_count"] = features.eq(0).sum()
distribution["iqr_flag_count"] = is_extreme.sum()


# ---------------------------------------------------------------- 6. Kiểm tra leakage
family = category.where(category.eq("Benign"),
                        category.str.split("-").str[:2].str.join("-"))
is_malware = (binary_labels == "Malware").astype(int)

leakage_checks = {"malware_families_and_benign": family.nunique()}

# Dùng master split cho cả phép thử feature và kiểm tra mẫu vượt split.
if SPLIT_FILE.exists():
    split_table = pd.read_csv(SPLIT_FILE)
    if not {"row_index", "split"}.issubset(split_table.columns):
        raise ValueError("Master split thiếu cột row_index hoặc split")
    if not np.array_equal(split_table["row_index"], np.arange(len(table))):
        raise ValueError("Master split không khớp số dòng hoặc thứ tự row_index của raw")

    split_of_row = split_table["split"]
    if split_of_row.isna().any() or set(split_of_row) != {"train", "validation", "test"}:
        raise ValueError("Master split phải có đủ train, validation và test")

    # Accuracy cao là tín hiệu cần xem xét thêm về feature.
    train_rows = split_of_row.eq("train")
    validation_rows = split_of_row.eq("validation")
    accuracy_of_single_feature = {}
    for column in features.columns:
        tree = DecisionTreeClassifier(max_depth=2, random_state=0)
        tree.fit(features.loc[train_rows, [column]], is_malware.loc[train_rows])
        accuracy_of_single_feature[column] = tree.score(
            features.loc[validation_rows, [column]], is_malware.loc[validation_rows]
        )
    single_feature_ranking = pd.Series(accuracy_of_single_feature).sort_values(ascending=False)
    print("\nTop 5 feature đơn lẻ (validation accuracy):\n", single_feature_ranking.head(5).round(4))
    leakage_checks["best_single_feature"] = single_feature_ranking.index[0]
    leakage_checks["best_single_feature_accuracy"] = round(single_feature_ranking.iloc[0], 5)
    leakage_checks["single_feature_fit_split"] = "train"
    leakage_checks["single_feature_score_split"] = "validation"

    sample_token = category.str.extract(r"^[^-]+-[^-]+-([0-9a-fA-F]+)-[0-9]+\.raw$")[0].str.lower()

    def count_keys_in_many_splits(keys):
        splits_per_key = pd.DataFrame({"key": keys, "split": split_of_row}).dropna() \
                           .groupby("key")["split"].nunique()
        return int((splits_per_key > 1).sum())

    leakage_checks["feature_groups_in_many_splits"] = count_keys_in_many_splits(feature_group)
    leakage_checks["sample_tokens_in_many_splits"] = count_keys_in_many_splits(sample_token)
    leakage_checks["families_present_in_all_splits"] = int(
        (pd.DataFrame({"family": family, "split": split_of_row})
         .groupby("family")["split"].nunique() == 3).sum())
else:
    print("(Chưa có master split nên bỏ qua phép thử feature và kiểm tra mẫu vượt split.)")
leakage_checks = pd.Series(leakage_checks, name="value")
print("\nKiểm tra leakage:\n", leakage_checks)


# ---------------------------------------------------------------- 7. Lưu kết quả
summary = pd.Series({
    "rows": len(table), "features": features.shape[1],
    "missing_cells": missing_cells, "infinite_cells": infinite_cells,
    "constant_features": len(constant_features),
    "invalid_category_rows": invalid_category_rows,
    "binary_label_mismatch_rows": mismatched_label_rows,
    "duplicate_feature_rows": duplicate_rows,
    "rows_in_duplicate_groups": rows_in_duplicate_groups,
    "binary_conflict_groups": binary_conflict_groups,
    "multiclass_conflict_groups": multiclass_conflict_groups,
}, name="value")

files_to_save = {
    "basic_audit_summary": (summary, "check"),
    "leakage_checks": (leakage_checks, "check"),
    "feature_distributions": (distribution, "feature"),
    "binary_counts": (binary_balance, "class"),
    "multiclass_counts": (multiclass_balance, "class"),
    "conflicting_feature_labels": (conflict_rows, "row_index_zero_based"),
}
for file_name, (content, index_name) in files_to_save.items():
    content.to_csv(REPORT_DIR / f"{file_name}.csv", index_label=index_name)
print("\nĐã lưu kết quả tại:", REPORT_DIR)
