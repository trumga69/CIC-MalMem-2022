import numpy as np
import pandas as pd
from load import load_data, feature_columns, derive_family
 
# Nhãn nhị phân "đáng lẽ phải có" ứng với mỗi family, dùng để đối chiếu
# ngược lại với cột Class thực tế.
EXPECTED_BINARY_OF_FAMILY = {
    "Benign": "Benign",
    "Ransomware": "Malware",
    "Spyware": "Malware",
    "Trojan": "Malware",
}
 
 
def run(df: pd.DataFrame, feat_cols: list, out_path: str = "audit/outputs/basic_audit_summary.csv") -> pd.DataFrame:
    """Compute the summary metrics and write basic_audit_summary.csv."""
 
    # --- 1. shape -----------------------------------------------------------
    n_rows = len(df)
    n_features = len(feat_cols)
 
    # --- 2. missing values ----------------------------------------------------
    # isna() gives a True/False grid the same shape as df; sum().sum() collapses
    # it to one number: total missing cells across the whole table.
    n_missing_values = int(df.isna().sum().sum())
 
    # --- 3. infinite values ---------------------------------------------------
    # np.isinf only works on numeric data, so we restrict to the feature columns.
    n_infinite_values = int(np.isinf(df[feat_cols]).sum().sum())
 
    # --- 4. constant columns ---------------------------------------------------
    # nunique() = number of distinct values in a column. If it's 1, every row
    # has the same value -> the column is constant.
    constant_columns = [c for c in feat_cols if df[c].nunique(dropna=False) == 1]
    n_constant_columns = len(constant_columns)
 
    # --- 5. duplicate rows -------------------------------------------------------
    # duplicated() flags every row that is an exact repeat of an earlier row
    # (keep='first' is the default, so the first occurrence is NOT counted,
    # only the repeats are).
    n_duplicate_rows = int(df.duplicated().sum())
 
    # --- 6. conflicting label groups ---------------------------------------------
    # Group rows by their full feature vector. If a group has more than one
    # distinct Class value, that's a conflict. We reuse this exact logic
    # (with full detail) in step 5; here we just need the COUNT of groups.
    grouped = df.groupby(feat_cols, dropna=False)["Class"].nunique()
    n_binary_conflict_groups = int((grouped > 1).sum())
    n_duplicate_feature_rows = int(df.duplicated(subset=feat_cols).sum())
 
    # --- 7 & 8. Category vs Class consistency ------------------------------------
    # family = tên family suy ra từ Category (Benign/Ransomware/Spyware/Trojan).
    # expected_binary = nhãn nhị phân "đáng lẽ" phải có, map từ family đó.
    family = derive_family(df)
    expected_binary = family.map(EXPECTED_BINARY_OF_FAMILY)
 
    # Nếu family không nằm trong 4 giá trị đã biết -> map() trả về NaN.
    n_invalid_category_rows = int(expected_binary.isna().sum())
 
    # So cột Class thực tế với expected_binary. .ne() = "not equal". Chỉ so
    # sánh ở những dòng mà expected_binary hợp lệ (không NaN), để tránh so
    # sánh nhầm với các dòng Category bất thường ở mục trên.
    valid_mask = expected_binary.notna()
    n_binary_label_mismatch_rows = int(
        df.loc[valid_mask, "Class"].ne(expected_binary.loc[valid_mask]).sum()
    )
 
    # --- assemble the summary ----------------------------------------------------
    summary = pd.DataFrame(
        {
            "metric": [
                "n_rows",
                "n_features",
                "n_missing_values",
                "n_infinite_values",
                "n_constant_columns",
                "n_duplicate_full_rows",
                "n_duplicate_feature_rows",
                "n_binary_conflict_groups",
                "n_invalid_category_rows",
                "n_binary_label_mismatch_rows",
            ],
            "value": [
                n_rows,
                n_features,
                n_missing_values,
                n_infinite_values,
                n_constant_columns,
                n_duplicate_rows,
                n_duplicate_feature_rows,
                n_binary_conflict_groups,
                n_invalid_category_rows,
                n_binary_label_mismatch_rows,
            ],
        }
    )
 
    print("[Step 1] basic_audit_summary:")
    print(summary.to_string(index=False))
    print("Label/metadata dtypes:", df[["Class", "Category"]].dtypes.to_dict())
    if constant_columns:
        print("  Constant columns found:", constant_columns)
 
    summary.to_csv(out_path, index=False)
    print(f"  Saved -> {out_path}\n")
    return summary

_df = load_data()
run(_df, feature_columns(_df))
