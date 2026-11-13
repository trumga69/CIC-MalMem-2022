'''
Step 5: conflicting_labels.csv
Tìm và ghi lại chi tiết các hàng bị "xung đột nhãn" (conflicting labels)
'''

import pandas as pd
from load import load_data, feature_columns, derive_family

df = load_data()
feat_cols = feature_columns(df)
df["multiclass"] = derive_family(df)
feature_groups = df.groupby(feat_cols, dropna=False, sort=False)
feature_group_id = feature_groups.ngroup()

records = []


for label_col in ["Class", "multiclass"]:
    group_sizes = feature_groups[label_col].transform("nunique")
    #Đây là phần cốt lõi của thuật toán. Nó gom nhóm (groupby) các hàng có cùng chung giá trị ở toàn bộ 55 cột đặc trưng.
    conflicting_mask = group_sizes > 1 

    conflicting_rows = df.loc[conflicting_mask, [label_col]].copy()
    #Trích xuất ra các hàng bị xung đột đó, nhưng chỉ giữ lại cột nhãn (label_col) để báo cáo. 
    # Dùng .copy() để tạo bản sao độc lập, tránh lỗi cảnh báo của Pandas.

    #Định dạng lại bảng kết quả
    conflicting_rows["row_index"] = conflicting_rows.index #Show error located 
    conflicting_rows["label_type"] = label_col #Ghi lại loại nhãn đang bị xung đột
    conflicting_rows = conflicting_rows.rename(columns={label_col: "label"}) 
    conflicting_rows["feature_group"] = feature_group_id.loc[conflicting_rows.index]
    print(f"{label_col}: {conflicting_rows['feature_group'].nunique()} conflict groups, {len(conflicting_rows)} rows")
    records.append(conflicting_rows[["label_type", "row_index", "feature_group", "label"]])

conflicts = pd.concat(records, ignore_index=True)
print(f"Conflicts found: {len(conflicts)} rows")
if len(conflicts):
    print(conflicts.to_string(index=False))
else: 
    print("No conflicting labels found")

conflicts.to_csv("audit/outputs/conflicting_labels.csv",index=False)
print("\n Saved -> audit/outputs/conflicting_labels.csv")