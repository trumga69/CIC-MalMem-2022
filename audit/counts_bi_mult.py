'''
Step 3 + 4: Counts binary and multiclass
Both are just "how many rows per label and what % of the dataset is that"
'''

import pandas as pd
from load import load_data, derive_family

df = load_data()
n_total = len(df) # Counts all rows in DataFrame để làm mẫu số khi tính tỷ lệ phần trăm

def counts_tables (labels: pd.Series) -> pd.DataFrame:
    vc = labels.value_counts(dropna=False)
    return pd.DataFrame(
        {
            "label": vc.index, 
            "count": vc.values,
            "pct": (100*vc.values / n_total).round(2),
        }
        
    )

#--- Step 3: binary (Benign vs Malware) ---
binary = counts_tables(df["Class"])
print("binary_counts:\n", binary.to_string(index=False))
binary.to_csv("audit/outputs/binary_counts.csv",index=False)

#--- Step 4: multiclass (Benign / Ransomware / Spyware / Trojan) ---
family = derive_family(df)
multiclass = counts_tables(family)
print("multiclass_counts\n", multiclass.to_string(index=False))
multiclass.to_csv("audit/outputs/multiclass_counts.csv", index= False)

print("\n Saved -> audit/outputs/binary_counts.csv ")
print("\n Saved -> audit/outputs/multiclass_counts.csv ")

'''
family = derive_family(df): Gọi hàm derive_family để xử lý cột "Category" 
(cắt bỏ phần đuôi) và lấy ra được tên chính xác của họ mã độc (Ransomware, Spyware, Trojan) hoặc Benign. 
Kết quả lưu vào biến family.
'''




