'''
S2: feature_distributions.csv

This file scan all feature cols to caculate các chỉ số thống kê cơ bản và indentify outliers based on IQR 

One row per feature, with:
- min, max, median
- 25th and 75th percentile (the two numbers you need to build the IQR rule)
- n_zero: how many rows have exactly 0.0 for this feature
- n_iqr_flagged: how many rows fall outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
                 (the classic "mild outlier" boundary, same one a boxplot
                 whisker is drawn at)
'''
import pandas as pd
from load import load_data, feature_columns

df = load_data()
feat_cols = feature_columns(df)

rows = []
for col in feat_cols:
    series = df[col]
    
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1 
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr 

    n_iqr_flagged = int(((series < lower_fence) | (series > upper_fence)).sum())
    n_zero = int((series == 0).sum())
    rows.append(
        {
            "features": col,
            "dtype": str(series.dtype),
            "min": series.min(),
            "q1": q1,
            "median": series.median(),
            "q3": q3,
            "max": series.max(),
            "n_zero": n_zero,
            "pct_zero": round(100 * n_zero / len(series), 2),
            "n_iqr_flagged": n_iqr_flagged,
            "pct_iqr_flagged": round(100 * n_iqr_flagged / len(series), 2),
        }
    )
dist = pd.DataFrame(rows)
print(dist.to_string(index = False))

dist.to_csv("audit/outputs/feature_distributions.csv", index=False)
print("\n Saved -> audit/outputs/feature_distributions.csv ")



'''
Tính toán quy tắc IQR (Interquartile Range - Khoảng tứ phân vị): Đây là phương pháp phổ biến nhất trong thống kê để tìm ra các điểm dữ liệu bất thường (outliers).

q1 = series.quantile(0.25): Tính điểm phân vị thứ 25 (Q1). 25% dữ liệu của cột có giá trị nhỏ hơn mức này.

q3 = series.quantile(0.75): Tính điểm phân vị thứ 75 (Q3). 75% dữ liệu nhỏ hơn mức này.

iqr = q3 - q1: Tính độ rộng của khoảng IQR (chứa 50% dữ liệu nằm ở giữa).

Xác định ranh giới bất thường (Fences):

lower_fence = q1 - 1.5 * iqr: Mức giới hạn dưới.

upper_fence = q3 + 1.5 * iqr: Mức giới hạn trên.

Bất kỳ giá trị nào nằm ngoài khoảng [lower_fence, upper_fence] đều bị coi là điểm bất thường (mild outlier).

Đếm số liệu:

n_iqr_flagged: Lọc ra các hàng có giá trị nhỏ hơn giới hạn dưới (< lower_fence) HOẶC (|) lớn hơn giới hạn trên (> upper_fence), sau đó tính tổng số lượng (.sum()).

n_zero: Đếm xem có bao nhiêu hàng mang giá trị chính xác bằng 0.0 ở cột này.

'''