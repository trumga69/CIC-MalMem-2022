'''
Preprocessing cho CIC-MalMem-2022 (Topic 05).
Tận dụng các hàm đã viết trong folder audit và dữ liệu trong folder data.
'''

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from audit.load import derive_family, derive_sample_token, feature_columns

DATA_PATH = ROOT / "data" / "raw" / "Obfuscated-MalMem2022.csv"
SPLIT_DIR = ROOT / "data" / "splits"
SPLIT_FILE = SPLIT_DIR / "master_split.csv"
METADATA_FILE = SPLIT_DIR / "metadata.json"

RANDOM_SEED = 42
SPLIT_RATIOS = {"train": 0.7, "validation": 0.15, "test": 0.15}
SPLIT_NAMES = ["train", "validation", "test"]
SPLIT_POLICY = (
    "connected exact-feature duplicates + Category sample token; "
    "greedy group allocation by multiclass counts, 70/15/15"
)


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_raw_data(path=DATA_PATH):
    """Đọc CSV gốc và tách X, nhãn binary, nhãn multiclass bằng các hàm bên audit."""
    df = pd.read_csv(path)
    X = df[feature_columns(df)]
    y_binary = df["Class"]
    y_multiclass = derive_family(df).rename("multiclass")
    return X, y_binary, y_multiclass, df


def find_related_groups(df):
    """Gom các dòng trùng toàn bộ features hoặc cùng mã mẫu (sample token) vào chung nhóm.
    group_id là chỉ số dòng nhỏ nhất trong nhóm.
    """
    feat_cols = feature_columns(df)
    feature_group = df.groupby(feat_cols, dropna=False, sort=False).ngroup()
    sample_token = derive_sample_token(df)

    n_rows = len(df)
    row_idx = pd.Series(range(n_rows))

    # Nối mỗi dòng với dòng đầu tiên có cùng feature vector hoặc cùng mã mẫu
    first_by_feature = row_idx.groupby(feature_group).transform("min")
    has_token = sample_token.notna()
    first_by_token = row_idx[has_token].groupby(sample_token[has_token]).transform("min")

    rows = np.concatenate([row_idx.to_numpy(), row_idx[has_token].to_numpy()])
    targets = np.concatenate([first_by_feature.to_numpy(), first_by_token.to_numpy()])

    # Tìm các thành phần liên thông để gom cả các nhóm bắc cầu
    graph = coo_matrix((np.ones(len(rows)), (rows, targets)), shape=(n_rows, n_rows))
    _, component = connected_components(graph, directed=False)
    return row_idx.groupby(component).transform("min").to_numpy()


def create_split_table(y_multiclass, group_ids):
    """Chia các nhóm vào train/validation/test theo tỷ lệ 70/15/15."""
    group_counts = pd.crosstab(group_ids, y_multiclass)

    # Xáo trộn ngẫu nhiên rồi ưu tiên xếp các nhóm lớn trước
    shuffled = group_counts.sample(frac=1, random_state=RANDOM_SEED)
    shuffled["total"] = shuffled.sum(axis=1)
    group_order = shuffled.sort_values("total", ascending=False, kind="mergesort").drop(columns="total")

    counts_arr = group_order.to_numpy()
    ratios_arr = np.array(list(SPLIT_RATIOS.values()))
    deficit = np.outer(ratios_arr, counts_arr.sum(axis=0))

    split_of_group = {}
    for gid, row in zip(group_order.index, counts_arr):
        # Chọn tập đang thiếu nhiều mẫu của các lớp trong nhóm này nhất
        best_idx = int(np.argmax(deficit @ row))
        split_of_group[gid] = SPLIT_NAMES[best_idx]
        deficit[best_idx] -= row

    return pd.DataFrame({
        "row_index": np.arange(len(group_ids)),
        "group_id": group_ids,
        "split": pd.Series(group_ids).map(split_of_group).to_numpy(),
    })


def check_split_table(split_table, group_ids, y_multiclass):
    """Kiểm tra bảng chia có hợp lệ và có bị rò rỉ nhóm giữa các tập không."""
    if list(split_table.columns) != ["row_index", "group_id", "split"]:
        raise ValueError("Sai tên cột trong master_split.csv")
    if not np.array_equal(split_table["row_index"], np.arange(len(group_ids))):
        raise ValueError("row_index không khớp thứ tự dòng gốc")
    if not np.array_equal(split_table["group_id"], group_ids):
        raise ValueError("group_id không khớp với dữ liệu hiện tại")
    if not split_table["split"].isin(SPLIT_NAMES).all():
        raise ValueError("Cột split có tên tập không hợp lệ")
    if (split_table.groupby("group_id")["split"].nunique() > 1).any():
        raise ValueError("Có nhóm mẫu liên quan bị nằm ở nhiều tập khác nhau")

    class_counts = pd.crosstab(split_table["split"], y_multiclass)
    if (class_counts == 0).any().any():
        raise ValueError("Có tập bị thiếu lớp dữ liệu")


def load_master_split(data_path=DATA_PATH, *, create=False):
    """Load the saved split. Allow creation only when explicitly requested."""
    missing = [path for path in (SPLIT_FILE, METADATA_FILE) if not path.exists()]
    if missing and (len(missing) != 2 or not create):
        names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(f"Missing master split files: {names}. "
                                "Restore the official master_split.csv and metadata.json.")

    X, y_binary, y_multiclass, df = read_raw_data(data_path)
    group_ids = find_related_groups(df)

    expected_meta = {
        "seed": RANDOM_SEED,
        "policy": SPLIT_POLICY,
        "ratios": SPLIT_RATIOS,
        "raw_sha256": checksum(data_path),
        "features": list(X.columns),
        "rows": len(X),
    }

    if not missing:
        meta = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
        for key, val in expected_meta.items():
            if meta.get(key) != val:
                raise ValueError(f"Metadata không khớp ở trường: {key}")
        if meta.get("split_sha256") != checksum(SPLIT_FILE):
            raise ValueError("File master_split.csv đã bị thay đổi so với metadata")

        split_table = pd.read_csv(SPLIT_FILE)
        check_split_table(split_table, group_ids, y_multiclass)
        return X, y_binary, y_multiclass, split_table

    # Nếu chưa có file split thì tạo mới và lưu lại
    split_table = create_split_table(y_multiclass, group_ids)
    check_split_table(split_table, group_ids, y_multiclass)

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    split_table.to_csv(SPLIT_FILE, index=False, mode="x")

    metadata = {
        **expected_meta,
        "split_sha256": checksum(SPLIT_FILE),
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    with METADATA_FILE.open("x", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
    return X, y_binary, y_multiclass, split_table


def load_splits(task="binary", data_path=DATA_PATH):
    """Trả về dict {'train': (X, y), 'validation': (X, y), 'test': (X, y)} cho notebook."""
    if task not in ("binary", "multiclass"):
        raise ValueError("task phải là 'binary' hoặc 'multiclass'")

    X, y_binary, y_multiclass, split_table = load_master_split(data_path)
    y = y_binary if task == "binary" else y_multiclass

    result = {}
    for name in SPLIT_NAMES:
        mask = (split_table["split"] == name).to_numpy()
        result[name] = (X[mask], y[mask])
    return result


class OutlierDetector(BaseEstimator, TransformerMixin):
    """Tính ngưỡng IQR giống audit/feature_distributions.py để gắn cờ outlier.
    Chỉ phát hiện chứ không xóa hay sửa dữ liệu.
    """

    def __init__(self, factor=1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        q1 = np.quantile(X_arr, 0.25, axis=0)
        q3 = np.quantile(X_arr, 0.75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self

    def outlier_mask(self, X):
        X_arr = np.asarray(X, dtype=float)
        return (X_arr < self.lower_) | (X_arr > self.upper_)

    def transform(self, X):
        return np.asarray(X, dtype=float)


def make_preprocessor():
    """Tạo Pipeline tiền xử lý (chưa fit) gồm: Imputer -> OutlierDetector -> StandardScaler."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("outliers", OutlierDetector()),
        ("scaler", StandardScaler()),
    ])
