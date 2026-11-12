"""Chuẩn bị dữ liệu: tạo/đọc bộ chia train-validation-test và pipeline tiền xử lý."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted

# ---------------------------------------------------------------- Cấu hình
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data/raw/Obfuscated-MalMem2022.csv"
SPLIT_DIR = ROOT / "data/splits"
SPLIT_FILE = SPLIT_DIR / "master_split.csv"
METADATA_FILE = SPLIT_DIR / "metadata.json"

RANDOM_SEED = 42
SPLIT_NAMES = ["train", "validation", "test"]
SPLIT_POLICY = ("connected exact-feature duplicates + Category sample token; "
                "SGKF5 folds 0=test,1=validation,2-4=train")

# Category có dạng "Loại-Họ-<mã mẫu>-<số thứ tự>.raw", ví dụ "Trojan-Zeus-0dd1...-4.raw".
SAMPLE_TOKEN_PATTERN = r"^[^-]+-[^-]+-([0-9a-fA-F]+)-[0-9]+\.raw$"
BINARY_LABEL_OF = {"Benign": "Benign", "Ransomware": "Malware",
                   "Spyware": "Malware", "Trojan": "Malware"}


def checksum(path):
    """Mã SHA-256 của một file, dùng để phát hiện file bị đổi."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def get_malware_family(category):
    """'Trojan-Zeus-<mã>-1.raw' -> 'Trojan-Zeus'. Benign giữ nguyên."""
    is_benign = category.eq("Benign")
    family = category.str.split("-").str[:2].str.join("-")
    return category.where(is_benign, family)


# ---------------------------------------------------------------- Đọc dữ liệu
def read_raw_data(path=DATA_PATH):
    """Đọc CSV và tách thành features, nhãn binary, nhãn multiclass và cột Category."""
    table = pd.read_csv(path)
    if not {"Category", "Class"}.issubset(table.columns):
        raise ValueError("Dataset thiếu cột Category hoặc Class")

    multiclass_labels = table["Category"].str.split("-").str[0]
    expected_binary = multiclass_labels.map(BINARY_LABEL_OF)
    if expected_binary.isna().any() or not table["Class"].eq(expected_binary).all():
        raise ValueError("Category không hợp lệ hoặc không khớp với Class")

    features = table.drop(columns=["Category", "Class"])
    if not features.select_dtypes(exclude="number").empty:
        raise ValueError("Features phải là kiểu số")
    if np.isinf(features.to_numpy()).any():
        raise ValueError("Features có giá trị vô cực, cần quyết định cách xử lý")

    return features, table["Class"], multiclass_labels, table["Category"]


# ---------------------------------------------------------------- Gom nhóm liên quan
def find_related_groups(features, category):
    """Gán mỗi dòng một mã nhóm. Hai dòng cùng nhóm nếu có features giống hệt nhau
    HOẶC cùng mã mẫu trong Category (kể cả liên kết bắc cầu qua dòng thứ ba).
    Mã nhóm = chỉ số dòng nhỏ nhất trong nhóm, nên luôn ổn định giữa các lần chạy."""
    sample_token = category.str.extract(SAMPLE_TOKEN_PATTERN)[0].str.lower()
    if (category.ne("Benign") & sample_token.isna()).any():
        raise ValueError("Có Category không đọc được mã mẫu, cần kiểm tra lại")

    feature_key = features.groupby(list(features.columns), dropna=False, sort=False).ngroup()

    # Nối mỗi dòng với dòng đầu tiên có cùng khóa, rồi tìm các cụm dòng được nối với nhau.
    row_count = len(features)
    row_numbers = pd.Series(np.arange(row_count))
    linked_from, linked_to = [], []
    for key in (feature_key, sample_token):
        # Dùng mảng khóa để nhóm theo vị trí dòng, không căn theo index pandas.
        first_row_with_same_key = row_numbers.groupby(key.to_numpy()).transform("min")
        has_key = first_row_with_same_key.notna()
        linked_from.append(row_numbers[has_key].to_numpy())
        linked_to.append(first_row_with_same_key[has_key].astype(int).to_numpy())
    linked_from, linked_to = np.concatenate(linked_from), np.concatenate(linked_to)

    links = coo_matrix((np.ones(len(linked_from)), (linked_from, linked_to)),
                       shape=(row_count, row_count))
    _, component_of_row = connected_components(links, directed=False)

    return row_numbers.groupby(component_of_row).transform("min").to_numpy()


def check_split_table(split_table, group_ids, multiclass_labels):
    """Báo lỗi nếu bộ chia bị thiếu dòng, tách nhóm liên quan, hoặc thiếu lớp."""
    if list(split_table.columns) != ["row_index", "group_id", "split"]:
        raise ValueError("Sai cấu trúc file split")
    if not np.array_equal(split_table["row_index"], np.arange(len(group_ids))):
        raise ValueError("File split phải có đủ mọi dòng, đúng thứ tự gốc")
    if not np.array_equal(split_table["group_id"], group_ids):
        raise ValueError("Nhóm liên quan đã thay đổi so với lúc tạo split")
    if set(split_table["split"]) != set(SPLIT_NAMES):
        raise ValueError("Tên tập không hợp lệ")
    if split_table.groupby("group_id")["split"].nunique().max() != 1:
        raise ValueError("Có nhóm liên quan bị tách sang nhiều tập")
    for split_name in SPLIT_NAMES:
        classes_in_split = set(multiclass_labels[split_table["split"].eq(split_name)])
        if classes_in_split != set(multiclass_labels):
            raise ValueError(f"Tập {split_name} thiếu một lớp")


# ---------------------------------------------------------------- Tạo / đọc bộ chia
def create_split_table(features, multiclass_labels, group_ids):
    """Chia 5 phần theo nhóm: phần 0 = test, phần 1 = validation, còn lại = train."""
    split_of_row = np.full(len(features), "train", dtype=object)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    folds = splitter.split(features, multiclass_labels, group_ids)
    for split_name, (_, row_positions) in zip(["test", "validation"], folds):
        split_of_row[row_positions] = split_name
    return pd.DataFrame({"row_index": np.arange(len(features)),
                         "group_id": group_ids, "split": split_of_row})


def load_master_split(data_path=DATA_PATH, split_dir=SPLIT_DIR):
    """Nếu đã có split thì đọc và kiểm tra; chưa có thì tạo và lưu. Không bao giờ ghi đè split cũ."""
    split_file = Path(split_dir) / "master_split.csv"
    metadata_file = Path(split_dir) / "metadata.json"

    features, binary_labels, multiclass_labels, category = read_raw_data(data_path)
    group_ids = find_related_groups(features, category)
    current_setup = {"seed": RANDOM_SEED, "policy": SPLIT_POLICY,
                     "raw_sha256": checksum(data_path),
                     "features": list(features.columns), "rows": len(features)}

    has_split, has_metadata = split_file.exists(), metadata_file.exists()
    if has_split != has_metadata:
        raise ValueError("Split lưu dở dang, cần kiểm tra thủ công trước khi tạo lại")

    if has_split:
        saved_setup = json.loads(metadata_file.read_text(encoding="utf-8"))
        if any(saved_setup.get(key) != value for key, value in current_setup.items()):
            raise ValueError("Dữ liệu hoặc cấu hình đã đổi; split cũ không bị ghi đè")
        if saved_setup.get("split_sha256") != checksum(split_file):
            raise ValueError("Checksum của file split không khớp")
        split_table = pd.read_csv(split_file)
        check_split_table(split_table, group_ids, multiclass_labels)
    else:
        split_table = create_split_table(features, multiclass_labels, group_ids)
        check_split_table(split_table, group_ids, multiclass_labels)
        Path(split_dir).mkdir(parents=True, exist_ok=True)
        split_table.to_csv(split_file, index=False)
        current_setup.update(split_sha256=checksum(split_file),
                             sklearn=sklearn.__version__, numpy=np.__version__,
                             pandas=pd.__version__)
        metadata_file.write_text(json.dumps(current_setup, indent=2), encoding="utf-8")

    return features, binary_labels, multiclass_labels, split_table


def load_splits(task, path=DATA_PATH, split_dir=SPLIT_DIR):
    """Trả về {'train': (X, y), 'validation': (X, y), 'test': (X, y)}.
    task = 'binary' (Benign/Malware) hoặc 'multiclass' (4 lớp). X chưa được biến đổi."""
    if task not in ("binary", "multiclass"):
        raise ValueError("task phải là 'binary' hoặc 'multiclass'")
    features, binary_labels, multiclass_labels, split_table = load_master_split(path, split_dir)
    labels = binary_labels if task == "binary" else multiclass_labels

    splits = {}
    for split_name in SPLIT_NAMES:
        rows_in_split = split_table["split"].eq(split_name)
        splits[split_name] = (features.loc[rows_in_split], labels.loc[rows_in_split])
    return splits


# ---------------------------------------------------------------- Pipeline tiền xử lý
class OutlierDetector(TransformerMixin, BaseEstimator):
    """Học ngưỡng IQR trên train và đánh dấu outlier. KHÔNG sửa hay xóa dữ liệu."""

    def fit(self, features, labels=None):
        features = check_array(features)
        self.n_features_in_ = features.shape[1]
        q1, q3 = np.quantile(features, [0.25, 0.75], axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - 1.5 * iqr
        self.upper_ = q3 + 1.5 * iqr
        return self

    def _check_input(self, features):
        check_is_fitted(self, ["lower_", "upper_"])
        features = check_array(features)
        if features.shape[1] != self.n_features_in_:
            raise ValueError("Số lượng features đã thay đổi")
        return features

    def outlier_mask(self, features):
        """Ma trận True/False: ô nào nằm ngoài ngưỡng IQR đã học."""
        features = self._check_input(features)
        return (features < self.lower_) | (features > self.upper_)

    def transform(self, features):
        return self._check_input(features)  # chỉ kiểm tra, giữ nguyên giá trị


# Giữ tên cũ để các notebook/test đã viết trước đó vẫn chạy.
IQRDetector = OutlierDetector


def make_preprocessor():
    """Pipeline mới, chưa học gì: điền thiếu bằng median -> đánh dấu outlier -> chuẩn hóa."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("outliers", OutlierDetector()),
        ("scaler", StandardScaler()),
    ])
