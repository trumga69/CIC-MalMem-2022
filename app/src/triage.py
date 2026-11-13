"""Module triage: nạp cấu hình, nạp pipeline, xác thực dữ liệu CSV và thực thi suy luận.

Hỗ trợ 4 models theo cấu hình model/configs/inference.json:
- Task A (Binary): 02_binary_lightgbm (mặc định, tau=0.95), 01_binary_random_forest (tau=0.95)
- Task B (Multiclass): 04_multiclass_lightgbm (mặc định, tau=0.80), 03_multiclass_random_forest (tau=0.70)
"""
import collections
import csv
import io
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd

# Bỏ qua cảnh báo phiên bản minor không ảnh hưởng khi unpickle
warnings.filterwarnings("ignore", category=UserWarning)
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass

# Import preprocessing để hỗ trợ nạp IQRDetector / OutlierDetector trong Pipeline
import preprocessing  # noqa: F401


def get_project_root() -> Path:
    """Trả về đường dẫn tuyệt đối đến thư mục gốc của dự án (PYTHON)."""
    curr = Path(__file__).resolve().parent
    for p in [curr] + list(curr.parents):
        if (p / "model" / "configs" / "inference.json").exists():
            return p
    return Path(__file__).resolve().parents[1]


def load_inference_config(config_path: Union[str, Path] = "model/configs/inference.json") -> Dict[str, Any]:
    """Đọc cấu hình inference.json từ đường dẫn tương đối hoặc tuyệt đối."""
    root = get_project_root()
    p = Path(config_path)
    if not p.is_absolute():
        p = root / p
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình inference: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model_report(report_path: Union[str, Path] = "model/reports/model_report.json") -> Dict[str, Any]:
    """Đọc báo cáo chuẩn hóa model_report.json để đối chiếu chỉ số."""
    root = get_project_root()
    p = Path(report_path)
    if not p.is_absolute():
        p = root / p
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def get_available_models(config: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    """Trả về danh mục các model đã được cấu hình trong inference.json."""
    if config is None:
        config = load_inference_config()
    return config.get("models", {})


def load_pipeline_for_model(
    model_id: str,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[Any, Dict[str, Any]]:
    """Nạp fitted pipeline tương ứng với model_id từ model/artifacts/*.joblib.

    Kiểm tra:
    1. Tồn tại artifact trên đĩa.
    2. classes_ khớp với nhiệm vụ (Binary: Benign/Malware, Multiclass: 4 lớp).
    3. feature_names_in_ có đúng 55 features.
    """
    if config is None:
        config = load_inference_config()

    models_cfg = config.get("models", {})
    if model_id not in models_cfg:
        raise ValueError(
            f"Model ID '{model_id}' không tồn tại trong cấu hình inference.json. "
            f"Các model hợp lệ: {list(models_cfg.keys())}"
        )

    model_info = dict(models_cfg[model_id])
    model_info["model_id"] = model_id

    root = get_project_root()
    rel_path = model_info["artifact"]
    artifact_path = root / rel_path

    if not artifact_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file artifact tại: {artifact_path}")

    pipeline = joblib.load(artifact_path)

    # Kiểm tra số lượng feature
    if hasattr(pipeline, "feature_names_in_"):
        n_feats = len(pipeline.feature_names_in_)
        expected_n = config.get("feature_count", 55)
        if n_feats != expected_n:
            raise ValueError(f"Pipeline {model_id} có {n_feats} đặc trưng, kỳ vọng {expected_n}.")

    # Kiểm tra tập nhãn classes_
    actual_classes = list(pipeline.classes_)
    task = model_info.get("task")
    if task == "binary":
        expected_classes = ["Benign", "Malware"]
        if actual_classes != expected_classes:
            raise ValueError(f"Nhãn binary của {model_id} không khớp: {actual_classes} vs {expected_classes}")
    elif task == "multiclass":
        expected_classes = ["Benign", "Ransomware", "Spyware", "Trojan"]
        if actual_classes != expected_classes:
            raise ValueError(f"Nhãn multiclass của {model_id} không khớp: {actual_classes} vs {expected_classes}")

    return pipeline, model_info


def validate_csv_input(
    input_source: Union[bytes, str, io.StringIO, io.BytesIO, Any],
    expected_features: List[str],
    auto_drop_labels: bool = True
) -> pd.DataFrame:
    """Kiểm tra và chuẩn hóa bảng dữ liệu đầu vào theo đúng 55 features của model.

    Các bước xác thực:
    1. Kiểm tra bảng rỗng / file rỗng.
    2. Kiểm tra header thô để phát hiện cột trùng tên trước khi pandas tự đổi tên.
    3. Hỗ trợ tự động loại bỏ cột nhãn (Class, Category, multiclass...) nếu người dùng tải raw dataset.
    4. Kiểm tra cột thiếu, cột dư so với pipeline.feature_names_in_.
    5. Sắp xếp lại đúng thứ tự 55 cột của model (cho phép thứ tự cột bị đảo).
    6. Kiểm tra ký tự chữ rác (non-numeric text). Giữ nguyên ô trống (NaN) để pipeline imputer tự điền.
    7. Kiểm tra giá trị vô cực (infinity, -inf).
    """
    if isinstance(input_source, (str, Path)):
        p = Path(input_source)
        if p.exists() and p.is_file():
            text = p.read_text(encoding="utf-8", errors="replace")
        else:
            text = str(input_source)
    elif isinstance(input_source, bytes):
        text = input_source.decode("utf-8", errors="replace")
    elif hasattr(input_source, "read"):
        content = input_source.read()
        if isinstance(content, bytes):
            text = content.decode("utf-8", errors="replace")
        else:
            text = str(content)
    elif isinstance(input_source, pd.DataFrame):
        return _validate_dataframe(input_source, expected_features, auto_drop_labels=auto_drop_labels)
    else:
        raise ValueError("Định dạng dữ liệu đầu vào không được hỗ trợ.")

    text_stripped = text.strip()
    if not text_stripped:
        raise ValueError("Tập tin CSV rỗng: không có dữ liệu hoặc tiêu đề.")

    # 1. Đọc header thô để bắt cột trùng tên
    reader = csv.reader(io.StringIO(text_stripped))
    try:
        raw_header = next(reader)
    except StopIteration:
        raise ValueError("Tập tin CSV không đọc được dòng tiêu đề (header).")

    if not raw_header or all(not h.strip() for h in raw_header):
        raise ValueError("Dòng tiêu đề của CSV rỗng.")

    clean_header = [h.strip() for h in raw_header]
    col_counts = collections.Counter(clean_header)
    duplicates = [col for col, count in col_counts.items() if count > 1]
    if duplicates:
        raise ValueError(
            f"CSV không hợp lệ: chứa {len(duplicates)} tên cột bị trùng lặp: {duplicates}."
        )

    # 2. Đọc vào DataFrame
    try:
        df = pd.read_csv(io.StringIO(text_stripped))
    except Exception as e:
        raise ValueError(f"Không thể phân tích cú pháp CSV: {str(e)}")

    if len(df) == 0:
        raise ValueError("Bảng dữ liệu rỗng: CSV có header nhưng không chứa dòng dữ liệu nào.")

    return _validate_dataframe(df, expected_features, auto_drop_labels=auto_drop_labels)


def _validate_dataframe(
    df: pd.DataFrame,
    expected_features: List[str],
    auto_drop_labels: bool = True
) -> pd.DataFrame:
    """Kiểm tra chi tiết schema và giá trị của DataFrame."""
    cols = list(df.columns)

    # Kiểm tra trùng tên cột nếu truyền trực tiếp df
    col_counts = collections.Counter(cols)
    duplicates = [col for col, count in col_counts.items() if count > 1]
    if duplicates:
        raise ValueError(f"Dữ liệu chứa cột trùng lặp: {duplicates}")

    # Tự động loại bỏ các cột nhãn Ground-Truth / Metadata phổ biến nếu có
    KNOWN_LABEL_COLS = {"Class", "Category", "multiclass", "Filename", "row_index", "group_id"}
    if auto_drop_labels:
        label_extras = [c for c in cols if c in KNOWN_LABEL_COLS and c not in expected_features]
        if label_extras:
            df = df.drop(columns=label_extras)
            cols = list(df.columns)

    missing = [c for c in expected_features if c not in cols]
    extra = [c for c in cols if c not in expected_features]

    if missing or extra:
        err_parts = []
        if missing:
            err_parts.append(f"Thiếu {len(missing)} cột: {missing}")
        if extra:
            err_parts.append(f"Dư {len(extra)} cột không thuộc schema 55 features: {extra}")
        raise ValueError("Không khớp schema 55 features: " + "; ".join(err_parts))

    # Sắp xếp lại đúng thứ tự model mong đợi (hỗ trợ cột bị đảo thứ tự)
    X = df[expected_features].copy()

    # Kiểm tra chữ rác (non-numeric strings)
    # Giữ ô trống hoặc NaN để SimpleImputer trong Pipeline xử lý
    for col in expected_features:
        series = X[col]
        coerced = pd.to_numeric(series, errors="coerce")
        orig_non_null = series.notna() & (series.astype(str).str.strip().str.lower() != "nan") & (series.astype(str).str.strip() != "")
        garbage_mask = orig_non_null & coerced.isna()
        if garbage_mask.any():
            sample_bad = series[garbage_mask].iloc[:3].tolist()
            raise ValueError(
                f"Cột '{col}' chứa giá trị chữ rác không phải số: {sample_bad}."
            )
        X[col] = coerced.astype(float)

    # Kiểm tra giá trị vô cực
    arr = X.to_numpy(dtype=float)
    if np.isinf(arr).any():
        inf_cols = [c for c in expected_features if np.isinf(X[c].dropna().to_numpy()).any()]
        raise ValueError(
            f"Dữ liệu chứa giá trị vô cực (infinity / -inf) tại các cột: {inf_cols}."
        )

    return X


def predict_batch(
    pipeline: Any,
    X_validated: pd.DataFrame,
    tau: float,
    rejection_label: str = "Uncertain"
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Chạy suy luận qua pipeline.predict_proba trên dữ liệu đã xác thực 55 features.

    Quy tắc nghiệp vụ:
    - So sánh p_max >= tau trước khi làm tròn số.
    - Nếu p_max >= tau: chấp nhận nhãn argmax (lớp ứng viên).
    - Nếu p_max < tau: gán nhãn Uncertain, vẫn lưu và hiển thị lớp ứng viên cùng toàn bộ điểm số.
    """
    probas = pipeline.predict_proba(X_validated)
    classes = list(pipeline.classes_)

    argmax_indices = np.argmax(probas, axis=1)
    p_max_values = probas[np.arange(len(probas)), argmax_indices]
    candidate_classes = [classes[idx] for idx in argmax_indices]

    # So sánh p_max >= tau TRƯỚC KHI làm tròn
    decisions = [
        candidate_classes[i] if p_max_values[i] >= tau else rejection_label
        for i in range(len(probas))
    ]

    records = []
    for i in range(len(probas)):
        rec = {
            "Dòng": i,
            "Kết quả": decisions[i],
            "Lớp ứng viên": candidate_classes[i],
            "p_max": float(p_max_values[i]),
            "Ngưỡng tau": float(tau),
            "Đạt ngưỡng": bool(p_max_values[i] >= tau)
        }
        for j, cls_name in enumerate(classes):
            rec[f"Score_{cls_name}"] = float(probas[i, j])
        records.append(rec)

    df_results = pd.DataFrame(records)
    return df_results, probas
