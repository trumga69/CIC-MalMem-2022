"""Module triage: nạp config, nạp và kiểm tra pipeline, xác thực dữ liệu CSV và suy luận."""
import collections
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import joblib
import numpy as np
import pandas as pd

# Cần import preprocessing để unpickle IQRDetector / OutlierDetector
import preprocessing  # noqa: F401


def get_project_root() -> Path:
    """Trả về đường dẫn tuyệt đối đến thư mục gốc của dự án."""
    return Path(__file__).resolve().parents[1]


def load_operating_config(config_path: Union[str, Path] = "configs/operating_threshold.json") -> Dict[str, Any]:
    """Đọc file operating_threshold.json từ đường dẫn tương đối hoặc tuyệt đối."""
    root = get_project_root()
    p = Path(config_path)
    if not p.is_absolute():
        p = root / p
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pipeline_for_task(task: str, config: Dict[str, Any] = None) -> Tuple[Any, Dict[str, Any]]:
    """Nạp LightGBM pipeline theo task ('binary' hoặc 'multiclass').

    Xác minh đường dẫn từ project root, kiểm tra SHA-256 và danh sách classes.
    """
    if task not in ("binary", "multiclass"):
        raise ValueError(f"Task không hợp lệ: '{task}'. Chỉ chấp nhận 'binary' hoặc 'multiclass'.")

    if config is None:
        config = load_operating_config()

    root = get_project_root()
    model_info = config["selected_models"][task]
    rel_path = model_info["pipeline_path"]
    pipeline_file = root / rel_path

    if not pipeline_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file pipeline tại: {pipeline_file}")

    # 1. Kiểm tra SHA-256
    file_bytes = pipeline_file.read_bytes()
    computed_sha256 = hashlib.sha256(file_bytes).hexdigest()
    expected_sha256 = model_info["pipeline_sha256"]
    if computed_sha256.lower() != expected_sha256.lower():
        raise ValueError(
            f"Xác minh SHA-256 thất bại cho {task} pipeline!\n"
            f"Kỳ vọng: {expected_sha256}\n"
            f"Thực tế: {computed_sha256}"
        )

    # 2. Nạp pipeline bằng joblib
    pipeline = joblib.load(pipeline_file)

    # 3. Kiểm tra classes_
    expected_classes = list(model_info["classes"])
    actual_classes = list(pipeline.classes_)
    if actual_classes != expected_classes:
        raise ValueError(
            f"Danh sách classes của pipeline không khớp với cấu hình!\n"
            f"Kỳ vọng: {expected_classes}\n"
            f"Thực tế: {actual_classes}"
        )

    return pipeline, model_info


def validate_csv_input(
    input_source: Union[bytes, str, io.StringIO, io.BytesIO, Any],
    expected_features: List[str]
) -> pd.DataFrame:
    """Kiểm tra và chuẩn hóa bảng dữ liệu đầu vào theo đúng 55 features của model.

    Các bước kiểm tra:
    1. Kiểm tra bảng rỗng / file rỗng.
    2. Kiểm tra dòng header thô để phát hiện cột trùng tên trước khi pandas tự đổi tên.
    3. Kiểm tra cột thiếu, cột dư so với pipeline.feature_names_in_.
    4. Sắp xếp lại đúng thứ tự cột của model (cho phép thứ tự đảo).
    5. Kiểm tra ký tự chữ rác (non-numeric text). Giữ nguyên ô trống (NaN) để imputer xử lý.
    6. Kiểm tra giá trị vô cực (infinity, -inf).
    """
    # Lấy nội dung text
    if isinstance(input_source, bytes):
        text = input_source.decode("utf-8", errors="replace")
    elif isinstance(input_source, str):
        # Có thể là chuỗi CSV hoặc đường dẫn file
        p = Path(input_source)
        if p.exists() and p.is_file():
            text = p.read_text(encoding="utf-8", errors="replace")
        else:
            text = input_source
    elif hasattr(input_source, "read"):
        content = input_source.read()
        if isinstance(content, bytes):
            text = content.decode("utf-8", errors="replace")
        else:
            text = str(content)
    elif isinstance(input_source, pd.DataFrame):
        # Trực tiếp xử lý dataframe
        return _validate_dataframe(input_source, expected_features)
    else:
        raise ValueError("Định dạng dữ liệu đầu vào không được hỗ trợ.")

    text_stripped = text.strip()
    if not text_stripped:
        raise ValueError("Tập tin CSV rỗng: không có dữ liệu hoặc tiêu đề.")

    # Kiểm tra header thô bằng csv.reader
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

    # Đọc vào DataFrame
    try:
        df = pd.read_csv(io.StringIO(text_stripped))
    except Exception as e:
        raise ValueError(f"Không thể phân tích cú pháp CSV: {str(e)}")

    if len(df) == 0:
        raise ValueError("Bảng dữ liệu rỗng: CSV có header nhưng không chứa dòng dữ liệu nào.")

    return _validate_dataframe(df, expected_features)


def _validate_dataframe(df: pd.DataFrame, expected_features: List[str]) -> pd.DataFrame:
    """Kiểm tra schema và giá trị của DataFrame."""
    cols = list(df.columns)

    # Kiểm tra cột trùng trong DataFrame nếu truyền trực tiếp df
    col_counts = collections.Counter(cols)
    duplicates = [col for col, count in col_counts.items() if count > 1]
    if duplicates:
        raise ValueError(f"Dữ liệu chứa cột trùng lặp: {duplicates}")

    missing = [c for c in expected_features if c not in cols]
    extra = [c for c in cols if c not in expected_features]

    if missing or extra:
        err_parts = []
        if missing:
            err_parts.append(f"Thiếu {len(missing)} cột: {missing}")
        if extra:
            err_parts.append(f"Dư {len(extra)} cột không thuộc schema 55 features: {extra}")
        raise ValueError("Không khớp schema 55 features: " + "; ".join(err_parts))

    # Sắp xếp lại các cột đúng theo thứ tự model yêu cầu
    X = df[expected_features].copy()

    # Kiểm tra chữ rác (non-numeric strings)
    # NaN hoặc chuỗi trống được chấp nhận để Pipeline imputer điền median
    for col in expected_features:
        series = X[col]
        # Coerce sang số
        coerced = pd.to_numeric(series, errors="coerce")
        # Tìm các ô ban đầu không rỗng nhưng bị ép thành NaN -> chữ rác
        orig_non_null = series.notna() & (series.astype(str).str.strip().str.lower() != "nan") & (series.astype(str).str.strip() != "")
        garbage_mask = orig_non_null & coerced.isna()
        if garbage_mask.any():
            sample_bad = series[garbage_mask].iloc[:3].tolist()
            raise ValueError(
                f"Cột '{col}' chứa giá trị chữ rác không phải số: {sample_bad}."
            )
        X[col] = coerced.astype(float)

    # Kiểm tra infinity
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
    """Chạy suy luận qua pipeline.predict_proba trên dữ liệu đã xác thực.

    Áp dụng quy tắc:
    - So sánh p_max >= tau trước khi làm tròn.
    - Nếu p_max >= tau: nhận nhãn argmax.
    - Nếu p_max < tau: gán Uncertain, vẫn giữ scores và ghi nhận lớp ứng viên.
    """
    probas = pipeline.predict_proba(X_validated)
    classes = list(pipeline.classes_)

    argmax_indices = np.argmax(probas, axis=1)
    p_max_values = probas[np.arange(len(probas)), argmax_indices]
    candidate_classes = [classes[idx] for idx in argmax_indices]

    # So sánh p_max >= tau trước khi làm tròn
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
            "Ngưỡng tau": tau,
            "Đạt ngưỡng": bool(p_max_values[i] >= tau)
        }
        for j, cls_name in enumerate(classes):
            rec[f"Score_{cls_name}"] = float(probas[i, j])
        records.append(rec)

    df_results = pd.DataFrame(records)
    return df_results, probas
