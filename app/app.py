"""Web Demo Streamlit - Topic 05: CIC-MalMem-2022 Forensic Memory Analysis.

Tệp app tự chứa đầy đủ (Self-Contained) cho TV5 bàn giao nhóm trưởng:
- Hỗ trợ toàn diện 4 mô hình theo model/configs/inference.json:
  + Task A (Binary Triage): 02_binary_lightgbm (Mặc định, tau=0.95), 01_binary_random_forest (tau=0.95)
  + Task B (Multiclass Threat Family): 04_multiclass_lightgbm (Mặc định, tau=0.80), 03_multiclass_random_forest (tau=0.70)
- Tự động dò tìm thư mục gốc dự án (hoạt động tốt dù đặt ở root hay trong thư mục app/)
- Xác thực hợp đồng dữ liệu nghiêm ngặt (55 features, bắt lỗi rỗng, trùng cột header, thiếu/dư cột, phi số, vô cực, tự động loại bỏ cột nhãn Class/Category/multiclass nếu nạp file thô)
- Suy luận xác suất, so sánh p_max >= tau trước khi làm tròn số, cơ chế từ chối Uncertain
- Giải thích cục bộ Local Top-3:
  + LightGBM: pred_contrib, đảo dấu Benign cho Binary, trích xuất khối cho Multiclass, kiểm chứng sai số khớp raw margin (|Δ| < 1e-4)
  + Random Forest: thông báo trạng thái chưa hỗ trợ theo đúng đặc tả bàn giao
- Giao diện High-Tech White SOC Cyberpunk với font Chakra Petch & JetBrains Mono hỗ trợ tiếng Việt.
"""
import collections
import csv
import io
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Bỏ qua các cảnh báo minor scikit-learn unpickle
warnings.filterwarnings("ignore", category=UserWarning)
try:
    from sklearn.exceptions import InconsistentVersionWarning
    warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
except ImportError:
    pass


# =============================================================================
# 1. HỆ THỐNG XÁC ĐỊNH PROJECT ROOT & IMPORT TƯƠNG THÍCH
# =============================================================================
def get_project_root() -> Path:
    """Tìm thư mục gốc của dự án (nơi chứa model/configs/inference.json)."""
    curr = Path(__file__).resolve().parent
    for p in [curr] + list(curr.parents):
        if (p / "model" / "configs" / "inference.json").exists():
            return p
    return curr

ROOT_DIR = get_project_root()
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
prep_dir = ROOT_DIR / "preprocessing_data"
if prep_dir.exists() and str(prep_dir) not in sys.path:
    sys.path.insert(0, str(prep_dir))

try:
    import preprocessing  # noqa: F401
except ImportError:
    pass


# =============================================================================
# 2. MODULE TRIAGE & NẠP MODEL
# =============================================================================
def load_inference_config(config_path: Union[str, Path] = "model/configs/inference.json") -> Dict[str, Any]:
    """Đọc file cấu hình inference.json."""
    p = Path(config_path)
    if not p.is_absolute():
        p = ROOT_DIR / p
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model_report(report_path: Union[str, Path] = "model/reports/model_report.json") -> Dict[str, Any]:
    """Đọc file báo cáo chuẩn hóa model_report.json."""
    p = Path(report_path)
    if not p.is_absolute():
        p = ROOT_DIR / p
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pipeline_for_model(model_id: str, config: Optional[Dict[str, Any]] = None) -> Tuple[Any, Dict[str, Any]]:
    """Nạp fitted pipeline và thông số của model_id."""
    if config is None:
        config = load_inference_config()

    models_cfg = config.get("models", {})
    if model_id not in models_cfg:
        raise ValueError(f"Model ID '{model_id}' không tồn tại trong cấu hình inference.json.")

    model_info = dict(models_cfg[model_id])
    model_info["model_id"] = model_id
    artifact_path = ROOT_DIR / model_info["artifact"]

    if not artifact_path.exists():
        raise FileNotFoundError(f"Không tìm thấy artifact tại: {artifact_path}")

    pipeline = joblib.load(artifact_path)
    return pipeline, model_info


def validate_csv_input(
    input_source: Union[bytes, str, Path, io.StringIO, io.BytesIO, Any],
    expected_features: List[str],
    auto_drop_labels: bool = True
) -> pd.DataFrame:
    """Xác thực và chuẩn hóa CSV theo đúng 55 features của model."""
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
        text = content.decode("utf-8", errors="replace") if isinstance(content, bytes) else str(content)
    elif isinstance(input_source, pd.DataFrame):
        return _validate_dataframe(input_source, expected_features, auto_drop_labels=auto_drop_labels)
    else:
        raise ValueError("Định dạng dữ liệu đầu vào không được hỗ trợ.")

    text_stripped = text.strip()
    if not text_stripped:
        raise ValueError("Tập tin CSV rỗng: không có dữ liệu hoặc tiêu đề.")

    reader = csv.reader(io.StringIO(text_stripped))
    try:
        raw_header = next(reader)
    except StopIteration:
        raise ValueError("Tập tin CSV không đọc được dòng tiêu đề.")

    if not raw_header or all(not h.strip() for h in raw_header):
        raise ValueError("Dòng tiêu đề của CSV rỗng.")

    clean_header = [h.strip() for h in raw_header]
    col_counts = collections.Counter(clean_header)
    duplicates = [col for col, count in col_counts.items() if count > 1]
    if duplicates:
        raise ValueError(f"CSV không hợp lệ: chứa {len(duplicates)} tên cột bị trùng lặp: {duplicates}.")

    try:
        df = pd.read_csv(io.StringIO(text_stripped))
    except Exception as e:
        raise ValueError(f"Không thể phân tích cú pháp CSV: {str(e)}")

    if len(df) == 0:
        raise ValueError("Bảng dữ liệu rỗng: CSV có header nhưng không chứa dòng dữ liệu nào.")

    return _validate_dataframe(df, expected_features, auto_drop_labels=auto_drop_labels)


def _validate_dataframe(df: pd.DataFrame, expected_features: List[str], auto_drop_labels: bool = True) -> pd.DataFrame:
    """Kiểm tra chi tiết DataFrame."""
    cols = list(df.columns)
    col_counts = collections.Counter(cols)
    duplicates = [col for col, count in col_counts.items() if count > 1]
    if duplicates:
        raise ValueError(f"Dữ liệu chứa cột trùng lặp: {duplicates}")

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
            err_parts.append(f"Dư {len(extra)} cột: {extra}")
        raise ValueError("Không khớp schema 55 features: " + "; ".join(err_parts))

    X = df[expected_features].copy()

    for col in expected_features:
        series = X[col]
        coerced = pd.to_numeric(series, errors="coerce")
        orig_non_null = series.notna() & (series.astype(str).str.strip().str.lower() != "nan") & (series.astype(str).str.strip() != "")
        garbage_mask = orig_non_null & coerced.isna()
        if garbage_mask.any():
            sample_bad = series[garbage_mask].iloc[:3].tolist()
            raise ValueError(f"Cột '{col}' chứa giá trị chữ rác không phải số: {sample_bad}.")
        X[col] = coerced.astype(float)

    arr = X.to_numpy(dtype=float)
    if np.isinf(arr).any():
        inf_cols = [c for c in expected_features if np.isinf(X[c].dropna().to_numpy()).any()]
        raise ValueError(f"Dữ liệu chứa giá trị vô cực (infinity / -inf) tại các cột: {inf_cols}.")

    return X


def predict_batch(
    pipeline: Any,
    X_validated: pd.DataFrame,
    tau: float,
    rejection_label: str = "Uncertain"
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Suy luận xác suất và áp dụng cơ chế từ chối p_max >= tau trước khi làm tròn."""
    probas = pipeline.predict_proba(X_validated)
    classes = list(pipeline.classes_)

    argmax_indices = np.argmax(probas, axis=1)
    p_max_values = probas[np.arange(len(probas)), argmax_indices]
    candidate_classes = [classes[idx] for idx in argmax_indices]

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

    return pd.DataFrame(records), probas


# =============================================================================
# 3. MODULE GIẢI THÍCH CỤC BỘ LOCAL TOP-3 & KIỂM CHỨNG PARITY
# =============================================================================
def compute_local_top3(
    pipeline: Any,
    row_features: pd.DataFrame,
    task: str,
    target_class_override: Optional[str] = None
) -> Dict[str, Any]:
    """Tính giải thích cục bộ Top-3 đặc trưng cho 1 dòng dữ liệu."""
    if len(row_features) != 1:
        row_features = row_features.iloc[[0]]

    classifier = pipeline.named_steps.get("classifier", None)
    if classifier is None:
        return {"supported": False, "message": "Không tìm thấy classifier step trong pipeline."}

    if not hasattr(classifier, "booster_"):
        return {
            "supported": False,
            "message": (
                "Trạng thái giải thích cục bộ: CHƯA HỖ TRỢ đối với mô hình Random Forest. "
                "Tính năng phân rã đóng góp cục bộ (Local Top-3) hiện chỉ được hỗ trợ trên thuật toán LightGBM "
                "thông qua cơ chế Tree Booster pred_contrib."
            )
        }

    feature_names = list(pipeline.feature_names_in_)
    classes = list(pipeline.classes_)

    probas = pipeline.predict_proba(row_features)[0]
    argmax_idx = int(np.argmax(probas))
    candidate_class = classes[argmax_idx]

    target_class = target_class_override if target_class_override is not None else candidate_class
    target_class_idx = classes.index(target_class)

    preprocessor = pipeline.named_steps["preprocessing"]
    X_transformed = preprocessor.transform(row_features)

    booster = classifier.booster_
    raw_scores = booster.predict(X_transformed, raw_score=True)[0]
    contribs = booster.predict(X_transformed, pred_contrib=True)[0]

    if task == "binary":
        feat_contribs = contribs[:55].copy()
        baseline = float(contribs[55])
        raw_malware = float(raw_scores)
        if target_class == "Benign":
            feat_contribs = -feat_contribs
            baseline = -baseline
            target_raw_score = -raw_malware
        else:
            target_raw_score = raw_malware

    elif task == "multiclass":
        contribs_matrix = contribs.reshape(len(classes), 56)
        cls_block = contribs_matrix[target_class_idx]
        feat_contribs = cls_block[:55].copy()
        baseline = float(cls_block[55])
        target_raw_score = float(raw_scores[target_class_idx])

    else:
        raise ValueError(f"Task không hỗ trợ: '{task}'.")

    sum_contributions = float(np.sum(feat_contribs))
    total_reconstructed = sum_contributions + baseline
    parity_diff = abs(total_reconstructed - target_raw_score)
    is_parity_ok = parity_diff < 1e-4

    orig_values = row_features.iloc[0].values
    df_contribs = pd.DataFrame({
        "feature": feature_names,
        "original_value": orig_values,
        "contribution": feat_contribs,
        "abs_contrib": np.abs(feat_contribs)
    })
    df_sorted = df_contribs.sort_values(by="abs_contrib", ascending=False).reset_index(drop=True)
    top3_df = df_sorted.head(3).copy()

    impact_labels = []
    for c in top3_df["contribution"]:
        if c > 0:
            impact_labels.append("Ủng hộ (Tăng xác suất)")
        elif c < 0:
            impact_labels.append("Phản đối (Giảm xác suất)")
        else:
            impact_labels.append("Trung tính")
    top3_df["impact"] = impact_labels

    return {
        "supported": True,
        "target_class": target_class,
        "candidate_class": candidate_class,
        "target_class_idx": target_class_idx,
        "classes": classes,
        "probas": probas,
        "p_max": float(np.max(probas)),
        "target_raw_score": target_raw_score,
        "baseline": baseline,
        "sum_contributions": sum_contributions,
        "total_reconstructed": total_reconstructed,
        "parity_diff": parity_diff,
        "is_parity_ok": is_parity_ok,
        "top3": top3_df,
        "all_contributions": df_sorted
    }


# =============================================================================
# 4. GIAO DIỆN STREAMLIT (UI / UX HIGH-TECH WHITE SOC)
# =============================================================================
st.set_page_config(
    page_title="CIC-MalMem-2022 // Memory Forensics Radar",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

CYBER_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:wght@400;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Chakra Petch', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #0f172a;
}

code, pre, .mono-font {
    font-family: 'JetBrains Mono', monospace !important;
}

.stApp {
    background-color: #f8fafc;
    background-image: 
        radial-gradient(#cbd5e1 1px, transparent 1px),
        linear-gradient(to right, #f1f5f9 1px, transparent 1px);
    background-size: 32px 32px;
}

.cyber-banner {
    background: linear-gradient(135deg, #ffffff 0%, #f0fdf4 50%, #f0f9ff 100%);
    border: 1.5px solid #0284c7;
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 24px;
    box-shadow: 0 4px 20px -2px rgba(2, 132, 199, 0.12);
    position: relative;
    overflow: hidden;
}

.cyber-banner::before {
    content: "[TOPIC-05 // CIC-MALMEM-2022 // FORENSIC TELEMETRY CONSOLE]";
    position: absolute;
    top: 6px;
    right: 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #0284c7;
    font-weight: 700;
    letter-spacing: 1.5px;
}

.cyber-title {
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    color: #0f172a;
    margin: 4px 0 6px 0;
    text-transform: uppercase;
}

.cyber-subtitle {
    font-size: 0.95rem;
    color: #475569;
    font-weight: 500;
    letter-spacing: 0.5px;
}

.cyber-badges {
    display: flex;
    gap: 8px;
    margin-top: 12px;
    flex-wrap: wrap;
}

.cyber-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    padding: 3px 10px;
    border-radius: 4px;
    font-weight: 600;
    letter-spacing: 0.5px;
    display: inline-flex;
    align-items: center;
}

.badge-radar { background: #ecfdf5; color: #047857; border: 1px solid #10b981; }
.badge-core { background: #f0f9ff; color: #0369a1; border: 1px solid #0284c7; }
.badge-tau { background: #fefce8; color: #b45309; border: 1px solid #f59e0b; }
.badge-task { background: #f5f3ff; color: #6d28d9; border: 1px solid #8b5cf6; }

.cyber-card {
    background: #ffffff;
    border: 1px solid #cbd5e1;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 20px;
    box-shadow: 0 2px 8px -1px rgba(0, 0, 0, 0.05);
}

.cyber-section-header {
    border-left: 4px solid #0284c7;
    padding-left: 12px;
    margin: 20px 0 12px 0;
    font-size: 1.15rem;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: 0.8px;
}

.metric-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-top: 3px solid #0284c7;
    border-radius: 6px;
    padding: 14px 18px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.metric-box-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.2;
}
.metric-box-lbl {
    font-size: 0.8rem;
    color: #64748b;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 4px;
}
</style>
"""
st.html(CYBER_CSS)


@st.cache_resource
def get_cached_pipeline(model_id: str):
    """Nạp và cache model pipeline."""
    cfg = load_inference_config()
    return load_pipeline_for_model(model_id, cfg)


config = load_inference_config()
model_report = load_model_report()


# =============================================================================
# 5. SIDEBAR: ĐIỀU KHIỂN & LỰA CHỌN 4 MODELS
# =============================================================================
st.sidebar.markdown("### 🎛️ ĐIỀU KHIỂN & CẤU HÌNH TÁC VỤ")

task_choice = st.sidebar.radio(
    "1. Chọn nhiệm vụ điều tra (Investigation Task):",
    options=["Task A (Binary Triage)", "Task B (Multiclass Threat Family)"],
    index=0
)
task_key = "binary" if "Binary" in task_choice else "multiclass"

algo_choice = st.sidebar.radio(
    "2. Chọn thuật toán phân loại (Algorithm Core):",
    options=["LightGBM (Mặc định // Primary)", "Random Forest (Tùy chọn // Secondary)"],
    index=0
)
algo_key = "lightgbm" if "LightGBM" in algo_choice else "random_forest"

if task_key == "binary":
    selected_model_id = "02_binary_lightgbm" if algo_key == "lightgbm" else "01_binary_random_forest"
else:
    selected_model_id = "04_multiclass_lightgbm" if algo_key == "lightgbm" else "03_multiclass_random_forest"

try:
    pipeline, model_info = get_cached_pipeline(selected_model_id)
    tau_value = float(model_info["tau"])
    rejection_label = config.get("uncertain_label", "Uncertain")
    expected_features = list(pipeline.feature_names_in_)
    pipeline_classes = list(pipeline.classes_)
except Exception as e:
    st.sidebar.error(f"Lỗi khi nạp model {selected_model_id}: {e}")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔒 THÔNG SỐ MODEL ĐANG CHẠY")
sidebar_info_html = f"""
<div style="background: #ffffff; border: 1px solid #cbd5e1; border-radius: 6px; padding: 12px 14px; font-size: 0.85rem; line-height: 1.6;">
    <div><b>Model ID:</b> <code style="color: #0284c7; font-weight: bold;">{selected_model_id}</code></div>
    <div><b>Thuật toán:</b> <span style="font-weight: 600;">{'LightGBM' if algo_key == 'lightgbm' else 'Random Forest'}</span></div>
    <div><b>Nhiệm vụ:</b> <span>{'Nhị phân (Binary)' if task_key == 'binary' else 'Đa lớp (Multiclass)'}</span></div>
    <div><b>Ngưỡng từ chối τ:</b> <code style="background: #fefce8; color: #b45309; padding: 2px 6px; border-radius: 4px; font-weight: bold;">{tau_value:.2f}</code></div>
    <div><b>Nhãn từ chối:</b> <span style="font-weight: 600; color: #b45309;">{rejection_label}</span></div>
    <div><b>Số features yêu cầu:</b> <code style="font-weight: bold;">{len(expected_features)}</code></div>
    <div><b>Tập classes:</b> <span style="color: #475569;">{', '.join(pipeline_classes)}</span></div>
</div>
"""
st.sidebar.html(sidebar_info_html)

if selected_model_id in model_report:
    rep_item = model_report[selected_model_id]
    test_metrics = rep_item.get("test", {})
    at_tau = test_metrics.get("at_selected_tau", {})
    acc_all = test_metrics.get("accuracy", 0.0)
    macro_f1 = test_metrics.get("classification_report", {}).get("macro avg", {}).get("f1-score", 0.0)
    cov = at_tau.get("coverage", 0.0)
    acc_acc = at_tau.get("accepted_accuracy", 0.0)
    unc_count = test_metrics.get("uncertain", 0)

    rep_html = f"""
    <div style="margin-top: 10px; background: #f8fafc; border: 1px dashed #94a3b8; border-radius: 6px; padding: 10px 12px; font-size: 0.8rem;">
        <div style="font-weight: 700; color: #334155; margin-bottom: 4px; text-transform: uppercase;">📊 Chuẩn Test (8.791 dòng)</div>
        <div>• Accuracy toàn test: <b>{acc_all*100:.2f}%</b></div>
        <div>• Macro-F1: <b>{macro_f1:.4f}</b></div>
        <div>• Coverage (τ={tau_value}): <b>{cov*100:.2f}%</b></div>
        <div>• Accuracy trên mẫu nhận: <b>{acc_acc*100:.2f}%</b></div>
        <div>• Số ca Uncertain: <b>{unc_count:,}</b></div>
    </div>
    """
    st.sidebar.html(rep_html)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ TẢI NHANH MẪU KIỂM THỬ")

if "demo_source" not in st.session_state:
    st.session_state.demo_source = None

col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    if st.button("🚀 Nạp 5 mẫu chuẩn"):
        demo_55_path = ROOT_DIR / "data" / "demo" / "demo_validation_55features.csv"
        if demo_55_path.exists():
            st.session_state.demo_source = demo_55_path.read_bytes()
            st.session_state.demo_name = "demo_validation_55features.csv"
            st.rerun()

with col_s2:
    if st.button("🏷️ Mẫu kèm nhãn thô"):
        raw_demo_path = ROOT_DIR / "data" / "demo" / "demo_with_groundtruth_labels.csv"
        if raw_demo_path.exists():
            st.session_state.demo_source = raw_demo_path.read_bytes()
            st.session_state.demo_name = "demo_with_groundtruth_labels.csv"
            st.rerun()

if st.session_state.demo_source is not None:
    if st.sidebar.button("🧹 Xóa mẫu / Dùng tệp tải lên"):
        st.session_state.demo_source = None
        st.session_state.demo_name = None
        st.rerun()


# =============================================================================
# 6. THÂN TRANG WEB CHÍNH
# =============================================================================
banner_html = f"""
<div class="cyber-banner">
    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: #0284c7; letter-spacing: 2px; font-weight: 700;">
        [DEFENSE PROTOCOL // FORENSIC MEMORY ANALYSIS CONSOLE]
    </div>
    <h1 class="cyber-title">CIC-MALMEM-2022 // THREAT TELEMETRY CONSOLE</h1>
    <div class="cyber-subtitle">
        HỆ THỐNG ĐIỀU TRA & GIÁM SÁT MÃ ĐỘC TỪ DẤU VẾT BỘ NHỚ RAM (55 FEATURES) // CƠ CHẾ TỪ CHỐI UNCERTAIN (τ = {tau_value:.2f}) // GIẢI THÍCH LOCAL TOP-3
    </div>
    <div class="cyber-badges">
        <span class="cyber-badge badge-radar">● RADAR: ONLINE</span>
        <span class="cyber-badge badge-core">ENGINE: {algo_key.upper()} [{selected_model_id}]</span>
        <span class="cyber-badge badge-task">TASK: {task_key.upper()}</span>
        <span class="cyber-badge badge-tau">CUTOFF TAU: {tau_value:.2f}</span>
    </div>
</div>
"""
st.html(banner_html)


st.markdown('<div class="cyber-section-header">1. NẠP DỮ LIỆU ĐẶC TRƯNG BỘ NHỚ (55 MEMORY VECTORS)</div>', unsafe_allow_html=True)

col_up, col_opt = st.columns([3, 1])
with col_up:
    uploaded_file = st.file_uploader(
        "Tải lên tệp CSV chứa đúng 55 đặc trưng bộ nhớ từ Volatility (hoặc tệp thô từ CIC-MalMem-2022):",
        type=["csv"],
        key="main_csv_uploader"
    )

with col_opt:
    auto_drop = st.checkbox("Tự động loại bỏ cột nhãn / metadata", value=True)

raw_data_input = None
source_label = ""
if uploaded_file is not None:
    raw_data_input = uploaded_file
    source_label = f"Tệp tải lên: {uploaded_file.name}"
elif st.session_state.demo_source is not None:
    raw_data_input = st.session_state.demo_source
    source_label = f"Mẫu kiểm thử nạp sẵn: {st.session_state.get('demo_name', 'demo.csv')}"

if raw_data_input is None:
    contract_guide_html = """
    <div class="cyber-card" style="border-left: 4px solid #0284c7;">
        <div style="font-weight: 700; font-size: 0.95rem; color: #0f172a; margin-bottom: 6px;">
            📋 HỢP ĐỒNG DỮ LIỆU ĐẦU VÀO (VOLATILITY 55 FEATURES CONTRACT)
        </div>
        <ul style="font-size: 0.88rem; color: #475569; margin: 0; padding-left: 20px; line-height: 1.6;">
            <li><b>Header:</b> Tệp CSV phải có dòng tiêu đề và không được chứa tên cột trùng lặp.</li>
            <li><b>Số lượng đặc trưng:</b> Đúng <b>55 features số</b> tương ứng các trích xuất Volatility (pslist, dlllist, handles, ldrmodules, malfind, svcscan, callbacks...).</li>
            <li><b>Cột nhãn / Metadata:</b> Khi bật tùy chọn tự động loại bỏ, hệ thống sẽ tự động gạt bỏ các cột <code style="color: #0284c7;">Class</code>, <code style="color: #0284c7;">Category</code>, <code style="color: #0284c7;">multiclass</code> để tránh data leakage.</li>
            <li><b>Ô trống (Missing values):</b> Được giữ nguyên dưới dạng NaN để <code>SimpleImputer(strategy='median')</code> trong pipeline tự động bù đắp.</li>
            <li><b>Xác thực nghiêm ngặt:</b> Phát hiện và từ chối các tệp rỗng, chứa chữ rác (non-numeric text) hoặc giá trị vô cực (<code>inf</code>).</li>
        </ul>
        <div style="margin-top: 10px; font-size: 0.85rem; color: #0284c7; font-weight: 600;">
            👉 Mẹo nhanh: Nhấn nút <b>"Nạp 5 mẫu chuẩn"</b> ở thanh bên trái để thử nghiệm ngay lập tức.
        </div>
    </div>
    """
    st.html(contract_guide_html)
    st.stop()

try:
    X_validated = validate_csv_input(
        raw_data_input,
        expected_features=expected_features,
        auto_drop_labels=auto_drop
    )
except ValueError as val_err:
    st.error(f"❌ [DATA CONTRACT VIOLATION] Xác thực dữ liệu thất bại:\n\n{val_err}")
    st.stop()
except Exception as err:
    st.error(f"❌ Lỗi không xác định khi đọc CSV: {err}")
    st.stop()

num_rows = len(X_validated)
st.success(f"✅ [DATA CONTRACT VERIFIED] Xác thực thành công {num_rows} dòng dữ liệu theo đúng 55 features của model. ({source_label})")

with st.expander("🔍 Xem bảng dữ liệu 55 đặc trưng đã chuẩn hóa (Tối đa 5 dòng đầu)"):
    st.dataframe(X_validated.head(5), use_container_width=True)


st.markdown('<div class="cyber-section-header">2. BẢNG KẾT QUẢ SUY LUẬN & THỐNG KÊ ĐỘ PHỦ TELEMETRY</div>', unsafe_allow_html=True)

try:
    results_df, probas = predict_batch(
        pipeline=pipeline,
        X_validated=X_validated,
        tau=tau_value,
        rejection_label=rejection_label
    )
except Exception as pred_err:
    st.error(f"❌ Lỗi trong quá trình suy luận pipeline: {pred_err}")
    st.stop()

num_accepted = int((results_df["Kết quả"] != rejection_label).sum())
num_uncertain = int((results_df["Kết quả"] == rejection_label).sum())
coverage_pct = (num_accepted / num_rows) * 100.0 if num_rows > 0 else 0.0

col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.html(f"""
    <div class="metric-box">
        <div class="metric-box-val">{num_rows}</div>
        <div class="metric-box-lbl">Tổng số mẫu (Total)</div>
    </div>
    """)
with col_m2:
    st.html(f"""
    <div class="metric-box" style="border-top-color: #10b981;">
        <div class="metric-box-val" style="color: #059669;">{num_accepted}</div>
        <div class="metric-box-lbl">Đạt ngưỡng tin cậy (Accepted)</div>
    </div>
    """)
with col_m3:
    st.html(f"""
    <div class="metric-box" style="border-top-color: #f59e0b;">
        <div class="metric-box-val" style="color: #d97706;">{num_uncertain}</div>
        <div class="metric-box-lbl">Bị từ chối (Uncertain)</div>
    </div>
    """)
with col_m4:
    st.html(f"""
    <div class="metric-box" style="border-top-color: #8b5cf6;">
        <div class="metric-box-val" style="color: #7c3aed;">{coverage_pct:.1f}%</div>
        <div class="metric-box-lbl">Độ phủ (Coverage at τ)</div>
    </div>
    """)

st.markdown("#### 📋 Chi tiết kết quả suy luận từng dòng:")

display_df = results_df.copy()
display_df["p_max"] = display_df["p_max"].apply(lambda v: f"{v:.4f}")
for cls in pipeline_classes:
    col_name = f"Score_{cls}"
    if col_name in display_df.columns:
        display_df[col_name] = display_df[col_name].apply(lambda v: f"{v:.4f}")

def highlight_results(val):
    if val == "Benign":
        return "background-color: #ecfdf5; color: #065f46; font-weight: bold;"
    elif val == "Uncertain":
        return "background-color: #fffbeb; color: #92400e; font-weight: bold;"
    else:
        return "background-color: #fef2f2; color: #991b1b; font-weight: bold;"

st.dataframe(
    display_df.style.applymap(highlight_results, subset=["Kết quả"]) if hasattr(display_df.style, "applymap")
    else display_df.style.map(highlight_results, subset=["Kết quả"]),
    use_container_width=True,
    height=min(320, 45 + num_rows * 38)
)

csv_buffer = io.StringIO()
results_df.to_csv(csv_buffer, index=False)
st.download_button(
    label="📥 Tải xuống CSV kết quả suy luận (Inference Report)",
    data=csv_buffer.getvalue(),
    file_name=f"inference_results_{selected_model_id}.csv",
    mime="text/csv"
)


st.markdown('<div class="cyber-section-header">3. KHÁM NGHIỆM CHI TIẾT DÒNG & GIẢI THÍCH CỤC BỘ (LOCAL TOP-3)</div>', unsafe_allow_html=True)

col_sel, col_stat = st.columns([1, 2])
with col_sel:
    selected_row_idx = st.selectbox(
        "Chọn dòng dữ liệu cần khám nghiệm chi tiết:",
        options=list(range(num_rows)),
        format_func=lambda i: f"Dòng #{i} — Kết quả: {results_df.iloc[i]['Kết quả']} (p_max = {results_df.iloc[i]['p_max']:.3f})"
    )

row_res = results_df.iloc[selected_row_idx]
row_features = X_validated.iloc[[selected_row_idx]]
is_accepted = bool(row_res["Đạt ngưỡng"])
dec_label = row_res["Kết quả"]
cand_class = row_res["Lớp ứng viên"]
row_pmax = float(row_res["p_max"])

with col_stat:
    status_bg = "#ecfdf5" if dec_label == "Benign" else ("#fffbeb" if dec_label == rejection_label else "#fef2f2")
    status_border = "#10b981" if dec_label == "Benign" else ("#f59e0b" if dec_label == rejection_label else "#ef4444")
    status_text_color = "#065f46" if dec_label == "Benign" else ("#92400e" if dec_label == rejection_label else "#991b1b")
    status_title = "ĐẠT NGƯỠNG AN TOÀN // TIN CẬY" if is_accepted else f"CẢNH BÁO // TỪ CHỐI DO KHÔNG ĐẠT NGƯỠNG TIN CẬY ({rejection_label.upper()})"

    row_card_html = f"""
    <div style="background: {status_bg}; border: 1.5px solid {status_border}; border-radius: 6px; padding: 12px 16px;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; color: {status_text_color}; font-weight: 700; letter-spacing: 1px;">
            [TRẠNG THÁI PHÂN ĐỊNH: {status_title}]
        </div>
        <div style="font-size: 1.25rem; font-weight: 700; color: {status_text_color}; margin: 4px 0;">
            Kết luận: <b>{dec_label}</b> {f"(Lớp ứng viên: <u>{cand_class}</u>)" if not is_accepted else ""}
        </div>
        <div style="font-size: 0.85rem; color: #334155;">
            Điểm tin cậy: <code style="font-weight: bold;">{row_pmax:.4f}</code> | Ngưỡng cắt τ: <code style="font-weight: bold;">{tau_value:.2f}</code> | 
            Điều kiện: <code>{row_pmax:.4f} {'>=' if is_accepted else '<'} {tau_value:.2f}</code>
        </div>
    </div>
    """
    st.html(row_card_html)

st.markdown("##### 📊 Phân bố điểm xác suất dự đoán (Predicted Class Scores):")
class_scores = {cls: float(row_res[f"Score_{cls}"]) for cls in pipeline_classes}
score_df = pd.DataFrame({
    "Class": list(class_scores.keys()),
    "Probability": list(class_scores.values())
})

col_chart, col_chart_info = st.columns([3, 2])
with col_chart:
    st.bar_chart(score_df.set_index("Class"), height=180, use_container_width=True)
with col_chart_info:
    st.markdown(f"""
    **Điểm số chi tiết từng lớp:**
    """ + "".join([f"\n- **{cls}**: `{score_df.loc[score_df['Class'] == cls, 'Probability'].values[0]:.4f}`" for cls in pipeline_classes]) + f"""
    \n*Ngưỡng quyết định:* $\\tau = {tau_value:.2f}$
    """)

st.markdown("##### 🔬 Phân tích giải thích cục bộ (Local Top-3 Feature Attribution):")

explanation_result = compute_local_top3(
    pipeline=pipeline,
    row_features=row_features,
    task=task_key,
    target_class_override=cand_class
)

if not explanation_result["supported"]:
    rf_notice_html = f"""
    <div style="background: #f1f5f9; border: 1.5px dashed #64748b; border-radius: 8px; padding: 18px 20px; margin: 12px 0;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; font-weight: 700; color: #475569; letter-spacing: 1px;">
            ⚠️ TRẠNG THÁI GIẢI THÍCH CỤC BỘ: CHƯA HỖ TRỢ ĐỐI VỚI RANDOM FOREST
        </div>
        <div style="font-size: 0.92rem; color: #334155; margin-top: 6px; line-height: 1.5;">
            {explanation_result['message']}
        </div>
        <div style="margin-top: 10px; font-size: 0.88rem; color: #0284c7; font-weight: 600;">
            💡 Gợi ý thao tác: Hãy chọn thuật toán <b>"LightGBM"</b> ở thanh bên trái để xem đầy đủ bảng giải thích Local Top-3 và cơ chế kiểm chứng tính khớp raw margin.
        </div>
    </div>
    """
    st.html(rf_notice_html)
else:
    top3_df = explanation_result["top3"]
    target_cls = explanation_result["target_class"]
    baseline_val = explanation_result["baseline"]
    raw_score_val = explanation_result["target_raw_score"]
    sum_contribs = explanation_result["sum_contributions"]
    total_recon = explanation_result["total_reconstructed"]
    parity_diff = explanation_result["parity_diff"]
    is_ok = explanation_result["is_parity_ok"]

    lgb_info_html = f"""
    <div style="background: #f0fdf4; border: 1px solid #86efac; border-radius: 6px; padding: 10px 16px; margin-bottom: 12px; font-size: 0.88rem;">
        <div><b>Đang giải thích cho lớp:</b> <span style="font-weight: 700; color: #166534;">{target_cls}</span> 
        {f"(Lớp ứng viên được chọn để phân tích do dòng này bị <i>Uncertain</i>)" if not is_accepted else ""}</div>
        <div style="color: #475569; font-size: 0.82rem; margin-top: 2px;">
            Cơ chế: Trích xuất <code>pred_contrib=True</code> từ LightGBM Booster sau khi chuẩn hóa qua pipeline fitted preprocessing.
        </div>
    </div>
    """
    st.html(lgb_info_html)

    top3_display = pd.DataFrame({
        "Thứ hạng": [f"Top #{i+1}" for i in range(len(top3_df))],
        "Đặc trưng bộ nhớ (Feature)": top3_df["feature"],
        "Giá trị gốc (Raw Value)": top3_df["original_value"].apply(lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else str(v)),
        "Đóng góp (Signed Raw Score)": top3_df["contribution"].apply(lambda v: f"{v:+.4f}"),
        "Mức độ ảnh hưởng (|Contrib|)": top3_df["abs_contrib"].apply(lambda v: f"{v:.4f}"),
        "Hướng tác động": top3_df["impact"]
    })
    st.table(top3_display)

    parity_status_color = "#047857" if is_ok else "#b91c1c"
    parity_status_text = "PASS (ĐẠT CHUẨN ĐỘ LỆCH < 1e-4)" if is_ok else "FAIL"

    parity_html = f"""
    <div style="background: #ffffff; border: 1px solid #cbd5e1; border-top: 3px solid #0284c7; border-radius: 6px; padding: 14px 18px; margin-top: 14px; font-size: 0.85rem;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; font-weight: 700; color: #0284c7; letter-spacing: 1px;">
            🔒 KIỂM CHỨNG TÍNH KHỚP TOÀN VẸN (RAW SCORE PARITY VERIFICATION)
        </div>
        <div style="margin-top: 8px; line-height: 1.7; font-family: 'JetBrains Mono', monospace; color: #334155;">
            <div>• Tổng 55 đặc trưng (Σ Contrib): <b>{sum_contribs:+.6f}</b></div>
            <div>• Điểm nền cơ sở (Baseline / Prior): <b>{baseline_val:+.6f}</b></div>
            <div>• Tổng tái tạo (Σ Contrib + Baseline): <b>{total_recon:+.6f}</b></div>
            <div>• Điểm biên thô gốc (Target Raw Score): <b>{raw_score_val:+.6f}</b></div>
            <div>• Chênh lệch sai số (|Δ|): <b>{parity_diff:.2e}</b></div>
            <div>• Trạng thái kiểm chứng: <span style="font-weight: 700; color: {parity_status_color}; background: #ecfdf5; padding: 2px 8px; border-radius: 4px;">{parity_status_text}</span></div>
        </div>
    </div>
    """
    st.html(parity_html)


st.markdown("---")
footer_html = """
<div style="text-align: center; color: #64748b; font-size: 0.8rem; font-family: 'JetBrains Mono', monospace; padding: 10px 0;">
    TOPIC 05 — CIC-MALMEM-2022 // OBFUSCATED MALWARE DETECTION FROM MEMORY FORENSICS<br>
    Triển khai Streamlit Web Demo cục bộ theo quy chuẩn bàn giao TV5 // Phiên bản 2.0 (4 Models Architecture)
</div>
"""
st.html(footer_html)
