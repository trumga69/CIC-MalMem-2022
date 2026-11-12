"""Ứng dụng Streamlit: Web Demo Topic 05 — CIC-MalMem-2022.

Phân loại bộ nhớ độc hại (Task A nhị phân và Task B đa lớp)
kèm cơ chế từ chối (Uncertain rejection với tau) và giải thích cục bộ Top-3.
Giao diện Cyber Defense / SOC Threat Intelligence Terminal (White Base + Chakra Petch & JetBrains Mono Fonts).
"""
import io
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.explanation import compute_local_top3
from src.triage import (
    get_project_root,
    load_operating_config,
    load_pipeline_for_task,
    predict_batch,
    validate_csv_input,
)

# Cấu hình font tiếng Việt và typography cho Matplotlib
plt.rcParams['font.sans-serif'] = ['Segoe UI', 'DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['font.family'] = 'sans-serif'

# Cấu hình trang chuẩn an ninh mạng
st.set_page_config(
    page_title="CIC-MalMem-2022 // CYBER THREAT TRIAGE HUD",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# SVG Icons chuẩn an ninh mạng (Vector sắc nét, độ phân giải cao)
# -----------------------------------------------------------------------------
SVG_SHIELD = """<svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 2L3 6V11.09C3 16.45 6.84 21.49 12 22.82C17.16 21.49 21 16.45 21 11.09V6L12 2Z" fill="#0284c7" fill-opacity="0.15" stroke="#0284c7" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M12 7V17M8 11L12 7L16 11" stroke="#0284c7" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>"""

SVG_CHIP = """<svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="4" y="4" width="16" height="16" rx="2" stroke="#0284c7" stroke-width="2" fill="#0284c7" fill-opacity="0.1"/><rect x="8" y="8" width="8" height="8" rx="1" fill="#0284c7"/><path d="M1 9H4M1 15H4M20 9H23M20 15H23M9 1V4M15 1V4M9 20V23M15 20V23" stroke="#0284c7" stroke-width="2" stroke-linecap="round"/></svg>"""

SVG_ANALYTICS = """<svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M3 3V21H21" stroke="#0284c7" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M7 16L12 11L15 14L20 8" stroke="#0284c7" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><circle cx="20" cy="8" r="2" fill="#0284c7"/></svg>"""

SVG_CROSSHAIR = """<svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="9" stroke="#0284c7" stroke-width="2"/><circle cx="12" cy="12" r="3" stroke="#0284c7" stroke-width="1.5"/><path d="M12 1V5M12 19V23M1 12H5M19 12H23" stroke="#0284c7" stroke-width="2" stroke-linecap="round"/></svg>"""

SVG_BENIGN_BIG = """<svg width="44" height="44" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 2L3 6V11.09C3 16.45 6.84 21.49 12 22.82C17.16 21.49 21 16.45 21 11.09V6L12 2Z" fill="#10b981" fill-opacity="0.18" stroke="#10b981" stroke-width="2"/><path d="M9 12L11 14L15 10" stroke="#059669" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/></svg>"""

SVG_THREAT_BIG = """<svg width="44" height="44" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M12 2L2 20H22L12 2Z" fill="#ef4444" fill-opacity="0.18" stroke="#ef4444" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/><path d="M12 9V14M12 17H12.01" stroke="#dc2626" stroke-width="2.5" stroke-linecap="round"/></svg>"""

SVG_UNCERTAIN_BIG = """<svg width="44" height="44" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="10" fill="#f59e0b" fill-opacity="0.18" stroke="#f59e0b" stroke-width="2"/><path d="M9.09 9C9.32 8.41 9.77 7.94 10.35 7.69C10.93 7.44 11.59 7.43 12.18 7.66C12.77 7.89 13.24 8.34 13.48 8.92C13.72 9.5 13.69 10.16 13.39 10.72L12 13M12 17H12.01" stroke="#d97706" stroke-width="2.5" stroke-linecap="round"/></svg>"""

# -----------------------------------------------------------------------------
# CSS Giao diện An ninh mạng (Hỗ trợ 100% tiếng Việt có dấu với Chakra Petch & JetBrains Mono)
# -----------------------------------------------------------------------------
st.html("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Chakra+Petch:ital,wght@0,400;0,500;0,600;0,700;1,500;1,700&family=JetBrains+Mono:ital,wght@0,400;0,500;0,600;0,700;0,800;1,400;1,600&family=Space+Mono:ital,wght@0,400;0,700;1,400&display=swap');

    /* Nền chung với lưới chấm vi mô công nghệ */
    .stApp {
        background-color: #ffffff;
        background-image: radial-gradient(#e2e8f0 1.1px, transparent 1.1px);
        background-size: 24px 24px;
        color: #0f172a;
        font-family: 'Chakra Petch', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }

    html, body, [class*="css"] {
        font-family: 'Chakra Petch', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    }

    /* Font Monospace chuẩn cho code, log, dataframe, metrics */
    code, pre, .stCodeBlock, [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', 'Consolas', monospace !important;
    }

    /* ---------------- Cyber Hero HUD Banner ---------------- */
    .cyber-hero-banner {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 2px solid #0284c7;
        border-radius: 8px;
        padding: 22px 28px;
        margin-bottom: 24px;
        position: relative;
        box-shadow: 0 8px 24px rgba(2, 132, 199, 0.1), 0 1px 3px rgba(0, 0, 0, 0.05);
        overflow: hidden;
    }

    .cyber-hero-banner::before {
        content: "[TOPIC-05 // CIC-MALMEM-2022 // LEVEL-4 CLASSIFIED]";
        position: absolute;
        top: 0;
        right: 0;
        background: #0284c7;
        color: #ffffff;
        font-family: 'Space Mono', 'JetBrains Mono', monospace !important;
        font-size: 0.72rem;
        padding: 3px 14px;
        letter-spacing: 2px;
        border-bottom-left-radius: 6px;
        font-weight: 700;
    }

    .hero-flex {
        display: flex;
        align-items: center;
        gap: 20px;
    }

    .hero-icon-box {
        background: #f0f9ff;
        border: 2px solid #38bdf8;
        border-radius: 10px;
        padding: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 0 16px rgba(56, 189, 248, 0.25);
    }

    .cyber-title {
        font-family: 'Chakra Petch', sans-serif !important;
        font-size: 2.1rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: 1.2px;
        margin: 0;
        text-transform: uppercase;
    }

    .cyber-subtitle {
        font-family: 'JetBrains Mono', 'Chakra Petch', monospace !important;
        font-size: 0.88rem;
        color: #475569;
        margin-top: 6px;
        margin-bottom: 12px;
        line-height: 1.5;
    }

    /* Badges trạng thái */
    .cyber-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }

    .cyber-badge {
        font-family: 'JetBrains Mono', 'Chakra Petch', monospace !important;
        font-size: 0.78rem;
        padding: 5px 12px;
        border-radius: 4px;
        letter-spacing: 1px;
        text-transform: uppercase;
        font-weight: 700;
        display: inline-flex;
        align-items: center;
        gap: 6px;
    }

    .badge-radar {
        background: #ecfdf5;
        color: #059669;
        border: 1px solid #10b981;
        box-shadow: 0 2px 6px rgba(16, 185, 129, 0.15);
    }

    .badge-hash {
        background: #f0f9ff;
        color: #0284c7;
        border: 1px solid #0284c7;
        box-shadow: 0 2px 6px rgba(2, 132, 199, 0.15);
    }

    .badge-tau {
        background: #fffbeb;
        color: #b45309;
        border: 1px solid #f59e0b;
        box-shadow: 0 2px 6px rgba(245, 158, 11, 0.15);
    }

    .badge-core {
        background: #faf5ff;
        color: #7e22ce;
        border: 1px solid #a855f7;
        box-shadow: 0 2px 6px rgba(168, 85, 247, 0.15);
    }

    /* ---------------- Cyber Section Titles có Khung Viền Nổi Bật ---------------- */
    .cyber-section-header {
        background: #ffffff;
        border: 1.5px solid #0284c7;
        border-left: 8px solid #0284c7;
        border-radius: 6px;
        padding: 10px 16px;
        margin-top: 24px;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 0 3px 10px rgba(2, 132, 199, 0.08);
    }

    .cyber-section-title-wrap {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .cyber-section-icon {
        background: #f0f9ff;
        border: 1px solid #bae6fd;
        border-radius: 6px;
        padding: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .cyber-section-title {
        font-family: 'Chakra Petch', sans-serif !important;
        font-size: 1.25rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin: 0;
    }

    .cyber-section-tag {
        font-family: 'Space Mono', 'JetBrains Mono', monospace !important;
        font-size: 0.74rem;
        color: #0284c7;
        background: #e0f2fe;
        border: 1px solid #7dd3fc;
        padding: 3px 10px;
        border-radius: 3px;
        letter-spacing: 1.5px;
        font-weight: 700;
    }

    /* ---------------- Metric Cards HUD ---------------- */
    .cyber-metric-card {
        background: #ffffff;
        border: 1.5px solid #cbd5e1;
        border-top: 4px solid #0284c7;
        border-radius: 6px;
        padding: 14px 18px;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.04);
        position: relative;
    }

    .cyber-metric-label {
        font-family: 'Chakra Petch', sans-serif !important;
        font-size: 0.8rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
        margin-bottom: 6px;
    }

    .cyber-metric-value {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 2rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
    }

    /* ---------------- Decision Cards ---------------- */
    .cyber-card-decision {
        border-radius: 8px;
        padding: 20px 24px;
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.05);
        display: flex;
        align-items: flex-start;
        gap: 18px;
    }

    .decision-accepted {
        background: #f0fdf4;
        border: 2px solid #10b981;
        border-left: 8px solid #059669;
    }

    .decision-uncertain {
        background: #fffbeb;
        border: 2px solid #f59e0b;
        border-left: 8px solid #d97706;
    }

    .decision-threat {
        background: #fef2f2;
        border: 2px solid #ef4444;
        border-left: 8px solid #dc2626;
    }

    /* ---------------- Cryptographic Terminal Box ---------------- */
    .cyber-terminal-box {
        background: #090d16;
        border: 1.5px solid #1e293b;
        border-left: 6px solid #0284c7;
        border-radius: 6px;
        padding: 18px 22px;
        font-family: 'JetBrains Mono', monospace !important;
        color: #38bdf8;
        font-size: 0.88rem;
        line-height: 1.65;
        box-shadow: inset 0 2px 10px rgba(0, 0, 0, 0.8), 0 4px 14px rgba(15, 23, 42, 0.1);
    }

    /* Sidebar HUD Styling */
    [data-testid="stSidebar"] {
        background-color: #f8fafc;
        border-right: 1.5px solid #cbd5e1;
    }
</style>
""")


@st.cache_resource(show_spinner="[SECURITY INIT] Đang nạp pipeline và xác thực tính toàn vẹn SHA-256...")
def get_model_and_config(task: str):
    """Nạp và kiểm tra SHA-256 + classes của model (cached)."""
    cfg = load_operating_config()
    pipe, model_info = load_pipeline_for_task(task, cfg)
    return pipe, model_info, cfg


def main():
    # ---------------------------------------------------- 0. Cyber Hero Header
    st.html(f"""
    <div class="cyber-hero-banner">
        <div class="hero-flex">
            <div class="hero-icon-box">
                {SVG_SHIELD}
            </div>
            <div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #0284c7; letter-spacing: 2.5px; font-weight: 700;">
                    [DEFENSE PROTOCOL // FORENSIC MEMORY ANALYSIS]
                </div>
                <h1 class="cyber-title">CIC-MALMEM-2022 // THREAT TELEMETRY CONSOLE</h1>
                <div class="cyber-subtitle">
                    HỆ THỐNG ĐIỀU TRA & GIÁM SÁT MÃ ĐỘC TỪ DẤU VẾT BỘ NHỚ RAM (55 FEATURES) //
                    CƠ CHẾ TỪ CHỐI UNCERTAIN (&tau; = 0.65) // GIẢI THÍCH LOCAL TOP-3
                </div>
                <div class="cyber-badges">
                    <span class="cyber-badge badge-radar">● SOC RADAR: ONLINE</span>
                    <span class="cyber-badge badge-hash">HASH SHA-256: VERIFIED</span>
                    <span class="cyber-badge badge-tau">CUTOFF: TAU = 0.65</span>
                    <span class="cyber-badge badge-core">CORE: LIGHTGBM [LOCKED]</span>
                </div>
            </div>
        </div>
    </div>
    """)

    # ---------------------------------------------------- Sidebar Cyber Panel
    with st.sidebar:
        st.html("""
        <div style="background: #ffffff; border: 1.5px solid #0284c7; border-left: 5px solid #0284c7; border-radius: 5px; padding: 8px 12px; margin-bottom: 16px;">
            <div style="font-family: 'Chakra Petch', sans-serif; font-weight: 700; font-size: 0.95rem; color: #0f172a; text-transform: uppercase;">
                🎛️ BẢNG ĐIỀU KHIỂN TÁC VỤ
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #0284c7; letter-spacing: 1px;">
                [TASK CONFIGURATION]
            </div>
        </div>
        """)

        task_choice = st.radio(
            "Chọn chế độ điều tra (Investigation Mode):",
            options=["Task A (Binary Triage)", "Task B (Multiclass Threat Family)"],
            index=0,
            help="Task A: Benign vs Malware. Task B: Benign vs Ransomware / Spyware / Trojan."
        )
        task_key = "binary" if "Task A" in task_choice else "multiclass"

        try:
            pipeline, model_info, config = get_model_and_config(task_key)
            tau = float(config.get("tau", 0.65))
            rejection_label = config.get("rejection_label", "Uncertain")
        except Exception as e:
            st.error(f"[SYSTEM_FAULT] Không thể nạp mô hình: {e}")
            st.stop()

        st.markdown("---")
        st.html("""
        <div style="background: #ffffff; border: 1px solid #cbd5e1; border-left: 4px solid #475569; border-radius: 5px; padding: 8px 12px; margin-bottom: 12px;">
            <div style="font-family: 'Chakra Petch', sans-serif; font-weight: 700; font-size: 0.9rem; color: #0f172a;">
                🛡️ THÔNG SỐ PIPELINE KHÓA
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #64748b;">
                [LOCKED RUN PARAMETERS]
            </div>
        </div>
        """)

        st.markdown(f"- **Thuật toán lõi:** `{model_info.get('algorithm', 'lightgbm').upper()}`")
        st.markdown(f"- **Run ID:** `{model_info.get('run_id')}`")
        st.markdown(f"- **Ngưỡng $\\tau$ (tau):** `{tau:.2f}`")
        st.markdown(f"- **Nhãn từ chối:** `{rejection_label}`")
        st.markdown(f"- **Số features yêu cầu:** `{len(pipeline.feature_names_in_)}`")
        st.markdown(f"- **Classes:** `{', '.join(pipeline.classes_)}`")
        st.caption(f"SHA-256: `{model_info.get('pipeline_sha256')[:16]}...` (đã đối chiếu khớp)")

        st.markdown("---")
        st.html("""
        <div style="background: #ffffff; border: 1px solid #cbd5e1; border-left: 4px solid #f59e0b; border-radius: 5px; padding: 8px 12px; margin-bottom: 12px;">
            <div style="font-family: 'Chakra Petch', sans-serif; font-weight: 700; font-size: 0.9rem; color: #0f172a;">
                ⚡ TẢI NHANH MẪU ĐỐI SOÁT
            </div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; color: #b45309;">
                [VALIDATION TELEMETRY]
            </div>
        </div>
        """)

        demo_path = get_project_root() / "demo_validation_features.csv"
        has_demo = demo_path.exists()
        use_demo = st.button("🧪 NẠP 3 DÒNG MẪU VALIDATION", disabled=not has_demo, use_container_width=True)
        if not has_demo:
            st.caption("Chưa có demo_validation_features.csv tại thư mục gốc.")

    # ---------------------------------------------------- 1. Data Ingestion Header
    st.html(f"""
    <div class="cyber-section-header">
        <div class="cyber-section-title-wrap">
            <div class="cyber-section-icon">
                {SVG_CHIP}
            </div>
            <h2 class="cyber-section-title">1. NẠP DỮ LIỆU ĐẶC TRƯNG BỘ NHỚ (55 MEMORY VECTORS)</h2>
        </div>
        <div class="cyber-section-tag">// STAGE: 01_INGESTION //</div>
    </div>
    """)

    uploaded_file = st.file_uploader(
        "Tải lên tệp CSV chứa đúng 55 đặc trưng bộ nhớ từ Volatility (loại trừ các cột nhãn Class / Category)",
        type=["csv"],
        help="Hợp đồng schema: đúng 55 features số. Bảng rỗng, thiếu, dư, trùng cột hoặc chữ rác/infinity sẽ bị từ chối."
    )

    input_data = None
    source_name = ""

    if use_demo and has_demo:
        input_data = demo_path.read_text(encoding="utf-8")
        source_name = "demo_validation_features.csv (Row ID 14627, 11611, 43302)"
        st.info(f"⚡ [TELEMETRY_LOADED] Đang sử dụng dữ liệu mẫu kiểm chứng: **{source_name}**")
    elif uploaded_file is not None:
        input_data = uploaded_file.getvalue()
        source_name = uploaded_file.name

    if input_data is None:
        st.html("""
        <div style="background: #ffffff; border: 1.5px dashed #0284c7; border-radius: 8px; padding: 26px; text-align: center; color: #475569; font-family: 'JetBrains Mono', monospace; box-shadow: 0 4px 12px rgba(2, 132, 199, 0.05);">
            <div style="font-size: 1.05rem; font-weight: 700; color: #0284c7; margin-bottom: 6px;">[STANDBY // WAITING FOR TELEMETRY INPUT]</div>
            Vui lòng tải file CSV hoặc nhấn nút <b>"NẠP 3 DÒNG MẪU VALIDATION"</b> ở bảng điều khiển bên trái để bắt đầu trích xuất & suy luận.
        </div>
        """)
        return

    # Xác thực dữ liệu
    expected_features = list(pipeline.feature_names_in_)
    try:
        X_validated = validate_csv_input(input_data, expected_features)
    except ValueError as ve:
        st.error(f"🚨 **[DATA CONTRACT VIOLATION] Xác thực thất bại:** {ve}")
        st.markdown("""
        > [!IMPORTANT]
        > **Quy chuẩn hợp đồng dữ liệu Volatility (55 features):**
        > - Đúng 55 đặc trưng số đã trích xuất từ RAM dump.
        > - Tuyệt đối không chứa cột nhãn (`Class`, `Category`, `Filename`, `row_index`, `group_id`).
        > - Không có cột trùng lặp, chữ rác không phải số, hoặc giá trị vô cực (`inf`).
        > - Cho phép cột đảo thứ tự (hệ thống sẽ tự động reorder theo đúng pipeline).
        """)
        return
    except Exception as ex:
        st.error(f"🚨 **[UNEXPECTED_ERROR] Lỗi đọc dữ liệu:** {ex}")
        return

    st.success(f"✅ [INTEGRITY_OK] Xác thực hợp lệ! Đã nhận diện **{len(X_validated)} mẫu bộ nhớ** với đủ 55 features theo thứ tự pipeline.")

    # ---------------------------------------------------- 2. Batch Inference
    st.html(f"""
    <div class="cyber-section-header">
        <div class="cyber-section-title-wrap">
            <div class="cyber-section-icon">
                {SVG_ANALYTICS}
            </div>
            <h2 class="cyber-section-title">2. BẢNG PHÂN LOẠI & GIÁM SÁT MÃ ĐỘC (BATCH TELEMETRY)</h2>
        </div>
        <div class="cyber-section-tag">// STAGE: 02_CLASSIFICATION //</div>
    </div>
    """)

    results_df, probas = predict_batch(pipeline, X_validated, tau=tau, rejection_label=rejection_label)

    # Thống kê nhanh
    n_total = len(results_df)
    n_accepted = int(results_df["Đạt ngưỡng"].sum())
    n_uncertain = n_total - n_accepted
    coverage = (n_accepted / n_total) * 100.0

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.html(f"""
        <div class="cyber-metric-card">
            <div class="cyber-metric-label">TỔNG SỐ MẪU (SAMPLES)</div>
            <div class="cyber-metric-value">{n_total}</div>
        </div>
        """)
    with col_m2:
        st.html(f"""
        <div class="cyber-metric-card" style="border-top-color: #10b981;">
            <div class="cyber-metric-label" style="color: #059669;">TIẾP NHẬN (ACCEPTED)</div>
            <div class="cyber-metric-value" style="color: #059669;">{n_accepted}</div>
        </div>
        """)
    with col_m3:
        st.html(f"""
        <div class="cyber-metric-card" style="border-top-color: #f59e0b;">
            <div class="cyber-metric-label" style="color: #d97706;">TỪ CHỐI (UNCERTAIN)</div>
            <div class="cyber-metric-value" style="color: #d97706;">{n_uncertain}</div>
        </div>
        """)
    with col_m4:
        st.html(f"""
        <div class="cyber-metric-card" style="border-top-color: #8b5cf6;">
            <div class="cyber-metric-label" style="color: #7c3aed;">ĐỘ BAO PHỦ (COVERAGE)</div>
            <div class="cyber-metric-value" style="color: #7c3aed;">{coverage:.2f}%</div>
        </div>
        """)

    st.write("")

    # Hiển thị bảng tổng hợp
    display_cols = ["Dòng", "Kết quả", "Lớp ứng viên", "p_max", "Ngưỡng tau", "Đạt ngưỡng"]
    score_cols = [f"Score_{c}" for c in pipeline.classes_]
    display_df = results_df[display_cols + score_cols].copy()

    # Định dạng số thực
    format_dict = {"p_max": "{:.6f}"}
    for sc in score_cols:
        format_dict[sc] = "{:.6f}"

    def highlight_threat(val):
        if val == "Uncertain":
            return "background-color: #fef3c7; color: #b45309; font-weight: 700; border-left: 4px solid #d97706;"
        elif val == "Benign":
            return "background-color: #dcfce7; color: #15803d; font-weight: 700; border-left: 4px solid #16a34a;"
        elif val in ("Malware", "Ransomware", "Spyware", "Trojan"):
            return "background-color: #fee2e2; color: #b91c1c; font-weight: 700; border-left: 4px solid #dc2626;"
        return ""

    styler = display_df.style.format(format_dict)
    if hasattr(styler, "map"):
        styled_df = styler.map(highlight_threat, subset=["Kết quả"])
    else:
        styled_df = styler.applymap(highlight_threat, subset=["Kết quả"])

    st.dataframe(
        styled_df,
        use_container_width=True,
        height=min(320, 45 + len(display_df) * 36)
    )

    st.caption(
        "⚡ Protocol Note: So sánh p_max >= tau (0.65) được thực hiện trên số thực nguyên bản trước khi hiển thị. "
        "Các cột Score thể hiện điểm dự đoán của model (predict_proba); chưa có bằng chứng đã được hiệu chỉnh (calibrated) "
        "để xem là xác suất thực tế ngoài phân phối."
    )

    # ---------------------------------------------------- 3. Row Details & Explanation
    st.html(f"""
    <div class="cyber-section-header">
        <div class="cyber-section-title-wrap">
            <div class="cyber-section-icon">
                {SVG_CROSSHAIR}
            </div>
            <h2 class="cyber-section-title">3. SOI CHI TIẾT MẪU & GIẢI THÍCH LOCAL TOP-3 (XAI TELEMETRY)</h2>
        </div>
        <div class="cyber-section-tag">// STAGE: 03_FORENSIC_XAI //</div>
    </div>
    """)

    selected_row_idx = st.selectbox(
        "Chọn mẫu dòng bộ nhớ để điều tra chi tiết:",
        options=list(range(len(X_validated))),
        format_func=lambda i: f"Mẫu #{i} ➜ Kết quả: [{results_df.loc[i, 'Kết quả']}] | p_max = {results_df.loc[i, 'p_max']:.6f} | Lớp ứng viên: {results_df.loc[i, 'Lớp ứng viên']}"
    )

    row_data = X_validated.iloc[[selected_row_idx]]
    row_result = results_df.iloc[selected_row_idx]
    final_res = row_result["Kết quả"]
    candidate = row_result["Lớp ứng viên"]
    p_max_val = row_result["p_max"]
    is_accepted = bool(row_result["Đạt ngưỡng"])

    col_d1, col_d2, col_d3 = st.columns([1.6, 0.9, 1.5])
    with col_d1:
        if is_accepted:
            if final_res == "Benign":
                st.html(f"""
                <div class="cyber-card-decision decision-accepted">
                    <div>{SVG_BENIGN_BIG}</div>
                    <div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #059669; letter-spacing: 1.5px; font-weight: 700;">
                            [STATUS: ACCESS VERIFIED // BENIGN]
                        </div>
                        <h3 style="color: #15803d; margin: 4px 0 8px 0; font-size: 1.8rem; font-weight: 700; font-family: 'Chakra Petch', sans-serif;">
                            KẾT LUẬN: {final_res}
                        </h3>
                        <div style="color: #334155; font-family: 'JetBrains Mono', monospace; font-size: 0.88rem; line-height: 1.5;">
                            Mẫu đạt ngưỡng tin cậy (p_max = <b>{p_max_val:.6f}</b> &ge; &tau; = <b>{tau:.2f}</b>). Dấu vết bộ nhớ an toàn / lành tính.
                        </div>
                    </div>
                </div>
                """)
            else:
                st.html(f"""
                <div class="cyber-card-decision decision-threat">
                    <div>{SVG_THREAT_BIG}</div>
                    <div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #dc2626; letter-spacing: 1.5px; font-weight: 700;">
                            [CRITICAL ALERT: MALICIOUS ARTIFACT DETECTED]
                        </div>
                        <h3 style="color: #b91c1c; margin: 4px 0 8px 0; font-size: 1.8rem; font-weight: 700; font-family: 'Chakra Petch', sans-serif;">
                            KẾT LUẬN: {final_res}
                        </h3>
                        <div style="color: #334155; font-family: 'JetBrains Mono', monospace; font-size: 0.88rem; line-height: 1.5;">
                            Mẫu đạt ngưỡng tin cậy (p_max = <b>{p_max_val:.6f}</b> &ge; &tau; = <b>{tau:.2f}</b>). Đã định danh hành vi mã độc.
                        </div>
                    </div>
                </div>
                """)
        else:
            st.html(f"""
            <div class="cyber-card-decision decision-uncertain">
                <div>{SVG_UNCERTAIN_BIG}</div>
                <div>
                    <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #d97706; letter-spacing: 1.5px; font-weight: 700;">
                        [WARNING: REJECTION ACTIVATED // UNCERTAIN]
                    </div>
                    <h3 style="color: #b45309; margin: 4px 0 8px 0; font-size: 1.8rem; font-weight: 700; font-family: 'Chakra Petch', sans-serif;">
                        TRẠNG THÁI: {rejection_label}
                    </h3>
                    <div style="color: #334155; font-family: 'JetBrains Mono', monospace; font-size: 0.88rem; line-height: 1.5;">
                        Từ chối quyết định do p_max = <b>{p_max_val:.6f}</b> &lt; &tau; = <b>{tau:.2f}</b>.<br>
                        👉 <b>Lớp ứng viên (điểm cao nhất):</b> <span style="color: #0284c7; font-size: 1.05rem; font-weight: bold;">{candidate}</span>
                    </div>
                </div>
            </div>
            """)

    with col_d2:
        st.html(f"""
        <div class="cyber-metric-card" style="margin-bottom: 12px;">
            <div class="cyber-metric-label">ĐIỂM CAO NHẤT (p_max)</div>
            <div class="cyber-metric-value" style="font-size: 1.5rem; color: #0284c7;">{p_max_val:.6f}</div>
        </div>
        <div class="cyber-metric-card" style="border-top-color: #ef4444;">
            <div class="cyber-metric-label">NGƯỠNG QUYẾT ĐỊNH (&tau;)</div>
            <div class="cyber-metric-value" style="font-size: 1.5rem; color: #ef4444;">{tau:.2f}</div>
        </div>
        """)

    with col_d3:
        # Biểu đồ phân phối scores trên nền trắng tinh tế
        fig, ax = plt.subplots(figsize=(5, 2.6), facecolor="#ffffff")
        ax.set_facecolor("#ffffff")

        class_names = list(pipeline.classes_)
        scores = [row_result[f"Score_{c}"] for c in class_names]

        # Bảng màu an ninh mạng rõ ràng trên nền sáng
        color_map = {
            "Benign": "#10b981",
            "Malware": "#ef4444",
            "Ransomware": "#e11d48",
            "Spyware": "#f59e0b",
            "Trojan": "#8b5cf6"
        }
        bar_colors = [color_map.get(c, "#0284c7") for c in class_names]

        bars = ax.barh(class_names, scores, color=bar_colors, alpha=0.88, edgecolor="none", height=0.55)
        ax.axvline(tau, color="#dc2626", linestyle="--", linewidth=1.6, label=f"tau = {tau:.2f} [CUTOFF]")
        ax.set_xlim(0, 1.05)

        # Style trục và grid nền sáng
        ax.tick_params(colors="#334155", labelsize=8.5)
        ax.xaxis.label.set_color("#334155")
        ax.set_xlabel("Scores dự đoán", fontsize=8.5, weight="bold")
        ax.set_title("PHÂN PHỐI ĐIỂM CÁC LỚP", fontsize=9.5, color="#0f172a", weight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#cbd5e1")
        ax.spines["bottom"].set_color("#cbd5e1")
        ax.grid(axis="x", color="#e2e8f0", linestyle=":", linewidth=0.8)
        ax.legend(loc="lower right", fontsize=7.5, facecolor="#f8fafc", edgecolor="#cbd5e1", labelcolor="#0f172a")

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ---------------------------------------------------- Local Top-3 Features Explanation
    st.markdown("#### 🎯 GIẢI THÍCH CỤC BỘ TOP-3 ĐẶC TRƯNG TÁC ĐỘNG CAO NHẤT (PRED_CONTRIB)")

    if not is_accepted:
        st.html(f"""
        <div style="background: #fffbeb; border: 1px solid #fde68a; border-left: 5px solid #d97706; padding: 12px 16px; border-radius: 6px; font-family: 'JetBrains Mono', monospace; color: #b45309; font-size: 0.88rem; margin-bottom: 14px;">
            ⚠️ [CAUTION // UNCERTAIN STATE]: Mẫu chưa đủ độ tin cậy để tiếp nhận. Giải thích dưới đây được tính cho <b>lớp ứng viên [{candidate}]</b> (lớp có điểm cao nhất).
        </div>
        """)
    else:
        st.html(f"""
        <div style="background: #f0f9ff; border: 1px solid #bae6fd; border-left: 5px solid #0284c7; padding: 12px 16px; border-radius: 6px; font-family: 'JetBrains Mono', monospace; color: #0369a1; font-size: 0.88rem; margin-bottom: 14px;">
            🔍 [EXPLANATION SCOPE]: Giải thích đóng góp cho lớp được phân loại: <b>[{final_res}]</b>.
        </div>
        """)

    try:
        explanation = compute_local_top3(pipeline, row_data, task=task_key)
        top3_df = explanation["top3"]
        target_cls = explanation["target_class"]
        raw_margin = explanation["target_raw_score"]
        baseline_val = explanation["baseline"]
        sum_contribs = explanation["sum_contributions"]
        parity_diff = explanation["parity_diff"]
        is_parity_ok = explanation["is_parity_ok"]

        st.caption(
            f"Đơn vị: **Đóng góp vào raw score (margin) của lớp {target_cls}** từ mô hình LightGBM (pred_contrib)."
        )

        col_t1, col_t2 = st.columns([1.2, 1])
        with col_t1:
            st.markdown("##### Bảng Top-3 Features ảnh hưởng lớn nhất:")
            table_top3 = top3_df[["feature", "original_value", "contribution"]].copy()
            table_top3.columns = ["Đặc trưng bộ nhớ (Feature)", "Giá trị gốc (RAM)", "Đóng góp (Contribution)"]
            table_top3["Chiều hướng"] = table_top3["Đóng góp (Contribution)"].apply(
                lambda x: "🟢 Thúc đẩy" if x > 0 else "🔴 Kìm hãm"
            )
            st.dataframe(
                table_top3.style.format({
                    "Giá trị gốc (RAM)": "{:,.4f}",
                    "Đóng góp (Contribution)": "{:+.6f}"
                }),
                use_container_width=True,
                hide_index=True
            )

        with col_t2:
            st.markdown("##### Biểu đồ đóng góp Top-3:")
            fig2, ax2 = plt.subplots(figsize=(5, 2.5), facecolor="#ffffff")
            ax2.set_facecolor("#ffffff")

            f_names = top3_df["feature"].tolist()
            c_vals = top3_df["contribution"].tolist()
            c_colors = ["#10b981" if v > 0 else "#ef4444" for v in c_vals]
            y_pos = np.arange(len(f_names))

            ax2.barh(y_pos, c_vals, color=c_colors, alpha=0.88, edgecolor="none", height=0.5)
            ax2.axvline(0, color="#94a3b8", linewidth=0.8, linestyle="--")
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(f_names, fontsize=8, color="#0f172a", fontfamily="monospace")
            ax2.invert_yaxis()
            ax2.tick_params(colors="#334155", labelsize=8)
            ax2.xaxis.label.set_color("#334155")
            ax2.set_xlabel("Đóng góp vào raw score", fontsize=8, weight="bold")
            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(False)
            ax2.spines["left"].set_color("#cbd5e1")
            ax2.spines["bottom"].set_color("#cbd5e1")
            ax2.grid(axis="x", color="#e2e8f0", linestyle=":", linewidth=0.8)

            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

        # Hộp kiểm chứng độ khớp Raw Score (Cryptographic Parity Log Terminal)
        with st.expander("🔍 [CRYPTOGRAPHIC PARITY LOG // ĐỐI SOÁT ĐỘ KHỚP RAW SCORE]"):
            st.html(f"""
            <div class="cyber-terminal-box">
                [TELEMETRY AUDIT LOG // PRED_CONTRIB PARITY CHECK]<br>
                =======================================================<br>
                &gt;&gt; TARGET CLASS              : {target_cls}<br>
                &gt;&gt; TARGET RAW MARGIN (LGBM)   : {raw_margin:+.8f}<br>
                &gt;&gt; MODEL BASELINE BIAS        : {baseline_val:+.8f}<br>
                &gt;&gt; SUM 55 FEATURE VECTORS    : {sum_contribs:+.8f}<br>
                &gt;&gt; RECONSTRUCTED RAW MARGIN  : {(sum_contribs + baseline_val):+.8f}<br>
                &gt;&gt; PARITY DELTA RESIDUAL     : {parity_diff:+.10e}<br>
                &gt;&gt; AUDIT STATUS              : {'[VERIFIED // EXACT MATCH ✅]' if is_parity_ok else '[PARITY MISMATCH ⚠️]'}<br>
                =======================================================<br>
                * Note: Với Binary Benign, toàn bộ 55 đóng góp và baseline được đảo dấu để giải thích margin đối nghịch. Multiclass trích xuất đúng khối lớp ứng viên từ ma trận (4 x 56).
            </div>
            """)

    except Exception as e_exp:
        st.error(f"[EXPLANATION_FAULT] Lỗi khi tính toán giải thích cục bộ: {e_exp}")


if __name__ == "__main__":
    main()
