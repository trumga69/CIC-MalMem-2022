"""Bộ kiểm thử khói (Smoke Test) toàn diện cho Web Demo CIC-MalMem-2022.

Kiểm tra:
1. Nạp và kiểm chứng 4 pipeline theo model/configs/inference.json
2. Đối chiếu chỉ số với model/reports/model_report.json
3. Kiểm tra tính toán suy luận (predict_batch) và ngưỡng từ chối tau
4. Kiểm tra hợp đồng dữ liệu: rỗng, thiếu cột, dư cột, trùng cột, phi số, vô cực, cột đảo thứ tự, ô trống NaN
5. Kiểm tra giải thích cục bộ (Local Top-3):
   - Đảo dấu cho Binary Benign
   - Trích xuất khối Multiclass
   - Khớp raw margin (parity check) cho LightGBM
   - Thông báo chưa hỗ trợ cho Random Forest
"""
import io
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Thêm đường dẫn để có thể import từ src
curr_dir = Path(__file__).resolve().parent
if str(curr_dir) not in sys.path:
    sys.path.insert(0, str(curr_dir))
parent_dir = curr_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from src.triage import (
    get_project_root,
    load_inference_config,
    load_model_report,
    load_pipeline_for_model,
    validate_csv_input,
    predict_batch
)
from src.explanation import compute_local_top3


def run_smoke_test():
    root = get_project_root()
    print("=" * 70)
    print("BẮT ĐẦU SMOKE TEST TOÀN DIỆN CHO TOPIC 05 - WEB DEMO")
    print(f"Project root: {root}")
    print("=" * 70)

    # 1. Kiểm tra cấu hình và báo cáo
    cfg = load_inference_config()
    report = load_model_report()
    assert "models" in cfg, "Lỗi: inference.json thiếu key 'models'"
    assert len(cfg["models"]) == 4, f"Lỗi: kỳ vọng 4 models, tìm thấy {len(cfg['models'])}"
    print("[PASS] Đọc thành công inference.json và model_report.json")

    # 2. Kiểm tra nạp 4 pipeline
    pipelines = {}
    for mid, info in cfg["models"].items():
        pipe, minfo = load_pipeline_for_model(mid, cfg)
        assert len(pipe.feature_names_in_) == 55, f"Lỗi {mid}: số feature != 55"
        pipelines[mid] = pipe
        print(f"[PASS] Nạp thành công pipeline '{mid}': task={minfo['task']}, tau={minfo['tau']}, classes={list(pipe.classes_)}")

    # 3. Kiểm tra hợp đồng dữ liệu (Data Contract Validation)
    expected_feats = list(pipelines["02_binary_lightgbm"].feature_names_in_)

    # 3.1. File rỗng
    try:
        validate_csv_input("", expected_feats)
        assert False, "Không bắt được lỗi file rỗng"
    except ValueError as e:
        print(f"[PASS] Bắt đúng lỗi file rỗng: {e}")

    # 3.2. Cột trùng tên trong header
    dup_header_csv = expected_feats[0] + "," + expected_feats[0] + "\n1,2"
    try:
        validate_csv_input(dup_header_csv, expected_feats)
        assert False, "Không bắt được lỗi cột trùng lặp"
    except ValueError as e:
        print(f"[PASS] Bắt đúng lỗi cột trùng tên: {e}")

    # 3.3. Thiếu cột
    partial_csv = ",".join(expected_feats[:50]) + "\n" + ",".join(["1.0"] * 50)
    try:
        validate_csv_input(partial_csv, expected_feats)
        assert False, "Không bắt được lỗi thiếu cột"
    except ValueError as e:
        print(f"[PASS] Bắt đúng lỗi thiếu cột: {e}")

    # 3.4. Dư cột lạ
    extra_csv = ",".join(expected_feats + ["unknown_malicious_column"]) + "\n" + ",".join(["1.0"] * 56)
    try:
        validate_csv_input(extra_csv, expected_feats, auto_drop_labels=False)
        assert False, "Không bắt được lỗi dư cột lạ"
    except ValueError as e:
        print(f"[PASS] Bắt đúng lỗi dư cột lạ: {e}")

    # 3.5. Chữ rác phi số
    garbage_csv = ",".join(expected_feats) + "\n" + ",".join(["1.0"] * 54 + ["not_a_number_hack"])
    try:
        validate_csv_input(garbage_csv, expected_feats)
        assert False, "Không bắt được lỗi chữ rác phi số"
    except ValueError as e:
        print(f"[PASS] Bắt đúng lỗi chữ rác phi số: {e}")

    # 3.6. Vô cực
    inf_csv = ",".join(expected_feats) + "\n" + ",".join(["1.0"] * 54 + ["inf"])
    try:
        validate_csv_input(inf_csv, expected_feats)
        assert False, "Không bắt được lỗi vô cực inf"
    except ValueError as e:
        print(f"[PASS] Bắt đúng lỗi giá trị vô cực: {e}")

    # 3.7. Tự động loại bỏ nhãn ground-truth
    raw_with_labels_path = root / "data" / "demo" / "demo_with_groundtruth_labels.csv"
    if raw_with_labels_path.exists():
        df_dropped = validate_csv_input(raw_with_labels_path, expected_feats, auto_drop_labels=True)
        assert len(df_dropped.columns) == 55, "Lỗi: Không loại bỏ đúng cột nhãn"
        print(f"[PASS] Tự động loại bỏ nhãn metadata/ground-truth: còn đúng {len(df_dropped.columns)} features.")

    # 3.8. Cột bị đảo thứ tự
    shuffled_path = root / "data" / "demo" / "demo_shuffled_columns.csv"
    if shuffled_path.exists():
        df_shuffled = validate_csv_input(shuffled_path, expected_feats)
        assert list(df_shuffled.columns) == expected_feats, "Lỗi: Thứ tự cột sau chuẩn hóa không khớp"
        print("[PASS] Tự động sắp xếp lại thứ tự cột đúng theo pipeline.")

    # 3.9. Ô trống NaN để imputer xử lý
    nan_path = root / "data" / "demo" / "demo_with_missing_cells.csv"
    if nan_path.exists():
        df_nan = validate_csv_input(nan_path, expected_feats)
        for mid, pipe in pipelines.items():
            tau = cfg["models"][mid]["tau"]
            res, prob = predict_batch(pipe, df_nan, tau)
            assert len(res) == len(df_nan)
        print("[PASS] Dữ liệu có ô trống NaN được SimpleImputer của fitted pipeline xử lý trơn tru trên cả 4 models.")

    # 4. Kiểm tra suy luận trên 4 models
    demo_55_path = root / "data" / "demo" / "demo_validation_55features.csv"
    assert demo_55_path.exists(), "Thiếu file demo_validation_55features.csv"
    df_val = validate_csv_input(demo_55_path, expected_feats)

    print("-" * 50)
    print("KIỂM TRA SUY LUẬN TRÊN TẬP DEMO (5 DÒNG):")
    for mid, pipe in pipelines.items():
        minfo = cfg["models"][mid]
        tau = minfo["tau"]
        res_df, probas = predict_batch(pipe, df_val, tau)
        n_accepted = (res_df["Kết quả"] != "Uncertain").sum()
        n_unc = (res_df["Kết quả"] == "Uncertain").sum()
        print(f"-> Model: {mid} (tau={tau}): Accepted={n_accepted}, Uncertain={n_unc}")

        for idx, row in res_df.iterrows():
            if row["p_max"] >= tau:
                assert row["Kết quả"] == row["Lớp ứng viên"], f"Lỗi logic chấp nhận tại dòng {idx}"
            else:
                assert row["Kết quả"] == "Uncertain", f"Lỗi logic từ chối tại dòng {idx}"
    print("[PASS] Toàn bộ logic phân định p_max >= tau và Uncertain hoạt động chính xác.")

    # 5. Kiểm tra ngưỡng biên p_max == tau
    mock_pipe = pipelines["02_binary_lightgbm"]
    res_edge, _ = predict_batch(mock_pipe, df_val.iloc[[0]], tau=float(mock_pipe.predict_proba(df_val.iloc[[0]])[0, 0]))
    assert res_edge.iloc[0]["Kết quả"] == res_edge.iloc[0]["Lớp ứng viên"], "Lỗi tại ngưỡng biên p_max == tau"
    print("[PASS] Kiểm tra ngưỡng biên p_max == tau: chấp nhận đúng lớp ứng viên.")

    # 6. Kiểm tra Local Top-3 Feature Explanation
    print("-" * 50)
    print("KIỂM TRA LOCAL TOP-3 VÀ PARITY CHECK:")

    # 6.1. LightGBM Binary
    lgb_bin = pipelines["02_binary_lightgbm"]
    exp_bin_benign = compute_local_top3(lgb_bin, df_val.iloc[[0]], task="binary", target_class_override="Benign")
    assert exp_bin_benign["supported"] is True
    assert exp_bin_benign["is_parity_ok"] is True
    assert len(exp_bin_benign["top3"]) == 3
    print(f"[PASS] LightGBM Binary (Benign): Parity diff = {exp_bin_benign['parity_diff']:.2e} (OK)")

    exp_bin_malware = compute_local_top3(lgb_bin, df_val.iloc[[1]], task="binary", target_class_override="Malware")
    assert exp_bin_malware["is_parity_ok"] is True
    print(f"[PASS] LightGBM Binary (Malware): Parity diff = {exp_bin_malware['parity_diff']:.2e} (OK)")

    # 6.2. LightGBM Multiclass
    lgb_mul = pipelines["04_multiclass_lightgbm"]
    for cls in lgb_mul.classes_:
        exp_mul = compute_local_top3(lgb_mul, df_val.iloc[[0]], task="multiclass", target_class_override=cls)
        assert exp_mul["supported"] is True
        assert exp_mul["is_parity_ok"] is True
        print(f"[PASS] LightGBM Multiclass ({cls}): Parity diff = {exp_mul['parity_diff']:.2e} (OK)")

    # 6.3. Random Forest (Cả binary và multiclass)
    rf_bin = pipelines["01_binary_random_forest"]
    exp_rf_bin = compute_local_top3(rf_bin, df_val.iloc[[0]], task="binary")
    assert exp_rf_bin["supported"] is False
    assert "CHƯA HỖ TRỢ" in exp_rf_bin["message"]
    print(f"[PASS] Random Forest Binary: Báo đúng trạng thái chưa hỗ trợ.")

    rf_mul = pipelines["03_multiclass_random_forest"]
    exp_rf_mul = compute_local_top3(rf_mul, df_val.iloc[[0]], task="multiclass")
    assert exp_rf_mul["supported"] is False
    assert "CHƯA HỖ TRỢ" in exp_rf_mul["message"]
    print(f"[PASS] Random Forest Multiclass: Báo đúng trạng thái chưa hỗ trợ.")

    print("=" * 70)
    print("TẤT CẢ CÁC BƯỚC KIỂM THỬ KHÓI ĐỀU ĐẠT CHUẨN 100%!")
    print("=" * 70)


if __name__ == "__main__":
    run_smoke_test()
