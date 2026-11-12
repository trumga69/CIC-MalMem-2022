"""Module explanation: tính giải thích cục bộ Top-3 đặc trưng cho dòng dữ liệu đang xem."""
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def compute_local_top3(
    pipeline: Any,
    row_features: pd.DataFrame,
    task: str,
    target_class_override: Optional[str] = None
) -> Dict[str, Any]:
    """Tính giải thích cục bộ Top-3 đặc trưng theo LightGBM pred_contrib.

    Quy trình:
    1. Transform dòng dữ liệu qua pipeline.named_steps['preprocessing'].
    2. Gọi booster.predict(..., pred_contrib=True).
    3. Binary:
       - 56 giá trị: 55 đóng góp features + 1 baseline ở vị trí cuối.
       - Dạng gốc là margin lớp dương Malware.
       - Với Benign: đảo dấu cả 55 đóng góp và baseline để giải thích margin đối nghịch.
    4. Multiclass:
       - 224 giá trị: reshape thành (4, 56).
       - Chọn khối của lớp argmax (hoặc lớp ứng viên khi Uncertain).
       - Phần tử thứ 56 (index 55) trong khối là baseline.
    5. Loại baseline khỏi ranking, chọn 3 features có trị tuyệt đối đóng góp lớn nhất (|contrib| desc).
    6. Giữ nguyên dấu đóng góp và giá trị gốc ban đầu.
    7. Kiểm chứng: tổng 55 đóng góp + baseline == raw_score của lớp đang giải thích.
    """
    if len(row_features) != 1:
        row_features = row_features.iloc[[0]]

    feature_names = list(pipeline.feature_names_in_)
    classes = list(pipeline.classes_)

    # 1. Dự đoán xác suất để xác định lớp giải thích (argmax / ứng viên)
    probas = pipeline.predict_proba(row_features)[0]
    argmax_idx = int(np.argmax(probas))
    candidate_class = classes[argmax_idx]

    target_class = target_class_override if target_class_override is not None else candidate_class
    target_class_idx = classes.index(target_class)

    # 2. Tiền xử lý dữ liệu qua fitted preprocessing
    preprocessor = pipeline.named_steps["preprocessing"]
    X_transformed = preprocessor.transform(row_features)

    # 3. Tính pred_contrib và raw_score từ booster
    classifier = pipeline.named_steps["classifier"]
    booster = classifier.booster_

    raw_scores = booster.predict(X_transformed, raw_score=True)[0]
    contribs = booster.predict(X_transformed, pred_contrib=True)[0]

    if task == "binary":
        if contribs.shape[0] != 56:
            raise ValueError(f"Kỳ vọng 56 giá trị pred_contrib cho binary, thực tế nhận được: {contribs.shape}")

        feat_contribs = contribs[:55].copy()
        baseline = float(contribs[55])
        raw_malware = float(raw_scores)

        # Malware là lớp 1, Benign là lớp 0
        if target_class == "Benign":
            # Đảo dấu cả đóng góp và baseline để giải thích margin đối nghịch
            feat_contribs = -feat_contribs
            baseline = -baseline
            target_raw_score = -raw_malware
        else:
            target_raw_score = raw_malware

    elif task == "multiclass":
        if contribs.shape[0] != 224:
            raise ValueError(f"Kỳ vọng 224 giá trị pred_contrib cho multiclass, thực tế nhận được: {contribs.shape}")

        contribs_matrix = contribs.reshape(len(classes), 56)
        cls_block = contribs_matrix[target_class_idx]
        feat_contribs = cls_block[:55].copy()
        baseline = float(cls_block[55])
        target_raw_score = float(raw_scores[target_class_idx])

    else:
        raise ValueError(f"Task không hỗ trợ: {task}")

    # 4. Kiểm chứng tính khớp: tổng đóng góp + baseline == raw score
    sum_contributions = float(np.sum(feat_contribs))
    total_reconstructed = sum_contributions + baseline
    parity_diff = abs(total_reconstructed - target_raw_score)
    is_parity_ok = parity_diff < 1e-4

    # 5. Xếp hạng: loại baseline, chọn top 3 theo |contrib| giảm dần, giữ dấu
    orig_values = row_features.iloc[0].values
    df_contribs = pd.DataFrame({
        "feature": feature_names,
        "original_value": orig_values,
        "contribution": feat_contribs,
        "abs_contrib": np.abs(feat_contribs)
    })

    df_sorted = df_contribs.sort_values(by="abs_contrib", ascending=False).reset_index(drop=True)
    top3_df = df_sorted.head(3).copy()

    return {
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
