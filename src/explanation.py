"""Module explanation: giải thích cục bộ (Local Top-3) theo cơ chế Tree Booster pred_contrib.

Quy định theo mục 4 của tài liệu bàn giao TV5:
1. LightGBM:
   - Tiền xử lý: pipeline.named_steps["preprocessing"].transform(row)
   - Dự đoán: pipeline.named_steps["classifier"].booster_.predict(..., pred_contrib=True)
   - Binary: 55 đóng góp + 1 baseline ở cuối. Margin gốc thuộc Malware. Với Benign: đảo dấu đóng góp và baseline.
   - Multiclass: 4 khối (mỗi khối 55 đóng góp + 1 baseline). Chọn khối của lớp ứng viên (hoặc lớp chỉ định).
   - Xếp hạng theo |contrib| giảm dần (sau khi bỏ baseline). Giữ nguyên dấu và đơn vị raw score.
   - Kiểm chứng tính toàn vẹn (Parity): sum(contributions) + baseline == target_raw_score.
2. Random Forest:
   - Hiển thị trạng thái giải thích cục bộ chưa hỗ trợ.
"""
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def compute_local_top3(
    pipeline: Any,
    row_features: pd.DataFrame,
    task: str,
    target_class_override: Optional[str] = None
) -> Dict[str, Any]:
    """Tính giải thích cục bộ Top-3 đặc trưng cho 1 dòng dữ liệu.

    Tự động nhận diện thuật toán:
    - Nếu là Random Forest: trả về trạng thái chưa hỗ trợ.
    - Nếu là LightGBM: tính toán pred_contrib, xếp hạng Top-3 và kiểm chứng parity.
    """
    if len(row_features) != 1:
        row_features = row_features.iloc[[0]]

    classifier = pipeline.named_steps.get("classifier", None)
    if classifier is None:
        return {
            "supported": False,
            "message": "Không tìm thấy classifier step trong pipeline."
        }

    # Kiểm tra thuật toán: Random Forest không có booster_
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

    # 1. Dự đoán xác suất để xác định lớp ứng viên
    probas = pipeline.predict_proba(row_features)[0]
    argmax_idx = int(np.argmax(probas))
    candidate_class = classes[argmax_idx]

    target_class = target_class_override if target_class_override is not None else candidate_class
    target_class_idx = classes.index(target_class)

    # 2. Tiền xử lý dữ liệu qua fitted preprocessing
    preprocessor = pipeline.named_steps["preprocessing"]
    X_transformed = preprocessor.transform(row_features)

    # 3. Tính pred_contrib và raw_score từ booster
    booster = classifier.booster_
    raw_scores = booster.predict(X_transformed, raw_score=True)[0]
    contribs = booster.predict(X_transformed, pred_contrib=True)[0]

    if task == "binary":
        if contribs.shape[0] != 56:
            raise ValueError(f"Kỳ vọng 56 giá trị pred_contrib cho binary, thực tế nhận được: {contribs.shape}")

        feat_contribs = contribs[:55].copy()
        baseline = float(contribs[55])
        raw_malware = float(raw_scores)

        # Malware là lớp dương (lớp 1), Benign là lớp 0
        if target_class == "Benign":
            # Đảo dấu cả 55 đóng góp và baseline để giải thích margin đối nghịch
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
        raise ValueError(f"Task không hỗ trợ: '{task}'. Chỉ chấp nhận 'binary' hoặc 'multiclass'.")

    # 4. Kiểm chứng tính khớp: tổng 55 đóng góp + baseline == raw_score
    sum_contributions = float(np.sum(feat_contribs))
    total_reconstructed = sum_contributions + baseline
    parity_diff = abs(total_reconstructed - target_raw_score)
    is_parity_ok = parity_diff < 1e-4

    # 5. Xếp hạng: bỏ baseline, chọn top 3 theo |contrib| giảm dần, giữ nguyên dấu
    orig_values = row_features.iloc[0].values
    df_contribs = pd.DataFrame({
        "feature": feature_names,
        "original_value": orig_values,
        "contribution": feat_contribs,
        "abs_contrib": np.abs(feat_contribs)
    })

    df_sorted = df_contribs.sort_values(by="abs_contrib", ascending=False).reset_index(drop=True)
    top3_df = df_sorted.head(3).copy()

    # Thêm mô tả hướng tác động
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
