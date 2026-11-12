"""Unit tests cho triage, kiểm tra hợp đồng dữ liệu, suy luận và giải thích cục bộ."""
import io
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing import load_splits
from src.explanation import compute_local_top3
from src.triage import (
    get_project_root,
    load_operating_config,
    load_pipeline_for_task,
    predict_batch,
    validate_csv_input,
)


class TestTriage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_operating_config()
        cls.tau = cls.config["tau"]
        cls.pipe_a, cls.info_a = load_pipeline_for_task("binary", cls.config)
        cls.pipe_b, cls.info_b = load_pipeline_for_task("multiclass", cls.config)
        cls.features_a = list(cls.pipe_a.feature_names_in_)
        cls.features_b = list(cls.pipe_b.feature_names_in_)

    def test_pipeline_loading_and_sha256(self):
        """Kiểm tra nạp pipeline, xác thực SHA-256 và classes_."""
        self.assertEqual(len(self.features_a), 55)
        self.assertEqual(len(self.features_b), 55)
        self.assertEqual(self.features_a, self.features_b)
        self.assertEqual(list(self.pipe_a.classes_), ["Benign", "Malware"])
        self.assertEqual(list(self.pipe_b.classes_), ["Benign", "Ransomware", "Spyware", "Trojan"])

        with self.assertRaises(ValueError):
            load_pipeline_for_task("invalid_task")

    def test_csv_validation_valid_and_reordered(self):
        """CSV đúng 55 features và CSV đảo thứ tự cột đều được xác thực và sắp xếp đúng."""
        # 1. Tạo mẫu từ validation
        val_X, _ = load_splits("multiclass")["validation"]
        sample_row = val_X.iloc[[0]].copy()

        # CSV chuẩn
        csv_str = sample_row.to_csv(index=False)
        validated = validate_csv_input(csv_str, self.features_a)
        self.assertEqual(list(validated.columns), self.features_a)
        self.assertEqual(len(validated), 1)

        # CSV đảo thứ tự cột
        rev_features = list(reversed(self.features_a))
        reordered_row = sample_row[rev_features]
        csv_reordered = reordered_row.to_csv(index=False)
        validated_rev = validate_csv_input(csv_reordered, self.features_a)
        self.assertEqual(list(validated_rev.columns), self.features_a)
        np.testing.assert_array_equal(validated.to_numpy(), validated_rev.to_numpy())

    def test_csv_validation_empty_table_and_no_data(self):
        """Báo lỗi khi file rỗng hoặc chỉ có header không có dòng dữ liệu."""
        with self.assertRaisesRegex(ValueError, "Tập tin CSV rỗng"):
            validate_csv_input("", self.features_a)

        header_only = ",".join(self.features_a) + "\n"
        with self.assertRaisesRegex(ValueError, "Bảng dữ liệu rỗng"):
            validate_csv_input(header_only, self.features_a)

    def test_csv_validation_duplicate_columns(self):
        """Báo lỗi khi có cột trùng lặp trong header."""
        header_with_dup = "colA,colB,colA\n1,2,3\n"
        with self.assertRaisesRegex(ValueError, "trùng lặp"):
            validate_csv_input(header_with_dup, ["colA", "colB"])

    def test_csv_validation_missing_and_extra_columns(self):
        """Báo lỗi rõ ràng khi thiếu cột hoặc dư cột ngoài schema 55 features."""
        val_X, _ = load_splits("multiclass")["validation"]
        sample = val_X.iloc[[0]].copy()

        # Thiếu cột
        missing_one = sample.drop(columns=[self.features_a[0]])
        with self.assertRaisesRegex(ValueError, "Thiếu 1 cột"):
            validate_csv_input(missing_one.to_csv(index=False), self.features_a)

        # Dư cột (ví dụ Class, Category, Filename)
        extra_df = sample.copy()
        extra_df["Class"] = "Benign"
        extra_df["Category"] = "Benign"
        with self.assertRaisesRegex(ValueError, "Dư 2 cột"):
            validate_csv_input(extra_df.to_csv(index=False), self.features_a)

    def test_csv_validation_garbage_text_and_infinity(self):
        """Báo lỗi khi có chữ rác hoặc infinity, nhưng giữ NaN cho imputer."""
        val_X, _ = load_splits("multiclass")["validation"]
        sample = val_X.iloc[[0]].copy()

        # Chữ rác
        bad_sample = sample.copy()
        bad_sample[self.features_a[0]] = "CORRUPTED_STRING"
        with self.assertRaisesRegex(ValueError, "chữ rác"):
            validate_csv_input(bad_sample.to_csv(index=False), self.features_a)

        # Infinity
        inf_sample = sample.copy()
        inf_sample[self.features_a[0]] = np.inf
        with self.assertRaisesRegex(ValueError, "vô cực"):
            validate_csv_input(inf_sample.to_csv(index=False), self.features_a)

        # NaN được giữ lại để SimpleImputer điền median
        nan_sample = sample.copy()
        nan_sample[self.features_a[0]] = np.nan
        validated_nan = validate_csv_input(nan_sample.to_csv(index=False), self.features_a)
        self.assertTrue(np.isnan(validated_nan[self.features_a[0]].iloc[0]))
        # Pipeline predict chạy thành công trên dữ liệu có NaN nhờ median imputer
        proba = self.pipe_a.predict_proba(validated_nan)
        self.assertEqual(proba.shape, (1, 2))

    def test_boundary_threshold_and_rejection(self):
        """Kiểm tra logic nhận/từ chối tại ngưỡng biên tau = 0.65."""
        # Giả lập xác suất: đúng 0.65 thì nhận, 0.649999 thì từ chối (Uncertain)
        class DummyPipeline:
            classes_ = np.array(["Benign", "Malware"])
            def __init__(self, p):
                self.p = p
            def predict_proba(self, X):
                return np.array([self.p])

        X_dummy = pd.DataFrame(np.zeros((1, 55)), columns=self.features_a)

        # Case 1: p_max == 0.650000 -> Chấp nhận
        p1 = [0.65, 0.35]
        res1, _ = predict_batch(DummyPipeline(p1), X_dummy, tau=0.65)
        self.assertEqual(res1["Kết quả"].iloc[0], "Benign")
        self.assertTrue(res1["Đạt ngưỡng"].iloc[0])

        # Case 2: p_max == 0.649999 -> Uncertain, ghi lớp ứng viên Benign
        p2 = [0.649999, 0.350001]
        res2, _ = predict_batch(DummyPipeline(p2), X_dummy, tau=0.65)
        self.assertEqual(res2["Kết quả"].iloc[0], "Uncertain")
        self.assertEqual(res2["Lớp ứng viên"].iloc[0], "Benign")
        self.assertFalse(res2["Đạt ngưỡng"].iloc[0])

    def test_known_validation_rows(self):
        """Kiểm chứng đúng mẫu xác nhận trong handoff:

        - Task A Row 14627 -> Benign, p_max ~ 0.999990
        - Task B Row 11611 -> Benign, p_max ~ 0.999997
        - Task B Row 43302 -> Uncertain (candidate: Spyware), p_max ~ 0.338679
        """
        X_val, _ = load_splits("multiclass")["validation"]

        # 1. Task A row 14627
        row_14627 = X_val.loc[[14627]]
        res_a, probas_a = predict_batch(self.pipe_a, row_14627, tau=self.tau)
        self.assertEqual(res_a["Kết quả"].iloc[0], "Benign")
        self.assertAlmostEqual(res_a["p_max"].iloc[0], 0.999990, places=5)

        # 2. Task B row 11611
        row_11611 = X_val.loc[[11611]]
        res_b1, probas_b1 = predict_batch(self.pipe_b, row_11611, tau=self.tau)
        self.assertEqual(res_b1["Kết quả"].iloc[0], "Benign")
        self.assertAlmostEqual(res_b1["p_max"].iloc[0], 0.999997, places=5)

        # 3. Task B row 43302
        row_43302 = X_val.loc[[43302]]
        res_b2, probas_b2 = predict_batch(self.pipe_b, row_43302, tau=self.tau)
        self.assertEqual(res_b2["Kết quả"].iloc[0], "Uncertain")
        self.assertEqual(res_b2["Lớp ứng viên"].iloc[0], "Spyware")
        self.assertAlmostEqual(res_b2["p_max"].iloc[0], 0.338679, places=5)

    def test_local_top3_explanation_parity(self):
        """Kiểm chứng độ khớp giữa tổng đóng góp + baseline với raw score của lớp giải thích."""
        X_val, _ = load_splits("multiclass")["validation"]

        # Binary: Row 14627 (Benign)
        exp_a = compute_local_top3(self.pipe_a, X_val.loc[[14627]], task="binary")
        self.assertEqual(exp_a["target_class"], "Benign")
        self.assertTrue(exp_a["is_parity_ok"])
        self.assertLess(exp_a["parity_diff"], 1e-4)
        self.assertEqual(len(exp_a["top3"]), 3)
        # Baseline không nằm trong top3
        self.assertNotIn("baseline", exp_a["top3"]["feature"].values)
        # Top 3 được sắp xếp theo trị tuyệt đối giảm dần
        abs_vals_a = exp_a["top3"]["abs_contrib"].values
        self.assertTrue(abs_vals_a[0] >= abs_vals_a[1] >= abs_vals_a[2])

        # Multiclass: Row 11611 (Benign)
        exp_b1 = compute_local_top3(self.pipe_b, X_val.loc[[11611]], task="multiclass")
        self.assertEqual(exp_b1["target_class"], "Benign")
        self.assertTrue(exp_b1["is_parity_ok"])
        self.assertLess(exp_b1["parity_diff"], 1e-4)
        self.assertEqual(len(exp_b1["top3"]), 3)

        # Multiclass: Row 43302 (Uncertain, candidate Spyware)
        exp_b2 = compute_local_top3(self.pipe_b, X_val.loc[[43302]], task="multiclass")
        self.assertEqual(exp_b2["target_class"], "Spyware")
        self.assertTrue(exp_b2["is_parity_ok"])
        self.assertLess(exp_b2["parity_diff"], 1e-4)
        self.assertEqual(len(exp_b2["top3"]), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
