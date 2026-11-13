# Bàn giao TV5 — Web demo CIC-MalMem-2022

Cập nhật ngày 13/11/2026. Phạm vi là web demo chạy cục bộ cho Topic 05, mục B/D/E, trang 16–17 của Final term.pdf.

## 1. Cấu hình model

LightGBM là model chính và mặc định ở cả hai task. Random Forest có thể chọn trong từng task để so sánh. Hai task chạy inference độc lập.

| Task | Thuật toán | ID trong config | τ | Vai trò |
|---|---|---|---:|---|
| A / binary | LightGBM | `02_binary_lightgbm` | 0,95 | Mặc định |
| A / binary | Random Forest | `01_binary_random_forest` | 0,95 | Tùy chọn |
| B / multiclass | LightGBM | `04_multiclass_lightgbm` | 0,80 | Mặc định |
| B / multiclass | Random Forest | `03_multiclass_random_forest` | 0,70 | Tùy chọn |

Nhãn của Task A là Benign và Malware. Nhãn của Task B là Benign, Ransomware, Spyware và Trojan. Dùng `pipeline.classes_` để ánh xạ đúng từng cột điểm.

## 2. Files và môi trường

| File hoặc thư mục | Mục đích |
|---|---|
| `model/configs/inference.json` | Nguồn cấu hình inference gồm model mặc định, đường dẫn artifact, τ của mỗi model và nhãn `Uncertain`. |
| `model/artifacts/` | Bốn fitted pipeline `.joblib` đã xuất. Resolve đường dẫn `artifact` từ project root và chỉ nạp artifact nội bộ đã bàn giao. |
| `preprocessing_data/preprocessing.py` | Module preprocessing cần có khi nạp pipeline. |
| `model/reports/model_report.json` | Báo cáo chung của bốn notebook, gồm kết quả validation và test theo từng model. |

Đã nạp lại cả bốn artifact và kiểm tra mỗi pipeline có 55 đặc trưng. Kết quả `predict_proba` khớp với kết quả trong notebook trên toàn bộ 8.791 dòng test.

Khi cần tái tạo bộ bàn giao, chạy lần lượt bốn notebook trong `model/notebooks/`. Hai cell cuối của mỗi notebook cập nhật mục model tương ứng trong báo cáo chung và xuất pipeline theo đường dẫn trong `inference.json`.

Môi trường model đã dùng Python 3.11.9, LightGBM 4.7.0, scikit-learn 1.9.0 và joblib 1.6.0. Ghi lại phiên bản Streamlit và dependencies đã kiểm chứng khi giao app. Dataset và master split dùng để sinh mẫu demo hoặc tái lập báo cáo.

## 3. Hợp đồng đầu vào và UI

1. Cho chọn task và thuật toán, upload CSV, xem bảng kết quả và chọn một dòng để xem chi tiết. Hiển thị model đang chạy.
2. CSV có header, ít nhất một dòng và đúng 55 features từ `pipeline.feature_names_in_`. Kiểm tra tên cột trùng trước khi pandas đổi tên. Báo rõ cột thiếu, dư hoặc trùng, giá trị phi số và infinity. Sắp lại cột theo thứ tự model và giữ ô trống dưới dạng NaN để pipeline impute.
3. Gọi `pipeline.predict_proba(X)` trực tiếp trên features đã xác thực. Fitted pipeline xử lý median imputation, IQR detection giữ nguyên giá trị và StandardScaler trước classifier.
4. Lấy lớp ứng viên theo `classes_` và điểm cao nhất `p_max`. So sánh điểm gốc với τ của model đang chạy. Nếu `p_max >= tau`, trả lớp ứng viên. Nếu thấp hơn, trả `Uncertain` và vẫn hiển thị lớp ứng viên.
5. Hiển thị kết quả, `p_max`, τ và biểu đồ điểm dự đoán từng lớp. Gọi các giá trị này là điểm dự đoán của model.

## 4. Local Top-3 cho LightGBM

Gọi `pipeline.named_steps["preprocessing"].transform(row)` rồi dùng `pipeline.named_steps["classifier"].booster_.predict(..., pred_contrib=True)`.

| Task | Dạng kết quả | Lớp giải thích |
|---|---|---|
| Binary | 55 đóng góp và một baseline | Margin gốc thuộc lớp dương Malware. Với Benign, đảo dấu đóng góp và baseline. |
| Multiclass | Bốn khối, mỗi khối có 55 đóng góp và một baseline | Chọn khối của lớp ứng viên theo `classes_`. Phần tử cuối khối là baseline. |

Xếp hạng theo trị tuyệt đối của đóng góp sau khi bỏ baseline. Hiển thị tên feature, giá trị gốc và đóng góp có dấu theo đơn vị raw score. Với `Uncertain`, ghi rõ đang giải thích lớp ứng viên. Kiểm tra tổng đóng góp cộng baseline khớp raw margin của lớp được giải thích. Với Random Forest, hiển thị trạng thái giải thích cục bộ chưa hỗ trợ.

## 5. Mẫu demo và nghiệm thu

Chọn mẫu từ validation của master split hiện tại. Tạo CSV chỉ gồm 55 features và lưu kỳ vọng theo từng cấu hình. Với mỗi cấu hình, đối chiếu bảng kết quả với `predict_proba` của artifact. Dùng mẫu validation dưới ngưỡng để demo `Uncertain` và kiểm tra riêng trường hợp `p_max == tau`.

| Model | Accuracy toàn test | Macro-F1 | Coverage | Accuracy trên mẫu được nhận | Uncertain |
|---|---:|---:|---:|---:|---:|
| Binary LightGBM | 100,00% | 1,0000 | 100,00% | 100,00% | 0 |
| Binary Random Forest | 99,95% | 0,9995 | 99,53% | 100,00% | 41 |
| Multiclass LightGBM | 84,42% | 0,7655 | 72,43% | 95,52% | 2.424 |
| Multiclass Random Forest | 83,90% | 0,7588 | 71,30% | 95,21% | 2.523 |

Các chỉ số trên lấy từ `model/reports/model_report.json` cho 8.791 dòng test. Khi trình bày accuracy trên mẫu được nhận, hiển thị kèm coverage.

- Kiểm tra mặc định LightGBM ở hai task, chuyển được sang Random Forest và ngưỡng cập nhật theo model.
- Kiểm tra cột đảo thứ tự, CSV thiếu, dư, trùng cột, rỗng, có giá trị phi số hoặc infinity.
- Kiểm tra điểm theo lớp, trạng thái `Uncertain` và Local Top-3 của hai model LightGBM.
- Giao mã app, dependency Streamlit, hướng dẫn chạy từ project root, mẫu CSV hoặc cách sinh mẫu và kết quả smoke test.
