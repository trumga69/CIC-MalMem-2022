# Bàn giao TV5 — Web demo CIC-MalMem-2022

Ngày 10/11/2026. Phạm vi là web demo chạy cục bộ cho Topic 05, mục B/D/E, trang 16–17 của Final term.pdf.

## 1. Hiện trạng

Đã huấn luyện RF và LightGBM cho cả hai task. LightGBM đứng đầu theo validation Macro-F1 ở mỗi task. Hai Pipeline đã khóa và test cuối đã được kiểm chứng với ngưỡng chung 0,65.

| Task | Nhãn theo thứ tự model | Run LightGBM đã khóa | Validation Macro-F1 |
|---|---|---|---:|
| A / binary | Benign, Malware | 20261110T020241Z_88a86804 | 1,000000 |
| B / multiclass | Benign, Ransomware, Spyware, Trojan | 20261110T020302Z_b7e98446 | 0,767168 |

A và B là hai chế độ inference riêng. Đầu vào là CSV chứa các đặc trưng bộ nhớ đã trích xuất.

## 2. Files và môi trường

| File | Mục đích |
|---|---|
| configs/operating_threshold.json | Nguồn quyết định. Đọc tau, rejection_label và selected_models.{binary,multiclass}. Mỗi mục chứa pipeline_path, pipeline_sha256, run_id và classes. |
| reports/evaluation/binary/lightgbm/20261110T020241Z_88a86804/candidate_pipeline.joblib | Pipeline A đã fit. |
| reports/evaluation/multiclass/lightgbm/20261110T020302Z_b7e98446/candidate_pipeline.joblib | Pipeline B đã fit. |
| preprocessing.py | Chứa IQRDetector mà joblib cần import khi nạp model. |
| requirements-task-a.txt | Môi trường model đã kiểm chứng. Bổ sung dependency Streamlit cho web. |
| reports/final_test/ | Metrics và predictions của test cuối để đưa vào báo cáo. |

Resolve pipeline_path từ thư mục gốc project. Xác minh SHA-256 trước khi nạp joblib từ workspace. Tên candidate_pipeline.joblib là tên file lúc train. Chọn chính xác theo hồ sơ khóa. Python đã dùng là 3.11.9, LightGBM 4.7.0, scikit-learn 1.9.0, joblib 1.6.0. Giữ các phiên bản model tương thích và ghi lại phiên bản Streamlit đã kiểm chứng. App chỉ cần config, hai model, preprocessing.py và dependencies khi inference. Dataset và master split cần cho việc sinh mẫu demo hoặc tái lập báo cáo.

Hồ sơ hiện còn final_test_status=pending và hai notebook còn RUN_FINAL_TEST=True dù kết quả đã kiểm chứng. Đây là việc đồng bộ hồ sơ của nhóm. TV5 không cần chạy lại test khi triển khai web.

## 3. Hợp đồng đầu vào và UI

1. Cho chọn Task A hoặc Task B, upload CSV một hoặc nhiều dòng, hiển thị bảng kết quả và cho chọn dòng để xem chi tiết.
2. CSV phải có header, có ít nhất một dòng và đúng bộ **55 features**. Lấy tên và thứ tự từ pipeline.feature_names_in_. Kiểm tra tên cột trùng trước khi pandas có thể đổi tên. Báo rõ cột thiếu, dư hoặc trùng. Cho phép cột đảo thứ tự và sắp xếp lại theo model. Class, Category, Filename, row_index và group_id không thuộc schema. Báo lỗi cho chữ rác hoặc infinity. Có thể giữ ô trống dưới dạng NaN để Pipeline impute.
3. Gọi pipeline.predict_proba(X) trực tiếp trên features đã xác thực. Pipeline có median imputation, IQR detection giữ nguyên giá trị, StandardScaler và classifier LightGBM. Dùng fitted Pipeline, không fit hoặc tiền xử lý thủ công trên upload.
4. Ánh xạ điểm theo pipeline.classes_ và đối chiếu config. Lấy argmax và p_max. Nếu p_max >= tau, trả nhãn argmax. Nếu p_max < tau, trả Uncertain. So sánh trước khi làm tròn để hiển thị. Tại đúng 0,65, mẫu được nhận.
5. Hiển thị nhãn hoặc Uncertain, p_max, tau và bar chart cho tất cả lớp. Với Uncertain, ghi rõ lớp điểm cao nhất là lớp ứng viên. Dùng cách gọi điểm dự đoán của model. Chưa có bằng chứng các scores được hiệu chỉnh để diễn giải là xác suất dự đoán đúng.

## 4. Local Top-3

Dùng LightGBM booster_.predict(..., pred_contrib=True) cho dòng đang xem. Gọi pipeline.named_steps['preprocessing'].transform(row) rồi đưa kết quả vào pipeline.named_steps['classifier'].booster_. Hai Pipeline giữ 55 vị trí feature. Đã kiểm tra trên mẫu validation rằng tổng đóng góp và baseline khớp raw_score.

| Model | Dạng dense đã kiểm tra | Lớp giải thích |
|---|---|---|
| Binary | 56 giá trị mỗi dòng, gồm 55 đóng góp và baseline | Dạng gốc là margin lớp dương Malware. Với Benign, đảo dấu cả đóng góp và baseline để giải thích margin đối nghịch. |
| Multiclass | 224 giá trị mỗi dòng, gồm 4 khối × 56 | Reshape thành 4 × 56. Chọn khối lớp argmax theo classes_. Phần tử cuối khối là baseline. |

Loại baseline khỏi xếp hạng. Chọn ba feature có trị tuyệt đối lớn nhất. Hiển thị tên, giá trị gốc và đóng góp có dấu. Ghi rõ đơn vị là đóng góp vào raw score của lớp đang giải thích. Với Uncertain, đặt tiêu đề là giải thích lớp ứng viên. Nếu tính giải thích lỗi, báo lỗi riêng. CSV global importance chỉ phục vụ báo cáo tổng thể.

Tham khảo [LightGBM Booster.predict](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.Booster.html#lightgbm.Booster.predict).

## 5. Mẫu demo đã xác nhận

| Task | Row ID validation | Lớp điểm cao nhất | p_max xấp xỉ | Kết quả tại tau 0,65 |
|---|---:|---|---:|---|
| A | 14627 | Benign | 0,999990 | Benign |
| B | 11611 | Benign | 0,999997 | Benign |
| B | 43302 | Spyware | 0,338679 | Uncertain |

Chạy từ project root, bằng Python trong .venv, để sinh CSV có đúng ba dòng trên:

```python
from preprocessing import load_splits

X, _ = load_splits("multiclass")["validation"]
X.loc[[14627, 11611, 43302]].to_csv("demo_validation_features.csv", index=False)
```

Hai task có cùng schema nên dùng chung CSV này. Để demo rejection, dùng dòng 3 với Task B. LightGBM Task A không có Uncertain tại 0,65 trên validation và test đã đánh giá.

## 6. Test cuối đã kiểm chứng

Mỗi task có 11.720 mẫu test. Các chỉ số tính trên toàn tập được trình bày riêng với kết quả sau khi áp dụng tau.

| Task | Accuracy toàn tập | Macro-F1 | Coverage | Accuracy trên mẫu được nhận | Uncertain |
|---|---:|---:|---:|---:|---:|
| A | 100,00% | 1,0000 | 100,00% | 100,00% | 0 |
| B | 85,30% | 0,7786 | 78,82% | 93,78% | 2.482 |

Coverage theo lớp thật của B: Benign 100%, Ransomware 51,79%, Spyware 65,40%, Trojan 55,51%. Trình bày accuracy trên mẫu được nhận cùng coverage. Kết luận giới hạn trong benchmark. Việc nhóm theo features trùng và token chưa chứng minh độc lập theo specimen hay host. Nguồn là test_metrics.json và test_predictions.csv trong reports/final_test/ của từng run khóa.

## 7. Nghiệm thu và bàn giao

- Giao app.py, dependency Streamlit đã kiểm chứng và hướng dẫn chạy trong README. Từ project root dùng `.\.venv\Scripts\python.exe -m streamlit run app.py`.
- Có CSV demo hoặc lệnh sinh mẫu, thao tác được cả hai task, xem scores và Top-3 đúng dòng, thấy Uncertain ở dòng 3 Task B.
- Cột đảo thứ tự vẫn ra cùng kết quả. CSV thiếu, dư, trùng cột, rỗng, có chữ rác hoặc infinity hiện lỗi rõ. Logic tại p_max bằng 0,65 nhận mẫu.
- Xác minh tổng local contributions cộng baseline khớp raw margin của lớp giải thích. Dùng mẫu validation để nghiệm thu. Giữ nguyên notebooks train, master split, model, tau và kết quả test.
- Gửi lại danh sách file đã tạo/sửa, lệnh cài và chạy, kết quả kiểm chứng và mục còn thiếu.

