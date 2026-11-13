# Prompt giao việc cho TV5

Triển khai web demo Streamlit chạy cục bộ cho Topic 05 — CIC-MalMem-2022. Đọc `app/handoff_tv5_web_demo.md` và `model/configs/inference.json` từ project root. Dùng bốn pipeline `.joblib` đã xuất và kiểm chứng trong `model/artifacts/`. Báo cáo chung `model/reports/model_report.json` dùng để đối chiếu kết quả.

1. Cho chọn Task A (Benign/Malware) hoặc Task B (Benign/Ransomware/Spyware/Trojan), rồi chọn thuật toán LightGBM hoặc Random Forest. LightGBM là model chính và mặc định ở cả hai task. Hỗ trợ đủ bốn pipeline theo config và hiển thị model đang dùng.
2. Cho upload CSV một hoặc nhiều dòng, xem bảng kết quả và chọn dòng để xem chi tiết. Lấy schema 55 features từ `pipeline.feature_names_in_`. Kiểm tra bảng rỗng, tên cột thiếu, dư hoặc trùng, giá trị phi số và infinity. Sắp xếp cột hợp lệ theo model và cho phép ô trống để pipeline impute.
3. Nạp artifact nội bộ theo config, resolve đường dẫn từ project root và kiểm tra tập nhãn theo task. Gọi `predict_proba` trên fitted pipeline. Hiển thị nhãn hoặc `Uncertain`, `p_max`, `tau` và biểu đồ điểm theo `classes_`. Áp dụng `p_max >= tau` trước khi làm tròn. Với `Uncertain`, hiển thị lớp ứng viên.
4. Hiển thị local Top-3 cho hai cấu hình LightGBM theo mục 4 của tài liệu bàn giao. Với Random Forest, hiển thị trạng thái giải thích cục bộ chưa hỗ trợ.
5. Giao mã app, dependencies và hướng dẫn chạy. Smoke test đủ bốn cấu hình bằng CSV validation, kiểm tra chuyển thuật toán, trường hợp `Uncertain`, ngưỡng biên, CSV lỗi, cột đảo thứ tự và độ khớp raw score của giải thích LightGBM.

Phạm vi inference là CSV đặc trưng bộ nhớ đã trích xuất. Báo lại danh sách file, lệnh chạy, kết quả kiểm chứng và phần còn thiếu.
