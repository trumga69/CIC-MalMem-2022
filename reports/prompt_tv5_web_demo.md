# Prompt giao việc cho TV5

Bạn phụ trách web demo Topic 05 — CIC-MalMem-2022. Đọc reports/handoff_tv5_web_demo.md, configs/operating_threshold.json và phần GUI/local explanation trong AI_SKILL/RULES.md. Triển khai Streamlit chạy local trong repository hiện tại.

1. Viết app.py cho phép chọn Task A (Benign/Malware) hoặc Task B (Benign/Ransomware/Spyware/Trojan), upload CSV 55 features, xem bảng kết quả và chọn một dòng để xem chi tiết. Nạp đúng hai Pipeline LightGBM theo selected_models trong config. Resolve đường dẫn từ project root, kiểm SHA-256 và classes.
2. Lấy schema từ pipeline.feature_names_in_. Kiểm tra cột thiếu, dư, trùng, bảng rỗng, chữ rác và infinity. Sắp xếp các cột hợp lệ theo thứ tự model. Có thể giữ ô trống cho imputer. Gọi pipeline.predict_proba trên CSV đã xác thực.
3. Hiển thị nhãn hoặc Uncertain, p_max, tau và biểu đồ scores từng lớp theo classes_. Đọc tau từ config. So sánh p_max >= tau trước khi làm tròn. Khi Uncertain, vẫn show scores và ghi lớp ứng viên.
4. Hiển thị local Top-3 cho dòng đang xem. Theo mục 4 báo cáo, dùng LightGBM pred_contrib sau fitted preprocessing, chọn đúng khối lớp ở multiclass hoặc dấu margin ở binary, loại baseline khỏi ranking. Xếp theo trị tuyệt đối, giữ dấu và ghi đơn vị raw score. Với Uncertain, giải thích lớp ứng viên.
5. Thêm dependency Streamlit đã kiểm chứng và hướng dẫn chạy từ root. Kiểm bằng validation row 14627 cho Task A, 11611 và 43302 cho Task B. Chứng minh app hiện Benign và Uncertain đúng mẫu, xử lý CSV lỗi, cột đảo thứ tự, ngưỡng biên và độ khớp raw score của giải thích.

Giữ nguyên Pipeline khóa, tau, master split, dữ liệu và kết quả test. Phạm vi là inference từ feature CSV và demo local. Báo lại danh sách file, lệnh chạy, kết quả kiểm chứng và phần còn thiếu.

