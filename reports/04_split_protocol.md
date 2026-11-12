# PHASE 3 — Task A baseline protocol (06/11/2026)

Nguồn mục tiêu: Final term.pdf Topic05 B/D; reports/01_objective_contract.md R02/R03/R05/R07/R09/R15. Người dùng yêu cầu trước mắt hai file train Task A. Freeze cấu hình trong configs/task_a_baselines.json trước các lần chạy mới; không thay đổi master split hoặc preprocessing.

Phase 1 SYNC: đã đọc source; Python 3.11.9 có pandas/sklearn/nbformat/nbclient/ipykernel/matplotlib; LightGBM thiếu ở Python gốc, tạo .venv --system-site-packages và thử cài riêng. Hiện trạng cũ không có notebook/model. Không giả rằng đã phục hồi notebook cũ: đây là hai notebook mới.

Phase 2 SYNC: dùng lại bằng chứng reports/objective_audit/observations.json: nhãn binary Class khớp mapping Category, 55 numeric predictors không chứa nhãn; không missing/inf; 0 exact/token/connected group overlap. Giữ nguyên xung đột multiclass, không xóa dữ liệu. Gate binary baseline: GO có giới hạn benchmark; taxonomy family không phải target của nhiệm vụ. Audit domain/provenance đầy đủ vẫn PARTIAL.

Phase 3 SYNC: master split seed42, SGKF5 connected groups; folds0=test,1=validation,2–4=train; train 35.156, validation 11.720, test 11.720. Binary labels giữ Benign/Malware, mọi dòng hợp lệ. Không gọi specimen-independent. Chấp nhận caveats bằng ghi nhận rõ theo skill §3 gate; không tuyên bố đã giải quyết near-duplicates/collection shortcuts. Pipeline median → IQR detection (không sửa/xóa giá trị) → StandardScaler → estimator, chỉ fit train.

Primary: validation macro-F1; tie-break macro-recall giảm dần, kích thước serialized model tăng dần, tên algorithm alphabet. Chọn kích thước thay runtime vì runtime nhiễu theo tải CPU; ghi nhận khác đề xuất Phase0 trước lần chạy mới. Hai fixed baseline configs, không search, không early stopping; cùng seed, cùng data/schema/preprocessing, tối đa 2 worker. 100 cây ở hai model không có nghĩa capacity ngang nhau. Không SMOTE vì binary cân bằng. Không calibration hoặc local XAI ở bước train; global importance không thay local Top-3.

Tau chỉ khảo sát grid cố định; chưa có chính sách vận hành nên selected_tau=null. Không tự đặt mức accepted accuracy tối thiểu. selected_model=null; không tự tuyên bố winner. Notebook không dự đoán/đánh giá classifier trên test, không có cell test có thể vô tình chạy. Final TEST thuộc Phase6 sau quyết định model/config/tau riêng.

Mỗi lần chạy lưu thư mục run ID mới, model candidate Pipeline, metadata/hashes/config/versions, validation predictions/metrics/CM/importance/coverage và figures thực. Reload model trong tiến trình mới, so probability parity trên validation. Không ghi đè experiment cũ. Giới hạn: lịch sử test exposure cũ không xác minh; score benchmark không chứng minh zero-day/deployment.
