# Task A — kết quả baseline thực chạy

Ngày 06/11/2026. Nguồn: Final term.pdf Topic05 B/D; configs/task_a_baselines.json; reports/04_split_protocol.md.

| Candidate | Run | Validation accuracy | Validation Macro-F1 | Train seconds | Reload process |
|---|---|---:|---:|---:|---|
| random_forest | 20261106T094717Z_9114ca94 | 0.999914675768 | 0.999914675767 | 1.304252 | True |
| lightgbm | 20261106T094725Z_697a8770 | 1.000000000000 | 1.000000000000 | 0.521153 | True |

Bảng được sinh từ validation_metrics.json, không nhập tay scores. Hai notebook mới có 6 code cells/notebook, execution counts đầy đủ, không error output. Jupyter kernel mới dùng .venv Python; LightGBM 4.7.0. Lần thử nbclient trong sandbox không khởi tạo được kernel và đã dừng; lần chạy có quyền mở loopback đã PASS cho cả hai notebook.

SYNC RESULT — PHASE 4 (Task A baseline candidates): train chỉ trên train; validation metrics/CM/per-class report/predictions/coverage/global importance có thật; candidate_pipeline.joblib và checksums trong từng run. Reload 32 validation vectors ở tiến trình mới probability parity PASS. Bốn unittest preprocessing chạy lại trong .venv PASS.

R02/R03/R05 cho phần binary: VERIFIED về chạy Pipeline/RF/LightGBM. R07: PARTIAL, có curves nhưng chưa chọn tau. R09: PARTIAL, có candidate models/reload; chưa final model/test datasets. R08/R11 và local XAI không thuộc bước này, chưa triển khai. Không tuyên bố Phase4 toàn dự án hoàn thành: multiclass chưa build.

selected_model=null; selected_tau=null; test_evaluated=false. LightGBM validation hoàn hảo không chứng minh zero-day/deployment. Còn caveats token/specimen, benign host/session, near-duplicates, collection artifacts và provenance. Không tuning theo test. Bước tiếp: review validation errors/global importance và chốt chính sách vận hành trước chọn model/tau; final test thuộc Phase6.

Để chạy: chọn interpreter .venv/Scripts/python.exe trong notebook UI, Restart Kernel → Run All. Notebook chạy được từ root hoặc notebooks/. Mỗi Run All tạo run ID mới và giữ kết quả cũ. Kernel name task-a trong output là kernel tạm phục vụ kiểm chứng; người dùng chọn .venv interpreter.
