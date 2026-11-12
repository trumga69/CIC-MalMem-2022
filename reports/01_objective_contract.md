# PHASE 0 — Objective Audit / Requirement-to-Evidence Matrix

Ngày audit: 06/11/2026. Gate: **LIMITED GO** — đủ hiểu phạm vi; tiếp tục audit/environment/taxonomy, chưa thiết kế/chạy model mới. Không đánh giá classifier bằng test trong phiên này.

## 1. Source of truth và phạm vi

- Đã đọc General Regulations and Policies trang 1–2 và toàn bộ Topic 05 từ trang 16 (bắt đầu sau Topic 04) đến phần References đầu trang 18 trong `Final term.pdf`. PDF có 28 trang; SHA-256 lưu ở `objective_audit/observations.json`. Không tìm thấy đề cương phiên bản khác hoặc rubric chấm điểm có trọng số. Mục E là tiêu chí đánh giá/bảo vệ hiện có, không suy đoán điểm số.
- Bộ quy tắc người dùng gọi BUILD_MODEL_SKILL nằm tại `AI_SKILL/build-model-malware-cic/SKILL.md` (heading BUILD_MODEL_SKILL); không có file riêng tên BUILD_MODEL_SKILL.md. Áp dụng cùng RULES.md và AI_SKILL.md trong repository.
- Điều chỉnh giảng viên được RULES.md mục 4 ghi nhận ngày 05/11/2026: train/evaluate tích hợp trong bốn notebook; không yêu cầu hai script riêng. Đây là điều chỉnh được ghi nhận, không phải nguyên văn PDF.
- Input: ma trận feature memory forensics đã trích xuất. Output: binary Benign/Malicious (nhãn local Malware) và multiclass Benign/Ransomware/Spyware/Trojan, scores theo lớp, Uncertain, local Top-3. Cả hai task dùng toàn bộ population hợp lệ, không có malware-only routing.
- Tiêu đề có Family nhưng A/B định nghĩa rõ multiclass 4 category; **không tự thêm 15-family/16-class, hierarchical hoặc DL**. Không xử lý raw memory/binary.

## 2. Requirement-to-Evidence Matrix

FOUND = yêu cầu rõ; AMBIGUOUS = chi tiết chưa xác định; MISSING = thiếu bằng chứng/đầu ra. Cột thực hiện tách khỏi trạng thái yêu cầu.

| ID | Yêu cầu / nguồn | Input → output | Implementation dự kiến/hiện có | Evidence / acceptance | Trạng thái yêu cầu; thực hiện |
|---|---|---|---|---|---|
| R01 | 16 B: audit phân phối, identifier, balance, dataset | CSV → profile | audit.py; kiểm tra độc lập | observations.json; audit CSV; provenance phải rõ | FOUND; PARTIAL: raw chưa đối chiếu UNB, 55 thay vì 56 features |
| R02 | 16–17 B: Pipeline missing/outlier/scaling | X → X transformed | make_preprocessor: median → IQRDetector → StandardScaler | 4 tests PASS; train-only statistics; không xóa dòng | FOUND; VERIFIED cho module, chưa tích hợp classifier hiện tại |
| R03 | 17 B Task 1 binary | 55 features → Benign/Malware | prepare_data/load_splits('binary'); notebook Binary-RF dự kiến | class mapping; predictions đúng index; notebook Run All | FOUND; PARTIAL: có predictions cũ, không có notebook/model |
| R04 | 17 B Task 2 multiclass | 55 features → 4 category gồm Benign | prepare_data/load_splits('multiclass') | đủ lớp/split; per-class metrics | FOUND; labels/split có, model MISSING |
| R05 | 17 B: RF và LightGBM/Gradient Boosting | mỗi task → candidates | 4 notebooks RF/LightGBM theo RULES | cùng split, tuning budget, validation; lưu log/config | FOUND; 4 notebook MISSING, lightgbm chưa có; không tự thay algorithm |
| R06 | 17 B: max probability < tau → Uncertain | scores → accepted/rejected | triage module dự kiến | p_max==tau nhận; class alignment; tau invalid; zero coverage=N/A | FOUND; runtime module MISSING; kết quả khảo sát cũ chưa chọn tau |
| R07 | 17 B: Coverage vs Accuracy | validation probabilities → curve | so sánh tau trên validation | lưu counts/coverage/accepted accuracy; không ép đường cong tăng | FOUND; có bảng Binary-RF cũ, thiếu các candidates và quy tắc chọn tau |
| R08 | 17 B/D: Streamlit/FastAPI upload | vector → category/scores/Top-3 | app.py + explanation dự kiến | schema errors, local attribution đúng sample/class, bar chart/flag | FOUND; MISSING |
| R09 | 17 D: model và test split datasets | fitted Pipeline → bundle + held-out data | joblib + features/classes/tau metadata | reload tiến trình mới, prediction parity; xuất/dẫn xuất test tái lập | FOUND; model MISSING; có manifest, chưa có gói test bàn giao |
| R10 | 17 D: technical report CM, per-class P/R/F1, importance | saved outputs → report | technical_report dự kiến | bảng truy được artifact; global khác local | FOUND; PARTIAL: chỉ outputs Binary-RF validation, chưa có technical report |
| R11 | 17 E: live inference, Uncertain, giải thích Trojan | sample → demo + oral defense | demo dự kiến | mẫu thực, không bịa contributions; giải thích malfind/handles/ldrmodules | FOUND; MISSING |
| R12 | 1–2: Git, branch/PR/review, commit cá nhân | source → hosted repo/history | workflow nhóm | repository URL, PR/review thực, không giả contribution | FOUND; MISSING trong workspace: không có .git |
| R13 | 2: README prerequisites/install/run/tests/architecture/tree/tasks/refs | project → reproducible handoff | README có một phần | clean install; dependency spec; phân công thật | FOUND; PARTIAL, thiếu requirements và trạng thái notebook sai |
| R14 | 16 A: hiểu Macro-F1; skill §4: freeze selection | validation → selected model | chưa chốt primary/tie-break | đăng ký trước comparison | AMBIGUOUS: PDF không ấn định primary/tie-break; cần freeze Phase 3 |
| R15 | Skill §2/4/9: no leakage, master split, train-only fit, test cuối | raw → partitions → final evaluation | connected groups + SGKF5 seed42 đã có | 0 group/exact/token overlap; test chỉ sau freeze | FOUND; PARTIAL: kiểm tra trực tiếp PASS, specimen identity/near-duplicates chưa chứng minh |

## 3. Objective contract còn cần freeze

Câu hỏi nghiên cứu: (1) hai kiến trúc đáp ứng binary và 4-category ra sao; (2) rejection thay đổi coverage/accepted accuracy thế nào; (3) giải thích dự đoán từng mẫu có đúng model/class và phục vụ analyst không. Đây là cách diễn đạt mục tiêu A/B/E, không thêm mục tiêu khoa học.

REQUIRED: Pipeline, hai task, ít nhất RF và LightGBM/Gradient Boosting, rejection, coverage benchmark, GUI, local Top-3, artifacts/test datasets, report, Git và hiểu để bảo vệ. IMPLEMENTATION CHOICE: split 60/20/20 gần đúng, connected grouping, median/IQR/StandardScaler, SHAP nếu chọn, dùng validation Macro-F1. OPTIONAL: calibration nâng cao/CI nếu có lý do. OUT OF SCOPE: family-only task, DL/thuật toán thứ ba, raw dumps, database/cloud, yêu cầu riêng Topic 04.

Primary metric chưa chính thức chốt. Đề xuất để xét ở Phase 3: validation macro-F1 cho từng task; tie-break macro-recall rồi chi phí đo thực, cuối cùng thứ tự cố định. Đây chưa phải quyết định model. Tau 0.65 trong đề chỉ là ví dụ; cần định nghĩa trade-off coverage/accuracy trước chọn tau, không tự bịa mức accuracy tối thiểu. Không chọn bằng test. Ngân sách CPU/RAM/thời gian thực, tuning budget và dependencies chưa kiểm kê đủ; con số runtime trong PDF là mô tả feasibility, không phải kết quả local.

## 4. Audit source, bằng chứng và leakage

Đã đọc toàn bộ audit.py, preprocessing.py và tests/test_preprocessing.py. audit.py thực thi ngay ở top level và ghi đè outputs audit cố định khi chạy/import; phiên này không chạy nó để bảo toàn bằng chứng cũ. Đã dùng script audit độc lập lưu cạnh kết quả. preprocessing.py không fit khi import; loại Category/Class, kiểm tra schema numeric/nhãn; connected components từ exact features và token; lưu/xác thực checksum split. Pipeline chưa fit chỉ học thống kê khi caller fit. Chưa có classifier/app source nên chưa thể xác nhận train-only fit toàn luồng.

- P1: Thiếu bốn notebook và mọi model .joblib/.pkl dù README ghi có Binary-RF. Có CSV/JSON predictions validation; checksum raw/split/preprocessing khớp; tái tính accuracy/macro-F1 từ predictions khớp JSON, row IDs đúng validation và labels đúng raw. Chứng minh nhất quán artifact, **không chứng minh training tái lập**, không xác nhận runtime cũ. selected_model/selected_tau null, test_evaluated false là khai báo metadata, không đủ chứng minh lịch sử chưa từng xem test.
- P1: 569 dòng dư exact-feature duplicate, 1.086 dòng thuộc nhóm trùng; 534 dòng dư trùng toàn dòng. Có 9 nhóm xung đột multiclass, 0 binary. Không tự xóa/sửa nhãn. Exact duplicates/group/token overlap giữa các split đều 0 theo kiểm tra trực tiếp.
- P1: Category chứa category/family/token/dump suffix; tuyệt đối chỉ target/group metadata. Token không được chứng minh là specimen ID thật; Benign không có host/session; near-duplicate và collection shortcuts chưa loại trừ. Không gọi split specimen-independent.
- P1: Ba Output CSV có Filename, thiếu Category/Class; README tuyên bố 850 mẫu đã nằm trong bản đầy đủ nhưng chưa đủ bằng chứng. Kiểm tra hash numeric exact shared features không tìm thấy match; cần kiểm tra convention/precision và provenance trước kết luận chúng độc lập hay trùng. Không nối hoặc dùng làm external test.
- P2: Full CSV 55 numeric features, không phải 56 như PDF; source README nói mirror GitHub, không có chứng nhận byte identity UNB. Không thêm feature để ép khớp đề.
- P2: 3 features hằng/giống nhau; có near-constant profile trong JSON. Không có missing/inf/negative. Giá trị không âm không chứng minh hợp lệ về ngữ nghĩa; cần domain checks cho ratios/counts và acquisition artifacts.
- P2: LightGBM chưa cài, chưa có requirements; không thay thuật toán hoặc cài package trong Phase 0.

Không huấn luyện, chọn model/tau hoặc tính classifier performance trên test. Có đọc toàn raw và class counts/split để audit cấu trúc; tuyệt đối không dùng thống kê toàn bộ này để fit preprocessing/model.

## 5. LEARN → PLAN → CODE → TEST → REVIEW

LEARN: đọc PDF/rules/source để tránh tiêu đề dẫn tới sai target. PLAN: map mỗi requirement tới acceptance trước model. CODE: chỉ script kiểm tra bằng chứng, không sửa training/preprocessing/raw/split. TEST: chạy 4 unittest hiện có và đối chiếu artifact; logs trong objective_audit. REVIEW: gate LIMITED GO do còn provenance, taxonomy/domain checks, reproducibility và selection contract. Không coi có code là đã hoàn thành GUI/model.

Lệnh tái chạy từ root PowerShell:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python reports/objective_audit/inspect_existing.py
python -m unittest discover -s tests -p test_preprocessing.py -v
```

## 6. Giải thích để bảo vệ

Tóm tắt 30–60 giây: Em bắt đầu từ yêu cầu có thể kiểm chứng trong đề: binary và bốn category gồm Benign. Em tách nhãn/identifier khỏi features để tránh model đọc đáp án. Em giữ các mẫu có cùng features hoặc token ở cùng split để giảm việc học thuộc. Kiểm tra hiện tại chưa chứng minh độc lập specimen/host, nên em giới hạn kết luận trong benchmark. Model và tau chỉ được chọn từ validation; test dành cho đánh giá sau khi cấu hình đã khóa. Artifacts cũ có thể kiểm tra tính nhất quán nhưng còn thiếu notebook để tái lập.

- Dễ — Vì sao không làm 15-family? A/B của Topic 05 yêu cầu binary và 4-category; title không thay hợp đồng I/O cụ thể.
- Vừa — Vì sao không random row split? Các dump/mẫu trùng có thể rơi cả train và test, làm metric phản ánh ghi nhớ; connected grouping giữ cả quan hệ bắc cầu.
- Khó — Zero overlap đã đủ chứng minh độc lập chưa? Chưa: token chưa xác minh nguồn, Benign thiếu host/session, near-duplicates/collection artifacts chưa kiểm hết.

## 7. SYNC RESULT — PHASE 0

Objective references: Final term.pdf p1–2, Topic05 p16–18; RULES.md §4; build-model skill §1/8/9.
Verified work: scope/matrix, đọc source; raw/split/provenance checks; 4 tests PASS; xác nhận thiếu notebooks/model và LightGBM.
Artifacts: reports/01_objective_contract.md; reports/02_data_audit.md; reports/objective_audit/{inspect_existing.py,observations.json,preprocessing_tests.txt}.
Risks/blockers: thiếu reproducible training source; grouping chưa specimen-verified; schema/provenance mismatch; metric/tau protocol chưa freeze.
Next phase: Phase 1 kiểm kê môi trường/dependencies và phục hồi nguồn notebook có provenance; Phase 2 hoàn thiện taxonomy/domain/provenance trước Phase 3 freeze. Chưa chuyển sang build.
