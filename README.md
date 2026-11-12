# Topic 05 — CIC-MalMem-2022

Workspace hiện tại: đã có audit, preprocessing dùng chung và master split được kiểm tra. Đã triển khai baseline Binary-RF và chạy các code cell trong tiến trình Python mới; chưa chốt model/tau, chưa test evaluation hoặc app.

## Chạy audit

Yêu cầu: Python 3, pandas, NumPy và scikit-learn. Các phiên bản môi trường chạy split được lưu trong `data/splits/metadata.json`. Chưa có `requirements.txt` để xác minh cài sạch.

```powershell
python audit.py
```

Script hiện tại đọc `Obfuscated-MalMem2022.csv` trong `data/raw/`, giữ nguyên raw và ghi kết quả vào `reports/audit/`. Lệnh chạy không phụ thuộc working directory khi dùng đường dẫn đầy đủ tới script. Tests cho preprocessing được mô tả bên dưới.

Khi có master split, audit kiểm tra thứ tự dòng trước khi đối chiếu mẫu vượt split. Phép thử từng feature dùng cây sâu 2, fit trên train và chấm trên validation. Accuracy cao là tín hiệu để xem xét thêm.

Luồng hiện có: CSV raw → audit.py → các bảng CSV tổng hợp, phân phối features và danh sách nhãn xung đột; CSV raw → preprocessing.py → master split dùng chung.

## Phạm vi đã chốt

Chỉ Topic 05 và quy định chung của `Final term.pdf`. Theo xác nhận người dùng ngày 05/11/2026, giảng viên đồng ý tích hợp train/evaluate vào bốn notebook đã chốt; không yêu cầu hai script riêng. Đã có notebook Binary-RF; ba notebook còn lại chưa triển khai.

## Tiến độ có bằng chứng

| Phase | Task | PDF Requirement | Status | Evidence |
|---|---|---|---|---|
| 02 | Audit CSV đầy đủ | Trang 16, Topic 05-B | COMPLETED | Source hiện tại chỉ đọc CSV đầy đủ; có `basic_audit_summary.csv`, `feature_distributions.csv`, `conflicting_feature_labels.csv`; phiên Split xác nhận lại shape/nhãn/group nhưng chưa chạy lại toàn bộ audit |
| 02 | So khớp bản GitHub | Lựa chọn triển khai để truy nguồn | IN PROGRESS | README cũ ghi VERIFIED nhưng `source_verification.json` không có trong workspace hiện tại; chưa xác minh lại remote ở nhiệm vụ Split |
| 02 | Xác minh byte identity với UNB master | Nguồn dataset chính thức, trang 16-B | IN PROGRESS | Chưa có bản đầy đủ hoặc checksum từ UNB để đối chiếu |
| 03 | Duplicate/related samples và master split | Trang 16–17, Topic 05-B; grouping là lựa chọn chống leakage | VERIFIED | `load_master_split()` qua lệnh bên dưới; `data/splits/master_split.csv`, `data/splits/metadata.json`; 58.596 dòng, 32.087 connected groups; không có nhóm/features trùng vượt split |
| 03 | Preprocessor chưa fit dùng chung | Trang 16–17, Topic 05-B | VERIFIED | `python -m unittest discover -s tests -p test_preprocessing.py -v`: 4 tests PASS; kiểm tra train thật (35.156 × 55) và transform validation/test hữu hạn, không mất dòng, không thay đổi thống kê train; chưa kiểm tra tích hợp classifier/notebook |

## Split & Preprocessing — STEP 03

Chạy từ project root:

```powershell
python -c "from preprocessing import load_master_split; import pandas as pd; X,b,m,t=load_master_split(); print(pd.crosstab(t['split'], m))"
python -m unittest discover -s tests -p test_preprocessing.py -v
```

`preprocessing.py` chỉ tạo split khi chưa tồn tại. Các lần sau xác thực và đọc lại; checksum CSV raw, checksum split, feature order và cấu hình được lưu trong metadata. Nếu dữ liệu/cấu hình hoặc split bị thay đổi, chương trình báo lỗi thay vì âm thầm chia lại. Không chỉnh sửa CSV raw, không nối ba Output vào bản đầy đủ.

Module không có khối `if __name__ == '__main__'` và không tự chạy tác vụ khi import. Chạy `python preprocessing.py` sẽ không in bảng hoặc tạo split; dùng lệnh trên hoặc gọi `load_splits()` trong notebook. Bản sửa ngày 06/11/2026 giữ cách rút gọn bằng `exists`, `zip` và dictionary comprehension của người dùng; validation chỉ gọi một lần, bỏ kiểm tra `len(table)` dư. Khôi phục IQRDetector trong Pipeline và ordered features/row count trong metadata. Chạy lại 4 tests PASS; checksum master split trước/sau giữ nguyên `88da6b733625a289fe244fb6018c85cbbb4034f1e8ef45d79dadf0cbd15797b6`.

Lựa chọn triển khai: seed 42, StratifiedGroupKFold 5 folds; fold 0 dành cho test, fold 1 validation, ba fold còn lại train (xấp xỉ 60/20/20). Đây không phải tỷ lệ bắt buộc trong PDF. Stratification theo bốn category giúp dùng cùng phân hoạch cho cả binary và multiclass; cân bằng gần đúng vì group không bị tách.

| Split | Benign | Ransomware | Spyware | Trojan | Tổng |
|---|---:|---:|---:|---:|---:|
| Train | 17580 | 5875 | 6009 | 5692 | 35156 |
| Validation | 5859 | 1958 | 2005 | 1898 | 11720 |
| Test | 5859 | 1958 | 2006 | 1897 | 11720 |

Nhóm là thành phần liên thông qua hai quan hệ: giống toàn bộ 55 features hoặc cùng token mẫu trong tên Category (bỏ category, family và hậu tố số `.raw`). Liên kết bắc cầu cũng được giữ cùng split. Token quan sát có độ dài 21/31/32/64; không gọi mọi token là hash đã xác minh. Giữ nguyên cả 9 nhóm xung đột multiclass, không tự sửa nhãn hoặc xóa để tăng điểm. Category/Class không nằm trong X; binary giữ tên lớp nguồn `Benign`/`Malware` (Malicious trong đề).

Giới hạn: Benign chỉ có nhãn `Benign`, thiếu thông tin máy nguồn/phiên thu thập; không thể chứng minh độc lập theo host/session hoặc loại hết near-duplicates. Token là metadata quan sát, chưa chứng minh danh tính executable. Dataset thực có 55 features số, khác mô tả 56 trong PDF. Ba cột hằng được giữ nguyên để schema đơn giản và thống nhất.

API dùng trong bốn notebook sau này:

```python
from preprocessing import load_splits, make_preprocessor
from sklearn.pipeline import Pipeline

splits = load_splits('multiclass')  # hoặc 'binary'
X_train, y_train = splits['train']
preprocessor = make_preprocessor()  # chưa fit; ghép classifier tại notebook
# model = Pipeline([('preprocessing', preprocessor), ('classifier', classifier)])
# model.fit(X_train, y_train)
```

Chuỗi xử lý: median imputation → IQRDetector → StandardScaler. Mọi thống kê học khi `fit` train trong Pipeline. IQRDetector giữ nguyên giá trị và số dòng; `outlier_mask()` cung cấp cờ vượt Q1−1,5×IQR / Q3+1,5×IQR. IQR bằng 0 có thể tạo nhiều cờ khi giá trị khác quartile; cờ không chứng minh dữ liệu sai. Không clipping hoặc thêm cờ vào feature schema. Scaling hỗ trợ yêu cầu đề; RF/LightGBM thường ít cần scaling. Imputation vẫn có trong pipeline dù dữ liệu hiện tại không thiếu.

Kiểm chứng thực tế ngày 06/11/2026: fit preprocessor trên train; số dòng có ít nhất một cờ IQR là train 10.544, validation 3.611, test 3.459. Đây là chẩn đoán preprocessing, không phải metrics classifier và không dùng chọn cấu hình theo test. Output cả ba tập hữu hạn, giữ 55 features và đủ dòng; bounds IQR và mean scaler không đổi sau transform. Automated tests kiểm tra thêm imputation bằng dữ liệu có NaN, extreme values không bị xóa, split persistence/checksum và joblib reload trong tiến trình mới. Chưa có classifier/model artifact; file joblib trong test chỉ là tạm thời.

Để bảo vệ: giải thích vì sao chia từng dòng có thể leakage; vì sao group cần liên kết bắc cầu; vì sao IQR chỉ phát hiện chứ không xóa; và vì sao `make_preprocessor()` phải trả instance chưa fit. Validation dành chọn model/config/tau; test chỉ đánh giá classifier sau khi chốt.

## File hiện có

- `AI_SKILL/`: quy tắc và quy trình mentor.
- `Final term.pdf`: đề học phần.
- `audit.py`: audit có thể tái lập.
- `data/raw/`: bốn CSV, không chỉnh sửa.
- `reports/audit/`: sáu bảng CSV audit hiện có; không thấy `dataset_audit.md` hoặc các JSON được README cũ nhắc đến.

Cấu trúc workspace đích giữ theo `AI_SKILL/RULES.md`; chưa tạo các thành phần ngoài nhiệm vụ audit. Phân công thành viên và cấu hình Git chưa được cung cấp; không suy đoán đóng góp.

## Nguồn

- [CIC-MalMem-2022 — UNB](https://www.unb.ca/cic/datasets/malmem-2022.html).
- [Bản sao CSV do người dùng cung cấp](https://github.com/dcevallossalas/PrivacyMalwareClassifiers/blob/main/Obfuscated-MalMem2022.csv).

Nguồn tải của bản đầy đủ hiện là mirror GitHub; không tuyên bố đã xác minh giống từng byte với UNB master. Không nối thêm ba Output vào bản đầy đủ vì 850 mẫu đã có trong đó.


## Model Development — STEP 04.1 (06/11/2026)

| Phase | Task | PDF Requirement | Status | Evidence |
|---|---|---|---|---|
| 04 | Binary-RF baseline | Trang 17, Topic 05-B Model Development | IN PROGRESS | `notebooks/01_binary_random_forest.ipynb`; nbformat schema và syntax PASS; chạy tuần tự mọi code cell trong tiến trình Python mới PASS. Chưa xác minh Restart Kernel → Run All qua Jupyter: hai lần nbclient không trả kết quả và đã dừng. |
| 04 | Binary-LightGBM và hai notebook multiclass | Trang 17, Topic 05-B | NOT STARTED | Python 3.11.9 hiện chưa có package lightgbm; chưa cài hoặc thay thuật toán. |

Notebook Binary-RF: mở `notebooks/01_binary_random_forest.ipynb` từ project root hoặc `notebooks/`, chọn kernel có pandas/numpy/scikit-learn, Restart Kernel → Run All. Các cell gọi `load_splits('binary')`, tạo Pipeline mới và chỉ fit train. Chức năng train/evaluate tích hợp trong notebook theo điều chỉnh giảng viên đã chấp thuận; test evaluation chưa thực hiện vì chưa chốt model/tau.

Bằng chứng thực chạy trong tiến trình Python mới: train 35.156 dòng, validation 11.720 dòng, 55 features; RF 100 cây, seed 42, n_jobs=-1. Validation accuracy = 0.999914675767918; Macro-F1 = 0.9999146757672969. Confusion matrix (hàng thật/cột dự đoán; Benign, Malware) = [[5859, 0], [1, 5860]]. Một Malware bị dự đoán Benign; chưa phân tích nguyên nhân mẫu lỗi. Tau 0.65 nhận 11.713/11.720 mẫu (coverage 0.9994027303754266), accepted accuracy 1.0. Đây là khảo sát validation, chưa chọn tau và không phải hiệu quả trên test/ngoài phân phối. Các feature global đứng đầu: svcscan.nservices, svcscan.kernel_drivers, dlllist.avg_dlls_per_proc; không dùng chúng như local Top-3.

Kết quả thật nằm trong `reports/evaluation/binary/random_forest/`: `validation_metrics.json`, `validation_confusion_matrix.csv`, `validation_coverage_accuracy.csv`, `validation_feature_importance.csv`, `validation_predictions.csv`. JSON ghi tham số, phiên bản, thời gian train thực tế, ordered features/classes, checksums raw/split/preprocessing; selected_model/selected_tau = null và test_evaluated = false. Không lưu model artifact ở bước baseline này. Notebook kiểm tra xác suất hữu hạn/tổng bằng 1, thống kê preprocessing không đổi sau predict validation, nhận mẫu khi p_max == tau, coverage 0 trả None, tau ngoài [0,1]/NaN bị từ chối.

Chạy lại `python -m unittest discover -s tests -p test_preprocessing.py -v`: 4 tests PASS (9.080s). Workspace hiện chưa là Git repository (`git status` báo not a git repository); chưa đáp ứng workflow branch/PR của quy định chung. Chưa tự init/commit/push hoặc suy đoán đóng góp thành viên.

## PHASE 0 — Objective Audit (06/11/2026, kiểm tra hiện trạng)

**Bảng dưới đây ưu tiên hơn các mô tả tiến độ cũ phía trên.** Kiểm kê hiện tại không tìm thấy thư mục notebooks hoặc model .joblib/.pkl. Các đoạn nói notebook Binary-RF đã có/chạy là ghi nhận lịch sử chưa tái lập được từ source hiện tại. Không dùng chúng làm trạng thái VERIFIED.

| Phase | Task | PDF Requirement | Status | Evidence |
|---|---|---|---|---|
| 0 | Requirement-to-Evidence Matrix | General p1–2, Topic05 p16–18 | VERIFIED | reports/01_objective_contract.md; đọc trực tiếp PDF và source |
| 0 | Audit raw/split và kết quả cũ | Topic05 B; skill Objective Gate | VERIFIED | reports/objective_audit/observations.json; reports/02_data_audit.md; phạm vi sơ bộ, chưa hoàn tất Phase2 |
| 0 | Kiểm tra preprocessing hiện tại | Topic05 B Pipeline | VERIFIED | reports/objective_audit/preprocessing_tests.txt: 4 tests PASS |
| 1–3 | Provenance, taxonomy/domain audit và freeze selection | Topic05 B và skill gates | IN PROGRESS | còn thiếu xác minh token/specimen, primary/tie-break/tau policy và environment spec |
| 4 | Bốn notebook và model tái lập | Topic05 B/D; điều chỉnh train/evaluate trong notebook | BLOCKED | không có notebook/model trong workspace hiện tại; chỉ có Binary-RF validation outputs cũ |

Gate: LIMITED GO cho audit/environment; chưa build/train. Full CSV có 55 numeric features, không phải 56. Không xác nhận tuyên bố cũ “850 mẫu đã có trong full”: ba Output có Filename, thiếu Class/Category; exact shared-feature fingerprints không match, còn cần kiểm tra precision và provenance. Không nối dữ liệu. Kết quả validation cũ khớp raw/split/preprocessing checksums và tính lại từ predictions khớp metrics, nhưng thiếu training source để tái lập. Chưa chạy classifier test evaluation.

## Task A — hai notebook baseline (hiện trạng mới nhất 06/11/2026)

Phần này thay trạng thái thiếu notebook binary ở các ghi nhận trước. Đã tạo mới và kiểm chứng kernel mới:

- notebooks/01_binary_random_forest.ipynb
- notebooks/02_binary_lightgbm.ipynb

Chọn interpreter `.venv/Scripts/python.exe` trong VS Code/Jupyter rồi Restart Kernel → Run All. `.venv` hiện dùng system-site-packages, chỉ thêm LightGBM riêng; để cài sạch ở môi trường khác:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-task-a.txt
```

Không tự chạy pip trong notebook. Requirements này chỉ phục vụ Task A, chưa gồm GUI/XAI. Không chọn kernel Python gốc thiếu LightGBM. Hai notebook dùng preprocessing.py và configs/task_a_baselines.json chung; train/evaluate nằm trong notebook theo điều chỉnh được giảng viên chấp thuận.

| Phase | Task | PDF Requirement | Status | Evidence |
|---|---|---|---|---|
| 1–3 | Binary baseline protocol và environment | Topic05 B; skill gates | VERIFIED | reports/04_split_protocol.md; configs/task_a_baselines.json; LightGBM 4.7.0 trong .venv; caveats data còn ghi rõ |
| 4 | Binary RF + LightGBM | Topic05 B/D | VERIFIED | Cả hai notebook Run All kernel mới PASS; reports/05_experiment_results.md dẫn tới run ID và artifacts thực |
| 4 | Candidate Pipeline reload | Topic05 D | VERIFIED | Hai validation_metrics.json ghi reload_new_process_parity=true, so probability ở tiến trình Python mới |
| 5–6 | Chọn model/tau và final test | Topic05 B/D | NOT STARTED | selected_model/selected_tau=null; test_evaluated=false |
| 4 | Hai notebook multiclass | Topic05 B | NOT STARTED | Nhiệm vụ hiện tại chỉ Task A |

Mỗi run tạo thư mục mới dưới reports/evaluation/binary/{algorithm}/{run_id}/ chứa candidate Pipeline và reports; không ghi đè outputs cũ. Các candidate này chưa là model deploy. Global feature importance không thay local Top-3. Không có cell test evaluation trong hai notebook.

### Output mỗi run Task A từ 09/11/2026

Mỗi run thành công lưu 6 file: `candidate_pipeline.joblib`, `config.json`, `validation_metrics.json`, `checksums.json`, `validation_predictions.csv` và `validation_feature_importance.csv`. Metrics JSON chứa báo cáo từng lớp, confusion matrix kèm thứ tự lớp, Coverage–Accuracy theo tau và kết quả reload. `accepted_accuracy` là `null` khi coverage bằng 0. Hai biểu đồ hiển thị trong notebook. Các run cũ vẫn có bộ file theo định dạng lúc chạy.

SYNC RESULT — Cả hai notebook Run All từ kernel mới PASS. Đã kiểm đúng 6 file mỗi run, đối chiếu confusion matrix và Coverage–Accuracy với predictions, xác minh checksums và reload 32 mẫu với sai lệch xác suất tối đa bằng 0. Xem output cuối notebook để lấy đường dẫn từng run.

## Web Demo Streamlit — TV5 (11/11/2026)

| Phase | Task | PDF Requirement | Status | Evidence |
|---|---|---|---|---|
| 07 | Streamlit Web Demo local | Trang 17, Topic 05-B/D/E | VERIFIED | `app.py`, `src/triage.py`, `src/explanation.py`; chạy từ root qua lệnh bên dưới; 13 unittest PASS |

### Hướng dẫn chạy Web Demo từ Project Root

Yêu cầu môi trường Python 3.11.9 trong `.venv` đã cài đặt `requirements-task-a.txt` (gồm `streamlit==1.67.0`).

1. **Khởi chạy ứng dụng web:**
   ```powershell
   .\.venv\Scripts\python.exe -m streamlit run app.py
   ```
2. **Chạy bộ kiểm thử tự động (13 tests):**
   ```powershell
   .\.venv\Scripts\python.exe -m unittest discover -s tests -v
   ```
3. **Sinh lại tệp CSV mẫu kiểm chứng nếu cần:**
   ```powershell
   .\.venv\Scripts\python.exe -c "from preprocessing import load_splits; X, _ = load_splits('multiclass')['validation']; X.loc[[14627, 11611, 43302]].to_csv('demo_validation_features.csv', index=False)"
   ```

### Chức năng và hợp đồng kiểm chứng
- **Nạp Pipeline khóa:** Tự động resolve từ root, kiểm tra mã SHA-256 (`7036c10c...` cho Binary, `729608bd...` cho Multiclass) và đối chiếu danh sách classes từ `configs/operating_threshold.json`.
- **Hợp đồng đầu vào (55 features):** Kiểm tra bảng rỗng, cột thiếu, cột dư (`Class`, `Category`, `Filename`, `row_index`, `group_id`), cột trùng lặp, chữ rác không phải số và giá trị vô cực (`inf`). Cho phép đảo thứ tự cột và giữ ô trống (`NaN`) cho SimpleImputer.
- **Ngưỡng tin cậy $\tau = 0.65$:** So sánh $p_{\max} \ge \tau$ trên số thực nguyên bản trước khi hiển thị. Mẫu dưới ngưỡng bị từ chối với trạng thái `Uncertain`, vẫn hiển thị phân phối scores và lớp ứng viên điểm cao nhất.
- **Giải thích cục bộ Top-3 (Local Explanation):** Dùng LightGBM `pred_contrib` sau fitted preprocessing. Với Binary Benign, đảo dấu toàn bộ 55 đóng góp và baseline để giải thích margin đối nghịch. Với Multiclass, trích xuất đúng khối 56 phần tử của lớp ứng viên. Tổng đóng góp 55 features cộng baseline khớp tuyệt đối raw score của mô hình.

