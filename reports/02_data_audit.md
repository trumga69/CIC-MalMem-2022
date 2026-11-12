# Data audit — bằng chứng sơ bộ phục vụ PHASE 0

Nguồn tính toán: `objective_audit/inspect_existing.py` → `objective_audit/observations.json`. Ngày 06/11/2026. Đây chưa phải tuyên bố Phase 2 hoàn thành.

| Kiểm tra | Kết quả trực tiếp |
|---|---:|
| Full CSV | 58.596 × 57 cột |
| Numeric predictors | 55 |
| Benign / Malware | 29.298 / 29.298 |
| Benign / Ransomware / Spyware / Trojan | 29.298 / 9.791 / 10.020 / 9.487 |
| Missing / infinity / negative cells | 0 / 0 / 0 |
| Constant columns | 3 |
| Duplicate full rows excess / feature rows excess | 534 / 569 |
| Rows in duplicate feature groups | 1.086 |
| Conflicting feature groups binary / multiclass | 0 / 9 |
| Connected groups | 32.087 |
| Groups / exact feature groups / tokens crossing splits | 0 / 0 / 0 |
| Unparsed malware Category tokens | 0 |

Cột hằng: pslist.nprocs64bit, handles.nport, svcscan.interactive_process_services; các cặp này có cùng giá trị theo từng hàng. Numeric feature groups được quan sát theo prefix: pslist, dlllist, handles, ldrmodules, malfind, psxview, modules, svcscan, callbacks. Không suy diễn nguồn Volatility/version chỉ từ tên cột.

Hashes, file sizes, dtypes, ordered feature list, class/split counts, near-constant ≥99%, duplicate-column pairs và checksums của artifacts được lưu trong JSON. Không sửa raw hoặc split. Các bảng phân phối cũ nằm reports/audit nhưng chưa tái chạy audit.py trong phiên này.

Ba Output CSV: 150/350/350 dòng, mỗi file 56 cột gồm Filename và 55 numeric features; không có Class/Category. So sánh exact numeric fingerprints trên shared features cho 0 match; không đủ để kết luận không liên quan vì cần kiểm tra rounding/nguồn thu thập. Không xác nhận tuyên bố README rằng 850 mẫu đã có trong full CSV.

Giới hạn/chưa làm: đối chiếu byte identity UNB và phiên bản nguồn; giải thích 55 vs 56; kiểm tra identifier convention gốc, family taxonomy không dùng làm task mới; near-duplicates theo tiêu chí đăng ký trước; ngữ nghĩa bounds/ratio/counts; shortcuts do môi trường thu thập. Chưa kiểm định deployment/OOD, không có final test metrics.

Xem Requirement-to-Evidence Matrix và mức ưu tiên rủi ro tại `01_objective_contract.md`.
