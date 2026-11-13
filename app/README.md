# BÀN GIAO THÀNH VIÊN 5 (TV5) — WEB DEMO CIC-MALMEM-2022

## 1. Giới thiệu
Tài liệu và mã nguồn ứng dụng Web Demo Streamlit phục vụ đề tài **Topic 05: CIC-MalMem-2022 Forensic Memory Analysis**.
Ứng dụng thực hiện phân loại dấu vết mã độc từ 55 đặc trưng trích xuất bộ nhớ RAM bằng Volatility, tích hợp cơ chế từ chối `Uncertain` theo ngưỡng tin cậy $\tau$ và giải thích cục bộ Local Top-3.

## 2. Cấu trúc thư mục bàn giao
```text
app/
├── app.py                      # File Streamlit chính (Self-Contained & Modular)
├── requirements.txt            # Danh sách dependencies chuẩn
├── smoke_test.py               # Script kiểm thử khói tự động toàn diện
├── handoff_tv5_web_demo.md     # Tài liệu đặc tả kỹ thuật bàn giao TV5
├── prompt_tv5_web_demo.md      # Prompt yêu cầu công việc
├── src/
│   ├── __init__.py
│   ├── triage.py               # Module nạp config, pipeline & xác thực 55 features
│   └── explanation.py          # Module giải thích cục bộ Local Top-3 & Parity check
└── README.md                   # Hướng dẫn chạy và nghiệm thu
```

*(Lưu ý: File `app.py` đã được thiết kế tự chứa đầy đủ (Self-Contained) và tự động nhận diện project root. Người dùng có thể chạy `streamlit run app/app.py` hoặc copy `app.py` ra thư mục gốc để chạy `streamlit run app.py` mà không lo thiếu thư viện nội bộ).*

## 3. Cấu hình 4 Pipeline Models (theo `model/configs/inference.json`)

| Task | Thuật toán | ID Model trong config | Ngưỡng $\tau$ | Tập Classes | Vai trò | Trạng thái Local Top-3 |
|---|---|---|:---:|---|---|:---:|
| **Task A (Binary)** | **LightGBM** | `02_binary_lightgbm` | **0.95** | `Benign`, `Malware` | **Mặc định** | **Hỗ trợ** (đảo dấu Benign) |
| **Task A (Binary)** | Random Forest | `01_binary_random_forest` | 0.95 | `Benign`, `Malware` | Tùy chọn | *Chưa hỗ trợ* |
| **Task B (Multiclass)** | **LightGBM** | `04_multiclass_lightgbm` | **0.80** | `Benign`, `Ransomware`, `Spyware`, `Trojan` | **Mặc định** | **Hỗ trợ** (trích khối 4 lớp) |
| **Task B (Multiclass)** | Random Forest | `03_multiclass_random_forest` | 0.70 | `Benign`, `Ransomware`, `Spyware`, `Trojan` | Tùy chọn | *Chưa hỗ trợ* |

## 4. Hướng dẫn chạy ứng dụng

### Bước 1: Cài đặt môi trường (nếu cài mới)
```powershell
pip install -r app/requirements.txt
```

### Bước 2: Chạy kiểm thử khói tự động (Smoke Test)
```powershell
python app/smoke_test.py
```
*(Xác nhận toàn bộ các bước nạp 4 model, kiểm tra hợp đồng dữ liệu, suy luận biên và parity check đều đạt PASS 100%).*

### Bước 3: Khởi chạy Web Demo Streamlit
Từ thư mục gốc dự án:
```powershell
streamlit run app/app.py
```
*(Hoặc chạy `streamlit run app.py`).*

Mở trình duyệt tại: **http://localhost:8501**

## 5. Các tính năng nổi bật đã tích hợp
1. **Lựa chọn độc lập Task A / Task B và LightGBM / Random Forest**: Mặc định luôn là LightGBM ở cả 2 task.
2. **Hợp đồng dữ liệu 55 đặc trưng**: Bắt lỗi tệp rỗng, trùng tên cột header, thiếu/thừa cột, phi số, vô cực `inf`. Tự động gạt bỏ các cột nhãn thô (`Class`, `Category`, `multiclass`) khi nạp file từ raw dataset. Cho phép ô trống `NaN` để `SimpleImputer` tự động bù đắp.
3. **Quy tắc phân định tin cậy**: So sánh $p_{\max} \ge \tau$ **trước khi làm tròn số**. Gán nhãn `Uncertain` khi dưới ngưỡng đồng thời hiển thị lớp ứng viên và phân bổ điểm số các lớp.
4. **Giải thích cục bộ Local Top-3 cho LightGBM**: Phân rã `pred_contrib` từ Booster, đảo dấu Benign cho Binary, trích xuất khối lớp ứng viên cho Multiclass. Có bảng kiểm chứng tính toàn vẹn (Cryptographic Parity Verification: $\sum \text{contrib} + \text{baseline} = \text{raw margin}$ với sai số $|\Delta| < 10^{-4}$).
5. **Thông báo chuẩn cho Random Forest**: Hiển thị card cảnh báo trạng thái *Chưa hỗ trợ* rõ ràng, không gây crash ứng dụng.
6. **Mẫu kiểm thử nạp nhanh**: Tích hợp nút bấm nạp nhanh 5 mẫu chuẩn (đại diện Benign, Malware và các ca Uncertain) trên Sidebar.
7. **Giao diện High-Tech White SOC Cyberpunk**: Nền sáng công nghệ cao, hỗ trợ chuẩn font tiếng Việt (`Chakra Petch` & `JetBrains Mono`), không bị cắt chữ.
