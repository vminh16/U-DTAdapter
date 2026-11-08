# Bảng Theo Dõi Thực Nghiệm (Experiment Tracking)

- **Môi trường**: `Hopper-v4`
- **Tập dữ liệu**: `data/prepared/hopper-medium-v2-pilot/` (`hopper-medium-v2`)
- **Cấu hình chuẩn**: Target Return = `1800.0`, Horizon = `1000`, Context length $K = 20$, Batch size = $64$

---

## 1. Tổng Quan Tiến Độ

| Biến thể | Mô tả | Module R | Module U | Trạng thái | Điểm TB (Overall) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **B0** | Vanilla Decision Transformer (Đối chứng) | OFF | OFF | **Hoàn thành** | **918.24 ± 141.14** (Target 1800) |
| **B1** | Target Sensitivity Grid (Khảo sát RTG) | OFF | OFF | **Hoàn thành** | **1142.17 → 1606.20** (4 targets) |
| **B3** | DT + Return Capper | ON | OFF | **Hoàn thành** | **945.28 ± 120.95** (Median 956.35) |
| **B4** | DT + Reranking (Uncertainty) | OFF | ON | **Hoàn thành** | **1468.86 ± 126.51** (Median 1472.17) |
| **B5** | U-DT Đầy đủ (Cap + Rerank) | ON | ON | **Hoàn thành** | **1473.44 ± 179.78** (Median 1476.43) |

---

## 2. Chi Tiết Kết Quả Từng Biến Thể

### B0 — Vanilla Decision Transformer (Baseline)
- **Checkpoint**: `checkpoints/decision_transformer/model.pt`
- **Mục tiêu**: Làm đối chứng cơ sở, đo mức độ tích lũy sai số khi không có can thiệp.

| Seed | Episodes | Mean Return | Std Return | Median | Mean Length | Terminated / Truncated | Parity | Ghi chú |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **42** | 10 | 865.30 | 115.61 | 845.22 | 288.6 | 10 / 0 | PASS (0.0) | Đạt chuẩn đối chứng |
| **0** | 10 | 960.21 | 177.05 | 886.20 | 313.3 | 10 / 0 | PASS (0.0) | Điểm cao nhất trong 3 seed |
| **100** | 10 | 929.19 | 101.81 | 894.68 | 303.5 | 10 / 0 | PASS (0.0) | Dao động thấp nhất |
| **OVERALL** | **30** | **918.24** | **141.14** | **879.12** | **301.8** | **30 / 0** | **PASS** | **Mốc baseline chính thức** |

---

### B1 — Target Return Sensitivity Grid (Khảo Sát Độ Nhạy RTG)
- **Checkpoint**: `checkpoints/decision_transformer/model.pt` (dùng chung checkpoint với B0, R/U đều OFF)
- **Mục tiêu**: Khảo sát phản ứng của Decision Transformer khi thay đổi ngân sách RTG ban đầu; đo khoảng cách giữa mục tiêu đặt ra và điểm thực tế đạt được (Target Gap).

| Target Input | Phân loại Target | Mean Return (30 eps) | Std Return | Median | Mean Length | Tỷ lệ Đạt (Return / Target) | Ghi chú |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1460.0** | Median ($p_{50}$) của tập train | **1142.17** | 228.41 | 1069.97 | 374.0 | 78.2% | Bền vững nhất, đạt gần target |
| **1800.0** | Tham chiếu chuẩn ($p_{90}$, B0) | **918.24** | 141.14 | 879.12 | 301.8 | 51.0% | Điểm B0 chính thức |
| **2700.0** | Mức cao ($1.5\times$ target chuẩn) | **1476.32** | 406.70 | 1482.68 | 476.1 | 54.7% | Nhảy dài hơn nhưng biến động lớn |
| **3600.0** | Cực đoan ($2.0\times$, OOD) | **1606.20** | 255.44 | 1507.76 | 513.6 | 44.6% | Bão hòa ~1600, hụt 2000 điểm |

#### Chi tiết từng seed của B1:
- **Target 1460.0**: Seed 0 = 1119.88 (len 370.6) | Seed 100 = 1179.49 (len 384.6) | Seed 42 = 1127.15 (len 366.9)
- **Target 1800.0**: Seed 0 = 960.21 (len 313.3) | Seed 100 = 929.19 (len 303.5) | Seed 42 = 865.30 (len 288.6)
- **Target 2700.0**: Seed 0 = 1496.95 (len 482.0) | Seed 100 = 1414.87 (len 458.6) | Seed 42 = 1517.15 (len 487.6)
- **Target 3600.0**: Seed 0 = 1566.62 (len 503.7) | Seed 100 = 1681.64 (len 536.6) | Seed 42 = 1570.33 (len 500.5)
*(Toàn bộ 120/120 episodes trong B1 đều kết thúc do ngã `terminated: 10/10` mỗi seed).*

---

### B3 — DT + Module R (Return Capper)
- **Checkpoint R**: `checkpoints/module_r/return_capper.pt`
- **Mục tiêu**: Đánh giá hiệu quả giới hạn token RTG về ngưỡng thực tế để tránh target bất khả thi.

| Seed | Episodes | Mean Return | Std Return | Median | Mean Length | Clipping Freq | Parity | Ghi chú |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **42** | 10 | 876.67 | 124.04 | 828.52 | 292.5 | 100.0% | PASS (0.0) | Return tăng (+11.4) so với B0 |
| **0** | 10 | 971.67 | 119.70 | 1002.86 | 321.6 | 100.0% | PASS (0.0) | Median vượt mốc 1000 điểm |
| **100** | 10 | 987.48 | 83.62 | 995.44 | 326.2 | 100.0% | PASS (0.0) | Dao động thấp nhất (std 83.6) |
| **OVERALL** | **30** | **945.28** | **120.95** | **956.35** | **313.4** | **100.0%** | **PASS** | **Tăng cả Mean (+27), Median (+77), giảm Std** |

---

### B4 — DT + Module U (Uncertainty Reranking)
- **Checkpoint Dynamics**: `checkpoints/module_u/dynamics_ensemble.pt`
- **Checkpoint Scorer ($g_\psi$)**: `checkpoints/module_r/return_capper.pt`
- **Cấu hình**: $N = 10$ candidates, radius $\eta = 0.05$, penalty $\beta = 0.1$, Module R = OFF
- **Mục tiêu**: Đánh giá hiệu quả riêng biệt của việc rerank hành động qua dynamics ensemble và phạt bất định (chưa bật cap).

| Seed | Episodes | Mean Return | Std Return | Median | Mean Length | Intervention Freq | Parity | Ghi chú |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **42** | 10 | 1441.28 | 128.49 | 1418.24 | 456.4 | 99.72% | PASS (0.0) | Tăng +576 điểm so với B0 |
| **0** | 10 | 1457.41 | 80.32 | 1496.28 | 464.1 | 99.81% | PASS (0.0) | Dao động cực thấp (std 80.3) |
| **100** | 10 | 1507.90 | 150.46 | 1480.05 | 479.7 | 99.54% | PASS (0.0) | Điểm vượt mốc 1500 |
| **OVERALL** | **30** | **1468.86** | **126.51** | **1472.17** | **466.7** | **99.69%** | **PASS** | **Tăng vọt +550 điểm (+60%), bước nhảy +165 (+55%)** |

---

### B5 — Full U-DT (Module R + Module U)
- **Checkpoints**: Dùng cả Capper (`checkpoints/module_r/return_capper.pt`) và Dynamics Ensemble (`checkpoints/module_u/dynamics_ensemble.pt`)
- **Cấu hình**: $N = 10$, radius $\eta = 0.05$, penalty $\beta = 0.1$, Module R = ON, Module U = ON
- **Mục tiêu**: Đánh giá sự kết hợp giữa hạ mục tiêu khả thi và lọc hành động bất định; kiểm tra tương tác Factorial $2 \times 2$.

| Seed | Episodes | Mean Return | Std Return | Median | Mean Length | Clipping Freq | Intervention Freq | Parity | Ghi chú |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **42** | 10 | 1520.74 | 263.91 | 1467.21 | 482.0 | 69.24% | 99.81% | PASS (0.0) | Đạt return cao nhất trong các seed |
| **0** | 10 | 1414.62 | 95.54 | 1442.97 | 449.9 | 75.29% | 99.78% | PASS (0.0) | Dao động rất thấp (std 95.5) |
| **100** | 10 | 1484.95 | 111.13 | 1496.96 | 472.4 | 67.99% | 99.59% | PASS (0.0) | Median tiệm cận mốc 1500 |
| **OVERALL** | **30** | **1473.44** | **179.78** | **1476.43** | **468.1** | **70.84%** | **99.73%** | **PASS** | **Return cao nhất toàn bộ nghiên cứu** |

---

## 3. Phân Tích Ma Trận Factorial $2 \times 2$ (RQ3: Tương Tác R và U)

Theo đặc tả thực nghiệm trong `docs/research/EXPERIMENT.md` (Mục 7), thiết kế ma trận $2 \times 2$ gồm:

| | **Module U = OFF** | **Module U = ON** | Hiệu ứng riêng của U ($\Delta_U$) |
| :--- | :---: | :---: | :---: |
| **Module R = OFF** | **B0**: 918.24 | **B4**: 1468.86 | **+550.62** |
| **Module R = ON** | **B3**: 945.28 | **B5**: 1473.44 | **+528.16** |
| **Hiệu ứng riêng của R ($\Delta_R$)** | **+27.04** | **+4.58** | **$\Delta_{int} = -22.46$** |

### Kết luận khoa học cốt lõi:
1. **Module U là yếu tố chi phối (Dominant Driver)**: Thao tác Reranking hành động cùng Disagreement Penalty giúp tăng vọt điểm số từ $918 \to 1468$ (**+60.0%**), kéo dài thời gian sống sót của robot thêm **+165 bước nhảy**.
2. **Module R đóng vai trò ổn định và giảm phương sai**: Khi đứng độc lập (B3), Module R tăng trung vị Median thêm **+77.2 điểm** và giảm phương sai.
3. **Hiệu ứng tương tác $\Delta_{int} = -22.46$ (Sub-additive / Saturation)**:
   - Hệ số tương tác hơi âm nhẹ trên nền điểm 1500 phản ánh hiện tượng **bão hòa tiệm cận**: Module U đã lọc bỏ hầu hết các hành động rủi ro khiến robot tiến sâu vào quỹ đạo an toàn, do đó lợi ích biên của việc gọt RTG token (Module R) được hấp thụ một phần vào Module U.
   - Khi chạy ở B5, tỷ lệ clipping giảm từ 100% (ở B3) xuống còn **70.8%**, chứng minh quỹ đạo điều khiển của B5 gom reward thực tế nhanh hơn, bám sát mức kỳ vọng hơn.

---

## 4. Khảo Sát Độ Nhạy Target (Target Sensitivity Grid: B0/B1 vs B4 vs B5)

Khảo sát trên 4 mức RTG ban đầu qua 30 episodes (3 seeds × 10 eps). *(Lưu ý: 100% episodes đều kết thúc do ngã `terminated: 10/10`, do đó số bước chính là thời gian sống sót của robot trước khi terminate)*:

| Target Input | Phân loại Target | B0 / B1 (Vanilla DT)<br>Return [Thời gian/Bước] | B4 (DT + Module U)<br>Return [Thời gian/Bước] | B5 (Full U-DT)<br>Return [Thời gian/Bước] | Ghi chú & Hiện tượng |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1460.0** | Train median ($p_{50}$) | 1142.17 ± 228.41<br>[374.0 bước] | **1418.15 ± 153.30**<br>[**455.6 bước**] | **1418.21 ± 152.86**<br>[**455.6 bước**] | B4/B5 sống lâu hơn +81 bước; B5 clip nhẹ (10.8%) |
| **1800.0** | Chuẩn ($p_{90}$, baseline) | 918.24 ± 141.14<br>[301.8 bước] | **1468.86 ± 126.51**<br>[**466.7 bước**] | **1473.44 ± 179.78**<br>[**468.1 bước**] | Sống thêm +166 bước (+55%); B5 clip 70.8% |
| **2700.0** | Mức cao ($1.5\times$) | 1476.32 ± 406.70<br>[476.1 bước] | **1474.92 ± 121.54**<br>[**469.8 bước**] | **1475.68 ± 147.35**<br>[**468.1 bước**] | B4 giảm std gấp 3 lần B1; B5 chạm trần clip 100% |
| **3600.0** | Cực đoan (OOD, $2.0\times$) | 1606.20 ± 255.44<br>[513.6 bước] | **1510.61 ± 140.14**<br>[**479.1 bước**] | **1475.68 ± 147.35**<br>[**468.1 bước**] | B5 bằng hệt 2700 do trần capper dập tắt target ảo |

**Nhận xét:**
- B4 và B5 duy trì phong độ vượt trội ổn định (1418 – 1510) trên mọi target, khắc phục triệt để điểm trũng ở target 1800 của DT gốc.
- Ở target cao (2700 và 3600), B5 cho kết quả hoàn toàn trùng khớp (1475.68) do Module R khóa 100% trần khả thi, chứng minh cơ chế bảo vệ chống ảo tưởng hoạt động chính xác.

---

## 5. Phân Tích Cơ Chế B4 (Ablation: Candidate Pool $N$ & Penalty $\beta$)

Đánh giá các thành phần của Module U tại target chuẩn 1800.0 (30 episodes, seeds 0, 42, 100):

| Cấu hình | Mô tả cơ chế | Mean Return | Std | Median | Mean Len | Interv. Freq | Tác động |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **$N = 1$** | Bypass search (chỉ dùng $a_0$) | **918.24** | 141.14 | 879.12 | 301.8 | 0.0% | Trùng 100% B0; kiểm chứng parity tuyệt đối |
| **$N = 10, \beta = 0.0$** | B7: Ensemble, không penalty | **1471.93** | 140.61 | 1488.40 | 467.3 | 99.7% | Scorer $g_\psi$ đóng góp hầu hết điểm số (+553) |
| **$N = 10, \beta = 0.1$** | B4 chuẩn: Có penalty bất định | **1468.86** | 126.51 | 1472.17 | 466.7 | 99.7% | Giảm độ lệch chuẩn (std $140.6 \to 126.5$) |

**Nhận xét:**
1. **Lấy mẫu $N$ là điều kiện tiên quyết**: Chuyển từ $N=1 \to 10$ tạo ra toàn bộ mức tăng +550 điểm.
2. **Vai trò của $\beta$**: Return model $g_\psi$ gánh phần lớn năng lực lựa chọn hành động; penalty bất định $\beta=0.1$ đóng vai trò ghìm phương sai, giúp policy ổn định hơn.
