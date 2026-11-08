# Bảng Theo Dõi Thực Nghiệm (Experiment Tracking — Exploratory Pilot)

> **Lưu ý khoa học:** Toàn bộ kết quả trong tài liệu này thuộc giai đoạn **thăm dò (exploratory pilot)** trên một training checkpoint duy nhất (`checkpoints/decision_transformer/model.pt`), được đánh giá qua các evaluation rollout seeds (0, 42, 100; mỗi seed 10 episodes). Đây **không** phải là các independent training seeds và **không** cấu thành bằng chứng kiểm chứng khẳng định (confirmatory evidence) hay bảo đảm an toàn/tính khả thi lý thuyết.

- **Môi trường**: `Hopper-v4`
- **Tập dữ liệu**: `data/prepared/hopper-medium-v2-pilot/` (`hopper-medium-v2`)
- **Cấu hình chuẩn**: Target Return = `1800.0`, Horizon = `1000`, Context length $K = 20$, Batch size = $64$

---

## 1. Tổng Quan Tiến Độ

| Biến thể | Mô tả | Module R | Module U | Trạng thái | Điểm TB (Overall across eval seeds) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **B0** | Vanilla Decision Transformer (Đối chứng) | OFF | OFF | **Hoàn thành (pilot)** | **918.24 ± 141.14** (Target 1800) |
| **B1** | Target Sensitivity Grid (Khảo sát RTG) | OFF | OFF | **Hoàn thành (pilot)** | **1142.17 → 1606.20** (4 targets) |
| **B3** | DT + Return Capper | ON | OFF | **Hoàn thành (pilot)** | **945.28 ± 120.95** (Median 956.35) |
| **B4** | DT + Reranking (Uncertainty) | OFF | ON | **Hoàn thành (pilot)** | **1468.86 ± 126.51** (Median 1472.17) |
| **B5** | U-DT Đầy đủ (Cap + Rerank) | ON | ON | **Hoàn thành (pilot)** | **1473.44 ± 179.78** (Median 1476.43) |

---

## 2. Chi Tiết Kết Quả Từng Biến Thể

### B0 — Vanilla Decision Transformer (Baseline)
- **Checkpoint**: `checkpoints/decision_transformer/model.pt`
- **Mục tiêu**: Làm đối chứng cơ sở trong pilot, đo lường hành vi policy khi không có can thiệp.

| Eval Seed | Episodes | Mean Return | Std Return | Median | Mean Length | Terminated / Truncated | Parity | Ghi chú |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **42** | 10 | 865.30 | 115.61 | 845.22 | 288.6 | 10 / 0 | PASS (0.0) | Evaluation seed 42 |
| **0** | 10 | 960.21 | 177.05 | 886.20 | 313.3 | 10 / 0 | PASS (0.0) | Evaluation seed 0 |
| **100** | 10 | 929.19 | 101.81 | 894.68 | 303.5 | 10 / 0 | PASS (0.0) | Evaluation seed 100 |
| **OVERALL** | **30** | **918.24** | **141.14** | **879.12** | **301.8** | **30 / 0** | **PASS** | **Mốc baseline đối chứng pilot** |

---

### B1 — Target Return Sensitivity Grid (Khảo Sát Độ Nhạy RTG)
- **Checkpoint**: `checkpoints/decision_transformer/model.pt` (dùng chung checkpoint với B0, R/U đều OFF)
- **Mục tiêu**: Khảo sát phản ứng của Decision Transformer khi thay đổi ngân sách RTG ban đầu; đo khoảng cách giữa mục tiêu đặt ra và điểm thực tế đạt được (Target Gap).

| Target Input | Phân loại Target | Mean Return (30 eps) | Std Return | Median | Mean Length | Tỷ lệ Đạt (Return / Target) | Ghi chú |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1460.0** | Median ($p_{50}$) của tập train | **1142.17** | 228.41 | 1069.97 | 374.0 | 78.2% | Target khả thi trong tập dữ liệu |
| **1800.0** | Tham chiếu chuẩn ($p_{90}$, B0) | **918.24** | 141.14 | 879.12 | 301.8 | 51.0% | Điểm B0 pilot |
| **2700.0** | Mức cao ($1.5\times$ target chuẩn) | **1476.32** | 406.70 | 1482.68 | 476.1 | 54.7% | Mean length cao hơn, biến động lớn |
| **3600.0** | Cực đoan ($2.0\times$, OOD) | **1606.20** | 255.44 | 1507.76 | 513.6 | 44.6% | Return trung bình đạt cao nhất trong grid |

#### Chi tiết từng evaluation seed của B1:
- **Target 1460.0**: Seed 0 = 1119.88 (len 370.6) | Seed 100 = 1179.49 (len 384.6) | Seed 42 = 1127.15 (len 366.9)
- **Target 1800.0**: Seed 0 = 960.21 (len 313.3) | Seed 100 = 929.19 (len 303.5) | Seed 42 = 865.30 (len 288.6)
- **Target 2700.0**: Seed 0 = 1496.95 (len 482.0) | Seed 100 = 1414.87 (len 458.6) | Seed 42 = 1517.15 (len 487.6)
- **Target 3600.0**: Seed 0 = 1566.62 (len 503.7) | Seed 100 = 1681.64 (len 536.6) | Seed 42 = 1570.33 (len 500.5)
*(Toàn bộ 120/120 episodes trong B1 đều kết thúc do `terminated: 10/10` mỗi seed).*

---

### B3 — DT + Module R (Return Capper)
- **Checkpoint R**: `checkpoints/module_r/return_capper.pt`
- **Mục tiêu**: Đánh giá thực nghiệm việc áp dụng upper-expectile conditioning token $g_\psi(z_t, t/H) + \delta$ (không phải bound khả thi đã chứng minh).

| Eval Seed | Episodes | Mean Return | Std Return | Median | Mean Length | Clipping Freq | Parity | Ghi chú |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **42** | 10 | 876.67 | 124.04 | 828.52 | 292.5 | 100.0% | PASS (0.0) | Return tăng nhẹ (+11.4) so với B0 |
| **0** | 10 | 971.67 | 119.70 | 1002.86 | 321.6 | 100.0% | PASS (0.0) | Median 1002.86 |
| **100** | 10 | 987.48 | 83.62 | 995.44 | 326.2 | 100.0% | PASS (0.0) | Std mẫu trong seed 83.62 |
| **OVERALL** | **30** | **945.28** | **120.95** | **956.35** | **313.4** | **100.0%** | **PASS** | **Mẫu 30 episodes: Mean (+27), Median (+77)** |

*(Lưu ý: Mức giảm độ lệch chuẩn mẫu qua 30 evaluation episodes trên một checkpoint duy nhất là quan sát thăm dò, chưa đủ bằng chứng kết luận về tính ổn định tổng quát).*

---

### B4 — DT + Module U (Uncertainty Reranking)
- **Checkpoint Dynamics**: `checkpoints/module_u/dynamics_ensemble.pt`
- **Checkpoint Scorer ($g_\psi$)**: `checkpoints/module_r/return_capper.pt`
- **Cấu hình**: $N = 10$ candidates, radius $\eta = 0.05$, penalty $\beta = 0.1$, Module R = OFF
- **Mục tiêu**: Đánh giá thực nghiệm việc rerank hành động qua candidate pool và heuristic scoring (chưa bật cap).

| Eval Seed | Episodes | Mean Return | Std Return | Median | Mean Length | Intervention Freq | Parity | Ghi chú |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **42** | 10 | 1441.28 | 128.49 | 1418.24 | 456.4 | 99.72% | PASS (0.0) | Evaluation seed 42 |
| **0** | 10 | 1457.41 | 80.32 | 1496.28 | 464.1 | 99.81% | PASS (0.0) | Evaluation seed 0 |
| **100** | 10 | 1507.90 | 150.46 | 1480.05 | 479.7 | 99.54% | PASS (0.0) | Evaluation seed 100 |
| **OVERALL** | **30** | **1468.86** | **126.51** | **1472.17** | **466.7** | **99.69%** | **PASS** | **Tăng điểm trung bình pilot tại target 1800** |

---

### B5 — Full U-DT (Module R + Module U)
- **Checkpoints**: Dùng cả Capper (`checkpoints/module_r/return_capper.pt`) và Dynamics Ensemble (`checkpoints/module_u/dynamics_ensemble.pt`)
- **Cấu hình**: $N = 10$, radius $\eta = 0.05$, penalty $\beta = 0.1$, Module R = ON, Module U = ON
- **Mục tiêu**: Khảo sát thực nghiệm sự kết hợp giữa RTG cap và candidate reranking trên thiết kế ma trận $2 \times 2$.

| Eval Seed | Episodes | Mean Return | Std Return | Median | Mean Length | Clipping Freq | Intervention Freq | Parity | Ghi chú |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **42** | 10 | 1520.74 | 263.91 | 1467.21 | 482.0 | 69.24% | 99.81% | PASS (0.0) | Evaluation seed 42 |
| **0** | 10 | 1414.62 | 95.54 | 1442.97 | 449.9 | 75.29% | 99.78% | PASS (0.0) | Evaluation seed 0 |
| **100** | 10 | 1484.95 | 111.13 | 1496.96 | 472.4 | 67.99% | 99.59% | PASS (0.0) | Evaluation seed 100 |
| **OVERALL** | **30** | **1473.44** | **179.78** | **1476.43** | **468.1** | **70.84%** | **99.73%** | **PASS** | **Mean return pilot tại target 1800** |

---

## 3. Phân Tích Ma Trận Factorial $2 \times 2$ (RQ3: Tương Tác R và U)

Theo đặc tả thực nghiệm trong `docs/research/EXPERIMENT.md` (Mục 7), thiết kế ma trận $2 \times 2$ gồm:

| | **Module U = OFF** | **Module U = ON** | Hiệu ứng riêng của U ($\Delta_U$) |
| :--- | :---: | :---: | :---: |
| **Module R = OFF** | **B0**: 918.24 | **B4**: 1468.86 | **+550.62** |
| **Module R = ON** | **B3**: 945.28 | **B5**: 1473.44 | **+528.16** |
| **Hiệu ứng riêng của R ($\Delta_R$)** | **+27.04** | **+4.58** | **$\Delta_{int} = -22.46$** |

### Nhận định thực nghiệm (Exploratory):
1. **Module U tạo ra phần lớn mức tăng return tại target 1800**: Reranking hành động qua candidate pool với heuristic score $J$ làm tăng mean return từ $918.24 \to 1468.86$ trên tập evaluation episodes này. Tuy nhiên, hiệu quả này đến chủ yếu từ việc lấy mẫu candidate $N=10$ và return model $g_\psi$, không thể tự động gán cho disagreement penalty khi chưa có paired confidence intervals.
2. **Module R trong pilot**: Khi đứng riêng (B3), Module R ghi nhận tăng điểm trung bình (+27.04) và tăng trung vị (+77.23). Tuy nhiên, đây là expectile heuristic trên dữ liệu tĩnh, không phải feasible bound hay cơ chế bảo vệ an toàn đã chứng minh.
3. **Hiệu ứng tương tác $\Delta_{int} = -22.46$**:
   - Giá trị tương tác mang dấu âm trong mẫu pilot này cho thấy sự kết hợp không mang tính cộng dồn đơn thuần (sub-additive). Cần chạy benchmark chính với nhiều independent training seeds và tính khoảng tin cậy (CI) trước khi đưa ra kết luận khẳng định về tương tác.
   - Việc tỷ lệ clipping giảm từ 100% (B3) xuống 70.84% (B5) chỉ phản ánh rằng ngân sách raw $B_t^{raw}$ ít vượt qua ngưỡng $C_t$ hơn trong các rollout của B5, không phải bằng chứng trực tiếp cho thấy reward được tích lũy nhanh hơn hay policy đạt bão hòa rủi ro.

---

## 4. Khảo Sát Độ Nhạy Target (Target Sensitivity Grid: B0/B1 vs B4 vs B5)

Khảo sát trên 4 mức RTG ban đầu qua 30 episodes (3 evaluation seeds × 10 eps, cùng 1 training checkpoint). *(100% episodes đều kết thúc do `terminated: 10/10`)*:

| Target Input | Phân loại Target | B0 / B1 (Vanilla DT)<br>Return [Thời gian/Bước] | B4 (DT + Module U)<br>Return [Thời gian/Bước] | B5 (Full U-DT)<br>Return [Thời gian/Bước] | Ghi chú & Hiện tượng quan sát |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **1460.0** | Train median ($p_{50}$) | 1142.17 ± 228.41<br>[374.0 bước] | **1418.15 ± 153.30**<br>[**455.6 bước**] | **1418.21 ± 152.86**<br>[**455.6 bước**] | B4/B5 đạt return cao hơn B1 tại target 1460 |
| **1800.0** | Chuẩn ($p_{90}$, baseline) | 918.24 ± 141.14<br>[301.8 bước] | **1468.86 ± 126.51**<br>[**466.7 bước**] | **1473.44 ± 179.78**<br>[**468.1 bước**] | B4/B5 đạt return cao hơn B0 tại target 1800 |
| **2700.0** | Mức cao ($1.5\times$) | **1476.32 ± 406.70**<br>[**476.1 bước**] | 1474.92 ± 121.54<br>[469.8 bước] | 1475.68 ± 147.35<br>[468.1 bước] | B1 đạt return trung bình tương đương B4/B5 |
| **3600.0** | Cực đoan (OOD, $2.0\times$) | **1606.20 ± 255.44**<br>[**513.6 bước**] | 1510.61 ± 140.14<br>[479.1 bước] | 1475.68 ± 147.35<br>[468.1 bước] | B1 đạt return cao hơn B4 và B5 |

**Nhận xét:**
- B4 và B5 cho kết quả cao hơn B0/B1 ở target 1460 và target chuẩn 1800 trong pilot này. Tuy nhiên, **B4/B5 không vượt B0/B1 ở mọi target**: tại target 2700, B1 đạt 1476.32 (tương đương B4/B5), và tại target 3600, B1 đạt 1606.20 (cao hơn cả B4 1510.61 và B5 1475.68).
- Tại target 2700 và 3600, kết quả B5 lặp lại ở mức 1475.68 do token bị gọt về cùng ngưỡng capper trong phần lớn bước chạy; theo RRD và EXPERIMENT.md, đường cong phẳng do clipping không đồng nghĩa với việc đã chứng minh chất lượng điều khiển tốt hơn.

---

## 5. Phân Tích Cơ Chế B4 (Ablation: Candidate Pool $N$ & Penalty $\beta$)

Đánh giá các thành phần của Module U tại target chuẩn 1800.0 (30 episodes, evaluation seeds 0, 42, 100, cùng 1 checkpoint):

| Cấu hình | Mô tả cơ chế | Mean Return | Std | Median | Mean Len | Interv. Freq | Tác động quan sát trong pilot |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **$N = 1$** | Bypass search (chỉ dùng $a_0$) | **918.24** | 141.14 | 879.12 | 301.8 | 0.0% | Khớp B0 tuyệt đối; kiểm chứng implementation parity |
| **$N = 10, \beta = 0.0$** | B7: Ensemble, không penalty | **1471.93** | 140.61 | 1488.40 | 467.3 | 99.7% | Scorer $g_\psi$ đóng góp chính vào mức tăng return (+553.69) |
| **$N = 10, \beta = 0.1$** | B4 chuẩn: Có penalty bất định | **1468.86** | 126.51 | 1472.17 | 466.7 | 99.7% | Mean return 1468.86 (chưa chứng minh được lợi ích riêng của $\beta$) |

**Nhận xét:**
1. **Lấy mẫu candidate $N=10$ kết hợp scorer $g_\psi$**: Chuyển từ $N=1 \to 10$ giải thích hầu như toàn bộ mức thay đổi return so với $N=1$.
2. **Vai trò của penalty bất định $\beta$**: So sánh giữa $\beta=0.0$ (1471.93) và $\beta=0.1$ (1468.86) trong pilot chưa cho thấy sự vượt trội về return của việc bổ sung penalty bất định. Sự khác biệt về độ lệch chuẩn mẫu (126.51 vs 140.61) trên 30 evaluation episodes không đủ để khẳng định tính ổn định khi chưa có paired confidence intervals trên nhiều independent training seeds.
