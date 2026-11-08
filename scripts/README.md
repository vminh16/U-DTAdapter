# Scripts & Thí nghiệm Baseline B0 (Frozen Decision Transformer)

Thư mục này chứa các script thực thi quy trình huấn luyện và đánh giá **Baseline B0** (Decision Transformer đóng băng, Module R = OFF, Module U = OFF) trên môi trường con lắc nhảy (`Hopper-v4` / `Hopper-medium-v2`).

---

## 1. Tổng quan Kiến trúc & Thí nghiệm

- **Mục tiêu**: Thiết lập mốc đối chứng (ground-truth baseline) chuẩn xác trước khi tích hợp Module R (cắt tỉa RTG) và Module U (lập kế hoạch bất định).
- **Mô hình**: Decision Transformer gốc (Chen et al., 2021) với ngữ cảnh $K = 20$, 3 tầng transformer, 1 attention head, hidden dimension $d = 128$.
- **Tái sử dụng Baseline**: `scripts/train_dt.py` gọi hàm `udt.dt_integration.train_decision_transformer`, sử dụng trực tiếp bộ tối ưu và quản lý sequence `SequenceTrainer` từ `baselines/decision_transformer/gym/`.
- **Dữ liệu huấn luyện**: `data/prepared/hopper-medium-v2/dataset.npz` (được phân tách triệt để: 1.966 episodes tập train theo `is_validation == False`, 219 episodes tập validation).
- **Môi trường mô phỏng**: Sử dụng `gymnasium[mujoco]` (`Hopper-v4`) và binary wheels chính thức của DeepMind MuJoCo (`mujoco==3.3.7`), hoàn toàn loại bỏ phụ thuộc vào legacy `mujoco-py` và Python 3.8.

---

## 2. Cài đặt Môi trường (Environment Setup)

Dự án sử dụng Python 3.10 virtual environment (`.venv`).

### Bước 2.1. Kích hoạt môi trường ảo
```bash
source .venv/bin/activate
```

### Bước 2.2. Cài đặt các gói phụ thuộc
Tất cả các thư viện cần thiết đã được định nghĩa trong `requirements.txt`:
```bash
pip install -r requirements.txt
```

Các thư viện chính được cài đặt bao gồm:
- `transformers==4.18.0`: Cung cấp GPT-2 backbone cho Decision Transformer (được shim tương thích với codebase baseline).
- `gymnasium[mujoco]==1.2.3`: API môi trường RL hiện đại của Farama Foundation.
- `mujoco==3.3.7`: Engine mô phỏng vật lý nhị phân chính thức (không cần cài `mujoco-py` hay cấu hình binary C++ thủ công).

### Bước 2.3. Kiểm tra môi trường hoạt động
Chạy lệnh kiểm tra nhanh xem Gymnasium và MuJoCo đã sẵn sàng:
```bash
python -c "import gymnasium as gym, mujoco; env = gym.make('Hopper-v4'); obs, _ = env.reset(); print('✓ Môi trường Hopper-v4 khởi tạo thành công! Shape obs:', obs.shape)"
```

---

## 3. Quy trình Thực thi Thí nghiệm

### Bước 3.1. Huấn luyện Decision Transformer (Frozen DT)

Script `scripts/train_dt.py` thực hiện:
- Tải dataset `data/prepared/hopper-medium-v2/dataset.npz` và `manifest.json`.
- Tách tập train (1.966 trajectories) và validation (219 trajectories).
- Huấn luyện mô hình thông qua `SequenceTrainer.train_step()` với AdamW và warmup schedule.
- Đóng băng toàn bộ trọng số (`requires_grad = False`).
- Lưu checkpoint đầy đủ cấu hình và thông số chuẩn hóa (`state_mean`, `state_std`) vào `checkpoints/decision_transformer/model.pt`.

**Lệnh chạy trên GPU (Khuyến nghị - khoảng 2–3 phút cho 10.000 gradient steps):**
```bash
python scripts/train_dt.py \
  --data data/prepared/hopper-medium-v2/dataset.npz \
  --manifest data/prepared/hopper-medium-v2/manifest.json \
  --output checkpoints/decision_transformer/model.pt \
  --num-steps 10000 \
  --batch-size 64 \
  --k 20 \
  --eval-interval 1000 \
  --device cuda
```

**Hoặc chạy thử nghiệm trên CPU (5.000 gradient steps):**
```bash
python scripts/train_dt.py \
  --data data/prepared/hopper-medium-v2/dataset.npz \
  --manifest data/prepared/hopper-medium-v2/manifest.json \
  --output checkpoints/decision_transformer/model.pt \
  --num-steps 5000 \
  --device cpu
```

---

### Bước 3.2. Chạy đánh giá Closed-Loop Baseline B0 & Kiểm tra Parity

Script `scripts/evaluate.py --variant B0` thực hiện chu trình kiểm định khép kín:
1. **Kiểm tra đóng băng**: Xác thực toàn bộ tham số của mô hình có `requires_grad == False` và ở chế độ `eval()`.
2. **Kiểm tra Parity**: So sánh trực tiếp hành động suy luận của `DTAdapter` với mô hình gốc để đảm bảo không bị lệch chuẩn (`max_abs_difference <= 1e-6`). Xuất báo cáo ra `checkpoints/decision_transformer/parity.json`.
3. **Rollout đa episode**: Chạy $N$ episodes trong môi trường MuJoCo `Hopper-v4` thông qua hàm đánh giá `evaluate_with_budget` (với Return Capper = None, không rerank).
4. **Tổng hợp số liệu**: Tính toán Mean, Std, Median, Min, Max Return và độ dài episode, lưu kết quả chi tiết vào `results/b0_evaluation.json`.

**Lệnh chạy đánh giá với Target Return $G_0 = 1800.0$ (Phân vị ~90% của tập Hopper-medium):**
```bash
python scripts/evaluate.py \
  --variant B0 \
  --checkpoint checkpoints/decision_transformer/model.pt \
  --env Hopper-v4 \
  --target-return 1800.0 \
  --episodes 10 \
  --horizon 1000 \
  --parity-output results/b0_parity.json \
  --results-output results/b0_evaluation.json \
  --device cuda
```

**Thử nghiệm với Target Return cao $G_0 = 3600.0$ (để quan sát hiện tượng suy thoái/drift của DT gốc):**
```bash
python scripts/evaluate.py \
  --variant B0 \
  --checkpoint checkpoints/decision_transformer/model.pt \
  --env Hopper-v4 \
  --target-return 3600.0 \
  --episodes 10 \
  --horizon 1000 \
  --parity-output results/b0_parity_high_target.json \
  --results-output results/b0_evaluation_high_target.json \
  --device cuda
```

---

### Bước 3.3. Kiểm tra Trạng thái Sẵn sàng (Pilot Readiness Audit)

Sau khi chạy xong bước đánh giá, file `results/b0_parity.json` đã được tạo với kết quả xác thực. Bạn có thể kiểm tra trạng thái readiness của hệ thống:

```bash
python scripts/check_readiness.py \
  --manifest data/prepared/hopper-medium-v2/manifest.json \
  --dataset data/prepared/hopper-medium-v2/dataset.npz \
  --return-checkpoint checkpoints/module_r/return_capper.pt \
  --parity results/b0_parity.json \
  --variants B0,B3
```

Chỉ số `[PASS] frozen_dt_parity: Frozen-DT parity passed against the real checkpoint` sẽ sáng xanh, giải tỏa hoàn toàn blocker này cho Baseline B0.

---

## 4. Danh mục File Đầu ra (Artifacts)

| Đường dẫn | Mô tả |
| :--- | :--- |
| `checkpoints/decision_transformer/model.pt` | Checkpoint mô hình Decision Transformer đã đóng băng trọng số |
| `results/b0_parity.json` | Báo cáo kiểm định sai số giữa DT gốc và DTAdapter (`passed: true`) |
| `results/b0_evaluation.json` | Kết quả phân phối return, độ dài và trạng thái từng episode của Baseline B0 |
