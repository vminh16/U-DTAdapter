# ĐẶC TẢ YÊU CẦU NGHIÊN CỨU — U-DTVerifier

**Tên đề tài:** Can thiệp nhẹ tại thời điểm suy luận cho Decision Transformer đóng băng bằng hiệu chỉnh Return-to-Go và lựa chọn hành động có phạt độ bất đồng mô hình.

**Tên làm việc:** U-DTVerifier. Giữ tên để truy vết đề xuất; “Verifier” **không** mang nghĩa chứng nhận tính khả thi, an toàn hoặc cải thiện chính sách. Trong mô tả kỹ thuật, dùng *RTG adjustment* và *disagreement-penalized action reranking*.

**Trạng thái:** Đặc tả nghiên cứu sửa theo phản biện toán học, chưa phải kết quả hay thuật toán đã được kiểm chứng. Mục tiêu trước mắt là một nghiên cứu thực nghiệm có thể tái lập; tính mới và lợi ích cần được xác nhận.

**Phân vai tài liệu:**

- Tài liệu này xác định bài toán, công thức, giả thuyết, giới hạn và yêu cầu bằng chứng.
- [EXPERIMENT.md](EXPERIMENT.md) xác định baselines, giao thức đánh giá, cấu hình và các mốc quyết định.
- [Mathematical Review](U-DTVerifier_Mathematical_Review.md) lưu phản biện đối với phiên bản đề xuất trước khi sửa. Các công thức cũ trong đó không phải đặc tả triển khai hiện hành.

## 1. Bài toán và phạm vi

### 1.1. Môi trường, dữ liệu và return

Xét MDP hữu hạn chân trời $H$, trạng thái $s_t\in\mathbb R^{d_s}$, hành động liên tục có biên $a_t\in\mathcal A$, reward $r_t$, transition $P$. Chỉ số $t=0,\ldots,H-1$; episode có thể kết thúc tự nhiên sớm tại bước cuối $T\le H-1$.

Phiên bản đầu dùng **return không chiết khấu**, tương ứng $\gamma=1$:

$$G_t=\sum_{k=t}^{T}r_k,\qquad G_t=r_t+G_{t+1},\qquad G_{T+1}=0.$$

Dữ liệu offline $\mathcal D$ gồm trajectories và các transitions có observation, action, reward, next observation, terminal, timeout và timestep khi có thể khôi phục đáng tin cậy. Không nối qua biên episode. Phải phân biệt:

- Kết thúc tự nhiên: không có continuation sau terminal.
- Đạt chân trời $H$ của mục tiêu hữu hạn: continuation bằng 0 theo định nghĩa tác vụ.
- Bị cắt do thu thập dữ liệu trước hai điều kiện trên: suffix return là quan sát bị kiểm duyệt, không mặc nhiên là full RTG. Phải loại khỏi nhãn return hoặc công bố cách xử lý trong giao thức.

Dùng một hằng số scale $c>0$ cho reward và return: $\rho_t=r_t/c$, $Y_t=G_t/c$. Mọi return target, output của mạng return, reward prediction và score đều dùng đơn vị này. Normalized benchmark score là đại lượng báo cáo riêng.

### 1.2. DT đóng băng và context

DT được huấn luyện bằng action MSE; tại inference ở chế độ evaluation, đầu ra là một hành động xác định:

$$a_{0,t}=\pi_\theta(h_t,z_t,B_t,t).$$

$z_t=(o_t-\mu_s)/\sigma_s$ là observation chuẩn hóa bằng thống kê training; trong điều kiện clean giả sử $o_t=s_t$. $h_t$ lưu các token quá khứ trong cửa sổ $K$, gồm observation, action và RTG đã thực sự cung cấp cho DT. $B_t$ là token RTG hiện tại. Không giả định observation nhiễu là trạng thái Markov đầy đủ.

Trọng số $\theta$ đóng băng trong toàn bộ so sánh chính; không cập nhật DT hay module phụ bằng rollout đánh giá. “Plug-and-play” chỉ có nghĩa không huấn luyện lại DT, **không** có nghĩa không cần dữ liệu hoặc huấn luyện module phụ.

Khi xét giá trị của chính sách, dùng context mở rộng $x_t$ chứa trạng thái môi trường, history, thời gian và trạng thái ngân sách/controller đủ để xác định diễn tiến. Không mặc nhiên viết giá trị của DT phụ thuộc lịch sử thành một hàm chỉ của $s_t$.

### 1.3. Câu hỏi trung tâm

Trên cùng DT checkpoint, hiệu chỉnh RTG và lựa chọn hành động cục bộ có cải thiện return hoặc robustness với overhead nhỏ không? Thành phần nào tạo lợi ích, khi nào can thiệp làm giảm hiệu năng?

Hai vấn đề khảo sát là target-conditioning mismatch và sai lệch phân phối khi chọn hành động. Không giả định RTG cao luôn gây suy giảm, hoặc disagreement luôn phát hiện OOD. Phạm vi không bao gồm global trajectory stitching, long-horizon planning hay certified safety.

## 2. Phương pháp phiên bản đầu

Có hai **thao tác can thiệp**: R điều chỉnh token RTG; U xếp hạng ứng viên. Chúng dùng mạng return và ensemble dynamics học từ training split. U vẫn dùng mạng return khi R tắt; vì vậy đây không phải hai module học hoàn toàn độc lập.

### 2.1. R — State–time-conditioned RTG cap

Học MLP $g_\psi(z_t,t/H)$ từ nhãn suffix return hợp lệ:

$$\mathcal L_g=\mathbb E_{\mathcal D_{train}}
\left[|\tau-\mathbb I(Y_t-g_\psi(z_t,t/H)<0)|
(Y_t-g_\psi(z_t,t/H))^2\right],\qquad \tau\in(0,1).$$

Upper expectile dùng $\tau>0.5$; $\tau=0.5$ là đối chứng mean regression. Timestep đưa vào để biểu diễn remaining horizon. Mạng chỉ xấp xỉ một thống kê có điều kiện của **return trong dữ liệu**, không phải mặc nhiên $V^{\pi_{DT}}$, $V^*$, quantile hay cận khả thi.

Tách ngân sách yêu cầu $B_t^{raw}$ khỏi token đưa vào DT:

$$B_t=\begin{cases}
\min\{B_t^{raw},g_\psi(z_t,t/H)+\delta\},&R\text{ bật},\\
B_t^{raw},&R\text{ tắt},
\end{cases}\qquad \delta\ge0,$$

$$B_{t+1}^{raw}=B_t^{raw}-\rho_t.$$

$\delta$ có đơn vị scaled return. Không dùng $B_t-\rho_t$ để cập nhật ngân sách trong phiên bản chính. History lưu $B_t$ đã dùng, không ghi đè các token quá khứ bằng ngân sách mới. Quy tắc cập nhật từ token đã clip là một biến thể riêng nếu khảo sát.

**Diễn giải được phép:** cap phụ thuộc state/time nhằm giảm độ nhạy với lựa chọn target. Không gọi là projection vào support hay probabilistic calibration. Expectile nằm ngoài support dữ liệu không đồng nghĩa mọi expected return bằng giá trị đó đều bất khả thi; nó chỉ không chứng nhận khả thi. Đường cong phẳng sau clipping cũng chưa chứng minh return tốt.

### 2.2. Proposal — Một DT forward, nhiều ứng viên cục bộ

Sau một lần tính $a_{0,t}$, tạo tập gồm $N$ hành động, bao gồm $a_{0,t}$:

$$a_t^{(0)}=a_{0,t},\qquad
a_t^{(k)}=\operatorname{clip}_{\mathcal A}
(a_{0,t}+\eta D_a\epsilon_k),\quad
\epsilon_k\sim\mathcal N(0,I),\quad k=1,\ldots,N-1,$$

với $D_a=\operatorname{diag}((a_{max}-a_{min})/2)$ và $\eta\ge0$. Với DT hiện tại, action gốc phải nằm trong biên môi trường; nếu cần adapter về biên, áp dụng đồng nhất cho mọi baseline.

Đây là proposal heuristic $q$, **không** phải lấy mẫu từ một phân phối stochastic do DT MSE học. Action bounds không bảo đảm candidate nằm trong support dữ liệu. Tăng $N$ có thể tăng model exploitation. $N=1$ phải cho đúng hành động DT tương ứng với trạng thái bật/tắt R.

### 2.3. Ensemble dynamics và disagreement

Dùng $M$ MLP độc lập, khởi tạo khác nhau và bootstrap theo trajectory. Mỗi mạng nhận $(z_t,a_t,t/H)$, học dự đoán delta observation chuẩn hóa, scaled reward và xác suất kết thúc tự nhiên:

$$f_{\phi_m}(z_t,a_t,t/H)
=(\widehat{\Delta z}_m,\hat\rho_m,\hat p_m^{term}),\qquad
\hat z'_m=z_t+\widehat{\Delta z}_m.$$

Loss gồm MSE delta/reward và binary cross-entropy terminal, với trọng số ghi trong cấu hình. Timeout do horizon không gán thành natural terminal. Mô hình học trên dữ liệu clean; under observation corruption, dùng input quan sát nhiễu là stress test, không giả định mô hình vẫn đúng. Shared encoder chỉ là ablation sau phiên bản độc lập.

Đặt:

$$\bar z'=\frac1M\sum_m\hat z'_m,\qquad
u(z_t,a,t)=\sqrt{\frac1M\sum_m\|\hat z'_m-\bar z'\|_2^2}.$$

$u$ là **ensemble disagreement proxy**, không phải posterior epistemic uncertainty đã hiệu chuẩn. Nó không bao phủ đầy đủ aleatoric uncertainty, reward error, terminal error hoặc sai số của $g$. Các mạng có thể đồng thuận nhưng cùng sai. Chuẩn hóa state giảm phụ thuộc đơn vị, không làm metric bất biến với mọi phép đổi tọa độ.

### 2.4. U — Heuristic action reranking

Với $t+1<H$, đặt $\hat v_m=g_\psi(\hat z'_m,(t+1)/H)$; khi $t+1\ge H$, đặt $\hat v_m=0$ mà không truy vấn $g$ ngoài horizon.

$$\hat S_t(a)=\frac1M\sum_{m=1}^M
\left[\hat\rho_m+(1-\hat p_m^{term})\hat v_m\right],$$

$$J_t(a)=\hat S_t(a)-\beta u(z_t,a,t),\qquad
 a_t^*=\arg\max_{a\in\mathcal A_c}J_t(a),\qquad\beta\ge0.$$

$\beta$ quy đổi disagreement sang scaled-return penalty; không diễn giải là confidence level. Nếu đồng điểm, ưu tiên action gốc rồi thứ tự candidate cố định. U tắt thì dùng $a_{0,t}$. Ghi action đã chọn vào history; nếu benchmark thêm actuator noise không quan sát được, history lưu action lệnh, không dùng action thật như thông tin đặc quyền.

Lấy trung bình giá trị theo head tránh thay trực tiếp bằng $g(\bar z')$. Tuy nhiên point-prediction ensemble vẫn không tích phân đầy đủ phân phối transition ngẫu nhiên. Tích $(1-\hat p^{term})g(\hat z')$ cũng là xấp xỉ, không mặc nhiên bằng kỳ vọng continuation có điều kiện sống sót.

**Giới hạn quyết định:** $J$ không phải Q-value được bảo đảm. Ngay khi dynamics đúng, expectile return của behavior data có thể không phản ánh khả năng tiếp diễn của DT hoặc controller can thiệp. Disagreement penalty không sửa được sai lệch này một cách tổng quát. Giữ action gốc trong pool cũng không bảo đảm thuật toán chọn hành động tốt hơn.

Không dùng terminal, reward hay next state thật chưa quan sát để xếp hạng candidates. Critic reranking là đối chứng bắt buộc trước khi kết luận ensemble dynamics là lựa chọn có lợi; nó không mặc nhiên là $Q^{\pi_{DT}}$ chính xác.

### 2.5. Chi phí

Phiên bản chính dùng một DT forward; phần thêm gồm $g$ cho cap, tạo candidates, $NM$ dự đoán dynamics và đánh giá $g$ theo head. Batching không loại bỏ lượng công việc $O(NM)$ và không chứng minh overhead dưới một ngưỡng thời gian. Đo riêng tổng latency, overhead, p50/p95, số tham số và memory trên phần cứng xác định.

## 3. Tính đúng đắn và giới hạn lý thuyết

Những công thức trên định nghĩa một thuật toán có thể triển khai; chúng **không** chứng minh thuật toán cải thiện return.

Mệnh đề tham chiếu: với continuation policy $\pi$ cố định, context mở rộng $x$, tập candidates hữu hạn và giả thiết đồng thời

$$|J(x,a)-Q^\pi(x,a)|\le\varepsilon\quad\forall a\in\mathcal A_c,$$

nếu $a^*=\arg\max J$ và $a^\dagger=\arg\max Q^\pi$ trên tập đó, thì

$$Q^\pi(x,a^*)\ge J(x,a^*)-\varepsilon
\ge J(x,a^\dagger)-\varepsilon
\ge Q^\pi(x,a^\dagger)-2\varepsilon.$$

Đây là bound xếp hạng có điều kiện, không phải đóng góp lý thuyết mới hay guarantee cho U-DTVerifier. Phương pháp chưa kiểm soát $\varepsilon$; không được thay nó bằng $\beta u$ mà không có lập luận bổ sung. Bound không tự chứng minh improvement khi controller can thiệp lặp lại ở mọi bước.

Các giới hạn còn mở: support của toàn bộ context; policy/value mismatch; uncertainty chưa hiệu chuẩn; partial observability khi thêm nhiễu; sai số tích lũy và tác động bất lợi của clipping. Không tuyên bố safety, optimality, OOD detection hoặc probabilistic coverage tổng quát.

## 4. Câu hỏi nghiên cứu và bằng chứng cần có

| Câu hỏi | Giả thuyết có thể bác bỏ | Bằng chứng cần có |
| --- | --- | --- |
| RQ1: R có hữu ích không? | State–time cap cải thiện return hoặc độ nhạy target hơn fixed/global/time-only cap trong một số điều kiện | Target curves kèm mức return, failure và clipping frequency; cả clean và inflated target |
| RQ2: U có xếp hạng tốt không? | Scorer cải thiện lựa chọn trong candidate pool; penalty có đóng góp thêm | So với action gốc, random, single model, ensemble không penalty, min-disagreement và critic |
| RQ3: R và U tương tác ra sao? | Hiệu quả reranking thay đổi khi bật cap | Factorial 2×2; interaction effect và CI, không chỉ B5 hơn B0 |
| RQ4: Lợi ích có đáng chi phí? | Có cấu hình cho trade-off return/robustness/latency hữu ích | Return, failure, parameters, memory và latency theo $N,M$ |

Prediction error nhỏ hay tương quan uncertainty–error cao chưa đủ trả lời RQ2. Cần kiểm tra quyết định trong cùng context và tổn thất khi can thiệp sai. Không đặt tăng score 10–20% hay giảm failure 30% thành điều kiện phải đạt; tránh tối ưu báo cáo theo kết quả kỳ vọng.

Mục tiêu kỹ thuật tham khảo là dưới 3 triệu tham số phụ, 500 MB VRAM bổ sung và 5 ms overhead trên RTX 2050 4 GB. Đây là ngân sách thiết kế cần đo, không phải cam kết hoặc lý do che giấu kết quả vượt ngân sách.

## 5. Yêu cầu khoa học và phạm vi kết luận

1. Dùng chung DT checkpoint, preprocessing, context length và dữ liệu được phép cho các biến thể chính; mô tả rõ ngân sách học thêm.
2. Tách trajectory train/validation; thống kê chuẩn hóa chỉ từ train. Công bố model selection và mọi tương tác môi trường dùng để tuning.
3. So sánh cả clean và stress; tách observation corruption, actuator noise và dynamics shift.
4. Dùng nhiều training seeds, báo cáo effect size và uncertainty; không coi evaluation episodes cùng checkpoint là các lần huấn luyện độc lập.
5. Một đường cong phẳng vì clip không đủ chứng minh chất lượng; một penalty tốt trên lỗi tổng thể không đủ chứng minh xếp hạng tốt.
6. Nếu chỉ một thao tác hữu ích, thu hẹp phương pháp và claim. Nếu không hơn heuristic/critic đơn giản, báo cáo và xem xét thay scorer hoặc dừng tích hợp.
7. Không suy rộng từ locomotion sang giao thông, robot thực hay safety khi chưa có bằng chứng riêng.

## 6. Định vị đóng góp

Đóng góp dự kiến là đánh giá có kiểm soát can thiệp nhẹ trên DT đóng băng, xác định thành phần tạo lợi ích và vùng thất bại, cùng chi phí suy luận. Không tuyên bố expectile, ensemble, uncertainty penalty hoặc value guidance là phát minh mới. Benchmark tái lập là mục tiêu chính; mức độ đủ để xuất bản phụ thuộc kết quả và đối chiếu nghiên cứu liên quan.

Cần đối chiếu QDT (relabel RTG khi huấn luyện), UNREST (uncertainty/conditioning), EDT (history selection), QT (value-guided Transformer) và TGDT (context trust cùng critic guidance). Chưa tuyên bố ưu tiên phát minh từ một danh sách tài liệu chưa toàn diện.

## 7. Sản phẩm và quyết định tiếp tục

Sản phẩm gồm code/config tái lập, checkpoints, manifests dữ liệu và seeds, bảng đối chứng, target/stress curves, ranking diagnostics, latency profile và báo cáo failure cases. Trình tự P0–P4, budgets và tiêu chí thu hẹp nằm trong [EXPERIMENT.md](EXPERIMENT.md).

Giữ hướng nếu bằng chứng cho thấy ít nhất một can thiệp có ích so với đối chứng thích hợp. Thay scorer nếu heuristic return không xếp hạng tốt; không suy ra toàn bộ hướng inference intervention sai chỉ vì một scorer thất bại. Không coi thiếu guarantee là bằng chứng hiệu quả âm, cũng không coi benchmark tăng là chứng minh guarantee.

## 8. Truy vết sửa đổi từ phản biện

| Mục trong Mathematical Review | Quyết định hiện hành |
| --- | --- |
| 1–2: support và thời gian | Bỏ “projector”; dùng state–time cap; phân biệt support và expected-return feasibility |
| 3: RTG/discount/scale | $\gamma=1$, chung scale $c$, ngân sách raw riêng, history lưu token đã dùng |
| 4–5: ổn định và value mismatch | Giữ claim thực nghiệm; công bố scorer heuristic và critic đối chứng |
| 6: phi tuyến/terminal | Trung bình score theo head; terminal prediction và horizon rõ; giữ giới hạn stochastic |
| 7–8: uncertainty/safety | Disagreement proxy chuẩn hóa, independent bootstrap, không chứng nhận safety |
| 9: policy improvement | Bound có điều kiện trên context mở rộng; chưa có bound cho sai số thực |
| 10–11: candidates/latency | Proposal Gaussian cục bộ, một DT forward, đo overhead |
| 12–14: metrics và tương tác | Giao thức riêng; interaction là giữa hai thao tác cap/rerank |

## Tài liệu tham khảo

1. Chen et al. (2021). [Decision Transformer](https://arxiv.org/abs/2106.01345).
2. Yu et al. (2020). [MOPO](https://proceedings.nips.cc/paper/2020/hash/a322852ce0df73e204b7e67cbbef0d0a-Abstract.html).
3. Kostrikov et al. (2022). [Implicit Q-Learning](https://arxiv.org/abs/2110.06169).
4. Yamagata et al. (2023). [Q-learning Decision Transformer](https://proceedings.mlr.press/v202/yamagata23a.html).
5. Tang et al. (2024). [Adversarially Robust Decision Transformer](https://proceedings.neurips.cc/paper_files/paper/2024/hash/349a45f211fb1b3850da1ccd829e869e-Abstract-Conference.html).
6. Li et al. (2025). [UNREST](https://proceedings.mlr.press/v270/li25b.html).
7. [Elastic Decision Transformer](https://arxiv.org/abs/2307.02484).
8. Hu et al. (2024). [Q-value Regularized Transformer](https://arxiv.org/abs/2405.17098).
9. Gautam et al. (2026). [Trust Guided Decision Transformer](https://arxiv.org/abs/2609.31586).
10. Farama Foundation. [D4RL](https://github.com/Farama-Foundation/D4RL), [Minari D4RL notes](https://minari.farama.org/datasets/D4RL/index.html).
