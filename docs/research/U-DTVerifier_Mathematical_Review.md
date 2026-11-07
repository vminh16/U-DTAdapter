# PHẢN BIỆN, CHỨNG MINH VÀ KIỂM ĐỊNH TOÁN HỌC CHO U-DTVerifier

**Loại tài liệu:** Mathematical Review & Research Critique  
**Phạm vi:** Đánh giá tính hợp lệ của các luận điểm trong `U-DTVerifier_RRD.md`, **không sửa công thức trong RRD**.  
**Mục đích:** Phân biệt mệnh đề có thể chứng minh, giả thuyết cần đo và khẳng định hiện không có cơ sở; đề xuất giả thiết đủ để phát triển lý thuyết trong phiên bản sau.

> **Ghi chú phiên bản:** Tài liệu này lưu phản biện đối với RRD trước lần chỉnh sửa tách giao thức thực nghiệm. [RRD hiện hành](U-DTVerifier_RRD.md) đã điều chỉnh công thức và claim; [EXPERIMENT.md](EXPERIMENT.md) quy định giao thức kiểm chứng. Các phát biểu “RRD dùng”, bảng mục 13 và đề xuất “chưa áp dụng” bên dưới được giữ trong ngữ cảnh lịch sử, không mô tả toàn bộ phiên bản hiện hành. Xem bảng truy vết ở mục 8 của RRD để biết cách xử lý từng phản biện.

## 0. Phương pháp phản biện

Một mệnh đề được xếp vào một trong ba loại:

- **Đúng theo định nghĩa/có chứng minh:** Suy ra trực tiếp từ giả thiết đã công bố.
- **Đúng có điều kiện:** Chỉ đúng nếu bổ sung giả thiết xác định; cần kiểm nghiệm giả thiết.
- **Chưa được bảo đảm/sai ở dạng tổng quát:** Tồn tại phản ví dụ; chỉ nên coi là giả thuyết thực nghiệm.

Ký hiệu: $D$ là tập dữ liệu quỹ đạo, $G_t=\sum_{k=t}^{T}r_k$ là RTG *không chiết khấu* trong RRD; $h_t$ là lịch sử của DT; $P_D$ là phân phối dữ liệu; $Q^\pi(s,a)$ là giá trị khi thực hiện $a$ rồi tiếp tục theo $\pi$; $\hat P_m$ là dynamics model thứ $m$.

## 1. Mệnh đề: “Upper expectile là cận trên khả thi trong phân phối”

**Phán quyết: Sai nếu hiểu là support bound hoặc bảo đảm tính khả thi.**

### 1.1. Phản ví dụ chính xác

Tại một trạng thái cố định $s$, cho phân phối RTG:

$$P(G=0\mid s)=P(G=100\mid s)=\tfrac12.$$

Với $g\in[0,100]$, hàm mất mát expectile trở thành:

$$L(g)=\tfrac12(1-\tau)g^2+\tfrac12\tau(100-g)^2.$$

Lấy đạo hàm:

$$L'(g)=(1-\tau)g-\tau(100-g)=g-100\tau.$$

Do $L''(g)=1>0$, nghiệm tối ưu duy nhất là:

$$\boxed{g_\tau(s)=100\tau.}$$

Với $\tau=0.9$, $g_{0.9}(s)=90$, nhưng

$$\operatorname{supp}(G\mid s)=\{0,100\};\qquad90\notin\operatorname{supp}(G\mid s).$$

Nếu $\hat R_t\ge 90$ và $\delta=0$, phép clip của RRD trả về $90$, **một giá trị không thuộc support**. Do đó:

$$\tilde R_t\le g_\psi(s_t)+\delta\quad\not\Rightarrow\quad
(s_t,\tilde R_t)\in\operatorname{supp}(P_D).$$

### 1.2. Vấn đề support của *toàn bộ context*

Ngay cả khi $\tilde R_t$ nằm trong support biên $P_D(G\mid s)$, DT cần phân phối có điều kiện theo $h_t$:

$$\pi_\theta(a_t\mid h_t,\tilde R_t).$$

Có thể tồn tại hai lịch sử $h^{(1)},h^{(2)}$ cùng đi đến trạng thái $s$, nhưng chỉ $h^{(1)}$ có RTG cao. Khi đó một RTG cao hợp lệ theo $P_D(G\mid s)$ vẫn có thể không hợp lệ theo $P_D(G\mid h^{(2)},s)$.

**Kết luận:** Tên gọi *return calibration* là phù hợp; tên *projection onto feasible support* chỉ có nghĩa nếu xây dựng riêng tập khả thi $\mathcal F(h,s)$ và chứng minh $\tilde R\in\mathcal F$.

**Cách kiểm định:** Đo độ lệch của $(h_t,\tilde R_t)$ so với phân phối ngữ cảnh training, đồng thời so sánh global cap, state-conditioned cap và target dữ liệu.

## 2. Mệnh đề: “Chỉ cần $g_\psi(s_t)$, không cần thời gian hoặc lịch sử”

**Phán quyết: Không đúng nói chung trong tác vụ có chân trời hữu hạn.**

Xét MDP chỉ có một trạng thái $s$, phần thưởng luôn bằng $1$, episode có $T$ bước. Khi đó trạng thái ở mọi thời điểm giống hệt nhau, nhưng:

$$G_t=T-t+1.$$

Đặc biệt, $G_1=T$ và $G_T=1$. Bất kỳ hàm $g(s)$ nào cũng phải xuất **cùng một số** ở đầu và cuối episode; nó không thể biểu diễn đồng thời hai target phù hợp theo thời gian.

Thực tế trạng thái MuJoCo có thể không biểu diễn “số bước còn lại”; nhiều RTG cao ở gần đầu episode và thấp ở gần cuối.

**Giả thiết đủ:** Nếu horizon vô hạn, giá trị được chiết khấu, trạng thái Markov đầy đủ và dynamics/reward dừng theo thời gian, $V(s)$ không cần $t$. Nhưng RRD học RTG hữu hạn không chiết khấu nên không thỏa trực tiếp điều kiện này.

**Hướng sửa sau này:** $g(s_t,t)$, $g(s_t,T-t)$ hoặc $g(h_t)$; so sánh thực nghiệm trước khi tăng độ phức tạp.

## 3. Mệnh đề: “RTG trong DT và giá trị một bước trong $J$ nhất quán”

**Phán quyết: Chỉ nhất quán trong những điều kiện nhất định.**

RRD định nghĩa:

$$G_t=\sum_{k=t}^{T}r_k\quad\Longrightarrow\quad G_{t+1}=G_t-r_t.$$

Đây là **undiscounted RTG**. Nhưng hàm lựa chọn sử dụng $r_t+\gamma g(s_{t+1})$ với $\gamma\in(0,1]$.

Nếu muốn cả hai biểu diễn cùng loại return, có hai phương án:

1. Giữ RTG không chiết khấu và dùng $\gamma=1$ trong score một bước.
2. Định nghĩa discounted RTG $G_t^\gamma=r_t+\gamma G_{t+1}^\gamma$. Lúc này phép cập nhật đúng là:

$$G_{t+1}^\gamma=\frac{G_t^\gamma-r_t}{\gamma},\qquad\gamma>0.$$

Không được dùng $G_{t+1}=G_t-r_t$ cho discounted return khi $\gamma\ne1$.

Ngoài ra, nếu DT dùng reward scale $c$, cần đưa $g$, $\bar r$ và RTG về cùng đơn vị. Nếu không, $J$ trộn những đại lượng có thang đo không tương thích.

**Vấn đề cập nhật sau clipping:** Hai quy tắc khác nhau là (i) giữ ngân sách gốc $G_{t+1}^{raw}=G_t^{raw}-r_t$ rồi clip mỗi bước, và (ii) lấy $\tilde G_t-r_t$ làm ngân sách bước sau. Chúng tạo ra các chuỗi RTG khác nhau; RRD cần chốt hoặc ablate thay vì vô tình trộn lẫn.

## 4. Mệnh đề: “Clipping bảo đảm DT ổn định khi target rất cao”

**Phán quyết: Không suy ra từ công thức clip.**

Phép $\min$ chỉ chứng minh được:

$$\tilde R_t\le g_\psi(s_t)+\delta.$$

Không tồn tại suy luận chung:

$$\tilde R_t\le\text{cap}\quad\Longrightarrow\quad
\text{return cao hoặc policy ổn định}.$$

### Phản ví dụ về chính sách

Xét policy quyết định $a=+1$ khi $R\ge R_0$ và $a=-1$ khi $R<R_0$. Nếu giá trị clip rơi từ phía trên sang phía dưới $R_0$, chỉ thay đổi RTG một lượng rất nhỏ cũng đổi toàn bộ hành động. Không có giả thiết liên tục/Lipschitz đối với $\pi_\theta$ thì không thể ràng buộc độ lệch hành động.

### Phản ví dụ về động lực học

Ngay cả khi sai lệch hành động chỉ là $\varepsilon$ ở mỗi bước, hệ:

$$x_{t+1}=2x_t+a_t$$

có thể khuếch đại sai lệch quỹ đạo theo $\varepsilon(2^t-1)$. Không có giả thiết ổn định của dynamics và chính sách thì sai số hành động nhỏ **không bảo đảm** return ổn định.

Chiều ngược lại cũng không đúng: đặt RTG cao hơn dataset không nhất thiết làm DT suy giảm. Nghiên cứu DT gốc ghi nhận một số trường hợp ngoại suy return tốt hơn mức cao nhất trong dữ liệu. Vì vậy H1 phải đo theo từng môi trường, không được coi là quy luật phổ quát [1].

**Hướng chứng minh hợp lý:** Nếu $\pi$ Lipschitz theo RTG, có thể chặn *độ lệch hành động cục bộ* khi thay RTG; để chặn sai lệch return còn cần giả thiết dynamics/reward, horizon và sai số xấp xỉ $g$.

## 5. Mệnh đề: “Expectile của RTG có thể dùng như value function trong $J$”

**Phán quyết: Là heuristic; chưa tương đương $V^\pi$ hoặc $V^*$**.

RRD ước lượng:

$$g_\psi(s)\approx\operatorname{Expectile}_\tau(G\mid S=s),$$

với $G$ thu thập từ nhiều behavior policies. Trong khi một action-value đúng nghĩa phải có dạng:

$$Q^\pi(s,a)=\mathbb E\left[r(s,a)+\gamma V^\pi(S')\mid s,a\right].$$

Nói chung:

$$\operatorname{Expectile}_\tau(G^{\pi_\beta}\mid s)
\ne V^{\pi_{DT}}(s)\ne V^*(s).$$

**Phản ví dụ:** Ở trạng thái $s'$ dataset chứa nhiều quỹ đạo nhận return $100$ vì chọn hành động $a_{good}$, trong khi DT đóng băng khi đến $s'$ thường chọn $a_{bad}$ và nhận return $0$. Mạng $g(s')$ có thể đánh giá rất cao $s'$; nhưng nếu tiếp tục theo DT thì $V^{\pi_{DT}}(s')$ thấp. Khi tối đa $J$, hệ thống có thể ưu tiên hành động dẫn đến $s'$ dù không thể khai thác phần thưởng được kỳ vọng bởi $g$.

**Khác biệt với IQL:** IQL học expectile của $Q(s,a)$ từ hành động trong dataset rồi kết hợp Bellman updates; nó không chỉ lấy expectile của Monte Carlo RTG và gọi đó là value [2].

**Hướng sửa:** (i) Giữ $g$ là return heuristic và đánh giá bằng ablation; hoặc (ii) huấn luyện thêm $V/Q$ với Bellman learning, so sánh trực tiếp. Không nên mặc nhiên nhận bảo đảm policy improvement từ IQL cho cách dùng $g$ hiện tại.

## 6. Mệnh đề: “Có thể thay $\mathbb E[g(S')]$ bằng $g(\mathbb E[S'])$”

**Phán quyết: Sai nói chung đối với $g$ phi tuyến.**

RRD dùng $g_\psi(\bar\mu)$ sau khi lấy trung bình dự đoán trạng thái. Nhưng:

$$g\big(\mathbb E[S']\big)\ne\mathbb E[g(S')].$$

**Phản ví dụ:** Hai dự đoán trạng thái một chiều $S'_1=-1$, $S'_2=1$; lấy $g(x)=x^2$. Khi đó:

$$g\!\left(\frac{-1+1}{2}\right)=0,
\qquad\frac{g(-1)+g(1)}2=1.$$

Sai khác bằng $1$ dù không có nhiễu số. Hơn nữa $\bar\mu$ có thể là một trạng thái trung bình **không xảy ra** trong thực tế.

**Hướng sửa sau này:** Tính $\frac1M\sum_m[\hat r_m+\gamma g(\hat s'_m)]$ thay cho $\bar r+\gamma g(\bar s')$ nếu muốn trung bình hóa giá trị dự đoán. Cũng cần nhân continuation value với $1-\mathrm{terminal}$ trong dữ liệu có kết thúc episode.

## 7. Mệnh đề: “Ensemble disagreement phát hiện chính xác OOD”

**Phán quyết: Không đúng nếu không có giả thiết hiệu chuẩn.**

RRD dùng:

$$u^2(s,a)=\frac1M\sum_m\|\mu_m(s,a)-\bar\mu(s,a)\|^2.$$

### 7.1. Phản ví dụ “đồng thuận nhưng cùng sai”

Giả sử dynamics thật có kỳ vọng $\mu^*(s,a)=0$, còn tất cả mô hình đều dự đoán $\mu_m(s,a)=10$. Khi đó:

$$u^2(s,a)=0,\qquad\|\bar\mu-\mu^*\|^2=100.$$

Do đó:

$$u(s,a)=0\quad\not\Rightarrow\quad\text{prediction error nhỏ}.$$

Shared backbone, bootstrap thiếu đa dạng hoặc sai lệch mô hình giống nhau đều có thể làm hiện tượng này trầm trọng hơn.

### 7.2. Aleatoric không phải epistemic

Về mặt xác suất, với tham số mô hình ngẫu nhiên $\Phi$ theo posterior, ta có phân rã:

$$\operatorname{Var}(S'\mid s,a,D)
=\mathbb E_\Phi[\operatorname{Var}(S'\mid s,a,\Phi)]
+\operatorname{Var}_\Phi(\mathbb E[S'\mid s,a,\Phi]).$$

Số hạng thứ nhất là thành phần ngẫu nhiên nội tại (aleatoric); số hạng thứ hai liên hệ epistemic. Disagreement giữa các **point predictions** chủ yếu nhắm đến số hạng thứ hai và chỉ là một xấp xỉ, không phải hậu nghiệm Bayesian chính xác.

Ví dụ: dynamics ngẫu nhiên cho $S'\in\{-1,+1\}$ đều nhau; mọi mô hình đều dự đoán mean $0$. Disagreement bằng $0$ nhưng uncertainty của trạng thái kế tiếp vẫn bằng $1$.

### 7.3. Phụ thuộc thang đo

Nếu thay hệ tọa độ trạng thái bằng $s'=As$, thì:

$$u_A^2=\frac1M\sum_m\|A(\mu_m-\bar\mu)\|^2,$$

không bất biến với $A$. Chỉ cần nhân một chiều trạng thái với $100$ có thể làm số đo biến thiên lớn hơn rất nhiều. Vì vậy cần chuẩn hóa state và định nghĩa rõ đơn vị của $\beta$.

**Hướng kiểm định:** Đo rank correlation, risk–coverage curve, calibration theo nhóm disagreement và lỗi dự đoán trên cả clean/OOD. Chỉ gọi uncertainty là hữu dụng khi nó có sức dự báo lỗi ở các tác vụ thực nghiệm. COMBO cho thấy sự thiếu tin cậy của uncertainty estimation trong model-based offline RL là một rủi ro có thật [3].

## 8. Mệnh đề: “Phạt $-\beta\sigma$ tự động loại bỏ hành động không an toàn”

**Phán quyết: Sai khi ‘an toàn’ không được định nghĩa trong reward/constraint.**

Hàm điểm của RRD chỉ sử dụng reward, future-return heuristic và model disagreement. Không có biến chi phí an toàn $C(s,a)$, xác suất tai nạn hay điều kiện reachability.

**Phản ví dụ:** Có hai hành động $a_s$ (an toàn) và $a_r$ (rủi ro). Giả sử cả hai được mô hình dự đoán chắc chắn ($\sigma=0$), cùng future value; reward tức thời lần lượt là $1$ và $2$. Khi đó:

$$J(s,a_r)>J(s,a_s),$$

nên thuật toán chọn $a_r$, bất kể một tiêu chuẩn an toàn ngoài reward xem hành động đó nguy hiểm như thế nào.

Ngay cả khi reward phạt tai nạn, rủi ro đuôi phân phối (tail risk) vẫn có thể bị bỏ qua nếu score dùng kỳ vọng.

**Hướng sửa sau này:** Định nghĩa safety cost $C$ và ràng buộc $\mathbb P(C>c_{max})\le\alpha$, hoặc dùng bộ điều khiển fallback với chứng nhận riêng. Trong RRD hiện tại, dùng từ **robustness** và **failure reduction**, không dùng “guaranteed safe”.

## 9. Mệnh đề: “Chọn $\arg\max J$ chắc chắn cải thiện return”

**Phán quyết: Chỉ có một bảo đảm có điều kiện theo sai số score.**

Một mệnh đề nhỏ, chứng minh được và có ích cho hướng nghiên cứu:

> **Mệnh đề (candidate ranking với sai số đều).** Với tập ứng viên hữu hạn $\mathcal A_c$, giả sử tồn tại $\varepsilon\ge0$ sao cho $|J(s,a)-Q^\pi(s,a)|\le\varepsilon$ cho **mọi** $a\in\mathcal A_c$. Gọi $a^*=\arg\max_{a\in\mathcal A_c}J(s,a)$ và $a^\dagger=\arg\max_{a\in\mathcal A_c}Q^\pi(s,a)$. Khi đó:

$$\boxed{Q^\pi(s,a^*)\ge Q^\pi(s,a^\dagger)-2\varepsilon.}$$

**Chứng minh:**

$$Q^\pi(s,a^*)\ge J(s,a^*)-\varepsilon
\ge J(s,a^\dagger)-\varepsilon
\ge Q^\pi(s,a^\dagger)-2\varepsilon.$$

Điều này cũng cho thấy nếu hành động DT gốc thuộc candidate pool thì $Q^\pi(s,a^*)\ge Q^\pi(s,a_{DT})-2\varepsilon$.

**Điểm giới hạn:** Mệnh đề không chứng minh $\varepsilon$ nhỏ. Khi $g$ không phải $V^\pi$, model bias cao, uncertainty chưa hiệu chuẩn hoặc candidates OOD, giả thiết có thể thất bại. Nó cũng chỉ so với **ứng viên tốt nhất**, không phải hành động tối ưu toàn cục.

**Liên hệ sai số dynamics:** Với một fixed policy, reward bị chặn $|r|\le R_{max}$, $0<\gamma<1$, sai số reward đều $\varepsilon_r$ và total variation của transition đều $\mathrm{TV}(P,\hat P)\le\varepsilon_P$, có thể chặn sai số giá trị kiểu simulation lemma:

$$\|V_P^\pi-V_{\hat P}^\pi\|_\infty
\le\frac{\varepsilon_r}{1-\gamma}
+\frac{2\gamma R_{max}\varepsilon_P}{(1-\gamma)^2},$$

với quy ước $\mathrm{TV}(P,Q)=\sup_A|P(A)-Q(A)|$ và các reward/model đều bị chặn thích hợp. **Phác thảo:** Trừ hai phương trình Bellman, chặn sai khác reward bởi $\varepsilon_r$, sai khác kỳ vọng transition bởi $2\varepsilon_P\|V\|_\infty$, sau đó chuyển số hạng $\gamma\|\Delta V\|$ sang vế trái.

RRD **không** có giả thiết hay cơ chế chứng minh $\mathrm{TV}(P,\hat P)$ được khống chế bởi $\sigma_{epi}$; cũng không có bound cho sai số $g$. Vì thế chưa thể áp dụng chặn trên để chứng minh hiệu quả cụ thể của U-DTVerifier. Với $\gamma=1$ và horizon hữu hạn cần một bound hữu hạn theo $T$ thay vì công thức mẫu số $(1-\gamma)$.

## 10. Mệnh đề: “DT liên tục có thể lấy nhiều mẫu action chỉ từ một forward pass”

**Phán quyết: Chỉ đúng nếu đầu ra policy định nghĩa một phân phối có thể lấy mẫu.**

Với DT được học bằng MSE, dự đoán tối ưu theo xác suất có dạng:

$$a_{MSE}(h,R)=\mathbb E[A\mid h,R].$$

**Phản ví dụ:** Trong dataset, $A\in\{-1,+1\}$ với hai mode có xác suất bằng nhau. MSE tối ưu là $a=0$; nhưng $0$ không thuộc support các hành động quan sát. Như vậy một đầu ra xác định của DT không phải mẫu từ $\pi_\theta(a\mid h,R)$.

Nếu sinh $a^{(k)}=a_{DT}+\epsilon_k$ bằng Gaussian noise thì phân phối ứng viên là một **proposal heuristic $q$**, không tự nhiên bằng phân phối DT học được. Nếu tạo ứng viên qua nhiều target RTG hoặc dropout và phải chạy lại Transformer, tuyên bố “một DT forward” có thể không đúng.

**Yêu cầu thực nghiệm:** Khai báo proposal $q$, kích thước noise, ràng buộc action bounds, candidate seed và kiểm tra độ lệch phân phối hành động. Khi so sánh reranking, mọi baseline phải nhận **cùng candidate pool**.

## 11. Mệnh đề: “Single-step batching bảo đảm độ trễ dưới 5 ms và overhead hằng số”

**Phán quyết: Không có bảo đảm định lượng chỉ từ kiến trúc.**

Đối với đúng biểu thức $J$ trong RRD (lấy $g$ một lần trên mean state cho mỗi ứng viên), công việc xấp xỉ mỗi bước là:

$$C_{test}\approx C_{DT}+O(N_{cand}M\,C_f)+O(N_{cand}C_g)+C_{other}.$$

Batching có thể giảm thời gian thực tế nhờ song song hóa, nhưng không triệt tiêu số phép tính và không bảo đảm một kernel GPU duy nhất. Pipeline vẫn có chuỗi phụ thuộc:

$$\text{DT}\rightarrow\text{candidate generation}\rightarrow
\text{dynamics ensemble}\rightarrow g\rightarrow\arg\max.$$

Nếu score được sửa để tính $g$ riêng cho từng dự đoán $\mu_m$, chi phí $g$ có thể tăng thành $O(N_{cand}M C_g)$.

Các yếu tố quyết định latency: GPU/CPU transfers, kích thước batch, kernel launch, dtype, framework, CUDA synchronization, warm-up và khả năng lấy mẫu action mà không lặp Transformer.

**Khuyến nghị chứng cứ:** Đo p50/p95 theo từng thành phần trên RTX 2050 4 GB. Tách **incremental model memory** khỏi VRAM tổng của DT và khỏi VRAM cần để huấn luyện. RO1 nên là target thực nghiệm, không phải theorem.

## 12. Mệnh đề: “Một thước đo OOD và một CFR dùng được cho mọi perturbation/môi trường”

**Phán quyết: Cần định nghĩa lại biến cố và không gian đo.**

### 12.1. OOD là thuộc tính của một phân phối đã chỉ định

Có thể định nghĩa $P_D(s,a)$, $P_D(s)$, $P_D(h,R)$ hoặc phân phối chuyển trạng thái. Chúng không tương đương:

$$s\in\operatorname{supp}(P_D(S))
\not\Rightarrow(s,a)\in\operatorname{supp}(P_D(S,A)).$$

Observation noise tác động lên $o_t$ đầu vào, không nhất thiết thay đổi trạng thái thật $s_t$; action noise thay đổi tín hiệu điều khiển; external force làm thay đổi quy luật/điều kiện chuyển trạng thái. Gom cả ba thành “state OOD” làm giả thuyết thiếu xác định.

### 12.2. CFR phải tương ứng một biến cố

Một định nghĩa hợp lệ là:

$$\mathrm{CFR}=\frac{1}{n}\sum_{i=1}^{n}
\mathbb I\{\text{episode }i\text{ vi phạm tiêu chí failure định trước}\}.$$

Trong Hopper/Walker2d, có thể dùng early unhealthy termination; trong HalfCheetah chuẩn, termination do ngã không phải event mặc định, nên không dùng cùng công thức diễn giải nếu không tạo điều kiện failure riêng.

### 12.3. Normalized score không bị chặn trong $[0,100]$

Với reference scores $R_{random}$ và $R_{expert}$:

$$\mathrm{Score}_{norm}=100\frac{R-R_{random}}{R_{expert}-R_{random}}.$$

Nếu $R>R_{expert}$, score lớn hơn $100$; nếu $R<R_{random}$, score âm. Vậy $0$ và $100$ là **hai mốc tham chiếu**, không phải biên cứng của thước đo [7].

## 13. Kiểm tra trực tiếp các giả thuyết H1–H3 và mục tiêu RO1–RO4

| Mệnh đề trong RRD | Kết luận logic | Bằng chứng tối thiểu cần có |
| --- | --- | --- |
| **H1:** Clip RTG cải thiện ổn định với inflated target | **Empirical, có điều kiện**; không đúng với mọi DT/môi trường | Sensitivity curves; fixed-target, global-clip, state-clip baselines |
| **H2:** Ensemble uncertainty giảm OOD model exploitation | **Empirical, có điều kiện** theo calibration và proposal support | Single model; ensemble $\beta=0$; ensemble $\beta>0$; uncertainty–error diagnostic |
| **H3:** $N_{cand}=20$, $M=5$ đạt latency dưới 5 ms | **Hardware-dependent empirical claim** | GPU synchronization; p50/p95; benchmark ứng viên stochastic vs deterministic |
| **RO1:** Dưới 3M params/500 MB/5 ms | Params kiểm tra tĩnh; VRAM/latency kiểm tra đo lường | Parameter count; peak allocated/incremental memory; timed inference |
| **RO2:** Giữ RTG trong miền khả thi và giữ chính sách ổn định | **Không có guarantee** từ expectile clipping | Đổi claim thành reduction in target sensitivity và đo trực tiếp |
| **RO3:** CFR giảm tối thiểu 30% | Không thể suy ra trước từ mô hình | Nhiều seeds; confidence interval; baseline và định nghĩa failure rõ |
| **RO4:** Score tăng 10–20% | Là mục tiêu, không phải chứng minh | Ablation cùng checkpoint; mean/CI; clean và stress |

## 14. Tương tác R và U: cơ sở toán học cho một đóng góp nghiên cứu

Gọi $P_{r,u}$ là performance khi bật/tắt module R và U với $r,u\in\{0,1\}$. Hiệu ứng tương tác:

$$\boxed{\Delta_{int}=P_{1,1}-P_{1,0}-P_{0,1}+P_{0,0}.}$$

Đây là *difference-in-differences* của thiết kế factorial 2×2.

- $\Delta_{int}>0$: Lợi ích của U lớn hơn khi đã có R (trên metric và môi trường đang xét).
- $\Delta_{int}=0$: Không quan sát thấy hiệu ứng tương tác trung bình.
- $\Delta_{int}<0$: Có hiệu ứng cản trở hoặc lợi ích U nhỏ hơn khi đã có R.

**Không suy ra tương tác dương từ $P_{1,1}>P_{0,0}$.** Ví dụ nếu $P_{0,0}=50$, $P_{1,0}=65$, $P_{0,1}=65$, $P_{1,1}=70$, hệ đầy đủ tốt hơn gốc 20 điểm nhưng $\Delta_{int}=70-65-65+50=-10$.

Đây là một cách định lượng đóng góp nghiên cứu mạnh hơn câu khẳng định chung “kết hợp hai kỹ thuật tốt hơn”. Cần báo cáo sai số/CI của $\Delta_{int}$ trên các seeds; kết quả có thể đổi dấu theo perturbation và dataset.

## 15. Các thay đổi toán học nên ưu tiên ở phiên bản RRD tiếp theo

> Phần này **chỉ là đề xuất**, chưa được áp dụng vào RRD đi kèm theo yêu cầu của tác giả.

| Ưu tiên | Điều chỉnh đề xuất | Lý do |
| --- | --- | --- |
| **P0** | Đổi “feasible/support projection” thành “state-conditioned RTG calibration” | Expectile không chứng minh support membership |
| **P0** | Chốt $\gamma$, reward scaling, time/terminal semantics | Tránh sai đơn vị, sai recurrence và leakage value tại terminal |
| **P0** | Khai báo proposal $q(a\mid h,R)$ sinh ứng viên | DT MSE không cho stochastic distribution một cách mặc nhiên |
| **P0** | Ghi rõ $J$ là heuristic, không phải Q-value đảm bảo | $g$ từ behavior RTG không Bellman-consistent |
| **P1** | Kiểm tra $g(s,t)$ so với $g(s)$ | RTG hữu hạn phụ thuộc remaining horizon |
| **P1** | Xem xét $\frac1M\sum_m g(\hat s'_m)$ thay vì $g(\bar\mu)$ | Khác biệt phi tuyến, tránh mean state bất khả thi |
| **P1** | Bootstrap/independent ensemble và uncertainty calibration | Disagreement thấp có thể cùng sai |
| **P2** | Dựng score error bound hoặc critic-based variant | Muốn có mệnh đề policy improvement cần kiểm soát $|J-Q^\pi|$ |
| **P2** | Safety cost/constraint riêng nếu muốn nói “safe” | Reward + disagreement không chứng minh safety |

## 16. Kết luận phản biện

Kiến trúc U-DTVerifier là **một đề xuất thực nghiệm hợp lý**, nhưng các công thức hiện tại chưa cho phép suy ra ba kết luận mạnh: (1) RTG được chiếu vào miền khả thi; (2) disagreement phát hiện OOD đáng tin cậy; (3) action reranking cải thiện chính sách hoặc bảo đảm an toàn.

Các phản ví dụ ở Mục 1, 5–8 bác bỏ những kết luận đó **trong dạng không có thêm giả thiết**. Mệnh đề về ranking ở Mục 9 cho thấy một con đường lý thuyết khả thi: phải kiểm soát sai số $J$ đối với giá trị thực của chính sách trên candidate set. Tuy nhiên, với phạm vi cử nhân, bước thích hợp trước mắt là kiểm chứng các giả thiết nền tảng bằng ablation và error diagnostics, rồi chỉ xây dựng chứng minh ở mức thực sự được dữ liệu và định nghĩa hỗ trợ.

## Tài liệu tham khảo chính

1. Chen, L. et al. (2021). [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://proceedings.neurips.cc/paper_files/paper/2021/hash/7f489f642a0ddb10272b5c31057f0663-Abstract.html). NeurIPS.
2. Kostrikov, I., Nair, A., Levine, S. (2022). [Offline Reinforcement Learning with Implicit Q-Learning](https://arxiv.org/abs/2110.06169). ICLR.
3. Yu, T. et al. (2021). [COMBO: Conservative Offline Model-Based Policy Optimization](https://proceedings.neurips.cc/paper_files/paper/2021/hash/f29a179746902e331572c483c45e5086-Abstract.html). NeurIPS.
4. Yu, T. et al. (2020). [MOPO: Model-based Offline Policy Optimization](https://proceedings.nips.cc/paper/2020/hash/a322852ce0df73e204b7e67cbbef0d0a-Abstract.html). NeurIPS.
5. Tang, X. et al. (2024). [Adversarially Robust Decision Transformer](https://proceedings.neurips.cc/paper_files/paper/2024/hash/349a45f211fb1b3850da1ccd829e869e-Abstract-Conference.html). NeurIPS.
6. Li, Z. et al. (2025). [Uncertainty-Aware Decision Transformer for Stochastic Driving Environments](https://proceedings.mlr.press/v270/li25b.html). CoRL/PMLR.
7. Farama Foundation. [D4RL repository and score normalization](https://github.com/Farama-Foundation/D4RL) và [Minari D4RL notes](https://minari.farama.org/datasets/D4RL/index.html).
8. Gautam, C. et al. (2026, preprint). [Trust Guided Decision Transformer](https://arxiv.org/abs/2609.31586).
