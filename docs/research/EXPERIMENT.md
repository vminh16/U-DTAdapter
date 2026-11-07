# ĐẶC TẢ THỰC NGHIỆM — U-DTVerifier

**Trạng thái:** Giao thức dự kiến, chưa có kết quả. Tài liệu này tách khỏi [RRD](U-DTVerifier_RRD.md): RRD quy định thuật toán và phạm vi claim; tài liệu này quy định cách kiểm chứng. Các cấu hình khởi đầu dưới đây là lựa chọn cho pilot, không phải cấu hình đã tối ưu.

## 1. Mục tiêu và nguyên tắc

Kiểm tra bốn câu hỏi của RRD: ích lợi của RTG cap, chất lượng reranking, tương tác hai thao tác và chi phí. Ưu tiên kết quả có thể tái lập, đối chứng hợp lý và phân tích thất bại; không đặt mức tăng benchmark bắt buộc.

Không chỉnh thuật toán sau khi xem final test rồi tiếp tục gọi tập đó là test độc lập. Mọi thay đổi sau pilot cần ghi lại cùng lý do, cấu hình cũ/mới và phạm vi kết luận. Phân biệt kết quả exploratory với confirmatory.

## 2. Môi trường và dữ liệu

| Giai đoạn | Phạm vi | Mục đích |
| --- | --- | --- |
| Pilot | Hopper Medium | Kiểm tra pipeline và cơ chế, không kết luận tổng quát |
| Benchmark chính | Hopper, Walker2d × Medium, Medium-Replay | Bốn task–dataset; báo cáo tất cả |
| Bổ sung | HalfCheetah; Medium-Expert | Chỉ sau khi benchmark chính ổn định |
| Ngoài phạm vi đầu tiên | SUMO/CityFlow, robot, multi-step planning | Không dùng để mở rộng scope trước khi trả lời RQ chính |

Chọn và khóa một stack dataset–environment–simulator. D4RL gốc hoặc một bản Minari phải ghi ID, phiên bản, wrappers, horizon, action bounds và reference scores. Không giả định dữ liệu tái tạo có thể so trực tiếp với bảng D4RL cũ. Với repo hiện tại, đối chiếu dataset `*-v2` và môi trường Gym `*-v3` trước khi dùng kết quả.

Chia trajectory 90% train / 10% offline validation bằng split seed cố định; lưu danh sách episode IDs. Không chia ngẫu nhiên transitions. DT và module phụ chỉ học trên train trong giao thức chính. Nếu dùng checkpoint ngoài đã học cả validation split, phải khai báo và không dùng split ấy để chứng minh generalization độc lập của toàn hệ.

Lưu `terminated`, `timeout`, timestep và next observation đúng episode. Kiểm tra tính đầy đủ của trajectories và horizon. Với trajectory bị cắt do thu thập dữ liệu, loại các nhãn full-RTG không quan sát được; vẫn có thể dùng transitions hợp lệ cho dynamics. Báo cáo số episode/transition bị loại và cách xác định. Nếu không phân biệt được nguyên nhân cắt, phải ghi hạn chế trước khi diễn giải giá trị return.

Chuẩn hóa observation chỉ từ train. Locomotion phiên bản đầu dùng reward/return scale $c=1000$ thống nhất với DT repo; lưu $\mu_s,\sigma_s,c,H$ cùng checkpoint. Không tái ước lượng thống kê khi evaluation.

## 3. Baselines và công bằng so sánh

| ID | Phương pháp | Vai trò |
| --- | --- | --- |
| B0 | Vanilla DT với target chuẩn được khóa trước test | Đối chứng chính; không cố tình chọn target xấu |
| B1 | DT với initial target chọn từ thống kê training trajectories | Target lựa chọn đơn giản; ngân sách vẫn giảm theo reward |
| B2 | DT + global cap | Kiểm tra có cần mạng phụ thuộc trạng thái/thời gian |
| B3 | DT + R: state–time cap | Hiệu quả riêng của thao tác cap |
| B4 | DT + U: reranking, không cap | Hiệu quả riêng của reranking; vẫn học và dùng $g$ |
| B5 | DT + R + U | Phương pháp kết hợp |
| B6 | Single dynamics model, không penalty | Đóng góp của ensemble |
| B7 | Ensemble với $\beta=0$ | Đóng góp của penalty |
| B8 | Chọn ngẫu nhiên trong cùng candidate pool | Kiểm tra tác động của proposal |
| B9 | DT + time-only cap $g(t/H)$ | Phân biệt lợi ích thời gian với state conditioning |
| B10 | Chọn candidate có disagreement nhỏ nhất | Kiểm tra uncertainty không có return scorer |
| B11 | Offline critic reranking trên cùng candidates | Kiểm tra có cần dynamics scorer |

B2 học global expectile trên nhãn scaled RTG của train, dùng cùng $\tau,\delta$ với B3 trong đối chứng cơ chế. B9 học expectile với đầu vào thời gian; ablation bổ sung dùng state-only $g(z)$. Các baseline được tune riêng trong ngân sách bằng nhau có thể báo cáo thêm, nhưng không trộn với bảng cố định hyperparameter để phân tích cơ chế.

B6–B8, B10–B11 mặc định so với B4 (R tắt). Khi dùng để phân tích B5 phải chạy phiên bản R bật tương ứng và ghi rõ. Chọn trước một critic offline, ví dụ Q từ IQL, khóa thuật toán và training budget sau pilot. Không gọi critic này là $Q^{\pi_{DT}}$ đã hiệu chuẩn. B11 dùng đúng candidates của nhánh tương ứng; không thay DT proposal bằng actor của critic.

**Hai mức candidate parity:**

- Diagnostic trên context cố định: cùng action gốc và cùng candidates chính xác cho mọi scorer được so sánh.
- Closed-loop rollout: dùng cùng initial/environment seeds và cùng noise streams cho proposal. Sau khi các policy chọn action khác nhau, states và candidates sẽ khác; không tuyên bố identical pools xuyên suốt rollout.

Khi so R bật/tắt, action gốc có thể đổi do RTG token; không ép chung candidates vì sẽ loại bỏ một phần hiệu ứng R. Tất cả biến thể giữ nguyên DT weights, context length và preprocessing trong mỗi training seed.

Mốc học thuật bổ sung: IQL, CQL và một DT-guidance gần nhất có code tái lập, ưu tiên đối chiếu TGDT/EDT theo khả năng tái lập. Chúng là contextual benchmarks nếu training/objective khác. Nếu thiếu đối chứng gần nhất, công bố giới hạn và không tuyên bố superiority toàn diện. B11 không thay thế đánh giá full IQL nếu báo cáo IQL như một policy benchmark.

## 4. Cấu hình pilot và model selection

Cấu hình khởi đầu: $\tau=0.9$, $\delta=0$, $M=3$, $N=10$ (gồm action gốc), proposal $\eta=0.05$, $\beta=0.1$ theo đơn vị score trong RRD. Đây là điểm bắt đầu, không phải khuyến nghị đã có bằng chứng.

| Thành phần | Phạm vi khảo sát ưu tiên |
| --- | --- |
| Expectile | $\tau\in\{0.5,0.7,0.9,0.99\}$ |
| Margin cap | $\delta=0$ cho chính; giá trị khác là ablation ghi bằng scaled-return units |
| Penalty | $\beta\in\{0,0.1,0.5,1,5\}$ |
| Ensemble | $M\in\{1,3,5\}$ |
| Candidates | $N\in\{1,5,10,20\}$ |
| Proposal radius | $\eta\in\{0.01,0.05,0.1\}$ |

Không chạy tích Descartes toàn bộ. Tune theo từng giai đoạn rồi khóa; báo cáo số cấu hình đã thử, kể cả thất bại. Điều chỉnh grid nếu thang score không phù hợp chỉ được làm trong pilot và phải ghi lại.

Trước P1, lưu manifest gồm: kiến trúc/activation của $g$ và mỗi head, optimizer, learning rate, batch size, số updates, early stopping/checkpoint rule, trọng số loss delta/reward/terminal, bootstrap IDs và seeds. Trước B11, lưu cấu hình tương đương cho critic. Không coi tài liệu hiện tại là một run configuration hoàn chỉnh khi các trường này chưa được khóa.

Dùng offline validation loss để chọn checkpoint của các supervised modules. Nếu dùng môi trường để chọn target hoặc hyperparameters, mặc định giới hạn **10 cấu hình mỗi baseline trên mỗi task–dataset, 20 episodes mỗi cấu hình** với validation seeds riêng; báo cáo số interactions thật và mọi ngân sách pilot bổ sung. Khóa ngân sách này trước tuning, không cấp thêm riêng cho phương pháp đề xuất sau khi xem test.

Giao thức có tuning qua rollout phải gọi là **offline training with online validation**, không phải strictly offline model selection. Một nhánh strictly offline nếu triển khai phải có quy tắc chọn cấu hình riêng từ dữ liệu và báo cáo riêng.

## 5. Điều kiện đánh giá

### 5.1. Clean và RTG sensitivity

B0 phải có target hợp lý trên clean. B1 dùng target từ percentile/top return training, chọn trong validation theo budget. Mỗi giá trị initial target chỉ thay token/ngân sách ban đầu, không huấn luyện lại DT.

Target grid chính gồm mức tham chiếu từ dữ liệu và các mức $1.5\times,2\times,5\times$ khi return dương và phép nhân có ý nghĩa. Nếu return không phù hợp, định nghĩa trước một grid cộng theo spread của training returns và lưu giá trị số cụ thể. So sánh các biến thể trên cùng grid, không chỉ trên target gây lỗi mạnh nhất cho B0.

Đo return, failure, clipping frequency và token RTG thực tế. Đường cong phẳng khi nhiều targets cùng bị clip về một giá trị không được dùng riêng làm bằng chứng R tốt.

### 5.2. Perturbations

| Điều kiện | Cách áp dụng | Giới hạn diễn giải |
| --- | --- | --- |
| Observation noise | Ở mọi bước, kể cả reset: $o_t=s_t+\alpha\,\sigma_s\odot\epsilon_t$, $\epsilon_t\sim\mathcal N(0,I)$; $\alpha\in\{0,0.01,0.05,0.1\}$ | Policy/module phụ nhận observation nhiễu; true state chỉ dùng cho diagnostic |
| Action noise | Sau chọn action: $a^{exec}=\operatorname{clip}_{\mathcal A}(a^*+\kappa D_a\epsilon_t)$; $\kappa\in\{0,0.01,0.05,0.1\}$ | Phân biệt proposal noise với actuator noise; controller history lưu action lệnh |
| Dynamics shift | Tùy chọn; khóa tham số vật lý hoặc force protocol trước test | Không suy rộng từ observation corruption sang thay đổi vật lý |

Đánh giá từng loại riêng trước, không gộp thành một mức “OOD”. Clean là mức 0. Dùng các RNG streams độc lập cho model training, candidate proposal, perturbation và environment; lưu seeds. Không cập nhật mô hình hoặc normalization khi stress test.

## 6. Diagnostics về mô hình và quyết định

### 6.1. Offline và rollout prediction error

Đo lỗi normalized next observation, scaled reward và terminal prediction trên validation. Với disagreement, báo cáo rank correlation, error theo các bins và risk–coverage curves. Báo cáo riêng clean/perturbed conditions và theo task. Trên môi trường stochastic, lỗi đối với một transition quan sát còn chứa aleatoric noise; không coi đó là nhãn epistemic error thuần túy.

Phân biệt lỗi dự đoán latent clean next state với lỗi quan sát bị corruption. Ghi rõ target nào được đo. Không suy ra OOD detection guarantee từ correlation, cũng không dùng mô hình phụ để tạo ground-truth error cho chính nó.

### 6.2. Counterfactual candidate-ranking diagnostic

Trên tập context cố định, thử candidates trong simulator để đo chất lượng xếp hạng. Đây là **diagnostic có simulator access**, không thuộc inference của phương pháp và không phải dữ liệu huấn luyện thêm.

1. Thu một tập contexts từ policy tham chiếu đã khóa, phủ nhiều timesteps; nếu bổ sung contexts từ policy can thiệp, báo cáo phân phối đó riêng.
2. Lưu đầy đủ simulator state, wrapper/time-limit state, RNG state, DT history, timestep và ngân sách raw. Observation đơn lẻ thường không đủ để reset đúng trạng thái.
3. Cố định candidates; restore snapshot cho mỗi candidate, thực thi action đó một lần rồi tiếp tục bằng **cùng baseline DT controller với quy tắc ngân sách cố định**. History của mỗi nhánh phải chứa action/reward thực tế tương ứng.
4. Ước lượng full-horizon return bằng repeated continuations khi có randomness. Nếu chỉ rollout ngắn, báo cáo đó là short-horizon proxy, không gọi là full $Q^\pi$.
5. Báo cáo probability of improvement, mean advantage so với action gốc, thiệt hại khi can thiệp sai, ranking correlation và regret so với candidate tốt nhất.

Pilot dự kiến 50 contexts/task, $N=10$, 5 continuations/candidate; lưu và công bố ngân sách thật. Ground-truth returns ở đây là Monte Carlo estimates, có sampling error; maximum candidate estimate có thể bị selection bias. Dùng continuations độc lập để kiểm tra candidate được chọn khi đánh giá oracle regret, hoặc nêu rõ đây là ước lượng exploratory có sai lệch.

Diagnostics dùng để phát triển phải lấy validation contexts. Diagnostic cuối dùng contexts/seeds độc lập và không feed back vào tuning. Nếu chưa triển khai snapshot đáng tin cậy, không báo cáo counterfactual ranking như đã kiểm chứng; thu hẹp claim còn closed-loop performance và prediction diagnostics.

## 7. Factorial, chỉ số và thống kê

Factorial bắt buộc gồm B0/B3/B4/B5, với hai yếu tố **cap bật/tắt** và **rerank bật/tắt**. $g$ vẫn tồn tại trong nhánh U-only. Với performance $P$ mà lớn hơn là tốt:

$$\Delta_{int}=P_{1,1}-P_{1,0}-P_{0,1}+P_{0,0}.$$

Tính CI của contrast trên các training-seed groups được ghép cặp. Với failure rate, đổi chiều metric hoặc giải thích dấu riêng. B5 hơn B0 không đủ chứng minh tương tác dương. Interaction phụ thuộc metric và task.

| Nhóm | Báo cáo bắt buộc |
| --- | --- |
| Performance | Raw return, D4RL normalized score với reference values/version |
| Robustness | Return theo noise level, early unhealthy termination; tách terminated/truncated |
| R | Target curves, clipping frequency, mean return đi kèm độ nhạy |
| U | Intervention frequency, candidate action distance, model errors và ranking diagnostic nếu hợp lệ |
| Efficiency | Parameters bổ sung, peak/incremental VRAM, latency mean/p50/p95 |

Normalized score:

$$100\frac{R-R_{random}}{R_{expert}-R_{random}},$$

không bị chặn trong $[0,100]$. Failure ở Hopper/Walker2d dựa vào unhealthy termination đã khai báo, không gộp time limit. HalfCheetah dùng return nếu chưa có định nghĩa failure riêng. Báo cáo giảm failure theo cả percentage points và relative reduction khi mẫu số khác 0.

Pilot có thể dùng một training seed nhưng không đủ kết luận. Benchmark chính dùng ít nhất 3, ưu tiên 5 training seeds; mỗi seed gồm DT và các module phụ với seed mapping được lưu. Mỗi cấu hình test dùng 100 evaluation episodes/seed, cùng seed list giữa các phương pháp. Final-test seeds khác validation seeds.

Báo cáo per-seed means, mean difference so với đối chứng, SD và 95% CI; ghép cặp theo training seed. Nếu bootstrap, resample training-seed groups ở cấp ngoài và episodes ở cấp trong, giữ pairing. Không pool episodes như các lần huấn luyện độc lập. Với ít training seeds, ghi rõ CI còn không ổn định.

Trước final test, khóa primary contrasts: B3 so với B2/B9, B5 so với B0, penalty so với $\beta=0$ trong cùng nhánh. B11 so với scorer là đối chứng phương pháp cần báo cáo. Trình bày toàn bộ bốn task–dataset; nếu chạy nhiều kiểm định, công bố cách xử lý multiplicity hoặc ghi exploratory. Không chọn seed/noise level tốt nhất làm kết luận chung.

## 8. Latency và tài nguyên

Đo trên phần cứng được ghi tên và software versions. Warm-up, dùng `eval` và inference/no-grad mode, synchronize CUDA đúng ranh giới đo. Đo batch size của một controller, bao gồm candidate generation, preprocessing và CPU/GPU transfers thực sự dùng khi điều khiển; tách simulator stepping khỏi model latency.

Báo cáo DT latency, tổng latency adapter và phần tăng thêm; p50/p95, số repetitions và điều kiện đo. Đếm tham số thực được load theo từng baseline. Tách VRAM allocated/reserved nếu đo cả hai, peak inference và training memory. Cấu hình đạt return tốt nhưng vượt mục tiêu 5 ms/500 MB vẫn phải xuất hiện trong báo cáo.

## 9. Các mốc thực thi và quyết định

| Mốc | Công việc | Điều kiện chuyển tiếp |
| --- | --- | --- |
| P0 | Khóa stack, split/seed/checkpoint; tái lập DT và target curves | Score/environment semantics xác minh được; không lấy việc target cao gây lỗi làm điều kiện bắt buộc |
| P1 | B0–B3, B9; state-only ablation nếu cần | Nếu B3 không có lợi ích rõ so với heuristic/time-only, thu hẹp R hoặc ghi kết quả âm |
| P2 | Dynamics và candidate diagnostics; B4, B6–B8, B10–B11 | Xác định scorer có ích hay không; correlation tổng thể chưa đủ để chuyển claim sang ranking |
| P3 | Factorial trên bốn task–dataset, clean và stress | Khóa model selection trước test; báo cáo adverse effects và CI |
| P4 | Efficiency, failure analysis và viết bài | Claim tương ứng đúng evidence; lưu đầy đủ artifacts |

Nếu chỉ R hoặc U hữu ích, paper có thể thu hẹp vào thao tác đó. Nếu penalty không giúp nhưng scorer có ích, không tiếp tục tuyên bố uncertainty là nguyên nhân cải thiện. Nếu critic tốt hơn về performance và chi phí, cân nhắc thay scorer và cập nhật định vị với prior work. Không kết luận “không có hiệu ứng” chỉ từ một pilot ít seeds hoặc một kiểm định không có ý nghĩa thống kê.

## 10. Artifacts và kiểm tra triển khai

Mỗi run lưu config, commit ID, environment/dataset versions và hashes nếu có, split IDs, preprocessing, checkpoint hashes, seed streams, training/validation budget, raw per-episode metrics, failure flags và latency samples. Log thêm cap value, RTG raw/token, action gốc/đã chọn và disagreement để phân tích cơ chế.

Các kiểm tra tính đúng đắn tối thiểu khi triển khai:

- R và U cùng tắt khớp baseline DT; $N=1$ khớp action DT của nhánh R tương ứng.
- Reward/return chung scale, recurrence raw đúng; history giữ token và action đã dùng.
- Continuation bằng 0 tại horizon; terminal khác timeout; không lấy next state thật để chọn action.
- Candidate bounds, inclusion của action gốc và tie-breaking đúng.
- Observation noise áp dụng mỗi bước; candidate noise và actuator noise là hai nguồn riêng.
- Normalization/train split không dùng validation; evaluation seeds và model selection tách biệt.

Repo gốc có state/return prediction heads nhưng action-only training loss; không dùng các heads đó như dynamics/value đã được huấn luyện. `mode='noise'` gốc chỉ thêm nhiễu lúc reset và dataset converter chưa giữ `timeouts`; phải xử lý các điểm này trước khi công bố giao thức đã được triển khai.
