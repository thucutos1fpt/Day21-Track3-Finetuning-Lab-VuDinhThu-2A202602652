# Lab 21 — Evaluation Report

**Họ tên:** Vũ Đình Thư

**MSSV:** 2A202602652

**Ngày:** 2026-10-07
**Tier:** T4 · **Base model:** `unsloth/Qwen3.5-4B` · **GPU thực tế:** Tesla T4 (14.6 GB khả dụng)

Mọi số liệu trong báo cáo được lấy từ output notebook và cần khớp với artefact trong `results/`.

## 1. Setup

| Thành phần | Cấu hình |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt, đầu ra JSON triage 4 trường |
| Train / validation | 225 / 25, seed 42 |
| `max_length` | 1024 — p95 đo được là 98 token |
| `MASK_MODE` | `assistant-only` |
| Epochs / max steps | 2 epochs / 30 steps |


NB1 gợi ý max_length=256 theo p95. Tôi giữ 1024 theo cấu hình tier T4 của lab để có thêm khoảng trống cho dữ liệu dài hơn. Cấu hình này đã chạy hết pipeline trên T4.
Template có giữ khối <think> không? Có. NB1 báo reasoning được giữ lại và an toàn để train trên traces. Tuy nhiên, valid_trace_rate của bản fine-tune ở NB5 là 0.0.
## 2. Mask proof (NB1)

| Kiểm tra | Kết quả |
|---|---:|
| `supervised_fraction` | 0.4149 |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi không nằm trong loss | `true` |


Đoạn được tính loss:
<|im_start|>assistant
<think>

</think>
{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
## 3. Ba baseline (NB2 — đo trước khi train)

| Run | Target | Regression | Format | Latency (ms) |
|---|---:|---:|---:|---:|
| (a) Base + naive prompt | 0.000 | 0.7911 | 0.000 | 3167.5 |
| (b) Base + optimized prompt | 0.765 | 0.7911 | 1.000 | 1045.3 |
| (c) LoRA fine-tune | 0.970 | 0.6556 | 1.000 | 1396.6 |


(b) có thật sự mạnh hơn (a) không? Có. Target tăng từ 0.000 lên 0.765 và format tăng từ 0.000 lên 1.000. Gatekeeper xác nhận prompt (b) chưa bị sửa sau khi đo.
Tôi không sửa OPTIMIZED_PROMPT; giữ nguyên prompt của lab để phép so sánh với baseline được công bằng.
## 4. Giải phẫu cấu hình sai (NB4)

| Run | Vị trí | r | Trainable params | LR | Train loss | Target (NB5 §4) | Train (s) | Peak VRAM (GB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `correct` | all-linear / text-linear | 16 | 32,464,896 | 0.0001 | 0.6270 | 0.970 | 407.4 | 8.78 |
| `attn_only` | q, v | 283 (matched) | 32,456,704 | 0.0001 | 0.5373 | 0.965 | 264.9 | 8.79 |
| `wrong_lr` | all-linear / text-linear | 16 | 32,464,896 | 0.00001 | 1.5702 | 0.000 | 394.0 | 8.78 |
| `qlora` | all-linear / text-linear | 16 | 32,464,896 | 0.0001 | 0.7058 | 0.940 | 464.7 | 3.86 |


4.1 — Vị trí adapter và rank
attn_only có 32,456,704 tham số huấn luyện, gần bằng correct với 32,464,896 tham số; sai lệch nhỏ hơn 5%. Trên tập target, correct đạt 0.970 còn attn_only đạt 0.965, tức correct nhỉnh hơn 0.005. Nhưng train loss của attn_only thấp hơn: 0.5373 so với 0.6270 của correct. Vì vậy thứ tự theo train loss không giống thứ tự theo target. Kết quả cho thấy train loss thấp hơn không đồng nghĩa với điểm đánh giá tốt hơn; vị trí adapter có thể ảnh hưởng, nhưng chênh lệch target trong lần chạy này rất nhỏ.
4.2 — Learning rate
wrong_lr dùng learning rate 1e-5, thấp hơn correct (1e-4) mười lần. Train loss cuối của wrong_lr là 1.5702, cao hơn nhiều so với 0.6270 của correct; log cũng có một số bước `grad_norm` là `nan`. Điểm target và format đều bằng 0.000, nên đây là một cấu hình thất bại rõ ràng. Tuy nhiên, bài học không phải là dùng loss để xếp hạng các run: loss chỉ cho tín hiệu tối ưu hoá trên dữ liệu train, còn target và format mới xác nhận mô hình có thực hiện được hành vi mong muốn trên tập chưa thấy hay không.
4.3 — QLoRA
QLoRA dùng 3.86 GB VRAM, trong khi correct dùng 8.78 GB, tiết kiệm khoảng 4.92 GB, tương đương 56%. Đổi lại, QLoRA mất 464.7 giây để train so với 407.4 giây của correct, và latency là 1753.3 ms so với 1396.6 ms. Điểm target của QLoRA là 0.940, thấp hơn correct ở mức 0.970 nhưng vẫn cao hơn baseline (b) ở mức 0.765. Trên T4, QLoRA có ích khi cần tiết kiệm VRAM; số liệu này cho thấy nó chậm hơn và target thấp hơn một chút, nhưng chưa đủ để kết luận QLoRA luôn không phù hợp với dòng model này.
## 5. Phán quyết (NB5)
Kết quả cổng hồi quy: FAILED
target Δ = +0.205 · regression Δ = -0.136 · valid_trace_rate = 0.0
Fine-tune cải thiện rõ nhiệm vụ chính: target tăng từ 0.765 của baseline (b) lên 0.970, chênh lệch +0.205. Định dạng JSON vẫn hợp lệ ở mức 1.000. Tuy nhiên, điểm regression giảm từ 0.7911 xuống 0.6556, tức giảm khoảng 0.136, vượt xa mức giảm tối đa 0.020 mà cổng cho phép. Vì vậy kết quả tổng thể là FAILED dù mô hình phân loại ticket tốt hơn. valid_trace_rate bằng 0.0 cũng cho thấy trace đầu ra không đạt điều kiện hợp lệ mà phép đánh giá yêu cầu. Kết quả gợi ý rằng dữ liệu fine-tune chuyên cho ticket đã làm mô hình tập trung mạnh vào tác vụ hẹp và giảm khả năng ở các câu hỏi phổ thông. Trước khi cân nhắc triển khai, cần thử thêm dữ liệu replay phổ thông và đánh giá lại cả regression lẫn trace.
## 6. Định tính — bắt buộc có cả ca thua

> **Việc cần hoàn tất trước khi nộp:** bảng dưới phải lấy từ `results/qualitative_comparison.json` hoặc output so sánh tương đương, có cả nhãn đúng, dự đoán baseline (b) và dự đoán fine-tune. Không được suy đoán các trường này từ điểm trung bình.
#	Ticket (rút gọn)	Nhãn đúng	(b) prompt	(c) fine-tune	Nhận xét
1	Điền từ qualitative.json	Điền nhãn	Điền dự đoán	Điền dự đoán	FT thắng/thua theo dữ liệu
2	Điền từ qualitative.json	Điền nhãn	Điền dự đoán	Điền dự đoán	FT thắng/thua theo dữ liệu
3	Điền từ qualitative.json	Điền nhãn	Điền dự đoán	Điền dự đoán	FT thua nếu dữ liệu xác nhận
4	Điền từ qualitative.json	Điền nhãn	Điền dự đoán	Điền dự đoán	FT thua nếu dữ liệu xác nhận
5	Điền từ qualitative.json	Điền nhãn	Điền dự đoán	Điền dự đoán	Theo dữ liệu


Sau khi đối chiếu 5 ví dụ, mô tả mẫu chung của các ca fine-tune thua nếu có; không kết luận chỉ từ điểm ft_score riêng lẻ.
## 7. Kết luận & điều tôi học được

### Kết luận
Trong thí nghiệm này, fine-tune LoRA giúp mô hình làm tốt hơn rõ rệt trên nhiệm vụ phân loại ticket CSKH. Điểm target tăng từ 0.765 của base model với prompt tối ưu lên 0.970; format vẫn đạt 1.000. Tuy vậy, tôi chưa nên triển khai bản fine-tune này như một mô hình dùng chung, vì điểm regression giảm từ 0.7911 xuống 0.6556, vượt ngưỡng suy giảm cho phép. Tốc độ trả lời của fine-tune cũng chậm hơn baseline (b): 1396.6 ms so với 1045.3 ms. Một bài học quan trọng là điểm target tăng không đủ để khẳng định mô hình tốt hơn toàn diện; cần xem đồng thời khả năng giữ năng lực chung, định dạng, độ trễ và trace. Trong các cấu hình, wrong_lr cho thấy learning rate không phù hợp làm kết quả rất kém; attn_only cho thấy train loss thấp hơn chưa chắc target tốt hơn; QLoRA giảm đáng kể VRAM nhưng chậm hơn và target thấp hơn correct một chút. Nếu tiếp tục phát triển, tôi sẽ bổ sung một lượng nhỏ dữ liệu replay phổ thông, kiểm tra nguyên nhân valid_trace_rate=0.0, rồi chạy lại NB5 trước khi quyết định triển khai.
### Ba điều tôi học được
1. Mask NB1 phải được kiểm chứng: lần chạy này chỉ 41.49% token được tính loss, câu trả lời được tính còn câu hỏi bị mask.
2. Train loss không thay thế được điểm đánh giá: attn_only có loss thấp hơn correct nhưng target thấp hơn một chút.
3. Fine-tune có thể cải thiện tác vụ mục tiêu nhưng làm giảm năng lực khác: target tăng 0.205 trong khi regression giảm 0.136.
**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** thêm 1–5% dữ liệu replay phổ thông, kiểm tra trace đầu ra và chạy lại đánh giá để xem có thể giữ điểm target cao mà giảm regression hay không.
## Phụ lục — Thưởng đã làm
- [ ] B1 NB6 merge + hot-swap
- [ ] B2 Dataset miền riêng (data/CUSTOM_DATASET.md)
- [ ] B3 Reasoning-trace collapse
- [ ] B4 Quét rank có kiểm soát
- [ ] B5 Hugging Face Hub — chưa thực hiện
