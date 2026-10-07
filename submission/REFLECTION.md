# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**

Điều làm tôi ngạc nhiên nhất là một bản LoRA có thể tăng điểm target rất mạnh nhưng vẫn không đạt phán quyết cuối cùng. Trong lần chạy này, target tăng từ 0.765 lên 0.970, nhưng regression giảm từ 0.7911 xuống 0.6556. Tôi cũng bất ngờ vì `attn_only` có train loss thấp hơn `correct` (0.5373 so với 0.6270) nhưng target lại thấp hơn một chút. Điều này làm tôi thấy rõ rằng loss không phải là câu trả lời cho câu hỏi mô hình có hữu ích trong thực tế hay không.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**

Tôi mất nhiều thời gian nhất ở phần chạy GPU: tải model, đo hai baseline trước train, rồi chạy ba contrast ở NB4. Ban đầu tôi nghĩ train `correct` mới là phần chậm nhất, nhưng thực tế NB4 tốn hơn vì phải lặp lại quá trình train cho `attn_only`, `wrong_lr` và `qlora` với cùng ngân sách step. Phần này chậm nhưng cần thiết, vì nhờ nó tôi mới so sánh được tác động của vị trí adapter, learning rate và lượng tử hoá một cách công bằng.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**

Trước lab, tôi tin rằng nếu fine-tuning làm train loss giảm và accuracy của task tăng thì mô hình chắc chắn tốt hơn. Tôi không còn tin điều đó nữa. `wrong_lr` và `attn_only` cho thấy các chỉ số trong quá trình train không thay thế được đánh giá trên dữ liệu holdout; kết quả target, format, regression và latency phải được đọc cùng nhau. Tôi cũng không còn cho rằng QLoRA luôn là lựa chọn mặc định: nó tiết kiệm VRAM rõ rệt, nhưng trong lần chạy này chậm hơn và điểm target thấp hơn LoRA 16-bit.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**

Tôi dùng AI assistant để đọc cấu trúc lab, giải thích ý nghĩa của loss mask, kiểm tra thứ tự chạy notebook và hỗ trợ chẩn đoán giới hạn VRAM của máy local. AI cũng giúp tôi diễn giải các số liệu như target delta và regression delta, nhưng nó không thể thay thế artefact thực tế. Ví dụ, AI không được phép tự bịa bảng định tính khi chưa có đủ dữ liệu từ `qualitative.json`; phần đó phải được lấy trực tiếp từ output NB5. Vì vậy tôi dùng AI để hỗ trợ quy trình và diễn giải, còn các kết luận số liệu vẫn phải dựa trên file `results/`.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**

Bước đầu tiên của tôi là xác định tiêu chí thành công và đóng băng một tập đánh giá đại diện trước khi train. Tôi sẽ đo base model với một prompt đã được tối ưu để biết fine-tuning có thật sự cần thiết hay không, sau đó kiểm tra chất lượng nhãn, định dạng đầu ra, các tình huống biên và một tập regression phản ánh năng lực không được phép mất. Với kết quả lab này, tôi còn sẽ chuẩn bị dữ liệu replay phổ thông ngay từ đầu để giảm rủi ro mô hình tăng điểm ở task hẹp nhưng suy giảm quá nhiều ở các yêu cầu khác.
