# UPDATE: Đưa implicit aspect vào Step 4 Multi-task Training

> **Ngày:** 2026-04-22  
> **Ảnh hưởng:** Step 1 (data pipeline) + Step 4 (ABSA training)  
> **Độ phức tạp:** Thấp — chỉ thay đổi data pipeline, không đổi kiến trúc model

---

## Vấn đề

Trong PLAN.md hiện tại, các mẫu implicit aspect (`target="NULL"`) bị **loại hoàn toàn** khỏi Step 4:

- Step 1: skip record có `target=NULL` → không tạo BIO record
- Step 4: chỉ train trên explicit records
- **Hệ quả:** Sentiment head mất ~25% dữ liệu training, đặc biệt mất các pattern diễn đạt gián tiếp (ví dụ: "It's overpriced", "We were not satisfied")

---

## Giải pháp

Đưa implicit records vào training Step 4, set toàn bộ BIO label = `-100` để BIO head tự động bỏ qua (cơ chế `ignore_index=-100` của PyTorch).

- **Mẫu explicit:** cả BIO head lẫn Sentiment head đều học → `Loss = L_bio + 0.5 * L_cls`
- **Mẫu implicit:** chỉ Sentiment head học, BIO head bỏ qua → `Loss = 0 + 0.5 * L_cls`

Không cần sửa kiến trúc model, không cần sửa loss function.

---

## Ảnh hưởng

| Khía cạnh | Ảnh hưởng |
|---|---|
| Sentiment cho implicit | **Tốt hơn** — model học thêm pattern diễn đạt gián tiếp |
| Sentiment cho explicit | **Tốt hơn** — thêm dữ liệu giúp generalize tốt hơn |
| BIO cho explicit | **Không đổi** — implicit sample có BIO loss = 0, không ảnh hưởng gradient BIO head |
| BIO cho implicit | **Đúng** — predict toàn O (vì không có aspect term), đây là kết quả expected |

---

## Hạn chế còn lại

Dù đưa implicit vào training, hệ thống vẫn **không thể tự phát hiện** một câu mới nói về aspect category nào (aspect category vẫn phải được cung cấp sẵn trong input). Để giải quyết hoàn toàn, cần thêm **Aspect Category Detection head** — nằm ngoài phạm vi MVP.
