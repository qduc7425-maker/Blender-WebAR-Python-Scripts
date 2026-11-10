# 🌊 Blender 3D WebAR - Mô Hình Phân Tử & Chu Trình Nước (KHTN 6)

Dự án chứa các đoạn mã Python tự động hóa trong Blender (Blender Python API) dùng để tạo lập, mô phỏng vật lý và tối ưu hóa học liệu 3D/WebAR phục vụ giảng dạy môn KHTN 6 (chủ đề: Hiện tượng chuyển thể của chất & Chu trình của nước trong tự nhiên).

---

## 📌 Tính Năng Nổi Bật

- **Mô phỏng 3D Phân tử H2O dán nhãn:** Tự động tạo khối 3D các phân tử nước có đính kèm nhãn chữ nổi **O** (Oxi) và **H** (Hydro) trên bề mặt nguyên tử.
- **Tự động Bake Keyframes & NLA Tracks:** Nướng toàn bộ chuyển động va chạm vật lý (*Rigid Body*) và đóng gói thành **1 Clip hoạt ảnh duy nhất**, chống lỗi đứng yên hoặc xé lẻ clip trên trình duyệt WebAR/3D Viewer.
- **Tối ưu vật liệu kính trong suốt (Alpha Blend):** Khắc phục triệt để lỗi vật liệu bị đục xám khi xuất file `.glb`.
- **Hiệu ứng Hơi nước bốc lên:** Tự động sinh các hạt hơi nước li ti chuyển động tuần hoàn từ mặt biển hướng lên đám mây.

---

## 🛠 Hướng Dẫn Sử Dụng Trong Blender

1. Mở phần mềm **Blender 5.2**.
2. Chuyển sang thẻ **Scripting** ở thanh menu trên cùng.
3. Tạo một tập lệnh mới (`New`), sao chép đoạn mã Python tương ứng trong repository này và dán vào.
4. Bấm **Run Script** (hoặc nhấn `Alt + P`).
5. **Xuất file `.glb` cho WebAR:**
   - Vào `File` > `Export` > `glTF 2.0 (.glb)`.
   - Mục **Animation**: Chọn *Animation Mode* là **`NLA Tracks`** hoặc **`Scene`**.
   - Mục **Materials**: Chọn *Alpha Mode* là **`BLEND`**.

---

## 📄 Ghi Công & Bản Quyền Mô Hình Tham Khảo

Mô hình 3D tham khảo "Water cycle" được sử dụng trong dự án theo giấy phép **Creative Commons Attribution (CC BY)**:
- **Tên mô hình:** [Water cycle](https://sketchfab.com/3d-models/water-cycle-8817dd5ea51c487fae89cce8a4bafda7)
- **Tác giả:** butterflybliss (`@ZorDob`) trên Sketchfab.
