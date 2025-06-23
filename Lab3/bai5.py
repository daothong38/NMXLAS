import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import imageio.v2 as iio
from scipy.ndimage import shift, rotate, zoom, gaussian_filter, map_coordinates
import matplotlib.pyplot as plt

class ImageTransformerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Biến đổi ảnh - XLAS")
        self.image = None
        self.result = None

        # GUI components
        ttk.Label(root, text="Chọn ảnh:").grid(row=0, column=0, padx=5, pady=5)
        self.combo = ttk.Combobox(root, values=["kiwi.jpg", "papaya.jpg", "mountain.jpg"], state="readonly")
        self.combo.current(0)
        self.combo.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(root, text="Tải ảnh", command=self.load_image).grid(row=0, column=2, padx=5)

        ttk.Label(root, text="Chọn phép biến đổi:").grid(row=1, column=0, padx=5)
        self.method = ttk.Combobox(root, values=[
            "Tịnh tiến", "Xoay", "Phóng to/Thu nhỏ", "Gaussian Blur", "Hiệu ứng sóng"
        ], state="readonly")
        self.method.current(0)
        self.method.grid(row=1, column=1, padx=5)
        ttk.Button(root, text="Thực hiện", command=self.apply_transform).grid(row=1, column=2, padx=5)

        ttk.Button(root, text="Lưu ảnh", command=self.save_image).grid(row=2, column=1, pady=10)

    def load_image(self):
        file = self.combo.get()
        try:
            self.image = iio.imread(file)
            messagebox.showinfo("Thành công", f"Đã tải ảnh: {file}")
        except:
            messagebox.showerror("Lỗi", "Không thể tải ảnh.")

    def apply_transform(self):
        if self.image is None:
            messagebox.showwarning("Chưa có ảnh", "Vui lòng chọn và tải ảnh trước.")
            return

        choice = self.method.get()

        if choice == "Tịnh tiến":
            dx = self.ask_number("Số pixel dịch X", 30)
            dy = self.ask_number("Số pixel dịch Y", 30)
            self.result = shift(self.image, shift=(dy, dx, 0))

        elif choice == "Xoay":
            angle = self.ask_number("Góc xoay (độ)", 45)
            reshape = messagebox.askyesno("Xoay", "Cho phép reshape (mở rộng ảnh để không bị cắt)?")
            self.result = rotate(self.image, angle, reshape=reshape, mode='reflect')

        elif choice == "Phóng to/Thu nhỏ":
            factor = self.ask_float("Hệ số phóng to (VD: 2.0, 0.5)", 1.5)
            self.result = zoom(self.image, (factor, factor, 1))

        elif choice == "Gaussian Blur":
            sigma = self.ask_float("Giá trị sigma cho làm mờ", 2.0)
            self.result = gaussian_filter(self.image, sigma=(sigma, sigma, 0))

        elif choice == "Hiệu ứng sóng":
            amp = self.ask_float("Biên độ sóng", 10)
            self.result = self.wave_transform(self.image, amp)

        if self.result is not None:
            plt.imshow(self.result.astype(np.uint8))
            plt.axis('off')
            plt.title("Kết quả")
            plt.show()

    def save_image(self):
        if self.result is None:
            messagebox.showinfo("Chưa có kết quả", "Bạn chưa thực hiện phép biến đổi nào.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".jpg")
        if path:
            iio.imwrite(path, self.result.astype(np.uint8))
            messagebox.showinfo("Đã lưu", f"Đã lưu ảnh tại:\n{path}")

    def wave_transform(self, img, amplitude):
        rows, cols, _ = img.shape
        x, y = np.meshgrid(np.arange(cols), np.arange(rows))
        y_wave = y + amplitude * np.sin(x / 20)
        coords = np.array([y_wave, x])
        result = np.zeros_like(img)
        for i in range(3):
            result[..., i] = map_coordinates(img[..., i], coords, order=1, mode='reflect')
        return result

    def ask_number(self, prompt, default):
        return int(simpledialog.askstring("Nhập số nguyên", f"{prompt}:", initialvalue=str(default)))

    def ask_float(self, prompt, default):
        return float(simpledialog.askstring("Nhập số thực", f"{prompt}:", initialvalue=str(default)))

# Setup ứng dụng
if __name__ == "__main__":
    from tkinter import simpledialog
    root = tk.Tk()
    app = ImageTransformerApp(root)
    root.mainloop()
