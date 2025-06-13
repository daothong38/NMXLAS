import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def load_image():
    global original_img, image_array
    filepath = filedialog.askopenfilename()
    if filepath:
        original_img = Image.open(filepath).convert('L')
        image_array = np.asarray(original_img)
        display_image(original_img)

def display_image(img):
    img_tk = ImageTk.PhotoImage(img.resize((256, 256)))
    panel.config(image=img_tk)
    panel.image = img_tk

def save_and_show(result_array, title):
    result_img = Image.fromarray(result_array)
    display_image(result_img)
    plt.imshow(result_img, cmap='gray')
    
def inverse():
    result = 255 - image_array
    save_and_show(result, 'inverse')

def gamma_correction():
    gamma = 0.5
    b1 = image_array.astype(float)
    b2 = np.max(b1)
    b3 = (b1+1)/b2
    b4 = np.log(b3) * gamma
    c = np.exp(b4) * 255.0
    result = c.astype(np.uint8)
    save_and_show(result, 'gamma')

def log_transform():
    b1 = image_array.astype(float)
    b2 = np.max(b1)
    c = (128.0 * np.log(1 + b1))/np.log(1 + b2)
    result = c.astype(np.uint8)
    save_and_show(result, 'log')

def histogram_equalization():
    result = cv2.equalizeHist(image_array)
    save_and_show(result, 'hist_eq')

def contrast_stretching():
    min_val = np.min(image_array)
    max_val = np.max(image_array)
    result = ((image_array - min_val) / (max_val - min_val)) * 255
    result = result.astype(np.uint8)
    save_and_show(result, 'contrast')

def fast_fourier():
    f = np.fft.fft2(image_array)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    result = np.asarray(magnitude_spectrum, dtype=np.uint8)
    save_and_show(result, 'fft')

def butterworth_lowpass():
    rows, cols = image_array.shape
    crow, ccol = rows // 2 , cols // 2
    D0 = 30
    n = 2
    
    dft = np.fft.fft2(image_array)
    dft_shift = np.fft.fftshift(dft)

    mask = np.zeros((rows, cols), np.float32)
    for u in range(rows):
        for v in range(cols):
            D = np.sqrt((u - crow)**2 + (v - ccol)**2)
            mask[u,v] = 1 / (1 + (D / D0)**(2*n))

    filtered = dft_shift * mask
    result = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered)))
    result = np.asarray(result, dtype=np.uint8)
    save_and_show(result, 'butter_low')

def butterworth_highpass():
    rows, cols = image_array.shape
    crow, ccol = rows // 2 , cols // 2
    D0 = 30
    n = 2
    
    dft = np.fft.fft2(image_array)
    dft_shift = np.fft.fftshift(dft)

    mask = np.ones((rows, cols), np.float32)
    for u in range(rows):
        for v in range(cols):
            D = np.sqrt((u - crow)**2 + (v - ccol)**2)
            mask[u,v] = 1 / (1 + (D0 / D)**(2*n)) if D != 0 else 0

    filtered = dft_shift * mask
    result = np.abs(np.fft.ifft2(np.fft.ifftshift(filtered)))
    result = np.asarray(result, dtype=np.uint8)
    save_and_show(result, 'butter_high')

# Setup GUI
os.makedirs('exercise', exist_ok=True)
root = tk.Tk()
root.title("Image Processing Menu")
btn = tk.Button(root, text="Load Image", command=load_image)
btn.pack()

panel = tk.Label(root)
panel.pack()

btn_frame = tk.Frame(root)
btn_frame.pack(pady=10)

# Buttons for each operation
tk.Button(btn_frame, text="Image Inverse", command=inverse).grid(row=0, column=0, padx=5, pady=5)
tk.Button(btn_frame, text="Gamma Correction", command=gamma_correction).grid(row=0, column=1, padx=5, pady=5)
tk.Button(btn_frame, text="Log Transformation", command=log_transform).grid(row=0, column=2, padx=5, pady=5)
tk.Button(btn_frame, text="Histogram Equalization", command=histogram_equalization).grid(row=1, column=0, padx=5, pady=5)
tk.Button(btn_frame, text="Contrast Stretching", command=contrast_stretching).grid(row=1, column=1, padx=5, pady=5)
tk.Button(btn_frame, text="Fast Fourier", command=fast_fourier).grid(row=2, column=0, padx=5, pady=5)
tk.Button(btn_frame, text="Butterworth Lowpass", command=butterworth_lowpass).grid(row=2, column=1, padx=5, pady=5)
tk.Button(btn_frame, text="Butterworth Highpass", command=butterworth_highpass).grid(row=2, column=2, padx=5, pady=5)

root.mainloop()