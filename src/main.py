import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.dct_compressor import compress_dct
from src.psnr_calculator import calculate_psnr
from src.visualizer import show_results
from src.wavelet_compressor import compress_wavelet
from src.file_handler import save_image

if __name__ == "__main__":
    # Завантаження зображення
    image = cv2.imread('../images/Preview.png', cv2.IMREAD_GRAYSCALE)

    # Стиснення за допомогою DCT
    dct_compressed = compress_dct(image, threshold=20)
    psnr_dct = calculate_psnr(image, dct_compressed)

    # Стиснення за допомогою Wavelet
    wavelet_compressed = compress_wavelet(image, threshold=20)
    psnr_wavelet = calculate_psnr(image, wavelet_compressed)

    # Відображення результатів
    show_results(image, dct_compressed, wavelet_compressed, psnr_dct, psnr_wavelet)

    # Збереження зображень
    save_image(image, "original.png")
    save_image(dct_compressed, "compressed_dct.png")
    save_image(wavelet_compressed, "compressed_wavelet.png")

    # Збереження графіка результатів
    plt.figure(figsize=(12, 8))

    # Оригінальне зображення
    plt.subplot(1, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')

    # DCT стиснення
    plt.subplot(1, 3, 2)
    plt.imshow(dct_compressed, cmap='gray')
    plt.title(f"DCT Compressed (PSNR: {psnr_dct:.2f} dB)")
    plt.axis('off')

    # Wavelet стиснення
    plt.subplot(1, 3, 3)
    plt.imshow(wavelet_compressed, cmap='gray')
    plt.title(f"Wavelet Compressed (PSNR: {psnr_wavelet:.2f} dB)")
    plt.axis('off')

    # Збереження графіка
    plt.tight_layout()
    plt.savefig("images/comparison_plot.png")
    print("Графік результатів збережено: images/comparison_plot.png")
    plt.show()
