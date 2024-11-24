import matplotlib.pyplot as plt


def show_results(original, dct_compressed, wavelet_compressed, psnr_dct, psnr_wavelet):
    """Відображає оригінал, результати стиснення та PSNR."""
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Оригінальне зображення")
    plt.imshow(original, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title(f"DCT (PSNR: {psnr_dct:.2f} dB)")
    plt.imshow(dct_compressed, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title(f"Вейвлет (PSNR: {psnr_wavelet:.2f} dB)")
    plt.imshow(wavelet_compressed, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
