import numpy as np
from scipy.fftpack import dct, idct
import cv2


def apply_dct(block):
    """Застосовує DCT до 8x8 блоку."""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')


def apply_idct(block):
    """Застосовує IDCT до 8x8 блоку."""
    return idct(idct(block.T, norm='ortho').T, norm='ortho')


def compress_dct(image, threshold=10):
    """Стискує зображення за допомогою DCT."""
    h, w = image.shape
    compressed_image = np.zeros_like(image, dtype=np.float32)

    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = image[i:i + 8, j:j + 8]
            dct_block = apply_dct(block)

            # Обнулення низькоінформативних коефіцієнтів
            dct_block[np.abs(dct_block) < threshold] = 0

            compressed_image[i:i + 8, j:j + 8] = apply_idct(dct_block)

    return compressed_image
