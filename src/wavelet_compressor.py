import numpy as np
import pywt


def compress_wavelet(image, wavelet='haar', level=2, threshold=20):
    coeffs = pywt.wavedec2(image, wavelet=wavelet, level=level)

    # Обнулення малозначущих коефіцієнтів
    coeffs_thresh = []
    for coeff in coeffs:
        if isinstance(coeff, tuple):
            coeffs_thresh.append(tuple(pywt.threshold(c, threshold, mode='hard') for c in coeff))
        else:
            coeffs_thresh.append(pywt.threshold(coeff, threshold, mode='hard'))

    # Відновлення зображення
    compressed_image = pywt.waverec2(coeffs_thresh, wavelet=wavelet)
    return np.clip(compressed_image, 0, 255).astype(np.uint8)
