import numpy as np


def haar_transform(block):
    """Виконує вейвлет-перетворення Хаара для блоку."""
    avg = (block[::2, ::2] + block[::2, 1::2] + block[1::2, ::2] + block[1::2, 1::2]) / 4
    hor = (block[::2, ::2] + block[::2, 1::2] - block[1::2, ::2] - block[1::2, 1::2]) / 4
    ver = (block[::2, ::2] - block[::2, 1::2] + block[1::2, ::2] - block[1::2, 1::2]) / 4
    diag = (block[::2, ::2] - block[::2, 1::2] - block[1::2, ::2] + block[1::2, 1::2]) / 4
    return avg, hor, ver, diag

def haar_inverse(avg, hor, ver, diag):
    """Відновлює блок із вейвлет-коефіцієнтів."""
    h, w = avg.shape
    restored = np.zeros((2 * h, 2 * w), dtype=np.float32)
    restored[::2, ::2] = avg + hor + ver + diag
    restored[::2, 1::2] = avg + hor - ver - diag
    restored[1::2, ::2] = avg - hor + ver - diag
    restored[1::2, 1::2] = avg - hor - ver + diag
    return restored

def compress_wavelet_numpy(image, threshold=10):
    """Стискування зображення за допомогою вейвлета Хаара (numpy)."""
    h, w = image.shape
    compressed_image = np.zeros_like(image, dtype=np.float32)

    for i in range(0, h, 2):
        for j in range(0, w, 2):
            block = image[i:i + 2, j:j + 2]
            avg, hor, ver, diag = haar_transform(block)

            # Обнулення низькоінформативних коефіцієнтів
            hor[np.abs(hor) < threshold] = 0
            ver[np.abs(ver) < threshold] = 0
            diag[np.abs(diag) < threshold] = 0

            compressed_image[i:i + 2, j:j + 2] = haar_inverse(avg, hor, ver, diag)

    return np.clip(compressed_image, 0, 255).astype(np.uint8)
