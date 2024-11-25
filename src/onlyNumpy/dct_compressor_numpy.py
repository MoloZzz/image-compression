import numpy as np

def dct_2d(block):
    """Обчислення 2D ДКП через матричні операції."""
    N = block.shape[0]
    basis = np.array([
        [(np.sqrt(1 / N) if u == 0 else np.sqrt(2 / N)) * np.cos((2 * x + 1) * u * np.pi / (2 * N))
         for u in range(N)]
        for x in range(N)
    ])
    return basis @ block @ basis.T


def idct_2d(dct_block):
    """Інверсне 2D ДКП (IDCT) через матричні операції."""
    N = dct_block.shape[0]
    basis = np.array([
        [(np.sqrt(1 / N) if u == 0 else np.sqrt(2 / N)) * np.cos((2 * x + 1) * u * np.pi / (2 * N))
         for u in range(N)]
        for x in range(N)
    ])
    return basis.T @ dct_block @ basis


def compress_dct_numpy(image, threshold=20):
    """Стискує зображення за допомогою DCT через NumPy."""
    h, w = image.shape
    compressed_image = np.zeros_like(image, dtype=np.float32)

    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = image[i:i + 8, j:j + 8]
            dct_block = dct_2d(block)

            # Обнулення малозначущих коефіцієнтів
            dct_block[np.abs(dct_block) < threshold] = 0

            compressed_image[i:i + 8, j:j + 8] = idct_2d(dct_block)

    return np.clip(compressed_image, 0, 255).astype(np.uint8)

