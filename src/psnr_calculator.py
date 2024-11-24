import numpy as np


def calculate_psnr(original, compressed):
    """Обчислює PSNR між оригінальним та стисненим зображенням."""
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return float('inf')  # Ідеальне співпадіння
    psnr = 10 * np.log10(255**2 / mse)
    return psnr
