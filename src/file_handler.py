import os
import cv2


def save_image(image, filename, folder="images"):
    """
    Зберігає зображення у вказану папку.

    :param image: Масив NumPy, що представляє зображення.
    :param filename: Назва файлу (наприклад, 'compressed_dct.png').
    :param folder: Папка для збереження (за замовчуванням 'images').
    """
    # Перевірка, чи існує папка
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Формуємо шлях до файлу
    filepath = os.path.join(folder, filename)

    # Зберігаємо зображення
    cv2.imwrite(filepath, image)
    print(f"Зображення збережено: {filepath}")
