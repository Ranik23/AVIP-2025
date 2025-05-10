import os
import numpy as np
from PIL import Image, ImageDraw
import cv2
from skimage.morphology import closing, square, dilation

SRC_PATH = "lab6/pictures_src/phrase3.bmp"
DST_DIR = "lab6/pictures_results"
os.makedirs(DST_DIR, exist_ok=True)

def to_binary(path: str) -> np.ndarray:
    """Конвертирует изображение в бинарный формат."""
    img = Image.open(path).convert("L")
    arr = np.array(img)
    return (arr < 128).astype(np.uint8)

def preprocess_image(bin_img: np.ndarray) -> np.ndarray:
    """Применяет морфологические операции для улучшения сегментации."""
    # Увеличиваем внимание на вертикальные структуры, используя вертикальное расширение
    vertical_kernel = np.ones((5, 1), np.uint8)
    dilated = cv2.dilate(bin_img, vertical_kernel)  # Увеличиваем вертикальные компоненты
    closed = closing(dilated, square(3))  # Закрытие для соединения частей букв
    return closed

def connected_components(bin_img: np.ndarray) -> list:
    """Находит компоненты связности и возвращает их ограничительные прямоугольники."""
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > 10:  # Отфильтровываем слишком маленькие компоненты
            boxes.append((x, y, x + w, y + h))
    return boxes

def save_letter_profiles(bin_img: np.ndarray, boxes: list):
    """Сохраняет вырезанные изображения символов и их профили."""
    for idx, (x0, y0, x1, y1) in enumerate(boxes):
        patch = bin_img[y0:y1, x0:x1]
        char_img = Image.fromarray((1 - patch) * 255).convert('L')
        bmp_name = f"{idx:02d}.bmp"
        char_img.save(os.path.join(DST_DIR, bmp_name))

        # Горизонтальные и вертикальные профили
        h_prof, v_prof = profiles(patch)
        txt_name = f"{idx:02d}.txt"
        with open(os.path.join(DST_DIR, txt_name), "w", encoding="utf-8") as f:
            f.write("horizontal:\n" + " ".join(map(str, h_prof.tolist())) + "\n")
            f.write("vertical:\n"   + " ".join(map(str, v_prof.tolist())))

def profiles(bin_img: np.ndarray):
    """Возвращает горизонтальные и вертикальные профили."""
    return bin_img.sum(axis=1), bin_img.sum(axis=0)

def draw_boxes(path: str, boxes: list):
    """Рисует ограничивающие прямоугольники для символов."""
    img = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for idx, (x0, y0, x1, y1) in enumerate(boxes):
        draw.rectangle([(x0, y0), (x1, y1)], outline="blue", width=2)
        draw.text((x0, y0 - 15), str(idx), fill="red")
    img.save(os.path.join(DST_DIR, "phrase_boxes.bmp"))

def main():
    """Основная функция."""
    bin_img = to_binary(SRC_PATH)  # Преобразуем изображение в бинарное
    bin_img = preprocess_image(bin_img)  # Применяем морфологию для улучшения сегментации

    boxes = connected_components(bin_img)  # Находим компоненты
    draw_boxes(SRC_PATH, boxes)  # Рисуем компоненты на изображении
    save_letter_profiles(bin_img, boxes)  # Сохраняем изображения символов и их профили

    print(f"Найдено символов: {len(boxes)}")
    print("Порядок символов:")
    for i, box in enumerate(boxes):
        print(f"{i}: {box}")
    print(f"Результаты сохранены в {DST_DIR}")

if __name__ == "__main__":
    main()
