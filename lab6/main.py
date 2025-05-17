import os
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label, find_objects
from skimage.morphology import square, dilation, closing, rectangle

SRC_PATH = "lab6/pictures_src/phrase3.bmp"
DST_DIR = "lab6/pictures_results"
os.makedirs(DST_DIR, exist_ok=True)

def to_binary(path: str) -> np.ndarray:
    """Конвертирует изображение в бинарный формат."""
    img = Image.open(path).convert("L")
    arr = np.array(img)
    return (arr < 128).astype(np.uint8)

def preprocess_image(bin_img: np.ndarray) -> np.ndarray:
    """Применяет морфологические операции для соединения элементов символов."""
    # Сильное вертикальное расширение для соединения точек, крючков и т.п.
    vertical_dilated = dilation(bin_img, rectangle(10, 1))
    
    # Затем горизонтальное расширение для соединения частей внутри буквы
    fully_dilated = dilation(vertical_dilated, rectangle(1, 5))
    
    # Закрытие — финальный шаг для устранения разрывов
    closed = closing(fully_dilated, square(3))
    return closed

def connected_components(bin_img: np.ndarray) -> list:
    """Находит компоненты связности и возвращает их ограничительные прямоугольники."""
    labeled, _ = label(bin_img)
    objects = find_objects(labeled)
    boxes = []
    for sl in objects:
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        if (x1 - x0) * (y1 - y0) > 10:
            boxes.append((x0, y0, x1, y1))
    return boxes

def profiles(bin_img: np.ndarray):
    """Возвращает горизонтальные и вертикальные профили."""
    return bin_img.sum(axis=1), bin_img.sum(axis=0)

def save_letter_profiles(bin_img: np.ndarray, boxes: list):
    """Сохраняет вырезанные изображения символов и их профили."""
    for idx, (x0, y0, x1, y1) in enumerate(boxes):
        patch = bin_img[y0:y1, x0:x1]
        char_img = Image.fromarray((1 - patch) * 255).convert('L')
        bmp_name = f"{idx:02d}.bmp"
        char_img.save(os.path.join(DST_DIR, bmp_name))

        h_prof, v_prof = profiles(patch)
        txt_name = f"{idx:02d}.txt"
        with open(os.path.join(DST_DIR, txt_name), "w", encoding="utf-8") as f:
            f.write("horizontal:\n" + " ".join(map(str, h_prof.tolist())) + "\n")
            f.write("vertical:\n"   + " ".join(map(str, v_prof.tolist())))

def draw_boxes(path: str, boxes: list):
    """Рисует ограничивающие прямоугольники для символов."""
    img = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for idx, (x0, y0, x1, y1) in enumerate(boxes):
        draw.rectangle([(x0, y0), (x1, y1)], outline="blue", width=2)
        draw.text((x0, y0 - 15), str(idx), fill="red")
    img.save(os.path.join(DST_DIR, "phrase_boxes.bmp"))

def main():
    bin_img = to_binary(SRC_PATH)
    bin_img = preprocess_image(bin_img)
    boxes = connected_components(bin_img)
    draw_boxes(SRC_PATH, boxes)
    save_letter_profiles(bin_img, boxes)

    print(f"Найдено символов: {len(boxes)}")
    print("Порядок символов:")
    for i, box in enumerate(boxes):
        print(f"{i}: {box}")
    print(f"Результаты сохранены в {DST_DIR}")

if __name__ == "__main__":
    main()
