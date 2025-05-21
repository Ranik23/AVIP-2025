from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

output_folder = "lab5/images"
os.makedirs(output_folder, exist_ok=True)

letters = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
font_path = "Times New Roman.ttf"
font = ImageFont.truetype(font_path, 200)

for letter in letters:
    # Создаем большой холст
    canvas_size = (300, 300)
    img = Image.new('L', canvas_size, color=255)
    draw = ImageDraw.Draw(img)

    # Центруем текст
    bbox = draw.textbbox((0, 0), letter, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pos = ((canvas_size[0] - text_w) // 2, (canvas_size[1] - text_h) // 2)
    draw.text(pos, letter, font=font, fill=0)

    # Преобразуем в массив и находим границы черных пикселей
    np_img = np.array(img)
    ys, xs = np.where(np_img < 255)
    if xs.size == 0 or ys.size == 0:
        continue  # пустая буква

    left, right = np.min(xs), np.max(xs)
    top, bottom = np.min(ys), np.max(ys)

    # Обрезаем по содержимому
    cropped = img.crop((left, top, right + 1, bottom + 1))

    # Добавляем белую рамку вокруг (например, 10 пикселей)
    padding = 1
    padded_size = (cropped.width + 2 * padding, cropped.height + 2 * padding)
    final_img = Image.new('L', padded_size, color=255)
    final_img.paste(cropped, (padding, padding))

    # Сохраняем
    filename = f"{letter}.png"
    final_img.save(os.path.join(output_folder, filename))
