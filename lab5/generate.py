from PIL import Image, ImageDraw, ImageFont
import os

output_folder = "images"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

letters = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"

font = ImageFont.truetype("Times New Roman.ttf", 52)

for letter in letters:
    img = Image.new('L', (100, 100), color=255)  # белый фон
    draw = ImageDraw.Draw(img)

    # Вычисление размеров текста
    text_bbox = draw.textbbox((0, 0), letter, font=font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    position = ((100 - text_width) // 2, (100 - text_height) // 2)

    draw.text(position, letter, fill=0, font=font)  # черный текст

    # Обрезаем по содержимому
    cropped_img = img.crop(img.getbbox())
    cropped_img.save(f"{output_folder}/{letter}.png")
