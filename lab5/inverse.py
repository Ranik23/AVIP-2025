from PIL import Image
import os

images_dir = 'lab5/images'
inverse_dir = 'lab5/inverse'
alphabet = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'

os.makedirs(inverse_dir, exist_ok=True)

def invert_image(image_path, save_path):
    img = Image.open(image_path)
    img = img.convert('L')
    img = Image.eval(img, lambda x: 255 - x)
    img.save(save_path)

for symbol in alphabet:
    image_path = os.path.join(images_dir, f'{symbol}.png')
    save_path = os.path.join(inverse_dir, f'{symbol}.png')

    if os.path.exists(image_path):
        invert_image(image_path, save_path)
        print(f'Inverted {symbol} saved to {save_path}')
    else:
        print(f'[!] File not found: {image_path}')
