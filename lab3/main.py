import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

# эрозия + дилатация
def apply_morphological_opening(image, kernel_size=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    eroded = cv2.erode(image, kernel, iterations=1)
    opened = cv2.dilate(eroded, kernel, iterations=1)
    return opened

def process_images(input_dir='pictures_src', output_dir='pictures_results'):

    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.bmp', '.png')):
            img_path = os.path.join(input_dir, filename)
            
            img = cv2.imread(img_path, cv2.IMREAD_ANYCOLOR)
            
            # Конвертация в полутоновое для цветных BMP
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img

            # Морфологическая обработка
            processed = apply_morphological_opening(gray)
            
            # Разностное изображение
            diff = cv2.absdiff(gray, processed)
            
            base_name = os.path.splitext(filename)[0]
            cv2.imwrite(os.path.join(output_dir, f'opened_{base_name}.bmp'), processed)
            cv2.imwrite(os.path.join(output_dir, f'diff_{base_name}.bmp'), diff)

            # Дополнительная обработка цветных изображений
            if len(img.shape) == 3:
                processed_color = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
                blended = cv2.addWeighted(img, 0.7, processed_color, 0.3, 0)
                cv2.imwrite(os.path.join(output_dir, f'color_{base_name}.bmp'), blended)

if __name__ == "__main__":
    process_images()