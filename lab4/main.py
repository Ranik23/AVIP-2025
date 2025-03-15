import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


kernel_gx = np.array([[6, 0, -6],
                      [0, 0, 0],
                      [-6, 0, 6]], dtype=np.float32)

kernel_gy = np.array([[-6, 0, 6],
                      [0, 0, 0],
                      [6, 0, -6]], dtype=np.float32)

input_folder = 'pictures_src'
output_folder = 'pictures_results'


if not os.path.exists(output_folder):
    os.makedirs(output_folder)

image_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]

def apply_kernel(image, kernel):
    kernel_size = kernel.shape[0]
    pad_size = kernel_size // 2
    padded_image = np.pad(image, ((pad_size, pad_size), mode='constant'))
    output = np.zeros_like(image, dtype=np.float32)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output[i, j] = np.sum(padded_image[i:i+kernel_size, j:j+kernel_size] * kernel)
    
    return output


for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)

    image = cv2.imread(image_path)

    if image is None:
        print(f"Ошибка: Не удалось загрузить изображение {image_file}. Пропускаем.")
        continue

    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gradient_x = cv2.filter2D(gray_image, cv2.CV_32F, kernel_gx)
    gradient_y = cv2.filter2D(gray_image, cv2.CV_32F, kernel_gy)

    gradient = np.sqrt(gradient_x ** 2 + gradient_y ** 2)

    if np.isnan(gradient).any() or np.isinf(gradient).any():
        gradient = np.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)

    gradient = gradient.astype(np.float32)

    gradient_normalized = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    _, binary_gradient = cv2.threshold(gradient_normalized, 50, 255, cv2.THRESH_BINARY)

    base_name = os.path.splitext(image_file)[0]
    cv2.imwrite(os.path.join(output_folder, f'{base_name}_gray.png'), gray_image)
    cv2.imwrite(os.path.join(output_folder, f'{base_name}_gradient_x.png'), gradient_x)
    cv2.imwrite(os.path.join(output_folder, f'{base_name}_gradient_y.png'), gradient_y)
    cv2.imwrite(os.path.join(output_folder, f'{base_name}_gradient_normalized.png'), gradient_normalized)
    cv2.imwrite(os.path.join(output_folder, f'{base_name}_binary_gradient.png'), binary_gradient)

    print(f"Обработано изображение: {image_file}")

print("Обработка всех изображений завершена.")