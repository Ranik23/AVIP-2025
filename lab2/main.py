from PIL import Image
import numpy as np
import os

def rgb_to_grayscale(image):

    image_array = np.array(image)
    grayscale_array = np.uint8(0.299 * image_array[:, :, 0] + 0.587 * image_array[:, :, 1] + 0.114 * image_array[:, :, 2])
    return Image.fromarray(grayscale_array)

def adaptive_threshold_nick(image, window_size=3, k=-0.2):

    image_array = np.array(image)
    half_size = window_size // 2
    padded = np.pad(image_array, pad_width=half_size, mode='edge')
    result = np.zeros_like(image_array)
    
    for i in range(image_array.shape[0]):
        for j in range(image_array.shape[1]):
            window = padded[i:i + window_size, j:j + window_size]
            mean = np.mean(window)
            std_dev = np.std(window)
            threshold = mean + k * std_dev
            result[i, j] = 255 if image_array[i, j] > threshold else 0
    
    return Image.fromarray(result)


input_folder = 'pictures_src/'
output_folder = 'pictures_results/'

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith('.png'):
        image_path = os.path.join(input_folder, filename)
        image = Image.open(image_path).convert('RGB')
        
        gray_image = rgb_to_grayscale(image)
        gray_output_path = os.path.join(output_folder, f'grayscale_{filename}')
        gray_image.save(gray_output_path)
        
        binary_image = adaptive_threshold_nick(gray_image, window_size=3)
        binary_output_path = os.path.join(output_folder, f'binarized_nick_{filename}')
        binary_image.save(binary_output_path)
        
        print(f"Обработано: {filename}, сохранено {gray_output_path}, {binary_output_path}")
