import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.feature import graycomatrix, graycoprops

# Директории
src_directory = 'lab8/pictures_src'
result_directory = 'lab8/pictures_results'

# Создание директории для результатов, если её нет
os.makedirs(result_directory, exist_ok=True)

# Функция для преобразования яркости
def contrast_image(image):
    hsl = color.rgb2hsv(image)
    luminance = hsl[:, :, 2]
    min_val = luminance.min()
    max_val = luminance.max()
    hsl[:, :, 2] = (luminance - min_val) / (max_val - min_val)
    return color.hsv2rgb(hsl), hsl[:, :, 2], luminance  # возвращаем также L до и после

# Функция для вычисления GLCM и визуализации
def compute_glcm(gray_image, filename, suffix):
    gray_uint8 = (gray_image * 255).astype(np.uint8)
    glcm = graycomatrix(gray_uint8, distances=[1], angles=[0, np.pi/2, np.pi, 3*np.pi/2], 
                        symmetric=True, normed=True)
    corr = graycoprops(glcm, 'correlation')

    for i, angle in enumerate(['0', '90', '180', '270']):
        matrix = glcm[:, :, 0, i]
        matrix_log = np.log1p(matrix)  # логарифмическое нормирование
        plt.imshow(matrix_log, cmap='gray')
        plt.title(f'GLCM {angle}° {suffix}')
        plt.colorbar()
        plt.savefig(os.path.join(result_directory, f'{filename}_glcm_{angle}_{suffix}.png'))
        plt.close()

    return corr

# Обработка изображений
for filename in os.listdir(src_directory):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(src_directory, filename)
        image = io.imread(image_path)

        # Преобразование яркости
        contrasted_rgb, luminance_contrast, luminance_orig = contrast_image(image)

        name_wo_ext = os.path.splitext(filename)[0]

        # Сохраняем изображения
        io.imsave(os.path.join(result_directory, f'{name_wo_ext}_gray_before.png'),
                  (luminance_orig * 255).astype(np.uint8))
        io.imsave(os.path.join(result_directory, f'{name_wo_ext}_gray_after.png'),
                  (luminance_contrast * 255).astype(np.uint8))

        # Гистограммы до и после
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].hist(luminance_orig.ravel(), bins=256, color='blue')
        axs[0].set_title('Before Contrast')
        axs[1].hist(luminance_contrast.ravel(), bins=256, color='green')
        axs[1].set_title('After Contrast')
        plt.tight_layout()
        plt.savefig(os.path.join(result_directory, f'{name_wo_ext}_histograms.png'))
        plt.close()

        # GLCM признаки
        corr_before = compute_glcm(luminance_orig, name_wo_ext, 'before')
        corr_after = compute_glcm(luminance_contrast, name_wo_ext, 'after')

        # Сравнение признаков корреляции
        print(f"Файл: {filename}")
        print("Correlation before contrast:", corr_before)
        print("Correlation after  contrast:", corr_after)
        print("-" * 50)

print("✅ Обработка изображений завершена!")
