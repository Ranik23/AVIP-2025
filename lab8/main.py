import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.feature import graycomatrix, graycoprops

# Директории
src_directory = 'lab8/pictures_src'
result_directory = 'lab8/pictures_results'
os.makedirs(result_directory, exist_ok=True)

# Функция для степенного преобразования яркости (гамма-коррекция)
def gamma_correction(image, gamma=1.0):
    hsl = color.rgb2hsv(image)
    luminance = hsl[:, :, 2]
    
    luminance_corrected = np.power(luminance, gamma)
    
    hsl[:, :, 2] = luminance_corrected
    return color.hsv2rgb(hsl), luminance_corrected, luminance

# Функция для вычисления GLCM
def compute_glcm(gray_image, filename, suffix):
    gray_uint8 = (gray_image * 255).astype(np.uint8)
    glcm = graycomatrix(gray_uint8, distances=[1], angles=[0, np.pi/2, np.pi, 3*np.pi/2], 
                        symmetric=True, normed=True)
    corr = graycoprops(glcm, 'correlation')
    
    # Визуализация GLCM
    for i, angle in enumerate(['0', '90', '180', '270']):
        plt.imshow(np.log1p(glcm[:, :, 0, i]), cmap='gray')
        plt.title(f'GLCM {angle}° {suffix} (gamma={gamma})')
        plt.colorbar()
        plt.savefig(os.path.join(result_directory, f'{filename}_glcm_{angle}_{suffix}.png'))
        plt.close()
    return corr


gamma = 0.7

# Обработка изображений
for filename in os.listdir(src_directory):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(src_directory, filename)
        image = io.imread(image_path)
        
        # Применяем гамма-коррекцию
        corrected_rgb, luminance_corrected, luminance_orig = gamma_correction(image, gamma)
        
        name_wo_ext = os.path.splitext(filename)[0]
        
        # Сохранение изображений
        io.imsave(os.path.join(result_directory, f'{name_wo_ext}_gray_before.png'),
                 (luminance_orig * 255).astype(np.uint8))
        io.imsave(os.path.join(result_directory, f'{name_wo_ext}_gray_after.png'),
                 (luminance_corrected * 255).astype(np.uint8))
        
        # Гистограммы
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].hist(luminance_orig.ravel(), bins=256, color='blue')
        axs[0].set_title('Before Gamma Correction')
        axs[1].hist(luminance_corrected.ravel(), bins=256, color='green')
        axs[1].set_title(f'After Gamma Correction (γ={gamma})')
        plt.tight_layout()
        plt.savefig(os.path.join(result_directory, f'{name_wo_ext}_histograms.png'))
        plt.close()
        
        # Анализ GLCM
        corr_before = compute_glcm(luminance_orig, name_wo_ext, 'before')
        corr_after = compute_glcm(luminance_corrected, name_wo_ext, 'after')
        
        print(f"Файл: {filename}")
        print(f"Gamma: {gamma}")
        print("Correlation before:", corr_before)
        print("Correlation after:", corr_after)
        print("-"*50)

print("✅ Обработка завершена!")