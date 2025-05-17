from PIL import Image
import numpy as np
import os
import csv
import matplotlib.pyplot as plt

inverse_path = 'lab5/inverse'
profiles_path = 'lab5/profiles'
output_csv_path = 'lab5/features.csv'

os.makedirs(profiles_path, exist_ok=True)

def binarize(image_array, threshold=128):
    # Возвращает бинарное изображение: 1 — пиксели символа, 0 — фон
    return (image_array < threshold).astype(np.float32)

def calculate_weight(image_array):
    bin_img = binarize(image_array)
    return np.sum(bin_img)

def calculate_center_of_mass(image_array):
    bin_img = binarize(image_array)
    total_weight = calculate_weight(image_array)
    y_indices, x_indices = np.indices(image_array.shape)
    center_y = np.sum(y_indices * bin_img) / total_weight
    center_x = np.sum(x_indices * bin_img) / total_weight
    return center_y, center_x

def calculate_inertia(image_array, center_y, center_x):
    bin_img = binarize(image_array)
    y_indices, x_indices = np.indices(image_array.shape)
    inertia_y = np.sum((x_indices - center_x)**2 * bin_img)
    inertia_x = np.sum((y_indices - center_y)**2 * bin_img)
    return inertia_y, inertia_x

# Функция для разделения изображения на четверти
def split_into_quarters(image_array):
    height, width = image_array.shape
    mid_x, mid_y = width // 2, height // 2
    quarter_I = image_array[:mid_y, :mid_x]      # Верхняя левая
    quarter_II = image_array[:mid_y, mid_x:]     # Верхняя правая
    quarter_III = image_array[mid_y:, :mid_x]    # Нижняя левая
    quarter_IV = image_array[mid_y:, mid_x:]     # Нижняя правая
    return quarter_I, quarter_II, quarter_III, quarter_IV

# Функция для расчета всех признаков
def calculate_features(image_array):
    height, width = image_array.shape
    total_pixels = height * width

    # Разделение на четверти
    quarter_I, quarter_II, quarter_III, quarter_IV = split_into_quarters(image_array)

    # Вес и относительный вес для каждой четверти (с бинаризацией)
    weight_I = calculate_weight(quarter_I)
    weight_II = calculate_weight(quarter_II)
    weight_III = calculate_weight(quarter_III)
    weight_IV = calculate_weight(quarter_IV)

    relative_weight_I = weight_I / (total_pixels / 4)
    relative_weight_II = weight_II / (total_pixels / 4)
    relative_weight_III = weight_III / (total_pixels / 4)
    relative_weight_IV = weight_IV / (total_pixels / 4)

    total_weight = weight_I + weight_II + weight_III + weight_IV
    relative_total_weight = total_weight / total_pixels

    # Центр тяжести
    center_y, center_x = calculate_center_of_mass(image_array)
    relative_center_y = center_y / height
    relative_center_x = center_x / width

    # Моменты инерции
    inertia_y, inertia_x = calculate_inertia(image_array, center_y, center_x)
    relative_inertia_y = inertia_y / (total_pixels * width**2)
    relative_inertia_x = inertia_x / (total_pixels * height**2)

    # Профили X и Y — считаем по бинарному изображению, чтобы не влиял оттенок серого
    bin_img = binarize(image_array)
    profile_x = np.sum(bin_img, axis=0)
    profile_y = np.sum(bin_img, axis=1)

    return {
        'weight_I': weight_I,
        'relative_weight_I': relative_weight_I,
        'weight_II': weight_II,
        'relative_weight_II': relative_weight_II,
        'weight_III': weight_III,
        'relative_weight_III': relative_weight_III,
        'weight_IV': weight_IV,
        'relative_weight_IV': relative_weight_IV,
        'total_weight': total_weight,
        'relative_total_weight': relative_total_weight,
        'center_y': center_y,
        'center_x': center_x,
        'relative_center_y': relative_center_y,
        'relative_center_x': relative_center_x,
        'inertia_y': inertia_y,
        'inertia_x': inertia_x,
        'relative_inertia_y': relative_inertia_y,
        'relative_inertia_x': relative_inertia_x,
        'profile_x': profile_x,
        'profile_y': profile_y,
    }

# Функция для сохранения профилей в виде изображений
def save_profile_image(profile, path, orientation='horizontal'):
    plt.figure()
    if orientation == 'horizontal':
        plt.bar(range(len(profile)), profile)
        plt.xlabel('X')
        plt.ylabel('Weight')
    else:
        plt.barh(range(len(profile)), profile)
        plt.xlabel('Weight')
        plt.ylabel('Y')
    plt.title('Profile')
    plt.savefig(path)
    plt.close()

with open(output_csv_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file, delimiter=';')
    writer.writerow([
        'letter', 'weight_I', 'relative_weight_I', 'weight_II', 'relative_weight_II',
        'weight_III', 'relative_weight_III', 'weight_IV', 'relative_weight_IV',
        'total_weight', 'relative_total_weight', 'center_y', 'center_x',
        'relative_center_y', 'relative_center_x', 'inertia_y', 'inertia_x',
        'relative_inertia_y', 'relative_inertia_x', 'profile_x', 'profile_y'
    ])

    for symbol in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя':
        image_path = os.path.join(inverse_path, f'{symbol}.png')
        image = Image.open(image_path).convert('L')
        image_array = np.array(image)

        features = calculate_features(image_array)

        save_profile_image(features['profile_x'], os.path.join(profiles_path, f'{symbol}_profile_x.png'), 'horizontal')
        save_profile_image(features['profile_y'], os.path.join(profiles_path, f'{symbol}_profile_y.png'), 'vertical')

        writer.writerow([
            symbol,
            features['weight_I'], features['relative_weight_I'],
            features['weight_II'], features['relative_weight_II'],
            features['weight_III'], features['relative_weight_III'],
            features['weight_IV'], features['relative_weight_IV'],
            features['total_weight'], features['relative_total_weight'],
            features['center_y'], features['center_x'],
            features['relative_center_y'], features['relative_center_x'],
            features['inertia_y'], features['inertia_x'],
            features['relative_inertia_y'], features['relative_inertia_x'],
            ';'.join(map(str, features['profile_x'])),
            ';'.join(map(str, features['profile_y'])),
        ])

print(f"Обработка завершена. Данные сохранены в {output_csv_path}, профили в папке {profiles_path}.")