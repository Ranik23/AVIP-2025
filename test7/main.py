import numpy as np
import cv2
import csv
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from PIL import ImageFont, ImageDraw, Image
import matplotlib.pyplot as plt
import os


bmp_path = "test7/pictures_src/phrase3.bmp"
alphabet_csv_path = "lab5/features.csv" 
ground_truth = "любите свою семью" 
output_path = "test7/results/output_results.txt" 
segment_dir = "test7/segments"
inv_dir = "test7/invert_segments"


def binarize(image_array, threshold=128):
    return (image_array < threshold).astype(np.float32)

def generate_text_image(text, font_path, font_size, image_size=(500, 100)):
    image = Image.new("L", image_size, "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_path, font_size)
    draw.text((10, 10), text, font=font, fill="black")
    return np.array(image)

def simple_segment(image, target_size=(150, 210)):
    _, binary_img = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])

    width_target, height_target = target_size  # width, height
    symbols = []
    inverted_symbols = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 5 and h > 10:
            symbol_img = image[y:y+h, x:x+w]

            scale_w = width_target / w
            scale_h = height_target / h
            scale = min(scale_w, scale_h)

            new_w = int(w * scale)
            new_h = int(h * scale)

            symbol_img_resized = cv2.resize(symbol_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            symbol_padded = 255 * np.ones((height_target, width_target), dtype=np.uint8)

            x_offset = (width_target - new_w) // 2
            y_offset = (height_target - new_h) // 2

            symbol_padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = symbol_img_resized

            symbols.append(symbol_padded)

            symbol_inverted = 255 - symbol_padded
            inverted_symbols.append(symbol_inverted)

    os.makedirs(segment_dir, exist_ok=True)
    os.makedirs(inv_dir, exist_ok=True)

    for i, symbol in enumerate(symbols):
        file_path = os.path.join(segment_dir, f"symbol_{i+1}.png")
        cv2.imwrite(file_path, symbol)

    for i, inv_symbol in enumerate(inverted_symbols):
        inv_file_path = os.path.join(inv_dir, f"symbol_inv_{i+1}.png")
        cv2.imwrite(inv_file_path, inv_symbol)

    return symbols, inverted_symbols

def split_into_quarters(image_array):
    height, width = image_array.shape
    mid_x, mid_y = width // 2, height // 2
    quarter_I = image_array[:mid_y, :mid_x]     
    quarter_II = image_array[:mid_y, mid_x:]   
    quarter_III = image_array[mid_y:, :mid_x]  
    quarter_IV = image_array[mid_y:, mid_x:]   
    return quarter_I, quarter_II, quarter_III, quarter_IV

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

def calculate_features(image):
    h, w = image.shape
    total_pixels = h * w

    quarter_I, quarter_II, quarter_III, quarter_IV = split_into_quarters(image)

    weight_I = calculate_weight(quarter_I)
    weight_II = calculate_weight(quarter_II)
    weight_III = calculate_weight(quarter_III)
    weight_IV = calculate_weight(quarter_IV)

    total_weight = weight_I + weight_II + weight_III + weight_IV
    relative_total_weight = total_weight / total_pixels  # отношение веса к количеству пикселей
    
    cy, cx = calculate_center_of_mass(image)
    rel_cy = cy / h
    rel_cx = cx / w

    iy, ix = calculate_inertia(image, cy, cx)
    # Для нормализации момента инерции делим на число пикселей и размер квадрата по соответствующей оси
    rel_iy = iy / (total_weight * (w ** 2)) if total_weight != 0 else 0
    rel_ix = ix / (total_weight * (h ** 2)) if total_weight != 0 else 0

    return [
        relative_total_weight,
        rel_cx,
        rel_cy,
        rel_ix,
        rel_iy
    ]


def load_alphabet_features_from_csv(path):
    labels = []
    features = []
    with open(path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        for row in reader:
            labels.append(row['letter'])
            feature_vector = [
                float(row['relative_total_weight']),
                float(row['relative_center_x']),
                float(row['relative_center_y']),
                float(row['relative_inertia_x']),
                float(row['relative_inertia_y']),
            ]
            features.append(feature_vector)
    return labels, features

def recognize_from_bmp(bmp_path, alphabet_csv_path, ground_truth, output_path="results.txt"):
    img = cv2.imread(bmp_path, cv2.IMREAD_GRAYSCALE)

    symbols, inverted_symbols = simple_segment(img)

    alphabet_labels, alphabet_features = load_alphabet_features_from_csv(alphabet_csv_path)

    input_features = [calculate_features(img) for img in inverted_symbols]

    print("Признаки входных символов:")
    for char, feature in zip(ground_truth, input_features):  
        print(f"Буква: {char}, Признаки: {[f'{f:.3f}' for f in feature]}")

    print("\nПризнаки алфавита:")
    for char, feature in zip(alphabet_labels, alphabet_features):
        print(f"Буква: {char}, Признаки: {[f'{f:.3f}' for f in feature]}")

    scaler = StandardScaler()
    all_features = alphabet_features + input_features
    scaler.fit(all_features)
    
    norm_alphabet = scaler.transform(alphabet_features)
    norm_input = scaler.transform(input_features)

    distances = cdist(norm_input, norm_alphabet, metric='cosine')
    similarities = 1 - distances

    hypotheses = []
    for sim_row in similarities:
        idxs = np.argsort(sim_row)[::-1]  # сортируем по убыванию похожести
        ranked = [(alphabet_labels[i], round(sim_row[i], 3)) for i in idxs[:5]]  # топ-5
        hypotheses.append(ranked)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(hypotheses, 1):
            f.write(f"{i}: {row}\n")

    recognized = ''.join(h[0][0] for h in hypotheses)
    correct = sum(1 for r, t in zip(recognized, ground_truth) if r == t)
    accuracy = correct / len(ground_truth) * 100

    print("Распознанная строка:", recognized)
    print("Эталонная строка:   ", ground_truth)
    print("Ошибок:             ", len(ground_truth) - correct)
    print(f"Точность:           {accuracy:.2f}%\n")

    print("Топ-5 кандидатов для каждого символа:")
    for i, (gt_char, top5) in enumerate(zip(ground_truth, hypotheses), 1):
        print(f"{i}: '{gt_char}' -> {[f'{c[0]} ({c[1]})' for c in top5]}")

    return recognized, accuracy, hypotheses



def main():
    recognized, accuracy, hypotheses = recognize_from_bmp(bmp_path, alphabet_csv_path, ground_truth, output_path)
if __name__ == "__main__":
    main()
