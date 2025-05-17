import numpy as np
import cv2
import csv
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from PIL import ImageFont, ImageDraw, Image
import matplotlib.pyplot as plt
import os


recognizedd = "всем привет"
bmp_path = "lab7new/pictures_src/phrase.bmp"
alphabet_csv_path = "lab5/features.csv" 
ground_truth = "всем привет" 
output_path = "lab7new/results/output_results.txt" 
segment_dir = "lab7new/segments"

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

    symbols = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 5 and h > 10:
            symbol_img = image[y:y+h, x:x+w]
            symbol_img_resized = cv2.resize(symbol_img, target_size, interpolation=cv2.INTER_AREA)
            symbols.append(symbol_img_resized)
    
    os.makedirs(segment_dir, exist_ok=True)

    for i, symbol in enumerate(symbols):
        file_path = os.path.join(segment_dir, f"symbol_{i+1}.png")
        cv2.imwrite(file_path, symbol)
        print(f"Сохранен символ {i+1} в {file_path}")

    return symbols

def split_into_quarters(image_array):
    height, width = image_array.shape
    mid_x, mid_y = width // 2, height // 2
    quarter_I = image_array[:mid_y, :mid_x]     
    quarter_II = image_array[:mid_y, mid_x:]   
    quarter_III = image_array[mid_y:, :mid_x]  
    quarter_IV = image_array[mid_y:, mid_x:]   
    return quarter_I, quarter_II, quarter_III, quarter_IV

def calculate_weight(image_array):
    return np.sum(image_array) / 255

def calculate_center_of_mass(image_array):
    total_weight = calculate_weight(image_array)
    y_indices, x_indices = np.indices(image_array.shape)
    center_y = np.sum(y_indices * image_array) / total_weight / 255
    center_x = np.sum(x_indices * image_array) / total_weight / 255
    return center_y, center_x

def calculate_inertia(image_array, center_y, center_x):
    y_indices, x_indices = np.indices(image_array.shape)
    inertia_y = np.sum((x_indices - center_x)**2 * image_array) / 255
    inertia_x = np.sum((y_indices - center_y)**2 * image_array) / 255
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
    relative_total_weight = total_weight / total_pixels
    
    cy, cx = calculate_center_of_mass(image)
    rel_cy = cy / h
    rel_cx = cx / w
    iy, ix = calculate_inertia(image, cy, cx)
    rel_iy = iy / (total_pixels * w**2)
    rel_ix = ix / (total_pixels * h**2)
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

    input_images = simple_segment(img)

    alphabet_labels, alphabet_features = load_alphabet_features_from_csv(alphabet_csv_path)

    input_features = [calculate_features(img) for img in input_images]

    print("Признаки входных символов:")
    for char, feature in zip(ground_truth, input_features):  
        print(f"Буква: {char}, Признаки: {[float(f) for f in feature]}")


    print("\nПризнаки алфавита:")
    for char, feature in zip(alphabet_labels, alphabet_features):
        print(f"Буква: {char}, Признаки: {[float(f) for f in feature]}")

    scaler = StandardScaler()
    all_features = alphabet_features + input_features
    scaler.fit(all_features)
    norm_alphabet = scaler.transform(alphabet_features)
    norm_input = scaler.transform(input_features)

    distances = cdist(norm_input, norm_alphabet, metric='euclidean')
    similarities = 1 / (1 + distances)

    hypotheses = []
    for sim_row in similarities:
        idxs = np.argsort(sim_row)[::-1]
        ranked = [(alphabet_labels[i], round(sim_row[i], 3)) for i in idxs]
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
    print(f"Точность:           {accuracy:.2f}%")

    return recognized, accuracy, hypotheses

def main():
    recognized, accuracy, hypotheses = recognize_from_bmp(bmp_path, alphabet_csv_path, ground_truth, output_path)
if __name__ == "__main__":
    main()
