import os
import numpy as np
from PIL import Image, ImageDraw

SRC_PATH = "lab6/pictures_src/phrase.bmp"
DST_DIR = "lab6/pictures_results"
os.makedirs(DST_DIR, exist_ok=True)

def to_binary(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.array(img)
    return (arr < 128).astype(np.uint8)

def flood_fill(img, x, y, label, labels):
    h, w = img.shape
    stack = [(x, y)]
    coords = []

    while stack:
        cx, cy = stack.pop()
        if cx < 0 or cx >= w or cy < 0 or cy >= h:
            continue
        if img[cy, cx] == 0 or labels[cy, cx] != 0:
            continue
        labels[cy, cx] = label
        coords.append((cx, cy))
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    stack.append((cx + dx, cy + dy))
    return coords

def connected_components(bin_img, min_area=10):
    h, w = bin_img.shape
    labels = np.zeros((h, w), dtype=int)
    label_id = 1
    boxes = []

    for y in range(h):
        for x in range(w):
            if bin_img[y, x] == 1 and labels[y, x] == 0:
                coords = flood_fill(bin_img, x, y, label_id, labels)
                if len(coords) >= min_area:
                    xs, ys = zip(*coords)
                    boxes.append((min(xs), min(ys), max(xs), max(ys)))
                    label_id += 1
    return sorted(boxes, key=lambda b: b[0])

def profiles(bin_img: np.ndarray):
    return bin_img.sum(axis=1), bin_img.sum(axis=0)

def save_letter_profiles(bin_img: np.ndarray, boxes: list[tuple]):
    for idx, (x0, y0, x1, y1) in enumerate(boxes):
        patch = bin_img[y0:y1+1, x0:x1+1]
        char_img = Image.fromarray((1 - patch) * 255)
        bmp_name = f"{idx:02d}.bmp"
        char_img.save(os.path.join(DST_DIR, bmp_name))

        h_prof, v_prof = profiles(patch)
        txt_name = f"{idx:02d}.txt"
        with open(os.path.join(DST_DIR, txt_name), "w", encoding="utf-8") as f:
            f.write("horizontal:\n" + " ".join(map(str, h_prof.tolist())) + "\n")
            f.write("vertical:\n"   + " ".join(map(str, v_prof.tolist())))

def draw_boxes(path: str, boxes: list[tuple]):
    img = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for x0, y0, x1, y1 in boxes:
        draw.rectangle([(x0, y0), (x1, y1)], outline="blue", width=2)
    img.save(os.path.join(DST_DIR, "phrase_boxes_manual.bmp"))

def main():
    bin_img = to_binary(SRC_PATH)
    boxes = connected_components(bin_img, min_area=10)
    draw_boxes(SRC_PATH, boxes)
    save_letter_profiles(bin_img, boxes)
    print(f"Сегментировано символов: {len(boxes)}")
    print(f"Результаты в {DST_DIR}")

if __name__ == "__main__":
    main()
