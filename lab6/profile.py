import os
import matplotlib.pyplot as plt

DST_DIR = "lab6/pictures_results"

def plot_profiles(idx):
    txt_path = os.path.join(DST_DIR, f"{idx:02d}.txt")
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        h_prof = list(map(int, lines[1].strip().split()))
        v_prof = list(map(int, lines[3].strip().split()))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Горизонтальный профиль (по строкам)
    ax1.barh(range(len(h_prof)), h_prof, color='blue')
    ax1.set_title(f'Горизонтальный профиль (символ {idx})')
    ax1.set_xlabel('Количество чёрных пикселей')
    ax1.invert_yaxis()
    
    # Вертикальный профиль (по столбцам)
    ax2.bar(range(len(v_prof)), v_prof, color='red')
    ax2.set_title(f'Вертикальный профиль (символ {idx})')
    ax2.set_ylabel('Количество чёрных пикселей')
    
    plt.tight_layout()
    plt.savefig(os.path.join(DST_DIR, f"{idx:02d}_profiles.png"))
    plt.close()

def main():
    # Находим все .txt файлы в папке результатов
    txt_files = [f for f in os.listdir(DST_DIR) if f.endswith('.txt')]
    
    for txt_file in txt_files:
        idx = int(txt_file.split('.')[0])
        plot_profiles(idx)
    
    print(f"Графики профилей сохранены в {DST_DIR}")

if __name__ == "__main__":
    main()