import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import wiener, savgol_filter
import soundfile as sf

SRC_DIR     = 'lab9/audio_src'
RESULTS_DIR = 'lab9/audio_results'
os.makedirs(SRC_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

AUDIO_FILE    = os.path.join(SRC_DIR, "record.wav")
FILTERED_WAV_WIENER = os.path.join(RESULTS_DIR, "denoised_wiener.wav")
FILTERED_WAV_SAVGOL = os.path.join(RESULTS_DIR, "denoised_savgol.wav")
PLOT_BEFORE   = os.path.join(RESULTS_DIR, "spec_before.png")
PLOT_WIENER   = os.path.join(RESULTS_DIR, "spec_wiener.png")
PLOT_SAVGOL   = os.path.join(RESULTS_DIR, "spec_savgol.png")

# 1. Функция для загрузки аудио
def load_audio(file_path):
    """Загружает аудио файл"""
    y, sr = librosa.load(file_path, sr=None, mono=True)
    return y, sr

# 2. Функция для фильтрации Винера
def denoise_wiener(y):
    """Применяет фильтр Винера"""
    return wiener(y)

# 3. Функция для фильтрации Савицкого-Голея
def denoise_savgol(y, window_length=51, polyorder=3):
    """Применяет фильтр Савицкого-Голея"""
    return savgol_filter(y, window_length=window_length, polyorder=polyorder)

# 4. Функция для построения спектрограммы
def plot_spectrogram(y, sr, title, out_path):
    """Строит спектрограмму"""
    D = librosa.stft(y, n_fft=2048, hop_length=512, window='hann')
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    plt.figure(figsize=(8, 4))
    librosa.display.specshow(S_db, sr=sr, hop_length=512,
                             x_axis='time', y_axis='log', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[✓] Спектрограмма сохранена: {out_path}")

# Главная функция обработки
def main():
    # Загружаем исходное аудио
    y, sr = load_audio(AUDIO_FILE)
    print(f"[i] Загружено: {AUDIO_FILE}, sr={sr}, длительность={len(y)/sr:.2f}s")

    # Строим спектрограмму до фильтрации
    plot_spectrogram(y, sr, 'Original', PLOT_BEFORE)

    # Применяем фильтрацию Винера
    y_wiener = denoise_wiener(y)
    sf.write(FILTERED_WAV_WIENER, y_wiener, sr)
    print(f"[✓] Фильтр Винера сохранён: {FILTERED_WAV_WIENER}")
    
    # Строим спектрограмму после фильтрации Винера
    plot_spectrogram(y_wiener, sr, 'Wiener Denoised', PLOT_WIENER)

    # Применяем фильтрацию Савицкого-Голея
    y_savgol = denoise_savgol(y)
    sf.write(FILTERED_WAV_SAVGOL, y_savgol, sr)
    print(f"[✓] Фильтр Савицкого-Голея сохранён: {FILTERED_WAV_SAVGOL}")

    # Строим спектрограмму после фильтрации Савицкого-Голея
    plot_spectrogram(y_savgol, sr, 'Savitzky-Golay Denoised', PLOT_SAVGOL)

if __name__ == "__main__":
    main()
