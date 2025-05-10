import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy.signal import wiener, savgol_filter
import soundfile as sf
from scipy.ndimage import maximum_filter

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
PEAKS_FILE_WIENER = os.path.join(RESULTS_DIR, "peaks_wiener.txt")
PEAKS_FILE_SAVGOL = os.path.join(RESULTS_DIR, "peaks_savgol.txt")

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
def plot_spectrogram(y, sr, title, out_path, peaks=None):
    """Строит спектрограмму"""
    D = librosa.stft(y, n_fft=2048, hop_length=512, window='hann')
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    plt.figure(figsize=(8, 4))
    librosa.display.specshow(S_db, sr=sr, hop_length=512,
                             x_axis='time', y_axis='log', cmap='magma')

    if peaks:
        # Добавляем пики на спектрограмму
        for peak in peaks:
            t, f, _ = peak
            plt.plot(t, f, 'ro')  # Рисуем пики как красные точки

    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[✓] Спектрограмма сохранена: {out_path}")

# 5. Функция для нахождения пиков в частотно-временной области
def find_time_freq_peaks(y, sr, n_fft=1024, hop_length=256, dt=0.1, df=50):
    """Находит пики в частотно-временной области"""
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))**2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop_length)

    t_win = int(dt * sr / hop_length)
    f_res = freqs[1] - freqs[0]
    f_win = max(int(df / f_res), 1)

    local_max = maximum_filter(S, size=(f_win, t_win))
    peaks = np.argwhere(S == local_max)

    peaks_list = [(times[t], freqs[f], S[f, t]) for f, t in peaks]
    peaks_list.sort(key=lambda x: x[2], reverse=True)
    return peaks_list[:10]

# 6. Функция для записи пиков в файл
def save_peaks_to_file(peaks, file_path):
    """Сохраняет пики в текстовый файл"""
    with open(file_path, 'w') as file:
        for peak in peaks:
            t, freq, amplitude = peak  # Переименовал f на freq
            file.write(f"Time: {t:.3f}s, Frequency: {freq:.2f}Hz, Amplitude: {amplitude:.2f}\n")
    print(f"[✓] Пики сохранены в файл: {file_path}")


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
    
    # Находим пики в спектрограмме после фильтрации Винера
    peaks_wiener = find_time_freq_peaks(y_wiener, sr)
    # Сохраняем пики в файл
    save_peaks_to_file(peaks_wiener, PEAKS_FILE_WIENER)
    # Строим спектрограмму после фильтрации Винера с пиками
    plot_spectrogram(y_wiener, sr, 'Wiener Denoised', PLOT_WIENER, peaks_wiener)

    # Применяем фильтрацию Савицкого-Голея
    y_savgol = denoise_savgol(y)
    sf.write(FILTERED_WAV_SAVGOL, y_savgol, sr)
    print(f"[✓] Фильтр Савицкого-Голея сохранён: {FILTERED_WAV_SAVGOL}")

    # Находим пики в спектрограмме после фильтрации Савицкого-Голея
    peaks_savgol = find_time_freq_peaks(y_savgol, sr)
    # Сохраняем пики в файл
    save_peaks_to_file(peaks_savgol, PEAKS_FILE_SAVGOL)
    # Строим спектрограмму после фильтрации Савицкого-Голея с пиками
    plot_spectrogram(y_savgol, sr, 'Savitzky-Golay Denoised', PLOT_SAVGOL, peaks_savgol)

if __name__ == "__main__":
    main()
