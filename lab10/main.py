import os
import glob
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

SRC_DIR     = 'lab10/src'
RESULTS_DIR = 'lab10/results'
os.makedirs(RESULTS_DIR, exist_ok=True)

def plot_spectrogram(y, sr, title, outpath):
    D = librosa.stft(y, n_fft=2048, hop_length=512, window='hann')
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    plt.figure(figsize=(8,4))
    librosa.display.specshow(S_db, sr=sr, hop_length=512,
                             x_axis='time', y_axis='log', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_f0_contour(y, sr, title, outpath):
    f0 = librosa.yin(y, fmin=50, fmax=800, sr=sr,
                     frame_length=2048, hop_length=512)
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=512)
    plt.figure(figsize=(8,4))
    plt.plot(times, f0, label='F0 contour')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_spectral_peaks(y, sr, harmonics, formants, title, outpath):
    D = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    spec_avg = D.mean(axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    spec_db = librosa.amplitude_to_db(spec_avg, ref=np.max)
    plt.figure(figsize=(8,4))
    plt.plot(freqs, spec_db, label='Average spectrum (dB)')

    for k, h in enumerate(harmonics, start=1):
        plt.axvline(x=h, linestyle='--', label=f'Harmonic {k}: {h:.1f} Hz')

    for i, f in enumerate(formants, start=1):
        plt.axvline(x=f, color='red', linestyle=':', label=f'Formant {i}: {f:.1f} Hz')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()


def min_max_frequency(y, sr, threshold_db=-60):
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    mean_spec = S_db.mean(axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    mask = mean_spec > threshold_db
    if not mask.any():
        return 0.0, 0.0
    fmin = freqs[mask].min()
    fmax = freqs[mask].max()
    return fmin, fmax


def estimate_f0_and_overtones(y, sr, fmin=50, fmax=800):
    f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr,
                     frame_length=2048, hop_length=512)
    f0_med = np.nanmedian(f0)
    D = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    spec_avg = D.mean(axis=1)

    harmonics = []
    k = 1
    while True:
        h_freq = f0_med * k
        if h_freq >= sr/2:
            break
        idx = np.argmin(np.abs(freqs - h_freq))
        if spec_avg[idx] > 0.5 * spec_avg.max():
            harmonics.append(h_freq)
        k += 1
    return f0_med, harmonics


def estimate_formants(y, sr, n_formants=3, lpc_order=None):
    # Автоматический выбор порядка LPC, если не задан
    if lpc_order is None:
        lpc_order = int(2 + sr / 1000)  # Эмпирическое правило для речевых сигналов
    
    # Выбираем сегмент сигнала для анализа (около 20-30 мс)
    frame_length = min(len(y), int(0.025 * sr))  # 25 мс
    if frame_length < lpc_order * 2:
        lpc_order = max(4, frame_length // 2)
    
    # Центральный сегмент сигнала
    center = len(y) // 2
    y_segment = y[center - frame_length//2 : center + frame_length//2]
    y_segment = y_segment / np.max(np.abs(y_segment))
    
    # Оконная функция
    window = np.hamming(len(y_segment))
    y_windowed = y_segment * window
    
    # LPC анализ
    a = librosa.lpc(y_windowed, order=lpc_order)
    roots = np.roots(a)
    
    # Фильтрация корней:
    # 1. Оставляем только корни с положительной мнимой частью (чтобы избежать дублирования)
    roots = roots[np.imag(roots) > 0]
    # 2. Устойчивые корни (внутри единичного круга)
    roots = roots[np.abs(roots) < 0.999]
    
    # Преобразование в частоты и полосы
    angles = np.arctan2(np.imag(roots), np.real(roots))
    freqs = angles * sr / (2 * np.pi)
    bandwidths = -0.5 * (sr / (2 * np.pi)) * np.log(np.abs(roots))
    
    # Фильтрация по полосе (отбрасываем слишком широкие форманты)
    mask = bandwidths < 400  # Эмпирический порог
    freqs = freqs[mask]
    bandwidths = bandwidths[mask]
    
    # Сортировка по частоте и выбор первых n_formants
    if len(freqs) > 0:
        idx = np.argsort(freqs)
        freqs = freqs[idx]
        # Отбираем только форманты в разумном диапазоне (50-5000 Гц)
        freqs = freqs[(freqs > 50) & (freqs < 5000)]
        if len(freqs) >= n_formants:
            return freqs[:n_formants]
    
    # Возвращаем значения по умолчанию, если не удалось найти
    default_formants = [500, 1500, 2500]  # Примерные средние значения для речи
    return np.array(default_formants[:n_formants])

def main():
    files = glob.glob(os.path.join(SRC_DIR, '*.wav'))
    report = []

    for path in files:
        name = os.path.splitext(os.path.basename(path))[0]
        y, sr = librosa.load(path, sr=None, mono=True)
        print(f"Processing {name} (sr={sr}, length={len(y)/sr:.2f}s)")

        plot_spectrogram(y, sr, f'Spectrogram: {name}', \
                         os.path.join(RESULTS_DIR, f'spec_{name}.png'))

        plot_f0_contour(y, sr, f'F0 Contour: {name}', \
                         os.path.join(RESULTS_DIR, f'f0_{name}.png'))

        fmin, fmax = min_max_frequency(y, sr)

        f0_med, harmonics = estimate_f0_and_overtones(y, sr)

        formants = estimate_formants(y, sr)

        plot_spectral_peaks(y, sr, harmonics, formants, \
                            f'Peaks: {name}', \
                            os.path.join(RESULTS_DIR, f'peaks_{name}.png'))

        report.append({
            'name': name,
            'fmin': fmin,
            'fmax': fmax,
            'f0': f0_med,
            'overtones': harmonics,
            'formants': formants.tolist()
        })

    with open(os.path.join(RESULTS_DIR, 'report.txt'), 'w', encoding='utf-8') as f:
        for item in report:
            f.write(f"File: {item['name']}\n")
            f.write(f"Min freq: {item['fmin']:.1f} Hz, Max freq: {item['fmax']:.1f} Hz\n")
            f.write(f"Fundamental (median): {item['f0']:.1f} Hz\n")
            f.write(f"Overtones: {', '.join(f'{h:.1f}' for h in item['overtones'])}\n")
            f.write(f"Formants: {', '.join(f'{f:.1f}' for f in item['formants'])} Hz\n")
            f.write("\n")
    print(f"Report saved to {os.path.join(RESULTS_DIR, 'report.txt')}")

if __name__ == '__main__':
    main()