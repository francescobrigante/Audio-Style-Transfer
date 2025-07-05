import os
import librosa
import numpy as np
from tqdm import tqdm

# Input directories
# Dataset united, not the one partitioned one (train, test and validation)
dataset2 = r"C:\Users\Lucia\Desktop\Uni\DL\Dataset\DATASET\Bach+ViolinEtudes_44khz" #delete row in case you want to analyze a dataset only
dataset1 = r"C:\Users\Lucia\Desktop\Uni\DL\Dataset\PianoMotion10M_ready"

# Extracts metrics from one or two datasets
def analyze_dataset(path):
    data = {
        "filenames": [],
        "durations": [],
        "rms_levels": [], # Root Mean Square: a measure of the energy/average volume
        "sample_rates": [], # Sampling frequency
        "mfcc_means": [] # The first MFCC coefficient: overall signal intensity (often represents the main spectral shape)
    }
    for fname in tqdm(os.listdir(path)):
        if not fname.lower().endswith((".mp3", ".wav")):
            continue
        fpath = os.path.join(path, fname)
        try:
            y, sr = librosa.load(fpath, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            rms = np.sqrt(np.mean(y**2))
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)

            data["filenames"].append(fname)
            data["durations"].append(duration)
            data["rms_levels"].append(rms)
            data["sample_rates"].append(sr)
            data["mfcc_means"].append(mfcc_mean)

        except Exception as e:
            print(f"Errore con {fname}: {e}")
    return data

print("\nAnalysis of the first dataset")
stats1 = analyze_dataset(dataset1)

print("\nanalysis of the second dataset") #delete row in case you want to analyze a dataset only
stats2 = analyze_dataset(dataset2) #delete row in case you want to analyze a dataset only

# Comparative summary function
def summarize_statistics(name, stats):
    print(f"\n{name}")
    print(f"• Files analyzed: {len(stats['filenames'])}")
    print(f"• Average duration: {np.mean(stats['durations']):.2f} sec")
    print(f"• Average RMS: {np.mean(stats['rms_levels']):.4f}")
    print(f"• Unique sample rates: {set(stats['sample_rates'])}")
    print(f"• Global average MFCC (first coefficient).: {np.mean([m[0] for m in stats['mfcc_means']]):.2f}")


summarize_statistics("🎻 Dataset 1", stats1)
summarize_statistics("🎼 Dataset 2", stats2) #delete row in case you want to analyze a dataset only
