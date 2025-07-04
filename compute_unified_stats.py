import os
import numpy as np
import torch
from tqdm import tqdm
from utilityFunctions import load_audio, get_STFT, get_CQT

# Configuration
dataset_dir = r"C:\Users\Lucia\Desktop\Uni\DL\Dataset\DATASET_partitioned\train"  # cartella con solo il TRAIN SET
output_path = r"C:\Users\Lucia\Desktop\Uni\DL\Dataloader"

# Initialization
sum_all = None
sum_sq_all = None
count_all = 0

files = []
for root, _, filenames in os.walk(dataset_dir):
    for f in sorted(filenames):
        if f.endswith(".mp3") or f.endswith(".wav"):
            files.append(os.path.join(root, f))

def concat_stft_cqt(stft, cqt):
    return torch.cat((stft, cqt), dim=2)

for fname in tqdm(files, desc="↻ Calcolo STFT+CQT stats"):
    path = os.path.join(dataset_dir, fname)
    try:
        audio, _ = load_audio(path)
        stft = get_STFT(audio)  # shape: (2, T, F1)
        cqt = get_CQT(audio)    # shape: (2, T, F2)
        merged = concat_stft_cqt(stft, cqt)  # shape: (2, T, F_total)

        # Media e varianza lungo T → resta (2, F)
        mean_clip = merged.mean(dim=1)  # (2, F)
        std_clip = merged.std(dim=1)    # (2, F)

        if sum_all is None:
            sum_all = mean_clip.clone()
            sum_sq_all = std_clip.pow(2).clone()
        else:
            sum_all += mean_clip
            sum_sq_all += std_clip.pow(2)
        count_all += 1

    except Exception as e:
        print(f"[Errore] {fname}: {e}")

# Final calculation
mean = sum_all / count_all
std = torch.sqrt(sum_sq_all / count_all)

# Separation between STFT and CQT 
F_total = mean.shape[1]
F_stft = get_STFT(audio).shape[2]
F_cqt = get_CQT(audio).shape[2]

stft_mean = mean[:, :F_stft].numpy()
cqt_mean = mean[:, F_stft:].numpy()
stft_std  = std[:, :F_stft].numpy()
cqt_std  = std[:, F_stft:].numpy()

# Saving
output_file = os.path.join(output_path, "stats_unified_stft_cqt.npz")
np.savez(output_file,
         stft_mean=stft_mean,
         stft_std=stft_std,
         cqt_mean=cqt_mean,
         cqt_std=cqt_std)

