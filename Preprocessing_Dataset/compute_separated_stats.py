import os
import numpy as np
import torch
from tqdm import tqdm
from utilityFunctions import load_audio, get_STFT, get_CQT

# Configuration
piano_dir = r"..\dataset\train\piano" # Folder with only the train set
violino_dir = r"..\dataset\train\violin"
output_path = r"..\train_set_stats"

def concat_stft_cqt(stft, cqt):
    return torch.cat((stft, cqt), dim=2)

# Generic function for statistics calculation
def compute_stats(file_list, label):
    sum_all, sum_sq_all, count_all = None, None, 0

    print(f"Compute stats for: {label}")
    for filepath in tqdm(file_list):
        try:
            audio, _ = load_audio(filepath)
            stft = get_STFT(audio)          # → shape: (2, T, F1)
            cqt = get_CQT(audio)            # → shape: (2, T, F2)
            merged = concat_stft_cqt(stft, cqt)  # → shape: (2, T, F_total)

            clip_mean = merged.mean(dim=1)  # → shape: (2, F)
            clip_std  = merged.std(dim=1)

            if sum_all is None:
                sum_all = clip_mean.clone()
                sum_sq_all = clip_std.pow(2).clone()
            else:
                sum_all += clip_mean
                sum_sq_all += clip_std.pow(2)
            count_all += 1

        except Exception as e:
            print(f"[Errore] {filepath}: {e}")

    mean = sum_all / count_all
    std  = torch.sqrt(sum_sq_all / count_all)
    return mean, std

# File collection
piano_files = [os.path.join(piano_dir, f) for f in os.listdir(piano_dir) if f.endswith((".mp3", ".wav"))]
violino_files = [os.path.join(violino_dir, f) for f in os.listdir(violino_dir) if f.endswith((".mp3", ".wav"))]

# Calculation by plan
mean_p, std_p = compute_stats(piano_files, "Piano")
F_stft = get_STFT(torch.zeros(1, 22050)).shape[2]  # Use audio dummy to infer size
stft_mean_p = mean_p[:, :F_stft].numpy()
cqt_mean_p  = mean_p[:, F_stft:].numpy()
stft_std_p  = std_p[:, :F_stft].numpy()
cqt_std_p   = std_p[:, F_stft:].numpy()

output_file = os.path.join(output_path, "stats_unified_stft_cqt.npz")
np.savez(output_file,
         stft_mean=stft_mean_p,
         stft_std=stft_std_p,
         cqt_mean=cqt_mean_p,
         cqt_std=cqt_std_p)

# Calculation for violin
mean_v, std_v = compute_stats(violino_files, "Violino")
stft_mean_v = mean_v[:, :F_stft].numpy()
cqt_mean_v  = mean_v[:, F_stft:].numpy()
stft_std_v  = std_v[:, :F_stft].numpy()
cqt_std_v   = std_v[:, F_stft:].numpy()

output_file = os.path.join(output_path, "stats_unified_stft_cqt.npz")
np.savez(output_file,
         stft_mean=stft_mean_v,
         stft_std=stft_std_v,
         cqt_mean=cqt_mean_v,
         cqt_std=cqt_std_v)

