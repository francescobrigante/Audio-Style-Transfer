import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from utilityFunctions import load_audio, get_CQT, get_STFT, get_overlap_windows

def normalize(x, mean, std, eps=1e-8):
    if mean.ndim == 2:
        mean = mean.unsqueeze(1)  
        std = std.unsqueeze(1)
    return (x - mean) / (std + eps)

def concat_stft_cqt(stft, cqt):
    return torch.cat((stft, cqt), dim=2)

class DualInstrumentDataset(Dataset):
    def __init__(self, piano_dir, violin_dir):
        self.piano_files = sorted([
            os.path.join(piano_dir, f)
            for f in os.listdir(piano_dir)
            if f.endswith(".mp3") or f.endswith(".wav")
        ])
        self.violin_files = sorted([
            os.path.join(violin_dir, f)
            for f in os.listdir(violin_dir)
            if f.endswith(".mp3") or f.endswith(".wav")
        ])
        self.length = min(len(self.piano_files), len(self.violin_files))

        # Caricamento statistiche separate
        piano_stats = np.load(r"C:\Users\Lucia\Desktop\Uni\DL\Dataloader\stats_stft_cqt_piano.npz")
        violin_stats = np.load(r"C:\Users\Lucia\Desktop\Uni\DL\Dataloader\stats_stft_cqt_violino.npz")

        self.stft_mean_piano = torch.tensor(piano_stats["stft_mean"]).float()
        self.stft_std_piano  = torch.tensor(piano_stats["stft_std"]).float()
        self.cqt_mean_piano  = torch.tensor(piano_stats["cqt_mean"]).float()
        self.cqt_std_piano   = torch.tensor(piano_stats["cqt_std"]).float()

        self.stft_mean_violin = torch.tensor(violin_stats["stft_mean"]).float()
        self.stft_std_violin  = torch.tensor(violin_stats["stft_std"]).float()
        self.cqt_mean_violin  = torch.tensor(violin_stats["cqt_mean"]).float()
        self.cqt_std_violin   = torch.tensor(violin_stats["cqt_std"]).float()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        piano_path = self.piano_files[idx]
        violin_path = self.violin_files[idx]

        audio_p, _ = load_audio(piano_path)
        audio_v, _ = load_audio(violin_path)

        # Compute STFT and CQT
        stft_p = get_STFT(audio_p)
        cqt_p  = get_CQT(audio_p)
        stft_v = get_STFT(audio_v)
        cqt_v  = get_CQT(audio_v)

        # Normalize separately
        stft_p = normalize(stft_p, self.stft_mean_piano, self.stft_std_piano)
        cqt_p  = normalize(cqt_p,  self.cqt_mean_piano,  self.cqt_std_piano)
        stft_v = normalize(stft_v, self.stft_mean_violin, self.stft_std_violin)
        cqt_v  = normalize(cqt_v,  self.cqt_mean_violin,  self.cqt_std_violin)

        # Concatenate and window
        conc_p = concat_stft_cqt(stft_p, cqt_p)
        conc_v = concat_stft_cqt(stft_v, cqt_v)

        sections_p = get_overlap_windows(conc_p)
        sections_v = get_overlap_windows(conc_v)

        return sections_p, 0, sections_v, 1

def collate_fn(batch):
    piano_tensors, violin_tensors = [], []
    labels = []

    for piano_tensor, piano_label, violin_tensor, violin_label in batch:
        piano_tensors.append(piano_tensor)
        violin_tensors.append(violin_tensor)

        labels += [piano_label] * piano_tensor.shape[0]
        labels += [violin_label] * violin_tensor.shape[0]

    X = torch.cat(piano_tensors + violin_tensors, dim=0)  
    Y = torch.tensor(labels)                              
    return X, Y

dataset = DualInstrumentDataset(
    piano_dir=r"C:\Users\Lucia\Desktop\Uni\DL\Dataset\DATASET_partitioned\test\PianoMotion10M_ready",
    violin_dir=r"C:\Users\Lucia\Desktop\Uni\DL\Dataset\DATASET_partitioned\test\Bach+ViolinEtudes_44khz"
    )

dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)


def diagnose_window_counts(piano_dir, violin_dir, max_files=10):
    print("Diagnostica delle finestre generate da STFT + CQT")
    print("-" * 80)

    piano_files = sorted([
        os.path.join(piano_dir, f)
        for f in os.listdir(piano_dir)
        if f.endswith(".mp3") or f.endswith(".wav")
    ])[:max_files]

    violin_files = sorted([
        os.path.join(violin_dir, f)
        for f in os.listdir(violin_dir)
        if f.endswith(".mp3") or f.endswith(".wav")
    ])[:max_files]

    for p_path, v_path in zip(piano_files, violin_files):
        # Piano file
        audio_p, sr_p = load_audio(p_path)
        tensor_p = concat_stft_cqt(get_STFT(audio_p), get_CQT(audio_p))  # → (2, T, F)
        windows_p = get_overlap_windows(tensor_p)
        duration_p = audio_p.shape[-1] / sr_p
        print(f"Piano: {os.path.basename(p_path):<35} | Duration: {duration_p:.2f}s | T: {tensor_p.shape[1]} | Windows: {windows_p.shape[0]}")

        # Violin file
        audio_v, sr_v = load_audio(v_path)
        tensor_v = concat_stft_cqt(get_STFT(audio_v), get_CQT(audio_v))  # → (2, T, F)
        windows_v = get_overlap_windows(tensor_v)
        duration_v = audio_v.shape[-1] / sr_v
        print(f"Violin: {os.path.basename(v_path):<35} | Duration: {duration_v:.2f}s | T: {tensor_v.shape[1]} | Windows: {windows_v.shape[0]}")
        
        print("-" * 80)

diagnose_window_counts(
    piano_dir=r"C:\Users\Lucia\Desktop\Uni\DL\Dataset\DATASET_partitioned\test\PianoMotion10M_ready",
    violin_dir=r"C:\Users\Lucia\Desktop\Uni\DL\Dataset\DATASET_partitioned\test\Bach+ViolinEtudes_44khz",
    max_files=5
)
