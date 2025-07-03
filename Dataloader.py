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
    def __init__(self, piano_dir, violin_dir, stats_path=None):
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

        # Loading statistics - make path configurable
        if stats_path is None:
            stats_path = "stats_stft_cqt.npz"  # Default path
        
        if os.path.exists(stats_path):
            stats = np.load(stats_path)
            self.stft_mean = torch.tensor(stats["stft_mean"]).float()
            self.stft_std  = torch.tensor(stats["stft_std"]).float()
            self.cqt_mean  = torch.tensor(stats["cqt_mean"]).float()
            self.cqt_std   = torch.tensor(stats["cqt_std"]).float()
        else:
            print(f"⚠️ Warning: Stats file {stats_path} not found. Using dummy normalization.")
            # Dummy stats for testing
            self.stft_mean = torch.zeros(2, 513)
            self.stft_std  = torch.ones(2, 513)
            self.cqt_mean  = torch.zeros(2, 84)
            self.cqt_std   = torch.ones(2, 84)

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        piano_path = self.piano_files[idx]
        violin_path = self.violin_files[idx]

        audio_p, _ = load_audio(piano_path)
        audio_v, _ = load_audio(violin_path)

        # Compute STFT and CQT
        stft_p = get_STFT(audio_p)          # shape: (2, T, F)
        cqt_p = get_CQT(audio_p)            # shape: (2, T, F)
        stft_v = get_STFT(audio_v)
        cqt_v = get_CQT(audio_v)

        # Normalize each (by channel and frequency bin)
        stft_p = normalize(stft_p, self.stft_mean, self.stft_std)
        cqt_p = normalize(cqt_p, self.cqt_mean, self.cqt_std)
        stft_v = normalize(stft_v, self.stft_mean, self.stft_std)
        cqt_v = normalize(cqt_v, self.cqt_mean, self.cqt_std)

        # Concatenation + Windowing
        conc_p = concat_stft_cqt(stft_p, cqt_p)    # shape: (2, T, F_new)
        conc_v = concat_stft_cqt(stft_v, cqt_v)
        sections_p = get_overlap_windows(conc_p)   # shape: (S, C=2, T, F)
        sections_v = get_overlap_windows(conc_v)

        # Return in format compatible with training pipeline
        # Randomly choose between piano and violin for this sample
        if torch.rand(1) < 0.5:
            return sections_p, 0  # Piano = class 0
        else:
            return sections_v, 1  # Violin = class 1

def collate_fn(batch):
    """
    Collate function compatible with training pipeline.
    Expected input format: list of (sections, label) tuples
    Expected output format: (B, S, 2, T, F), (B,)
    """
    sections_list = []
    labels_list = []
    
    for sections, label in batch:
        sections_list.append(sections)  # sections: (S, 2, T, F)
        labels_list.append(label)
    
    # Stack sections to create batch dimension
    # sections_list: [(S1, 2, T, F), (S2, 2, T, F), ...]
    # We need to handle variable sequence lengths
    
    # Find minimum sequence length to ensure all samples have same S
    min_seq_len = min(sections.shape[0] for sections in sections_list)
    
    # Truncate all sequences to minimum length
    truncated_sections = []
    for sections in sections_list:
        if sections.shape[0] >= min_seq_len:
            truncated_sections.append(sections[:min_seq_len])  # Take first min_seq_len sections
        else:
            # This shouldn't happen given our min calculation, but just in case
            truncated_sections.append(sections)
    
    # Stack to create batch
    X = torch.stack(truncated_sections, dim=0)  # (B, S, 2, T, F)
    Y = torch.tensor(labels_list)               # (B,)
    
    return X, Y

# Example usage - update paths for your system
def get_dataloader(piano_dir, violin_dir, batch_size=8, shuffle=True, stats_path=None):
    """
    Create a dataloader compatible with the training pipeline.
    
    Args:
        piano_dir: Path to piano audio files
        violin_dir: Path to violin audio files
        batch_size: Batch size for training
        shuffle: Whether to shuffle the dataset
        stats_path: Path to normalization statistics file
    
    Returns:
        DataLoader that returns (B, S, 2, T, F), (B,) format
    """
    dataset = DualInstrumentDataset(piano_dir, violin_dir, stats_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

# Example paths - update these for your system
if __name__ == "__main__":
    # Test paths - update these!
    piano_dir = r"dataset/piano"  # Update this path
    violin_dir = r"dataset/violin"  # Update this path
    
    # Only run if directories exist
    if os.path.exists(piano_dir) and os.path.exists(violin_dir):
        dataset = DualInstrumentDataset(piano_dir, violin_dir)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
        
        # Test a batch
        for batch_idx, (x, labels) in enumerate(dataloader):
            print(f"Batch {batch_idx}:")
            print(f"  X shape: {x.shape}")  # Should be (B, S, 2, T, F)
            print(f"  Labels shape: {labels.shape}")  # Should be (B,)
            print(f"  Labels: {labels}")
            if batch_idx == 0:  # Just test first batch
                break
    else:
        print("⚠️ Dataset directories not found. Update paths in the script.")

# Legacy code for compatibility
# dataset = DualInstrumentDataset(
#     piano_dir=r"C:\Users\Lucia\Desktop\Uni\DL\Dataset\DATASET_partitioned\test\PianoMotion10M_ready",
#     violin_dir=r"C:\Users\Lucia\Desktop\Uni\DL\Dataset\DATASET_partitioned\test\Bach+ViolinEtudes_44khz"
# )
# dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)


def diagnose_window_counts(piano_dir, violin_dir, max_files=10):
    """
    Diagnose window counts for debugging.
    Update paths as needed for your system.
    """
    print("🧠 Diagnostica delle finestre generate da STFT + CQT")
    print("-" * 80)
    
    if not os.path.exists(piano_dir) or not os.path.exists(violin_dir):
        print(f"⚠️ Warning: Directories not found:")
        print(f"  Piano: {piano_dir}")
        print(f"  Violin: {violin_dir}")
        return

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
        print(f"🎹 Piano: {os.path.basename(p_path):<35} | Duration: {duration_p:.2f}s | T: {tensor_p.shape[1]} | Windows: {windows_p.shape[0]}")

        # Violin file
        audio_v, sr_v = load_audio(v_path)
        tensor_v = concat_stft_cqt(get_STFT(audio_v), get_CQT(audio_v))  # → (2, T, F)
        windows_v = get_overlap_windows(tensor_v)
        duration_v = audio_v.shape[-1] / sr_v
        print(f"🎻 Violin: {os.path.basename(v_path):<35} | Duration: {duration_v:.2f}s | T: {tensor_v.shape[1]} | Windows: {windows_v.shape[0]}")
        
        print("-" * 80)
