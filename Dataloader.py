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

        # Return both piano and violin sections with their labels
        # This will be handled by the collate_fn to ensure proper batch structure
        return {
            'piano': sections_p,
            'violin': sections_v,
            'piano_label': 0,
            'violin_label': 1
        }

def collate_fn(batch):
    """
    Collate function that ensures the first half of each batch contains piano samples (label 0)
    and the second half contains violin samples (label 1).
    
    Expected input format: list of dictionaries with 'piano', 'violin', 'piano_label', 'violin_label'
    Expected output format: (B, S, 2, T, F), (B,)
    """
    # Extract piano and violin sections from batch
    piano_sections_list = []
    violin_sections_list = []
    
    for item in batch:
        piano_sections_list.append(item['piano'])    # (S, 2, T, F)
        violin_sections_list.append(item['violin'])  # (S, 2, T, F)
    
    # Find minimum sequence length for both piano and violin
    min_seq_len_piano = min(sections.shape[0] for sections in piano_sections_list)
    min_seq_len_violin = min(sections.shape[0] for sections in violin_sections_list)
    min_seq_len = min(min_seq_len_piano, min_seq_len_violin)
    
    # Truncate all sequences to minimum length
    truncated_piano = []
    truncated_violin = []
    
    for sections in piano_sections_list:
        truncated_piano.append(sections[:min_seq_len])
    
    for sections in violin_sections_list:
        truncated_violin.append(sections[:min_seq_len])
    
    # Stack to create batch dimensions
    piano_batch = torch.stack(truncated_piano, dim=0)    # (B, S, 2, T, F)
    violin_batch = torch.stack(truncated_violin, dim=0)  # (B, S, 2, T, F)
    
    # Concatenate piano and violin batches
    # First half: piano (label 0), Second half: violin (label 1)
    X = torch.cat([piano_batch, violin_batch], dim=0)    # (2*B, S, 2, T, F)
    
    # Create labels: first half = 0 (piano), second half = 1 (violin)
    batch_size = len(batch)
    piano_labels = torch.zeros(batch_size, dtype=torch.long)  # (B,) with all 0s
    violin_labels = torch.ones(batch_size, dtype=torch.long)  # (B,) with all 1s
    Y = torch.cat([piano_labels, violin_labels], dim=0)       # (2*B,)
    
    return X, Y

# Example usage - update paths for your system
def get_dataloader(piano_dir, violin_dir, batch_size=8, shuffle=True, stats_path=None):
    """
    Create a dataloader that ensures structured batches for robust training.
    
    Args:
        piano_dir: Path to piano audio files
        violin_dir: Path to violin audio files
        batch_size: Batch size for training (actual batch size will be 2*batch_size)
        shuffle: Whether to shuffle the dataset
        stats_path: Path to normalization statistics file
    
    Returns:
        DataLoader that returns (2*B, S, 2, T, F), (2*B,) format where:
        - First half of batch contains piano samples (label 0)
        - Second half of batch contains violin samples (label 1)
        - This ensures every batch has both classes for robust adversarial training
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
            print(f"  X shape: {x.shape}")  # Should be (2*B, S, 2, T, F)
            print(f"  Labels shape: {labels.shape}")  # Should be (2*B,)
            print(f"  Labels: {labels}")
            print(f"  First half (piano): {labels[:len(labels)//2]}")
            print(f"  Second half (violin): {labels[len(labels)//2:]}")
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
