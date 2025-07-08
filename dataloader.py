import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from utilityFunctions import load_audio, get_CQT, get_STFT, get_overlap_windows
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize(x, mean, std, eps=1e-8):
    if mean.ndim == 2:
        mean = mean.unsqueeze(1).to(x.device)  
        std = std.unsqueeze(1).to(x.device)
    return (x - mean) / (std + eps)

def concat_stft_cqt(stft, cqt):
    stft = stft.to(device)
    cqt = cqt.to(device)
    return torch.cat((stft, cqt), dim=2)

class DualInstrumentDataset(Dataset):
    def __init__(self, piano_dir, violin_dir, stats_path=None, use_separate_stats=True):
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
        self.use_separate_stats = use_separate_stats

        # Loading statistics
        if use_separate_stats:
            self._load_separate_stats()
        else:
            self._load_combined_stats(stats_path)

    def _load_separate_stats(self):
        """Load separate statistics for piano and violin"""
        piano_stats_path = "train_set_stats/stats_stft_cqt_piano.npz"
        violin_stats_path = "train_set_stats/stats_stft_cqt_violin.npz"
        
        if os.path.exists(piano_stats_path) and os.path.exists(violin_stats_path):
            # Load piano statistics
            piano_stats = np.load(piano_stats_path)
            self.stft_mean_piano = torch.tensor(piano_stats["stft_mean"]).float()
            self.stft_std_piano = torch.tensor(piano_stats["stft_std"]).float()
            self.cqt_mean_piano = torch.tensor(piano_stats["cqt_mean"]).float()
            self.cqt_std_piano = torch.tensor(piano_stats["cqt_std"]).float()
            
            # Load violin statistics
            violin_stats = np.load(violin_stats_path)
            self.stft_mean_violin = torch.tensor(violin_stats["stft_mean"]).float()
            self.stft_std_violin = torch.tensor(violin_stats["stft_std"]).float()
            self.cqt_mean_violin = torch.tensor(violin_stats["cqt_mean"]).float()
            self.cqt_std_violin = torch.tensor(violin_stats["cqt_std"]).float()
            
            print(f"✅ Loaded separate statistics:")
            print(f"  Piano: {piano_stats_path}")
            print(f"  Violin: {violin_stats_path}")
        else:
            print(f"⚠️ Warning: Separate stats files not found. Using dummy normalization.")
            print(f"  Expected: {piano_stats_path}, {violin_stats_path}")
            self._create_dummy_separate_stats()

    def _load_combined_stats(self, stats_path):
        """Load combined statistics (fallback to original behavior)"""
        if stats_path is None:
            stats_path = "train_set_stats/stats_unified_stft_cqt.npz"
        
        if os.path.exists(stats_path):
            stats = np.load(stats_path)
            # Use same stats for both instruments
            self.stft_mean_piano = self.stft_mean_violin = torch.tensor(stats["stft_mean"]).float()
            self.stft_std_piano = self.stft_std_violin = torch.tensor(stats["stft_std"]).float()
            self.cqt_mean_piano = self.cqt_mean_violin = torch.tensor(stats["cqt_mean"]).float()
            self.cqt_std_piano = self.cqt_std_violin = torch.tensor(stats["cqt_std"]).float()
            
            print(f"✅ Loaded combined statistics from: {stats_path}")
        else:
            print(f"⚠️ Warning: Combined stats file {stats_path} not found. Using dummy normalization.")
            self._create_dummy_separate_stats()

    def _create_dummy_separate_stats(self):
        """Create dummy statistics for testing"""
        # Dummy stats for piano
        self.stft_mean_piano = torch.zeros(2, 513)
        self.stft_std_piano = torch.ones(2, 513)
        self.cqt_mean_piano = torch.zeros(2, 84)
        self.cqt_std_piano = torch.ones(2, 84)
        
        # Dummy stats for violin
        self.stft_mean_violin = torch.zeros(2, 513)
        self.stft_std_violin = torch.ones(2, 513)
        self.cqt_mean_violin = torch.zeros(2, 84)
        self.cqt_std_violin = torch.ones(2, 84)

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

        # Normalize using separate statistics
        stft_p = normalize(stft_p, self.stft_mean_piano, self.stft_std_piano)
        cqt_p = normalize(cqt_p, self.cqt_mean_piano, self.cqt_std_piano)
        stft_v = normalize(stft_v, self.stft_mean_violin, self.stft_std_violin)
        cqt_v = normalize(cqt_v, self.cqt_mean_violin, self.cqt_std_violin)

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
    Efficient collate function that creates balanced batches with equal piano/violin samples.
    
    Expected input format: list of dictionaries with 'piano', 'violin', 'piano_label', 'violin_label'
    Expected output format: (B, S, 2, T, F), (B,) where B = len(batch)
    
    """
    batch_size = len(batch)
    half_batch = batch_size // 2
    
    # Extract sections efficiently
    piano_sections = [batch[i]['piano'] for i in range(half_batch)]
    violin_sections = [batch[i]['violin'] for i in range(half_batch)]
    
    # Find minimum sequence length in one pass
    all_sections = piano_sections + violin_sections
    min_seq_len = min(sections.shape[0] for sections in all_sections)
    
    # Pre-allocate the final tensor for better memory efficiency
    # Get dimensions from first sample
    sample_shape = piano_sections[0].shape  # (S, 2, T, F)
    _, channels, time_steps, freq_bins = sample_shape
    
    # Create output tensor
    result_tensor = torch.empty((batch_size, min_seq_len, channels, time_steps, freq_bins), 
                               dtype=piano_sections[0].dtype)
    
    # Fill tensor 
    for i in range(half_batch):
        # Piano samples (first half)
        result_tensor[i] = piano_sections[i][:min_seq_len]
        # Violin samples (second half)
        result_tensor[i + half_batch] = violin_sections[i][:min_seq_len]
    
    # Create labels tensor directly - more efficient than list + conversion
    labels = torch.cat([
        torch.zeros(half_batch, dtype=torch.long),  # Piano labels
        torch.ones(half_batch, dtype=torch.long)    # Violin labels
    ])
    
    return result_tensor, labels

# Example usage - update paths for your system
def get_dataloader(piano_dir, violin_dir, batch_size=8, shuffle=True, stats_path=None, use_separate_stats=True):
    """
    Create a dataloader that returns balanced batches with equal piano/violin samples.
    
    Args:
        piano_dir: Path to piano audio files
        violin_dir: Path to violin audio files
        batch_size: Total batch size (must be even number for balanced batches)
        shuffle: Whether to shuffle the dataset
        stats_path: Path to normalization statistics file (used only if use_separate_stats=False)
        use_separate_stats: Whether to use separate statistics for piano and violin
    
    Returns:
        DataLoader that returns (B, S, 2, T, F), (B,) format where:
        - B = batch_size (actual requested batch size)
        - First half of batch contains piano samples (label 0)
        - Second half of batch contains violin samples (label 1)
        - Example: batch_size=8 → 4 piano + 4 violin samples
    
    Note:
        batch_size should be even to ensure balanced piano/violin distribution
    """
    if batch_size % 2 != 0:
        print(f"Warning: batch_size={batch_size} is odd. Rounding down to {batch_size-1} for balanced batches.")
        batch_size = batch_size - 1
    
    dataset = DualInstrumentDataset(piano_dir, violin_dir, stats_path, use_separate_stats)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, drop_last=True)

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

# Example usage with separate statistics (recommended)
if __name__ == "__main__":
    # Using separate statistics (default behavior)
    dataloader = get_dataloader(
        piano_dir="dataset/train",
        violin_dir="C:/Users/Lucia/Desktop/Uni/DL/Dataset/DATASET_partitioned/test/Bach+ViolinEtudes_44khz",
        batch_size=8,
        use_separate_stats=True
    )
    
    # Alternative: using combined statistics
    # dataloader = get_dataloader(
    #     piano_dir="dataset/train",
    #     violin_dir="C:/Users/Lucia/Desktop/Uni/DL/Dataset/DATASET_partitioned/test/Bach+ViolinEtudes_44khz",
    #     batch_size=8,
    #     use_separate_stats=False,
    #     stats_path="stats_stft_cqt.npz"
    # )