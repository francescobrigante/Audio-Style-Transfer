import torch
import torchaudio
import librosa
import numpy as np
import matplotlib.pyplot as plt
import math

WINDOW_SIZE = 287
OVERLAP_PERCENTAGE = 0.3
OVERLAP_FRAMES = 96

def get_STFT(waveform, n_fft=1024, hop_length=256):
    """
    Input: audio of shape (channels, samples)
    
    n_fft: number of samples in each FFT window
    hop_length: number of samples between successive frames
    
    Output: torch.Tensor (frames, freq, 2) where 2 is [real, imaginary]
    """
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)  # (1, samples)
    
    window = torch.hann_window(n_fft)
    
    stft = torch.stft(
        waveform, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True
    )
    stft = stft.squeeze(0)  # remove channel dim -> shape: (freq, time)
    
    real_part = torch.real(stft)  # (freq, time)
    imag_part = torch.imag(stft)  # (freq, time)
    
    stft_tensor = torch.stack([real_part, imag_part], dim=-1)  # (freq, time, 2)
    stft_tensor = stft_tensor.permute(2, 1, 0)  # (2, time, freq)
    
    return stft_tensor

def get_CQT(waveform, sample_rate=22050, n_bins=84, hop_length=256):
    """
    Input: audio of shape (channels, samples)
    
    n_bins: number of frequency bins (typically 84 for 7 octaves)
    
    Output: torch.Tensor (frames, freq, 2) where 2 is [real, imaginary]
    """
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.cpu().numpy()
        
    waveform = waveform.squeeze()  # remove channel dim if present
    
    cqt = librosa.cqt(waveform, sr=sample_rate, n_bins=n_bins, hop_length=hop_length)  # (freq, time), complex
    
    real_part = np.real(cqt)  # (freq, time)
    imag_part = np.imag(cqt)  # (freq, time)
    
    cqt_tensor = np.stack([real_part, imag_part], axis=-1)  # (freq, time, 2)
    cqt_tensor = np.transpose(cqt_tensor, (2, 1, 0))  # (2, time, freq)
    
    return torch.from_numpy(cqt_tensor).float()

def inverse_STFT(stft_tensor, n_fft=1024, hop_length=256):
    """
    Input: torch.Tensor (2, time, freq) where 2 is [real, imaginary]
    
    Output: torch.Tensor (samples,) - reconstructed waveform
    """
    stft_tensor = stft_tensor.permute(0, 2, 1)  # (2, freq, time)
    
    real_part = stft_tensor[0, :, :]  # (freq, frames)
    imag_part = stft_tensor[1, :, :]  # (freq, frames)
    stft_complex = torch.complex(real_part, imag_part)  # (freq, frames)
    
    stft_complex = stft_complex.unsqueeze(0)  # (1, freq, frames)
    
    window = torch.hann_window(n_fft)
    
    waveform = torch.istft(
        stft_complex, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=False
    )
    
    return waveform.squeeze(0)  # (samples,)

def inverse_CQT(cqt_tensor, sample_rate=22050, n_bins=84, hop_length=256):
    """
    Input: torch.Tensor (frames, freq, 2) where 2 is [real, imaginary]
    
    Output: torch.Tensor (samples,) - reconstructed waveform
    """
    if isinstance(cqt_tensor, torch.Tensor):
        cqt_np = cqt_tensor.cpu().numpy()
    else:
        cqt_np = cqt_tensor
    
    cqt_np = np.transpose(cqt_np, (0, 2, 1))  # (freq, frames, 2)
    
    real_part = cqt_np[0, :, :]  # (freq, frames)
    imag_part = cqt_np[1, :, :]  # (freq, frames)
    cqt_complex = real_part + 1j * imag_part  # (freq, frames)
    
    waveform = librosa.icqt(cqt_complex, sr=sample_rate, hop_length=hop_length)  # (samples,)
    
    return torch.from_numpy(waveform).float()

def load_audio(file_path, sample_rate=22050, cut_time_seconds=10):
    """
    Loads an audio file, resamples it to a specified sample rate, and cuts it to a specified duration.
    """
    waveform, orig_sample_rate = torchaudio.load(file_path)
    cut_samples = int(cut_time_seconds * orig_sample_rate)
    if waveform.shape[-1] < cut_samples:
        padding = torch.zeros((waveform.shape[0], cut_samples - waveform.shape[-1]))
        waveform = torch.cat([waveform, padding], dim=-1)
    waveform = waveform[:, :cut_samples]
    
    if orig_sample_rate != sample_rate:
        waveform = torchaudio.functional.resample(waveform, orig_sample_rate, sample_rate)
    
    if waveform.shape[0] == 2:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    return waveform, sample_rate

def plot_stft(spectrogram, sr=22050, hop_length=256, log_scale=True):
    """
    Plot STFT magnitude (in dB or not) and phase from a tensor.
    Works with both single and multiple sections of STFT.
    """
    arr = spectrogram.detach().cpu()
    if arr.ndim == 3:
        sections = [arr]
    elif arr.ndim == 4:
        sections = [arr[i] for i in range(arr.shape[0])]
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got shape {arr.shape}")

    for idx, sec in enumerate(sections):
        real, imag = sec[0], sec[1]
        
        stft_magnitude = torch.hypot(real, imag)
        if log_scale:
            stft_magnitude = 20 * torch.log10(stft_magnitude + 1e-8)
        stft_magnitude = stft_magnitude.cpu().numpy()
    
        plt.figure(figsize=(8, 4))
        plt.imshow(
            stft_magnitude.T,
            origin='lower',
            aspect='auto',
            extent=[0, sec.shape[1] * hop_length / sr, 0, sr/2]
        )
        plt.colorbar(label='Magnitude (dB)')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title(f"{'Section '+str(idx)+': ' if len(sections)>1 else ''}STFT Magnitude")
        plt.tight_layout()
        plt.show()

        phase = torch.atan2(imag, real)
        phase = phase.cpu().numpy()
        
        plt.figure(figsize=(8, 4))
        plt.imshow(
            phase.T,
            origin='lower',
            cmap='hsv',
            aspect='auto',
            extent=[0, sec.shape[1] * hop_length / sr, 0, sr/2]
        )
        plt.colorbar(label='Phase (rad)')
        plt.xlabel('Time (s)')
        plt.ylabel('Frequency (Hz)')
        plt.title(f"{'Section '+str(idx)+': ' if len(sections)>1 else ''}STFT Phase")
        plt.tight_layout()
        plt.show()

def plot_cqt(spectrogram, sr=22050, hop_length=256, log_scale=True):
    """
    Visualizza magnitudine e fase di una (o più) CQT.

    Args:
        tensor: torch.Tensor di shape:
            - (2, time, freq)            → una sola CQT
            - (n_sections, 2, time, freq) → più sezioni
            dove channel=0 è reale, channel=1 è immaginario.
        sr: sample rate.
        hop_length: hop tra i frame temporali (per asse tempo).
    """
    arr = spectrogram.detach().cpu()
    if arr.ndim == 3:
        sections = [arr]
    elif arr.ndim == 4:
        sections = [arr[i] for i in range(arr.shape[0])]
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got shape {arr.shape}")

    for idx, sec in enumerate(sections):
        real = sec[0]
        imag = sec[1]

        cqt_magnitude = torch.hypot(real, imag)
        if log_scale:
            cqt_magnitude = librosa.amplitude_to_db(cqt_magnitude, ref=np.max)
        
        plt.figure(figsize=(8, 4))
        plt.imshow(
            cqt_magnitude.T,
            origin='lower',
            aspect='auto',
            extent=[0, sec.shape[1] * hop_length / sr, 0, sec.shape[2]]
        )
        
        plt.colorbar(label='Magnitude (dB)')
        plt.xlabel('Time (s)')
        plt.ylabel('CQT bins')
        title = f"{'Section '+str(idx)+': ' if len(sections)>1 else ''}CQT Magnitude"
        plt.title(title)
        plt.tight_layout()
        plt.show()

        phase = torch.atan2(imag, real)
        phase = phase.cpu().numpy()
        
        plt.figure(figsize=(8, 4))
        plt.imshow(
            phase.T,
            origin='lower',
            aspect='auto',
            cmap='hsv',
            extent=[0, sec.shape[1] * hop_length / sr, 0, sec.shape[2]]
        )
        plt.colorbar(label='Phase (rad)')
        plt.xlabel('Time (s)')
        plt.ylabel('CQT bins')
        title = f"{'Section '+str(idx)+': ' if len(sections)>1 else ''}CQT Phase"
        plt.title(title)
        plt.tight_layout()
        plt.show()

def get_overlap_windows(spectrogram, window_size=WINDOW_SIZE, overlap_frames=OVERLAP_FRAMES):
    """
    Input: spectrogram of shape (2, time, freq)
    Output: torch.Tensor of shape (n_sections, 2, window_size, freq)
    """
    channels, n_time, n_freq = spectrogram.shape
    step_size = window_size - overlap_frames
    sections = []
    
    for start_time in range(0, n_time, step_size):
        end_time = min(start_time + window_size, n_time)
        if end_time - start_time < window_size * 0.5:  # Escludi finestre troppo corte
            break
        section = spectrogram[:, start_time:end_time, :]
        slice_len = end_time - start_time
        pad_size = window_size - slice_len
        if pad_size > 0:
            pad_tensor = torch.zeros((channels, pad_size, n_freq), device=spectrogram.device)
            section = torch.cat([section, pad_tensor], dim=1)
        sections.append(section)
        if end_time == n_time:
            break
    
    return torch.stack(sections, dim=0)

def sections2spectrogram(sections, original_size, overlap=OVERLAP_FRAMES):
    """
    Reconstructs a full spectrogram from overlapping sections.
    """
    n_sections, _, wind_size, n_freq = sections.shape
    hop = wind_size - overlap
    n_time = hop * (n_sections - 1) + wind_size
    
    full = torch.zeros((2, n_time, n_freq), device=sections.device)
    count = torch.zeros((1, n_time, 1), device=sections.device)
    
    for i in range(n_sections):
        start = i * hop
        end = start + wind_size
        full[:, start:end, :] += sections[i]
        count[:, start:end, :] += 1.0

    full = full / count.clamp(min=1.0)
    return full[:, :original_size, :]

def concat_stft_cqt(stft, cqt):
    """
    Concats stft and cqt tensors along the frequency axis.
        stft: torch.Tensor, shape (2, T, F1)
        cqt:  torch.Tensor, shape (2, T, F2)
    Returns: torch.Tensor di shape (2, T, F1+F2)
    """
    if stft.ndim != 3 or cqt.ndim != 3:
        raise ValueError(f"Both tensors must be 3D, got {stft.ndim}D e {cqt.ndim}D.")
    if stft.shape[0] != cqt.shape[0] or stft.shape[1] != cqt.shape[1]:
        raise ValueError(
            f"Channel/Time mismatch: stft {stft.shape[:2]} vs cqt {cqt.shape[:2]}"
        )
    
    return torch.cat([stft, cqt], dim=2)
