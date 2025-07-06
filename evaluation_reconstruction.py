#!pip install torch torchaudio librosa soundfile scipy
import torch
import torchaudio
import librosa
import numpy as np
import soundfile as sf
import os
import glob
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
from content_encoder import ContentEncoder
from new_decoder import Decoder
from style_encoder import StyleEncoder
from utilityFunctions import get_STFT, get_CQT, inverse_STFT, get_overlap_windows, sections2spectrogram, concat_stft_cqt

'''
Content reconstruction metrics:
    - Chroma Distance: Evaluates melody preservation.
    - Onset Accuracy: Measures rhythmic coherence.
    - Pitch Correlation: Compares pitch contours.
    - MSE Spectrogram: Quantifies the difference on STFT spectrograms, as required.
'''

# Configurations
SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_BINS = 84
WINDOW_SIZE = 287
OVERLAP_PERCENTAGE = 0.3
OVERLAP_FRAMES = int(WINDOW_SIZE * OVERLAP_PERCENTAGE)
TRANSFORMER_DIM = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SECTION_LENGTH = 1.0

# Path
TEST_DIR = r"dataset\test"
OUTPUT_DIR = r"result_evaluation_test"
SAVED_MODELS_DIR = r"checkpoints"

# Metrics for content reconstruction
def chroma_distance(original_audio, generated_audio, sr=22050):
    chroma_orig = librosa.feature.chroma_stft(y=original_audio, sr=sr)
    chroma_gen = librosa.feature.chroma_stft(y=generated_audio, sr=sr)
    return np.mean(np.sqrt(np.sum((chroma_orig - chroma_gen) ** 2, axis=0)))

def onset_accuracy(original_audio, generated_audio, sr=22050):
    onset_frames_orig = librosa.onset.onset_detect(y=original_audio, sr=sr)
    onset_frames_gen = librosa.onset.onset_detect(y=generated_audio, sr=sr)
    
    # Use the maximum frame index or audio length in frames
    max_frame_idx = max(
        max(onset_frames_orig, default=0),
        max(onset_frames_gen, default=0)
    ) if len(onset_frames_orig) > 0 or len(onset_frames_gen) > 0 else 0
    
    # Estimate total frames from audio length
    total_frames = max(int(len(original_audio) / HOP_LENGTH) + 1, max_frame_idx + 1)
    
    y_true = np.zeros(total_frames)
    y_pred = np.zeros(total_frames)
    
    y_true[onset_frames_orig] = 1
    y_pred[onset_frames_gen] = 1
    
    return f1_score(y_true, y_pred, average='binary')

def pitch_correlation(original_audio, generated_audio, sr=22050):
    pitches_orig, _ = librosa.piptrack(y=original_audio, sr=sr)
    pitches_gen, _ = librosa.piptrack(y=generated_audio, sr=sr)
    pitch_mean_orig = np.mean(pitches_orig, axis=0)
    pitch_mean_gen = np.mean(pitches_gen, axis=0)
    return pearsonr(pitch_mean_orig, pitch_mean_gen)[0]

def mse_spectrogram(original_audio, generated_audio, sr=22050):
    spec_orig = np.abs(librosa.stft(original_audio, n_fft=N_FFT, hop_length=HOP_LENGTH))
    spec_gen = np.abs(librosa.stft(generated_audio, n_fft=N_FFT, hop_length=HOP_LENGTH))
    return np.mean((spec_orig - spec_gen) ** 2)

# Function to generate class embeddings using StyleEncoder
def generate_class_embeddings(style_encoder, piano_file, violin_file, device):
    style_encoder.eval()
    class_embeddings = {}
    
    # Process piano file
    waveform, sr = torchaudio.load(piano_file)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    if waveform.shape[0] == 2:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    waveform = waveform.to(device)
    
    stft = get_STFT(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH).to(device)
    cqt = get_CQT(waveform, sample_rate=SAMPLE_RATE, n_bins=N_BINS, hop_length=HOP_LENGTH).to(device)
    input_spectrogram = concat_stft_cqt(stft, cqt)
    sections = get_overlap_windows(input_spectrogram, window_size=WINDOW_SIZE, overlap_frames=OVERLAP_FRAMES)
    sections = sections.unsqueeze(0)
    
    with torch.no_grad():
        _, class_emb = style_encoder(sections, torch.tensor([0]).to(device))  # 0 for piano
    class_embeddings["piano"] = class_emb.squeeze(0).cpu()  # Shape: [256]
    
    # Process violin file
    waveform, sr = torchaudio.load(violin_file)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    if waveform.shape[0] == 2:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    waveform = waveform.to(device)
    
    stft = get_STFT(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH).to(device)
    cqt = get_CQT(waveform, sample_rate=SAMPLE_RATE, n_bins=N_BINS, hop_length=HOP_LENGTH).to(device)
    input_spectrogram = concat_stft_cqt(stft, cqt)
    sections = get_overlap_windows(input_spectrogram, window_size=WINDOW_SIZE, overlap_frames=OVERLAP_FRAMES)
    sections = sections.unsqueeze(0)
    
    with torch.no_grad():
        _, class_emb = style_encoder(sections, torch.tensor([1]).to(device))  # 1 for violin
    class_embeddings["violin"] = class_emb.squeeze(0).cpu()  # Shape: [256]
    
    return class_embeddings

# Reconstruction function
def process_audio(waveform, sr, content_encoder, decoder, source_class, class_embeddings):
    stft = get_STFT(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH).to(DEVICE)
    cqt = get_CQT(waveform, sample_rate=SAMPLE_RATE, n_bins=N_BINS, hop_length=HOP_LENGTH).to(DEVICE)
    input_spectrogram = concat_stft_cqt(stft, cqt)
    sections = get_overlap_windows(input_spectrogram, window_size=WINDOW_SIZE, overlap_frames=OVERLAP_FRAMES)
    sections = sections.unsqueeze(0)
    
    content_encoder.eval()
    with torch.no_grad():
        content_emb = content_encoder(sections)
    
    # Select class embedding based on source_class
    class_emb = class_embeddings[source_class].unsqueeze(0).to(DEVICE)  # Shape: [1, TRANSFORMER_DIM]
    
    decoder.eval()
    with torch.no_grad():
        output_stft = decoder(content_emb, class_emb, target_length=content_emb.size(1))
    
    output_stft = output_stft.squeeze(0)
    original_time = stft.size(1)
    full_spectrogram = sections2spectrogram(output_stft, original_size=original_time, overlap=OVERLAP_FRAMES)
    
    output_audio = inverse_STFT(full_spectrogram, n_fft=N_FFT, hop_length=HOP_LENGTH)
    
    return output_audio.cpu(), sr

# Function to evaluate reconstruction
def eval_reconstruction(audio_path, content_encoder, decoder, source_class, output_dir, class_embeddings):
    waveform, sr = torchaudio.load(audio_path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        sr = SAMPLE_RATE
    if waveform.shape[0] == 2:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Reconstruction (same timbre)
    x_ricostruito, sr = process_audio(waveform, sr, content_encoder, decoder, source_class, class_embeddings)
    
    # Calculate content metrics
    waveform_np = waveform.numpy()[0]
    x_ricostruito_np = x_ricostruito.numpy()
    chroma_dist = chroma_distance(waveform_np, x_ricostruito_np, sr=sr)
    onset_acc = onset_accuracy(waveform_np, x_ricostruito_np, sr=sr)
    pitch_corr = pitch_correlation(waveform_np, x_ricostruito_np, sr=sr)
    mse_spec = mse_spectrogram(waveform_np, x_ricostruito_np, sr=sr)
    
    print("Content reconstruction metrics:")
    print(f" - Chroma Distance: {chroma_dist:.4f}")
    print(f" - Onset Accuracy: {onset_acc:.4f}")
    print(f" - Pitch Correlation: {pitch_corr:.4f}")
    print(f" - MSE Spectrogram: {mse_spec:.4f}")
    
    output_filename = f"{source_class}_reconstructed_{os.path.basename(audio_path)}"
    output_path = os.path.join(output_dir, output_filename)
    sf.write(output_path, x_ricostruito_np, sr)
    print(f"Saved: {output_path}")
    
    return {
        "chroma_distance": chroma_dist,
        "onset_accuracy": onset_acc,
        "pitch_correlation": pitch_corr,
        "mse_spectrogram": mse_spec
    }

# Function to process the test dataset
def process_test_set(test_dir, output_dir):
    piano_reconstruction_dir = os.path.join(output_dir, "piano_reconstruction")
    violin_reconstruction_dir = os.path.join(output_dir, "violin_reconstruction")
    Path(piano_reconstruction_dir).mkdir(parents=True, exist_ok=True)
    Path(violin_reconstruction_dir).mkdir(parents=True, exist_ok=True)
    
    # Load models
    content_encoder = ContentEncoder(
        cnn_out_dim=TRANSFORMER_DIM,
        transformer_dim=TRANSFORMER_DIM,
        num_heads=4,
        num_layers=4,
        channels_list=[16, 32, 64, 128, 256]
    ).to(DEVICE)
    decoder = Decoder(
        d_model=TRANSFORMER_DIM,
        nhead=4,
        num_layers=4
    ).to(DEVICE)
    style_encoder = StyleEncoder(
        transformer_dim=TRANSFORMER_DIM,
        num_heads=4,
        num_layers=4
    ).to(DEVICE)
    
    # Load pre-trained weights
    checkpoint_path = os.path.join(SAVED_MODELS_DIR, f"NEWcheckpoint_epoch_100.pth")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    content_encoder.load_state_dict(checkpoint['content_encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    style_encoder.load_state_dict(checkpoint['style_encoder'])
    
    # Get one sample audio file for each class to generate embeddings
    piano_dir = os.path.join(test_dir, "piano")
    violin_dir = os.path.join(test_dir, "violin")
    piano_files = glob.glob(os.path.join(piano_dir, "*.mp3"))
    violin_files = glob.glob(os.path.join(violin_dir, "*.mp3"))
    
    if len(piano_files) == 0 or len(violin_files) == 0:
        raise ValueError(f"Empty dataset: piano({len(piano_files)}), violin ({len(violin_files)})")
    
    # Use the first file from each class to generate embeddings
    piano_sample_file = piano_files[0]
    violin_sample_file = violin_files[0]
    
    # Generate class embeddings
    class_embeddings = generate_class_embeddings(style_encoder, piano_sample_file, violin_sample_file, DEVICE)
    
    # Get all audio files
    piano_files = glob.glob(os.path.join(piano_dir, "*.mp3"))
    violin_files = glob.glob(os.path.join(violin_dir, "*.mp3"))
    
    # Metrics collection
    metrics = {
        "piano_reconstruction": [],
        "violin_reconstruction": []
    }
    
    # Process piano files
    print(f"Piano file processing ({len(piano_files)} samples):")
    for audio_path in piano_files:
        # Reconstruction evaluation (piano → piano)
        recon_metrics = eval_reconstruction(
            audio_path, content_encoder, decoder, 
            source_class="piano", output_dir=piano_reconstruction_dir,
            class_embeddings=class_embeddings
        )
        metrics["piano_reconstruction"].append(recon_metrics)
    
    # Process violin files
    print(f"\nViolin file processing ({len(violin_files)} samples):")
    for audio_path in violin_files:
        # Reconstruction evaluation (violin → violin)
        recon_metrics = eval_reconstruction(
            audio_path, content_encoder, decoder, 
            source_class="violin", output_dir=violin_reconstruction_dir,
            class_embeddings=class_embeddings
        )
        metrics["violin_reconstruction"].append(recon_metrics)
    
    # Calculate aggregate statistics
    for transformation in metrics:
        print(f"\nStatistics for {transformation}:")
        metric_keys = metrics[transformation][0].keys() if metrics[transformation] else []
        for metric in metric_keys:
            values = [r[metric] for r in metrics[transformation] if r[metric] is not None]
            if values:
                print(f" - {metric.replace('_', ' ').title()}: Mean = {np.mean(values):.4f}, Std = {np.std(values):.4f}")

# Execution
process_test_set(TEST_DIR, OUTPUT_DIR)