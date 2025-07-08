#!pip install torch torchaudio librosa soundfile scipy
import torch
import torchaudio
import librosa
import numpy as np
import soundfile as sf
import os
import glob
import json
from pathlib import Path
from scipy.stats import pearsonr
from sklearn.metrics import f1_score
from content_encoder import ContentEncoder
from style_encoder import StyleEncoder
# from SimpleDecoder_TransformerOnly import Decoder  # Use the new decoder
from new_decoder import Decoder  # Use the new decoder
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

# Paths
TEST_DIR = r"dataset\test"
OUTPUT_DIR = r"result_evaluation_test"
SAVED_MODELS_DIR = r"checkpoints"

# ===========================
# METRICS FUNCTIONS
# ===========================

def chroma_distance(original_audio, generated_audio, sr=22050):
    """Calculate chroma distance between original and generated audio"""
    try:
        chroma_orig = librosa.feature.chroma_stft(y=original_audio, sr=sr)
        chroma_gen = librosa.feature.chroma_stft(y=generated_audio, sr=sr)
        
        # Handle different lengths
        min_frames = min(chroma_orig.shape[1], chroma_gen.shape[1])
        chroma_orig = chroma_orig[:, :min_frames]
        chroma_gen = chroma_gen[:, :min_frames]
        
        return np.mean(np.sqrt(np.sum((chroma_orig - chroma_gen) ** 2, axis=0)))
    except Exception as e:
        print(f"Error in chroma_distance: {e}")
        return float('inf')

def onset_accuracy(original_audio, generated_audio, sr=22050):
    """Calculate onset detection accuracy"""
    try:
        onset_frames_orig = librosa.onset.onset_detect(y=original_audio, sr=sr)
        onset_frames_gen = librosa.onset.onset_detect(y=generated_audio, sr=sr)
        
        # Handle empty onsets
        if len(onset_frames_orig) == 0 and len(onset_frames_gen) == 0:
            return 1.0  # Perfect match if both have no onsets
        if len(onset_frames_orig) == 0 or len(onset_frames_gen) == 0:
            return 0.0  # No match if one has onsets and other doesn't
        
        # Use the maximum frame index or audio length in frames
        max_frame_idx = max(
            max(onset_frames_orig, default=0),
            max(onset_frames_gen, default=0)
        )
        
        # Estimate total frames from audio length
        total_frames = max(int(len(original_audio) / HOP_LENGTH) + 1, max_frame_idx + 1)
        
        y_true = np.zeros(total_frames)
        y_pred = np.zeros(total_frames)
        
        y_true[onset_frames_orig] = 1
        y_pred[onset_frames_gen] = 1
        
        return f1_score(y_true, y_pred, average='binary')
    except Exception as e:
        print(f"Error in onset_accuracy: {e}")
        return 0.0

def pitch_correlation(original_audio, generated_audio, sr=22050):
    """Calculate pitch correlation between original and generated audio"""
    try:
        pitches_orig, _ = librosa.piptrack(y=original_audio, sr=sr)
        pitches_gen, _ = librosa.piptrack(y=generated_audio, sr=sr)
        
        pitch_mean_orig = np.mean(pitches_orig, axis=0)
        pitch_mean_gen = np.mean(pitches_gen, axis=0)
        
        # Handle different lengths
        min_length = min(len(pitch_mean_orig), len(pitch_mean_gen))
        pitch_mean_orig = pitch_mean_orig[:min_length]
        pitch_mean_gen = pitch_mean_gen[:min_length]
        
        if min_length == 0:
            return 0.0
        
        corr, _ = pearsonr(pitch_mean_orig, pitch_mean_gen)
        return corr if not np.isnan(corr) else 0.0
    except Exception as e:
        print(f"Error in pitch_correlation: {e}")
        return 0.0

def mse_spectrogram(original_audio, generated_audio, sr=22050):
    """Calculate MSE between spectrograms"""
    try:
        spec_orig = np.abs(librosa.stft(original_audio, n_fft=N_FFT, hop_length=HOP_LENGTH))
        spec_gen = np.abs(librosa.stft(generated_audio, n_fft=N_FFT, hop_length=HOP_LENGTH))
        
        # Handle different lengths
        min_time = min(spec_orig.shape[1], spec_gen.shape[1])
        spec_orig = spec_orig[:, :min_time]
        spec_gen = spec_gen[:, :min_time]
        
        return np.mean((spec_orig - spec_gen) ** 2)
    except Exception as e:
        print(f"Error in mse_spectrogram: {e}")
        return float('inf')

# ===========================
# CLASS EMBEDDINGS GENERATION
# ===========================

def generate_class_embeddings_from_dataloader(style_encoder, test_loader, device):
    """
    Generate class embeddings using the first batch from dataloader
    """
    style_encoder.eval()
    
    with torch.no_grad():
        # Get first batch
        sections, labels = next(iter(test_loader))
        sections = sections.to(device)  # (B, S, 2, T, F)
        labels = labels.to(device)      # (B,)
        
        print(f"📊 Generating class embeddings from batch shape: {sections.shape}")
        print(f"📋 Available labels: {labels}")
        
        class_embeddings = {}
        
        # Find piano and violin samples
        piano_idx = torch.where(labels == 0)[0]
        violin_idx = torch.where(labels == 1)[0]
        
        if len(piano_idx) > 0:
            piano_sections = sections[piano_idx[0]:piano_idx[0]+1]  # (1, S, 2, T, F)
            _, piano_class_emb = style_encoder(piano_sections, torch.tensor([0]).to(device))
            class_embeddings["piano"] = piano_class_emb.squeeze(0).cpu()
            print(f"✅ Piano class embedding generated: {piano_class_emb.shape}")
        
        if len(violin_idx) > 0:
            violin_sections = sections[violin_idx[0]:violin_idx[0]+1]  # (1, S, 2, T, F)
            _, violin_class_emb = style_encoder(violin_sections, torch.tensor([1]).to(device))
            class_embeddings["violin"] = violin_class_emb.squeeze(0).cpu()
            print(f"✅ Violin class embedding generated: {violin_class_emb.shape}")
        
        if len(class_embeddings) != 2:
            raise ValueError(f"Could not generate embeddings for both classes. Found: {list(class_embeddings.keys())}")
    
    return class_embeddings

# ===========================
# AUDIO RECONSTRUCTION
# ===========================

def reconstruct_audio_from_sections(stft_sections, batch_idx, sample_idx):
    """
    Reconstruct audio from STFT sections
    
    Args:
        stft_sections: (1, S, 2, T, F) tensor
        batch_idx: batch index for debugging
        sample_idx: sample index for debugging
    
    Returns:
        numpy array of reconstructed audio
    """
    try:
        # Remove batch dimension: (S, 2, T, F)
        stft_sections = stft_sections.squeeze(0)
        
        # Simple approach: concatenate all sections
        if stft_sections.size(0) == 1:
            # Single section
            stft_single = stft_sections[0]  # (2, T, F)
        else:
            # Multiple sections - use middle section or concatenate
            # For simplicity, let's use the first section
            stft_single = stft_sections[0]  # (2, T, F)
        
        # Convert to complex tensor for inverse STFT
        real_part = stft_single[0]  # (T, F)
        imag_part = stft_single[1]  # (T, F)
        complex_stft = torch.complex(real_part, imag_part)  # (T, F)
        
        # Transpose for torchaudio format (F, T)
        complex_stft = complex_stft.transpose(0, 1)  # (F, T)
        
        # Inverse STFT
        audio = torch.istft(
            complex_stft,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            window=torch.hann_window(WIN_LENGTH).to(complex_stft.device),
            return_complex=False
        )
        
        return audio.cpu().numpy()
        
    except Exception as e:
        print(f"⚠️ Error in audio reconstruction (batch {batch_idx}, sample {sample_idx}): {e}")
        # Return silence as fallback
        return np.zeros(22050)  # 1 second of silence

def calculate_reconstruction_metrics(original_audio, reconstructed_audio, sr):
    """
    Calculate reconstruction metrics with error handling
    """
    try:
        # Ensure same length
        min_length = min(len(original_audio), len(reconstructed_audio))
        if min_length == 0:
            print("⚠️ Empty audio detected")
            return {
                "chroma_distance": float('inf'),
                "onset_accuracy": 0.0,
                "pitch_correlation": 0.0,
                "mse_spectrogram": float('inf')
            }
        
        original_audio = original_audio[:min_length]
        reconstructed_audio = reconstructed_audio[:min_length]
        
        # Calculate metrics
        chroma_dist = chroma_distance(original_audio, reconstructed_audio, sr)
        onset_acc = onset_accuracy(original_audio, reconstructed_audio, sr)
        pitch_corr = pitch_correlation(original_audio, reconstructed_audio, sr)
        mse_spec = mse_spectrogram(original_audio, reconstructed_audio, sr)
        
        return {
            "chroma_distance": chroma_dist,
            "onset_accuracy": onset_acc,
            "pitch_correlation": pitch_corr if not np.isnan(pitch_corr) else 0.0,
            "mse_spectrogram": mse_spec
        }
    except Exception as e:
        print(f"⚠️ Error calculating metrics: {e}")
        return {
            "chroma_distance": float('inf'),
            "onset_accuracy": 0.0,
            "pitch_correlation": 0.0,
            "mse_spectrogram": float('inf')
        }

# ===========================
# MAIN PROCESSING FUNCTION
# ===========================

def process_test_set_with_dataloader(test_dir, output_dir):
    """
    Process test set using dataloader for consistent preprocessing
    """
    # Create output directories
    piano_reconstruction_dir = os.path.join(output_dir, "piano_reconstruction")
    violin_reconstruction_dir = os.path.join(output_dir, "violin_reconstruction")
    Path(piano_reconstruction_dir).mkdir(parents=True, exist_ok=True)
    Path(violin_reconstruction_dir).mkdir(parents=True, exist_ok=True)
    
    # Load models
    print("🔧 Loading models...")
    
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
        cnn_out_dim=TRANSFORMER_DIM,
        transformer_dim=TRANSFORMER_DIM,
        num_heads=4,
        num_layers=4
    ).to(DEVICE)
    
    # Load pre-trained weights
    checkpoint_path = os.path.join(SAVED_MODELS_DIR, f"checkpoint_epoch_100.pth")
    
    if os.path.exists(checkpoint_path):
        print(f"📂 Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        
        try:
            content_encoder.load_state_dict(checkpoint['content_encoder'])
            style_encoder.load_state_dict(checkpoint['style_encoder'])
            decoder.load_state_dict(checkpoint['decoder'])
            print("✅ All models loaded successfully!")
        except Exception as e:
            print(f"⚠️ Error loading checkpoint: {e}")
            print("🔧 Using randomly initialized models...")
    else:
        print(f"⚠️ Checkpoint not found: {checkpoint_path}")
        print("🔧 Using randomly initialized models...")
    
    # Create test dataloader
    print("🔧 Creating test dataloader...")
    from dataloader import get_dataloader
    
    piano_test_dir = os.path.join(test_dir, "piano")
    violin_test_dir = os.path.join(test_dir, "violin")
    
    # Check if test directories exist
    if not os.path.exists(piano_test_dir) or not os.path.exists(violin_test_dir):
        raise FileNotFoundError(f"Test directories not found: {piano_test_dir}, {violin_test_dir}")
    
    # Use small batch size for testing
    test_loader = get_dataloader(
        piano_dir=piano_test_dir,
        violin_dir=violin_test_dir,
        batch_size=2,  # Small batch for testing
        shuffle=False,  # No shuffle for reproducible results
        stats_path="stats_stft_cqt.npz"  # Use existing stats
    )
    
    print(f"📊 Test dataloader created with {len(test_loader)} batches")
    
    # Generate class embeddings using the first batch
    print("🔧 Generating class embeddings...")
    class_embeddings = generate_class_embeddings_from_dataloader(
        style_encoder, test_loader, DEVICE
    )
    
    # Process all batches
    metrics = {
        "piano_reconstruction": [],
        "violin_reconstruction": []
    }
    
    # Set models to eval mode
    content_encoder.eval()
    decoder.eval()
    style_encoder.eval()
    
    print("🚀 Starting evaluation...")
    with torch.no_grad():
        for batch_idx, (sections, labels) in enumerate(test_loader):
            sections = sections.to(DEVICE)  # Shape: (B, S, 2, T, F)
            labels = labels.to(DEVICE)      # Shape: (B,)
            
            print(f"\n🔄 Processing batch {batch_idx + 1}/{len(test_loader)}")
            print(f"   Batch shape: {sections.shape}")
            print(f"   Labels: {labels}")
            
            # Process each sample in the batch
            for i in range(sections.size(0)):
                sample_sections = sections[i:i+1]  # (1, S, 2, T, F)
                sample_label = labels[i].item()
                
                # Get content embedding
                content_emb = content_encoder(sample_sections)  # (1, S, D)
                
                # Determine source class
                source_class = "piano" if sample_label == 0 else "violin"
                
                # Get class embedding
                class_emb = class_embeddings[source_class].unsqueeze(0).to(DEVICE)
                
                # Expand class embedding to match batch size
                B = content_emb.size(0)
                class_emb_expanded = class_emb.repeat(B, 1)
                
                # Decode using the new decoder
                # Extract STFT part from original sections for target_length
                stft_sections = sample_sections[:, :, :, :, :513]  # (1, S, 2, T, 513)
                target_length = stft_sections.size(1)  # S
                
                reconstructed_stft = decoder(
                    content_emb, 
                    class_emb_expanded, 
                    target_length=target_length
                )  # (1, S, 2, T, 513)
                
                # Convert back to audio for metrics calculation
                reconstructed_audio = reconstruct_audio_from_sections(
                    reconstructed_stft, batch_idx, i
                )
                original_audio = reconstruct_audio_from_sections(
                    stft_sections, batch_idx, i
                )
                
                # Calculate metrics
                try:
                    metrics_result = calculate_reconstruction_metrics(
                        original_audio, reconstructed_audio, SAMPLE_RATE
                    )
                    
                    if source_class == "piano":
                        metrics["piano_reconstruction"].append(metrics_result)
                    else:
                        metrics["violin_reconstruction"].append(metrics_result)
                    
                    print(f"   Sample {i} ({source_class}):")
                    for metric_name, value in metrics_result.items():
                        if np.isfinite(value):
                            print(f"     {metric_name}: {value:.4f}")
                        else:
                            print(f"     {metric_name}: {value}")
                    
                    # Save reconstructed audio
                    output_subdir = piano_reconstruction_dir if source_class == "piano" else violin_reconstruction_dir
                    output_filename = f"{source_class}_batch{batch_idx}_sample{i}_reconstructed.wav"
                    output_path = os.path.join(output_subdir, output_filename)
                    
                    # Ensure audio is not empty
                    if len(reconstructed_audio) > 0:
                        sf.write(output_path, reconstructed_audio, SAMPLE_RATE)
                        print(f"     Saved: {output_filename}")
                    else:
                        print(f"     ⚠️ Skipped saving empty audio: {output_filename}")
                    
                except Exception as e:
                    print(f"   ⚠️ Error processing sample {i}: {e}")
    
    # Calculate aggregate statistics
    print_aggregate_statistics(metrics)
    
    return metrics

# ===========================
# STATISTICS FUNCTIONS
# ===========================

def print_aggregate_statistics(metrics):
    """
    Print aggregate statistics for all metrics
    """
    print("\n" + "="*60)
    print("📊 AGGREGATE STATISTICS")
    print("="*60)
    
    for transformation in metrics:
        print(f"\n📈 Statistics for {transformation.replace('_', ' ').title()}:")
        if not metrics[transformation]:
            print("   No data available")
            continue
            
        metric_keys = metrics[transformation][0].keys()
        for metric in metric_keys:
            values = [
                r[metric] for r in metrics[transformation] 
                if r[metric] is not None and np.isfinite(r[metric])
            ]
            
            if values:
                print(f"   {metric.replace('_', ' ').title()}:")
                print(f"     Mean = {np.mean(values):.4f}")
                print(f"     Std = {np.std(values):.4f}")
                print(f"     Min = {np.min(values):.4f}")
                print(f"     Max = {np.max(values):.4f}")
                print(f"     Valid samples = {len(values)}")
            else:
                print(f"   {metric.replace('_', ' ').title()}: No valid values")

# ===========================
# MAIN EXECUTION
# ===========================

if __name__ == "__main__":
    print("🎵 Starting Audio Style Transfer Evaluation")
    print(f"🔧 Device: {DEVICE}")
    print(f"📂 Test directory: {TEST_DIR}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    
    try:
        # Run evaluation with dataloader
        metrics = process_test_set_with_dataloader(TEST_DIR, OUTPUT_DIR)
        
        # Save results to JSON
        print("\n💾 Saving results...")
        results_path = os.path.join(OUTPUT_DIR, "evaluation_results.json")
        
        # Convert numpy values to Python types for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif np.isnan(obj) or np.isinf(obj):
                return None
            return obj
        
        # Convert metrics
        json_metrics = {}
        for key, value_list in metrics.items():
            json_metrics[key] = [
                {k: convert_for_json(v) for k, v in item.items()}
                for item in value_list
            ]
        
        with open(results_path, 'w') as f:
            json.dump(json_metrics, f, indent=2)
        
        print(f"📊 Results saved to: {results_path}")
        print("\n🎉 Evaluation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()