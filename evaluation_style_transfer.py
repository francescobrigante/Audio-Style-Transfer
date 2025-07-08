import torch
import torchaudio
import librosa
import numpy as np
import os
import glob
from pathlib import Path
import random
from scipy.stats import pearsonr
from content_encoder import ContentEncoder
from new_decoder import Decoder
# from SimpleDecoder_TransformerOnly import Decoder
from style_encoder import StyleEncoder
from discriminator import Discriminator
from utilityFunctions import get_STFT, get_CQT, inverse_STFT, get_overlap_windows, sections2spectrogram, concat_stft_cqt
from torch.utils.data import DataLoader
from dataloader import DualInstrumentDataset, custom_collate_fn

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
TEST_DIR = r"dataset/test"
OUTPUT_DIR = r"result_evaluation_style_transfer"
SAVED_MODELS_DIR = r"checkpoints"


# funzione per generare class_embeddings dal primo batch
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

# per compatibilità delle chiavi in class_embeddings
id_to_name = {0: "piano", 1: "violin"}

def chroma_similarity(generated_audio, original_audio, sr=22050):
    try:
        chroma_gen = librosa.feature.chroma_stft(y=generated_audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        chroma_orig = librosa.feature.chroma_stft(y=original_audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH)
        min_len = min(chroma_gen.shape[1], chroma_orig.shape[1])
        chroma_gen = chroma_gen[:, :min_len]
        chroma_orig = chroma_orig[:, :min_len]
        similarities = [np.corrcoef(chroma_gen[i], chroma_orig[i])[0, 1] for i in range(chroma_gen.shape[0])]
        similarities = [s for s in similarities if not np.isnan(s)]  # Filtra i NaN
        if similarities:
            return np.mean(similarities)
        else:
            print(f"Warning: No valid chroma similarities computed for audio (all NaN). Returning 0.0")
            return 0.0
    except Exception as e:
        print(f"Error computing chroma similarity: {e}. Returning 0.0")
        return 0.0
    

def mfcc_distance(generated_audio, reference_audio, sr=22050, n_mfcc=13):
    try:
        mfcc_gen = librosa.feature.mfcc(y=generated_audio, sr=sr, n_mfcc=n_mfcc, hop_length=HOP_LENGTH)
        mfcc_ref = librosa.feature.mfcc(y=reference_audio, sr=sr, n_mfcc=n_mfcc, hop_length=HOP_LENGTH)
        min_len = min(mfcc_gen.shape[1], mfcc_ref.shape[1])
        mfcc_gen = mfcc_gen[:, :min_len]
        mfcc_ref = mfcc_ref[:, :min_len]
        return np.mean(np.sqrt(np.sum((mfcc_gen - mfcc_ref) ** 2, axis=0)))
    except Exception as e:
        print(f"Error computing MFCC distance: {e}. Returning None")
        return None
    
def instrumentation_similarity(audio1, audio2, sr=22050):
    S1 = np.abs(librosa.stft(audio1))
    S2 = np.abs(librosa.stft(audio2))
    energy1 = np.sum(S1, axis=1)
    energy2 = np.sum(S2, axis=1)

    min_len = min(len(energy1), len(energy2))
    corr, _ = pearsonr(energy1[:min_len], energy2[:min_len])
    return corr if not np.isnan(corr) else 0.0

def self_similarity_distance(audio1, audio2, sr=22050):
    mfcc1 = librosa.feature.mfcc(y=audio1, sr=sr)
    mfcc2 = librosa.feature.mfcc(y=audio2, sr=sr)

    SSM1 = librosa.segment.recurrence_matrix(mfcc1.T)
    SSM2 = librosa.segment.recurrence_matrix(mfcc2.T)

    min_len = min(SSM1.shape[0], SSM2.shape[0])
    SSM1 = SSM1[:min_len, :min_len].astype(int)
    SSM2 = SSM2[:min_len, :min_len].astype(int)

    dist = np.mean(np.abs(SSM1 - SSM2))
    return dist

def process_audio(waveform, sr, content_encoder, decoder, class_embeddings, target_class_id):
    stft = get_STFT(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH).to(DEVICE)
    cqt = get_CQT(waveform, sample_rate=SAMPLE_RATE, n_bins=N_BINS, hop_length=HOP_LENGTH).to(DEVICE)
    input_spectrogram = concat_stft_cqt(stft, cqt)
    sections = get_overlap_windows(input_spectrogram, window_size=WINDOW_SIZE, overlap_frames=OVERLAP_FRAMES)
    sections = sections.unsqueeze(0)
    
    content_encoder.eval()
    with torch.no_grad():
        content_emb = content_encoder(sections)
    
    class_name = id_to_name[target_class_id]
    class_emb = class_embeddings[class_name].unsqueeze(0)
    
    decoder.eval()
    with torch.no_grad():
        output_stft = decoder(content_emb, class_emb, target_length=content_emb.size(1))
    
    output_stft = output_stft.squeeze(0)
    original_time = stft.size(1)
    full_spectrogram = sections2spectrogram(output_stft, original_size=original_time, overlap=OVERLAP_FRAMES)
    
    output_audio = inverse_STFT(full_spectrogram, n_fft=N_FFT, hop_length=HOP_LENGTH)
    
    return output_audio.cpu(), sr

def eval_style_transfer(audio_path, content_encoder, style_encoder, decoder, discriminator, class_embeddings, source_class, target_class_id, target_class, output_dir):
    waveform, sr = torchaudio.load(audio_path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        sr = SAMPLE_RATE
    if waveform.shape[0] == 2:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    waveform_np = waveform[0].numpy()
    
    x_con_st_tr, sr = process_audio(waveform, sr, content_encoder, decoder, class_embeddings, target_class_id)
    x_con_st_tr_np = x_con_st_tr.numpy()
    
    reference_files = glob.glob(os.path.join(TEST_DIR, target_class, "*.mp3"))
    if not reference_files:
        print(f"Notice: No audio reference for {target_class} in {TEST_DIR}/{target_class}/")
        reference_audio = None
    else:
        reference_path = random.choice(reference_files)
        reference_audio, ref_sr = torchaudio.load(reference_path)
        if ref_sr != SAMPLE_RATE:
            reference_audio = torchaudio.functional.resample(reference_audio, ref_sr, SAMPLE_RATE)
        reference_audio = reference_audio.mean(dim=0).numpy() if reference_audio.shape[0] == 2 else reference_audio[0].numpy()
    
    st_metrics = eval_style_transfer_metrics(x_con_st_tr_np, waveform_np, reference_audio, sr=sr)
    
    output_filename = f"{source_class}_to_{target_class}_{os.path.basename(audio_path)}.txt"
    output_path = os.path.join(output_dir, output_filename)
    save_metrics(st_metrics, output_path)
    
    return st_metrics

def eval_style_transfer_metrics(generated_audio, original_audio, reference_audio, sr):
    # Computes chroma similarity between the generated audio and the original
    chroma_sim = chroma_similarity(generated_audio, original_audio, sr=sr)
    
    # Computes MFCC distance with respect to the reference audio (target instrument)
    mfcc_dist = mfcc_distance(generated_audio, reference_audio, sr=sr) if reference_audio is not None else None
        
    # Computes instrumentation similarity with respect to the reference audio (target instrument)
    instr_sim = instrumentation_similarity(generated_audio, reference_audio, sr=sr) if reference_audio is not None else None
    
    # Computes the self-similarity matrix distance with respect to the reference audio (target instrument)
    self_sim_dist = self_similarity_distance(generated_audio, reference_audio, sr=sr) if reference_audio is not None else None

    return {
        "chroma_similarity": chroma_sim,
        "mfcc_distance": mfcc_dist,
        "instrumentation_similarity": instr_sim,
        "self_similarity_distance": self_sim_dist
    }


def process_test_set(test_dir, output_dir, batch_size=8):
    piano_to_violin_dir = os.path.join(output_dir, "from_piano_to_violin")
    violin_to_piano_dir = os.path.join(output_dir, "from_violin_to_piano")
    Path(piano_to_violin_dir).mkdir(parents=True, exist_ok=True)
    Path(violin_to_piano_dir).mkdir(parents=True, exist_ok=True)
    
    content_encoder = ContentEncoder(
        cnn_out_dim=TRANSFORMER_DIM,
        transformer_dim=TRANSFORMER_DIM,
        num_heads=4,
        num_layers=4,
        # channels_list=[16, 32, 64, 128, 256]          old example old model
        channels_list = [32, 64, 128, 256, 512, 512]
    ).to(DEVICE)
    style_encoder = StyleEncoder(
        cnn_out_dim=TRANSFORMER_DIM,
        transformer_dim=TRANSFORMER_DIM,
        num_heads=4,
        num_layers=4
    ).to(DEVICE)
    decoder = Decoder(
        d_model=TRANSFORMER_DIM,
        nhead=4,
        num_layers=4
    ).to(DEVICE)
    discriminator = Discriminator(
        input_dim=TRANSFORMER_DIM,
        hidden_dim=128
    ).to(DEVICE)
    
    # load pre-trained weights here
    checkpoint_path = os.path.join(SAVED_MODELS_DIR, 'NEWDECODERcheckpoint_epoch_70.pth')
    
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    content_encoder.load_state_dict(checkpoint['content_encoder'])
    style_encoder.load_state_dict(checkpoint['style_encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    discriminator.load_state_dict(checkpoint['discriminator'])
    #class_embeddings = torch.load(path_class_embeddings).to(DEVICE)
    
    piano_dir = os.path.join(test_dir, "piano")
    violin_dir = os.path.join(test_dir, "violin")
    
    test_dataset = DualInstrumentDataset(
        piano_dir=piano_dir,
        violin_dir=violin_dir,
        use_separate_stats=True
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=custom_collate_fn
    )
    
    metrics = {
        "piano_to_violin": [],
        "violin_to_piano": []
    }

    
    # Genera le class embeddings dal test_loader (prima batch bilanciata piano/violino)
    class_embeddings = generate_class_embeddings_from_dataloader(style_encoder, test_dataloader, DEVICE)
    
    #print(f"Processing test dataset ({len(test_dataset)} samples):")
    for batch_idx, (sections, labels) in enumerate(test_dataloader):
        sections = sections.to(DEVICE)
        labels = labels.to(DEVICE)
        
        batch_size_actual = sections.size(0)
        half_batch = batch_size_actual // 2
        
        #print(f"\nProcessing piano samples in batch {batch_idx + 1} ({half_batch} samples):")
        for i in range(half_batch):
            piano_sections = sections[i]
            piano_label = labels[i]
            
            num_sections = piano_sections.size(0)
            section_time_frames = piano_sections.size(2)
            total_time_frames = (num_sections - 1) * (section_time_frames - OVERLAP_FRAMES) + section_time_frames
            original_size = total_time_frames
            
            full_spectrogram = sections2spectrogram(piano_sections, original_size=original_size, overlap=OVERLAP_FRAMES)
            #print(f"piano full_spectrogram shape: {full_spectrogram.shape}")
            
            stft_bins = N_FFT // 2 + 1
            stft_spectrogram = full_spectrogram[:, :, :stft_bins]
            #print(f"piano stft_spectrogram shape: {stft_spectrogram.shape}")
            
            waveform = inverse_STFT(stft_spectrogram, n_fft=N_FFT, hop_length=HOP_LENGTH)
            waveform_np = waveform.cpu().numpy()
            #print(f"Waveform length: {len(waveform_np)}")
            
            x_con_st_tr, sr = process_audio(
                waveform, SAMPLE_RATE, content_encoder, decoder, class_embeddings, target_class_id=1
            )
            x_con_st_tr_np = x_con_st_tr.numpy()
            
            reference_files = glob.glob(os.path.join(test_dir, "violin", "*.mp3"))
            if not reference_files:
                print(f"Notice: No audio reference for violin in {test_dir}/violin/")
                reference_audio = None
            else:
                reference_path = random.choice(reference_files)
                reference_audio, ref_sr = torchaudio.load(reference_path)
                if ref_sr != SAMPLE_RATE:
                    reference_audio = torchaudio.functional.resample(reference_audio, ref_sr, SAMPLE_RATE)
                reference_audio = reference_audio.mean(dim=0).numpy() if reference_audio.shape[0] == 2 else reference_audio[0].numpy()
            
            print(f"Generated audio length: {len(x_con_st_tr_np)}, Reference audio length: {len(reference_audio) if reference_audio is not None else 'None'}")
            st_metrics = eval_style_transfer_metrics(
                x_con_st_tr_np, waveform_np, reference_audio, sr=SAMPLE_RATE
            )
            
            output_filename = f"piano_to_violin_batch{batch_idx}_sample{i}.txt"
            output_path = os.path.join(piano_to_violin_dir, output_filename)
            save_metrics(st_metrics, output_path)
            metrics["piano_to_violin"].append(st_metrics)
        
        #print(f"Processing violin samples in batch {batch_idx + 1} ({half_batch} samples):")
        for i in range(half_batch, batch_size_actual):
            violin_sections = sections[i]
            violin_label = labels[i]
            
            num_sections = violin_sections.size(0)
            section_time_frames = violin_sections.size(2)
            total_time_frames = (num_sections - 1) * (section_time_frames - OVERLAP_FRAMES) + section_time_frames
            original_size = total_time_frames
            
            full_spectrogram = sections2spectrogram(violin_sections, original_size=original_size, overlap=OVERLAP_FRAMES)
            #print(f"violin full_spectrogram shape: {full_spectrogram.shape}")
            
            stft_bins = N_FFT // 2 + 1
            stft_spectrogram = full_spectrogram[:, :, :stft_bins]
            #print(f"violin stft_spectrogram shape: {stft_spectrogram.shape}")
            
            waveform = inverse_STFT(stft_spectrogram, n_fft=N_FFT, hop_length=HOP_LENGTH)
            waveform_np = waveform.cpu().numpy()
            #print(f"Waveform length: {len(waveform_np)}")
            
            x_con_st_tr, sr = process_audio(
                waveform, SAMPLE_RATE, content_encoder, decoder, class_embeddings, target_class_id=0
            )
            x_con_st_tr_np = x_con_st_tr.numpy()
            
            reference_files = glob.glob(os.path.join(test_dir, "piano", "*.mp3"))
            if not reference_files:
                print(f"Notice: No audio reference for piano in {test_dir}/piano/")
                reference_audio = None
            else:
                reference_path = random.choice(reference_files)
                reference_audio, ref_sr = torchaudio.load(reference_path)
                if ref_sr != SAMPLE_RATE:
                    reference_audio = torchaudio.functional.resample(reference_audio, ref_sr, SAMPLE_RATE)
                reference_audio = reference_audio.mean(dim=0).numpy() if reference_audio.shape[0] == 2 else reference_audio[0].numpy()
            
            print(f"Generated audio length: {len(x_con_st_tr_np)}, Reference audio length: {len(reference_audio) if reference_audio is not None else 'None'}")
            st_metrics = eval_style_transfer_metrics(
                x_con_st_tr_np, waveform_np, reference_audio, sr=SAMPLE_RATE
            )
            
            output_filename = f"violin_to_piano_batch{batch_idx}_sample{i-half_batch}.txt"
            output_path = os.path.join(violin_to_piano_dir, output_filename)
            save_metrics(st_metrics, output_path)
            metrics["violin_to_piano"].append(st_metrics)
    return metrics

def eval_style_transfer_metrics(generated_audio, original_audio, reference_audio, sr):
   # Compute chroma similarity between generated audio and the original
    chroma_sim = chroma_similarity(generated_audio, original_audio, sr=sr)

    # Compute MFCC distance with respect to the reference audio (target instrument)
    mfcc_dist = mfcc_distance(generated_audio, reference_audio, sr=sr) if reference_audio is not None else None

    # Compute instrumentation similarity with respect to the reference audio (target instrument)
    instr_sim = instrumentation_similarity(generated_audio, reference_audio, sr=sr) if reference_audio is not None else None

    # Compute self-similarity matrix distance with respect to the reference audio (target instrument)
    self_sim_dist = self_similarity_distance(generated_audio, reference_audio, sr=sr) if reference_audio is not None else None
    
    return {
        "chroma_similarity": chroma_sim,
        "mfcc_distance": mfcc_dist,
        "instrumentation_similarity": instr_sim,
        "self_similarity_distance": self_sim_dist
    }

def save_metrics(metrics, output_path):
    with open(output_path, 'w') as f:
        f.write(f" - Chroma Similarity: {metrics['chroma_similarity']:.4f}\n")
        if metrics['mfcc_distance'] is not None:
            f.write(f" - MFCC Distance: {metrics['mfcc_distance']:.4f}\n")
        else:
            f.write(" - MFCC Distance: None\n")
        if metrics['instrumentation_similarity'] is not None:
            f.write(f" - Instrumentation Similarity: {metrics['instrumentation_similarity']:.4f}\n")
        else:
            f.write(" - Instrumentation Similarity: None\n")
        if metrics['self_similarity_distance'] is not None:
            f.write(f" - Self Similarity Distance: {metrics['self_similarity_distance']:.4f}\n")
        else:
            f.write(" - Self Similarity Distance: None\n")
    print(f"Saved results: {output_path}")

def save_global_statistics(metrics_dict, output_dir, filename="global_statistics.txt"):
    stats_path = os.path.join(output_dir, filename)

    with open(stats_path, 'w') as f:
        f.write("=== Global Style Transfer Statistics ===\n\n")

        for direction in metrics_dict:
            f.write(f"Transformation: {direction.replace('_', ' ').title()}\n")
            entries = metrics_dict[direction]
            if not entries:
                f.write("  No data available.\n\n")
                continue

            metric_keys = entries[0].keys()
            for metric in metric_keys:
                values = [entry[metric] for entry in entries if entry[metric] is not None]
                if values:
                    mean = np.mean(values)
                    std = np.std(values)
                    f.write(f"  - {metric.replace('_', ' ').title()}: Mean = {mean:.4f}, Std = {std:.4f}\n")
                else:
                    f.write(f"  - {metric.replace('_', ' ').title()}: None\n")
            f.write("\n")

    print(f"📊 Global statistics saved to: {stats_path}")
    

if __name__ == "__main__":
    all_metrics = process_test_set(TEST_DIR, OUTPUT_DIR, batch_size=8)
    save_global_statistics(all_metrics, OUTPUT_DIR)

