### PENSATO PER COLAB + GOOGLE DRIVE, IN CASO DA MODIFICARE

# Installa le dipendenze necessarie
!pip install torch torchaudio librosa soundfile

import torch
import torchaudio
import librosa
import numpy as np
import soundfile as sf
import os
import glob
from pathlib import Path
import random
from IPython.display import Audio, display
from google.colab import drive

### Sostituire nomi con quelli dei file corretti
from content_encoder import ContentEncoder
from new_decoder import Decoder
from utilityFunctions import get_STFT, get_CQT, inverse_STFT, get_overlap_windows, sections2spectrogram, concat_stft_cqt

# Monta Google Drive
drive.mount('/content/drive')

# Path input/output dir
'''
TEST DIR:
/content/drive/MyDrive/test_dataset
          -> /piano
          -> /violin

OUTPUT DIR:

/content/drive/MyDrive/output
          -> /from_piano_to_violin
          -> /from_violin_to_piano
          (li crea dopo)
'''

TEST_DIR = "/content/drive/MyDrive/test_dataset"  # Sostituisci con il percorso su Google Drive
OUTPUT_DIR = "/content/drive/MyDrive/output"  # Sostituisci con il percorso di output


SAMPLES_PER_CLASS = 5  # Numero di campioni casuali per classe

### class_embeddings pensato come file .pth con tensore [2, d_enc]
path_class_embeddings = "/content/drive/MyDrive/class_embeddings.pth"


# Configurazioni
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


# Funzione di style transfer
def style_transfer(waveform, sr, content_encoder, decoder, class_embeddings, target_class_id):
    stft = get_STFT(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH).to(DEVICE)
    cqt = get_CQT(waveform, sample_rate=SAMPLE_RATE, n_bins=N_BINS, hop_length=HOP_LENGTH).to(DEVICE)
    input_spectrogram = concat_stft_cqt(stft, cqt)
    sections = get_overlap_windows(input_spectrogram, window_size=WINDOW_SIZE, overlap_frames=OVERLAP_FRAMES)
    sections = sections.unsqueeze(0)
    
    content_encoder.eval()
    with torch.no_grad():
        content_emb = content_encoder(sections)
    
    class_emb = class_embeddings[target_class_id].unsqueeze(0)
    
    decoder.eval()
    with torch.no_grad():
        output_stft = decoder(content_emb, class_emb, target_length=content_emb.size(1))
    
    output_stft = output_stft.squeeze(0)
    original_time = stft.size(1)
    full_spectrogram = sections2spectrogram(output_stft, original_size=original_time, overlap=OVERLAP_FRAMES)
    
    output_audio = inverse_STFT(full_spectrogram, n_fft=N_FFT, hop_length=HOP_LENGTH)
    
    return output_audio.cpu(), sr

# Funzione per processare il dataset di test
def process_test_set(test_dir, output_dir, samples_per_class=5):
    """
    Processa un numero specificato di file casuali da ogni cartella di test.
    
    Args:
        test_dir: str - directory del dataset di test con sottocartelle 'piano' e 'violin'
        output_dir: str - directory per salvare gli audio generati
        samples_per_class: int - numero di campioni casuali per classe da processare
    """
    # Crea directory di output (una per piano->violino + una violino -> piano)
    piano_to_violin_dir = os.path.join(output_dir, "from_piano_to_violin")
    violin_to_piano_dir = os.path.join(output_dir, "from_violin_to_piano")
    Path(piano_to_violin_dir).mkdir(parents=True, exist_ok=True)
    Path(violin_to_piano_dir).mkdir(parents=True, exist_ok=True)
    
    # Carica i modelli
    content_encoder = ContentEncoder().to(DEVICE)
    decoder = Decoder().to(DEVICE)
    
    ### Carica i pesi dei modelli (presi sempre dal drive)
    # content_encoder.load_state_dict(torch.load('/content/drive/MyDrive/content_encoder.pth'))
    # decoder.load_state_dict(torch.load('/content/drive/MyDrive/decoder.pth'))

    class_embeddings = torch.load(path_class_embeddings).to(DEVICE)
    
    # Directory delle classi
    piano_dir = os.path.join(test_dir, "piano")
    violin_dir = os.path.join(test_dir, "violin")
    
    # Ottieni i file audio e seleziona campioni casuali
    piano_files = glob.glob(os.path.join(piano_dir, "*.wav"))
    violin_files = glob.glob(os.path.join(violin_dir, "*.wav"))
    
    if len(piano_files) < samples_per_class or len(violin_files) < samples_per_class:
        raise ValueError(f"Non abbastanza file: piano ({len(piano_files)}), violino ({len(violin_files)})")
    
    piano_files = random.sample(piano_files, samples_per_class)
    violin_files = random.sample(violin_files, samples_per_class)
    
    # Processa i file
    print("Processamento file piano → violino:")
    for audio_path in piano_files:
        output_audio, sr = process_file(audio_path, content_encoder, decoder, class_embeddings, 
                                      source_class="piano", target_class_id=1, target_class="violin",
                                      output_dir=piano_to_violin_dir)
    
    print("\nProcessamento file violino → piano:")
    for audio_path in violin_files:
        output_audio, sr = process_file(audio_path, content_encoder, decoder, class_embeddings, 
                                      source_class="violin", target_class_id=0, target_class="piano",
                                      output_dir=violin_to_piano_dir)

def process_file(audio_path, content_encoder, decoder, class_embeddings, source_class, target_class_id, target_class, output_dir):
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] == 2:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    print(f"\nFile: {os.path.basename(audio_path)} ({source_class} → {target_class})")
    print("Audio originale:")
    display(Audio(waveform.numpy(), rate=sr))
    
    output_audio, sr = style_transfer(waveform, sr, content_encoder, decoder, class_embeddings, target_class_id)
    
    print(f"Audio con stile trasferito ({target_class}):")
    display(Audio(output_audio.numpy(), rate=sr))
    
    output_filename = f"{source_class}_to_{target_class}_{os.path.basename(audio_path)}"
    output_path = os.path.join(output_dir, output_filename)
    sf.write(output_path, output_audio.numpy(), sr)
    print(f"Salvato: {output_path}")
    
    return output_audio, sr


process_test_set(TEST_DIR, OUTPUT_DIR, samples_per_class=SAMPLES_PER_CLASS)