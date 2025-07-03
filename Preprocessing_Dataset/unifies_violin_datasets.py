import os
import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
from tqdm import tqdm
import tempfile

# FFmpeg Configuration
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg-7.1.1-essentials-shared-win-arm64\bin"
AudioSegment.converter = r"C:\ffmpeg-7.1.1-essentials-shared-win-arm64\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\ffmpeg-7.1.1-essentials-shared-win-arm64\bin\ffprobe.exe"

# Input and output directories
bach_dir = r"C:\Users\Lucia\Desktop\Uni\DL\Dataset\BachViolinDatsetPronto"
etudes_dir = r"C:\Users\Lucia\Desktop\Uni\DL\Dataset\ViolinEtudesPronto"
merged_dir = r"C:\Users\Lucia\Desktop\Uni\DL\Dataset\DatasetUnificato_44100"
os.makedirs(merged_dir, exist_ok=True)

# Audio preprocessing parameters 
target_sr = 44100      # Desired sample rate in Hz
target_rms = 0.07      # Target RMS amplitude for normalization

# Applies RMS normalization to an audio signal.
def rms_normalize(y, target_rms):
    current_rms = np.sqrt(np.mean(y ** 2))
    if current_rms == 0:
        return y  # Avoid division by zero
    gain = target_rms / current_rms
    return y * gain

def process_and_export(directory, prefix):
    """
    Processes all .mp3 files in a specified directory by:
      - Loading and converting to mono
      - Resampling to the target sample rate
      - RMS-normalizing to a fixed energy level
      - Exporting as high-quality .mp3 with consistent naming
    """
    for file in tqdm(os.listdir(directory), desc=f"[Processing] {prefix} dataset"):
        if not file.lower().endswith(".mp3"):
            continue

        path = os.path.join(directory, file)
        try:
            y, sr = librosa.load(path, sr=None)

            if sr != target_sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)

            y = rms_normalize(y, target_rms)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
                tmp_path = tmpfile.name
            sf.write(tmp_path, y, samplerate=target_sr)

            audio = AudioSegment.from_wav(tmp_path)
            new_name = f"{prefix}_{file}"
            destination = os.path.join(merged_dir, new_name)
            audio.export(destination, format="mp3", bitrate="192k")

            os.remove(tmp_path)

        except Exception as e:
            print(f"[Error] Processing failed for {file}: {e}")

# Normalization for both datasets 
process_and_export(bach_dir, "Bach")
process_and_export(etudes_dir, "Violin")
