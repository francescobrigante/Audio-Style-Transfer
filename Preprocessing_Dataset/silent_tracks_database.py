import os
import numpy as np
import librosa

# Configuration
dataset_directory = r"C:\Users\Lucia\Desktop\Uni\DL\Dataset\PianoMotion10M_ready"
supported_extensions = (".wav", ".mp3")
rms_threshold = 0.005                    # RMS amplitude threshold below which a frame is considered silent
silence_ratio_threshold = 0.3           # Proportion of silent frames required to consider a track "silent"

flagged_tracks = []

# Silence Analysis 
for filename in sorted(os.listdir(dataset_directory)):
    if not filename.lower().endswith(supported_extensions):
        continue

    path = os.path.join(dataset_directory, filename)
    try:
        y, sr = librosa.load(path, sr=None)
        rms = librosa.feature.rms(y=y)[0]  # Extract RMS energy per frame

        silent_frames = np.sum(rms < rms_threshold)
        silence_ratio = silent_frames / len(rms)

        if silence_ratio >= silence_ratio_threshold:
            flagged_tracks.append((filename, silence_ratio))

    except Exception as e:
        print(f"[Error] Failed to process {filename}: {e}")

print("Critical tracks found:\n")
print(flagged_tracks)
