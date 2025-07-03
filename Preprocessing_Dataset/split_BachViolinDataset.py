import os
import numpy as np
from pydub import AudioSegment

# FFmpeg Configuration
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg-7.1.1-essentials-shared-win-arm64\bin"
AudioSegment.converter = r"C:\ffmpeg-7.1.1-essentials-shared-win-arm64\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\ffmpeg-7.1.1-essentials-shared-win-arm64\bin\ffprobe.exe"

# Input and output directories
input_folder = r"C:\Users\Lucia\Desktop\Uni\DL\Dataset\BachVilinDatset_Unito"
output_folder = r"C:\Users\Lucia\Desktop\Uni\DL\Dataset\BachViolinDatsetPronto"
os.makedirs(output_folder, exist_ok=True)

# Audio segmentation parameters 
segment_length = 10_000       
skip_ms = 15_000             
frame_size = 100             
silence_threshold = -45       
min_sound_ratio = 0.6         
count = 1                    

# To verify that the track contains mainly sound
def is_mostly_sound(clip):
    num_frames = segment_length // frame_size
    sound_frames = sum(
        clip[i * frame_size:(i + 1) * frame_size].dBFS > silence_threshold
        for i in range(num_frames)
    )
    return (sound_frames / num_frames) >= min_sound_ratio

# Returns the number of segments to export, given the length of the trace
def get_num_segments(duration_ms):
    if duration_ms < 120_000:
        return 2
    elif duration_ms < 300_000:
        return 4
    elif duration_ms < 1_020_000:
        return 8
    else:
        return 10


for filename in sorted(os.listdir(input_folder)):
    if not filename.lower().endswith(".mp3"):
        continue

    filepath = os.path.join(input_folder, filename)
    audio = AudioSegment.from_mp3(filepath)
    duration = len(audio)
    num_segments = get_num_segments(duration)
    effective_duration = duration - 2 * skip_ms

    interval = (effective_duration - segment_length) // (num_segments - 1) if num_segments > 1 else 0
    valid_segments = 0
    attempts = 0
    max_attempts = 20

    # It try to find a mostly sound segment for 20 times before skip it 
    for i in range(num_segments):
        success = False
        start = skip_ms + i * interval

        while attempts < max_attempts:
            clip = audio[start:start + segment_length]
            if is_mostly_sound(clip):
                clip.export(os.path.join(output_folder, f"{count}_{i + 1}.mp3"), format="mp3")
                valid_segments += 1
                success = True
                break
            else:
                start += 1000  
                attempts += 1

        if not success:
            print(f"Silent segment detected and skipped at index {i+1} in {filename}")

    print(f"Valid segments extracted from {filename}: {valid_segments}")

    count += 1

