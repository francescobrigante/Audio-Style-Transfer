import os
import random
from pydub import AudioSegment

# FFmpeg Configuration 
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg-7.1.1-essentials-shared-win-arm64\bin"
AudioSegment.converter = r"C:\ffmpeg-7.1.1-essentials-shared-win-arm64\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\ffmpeg-7.1.1-essentials-shared-win-arm64\bin\ffprobe.exe"

# Input and output directories 
input_dir = r"C:\Users\Lucia\Desktop\Uni\DL\Dataset\ViolinEtudes_Unito"
output_dir = r"C:\Users\Lucia\Desktop\Uni\DL\Dataset\ViolinEtudesPronto"

# Audio segmentation parameters 
segment_duration_ms = 10_000       
min_distance_ms = 25_000           
global_track_index = 1

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Returns the number of segments to export, given the length of the trace
def get_num_segments(duration_ms):
    if duration_ms < 40_000:
        return 1
    elif duration_ms < 70_000:
        return 2
    elif duration_ms < 200_000:
        return 3
    elif duration_ms < 300_000:
        return 4
    else:
        return 5 + (duration_ms // 120_000)


for filename in sorted(os.listdir(input_dir)):
    if not filename.lower().endswith(".mp3"):
        continue

    audio_path = os.path.join(input_dir, filename)
    audio = AudioSegment.from_mp3(audio_path)
    duration_ms = len(audio)

    num_segments = get_num_segments(duration_ms)
    available_range = duration_ms - segment_duration_ms

    used_starts = []
    segments_extracted = 0
    attempts = 0
    max_attempts = num_segments * 4  # Allow extra random attempts if overlap occurs

    '''Extracts random portions of audio, preventing segments 
       from being too close together, for up to 4 attempts'''
    while segments_extracted < num_segments and attempts < max_attempts:
        start = random.randint(0, available_range)

        # Ensure that this new segment does not overlap with already extracted ones
        if all(abs(start - s) >= min_distance_ms for s in used_starts):
            clip = audio[start:start + segment_duration_ms]
            segment_filename = f"{global_track_index}_{segments_extracted + 1}.mp3"
            output_path = os.path.join(output_dir, segment_filename)

            clip.export(output_path, format="mp3")
            used_starts.append(start)
            segments_extracted += 1

        attempts += 1

    global_track_index += 1

