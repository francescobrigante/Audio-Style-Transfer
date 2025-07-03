import os
import random
from pydub import AudioSegment

# FFmpeg Configuration 
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg-7.1.1-essentials-shared-win-arm64\bin"
AudioSegment.converter = r"C:\ffmpeg-7.1.1-essentials-shared-win-arm64\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\ffmpeg-7.1.1-essentials-shared-win-arm64\bin\ffprobe.exe"

# Input and output directories 
source_directory = r"C:\Users\Lucia\Desktop\Uni\DL\Dataset\PianoMotion10M_selected"
target_directory = r"C:\Users\Lucia\Desktop\Uni\DL\Dataset\PianoMotion10M_ready"
os.makedirs(target_directory, exist_ok=True)

segment_duration_ms = 10_000  # =10 seconds

global_index = 1
number_audio = 768


'''Takes 768 tracks (like the number of tracks in the violin datset) and for 
   each of them extracts a segment from the 10-second center'''

for filename in sorted(os.listdir(source_directory)):
    if not filename.lower().endswith(".mp3"):
        continue

    filepath = os.path.join(source_directory, filename) 
    
    if number_audio > 0:    
        audio = AudioSegment.from_mp3(filepath)
        duration = len(audio)  

        splits = []
        
        center = duration // 2
        start = max(0, center - segment_duration_ms // 2)
        splits = [start]

        
        clip = audio[start : start + segment_duration_ms]
        output_name = f"{global_index}.mp3"
        output_path = os.path.join(target_directory, output_name)
        clip.export(output_path, format="mp3", bitrate="192k")

        global_index += 1
        number_audio -= 1
    

    