import os
import subprocess
import glob

from amadeus import paths

def prepare_voice_dataset():
    audio_dir = str(paths.RAW_AUDIO_DIR)
    output_wav = str(paths.DATA_DIR / 'kurisu_reference.wav')
    list_file = str(paths.DATA_DIR / 'audio_list.txt')
    
    ogg_files = sorted(glob.glob(os.path.join(audio_dir, 'CRS_*.OGG')))
    selected_files = ogg_files[:300]
    
    print(f"Selected {len(selected_files)} audio files for the training set.")
    
    with open(list_file, 'w') as f:
        for audio_file in selected_files:
            f.write(f"file '{audio_file}'\n")
            
    print("Concatenating audio with FFmpeg...")
    
    try:
        subprocess.run([
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', list_file, 
            '-ar', '44100', '-ac', '1', '-c:a', 'pcm_s16le', output_wav
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"Success! Reference audio saved to {output_wav}")
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg failed: {e}")
        
    if os.path.exists(list_file):
        os.remove(list_file)

if __name__ == '__main__':
    prepare_voice_dataset()