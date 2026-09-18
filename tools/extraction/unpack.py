import os
import subprocess
from pathlib import Path

def extract_kurisu_audio():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(base_dir, 'data', 'raw_audio')
    os.makedirs(output_dir, exist_ok=True)

    # Prompt for the Steins;Gate game directory (e.g., your Steam library path)
    game_dir = input("Enter the path to your Steins;Gate 'voice' folder (or game root): ").strip()
    
    if not os.path.exists(game_dir):
        print("Error: Directory not found.")
        return

    print(f"Scanning {game_dir} for Kurisu's voice files...")
    
    # In S;G, Kurisu's voice lines generally have specific prefixes (like SG01, CRS, etc.)
    # We will grab a sample of them and ensure they are 24kHz/44kHz WAVs for GPT-SoVITS
    count = 0
    for file_path in Path(game_dir).rglob('*.*'):
        file_name = file_path.name.upper()
        # Look for typical Kurisu voice prefixes or just grab the SG01_01 series
        if file_name.startswith("SG01_") and file_name.endswith((".OGG", ".WAV")):
            out_file = os.path.join(output_dir, f"{file_path.stem}.wav")
            
            # Use FFmpeg to convert/copy to standard WAV format
            cmd = [
                "ffmpeg", "-y", "-i", str(file_path),
                "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "1",
                "-loglevel", "error", out_file
            ]
            subprocess.run(cmd)
            count += 1
            print(f"Extracted: {out_file}")
            
            # We only need a handful to pick a good reference clip
            if count >= 50:
                break
                
    print(f"\nExtraction complete! {count} voice files saved to {output_dir}")

if __name__ == "__main__":
    extract_kurisu_audio()
