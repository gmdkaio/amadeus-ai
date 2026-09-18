import os
import struct
import subprocess

def extract_kurisu_voices():
    mpk_path = "/home/kaio/.local/share/Steam/steamapps/common/STEINS;GATE/USRDIR/voice.mpk"
    out_dir = "/home/kaio/amadeus-ai/data/raw_audio"
    
    if not os.path.exists(mpk_path):
        print(f"Error: Could not find {mpk_path}")
        return

    print("Cracking MAGES. MPK Archive Header...")
    with open(mpk_path, "rb") as f:
        magic = f.read(4)
        if magic != b'MPK\x00':
            print("Invalid archive format!")
            return
            
        f.seek(8)
        file_count = struct.unpack("<Q", f.read(8))[0]
        print(f"Archive contains {file_count} audio files. Searching for Kurisu (CRS)...")
        
        f.seek(64)
        
        extracted = 0
        for _ in range(file_count):
            entry = f.read(256)
            if not entry: break
            
            is_comp, idx, offset, size, uncomp_size = struct.unpack("<IIQQQ", entry[:32])
            filename = entry[32:].split(b'\x00', 1)[0].decode('utf-8', errors='ignore')
            
            if "CRS" in filename.upper():
                current_pos = f.tell()
                
                print(f"Extracting: {filename}")
                f.seek(offset)
                audio_data = f.read(size)
                
                temp_ogg = os.path.join(out_dir, "temp.ogg")
                out_wav = os.path.join(out_dir, filename.replace('.ogg', '.wav'))
                
                with open(temp_ogg, "wb") as out:
                    out.write(audio_data)
                    
                subprocess.run([
                    "ffmpeg", "-y", "-i", temp_ogg, 
                    "-ar", "44100", "-ac", "1", out_wav
                ], stderr=subprocess.DEVNULL)
                
                os.remove(temp_ogg)
                extracted += 1
                
                f.seek(current_pos)
                
                # Upped the limit to 50 files
                if extracted >= 50:
                    break
                    
    print("\nExtraction complete! Check your data/raw_audio folder.")

if __name__ == "__main__":
    extract_kurisu_voices()
