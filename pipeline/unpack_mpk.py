import os
import subprocess
import tempfile

from amadeus import paths
from pipeline.games import archive_path
from pipeline.mpk import MpkArchive

# Upped the limit to 50 files
LIMIT = 50


def extract_kurisu_voices():
    mpk_path = archive_path("sg", "voice")
    out_dir = paths.RAW_AUDIO_DIR
    os.makedirs(out_dir, exist_ok=True)

    if not mpk_path.exists():
        print(f"Error: Could not find {mpk_path}")
        return

    with MpkArchive(mpk_path) as archive:
        print(f"Archive contains {len(archive)} audio files. Searching for Kurisu (CRS)...")
        entries = archive.filter("CRS*")[:LIMIT]

        with tempfile.TemporaryDirectory() as tmp:
            for entry in entries:
                print(f"Extracting: {entry.name}")
                temp_ogg = archive.extract(entry, tmp)
                subprocess.run([
                    "ffmpeg", "-y", "-i", str(temp_ogg),
                    "-ar", "44100", "-ac", "1", str(out_dir / entry.name)
                ], stderr=subprocess.DEVNULL)

    print("\nExtraction complete! Check your data/raw_audio folder.")


if __name__ == "__main__":
    extract_kurisu_voices()
