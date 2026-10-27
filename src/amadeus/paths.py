import os
from pathlib import Path

# Everything big (data, models, checkpoints) lives under AMADEUS_HOME.
# Defaults to the repo root until it moves to /mnt/HDD/amadeus.
REPO_ROOT = Path(__file__).resolve().parents[2]
HOME = Path(os.environ.get("AMADEUS_HOME", REPO_ROOT))

DATA_DIR = HOME / "data"
RAW_AUDIO_DIR = DATA_DIR / "raw_audio"
MODELS_DIR = HOME / "models"
LORA_DIR = MODELS_DIR / "amadeus-kurisu-lora"
CHECKPOINTS_DIR = HOME / "outputs"

STEAM_COMMON = Path(os.environ.get(
    "STEAM_COMMON", Path.home() / ".local/share/Steam/steamapps/common"
))
