import warnings
warnings.filterwarnings("ignore")

from transformers import pipeline
import librosa

print("Loading Amadeus Hearing Module (Whisper-Small)...")
# Load the model directly onto your RTX 4060
pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=0)

for filename in ["CRS_0013.OGG", "CRS_0014.OGG"]:
    filepath = f"data/raw_audio/{filename}"
    print(f"\n--- Processing {filename} ---")
    
    # Whisper expects 16kHz audio
    audio, sr = librosa.load(filepath, sr=16000)
    
    # Force the AI to output Japanese characters
    result = pipe(audio, generate_kwargs={"language": "japanese"})
    print(f"Transcript: {result['text'].strip()}")
