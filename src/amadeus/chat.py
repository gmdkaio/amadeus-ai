import os
import re
import subprocess
import tempfile

import requests
import torch
from unsloth import FastLanguageModel
import transformers

from amadeus import paths

# Suppress warnings
transformers.logging.set_verbosity_error()

BASE_MODEL = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"

SYSTEM_PROMPT = (
    "You are Kurisu Makise (also known as Christina) from Steins;Gate. "
    "You are highly logical, easily embarrassed, slightly tsundere, and skeptical of Okabe's chuunibyou. "
    "Keep your answers conversational and concise. "
    "You MUST respond exclusively in Japanese so your vocal synthesizer can pronounce it correctly."
)

TTS_API_URL = "http://127.0.0.1:9880/"
REF_AUDIO = paths.RAW_AUDIO_DIR / "CRS_0013.OGG"
REF_TEXT = "私に何か言おうとしましたよね すごく悲しそうな顔をして"

# System prompt plus this many recent messages
HISTORY_LIMIT = 10


def load_model():
    print("Loading Amadeus Base Brain (Pre-Quantized Qwen 2.5 7B 4-bit)...")

    # Load the EXACT base model the LoRA was trained on
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )

    print("Snapping Kurisu Personality (LoRA) onto Base Brain...")
    model.load_adapter(str(paths.LORA_DIR))
    FastLanguageModel.for_inference(model)

    # Dump the initialization memory back to the GPU
    torch.cuda.empty_cache()
    return model, tokenizer


def generate(model, tokenizer, messages):
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=150,
        temperature=0.65,
        top_p=0.9,
        repetition_penalty=1.15,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
    return re.sub(r'\[color.*?\]', '', response).replace('“”', '').strip()


def speak(text, wav_path):
    params = {
        "refer_wav_path": str(REF_AUDIO),
        "prompt_text": REF_TEXT,
        "prompt_language": "ja",
        "text": text,
        "text_language": "ja",
        "cut_punc": "。"
    }

    try:
        tts_req = requests.get(TTS_API_URL, params=params)
    except requests.RequestException:
        print("[System Error: Cannot reach Voice API. Is GPT-SoVITS api.py running?]")
        return

    if tts_req.status_code != 200:
        print(f"[System Error: Voice API returned {tts_req.status_code}]")
        return

    with open(wav_path, "wb") as f:
        f.write(tts_req.content)
    subprocess.run(["mpv", "--no-video", "--msg-level=all=error", wav_path])


def run(voice=True):
    model, tokenizer = load_model()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    wav_path = os.path.join(tempfile.gettempdir(), "amadeus_reply.wav")

    print("\n" + "="*40)
    print("      AMADEUS SYSTEM FULLY ONLINE")
    print("="*40)

    while True:
        try:
            user_input = input("\nOkabe: ")
        except EOFError:
            break
        if user_input.lower() in ["exit", "quit"]:
            break

        messages.append({"role": "user", "content": user_input})
        response = generate(model, tokenizer, messages)

        print(f"\nKurisu: {response}")
        messages.append({"role": "assistant", "content": response})

        if voice:
            speak(response, wav_path)

        if len(messages) > HISTORY_LIMIT + 1:
            messages = [messages[0]] + messages[-HISTORY_LIMIT:]
