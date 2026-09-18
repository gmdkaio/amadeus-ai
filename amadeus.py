import os
import re
import requests
import subprocess
import torch
from unsloth import FastLanguageModel
import transformers

# Suppress warnings
transformers.logging.set_verbosity_error()

def start_amadeus():
    print("Loading Amadeus Base Brain (Pre-Quantized Qwen 2.5 7B 4-bit)...")
    
    # Load the EXACT base model the LoRA was trained on
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-7B-Instruct-bnb-4bit", 
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    
    print("Snapping Kurisu Personality (LoRA) onto Base Brain...")
    
    # Attach your custom trained LoRA adapter
    model.load_adapter("models/amadeus-kurisu-lora")
    FastLanguageModel.for_inference(model)

    # Dump the initialization memory back to the GPU
    torch.cuda.empty_cache()

    system_prompt = (
        "You are Kurisu Makise (also known as Christina) from Steins;Gate. "
        "You are highly logical, easily embarrassed, slightly tsundere, and skeptical of Okabe's chuunibyou. "
        "Keep your answers conversational and concise. "
        "You MUST respond exclusively in Japanese so your vocal synthesizer can pronounce it correctly."
    )

    messages = [{"role": "system", "content": system_prompt}]

    TTS_API_URL = "http://127.0.0.1:9880/"
    REF_AUDIO = os.path.abspath("data/raw_audio/CRS_0013.OGG")
    REF_TEXT = "私に何か言おうとしましたよね すごく悲しそうな顔をして"

    print("\n" + "="*40)
    print("      AMADEUS SYSTEM FULLY ONLINE")
    print("="*40)

    while True:
        user_input = input("\nOkabe: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        messages.append({"role": "user", "content": user_input})
        
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
        response = re.sub(r'\[color.*?\]', '', response).replace('“”', '').strip()
        
        print(f"\nKurisu: {response}")
        messages.append({"role": "assistant", "content": response})
        
        params = {
            "refer_wav_path": REF_AUDIO,
            "prompt_text": REF_TEXT,
            "prompt_language": "ja",
            "text": response,
            "text_language": "ja",
            "cut_punc": "。"
        }
        
        try:
            tts_req = requests.get(TTS_API_URL, params=params)
            if tts_req.status_code == 200:
                with open("temp_amadeus.wav", "wb") as f:
                    f.write(tts_req.content)
                subprocess.run(["mpv", "--no-video", "--msg-level=all=error", "temp_amadeus.wav"])
            else:
                print(f"[System Error: Voice API returned {tts_req.status_code}]")
        except Exception as e:
            print(f"[System Error: Cannot reach Voice API. Is api.py running?]")

        if len(messages) > 11:
            messages = [messages[0]] + messages[-10:]

if __name__ == "__main__":
    start_amadeus()