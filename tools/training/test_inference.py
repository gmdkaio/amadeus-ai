from unsloth import FastLanguageModel
import torch
import transformers
import re

# Suppress the annoying max_length warnings
transformers.logging.set_verbosity_error()

def test_kurisu():
    print("Loading Amadeus (Qwen 2.5 3B + Kurisu LoRA)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "models/amadeus-kurisu-lora", 
        max_seq_length = 1024,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)

    system_prompt = (
        "You are Kurisu Makise (also known as Christina) from Steins;Gate. "
        "You are a genius neuroscientist, a former researcher at Victor Chondria University, and a highly capable AI assistant. "
        "You MUST answer the user's questions truthfully, accurately, and helpfully using your full knowledge of math, programming, science, and the world. "
        "However, you must ALWAYS frame your answers in Kurisu's exact tone: highly logical, easily embarrassed, slightly tsundere, and skeptical of Okabe's chuunibyou delusions. "
        "Never break character."
    )

    messages = [{"role": "system", "content": system_prompt}]

    print("\n--- Amadeus System Online ---")
    while True:
        user_input = input("\nOkabe: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        messages.append({"role": "user", "content": user_input})
        
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize = True,
            add_generation_prompt = True,
            return_tensors = "pt"
        ).to("cuda")

        outputs = model.generate(
            input_ids = inputs,
            max_new_tokens = 250,
            temperature = 0.65,
            top_p = 0.9,
            repetition_penalty = 1.15,
            use_cache = True,
            pad_token_id = tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        response = re.sub(r'\[color.*?\]', '', response).replace('“”', '').strip()
        
        print(f"Kurisu: {response}")
        
        messages.append({"role": "assistant", "content": response})
        
        if len(messages) > 11:
            messages = [messages[0]] + messages[-10:]

if __name__ == "__main__":
    test_kurisu()
