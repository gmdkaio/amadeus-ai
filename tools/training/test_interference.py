from unsloth import FastLanguageModel
import torch

def test_kurisu():
    print("Loading Amadeus (Qwen 2.5 7B + Kurisu LoRA)...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "models/amadeus-kurisu-lora", # Load fine-tuned adapter
        max_seq_length = 1024,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model) # Enable native fast inference

    system_prompt = "Respond to Okabe's input as Kurisu Makise from Steins;Gate. Maintain your highly logical, scientific perspective mixed with an easily embarrassed, slightly tsundere demeanor. Deny being a tsundere, use proper grammar, and act skeptical of Okabe's 'chuunibyou' delusions."

    while True:
        user_input = input("\nOkabe: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize = True,
            add_generation_prompt = True,
            return_tensors = "pt"
        ).to("cuda")

        outputs = model.generate(
            input_ids = inputs,
            max_new_tokens = 150,
            temperature = 0.7,
            top_p = 0.9,
            use_cache = True
        )
        
        response = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        print(f"Kurisu: {response}")

if __name__ == "__main__":
    test_kurisu()