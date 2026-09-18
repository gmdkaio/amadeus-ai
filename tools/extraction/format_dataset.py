import os
import json
import re

def build_dataset():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    input_script = os.path.join(base_dir, 'data', 'raw_script.txt')
    output_file = os.path.join(base_dir, 'data', 'kurisu_finetune.jsonl')
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Note: If your raw_script.txt is missing, you'll need to drop it back into data/
    if not os.path.exists(input_script):
        print(f"Waiting on raw script data at: {input_script}")
        return

    instruction = (
        "You are Kurisu Makise (also known as Christina) from Steins;Gate. "
        "You are a genius neuroscientist, highly logical, easily embarrassed, slightly tsundere, and skeptical. "
        "Never break character."
    )

    clean_pairs = []
    
    with open(input_script, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # Mock parser for standard VN dialogue structures
    for i in range(len(lines) - 1):
        if "Okabe:" in lines[i] and "Kurisu:" in lines[i+1]:
            okabe_text = lines[i].replace("Okabe:", "").strip()
            kurisu_text = lines[i+1].replace("Kurisu:", "").strip()
            
            # The holy grail regex cleaning we added last night
            kurisu_text = re.sub(r'\[%.*?\]', '', kurisu_text)
            kurisu_text = re.sub(r'\[color.*?\]', '', kurisu_text)
            kurisu_text = kurisu_text.replace('“”', '').strip()
            
            if kurisu_text and okabe_text:
                clean_pairs.append({
                    "instruction": instruction,
                    "input": okabe_text,
                    "output": kurisu_text
                })

    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in clean_pairs:
            f.write(json.dumps(pair) + '\n')
            
    print(f"Dataset formatted successfully! {len(clean_pairs)} clean pairs saved to {output_file}")

if __name__ == "__main__":
    build_dataset()
