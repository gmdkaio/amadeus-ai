import os
import glob
import json
import re

def build_dataset():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    txt_dir = os.path.join(base_dir, 'data', 'raw_scripts', 'txt')
    output_file = os.path.join(base_dir, 'data', 'kurisu_finetune.jsonl')

    if not os.path.exists(txt_dir):
        print(f"Error: {txt_dir} not found. Did sc3tools run correctly?")
        return

    dataset = []
    
    # Standard instruction for the LLM
    system_prompt = "Respond to Okabe's input as Kurisu Makise from Steins;Gate. Maintain your highly logical, scientific perspective mixed with an easily embarrassed, slightly tsundere demeanor. Deny being a tsundere, use proper grammar, and act skeptical of Okabe's 'chuunibyou' delusions."

    print("Parsing text files for Okabe -> Kurisu interactions...")

    for txt_file in sorted(glob.glob(os.path.join(txt_dir, '*.txt'))):
        with open(txt_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        last_speaker = None
        last_line = ""

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Match [name]Speaker[line]Dialogue
            name_match = re.match(r'^\[name\](.*?)\[line\]', line)
            
            if name_match:
                speaker = name_match.group(1)
                dialogue = line[len(name_match.group(0)):].strip()
                
                # Strip out engine tags like [%p], [%e], etc.
                dialogue = re.sub(r'\[%.*?\]', '', dialogue)
                dialogue = re.sub(r'\[color.*?\]', '', dialogue)
                
                # Remove the quotes that wrap the dialogue
                if dialogue.startswith('“') and dialogue.endswith('”'):
                    dialogue = dialogue[1:-1]
                elif dialogue.startswith('"') and dialogue.endswith('"'):
                    dialogue = dialogue[1:-1]
                
                # Check for a valid conversational pair (Rintaro -> Kurisu)
                if speaker in ['Kurisu', 'Makise', '紅莉栖', 'クリス']:
                    if last_speaker in ['Rintaro', 'Okabe', 'Okarin', '岡部', 'オカベ'] and last_line:
                        dataset.append({
                            "instruction": system_prompt,
                            "input": last_line,
                            "output": dialogue
                        })
                
                last_speaker = speaker
                last_line = dialogue

    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"Successfully generated {len(dataset)} training pairs.")
    print(f"Saved to {output_file}")

if __name__ == '__main__':
    build_dataset()