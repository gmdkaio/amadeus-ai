import os
import json
import glob

def extract_dialogue(scx_path):
    with open(scx_path, 'rb') as f:
        data = f.read()
        
    strings = []
    # MAGES strings are null-terminated. Split the entire binary by 0x00
    for block in data.split(b'\x00'):
        if len(block) > 4:
            try:
                decoded = block.decode('shift_jis')
                # Keep only strings that contain actual Japanese text (ignores ASCII engine code)
                if not decoded.isascii() and not decoded.startswith('SG'):
                    strings.append(decoded)
            except UnicodeDecodeError:
                pass
                
    return strings

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    script_dir = os.path.join(base_dir, 'data', 'raw_scripts')
    output_file = os.path.join(base_dir, 'data', 'kurisu_corpus.json')
    
    if not os.path.exists(script_dir):
        print(f"Error: Could not find {script_dir}")
        return

    all_text = []
    print("Scanning SCX files...")
    
    for scx_file in glob.glob(os.path.join(script_dir, '*.SCX')):
        lines = extract_dialogue(scx_file)
        if lines:
            all_text.extend(lines)
            
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_text, f, ensure_ascii=False, indent=2)
        
    print(f"Extracted {len(all_text)} text fragments to {output_file}")

if __name__ == '__main__':
    main()