import os
import struct
import sys
import zlib

def unpack_mpk(mpk_path, out_dir):
    with open(mpk_path, 'rb') as f:
        if f.read(4) != b'MPK\x00':
            print("Error: Not a valid MPK file.")
            return
        
        f.read(4) 
        file_count = struct.unpack('<Q', f.read(8))[0]
        f.read(48) 
        
        print(f"Found {file_count} files in {os.path.basename(mpk_path)}. Filtering and extracting...")
        
        entries = []
        for _ in range(file_count):
            entry_data = f.read(256)
            is_comp, index, pos, comp_size, uncomp_size = struct.unpack('<IIQQQ', entry_data[:32])
            
            name = entry_data[32:].split(b'\x00')[0].decode('shift_jis', errors='ignore')
            name = name.replace('\\', '/')
            entries.append((name, pos, comp_size, is_comp))
            
        os.makedirs(out_dir, exist_ok=True)
        
        extracted_count = 0
        for name, pos, comp_size, is_comp in entries:
            
            if name.upper().endswith('.OGG') and not name.upper().startswith('CRS_'):
                continue
                
            f.seek(pos)
            data = f.read(comp_size)
            
            if is_comp:
                try:
                    data = zlib.decompress(data)
                except Exception as e:
                    print(f"Failed to decompress {name}: {e}")
                    
            out_path = os.path.join(out_dir, name)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            
            with open(out_path, 'wb') as out_f:
                out_f.write(data)
            
            extracted_count += 1
                
    print(f"Extraction complete! Saved {extracted_count} target files to {out_dir}")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python unpack.py <input.mpk> <output_directory>")
        sys.exit(1)
    unpack_mpk(sys.argv[1], sys.argv[2])