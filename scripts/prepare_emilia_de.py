import sys
import os
import tarfile
import tempfile
import shutil

sys.path.append(os.getcwd())

from pathlib import Path
import json
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from datasets.arrow_writer import ArrowWriter
from model.utils import repetition_found

# Define problematic German samples to filter out (if any)
out_de = set()  # Add any problematic German sample IDs here if found during testing
de_filters = []  # Add any German-specific filters if needed

def process_tar_file(tar_path):
    """Process a single tar file and return the processed data."""
    sub_result, durations = [], []
    vocab_set = set()
    bad_case_de = 0
    
    # Create a temporary directory to extract files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract tar file
        with tarfile.open(tar_path, 'r') as tar:
            tar.extractall(path=temp_dir)
        
        # Process the extracted files
        temp_path = Path(temp_dir)
        json_files = list(temp_path.glob('**/*.json'))  # Recursively find all JSON files
        
        if not json_files:
            print(f"No JSON files found in {tar_path}")
            return [], [], set(), 0
        
        for json_file in json_files:
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                    
                text = obj.get("text", "")
                
                # Skip if text is empty or contains unwanted patterns
                if not text or any(f in text for f in de_filters) or repetition_found(text, length=4):
                    bad_case_de += 1
                    continue
                
                # German-specific text normalization
                text = text.translate(
                    str.maketrans({
                        ",": ",",
                        "!": "!",
                        "?": "?",
                        "„": "\"",  # German quotation marks
                        "–": "-"    # German dash
                    })
                )
                
                # Get corresponding mp3 file path
                mp3_path = str(json_file).replace('.json', '.mp3')
                if not os.path.exists(mp3_path):
                    continue
                
                # Calculate duration (you might want to use a proper audio duration calculation)
                duration = obj.get("duration", 0)
                if duration <= 0:
                    continue
                
                # Construct the audio path relative to dataset directory
                tar_name = Path(tar_path).stem
                relative_mp3_path = os.path.relpath(mp3_path, temp_dir)
                audio_path = f"{dataset_dir}/DE/{tar_name}/{relative_mp3_path}"
                
                sub_result.append({
                    "audio_path": audio_path,
                    "text": text,
                    "duration": duration
                })
                durations.append(duration)
                vocab_set.update(list(text))
                
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue
    
    return sub_result, durations, vocab_set, bad_case_de

def main():
    result = []
    duration_list = []
    text_vocab_set = set()
    total_bad_case_de = 0

    # Get list of all tar files
    dataset_path = Path(dataset_dir) / "DE"
    tar_files = list(dataset_path.glob("*.tar"))
    
    # Process tar files in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_tar_file, str(tar_file)) for tar_file in tar_files]
        
        for future in tqdm(futures, total=len(futures), desc="Processing tar files"):
            sub_result, durations, vocab_set, bad_case_de = future.result()
            result.extend(sub_result)
            duration_list.extend(durations)
            text_vocab_set.update(vocab_set)
            total_bad_case_de += bad_case_de

    # Save preprocessed dataset
    output_dir = Path(f"data/{dataset_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving to {output_dir} ...")
    
    # Save to Arrow format
    with ArrowWriter(path=str(output_dir / "raw.arrow")) as writer:
        for line in tqdm(result, desc="Writing to raw.arrow ..."):
            writer.write(line)

    # Save duration information
    with open(output_dir / "duration.json", "w", encoding="utf-8") as f:
        json.dump({"duration": duration_list}, f, ensure_ascii=False)

    # Save vocabulary (including German-specific characters)
    text_vocab_set.update([chr(i) for i in range(32, 127)] + 
                         ['ä', 'ö', 'ü', 'ß', 'Ä', 'Ö', 'Ü'])
    
    with open(output_dir / "vocab.txt", "w", encoding="utf-8") as f:
        for vocab in sorted(text_vocab_set):
            f.write(vocab + "\n")

    print(f"\nFor {dataset_name}:")
    print(f"Sample count: {len(result)}")
    print(f"Vocab size: {len(text_vocab_set)}")
    print(f"Total duration: {sum(duration_list)/3600:.2f} hours")
    print(f"Bad German transcription cases: {total_bad_case_de}\n")

if __name__ == "__main__":
    max_workers = 32
    tokenizer = "pinyin"  # Using character-based tokenization for German
    
    dataset_dir = "/raid/shared/datasets/Emilia-Dataset"
    dataset_name = f"Emilia_DE_{tokenizer}"
    print(f"\nPreparing {dataset_name}\n")

    main()