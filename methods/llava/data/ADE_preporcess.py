# NEW: Aurora-only file; not present in upstream LLAVA.
# NEW: Baseline https://github.com/haotian-liu/LLaVA.git @ v1.2.2.post1 (24fa1d065bbeac8a145a796ab7218c6945a2536e).
# NEW: Aurora path: data/ADE_preporcess.py

from datasets import load_dataset
import os
from pathlib import Path

# Load ADE20K dataset
ds = load_dataset("1aurent/ADE20K")

# Output directory where images are already saved
output_dir = Path(os.environ.get("ADE20K_OUTPUT_DIR", str(Path(__file__).resolve().parent / "ADE20K")))

# Loop through splits
for split in ds.keys():  # e.g., 'train', 'validation', 'test'
    split_dir = output_dir / split
    
    if not split_dir.exists():
        print(f"Split directory {split_dir} does not exist, skipping...")
        continue
    
    print(f"Renaming {split} images in {split_dir}...")
    
    for idx, example in enumerate(ds[split]):
        # Get the actual filename from dataset
        filename = example["filename"]
        
        # Current file path (based on index)
        current_file = split_dir / f"{idx}.png"
        
        # New file path (based on actual filename)
        new_file = split_dir / filename
        
        # Rename if the current file exists and new name is different
        if current_file.exists() and current_file != new_file:
            os.rename(current_file, new_file)
            print(f"Renamed: {idx}.png -> {filename}")
        elif not current_file.exists():
            print(f"Warning: {current_file} does not exist, skipping...")

    print(f"Completed renaming {split} images.")

print("All images have been renamed!")
