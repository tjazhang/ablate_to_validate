# NEW: Aurora-only file; not present in upstream LLAVA.
# NEW: Baseline https://github.com/haotian-liu/LLaVA.git @ v1.2.2.post1 (24fa1d065bbeac8a145a796ab7218c6945a2536e).
# NEW: Aurora path: data/toWeb.py

import os
import json
import tarfile
from pathlib import Path
# from PIL import Image # PIL Image is not strictly needed for just adding files
import io
import math

# --- Configuration ---
data_dir = Path(os.environ.get("LLAVA_DATA_ROOT", str(Path(__file__).resolve().parent)))
data_name = "ADE20K/train_annealing_data_10"
input_json_path = data_dir / f"{data_name}.json"
output_dir = data_dir / f"{data_name}_wds" # Changed suffix to _wds

samples_per_shard = 1000

os.makedirs(output_dir, exist_ok=True)

print(f"Loading dataset description from: {input_json_path}")
# Load the dataset description
try:
    with open(input_json_path, 'r') as f:
        dataset = json.load(f)
except FileNotFoundError:
    print(f"Error: Input JSON file not found at {input_json_path}")
    exit(1)
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {input_json_path}")
    exit(1)

print(f"Found {len(dataset)} samples in the dataset description.")

# --- Sharding Logic ---
num_samples = len(dataset)
num_shards = math.ceil(num_samples / samples_per_shard)
shard_paths = []
current_shard_index = 0
samples_in_current_shard = 0
tar_writer = None

print(f"Preparing to create {num_shards} shards with approx {samples_per_shard} samples each.")

try:
    for i, item in enumerate(dataset):
        # --- Determine Current Shard ---
        if samples_in_current_shard == 0:
            # Close previous tar writer if it exists
            if tar_writer:
                tar_writer.close()
                print(f"Closed shard: {shard_filename}")

            # Start a new shard
            shard_filename = output_dir / f"shard-{current_shard_index:06d}.tar"
            shard_paths.append(str(shard_filename)) # Store path as string for JSON
            print(f"Opening new shard: {shard_filename}")
            tar_writer = tarfile.open(shard_filename, "w") # Use 'w:' for uncompressed, 'w:gz' or 'w:bz2' for compressed
            current_shard_index += 1

        # --- Prepare Sample Data ---
        try:
            image_path_str = item.get("image")
            item_id = item.get("id")
            conversations = item.get("conversations")

            if not image_path_str or not item_id or conversations is None:
                print(f"Warning: Skipping item {i} due to missing 'image', 'id', or 'conversations' field.")
                continue

            image_path = Path(image_path_str) # Convert to Path object for exists check

            # Ensure PosixPath is converted to string if needed for some os functions,
            # but tarfile.add usually handles Path objects fine.
            if not image_path.exists():
                print(f"Warning: Skipping item {item_id}. Image file not found: {image_path}")
                continue

            # Use item['id'] as the base name for files *within* the tar archive
            base_arcname = str(item_id) # Ensure it's a string

            # 1. Add Image File
            # Derive the extension from the original file path
            img_extension = image_path.suffix # e.g., '.jpg', '.png'
            if not img_extension:
                print(f"Warning: Skipping item {item_id}. Cannot determine image extension from path: {image_path}")
                continue
            img_arcname = f"{base_arcname}{img_extension}"
            tar_writer.add(image_path, arcname=img_arcname)

            # 2. Add Metadata JSON File
            meta = {
                "__key__": base_arcname, # Optional: Common practice to include the key
                "id": item_id,
                "conversations": conversations
                # Add any other metadata associated with the sample
            }
            meta_bytes = json.dumps(meta).encode("utf-8")
            meta_arcname = f"{base_arcname}.json"
            info = tarfile.TarInfo(name=meta_arcname)
            info.size = len(meta_bytes)
            tar_writer.addfile(tarinfo=info, fileobj=io.BytesIO(meta_bytes))

            # Increment sample count for the current shard
            samples_in_current_shard += 1

            # Reset sample count if shard is full
            if samples_in_current_shard >= samples_per_shard:
                samples_in_current_shard = 0 # Trigger new shard creation on next iteration

        except Exception as e:
            print(f"Error processing item {i} (ID: {item.get('id', 'N/A')}): {e}")
            # Decide whether to continue or stop based on the error
            continue # Skip this item and continue with the next

finally:
    # Ensure the last tar writer is closed
    if tar_writer:
        tar_writer.close()
        print(f"Closed final shard: {shard_filename}")

print(f"\nFinished creating {len(shard_paths)} shards.")

# --- Save paths to JSON (optional but can be useful) ---
output_json_path = output_dir / "shard_list.json"
print(f"Saving list of created shards to: {output_json_path}")
with open(output_json_path, "w") as f:
    json.dump(shard_paths, f, indent=2)

print("Script finished.")
