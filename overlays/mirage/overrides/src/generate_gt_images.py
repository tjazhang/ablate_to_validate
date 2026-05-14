import json
import os
import argparse
from tqdm import tqdm
from task import find_shortest_path_bfs, ACTION_MAP
from PIL import Image, ImageDraw

def overlay_path_on_image(image_path, map_desc, path_actions):
    """
    Overlays the path arrows onto the original input image.
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None

    draw = ImageDraw.Draw(img)
    rows = len(map_desc)
    cols = len(map_desc[0]) if rows > 0 else 0
    
    img_width, img_height = img.size
    cell_width = img_width / cols
    cell_height = img_height / rows
    
    # Use the smaller dimension to define arrow size relative to cell
    cell_size = min(cell_width, cell_height)

    # Find start position
    start = None
    for r in range(rows):
        for c in range(cols):
            if map_desc[r][c] == 1:
                start = (r, c)
                break
        if start is not None:
            break
            
    if start is None or path_actions is None:
        return img

    # Trace the path and collect points
    current_position = start
    path_points = [(current_position[1] * cell_width + cell_width / 2,
                    current_position[0] * cell_height + cell_height / 2)]

    for action in path_actions:
        if action not in ACTION_MAP:
            break
        
        dr, dc = ACTION_MAP[action]
        new_r = current_position[0] + dr
        new_c = current_position[1] + dc
        
        # Check boundaries
        if new_r < 0 or new_r >= rows or new_c < 0 or new_c >= cols:
            continue
        
        current_position = (new_r, new_c)
        
        center_x = new_c * cell_width + cell_width / 2
        center_y = new_r * cell_height + cell_height / 2
        path_points.append((center_x, center_y))

    # Define colors
    COLOR_ARROW = (255, 0, 0)         # Red for arrows

    # Draw arrows for each step
    if len(path_points) > 1:
        arrow_width = max(3, int(cell_size / 15))
        arrow_head_size = max(8, int(cell_size / 5))
        
        for i in range(len(path_points) - 1):
            p1 = path_points[i]
            p2 = path_points[i+1]
            
            x1, y1 = p1
            x2, y2 = p2
            
            # Draw line
            draw.line([p1, p2], fill=COLOR_ARROW, width=arrow_width)
            
            # Draw arrowhead at p2
            if abs(x1 - x2) < 1e-6: # Vertical
                if y2 > y1: # Down
                    points = [(x2, y2), (x2 - arrow_head_size//2, y2 - arrow_head_size), (x2 + arrow_head_size//2, y2 - arrow_head_size)]
                else: # Up
                    points = [(x2, y2), (x2 - arrow_head_size//2, y2 + arrow_head_size), (x2 + arrow_head_size//2, y2 + arrow_head_size)]
            elif abs(y1 - y2) < 1e-6: # Horizontal
                if x2 > x1: # Right
                    points = [(x2, y2), (x2 - arrow_head_size, y2 - arrow_head_size//2), (x2 - arrow_head_size, y2 + arrow_head_size//2)]
                else: # Left
                    points = [(x2, y2), (x2 + arrow_head_size, y2 - arrow_head_size//2), (x2 + arrow_head_size, y2 + arrow_head_size//2)]
            else:
                continue
                
            draw.polygon(points, fill=COLOR_ARROW)
            
    return img

def main():
    parser = argparse.ArgumentParser(description="Generate ground truth helper images (reasoning paths) for VSP tasks.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the JSONL data file.")
    parser.add_argument("--task", type=str, default="vsp-spatial-planning", help="Task name.")
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Error: Data path {args.data_path} does not exist.")
        return

    print(f"Loading data from {args.data_path}...")
    with open(args.data_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    print(f"Processing {len(data)} samples...")
    generated_count = 0
    skipped_count = 0
    error_count = 0

    for i, sample in tqdm(enumerate(data), total=len(data)):
        map_desc = sample.get("map_desc")
        image_input_path = sample.get("image_input")

        if not map_desc:
            skipped_count += 1
            continue

        if not image_input_path:
            # Try to handle cases where image_input might be a list (like in blink-jigsaw)
            image_input = sample.get("image_input")
            if isinstance(image_input, list) and len(image_input) > 0:
                image_input_path = image_input[0]
            else:
                skipped_count += 1
                continue

        try:
            # Find shortest path
            gt_path = find_shortest_path_bfs(map_desc)
            
            if gt_path is not None:
                # Check if image path exists relative to current dir
                if not os.path.exists(image_input_path):
                     # Try relative to data_path dir
                     data_dir = os.path.dirname(args.data_path)
                     alt_path = os.path.join(data_dir, image_input_path)
                     if os.path.exists(alt_path):
                         image_input_path = alt_path
                
                gt_image = overlay_path_on_image(image_input_path, map_desc, gt_path)
                
                if gt_image is None:
                    error_count += 1
                    continue

                # Determine save path
                image_output_path = sample.get("image_output")
                if image_output_path:
                    # Check if path exists or needs to be relative to data_path
                    if not os.path.exists(image_output_path):
                         data_dir = os.path.dirname(args.data_path)
                         alt_path = os.path.join(data_dir, image_output_path)
                         # Even if it doesn't exist, we'll use this as the save location
                         save_path = alt_path
                    else:
                         save_path = image_output_path
                    save_dir = os.path.dirname(save_path)
                else:
                    # Save the image with a unique filename based on input image
                    save_dir = os.path.dirname(image_input_path)
                    if not save_dir:
                        save_dir = "."
                    
                    # Create unique filename based on original image name
                    original_filename = os.path.basename(image_input_path)
                    name_without_ext = os.path.splitext(original_filename)[0]
                    
                    # Use naming convention: {id}_reasoning_path.png for test, map_reasoning_path.png for train
                    if "imgs_test" in image_input_path:
                        output_filename = f"{name_without_ext}_reasoning_path.png"
                    else:
                        # If it's in a subdirectory (like the ID dir in train), use map_reasoning_path.png
                        # Otherwise fallback to the {id}_reasoning_path.png
                        parent_dir = os.path.basename(save_dir)
                        if parent_dir.isdigit() or parent_dir.startswith("level"):
                            output_filename = "map_reasoning_path.png"
                        else:
                            output_filename = f"{name_without_ext}_reasoning_path.png"
                            
                    save_path = os.path.join(save_dir, output_filename)
                
                if not os.path.exists(save_dir) and save_dir != ".":
                    os.makedirs(save_dir, exist_ok=True)
                
                gt_image.save(save_path)
                generated_count += 1
            else:
                skipped_count += 1
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            error_count += 1

    print(f"\nGeneration complete!")
    print(f"Generated: {generated_count}")
    print(f"Skipped: {skipped_count} (no map_desc or no path found)")
    print(f"Errors: {error_count}")

if __name__ == "__main__":
    main()
