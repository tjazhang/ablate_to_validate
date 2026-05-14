#!/usr/bin/env python3
# NEW: Aurora-only file; not present in upstream LLAVA.
# NEW: Baseline https://github.com/haotian-liu/LLaVA.git @ v1.2.2.post1 (24fa1d065bbeac8a145a796ab7218c6945a2536e).
# NEW: Aurora path: data/depth_point_generator.py

"""
Depth Point Generator Script
Based on add_5_points.ipynb - generates depth-based question-answer pairs with random point selection.
Uses DEPTH-FIRST approach: starts from depth maps and finds corresponding RGB images.
"""

import numpy as np
import os
import json
import random
import argparse
import uuid
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import warnings
from numba import njit

warnings.filterwarnings("ignore")

@njit
def find_valid_points_numba(num_points, target_size, min_dist_sq, min_depth_diff, depth_array, max_attempts, y_min, y_max):
    """
    A Numba-jitted function to find valid points.
    Note: Numba works best with simple types like numbers and arrays.
    """
    points = np.zeros((num_points, 2), dtype=np.int32)
    depths = np.zeros(num_points, dtype=np.float64)
    
    for attempt in range(max_attempts):
        num_found = 0
        for point_attempt in range(max_attempts):
            if num_found == num_points:
                break
                
            x = np.random.randint(10, target_size - 11)
            y = np.random.randint(y_min, y_max)
            
            current_depth = depth_array[y, x]
            if current_depth == 0:
                continue

            # Check distance and depth against already found points
            is_valid = True
            for i in range(num_found):
                # Squared distance check
                dist_sq = (x - points[i, 0])**2 + (y - points[i, 1])**2
                if dist_sq < min_dist_sq:
                    is_valid = False
                    break
                
                # Depth check
                if abs(current_depth - depths[i]) < min_depth_diff:
                    is_valid = False
                    break
            
            if is_valid:
                points[num_found, 0] = x
                points[num_found, 1] = y
                depths[num_found] = current_depth
                num_found += 1
        
        if num_found == num_points:
            # Sort points by depth
            sorted_indices = np.argsort(depths)
            return points[sorted_indices], depths[sorted_indices]
            
    return None, None

DATA_ROOT = Path(os.environ.get("LLAVA_DATA_ROOT", str(Path(__file__).resolve().parent / "ADE20K")))
DATASET_PATH = str(Path(os.environ.get("ADE20K_IMAGE_ROOT", str(DATA_ROOT / "train"))))
DEPTH_PATH = str(Path(os.environ.get("DEPTH_MAP_DIR", str(DATA_ROOT / "depth_maps"))))
DEFAULT_DEPTH_CODES_PATH = os.environ.get("DEPTH_CODES_PATH")

# Special images that need different y-coordinate range (from notebook)
SPECIAL_IMAGES = [""]

class DepthPointGenerator:
    def __init__(self, seed=42, target_size=336, min_distance=20, min_depth_diff=20, output_dir="depth_point_output"):
        """
        Initialize the depth point generator.
        
        Args:f
            seed (int): Random seed for reproducibility
            target_size (int): Target image size (width and height)
            min_distance (int): Minimum pixel distance between points
            min_depth_diff (int): Minimum depth difference between points
            output_dir (str): Base output directory for drawn images
        """
        self.seed = seed
        self.target_size = target_size
        self.min_distance = min_distance
        self.min_depth_diff = min_depth_diff
        self.output_dir = output_dir
        
        # Set random seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize data storage
        self.data_dict = {}
        self.point_data = {}
        self.drawn_image_paths = {}  # Store paths to drawn images
        
        # Load depth codes dictionary
        self.depth_codes = None
        self.load_depth_codes()
        
    def load_depth_codes(self):
        """Load the depth codes dictionary from the .npy file."""
        depth_codes_path = DEFAULT_DEPTH_CODES_PATH
        try:
            if depth_codes_path and os.path.exists(depth_codes_path):
                data = np.load(depth_codes_path, allow_pickle=True)
                self.depth_codes = data.flat[0]  # Extract the dictionary
                print(f"Loaded depth codes dictionary with {len(self.depth_codes)} entries")
            else:
                print("Warning: Depth codes file not configured. Set DEPTH_CODES_PATH if you need discrete depth code lookups.")
                self.depth_codes = {}
        except Exception as e:
            print(f"Error loading depth codes: {e}")
            self.depth_codes = {}
    
    def get_depth_value_for_image(self, img_name):
        """Get depth value for a specific image from the loaded dictionary."""
        if self.depth_codes is None or len(self.depth_codes) == 0:
            return "<DEPTH_TOKEN>"  # Fallback to original placeholder
        
        # Convert image name to depth map name with _depth suffix
        depth_map_name = img_name.replace('.jpg', '_depth.png')
        
        # Try different key formats that might match the depth map name
        possible_keys = [
            depth_map_name,  # full depth map name with _depth.png
            depth_map_name.replace('.png', ''),  # without .png extension
            os.path.basename(depth_map_name),  # just filename
            os.path.basename(depth_map_name).replace('.png', ''),  # filename without extension
        ]
        
        for key in possible_keys:
            if key in self.depth_codes:
                return str(self.depth_codes[key])
        
        # If no match found, return the placeholder
        print(f"Warning: No depth value found for depth map {depth_map_name}")
        return "<DEPTH_TOKEN>"
        
    def load_image_pairs(self, max_images=None, order='random'):
        """Load depth map and image pairs from the dataset (DEPTH-FIRST APPROACH).
        
        Args:
            max_images (int): Maximum number of image pairs to load (None for all)
            order (str): Loading order - 'random', 'sequential', or 'reverse'
        """
        print("Loading image pairs (DEPTH-FIRST approach)...")
        
        # Get all depth files from the depth directory first
        if not os.path.exists(DEPTH_PATH):
            print(f"Depth path does not exist: {DEPTH_PATH}")
            return 0
            
        # Look for depth files with "_depth" suffix
        all_depth_files = []
        for f in os.listdir(DEPTH_PATH):
            if f.endswith('_depth.png') or f.endswith('_depth.jpg'):
                all_depth_files.append(f)
        
        # Apply ordering strategy
        if order == 'sequential':
            all_depth_files.sort()  # Sort for consistent ordering
        elif order == 'reverse':
            all_depth_files.sort(reverse=True)  # Reverse alphabetical order
        elif order == 'random':
            random.shuffle(all_depth_files)  # Random order
        else:
            print(f"Warning: Unknown order '{order}', using random order")
            random.shuffle(all_depth_files)
        
        # Limit the number of files if specified
        if max_images is not None:
            all_depth_files = all_depth_files[:max_images]
        
        print(f"Found {len(all_depth_files)} depth files in depth directory")
        
        # Create dictionary to store depth maps and their corresponding images
        loaded_count = 0
        for depth_name in all_depth_files:
            # Build paths for depth map and RGB image
            depth_path = os.path.join(DEPTH_PATH, depth_name)
            
            # Extract base name from depth file (remove _depth.png or _depth.jpg)
            if depth_name.endswith('_depth.png'):
                base_name = depth_name[:-10]  # Remove _depth.png
            elif depth_name.endswith('_depth.jpg'):
                base_name = depth_name[:-10]  # Remove _depth.jpg
            else:
                continue  # Skip if doesn't match expected pattern
            
            # Try to find corresponding RGB image
            img_name_candidates = [
                f"{base_name}.jpg",
                f"{base_name}.png",
            ]
            
            # Find the first existing image file
            img_path_full = None
            img_name_found = None
            for candidate in img_name_candidates:
                candidate_path = os.path.join(DATASET_PATH, candidate)
                if os.path.exists(candidate_path):
                    img_path_full = candidate_path
                    img_name_found = candidate
                    break
            
            # Check if both files exist
            if os.path.exists(depth_path) and img_path_full is not None:
                # Store in dictionary with sub-dictionary structure (use image name as key for consistency)
                self.data_dict[img_name_found] = {
                    'image': img_path_full,
                    'depth': depth_path,
                }
                loaded_count += 1
            else:
                if loaded_count < 10:  # Only show first 10 warnings to avoid spam
                    print(f"Warning: Missing files for {depth_name}")
                    print(f"  Depth exists: {os.path.exists(depth_path)}")
                    print(f"  Tried image paths: {[os.path.basename(c) for c in [os.path.join(DATASET_PATH, candidate) for candidate in img_name_candidates]]}")
                elif loaded_count == 10:
                    print("  ... (suppressing further missing file warnings)")
        
        print(f"Successfully loaded {loaded_count} complete image pairs (depth-first)")
        return loaded_count
    
    def add_point(self, text, draw, center_x, center_y, font_size=12):
        """
        Add a point marker with text label to an image.
        
        Args:
            text (str): Label text to display
            draw (ImageDraw.Draw): PIL ImageDraw object
            center_x (int): X coordinate of point center
            center_y (int): Y coordinate of point center
            font_size (int): Font size for text
        """
        radius = 4  # Radius of the circle
        draw.ellipse([center_x - radius, center_y - radius, center_x + radius, center_y + radius], 
                    outline='red', width=2)
        
        # Use default font
        font = ImageFont.load_default()
        text_width = draw.textlength(text, font=font)
        text_height = font_size
        text_x, text_y = center_x, center_y - 25  # Position where the text will start

        # Define background dimensions and position based on text dimensions
        background_margin = 1  # Margin between text and background edge
        background_x0 = text_x - background_margin
        background_y0 = text_y - background_margin
        background_x1 = text_x + text_width + background_margin
        background_y1 = text_y + text_height + background_margin
        draw.rectangle([background_x0, background_y0, background_x1, background_y1], fill='black')
        
        # Draw text
        draw.text((text_x, text_y), text, font=font, fill='white')
    
    def draw_markers_and_save(self, img_name, points, depths, labels, num_points, output_json_path):
        """
        Draw markers on image and save to a single images folder for HuggingFace compatibility.
        
        Args:
            img_name (str): Original image filename
            points (list): List of (x, y) coordinates
            depths (list): List of depth values
            labels (list): List of point labels
            num_points (int): Number of points (for folder organization)
            output_json_path (str): Path to the output JSON file (used to determine base directory)
            
        Returns:
            str: Path to the saved drawn image
        """
        # Use the same directory as the output JSON file
        json_dir = os.path.dirname(output_json_path)
        
        # Create a single 'images' folder instead of separate point folders
        output_folder = os.path.join(json_dir, "images")
        os.makedirs(output_folder, exist_ok=True)
        
        # Load and resize image
        img = Image.open(self.data_dict[img_name]['image'])
        img = img.resize((self.target_size, self.target_size))
        
        draw = ImageDraw.Draw(img)
        
        # Draw points with labels
        for i, (x, y) in enumerate(points):
            self.add_point(labels[i], draw, x, y)
        
        # Generate output filename with point count prefix for uniqueness
        base_name = os.path.splitext(img_name)[0]  # Remove extension
        output_filename = f"{num_points}pts_{base_name}.jpg"  # Add point count prefix
        output_path = os.path.join(output_folder, output_filename)
        
        # Save the drawn image
        img.save(output_path)
        
        return output_path
    
    def select_random_points(self, img_name, num_points):
        """
        Select random points on an image with distance and depth constraints.
        Optimized version using Numba-jitted function for better performance.
        
        Args:
            img_name (str): Name of the image file
            num_points (int): Number of points to select (3-5)
            
        Returns:
            tuple: (points, depths) or (None, None) if failed
        """
        # Load and resize images ONCE outside the loop (notebook optimization)
        img = Image.open(self.data_dict[img_name]['image'])
        img = img.resize((self.target_size, self.target_size))
        depth_img = Image.open(self.data_dict[img_name]['depth']).convert('L')
        depth_img = depth_img.resize((self.target_size, self.target_size))
        depth_array = np.array(depth_img, dtype=np.float64)
        
        # Determine y-coordinate range based on special images
        y_min, y_max = (95, 225)
        if img_name in SPECIAL_IMAGES:
            y_min, y_max = (10, self.target_size - 11)

        # Use Numba-optimized function
        points_arr, depths_arr = find_valid_points_numba(
            num_points, self.target_size, self.min_distance**2, 
            self.min_depth_diff, depth_array, 10000, y_min, y_max
        )

        if points_arr is not None:
            # Convert back to list of tuples with native Python types for JSON serialization
            return [tuple(int(coord) for coord in p) for p in points_arr], [float(d) for d in depths_arr]
        
        return None, None
    
    def process_images(self, num_images=None, output_json_path=None):
        """
        Process images to generate point data.
        
        Args:
            num_images (int): Number of images to process (None for all loaded images)
            output_json_path (str): Path to the output JSON file (used for image directory structure)
        """
        print("Processing images to generate point data...")
        
        available_images = list(self.data_dict.keys())
        successful_count = 0
        
        if num_images is None:
            # Process all available images
            selected_images = available_images
            target_count = len(available_images)
        else:
            # We need to keep trying until we get the requested number of successful images
            target_count = min(num_images, len(available_images))
            selected_images = available_images.copy()  # Copy so we can modify it
            random.shuffle(selected_images)  # Randomize the order
        
        processed_images = []
        image_index = 0
        
        with tqdm(total=target_count, desc="Processing images") as pbar:
            while successful_count < target_count and image_index < len(selected_images):
                img_name = selected_images[image_index]
                image_index += 1
                
                # Skip if already processed (shouldn't happen in current logic, but safety check)
                if img_name in processed_images:
                    continue
                    
                # Randomly select number of points (3-5)
                num_points = random.randint(3, 5)
                # TODO testing
                # num_points = 2
                
                points, depths = self.select_random_points(img_name, num_points)
                
                if points is not None and depths is not None:
                    # Create point labels based on number of points
                    all_labels = ['A', 'B', 'C', 'D', 'E']
                    labels = all_labels[:num_points]
                    random.shuffle(labels)
                    
                    # Draw markers and save image
                    if output_json_path:
                        drawn_image_path = self.draw_markers_and_save(img_name, points, depths, labels, num_points, output_json_path)
                    else:
                        # Fallback to output_dir if no JSON path provided
                        fallback_path = os.path.join(self.output_dir, "temp.json")
                        drawn_image_path = self.draw_markers_and_save(img_name, points, depths, labels, num_points, fallback_path)
                    
                    self.point_data[img_name] = {
                        'coordinates': points,
                        'depths': depths,
                        'num_points': num_points,
                        'labels': labels
                    }
                    
                    # Store the path to the drawn image
                    self.drawn_image_paths[img_name] = drawn_image_path
                    processed_images.append(img_name)
                    successful_count += 1
                    pbar.update(1)
                
                # If we've tried all available images but still need more, break
                if image_index >= len(selected_images) and successful_count < target_count:
                    print(f"Warning: Only {successful_count} out of {target_count} images could be processed successfully")
                    break
        
        print(f"Successfully processed {successful_count} images with valid point configurations")
        return successful_count
    
    def generate_qa_pairs(self):
        """Generate question-answer pairs from point data."""
        print("Generating question-answer pairs...")
        
        qa_pairs = {}
        point_mappings = {}
        
        for img_name in tqdm(self.point_data.keys(), desc="Generating QA pairs"):
            points = self.point_data[img_name]['coordinates']
            depths = self.point_data[img_name]['depths']
            num_points = self.point_data[img_name]['num_points']
            labels = self.point_data[img_name]['labels']  # Use pre-generated labels
            
            # Create point label mapping
            point_label_mapping = {}
            for i, (x, y) in enumerate(points):
                point_label_mapping[labels[i]] = {
                    'coordinate': points[i],
                    'depth': depths[i]
                }
            
            # Find the point with maximum depth (closest to camera)
            max_depth_idx = depths.index(max(depths))
            max_depth_label = labels[max_depth_idx]
            
            # Create query-answer pair
            query = "Multiple points are circled on the image, labeled by letters beside each circle. Which point is the closest to the camera?"
            answer = f"{max_depth_label}"
            
            qa_pairs[img_name] = {
                "query": query,
                "answer": answer
            }
            point_mappings[img_name] = point_label_mapping
        
        return qa_pairs, point_mappings
    
    def prepare_conversation_format(self, qa_pairs, point_mappings, output_file, answer_type='both'):
        """
        Prepare the final dataset in conversation format.
        
        Args:
            qa_pairs (dict): Question-answer pairs
            point_mappings (dict): Point label mappings
            output_file (str): Output JSON file path
            answer_type (str): Type of answers to generate ('short', 'long', or 'both')
        """
        print("Preparing conversation format dataset...")
        
        data = []
        entry_id = 0
        
        for i, img_name in enumerate(qa_pairs.keys()):
            question = qa_pairs[img_name]['query']
            answer = qa_pairs[img_name]['answer']
            mapping = point_mappings[img_name]
            
            # Build coordinate information for the answer
            coord_info = []
            for label in sorted(mapping.keys()):
                x, y = mapping[label]['coordinate']
                coord_info.append(f"Point {label} is located at (x = {x} y = {y})")
            
            coord_text = ",".join(coord_info)
            
            # Get depth value from loaded dictionary if available
            depth_value = self.get_depth_value_for_image(img_name)
            
            # Create the detailed answer format
            detailed_answer = (
                f"{coord_text}.The depth map for the image is {depth_value}. "
                f"Since point {answer} has a higher pixel value on the depth map, "
                f"the answer is that point {answer} is closer to the camera."
            )
            
            # Use drawn image path instead of original image path
            image_path = self.drawn_image_paths[img_name]
            
            # Use original filename for embedding (without reformatting)
            embedding_filename = img_name.replace('.jpg', '.npy')
            
            # Create SHORT form conversation entry
            short_entry = {
                "id": f"depth_point_{entry_id}",
                "image": image_path,
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>\n{question}"
                    },
                    {
                        "from": "gpt",
                        "value": f"({answer})"
                    }
                ],
                "embedding": os.path.join(str(DATA_ROOT), "train_annealing_data", "<encoder_stub>", embedding_filename)
            }
            
            # Create LONG form conversation entry  
            long_entry_id = entry_id + (1 if answer_type in ['short', 'both'] else 0)
            long_entry = {
                "id": f"depth_point_{long_entry_id}",
                "image": image_path,
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<image>\n{question}\nTo answer this question, let's think through it step by step, and we know the image is {self.target_size} x {self.target_size}. First, what are the coordinates of points in the image? Second, what is the depth map for the image? Which point has a higher pixel value on the depth map? Remember, higher values indicate that the point is closer to the camera."
                    },
                    {
                        "from": "gpt",
                        "value": detailed_answer
                    }
                ],
                "embedding": os.path.join(str(DATA_ROOT), "train_annealing_data", "<encoder_stub>", embedding_filename)
            }
            
            # Add entries based on answer_type parameter
            if answer_type in ['short', 'both']:
                data.append(short_entry)
                entry_id += 1
            if answer_type in ['long', 'both']:
                data.append(long_entry)
                entry_id += 1
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")
        
        # Save to JSON file
        try:
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving output file {output_file}: {e}")
            # Try saving to current directory as fallback
            fallback_file = os.path.basename(output_file)
            print(f"Trying fallback location: {fallback_file}")
            with open(fallback_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved to fallback location: {fallback_file}")
            return len(data)
        
        print(f"Saved {len(data)} conversation entries to {output_file}")
        return len(data)
    
    def _load_and_format_readme_template(self, prefix, total_samples, image_resolution, output_json_path):
        """
        Load the README template from external file and format it with dynamic values.
        
        Args:
            prefix (str): Dataset prefix for filenames
            total_samples (int): Total number of samples in the dataset
            image_resolution (int): Image resolution (width/height)
            output_json_path (str): Path to the output JSON file
            
        Returns:
            str: Formatted README content
        """
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        template_path = os.path.join(script_dir, 'hf_readme_template.md')
        
        try:
            with open(template_path, 'r') as f:
                template_content = f.read()
            
            # Format the template with dynamic values
            formatted_content = template_content.format(
                prefix=prefix,
                total_samples=total_samples,
                image_resolution=image_resolution,
                output_json_path=output_json_path
            )
            
            return formatted_content
            
        except FileNotFoundError:
            print(f"Warning: README template not found at {template_path}")
            # Fallback to a basic README
            return f"""---
license: mit
task_categories:
- visual-question-answering
- depth-estimation
language:
- en
tags:
- depth
- spatial-reasoning
- computer-vision
- multimodal
size_categories:
- 1K<n<10K
---

# Depth Point Dataset

This dataset contains {total_samples} depth-based question-answer pairs for training vision-language models.

- **Total samples**: {total_samples}
- **Image resolution**: {image_resolution}x{image_resolution}
- **Task**: Identify which labeled point is closest to the camera based on depth
"""
        except Exception as e:
            print(f"Warning: Error loading README template: {e}")
            return f"# Depth Point Dataset\n\nThis dataset contains {total_samples} samples."
    
    def save_intermediate_files(self, qa_pairs, point_mappings, output_json_path):
        """Save intermediate debugging files for analysis."""
        json_dir = os.path.dirname(output_json_path)
        
        # Save point data
        point_data_file = os.path.join(json_dir, 'depth_point_data.json')
        with open(point_data_file, 'w') as f:
            json.dump(self.point_data, f, indent=2)
        print(f"Saved point data to: {point_data_file}")
        
        # Save QA pairs
        qa_file = os.path.join(json_dir, 'depth_qa_pairs.json')
        with open(qa_file, 'w') as f:
            json.dump(qa_pairs, f, indent=2)
        print(f"Saved QA pairs to: {qa_file}")
        
        # Save label mappings
        mappings_file = os.path.join(json_dir, 'depth_label_mappings.json')
        with open(mappings_file, 'w') as f:
            json.dump(point_mappings, f, indent=2)
        print(f"Saved label mappings to: {mappings_file}")
    
    def create_hf_comprehensive_dataset(self, qa_pairs, point_mappings, output_json_path, prefix="depth_point", answer_type='both'):
        """Create HuggingFace-compatible dataset with proper image path handling."""
        print("Creating HuggingFace-compatible dataset...")
        
        # Use the same directory as the output JSON file
        json_dir = os.path.dirname(output_json_path)
        os.makedirs(json_dir, exist_ok=True)
        
        # Create HuggingFace viewer-friendly dataset (flat structure)
        hf_viewer_data = []
        
        # Also maintain the LLaVA training format
        llava_training_data = []
        
        entry_id = 0
        
        # Process each image
        for img_name in qa_pairs.keys():
            question = qa_pairs[img_name]['query']
            answer = qa_pairs[img_name]['answer']
            mapping = point_mappings[img_name]
            point_data = self.point_data[img_name]
            
            # Build coordinate information
            coord_info = []
            point_coordinates = {}
            point_depths = {}
            
            for label in sorted(mapping.keys()):
                x, y = mapping[label]['coordinate']
                depth = mapping[label]['depth']
                coord_info.append(f"Point {label} is located at (x = {x} y = {y})")
                point_coordinates[f"point_{label}_x"] = x
                point_coordinates[f"point_{label}_y"] = y
                point_depths[f"point_{label}_depth"] = depth
            
            coord_text = ",".join(coord_info)
            
            # Get depth value
            depth_value = self.get_depth_value_for_image(img_name)
            
            # Create detailed answer
            detailed_answer = (
                f"{coord_text}.The depth map for the image is {depth_value}. "
                f"Since point {answer} has a higher pixel value on the depth map, "
                f"the answer is that point {answer} is closer to the camera."
            )
            
            # Get paths
            original_image_path = self.data_dict[img_name]['image']
            drawn_image_path = self.drawn_image_paths[img_name]
            embedding_filename = img_name.replace('.jpg', '.npy')
            embedding_path = os.path.join(str(DATA_ROOT), "train_annealing_data", "<encoder_stub>", embedding_filename)
            
            # Create HuggingFace-compatible image path (just the filename within images folder)
            hf_image_filename = os.path.join("images", os.path.basename(drawn_image_path))
            
            # Create HuggingFace viewer entries (flat structure for easy viewing)
            if answer_type in ['short', 'both']:
                hf_entry = {
                    "id": f"depth_point_{entry_id}",
                    "image": hf_image_filename,  # Just filename - HF will look in same directory
                    "image_filename": hf_image_filename,
                    "original_image": img_name,
                    "question": question,
                    "answer": f"({answer})",
                    "answer_letter": answer,
                    "answer_type": "short",
                    "num_points": point_data['num_points'],
                    "point_labels": point_data['labels'],
                    "depth_token": depth_value,
                    **point_coordinates,  # Flatten point coordinates
                    **point_depths,  # Flatten point depths
                }
                hf_viewer_data.append(hf_entry)
                
                # LLaVA training entry (unchanged, uses absolute paths)
                llava_entry = {
                    "id": f"depth_point_{entry_id}",
                    "image": drawn_image_path,  # Keep absolute path for LLaVA
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"<image>\n{question}"
                        },
                        {
                            "from": "gpt",
                            "value": f"({answer})"
                        }
                    ],
                    "embedding": embedding_path
                }
                llava_training_data.append(llava_entry)
                entry_id += 1
            
            if answer_type in ['long', 'both']:
                long_question = f"{question}\nTo answer this question, let's think through it step by step, and we know the image is {self.target_size} x {self.target_size}. First, what are the coordinates of points in the image? Second, what is the depth map for the image? Which point has a higher pixel value on the depth map? Remember, higher values indicate that the point is closer to the camera."
                
                hf_entry_long = {
                    "id": f"depth_point_{entry_id}",
                    "image": hf_image_filename,  # Just filename - HF will look in same directory
                    "image_filename": hf_image_filename,
                    "original_image": img_name,
                    "question": long_question,
                    "answer": detailed_answer,
                    "answer_letter": answer,
                    "answer_type": "long",
                    "num_points": point_data['num_points'],
                    "point_labels": point_data['labels'],
                    "depth_token": depth_value,
                    **point_coordinates,  # Flatten point coordinates
                    **point_depths,  # Flatten point depths
                }
                hf_viewer_data.append(hf_entry_long)
                
                # LLaVA training entry (unchanged, uses absolute paths)
                llava_entry_long = {
                    "id": f"depth_point_{entry_id}",
                    "image": drawn_image_path,  # Keep absolute path for LLaVA
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"<image>\n{long_question}"
                        },
                        {
                            "from": "gpt",
                            "value": detailed_answer
                        }
                    ],
                    "embedding": embedding_path
                }
                llava_training_data.append(llava_entry_long)
                entry_id += 1
        
        # Save HuggingFace viewer-friendly dataset
        hf_viewer_file = os.path.join(json_dir, f'{prefix}_hf_viewer.json')
        try:
            with open(hf_viewer_file, 'w') as f:
                json.dump(hf_viewer_data, f, indent=2)
            print(f"Saved HuggingFace viewer dataset to: {hf_viewer_file}")
        except Exception as e:
            print(f"Error saving HF viewer dataset: {e}")
            return False
        
        # Save LLaVA training format (this is what mix_depth.json will point to)
        llava_training_file = output_json_path  # Use the original output path for LLaVA
        try:
            with open(llava_training_file, 'w') as f:
                json.dump(llava_training_data, f, indent=2)
            print(f"Saved LLaVA training format to: {llava_training_file}")
        except Exception as e:
            print(f"Error saving LLaVA training file: {e}")
            return False

        # Create a README for HuggingFace upload
        readme_content = self._load_and_format_readme_template(
            prefix=prefix,
            total_samples=len(hf_viewer_data),
            image_resolution=self.target_size,
            output_json_path=os.path.basename(output_json_path)
        )
        
        readme_file = os.path.join(json_dir, 'README.md')
        try:
            with open(readme_file, 'w') as f:
                f.write(readme_content)
            print(f"Created README.md for HuggingFace: {readme_file}")
        except Exception as e:
            print(f"Warning: Could not create README: {e}")
        
        # Create dataset_infos.json for HuggingFace
        dataset_info = {
            "default": {
                "description": "Depth-based question-answer pairs for vision-language models",
                "features": {
                "id": { "_type": "Value", "dtype": "string" },
                "image": { "_type": "Image" },
                "image_filename": { "_type": "Value", "dtype": "string" },
                "original_image": { "_type": "Value", "dtype": "string" },
                "question": { "_type": "Value", "dtype": "string" },
                "answer": { "_type": "Value", "dtype": "string" },
                "answer_letter": { "_type": "Value", "dtype": "string" },
                "answer_type": { "_type": "Value", "dtype": "string" },
                "num_points": { "_type": "Value", "dtype": "int32" },
                "point_labels": { "_type": "Sequence", "feature": { "_type": "Value", "dtype": "string" } },
                "depth_token": { "_type": "Value", "dtype": "string" },

                "point_A_x": { "_type": "Value", "dtype": "float32" },
                "point_A_y": { "_type": "Value", "dtype": "float32" },
                "point_A_depth": { "_type": "Value", "dtype": "float32" },
                "point_B_x": { "_type": "Value", "dtype": "float32" },
                "point_B_y": { "_type": "Value", "dtype": "float32" },
                "point_B_depth": { "_type": "Value", "dtype": "float32" },
                "point_C_x": { "_type": "Value", "dtype": "float32" },
                "point_C_y": { "_type": "Value", "dtype": "float32" },
                "point_C_depth": { "_type": "Value", "dtype": "float32" },
                "point_D_x": { "_type": "Value", "dtype": "float32" },
                "point_D_y": { "_type": "Value", "dtype": "float32" },
                "point_D_depth": { "_type": "Value", "dtype": "float32" },
                "point_E_x": { "_type": "Value", "dtype": "float32" },
                "point_E_y": { "_type": "Value", "dtype": "float32" },
                "point_E_depth": { "_type": "Value", "dtype": "float32" }
                },
                "splits": {
                "train": { "name": "train", "num_examples": len(hf_viewer_data), "dataset_name": "ade20k_depth_discrete_long_only" }
                },
                "license": "mit",
                "homepage": ""
            }
            }

        
        dataset_info_file = os.path.join(json_dir, 'dataset_infos.json')
        try:
            with open(dataset_info_file, 'w') as f:
                json.dump(dataset_info, f, indent=2)
            print(f"Created dataset_infos.json: {dataset_info_file}")
        except Exception as e:
            print(f"Warning: Could not create dataset_infos.json: {e}")
        
        print(f"\n=== HuggingFace Dataset Created ===")
        print(f"HF Viewer Format: {hf_viewer_file} ({len(hf_viewer_data)} entries)")
        print(f"LLaVA Training: {llava_training_file} ({len(llava_training_data)} entries)")
        print(f"Images Directory: {os.path.join(json_dir, 'images')}")
        print(f"README: {readme_file}")
        print(f"\nThe dataset is now ready for HuggingFace upload!")
        print(f"Directory structure:")
        print(f"  {json_dir}/")
        print(f"  ├── {os.path.basename(hf_viewer_file)}")
        print(f"  ├── {os.path.basename(llava_training_file)}")
        print(f"  ├── README.md")
        print(f"  ├── dataset_infos.json")
        print(f"  └── images/")
        print(f"      ├── 2pts_image1.jpg")
        print(f"      ├── 3pts_image2.jpg")
        print(f"      └── ...")
        
        return True
    
    def load_existing_hf_viewer_data(self, hf_viewer_json_path):
        """
        Load existing HuggingFace viewer JSON data.
        
        Args:
            hf_viewer_json_path (str): Path to existing depth_point_hf_viewer.json file
            
        Returns:
            list: Loaded data entries or empty list if failed
        """
        try:
            with open(hf_viewer_json_path, 'r') as f:
                data = json.load(f)
            print(f"Successfully loaded {len(data)} entries from {hf_viewer_json_path}")
            return data
        except FileNotFoundError:
            print(f"Error: File not found: {hf_viewer_json_path}")
            return []
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format in {hf_viewer_json_path}: {e}")
            return []
        except Exception as e:
            print(f"Error loading file {hf_viewer_json_path}: {e}")
            return []
    
    def convert_to_format(self, loaded_data, output_format='both', output_file=None):
        """
        Convert loaded HF viewer data to specified format (short, long, or both).
        This process does not redraw images or change image paths.
        
        Args:
            loaded_data (list): Data loaded from HF viewer JSON
            output_format (str): 'short', 'long', or 'both'
            output_file (str): Output JSON file path (optional)
            
        Returns:
            list: Converted data in LLaVA training format
        """
        print(f"Converting {len(loaded_data)} entries to {output_format} format...")
        
        converted_data = []
        entry_id = 0
        
        # Group entries by original image to avoid duplicates
        image_entries = {}
        for entry in loaded_data:
            original_image = entry.get('original_image', '')
            if original_image not in image_entries:
                image_entries[original_image] = []
            image_entries[original_image].append(entry)
        
        for original_image, entries in image_entries.items():
            # Use the first entry as the base (they should have the same image data)
            base_entry = entries[0]
            
            # Extract the absolute image path from the existing entry
            # Convert relative HF path to absolute path
            hf_image_path = base_entry.get('image', '')
            if hf_image_path.startswith('images/'):
                # Construct absolute path based on the original structure
                base_dir = os.path.dirname(output_file) if output_file else str(DATA_ROOT / "mixed_depth")
                absolute_image_path = os.path.join(base_dir, hf_image_path)
            else:
                absolute_image_path = hf_image_path
            
            # Get embedding path
            original_img_name = base_entry.get('original_image', '')
            embedding_filename = original_img_name.replace('.jpg', '.npy')
            embedding_path = os.path.join(str(DATA_ROOT), "train_annealing_data", "<encoder_stub>", embedding_filename)
            
            # Basic question
            basic_question = "Multiple points are circled on the image, labeled by letters beside each circle. Which point is the closest to the camera?"
            
            # Extended question for long format
            extended_question = f"{basic_question}\nTo answer this question, let's think through it step by step, and we know the image is 336 x 336. First, what are the coordinates of points in the image? Second, what is the depth map for the image? Which point has a higher pixel value on the depth map? Remember, higher values indicate that the point is closer to the camera."
            
            # Get answer information
            answer_letter = base_entry.get('answer_letter', '')
            
            # Build detailed answer from the base entry data
            point_coordinates = []
            for label in ['A', 'B', 'C', 'D', 'E']:
                x_key = f'point_{label}_x'
                y_key = f'point_{label}_y'
                if x_key in base_entry and y_key in base_entry:
                    x = base_entry[x_key]
                    y = base_entry[y_key]
                    point_coordinates.append(f"Point {label} is located at (x = {int(x)} y = {int(y)})")
            
            coord_text = ",".join(point_coordinates)
            depth_token = base_entry.get('depth_token', '<DEPTH_TOKEN>')
            
            detailed_answer = (
                f"{coord_text}.The depth map for the image is {depth_token}. "
                f"Since point {answer_letter} has a higher pixel value on the depth map, "
                f"the answer is that point {answer_letter} is closer to the camera."
            )
            
            # Create entries based on output format
            if output_format in ['short', 'both']:
                short_entry = {
                    "id": f"depth_point_{entry_id}",
                    "image": absolute_image_path,
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"<image>\n{basic_question}"
                        },
                        {
                            "from": "gpt",
                            "value": f"({answer_letter})"
                        }
                    ],
                    "embedding": embedding_path
                }
                converted_data.append(short_entry)
                entry_id += 1
            
            if output_format in ['long', 'both']:
                long_entry = {
                    "id": f"depth_point_{entry_id}",
                    "image": absolute_image_path,
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"<image>\n{extended_question}"
                        },
                        {
                            "from": "gpt",
                            "value": detailed_answer
                        }
                    ],
                    "embedding": embedding_path
                }
                converted_data.append(long_entry)
                entry_id += 1
        
        print(f"Converted to {len(converted_data)} entries in LLaVA training format")
        
        # Save to file if specified
        if output_file:
            # Add suffix to filename based on output format
            if output_format == 'short':
                output_file = self._add_suffix_to_filename(output_file, '_short')
            elif output_format == 'long':
                output_file = self._add_suffix_to_filename(output_file, '_long')
            # For 'both', keep original filename
            
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            
            try:
                with open(output_file, 'w') as f:
                    json.dump(converted_data, f, indent=2)
                print(f"Saved converted data to: {output_file}")
            except Exception as e:
                print(f"Error saving converted data: {e}")
        
        return converted_data
    
    def _add_suffix_to_filename(self, filepath, suffix):
        """
        Add a suffix to a filename before the extension.
        
        Args:
            filepath (str): Original file path
            suffix (str): Suffix to add (e.g., '_short', '_long')
            
        Returns:
            str: Modified file path with suffix
        """
        if not filepath:
            return filepath
        
        # Split the path into directory, name, and extension
        dir_path = os.path.dirname(filepath)
        filename = os.path.basename(filepath)
        name, ext = os.path.splitext(filename)
        
        # Add suffix to the name
        new_filename = f"{name}{suffix}{ext}"
        
        # Reconstruct the full path
        return os.path.join(dir_path, new_filename)

    def upload_to_huggingface(self, dataset_dir, repo_id, token=None):
        """
        Upload the dataset to Hugging Face Hub.
        
        Args:
            dataset_dir (str): Directory containing the dataset files
            repo_id (str): Repository ID in format "username/repo-name"
            token (str): HuggingFace token (optional, can be set via environment)
        """
        try:
            from huggingface_hub import HfApi
        except ImportError:
            print("Error: huggingface_hub is not installed. Install with: pip install huggingface_hub")
            return False
        
        print(f"\n=== Uploading to Hugging Face ===")
        print(f"Repository: {repo_id}")
        print(f"Dataset directory: {dataset_dir}")
        
        api = HfApi(token=token)
        
        try:
            # Check if directory exists
            if not os.path.exists(dataset_dir):
                print(f"Error: Dataset directory does not exist: {dataset_dir}")
                return False
            
            # Upload the entire dataset folder
            print("Uploading dataset folder...")
            api.upload_folder(
                folder_path=dataset_dir,
                path_in_repo="",  # Upload to root of repository
                repo_id=repo_id,
                repo_type="dataset",
                commit_message="Upload depth point dataset"
            )
            
            print(f"✓ Successfully uploaded dataset to https://huggingface.co/datasets/{repo_id}")
            return True
            
        except Exception as e:
            print(f"Error uploading to Hugging Face: {e}")
            print("Make sure you have:")
            print("1. A valid Hugging Face token set (via --hf-token or HF_TOKEN environment variable)")
            print("2. Created the repository on Hugging Face first")
            print("3. Have write access to the repository")
            return False

def main():
    parser = argparse.ArgumentParser(description='Generate depth-based question-answer pairs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--num-images', type=int, default=None, help='Number of images to process (default: all)')
    parser.add_argument('--output', type=str, default='depth_point_dataset.json', help='Output JSON file')
    parser.add_argument('--output-dir', type=str, default='depth_point_output', help='Output directory for drawn images')
    parser.add_argument('--save-intermediates', action='store_true', 
                        help='Save intermediate debugging files (point_data.json, qa.json, label_mappings.json) for analysis')
    parser.add_argument('--min-distance', type=int, default=20, help='Minimum pixel distance between points')
    parser.add_argument('--min-depth-diff', type=int, default=20, help='Minimum depth difference between points')
    parser.add_argument('--order', type=str, default='random', choices=['random', 'sequential', 'reverse'],
                        help='Image loading order: random (default), sequential (alphabetical), or reverse (reverse alphabetical)')
    parser.add_argument('--answer-type', type=str, default='both', choices=['short', 'long', 'both'],
                        help='Type of answers to generate: short (brief answers), long (detailed answers), or both (default)')
    parser.add_argument('--hf-upload', action='store_true',
                        help='Create comprehensive HuggingFace-compatible dataset with merged intermediate states')
    parser.add_argument('--hf-repo', type=str, default=None,
                        help='HuggingFace repository ID (username/repo-name) for upload')
    parser.add_argument('--hf-token', type=str, default=None,
                        help='HuggingFace token for authentication (can also use HF_TOKEN env var)')
    
    # New arguments for loading existing data
    parser.add_argument('--load-existing', type=str, default=None,
                        help='Path to existing depth_point_hf_viewer.json file to load and convert')
    parser.add_argument('--convert-only', action='store_true',
                        help='Only convert existing data to specified format, skip image processing')
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = DepthPointGenerator(
        seed=args.seed,
        target_size=336,
        min_distance=args.min_distance,
        min_depth_diff=args.min_depth_diff,
        output_dir=args.output_dir
    )
    
    # Handle loading existing data mode
    if args.load_existing or args.convert_only:
        # Load existing HF viewer data
        hf_viewer_path = args.load_existing or str(DATA_ROOT / "mixed_depth" / "depth_point_hf_viewer.json")
        
        if not os.path.exists(hf_viewer_path):
            print(f"Error: Existing data file not found: {hf_viewer_path}")
            return
        
        print(f"Loading existing data from: {hf_viewer_path}")
        existing_data = generator.load_existing_hf_viewer_data(hf_viewer_path)
        
        if not existing_data:
            print("Failed to load existing data!")
            return
        
        # Convert to specified format
        print(f"Converting to {args.answer_type} format...")
        converted_data = generator.convert_to_format(existing_data, args.answer_type, args.output)
        
        # Determine actual output filename with suffix
        actual_output_file = args.output
        if args.answer_type == 'short':
            actual_output_file = generator._add_suffix_to_filename(args.output, '_short')
        elif args.answer_type == 'long':
            actual_output_file = generator._add_suffix_to_filename(args.output, '_long')
        
        # Print summary
        print(f"\n=== Conversion Summary ===")
        print(f"Loaded {len(existing_data)} entries from existing data")
        print(f"Converted to {len(converted_data)} entries in {args.answer_type} format")
        print(f"Output saved to: {actual_output_file}")
        print("Note: No images were redrawn, existing image paths preserved")
        
        return
    
    # Original workflow for generating new data
    # Load image pairs (limit loading if num_images specified)
    num_pairs = generator.load_image_pairs(args.num_images, args.order)
    if num_pairs == 0:
        print("No image pairs found! Check dataset paths.")
        return
    
    # Process images (will process all loaded images)
    successful_count = generator.process_images(output_json_path=args.output)
    if successful_count == 0:
        print("No images were successfully processed!")
        return
    
    # Generate QA pairs
    qa_pairs, point_mappings = generator.generate_qa_pairs()
    
    # Handle different output modes
    if args.hf_upload:
        # Create comprehensive HuggingFace dataset (includes everything)
        success = generator.create_hf_comprehensive_dataset(qa_pairs, point_mappings, args.output, answer_type=args.answer_type)
        if not success:
            print("Failed to create comprehensive HuggingFace dataset!")
            return
        
        # Upload to HuggingFace if repo specified
        if args.hf_repo:
            dataset_dir = os.path.dirname(args.output)
            # Get token from argument or environment variable
            token = args.hf_token or os.environ.get('HF_TOKEN')
            upload_success = generator.upload_to_huggingface(dataset_dir, args.hf_repo, token)
            if not upload_success:
                print("Dataset creation succeeded but upload failed.")
        elif args.hf_upload:
            print("Note: Dataset created but not uploaded. Use --hf-repo to specify repository for upload.")
        
        # Count entries for summary (both short and long)
        if args.answer_type == 'short':
            num_entries = successful_count
        elif args.answer_type == 'long':
            num_entries = successful_count
        else:  # both
            num_entries = successful_count * 2
    else:
        # Original workflow
        # Save intermediate files if requested
        if args.save_intermediates:
            generator.save_intermediate_files(qa_pairs, point_mappings, args.output)
        
        # Prepare final conversation format
        num_entries = generator.prepare_conversation_format(qa_pairs, point_mappings, args.output, args.answer_type)
    
    print(f"\n=== Summary ===")
    print(f"Loaded {num_pairs} image pairs")
    print(f"Successfully processed {successful_count} images")
    
    # Calculate short and long counts based on answer_type
    if args.answer_type == 'short':
        short_count = num_entries
        long_count = 0
    elif args.answer_type == 'long':
        short_count = 0
        long_count = num_entries
    else:  # both
        short_count = num_entries // 2
        long_count = num_entries // 2
    
    print(f"Generated {num_entries} conversation entries ({short_count} short + {long_count} long form)")
    json_dir = os.path.dirname(args.output)
    
    if args.hf_upload:
        print(f"Images saved to: {json_dir}/images/")
        print(f"=== HuggingFace Dataset Created ===")
        print(f"HF Viewer Format: {json_dir}/depth_point_hf_viewer.json")
        print(f"LLaVA Training Format: {args.output}")
        print(f"Features:")
        print(f"  ✓ LLaVA training compatibility maintained")
        print(f"  ✓ HuggingFace dataset viewer compatible")
        print(f"  ✓ Single images directory for proper path resolution")
        print(f"  ✓ HuggingFace upload ready")
        if args.hf_repo:
            print(f"  ✓ Uploaded to https://huggingface.co/datasets/{args.hf_repo}")
        else:
            print(f"  ⚠ Use --hf-repo to upload to HuggingFace")
    else:
        print(f"Drawn images saved to: {json_dir}/images/")
        print(f"Dataset JSON saved to: {args.output}")
        if args.save_intermediates:
            print(f"Intermediate files saved to: {json_dir}/depth_point_*.json")

if __name__ == "__main__":
    main()
