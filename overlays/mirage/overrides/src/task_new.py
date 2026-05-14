from PIL import Image
from collections import deque
import os

def get_reasoning_image_path(sample):
    """
    Get the path to the reasoning image (ground truth image) for a sample.
    Handles different directory structures for training and testing.
    """
    # 1. Check if image_output is already in sample and exists
    if "image_output" in sample and sample["image_output"]:
        if os.path.exists(sample["image_output"]):
            return sample["image_output"]
            
    # 2. Try to derive it from image_input
    image_input = sample.get("image_input")
    if not image_input:
        return None
        
    if isinstance(image_input, list):
        if len(image_input) == 0:
            return None
        image_input = image_input[0]
        
    save_dir = os.path.dirname(image_input)
    map_id = str(sample.get("map_id", ""))
    
    # Possible paths
    # Train structure: .../level_3/ID/map_reasoning_path.png
    # Test structure: .../imgs_test/level3/img/ID_reasoning_path.png
    possible_paths = [
        os.path.join(save_dir, "map_reasoning_path.png"),
        os.path.join(save_dir, f"{map_id}_reasoning_path.png") if map_id else None,
        os.path.join(save_dir, f"{os.path.splitext(os.path.basename(image_input))[0]}_reasoning_path.png")
    ]
    
    for p in possible_paths:
        if p and os.path.exists(p):
            return p
            
    return None

def single_input_image_preprocess_function(sample):
    # Load images
    
    image = Image.open(sample["image_input"]).convert("RGB") 
    
    image_output_path = get_reasoning_image_path(sample)
    if image_output_path is None:
        # Fallback to what was there if possible, but it will likely fail if it doesn't exist
        image_output_path = sample.get("image_output", "")
        
    if not image_output_path or not os.path.exists(image_output_path):
        # We need image_output for training; if we can't find it, we might want to log or raise
        # For now, let's just try to open it and let it fail if it doesn't exist, 
        # but we've tried our best to find it.
        pass

    image_output = Image.open(image_output_path).convert("RGB")

    # Format conversations
    conversations = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": sample["text_input"]},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "image", "image": image_output},
                {"type": "text", "text": sample["text_output"]},
                ],
        }
    ]

    return conversations

def single_input_image_test_preprocess_function(sample):
    # Load images
    image = Image.open(sample["image_input"]).convert("RGB") 

    # Format conversations
    conversations = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": sample["text_input"]},
            ],
        },
    ]

    return conversations

def multiple_input_images_preprocess_function(sample):

    # Multiple input images
    user_content = []
    for image in sample['image_input']:
        user_content.append({"type": "image", "image": Image.open(image).convert("RGB") })
    user_content.append({"type": "text", "text": sample["text_input"]})

    image_output_path = get_reasoning_image_path(sample)
    if image_output_path is None:
        image_output_path = sample.get("image_output", "")
    
    image_output = Image.open(image_output_path).convert("RGB")

    conversations = [
        {
            "role": "user", 
            "content": user_content
        }, 
        {
            "role": "assistant", 
            "content": [
                {"type": "image", "image": image_output}, 
                {"type": "text", "text": sample["text_output"]}
                ],
        },
    ]

    return conversations

def multiple_input_images_test_preprocess_function(sample):

    # Multiple input images
    user_content = []
    for image in sample['image_input']:
        user_content.append({"type": "image", "image": Image.open(image).convert("RGB") })
    user_content.append({"type": "text", "text": sample["text_input"]})

    conversations = [
        {
            "role": "user", 
            "content": user_content
        }, 
    ]

    return conversations

task_preporcess_config = {
    'vsp-spatial-reasoning': single_input_image_preprocess_function,
    'vsp-spatial-planning': single_input_image_preprocess_function,
    'blink-jigsaw': multiple_input_images_preprocess_function,
    'sat': multiple_input_images_preprocess_function,
    'depth-reasoning': single_input_image_preprocess_function,
}

task_test_preporcess_config = {
    'vsp-spatial-reasoning': single_input_image_test_preprocess_function,
    'vsp-spatial-planning': single_input_image_test_preprocess_function,
    'blink-jigsaw': multiple_input_images_test_preprocess_function,
    'sat': multiple_input_images_test_preprocess_function,
    'depth-reasoning': single_input_image_test_preprocess_function,
}

# Define VSP valid actions and their corresponding (row, column) changes.
ACTION_MAP = {
    "LEFT":  (0, -1),
    "DOWN":  (1,  0),
    "RIGHT": (0,  1),
    "UP":    (-1, 0),
    "L":  (0, -1),
    "D":  (1,  0),
    "R": (0,  1),
    "U":  (-1, 0)
}

def parse_action_sequence(path_str):

    path_str = path_str.upper()
    action_sequence = [char for char in list(path_str) if char in ['U', 'D', 'R', 'L']]

    return action_sequence

def simulate_vsp(map_desc, path_str):
    """
    Simulate the action sequence on the provided map.
    """
    action_sequence = parse_action_sequence(path_str)

    start = None
    for r, row in enumerate(map_desc):
        for c, val in enumerate(row):
            if val == 1:
                start = (r, c)
                break
        if start is not None:
            break

    if start is None:
        raise ValueError("The map description does not contain a start position (cell value 1).")
    
    current_position = start
    for action in action_sequence:
        if action not in ACTION_MAP:
            return {
                "success": False,
                "status": "Invalid action",
                "invalid": True,
            }
            raise ValueError(f"Invalid action: {action}. Valid actions are {list(ACTION_MAP.keys())}.")
        
        dr, dc = ACTION_MAP[action]
        new_r = current_position[0] + dr
        new_c = current_position[1] + dc

        # Check boundaries; ignore the move if it goes out-of-bounds.
        if new_r < 0 or new_r >= len(map_desc) or new_c < 0 or new_c >= len(map_desc[0]):
            continue
        
        current_position = (new_r, new_c)
        
        # If the new cell is a hole (-1), immediately return failure.
        if map_desc[new_r][new_c] == -1:
            return {
                "success": False,
                "status": "Fell in hole",
                "invalid": False,
            }

    # Check if the final position is the goal (2).
    final_r, final_c = current_position
    if map_desc[final_r][final_c] == 2:
        return {
            "success": True,
            "status": "Reached goal",
            "invalid": False,
        }
    else:
        return {
            "success": False,
            "status": "Did not reach goal",
            "invalid": False,
        }


def find_shortest_path_bfs(map_desc):
    """
    Find the shortest path from start (1) to goal (2) using BFS.
    Returns a list of actions ['U', 'D', 'L', 'R'] or None if no path exists.
    """
    rows = len(map_desc)
    cols = len(map_desc[0]) if rows > 0 else 0
    
    # Find start and goal positions
    start = None
    goal = None
    for r in range(rows):
        for c in range(cols):
            if map_desc[r][c] == 1:
                start = (r, c)
            elif map_desc[r][c] == 2:
                goal = (r, c)
    
    if start is None or goal is None:
        return None
    
    # BFS
    queue = deque([(start, [])])  # (position, path)
    visited = {start}
    
    # Action mapping: (dr, dc) -> action_name
    action_reverse_map = {
        (-1, 0): 'U',
        (1, 0): 'D',
        (0, -1): 'L',
        (0, 1): 'R'
    }
    
    while queue:
        (r, c), path = queue.popleft()
        
        # Check if we reached the goal
        if (r, c) == goal:
            return path
        
        # Try all 4 directions
        for (dr, dc), action in action_reverse_map.items():
            new_r, new_c = r + dr, c + dc
            
            # Check boundaries
            if new_r < 0 or new_r >= rows or new_c < 0 or new_c >= cols:
                continue
            
            # Check if already visited
            if (new_r, new_c) in visited:
                continue
            
            # Check if it's a hole
            if map_desc[new_r][new_c] == -1:
                continue
            
            # Valid move
            visited.add((new_r, new_c))
            queue.append(((new_r, new_c), path + [action]))
    
    # No path found
    return None