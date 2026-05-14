# NEW: Aurora-only file; not present in upstream LLAVA.
# NEW: Baseline https://github.com/haotian-liu/LLaVA.git @ v1.2.2.post1 (24fa1d065bbeac8a145a796ab7218c6945a2536e).
# NEW: Aurora path: llava/model/language_model/checkmodel.py

import argparse
import torch
import os

def compute_weight_diff(w1, w2):
    """Compute various metrics between two weight tensors."""
    abs_diff = (w1 - w2).abs()
    is_equal = torch.all(w1 == w2).item()
    return {
        'max_diff': abs_diff.max().item(),
        'mean_diff': abs_diff.mean().item(),
        'l2_diff': torch.norm(w1 - w2).item(),
        'relative_diff': torch.norm(w1 - w2).item() / torch.norm(w1).item(),
        'is_equal': is_equal
    }

def load_model(args):
    model_path = os.path.expanduser(args.model_path)

    # Find all available checkpoints
    checkpoint_dirs = [d for d in os.listdir(model_path) if d.startswith('checkpoint-')]
    checkpoint_steps = [int(d.split('-')[1]) for d in checkpoint_dirs]
    checkpoint_steps.sort()  # Sort steps in ascending order

    # Check weights across checkpoints
    mydict = {}
    for checkpoint in checkpoint_steps:
        checkpoint_path = os.path.join(args.model_path, f"checkpoint-{checkpoint}", "non_lora_trainables.bin")
        if os.path.exists(checkpoint_path):
            print(f"\nLoading checkpoint {checkpoint} from {checkpoint_path}")
            try:
                state_dict = torch.load(checkpoint_path, map_location="cpu")
                mydict[str(checkpoint)] = {}
                
                # Check if model has new_head
                if 'base_model.model.new_head.weight' in state_dict:
                    mydict[str(checkpoint)]['new_head'] = state_dict['base_model.model.new_head.weight']
                    print(f"Found new_head in checkpoint {checkpoint}")
                    print(f"  new_head norm: {torch.norm(mydict[str(checkpoint)]['new_head']).item():.6f}")
                # Check if model has depth components
                elif 'base_model.model.depth_head.weight' in state_dict or 'base_model.model.depth_projector.weight' in state_dict:
                    if 'base_model.model.depth_head.weight' in state_dict:
                        mydict[str(checkpoint)]['depth_head'] = state_dict['base_model.model.depth_head.weight']
                        print(f"Found depth_head in checkpoint {checkpoint}")
                        print(f"  depth_head norm: {torch.norm(mydict[str(checkpoint)]['depth_head']).item():.6f}")
                    if 'base_model.model.depth_projector.weight' in state_dict:
                        mydict[str(checkpoint)]['depth_projector'] = state_dict['base_model.model.depth_projector.weight']
                        print(f"Found depth_projector in checkpoint {checkpoint}")
                        print(f"  depth_projector norm: {torch.norm(mydict[str(checkpoint)]['depth_projector']).item():.6f}")
                else:
                    print(f"Warning: No new_head or depth components found in checkpoint {checkpoint}")
                    continue
            except KeyError as e:
                print(f"Error: Could not find expected keys in checkpoint {checkpoint}. Error: {e}")
                print(f"Available keys: {list(state_dict.keys())}")
                continue
        else:
            print(f"Warning: Checkpoint {checkpoint} not found at {checkpoint_path}")

    # Check consecutive checkpoints for weight updates
    print("\nAnalyzing weight updates between checkpoints:")
    checkpoints = sorted([int(k) for k in mydict.keys()])
    for i in range(len(checkpoints)-1):
        curr_checkpoint = str(checkpoints[i])
        next_checkpoint = str(checkpoints[i+1])
        
        print(f"\nComparing checkpoint {curr_checkpoint} → {next_checkpoint}:")
        
        # Compare new_head weights if present
        if 'new_head' in mydict[curr_checkpoint] and 'new_head' in mydict[next_checkpoint]:
            head_diff = compute_weight_diff(
                mydict[curr_checkpoint]['new_head'],
                mydict[next_checkpoint]['new_head']
            )
            print(f"new_head changes:")
            print(f"  Max difference: {head_diff['max_diff']:.2e}")
            print(f"  Mean difference: {head_diff['mean_diff']:.2e}")
            print(f"  L2 norm of difference: {head_diff['l2_diff']:.2e}")
            print(f"  Relative change: {head_diff['relative_diff']:.2%}")
            print(f"  Weights are equal: {head_diff['is_equal']}")
            
            if head_diff['relative_diff'] < 1e-6:
                print("\n  WARNING: Very small or no changes detected in new_head between checkpoints!")
        
        # Compare depth components if present
        if 'depth_head' in mydict[curr_checkpoint] and 'depth_head' in mydict[next_checkpoint]:
            head_diff = compute_weight_diff(
                mydict[curr_checkpoint]['depth_head'],
                mydict[next_checkpoint]['depth_head']
            )
            print(f"depth_head changes:")
            print(f"  Max difference: {head_diff['max_diff']:.2e}")
            print(f"  Mean difference: {head_diff['mean_diff']:.2e}")
            print(f"  L2 norm of difference: {head_diff['l2_diff']:.2e}")
            print(f"  Relative change: {head_diff['relative_diff']:.2%}")
            print(f"  Weights are equal: {head_diff['is_equal']}")
            
            if head_diff['relative_diff'] < 1e-6:
                print("\n  WARNING: Very small or no changes detected in depth_head between checkpoints!")

        if 'depth_projector' in mydict[curr_checkpoint] and 'depth_projector' in mydict[next_checkpoint]:
            proj_diff = compute_weight_diff(
                mydict[curr_checkpoint]['depth_projector'],
                mydict[next_checkpoint]['depth_projector']
            )
            print(f"depth_projector changes:")
            print(f"  Max difference: {proj_diff['max_diff']:.2e}")
            print(f"  Mean difference: {proj_diff['mean_diff']:.2e}")
            print(f"  L2 norm of difference: {proj_diff['l2_diff']:.2e}")
            print(f"  Relative change: {proj_diff['relative_diff']:.2%}")
            print(f"  Weights are equal: {proj_diff['is_equal']}")
            
            if proj_diff['relative_diff'] < 1e-6:
                print("\n  WARNING: Very small or no changes detected in depth_projector between checkpoints!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model checkpoints directory")
    parser.add_argument("--model-base", type=str, help="Path to base model (not used)")
    args = parser.parse_args()

    load_model(args)
