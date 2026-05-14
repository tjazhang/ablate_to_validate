# NEW: Aurora-only file; not present in upstream LLAVA.
# NEW: Baseline https://github.com/haotian-liu/LLaVA.git @ v1.2.2.post1 (24fa1d065bbeac8a145a796ab7218c6945a2536e).
# NEW: Aurora path: llava/model/language_model/debug_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging # Optional, for warnings

# --- Constants ---
IGNORE_INDEX = -100
# Use placeholder IDs for the test script
# In a real scenario, these would come from the model config/tokenizer
DEPTH_START_ID = 47
DEPTH_END_ID = 48
DEPTH_TOKEN_ID = 49 # Placeholder ID used in labels between start/end

# --- Standalone combinedLoss Function (Adapted for Debugging) ---

def combinedLoss_debug(
    logits,
    labels,
    hidden_states,
    depth_targets,
    img_head, # Pass the mock img_head layer
    vocab_size,
    depth_start_id=DEPTH_START_ID, # Pass token IDs
    depth_end_id=DEPTH_END_ID,
    ignore_index=IGNORE_INDEX,
    lambda_depth=1.0,
    num_depth_tokens=None # Expected number of depth tokens between markers
):
    """
    Isolated combined loss function for debugging.

    Args:
        logits: Model output logits [batch_size, seq_len, vocab_size].
        labels: Ground truth labels [batch_size, seq_len].
        hidden_states: Hidden states from the model [batch_size, seq_len, hidden_size].
        depth_targets: Ground truth depth features [batch_size, num_depth_tokens, depth_embedding_dim] or None.
        img_head: A callable (like nn.Linear) mapping hidden_size to depth_embedding_dim.
        vocab_size: Size of the vocabulary.
        depth_start_id: Token ID marking the start of depth sequence.
        depth_end_id: Token ID marking the end of depth sequence.
        ignore_index: Label value to ignore in CE loss.
        lambda_depth: Weighting factor for the depth MSE loss.
        num_depth_tokens: Expected number of tokens between depth markers (for target alignment).

    Returns:
        Tuple: (total_loss, loss_ce, loss_mse)
    """
    if logits is None or labels is None or hidden_states is None:
        print("Error: logits, labels, and hidden_states must be provided.")
        return None, None, None

    batch_size, seq_length, _ = logits.shape
    loss_ce = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    loss_mse = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    # --- Calculate Depth Flag ---
    # Identify tokens *between* the start and end markers
    start_mask = (labels == depth_start_id)
    end_mask = (labels == depth_end_id)
    # Cumulative sum approach to find regions *after* start and *before* end
    shifted_start_mask = torch.cat([torch.zeros_like(start_mask[:, :1]), start_mask[:, :-1]], dim=1)
    depth_flag = (torch.cumsum(shifted_start_mask.int(), dim=1) - torch.cumsum(end_mask.int(), dim=1)) > 0
    print(f"Depth Start Mask:\n{start_mask.int()}")
    print(f"Depth End Mask:\n{end_mask.int()}")
    print(f"Depth Flag (Tokens between markers):\n{depth_flag.int()}")


    # --- 1. Cross-Entropy Loss (Language) ---
    # Shift logits and labels for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Shift the depth flag as well to align with shifted labels/logits
    shift_depth_flag = depth_flag[..., 1:].contiguous()

    # Create a copy of shifted labels to modify
    ce_labels = shift_labels.clone()
    # Ignore depth tokens in CE loss calculation
    ce_labels[shift_depth_flag] = ignore_index
    print(f"Shifted Logits shape: {shift_logits.shape}")
    print(f"CE Labels (Ignore Depth):\n{ce_labels}")


    # Flatten tokens for CE loss calculation
    try:
        loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
        loss_ce = loss_fct(shift_logits.view(-1, vocab_size), ce_labels.view(-1))
        print(f"\nCross-Entropy Loss (CE): {loss_ce.item():.4f}")
    except Exception as e:
        print(f"Error calculating CE loss: {e}")


    # --- 2. MSE Loss (Depth) ---
    if depth_targets is None:
        print("No depth_targets provided. MSE loss is 0.")
        loss_mse = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    else:
        # Extract the *unshifted* hidden states corresponding to the *original* depth flag
        depth_hidden_states = hidden_states[depth_flag]
        print(f"\nExtracted Depth Hidden States shape: {depth_hidden_states.shape}") # Should be [num_total_depth_tokens_in_batch, hidden_size]

        if depth_hidden_states.numel() == 0:
            print("No depth tokens found based on flags. MSE loss is 0.")
            loss_mse = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        else:
            # Ensure it's 2D: [N, hidden_size]
            if depth_hidden_states.ndim > 2:
                 depth_hidden_states = depth_hidden_states.reshape(-1, depth_hidden_states.size(-1))

            # Project hidden states to depth embedding dimension using the passed img_head
            predicted_depth_features = img_head(depth_hidden_states)
            print(f"Predicted Depth Features shape: {predicted_depth_features.shape}") # Should be [N, depth_embedding_dim]

            # Prepare targets
            # Assuming depth_targets shape: [batch_size, num_depth_tokens_per_sample, depth_embedding_dim]
            if depth_targets.ndim == 3:
                 # We need to flatten targets in the same order as predictions were extracted
                 # This requires knowing how many depth tokens belong to each batch item
                 num_depth_per_batch = depth_flag.sum(dim=1) # Count depth tokens per batch item
                 target_segments = []
                 expected_total_tokens = 0
                 actual_tokens_in_batch = predicted_depth_features.shape[0]

                 # Check if the number of expected depth tokens per sample matches the target shape
                 # This assumes a *fixed* number of depth tokens per sample in targets
                 if num_depth_tokens is None:
                      num_depth_tokens = depth_targets.shape[1]
                      print(f"Inferring num_depth_tokens from target shape: {num_depth_tokens}")

                 processed_targets = 0
                 for i in range(batch_size):
                     num_actual = num_depth_per_batch[i].item()
                     if num_actual > 0:
                         # Basic check: Does the number of found flags match expected?
                         if num_actual != num_depth_tokens:
                            print(f"Warning: Batch {i}: Found {num_actual} depth tokens, but target has {num_depth_tokens}. Alignment might be incorrect.")
                         # Take the targets for this batch item
                         target_segments.append(depth_targets[i, :num_actual, :]) # Slice targets to match actual found tokens
                         expected_total_tokens += num_actual
                     else:
                         # No depth tokens found for this batch item based on flags
                         pass

                 # Only proceed if we have segments and counts match
                 if target_segments and actual_tokens_in_batch == expected_total_tokens:
                     depth_targets_flat = torch.cat(target_segments, dim=0)
                     print(f"Flattened Depth Targets shape: {depth_targets_flat.shape}")
                     try:
                        loss_mse = F.mse_loss(predicted_depth_features, depth_targets_flat, reduction="mean")
                        print(f"Mean Squared Error Loss (MSE): {loss_mse.item():.4f}")
                     except Exception as e:
                        print(f"Error calculating MSE loss: {e}")
                 else:
                      print("Warning: Mismatch between predicted depth tokens and target structure or count. Skipping MSE calculation.")
                      loss_mse = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

            else: # Handle case where targets might already be flat? Unlikely.
                print("Warning: depth_targets is not 3D. MSE calculation might be incorrect.")
                # Attempt MSE anyway if shapes match somehow? Risky.
                if predicted_depth_features.shape == depth_targets.shape:
                     loss_mse = F.mse_loss(predicted_depth_features, depth_targets, reduction="mean")
                     print(f"Mean Squared Error Loss (MSE) (Targets were not 3D): {loss_mse.item():.4f}")
                else:
                     print("Shapes mismatch between predicted depth features and non-3D targets. Skipping MSE.")
                     loss_mse = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)


    # --- Total Loss ---
    total_loss = loss_ce + lambda_depth * loss_mse
    print(f"\nLambda Depth: {lambda_depth}")
    print(f"Total Combined Loss: {total_loss.item():.4f}")

    return total_loss, loss_ce, loss_mse

# --- Synthetic Data Setup ---
batch_size = 2
seq_len = 15
vocab_size = 50  # Small vocab for testing
hidden_size = 32 # Small hidden dim
depth_embedding_dim = 16 # Dim of depth features
num_depth_tokens_target = 5 # Expected depth tokens between markers for targets

# Mock img_head layer
mock_img_head = nn.Linear(hidden_size, depth_embedding_dim)

# --- Input Tensors ---
# Logits (random)
logits = torch.randn(batch_size, seq_len, vocab_size)

# Hidden States (random)
hidden_states = torch.randn(batch_size, seq_len, hidden_size)

# Labels - Carefully crafted
# Batch 1: Text -> Depth Sequence -> Text
# Batch 2: Only Text (no depth markers)
labels = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long)
# Make Batch 1 have depth sequence
labels[0, 3] = DEPTH_START_ID
labels[0, 4:4+num_depth_tokens_target] = DEPTH_TOKEN_ID # Placeholders
labels[0, 4+num_depth_tokens_target] = DEPTH_END_ID
# Optionally add ignore index somewhere
labels[0, -1] = IGNORE_INDEX
labels[1, 5:8] = IGNORE_INDEX # Ignore some tokens in batch 2

print("--- Input Labels ---")
print(labels)

# Depth Targets (random, only needed if depth markers exist)
# Shape: [batch_size, num_depth_tokens_target, depth_embedding_dim]
# Targets only for the first batch item in this example
depth_targets = torch.randn(batch_size, num_depth_tokens_target, depth_embedding_dim)
# If a batch item has no depth sequence, its target doesn't matter for MSE calculation
# but it needs to be provided with the correct shape if the function expects a batch.
# Alternatively, pass None and handle it inside the function. Let's try passing the full tensor.

print(f"\n--- Running combinedLoss_debug ---")
total_loss, loss_ce, loss_mse = combinedLoss_debug(
    logits=logits,
    labels=labels,
    hidden_states=hidden_states,
    depth_targets=depth_targets, # Provide targets for batch 0
    img_head=mock_img_head,
    vocab_size=vocab_size,
    depth_start_id=DEPTH_START_ID,
    depth_end_id=DEPTH_END_ID,
    ignore_index=IGNORE_INDEX,
    lambda_depth=1.0,
    num_depth_tokens=num_depth_tokens_target # Inform function about expected target length
)

print(f"\n--- Final Losses ---")
print(f"Total: {total_loss.item() if total_loss is not None else 'N/A'}")
print(f"CE:    {loss_ce.item() if loss_ce is not None else 'N/A'}")
print(f"MSE:   {loss_mse.item() if loss_mse is not None else 'N/A'}")

# print("\n--- Testing with No Depth Targets ---")
# total_loss_no_dt, loss_ce_no_dt, loss_mse_no_dt = combinedLoss_debug(
#     logits=logits,
#     labels=labels,
#     hidden_states=hidden_states,
#     depth_targets=None, # Test case with None targets
#     img_head=mock_img_head,
#     vocab_size=vocab_size
# )
# print(f"\n--- Final Losses (No Depth Targets) ---")
# print(f"Total: {total_loss_no_dt.item() if total_loss_no_dt is not None else 'N/A'}")
# print(f"CE:    {loss_ce_no_dt.item() if loss_ce_no_dt is not None else 'N/A'}")
# print(f"MSE:   {loss_mse_no_dt.item() if loss_mse_no_dt is not None else 'N/A'}")


# print("\n--- Testing with Label Mismatch ---")
# # Make labels have fewer depth tokens than target expects
# labels_mismatch = labels.clone()
# labels_mismatch[0, 4+num_depth_tokens_target-1] = 99 # Change one placeholder token
# labels_mismatch[0, 4+num_depth_tokens_target] = DEPTH_END_ID # Shift end marker

# print("Input Labels (Mismatch):")
# print(labels_mismatch)

# total_loss_mm, loss_ce_mm, loss_mse_mm = combinedLoss_debug(
#     logits=logits,
#     labels=labels_mismatch, # Use mismatch labels
#     hidden_states=hidden_states,
#     depth_targets=depth_targets, # Provide targets
#     img_head=mock_img_head,
#     vocab_size=vocab_size,
#     num_depth_tokens=num_depth_tokens_target # Still expecting 5 in targets
# )
# print(f"\n--- Final Losses (Label Mismatch) ---")
# print(f"Total: {total_loss_mm.item() if total_loss_mm is not None else 'N/A'}")
# print(f"CE:    {loss_ce_mm.item() if loss_ce_mm is not None else 'N/A'}")
# print(f"MSE:   {loss_mse_mm.item() if loss_mse_mm is not None else 'N/A'} (Should be 0 or skipped due to mismatch)")