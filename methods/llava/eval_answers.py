# NEW: Aurora-only file; not present in upstream LLAVA.
# NEW: Baseline https://github.com/haotian-liu/LLaVA.git @ v1.2.2.post1 (24fa1d065bbeac8a145a796ab7218c6945a2536e).
# NEW: Aurora path: eval_answers.py

import json
import os
import argparse
from tqdm import tqdm
import numpy as np
from collections import defaultdict

def load_answers(file_path):
    """Load answers from a JSONL file into a dictionary keyed by question_id."""
    answers = {}
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            answers[data['question_id']] = data
    return answers

def evaluate_answers(ground_truth, model_answers, verbose=False, enable_point_patterns=False):
    """Evaluate model answers against ground truth."""
    metrics = {
        'total': 0,
        'correct': 0,
        'missing': 0,
        'extra': 0,
        'total_loss': 0.0,
        'total_mse_loss': 0.0,
        'loss_count': 0
    }
    
    # Check each ground truth answer
    for q_id, gt_data in ground_truth.items():
        metrics['total'] += 1
        if q_id not in model_answers:
            metrics['missing'] += 1
            continue
            
        model_answer = model_answers[q_id]['text'].strip().lower()
        gt_answer = gt_data['text'].strip().lower()
        
        if verbose:
            print(f"Evaluating Q{q_id}: GT='{gt_answer}' vs Model='{model_answer[:100]}...'")
        
        # Check if loss information is available
        if 'loss' in model_answers[q_id]:
            metrics['total_loss'] += model_answers[q_id]['loss']
            metrics['loss_count'] += 1
            
        if 'mse_loss' in model_answers[q_id]:
            metrics['total_mse_loss'] += model_answers[q_id]['mse_loss']
        
        # Check for answer match - handle both short formatted and long embedded answers
        answer_match = False
        
        import re
        
        # First, extract the ground truth answer character
        gt_char = None
        # Check if GT is formatted like (A), "A", etc.
        # Include both straight quotes and Unicode curly quotes
        gt_patterns = re.findall(r'[\(\[]([a-z])[\)\]]|["\"\u201c\u201d\'\u2018\u2019]([a-z])["\"\u201c\u201d\'\u2018\u2019]', gt_answer)
        gt_chars = [char for group in gt_patterns for char in group if char]
        
        if gt_chars:
            gt_char = gt_chars[0].lower()
        elif len(gt_answer.strip()) == 1 and gt_answer.strip().isalpha():
            # GT is a single character
            gt_char = gt_answer.strip().lower()
        
        if gt_char:
            # Method 1: Check for formatted patterns in model answer (short format)
            # Include both straight quotes and Unicode curly quotes, allow punctuation inside quotes
            model_patterns = re.findall(r'[\(\[]([a-z])[\)\]]|["\"\u201c\u201d\'\u2018\u2019]([a-z])[.,;:!?]*["\"\u201c\u201d\'\u2018\u2019]', model_answer)
            model_chars = [char.lower() for group in model_patterns for char in group if char]
            
            if gt_char in model_chars:
                answer_match = True
                if verbose:
                    print(f"Match found via formatted pattern: GT='{gt_char}' in model_chars={model_chars}")
            else:
                # Method 2: Check for embedded letters in sentences (long format)
                # Look for patterns like "point is at a", "answer is a", "located at a", etc.
                embedded_patterns = [
                    r'the\s+answer\s+is\s+that\s+point\s+([a-z])\s+is\s+closer\s+to\s+the\s+camera\b(?:\.)?',  # "the answer is that point a is closer to the camera."
                ]
                enable_point_patterns = True
                # Add additional patterns if flag is enabled
                if enable_point_patterns:
                    embedded_patterns.extend([
                        r'depth_end.*?point\s+([a-z])\b', # "DEPTH_END ... point a" (check for point after DEPTH_END)
                        r'since\s+point\s+([a-z])\s+has\b', # "since point a has", "since point b has" (more specific)
                        r'point\s+([a-z])\s+has\s+(?:a\s+)?higher', # "point a has higher", "point a has a higher"
                        r'point\s+([a-z])\s+is\s+closer', # "point a is closer"
                        r'point\s+([a-z])\s+appears\s+closer', # "point a appears closer"
                        r'the\s+answer\s+is\s+([a-z])\b', # "the answer is a"
                        r'answer\s+is\s+([a-z])\b',    # "answer is a"
                        r'(?:the\s+)?letter\s+([a-z])\b', # "letter a" or "the letter a" without quotes
                        r'labeled\s+([a-z])\b', # "labeled a" without quotes
                    ])
                
                embedded_chars = []
                for pattern in embedded_patterns:
                    matches = re.findall(pattern, model_answer, re.IGNORECASE)
                    embedded_chars.extend([char.lower() for char in matches])
                
                if gt_char in embedded_chars:
                    answer_match = True
                    if verbose:
                        print(f"Match found via embedded pattern: GT='{gt_char}' in embedded_chars={embedded_chars}")
                else:
                    if verbose:
                        print(f"No match found: GT='{gt_char}', model_chars={model_chars}, embedded_chars={embedded_chars}")
        else:
            if verbose:
                print(f"Could not extract GT character from: '{gt_answer}'")
        
        if answer_match:
            metrics['correct'] += 1
    
    # Count extra answers (answers in model that aren't in ground truth)
    metrics['extra'] = len(set(model_answers.keys()) - set(ground_truth.keys()))
    
    # Calculate accuracy
    if metrics['total'] > 0:
        metrics['accuracy'] = metrics['correct'] / metrics['total']
    else:
        metrics['accuracy'] = 0.0
        
    # Calculate average losses
    if metrics['loss_count'] > 0:
        metrics['avg_loss'] = metrics['total_loss'] / metrics['loss_count']
        metrics['avg_mse_loss'] = metrics['total_mse_loss'] / metrics['loss_count']
    else:
        metrics['avg_loss'] = 0.0
        metrics['avg_mse_loss'] = 0.0
        
    return metrics

def print_metrics(metrics, file_name):
    """Print evaluation metrics in a formatted way."""
    print(f"\nResults for {file_name}:")
    print(f"Total questions: {metrics['total']}")
    print(f"Correct answers: {metrics['correct']}")
    print(f"Missing answers: {metrics['missing']}")
    print(f"Extra answers: {metrics['extra']}")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    if metrics['loss_count'] > 0:
        print(f"Average Loss: {metrics['avg_loss']:.4f}")
        print(f"Average MSE Loss: {metrics['avg_mse_loss']:.4f}")

def evaluate_file(gt_path, model_path, verbose=False, enable_point_patterns=False):
    """Evaluate a single pair of ground truth and model answer files."""
    if not os.path.exists(gt_path):
        print(f"Warning: Ground truth file not found: {gt_path}")
        return None
    if not os.path.exists(model_path):
        print(f"Warning: Model answers file not found: {model_path}")
        return None
        
    ground_truth = load_answers(gt_path)
    model_answers = load_answers(model_path)
    
    return evaluate_answers(ground_truth, model_answers, verbose, enable_point_patterns)

def main():
    parser = argparse.ArgumentParser(description='Evaluate model answers against ground truth.')
    parser.add_argument('--gt_dir', type=str, required=True,
                      help='Directory containing ground truth answer files')
    parser.add_argument('--model_dir', type=str, required=True,
                      help='Directory containing model answer files')
    parser.add_argument('--files', type=str, nargs='+', required=True,
                      help='List of answer files to evaluate')
    parser.add_argument('--gt_files', type=str, nargs='+', default=None,
                      help='List of ground truth filenames (if different from --files)')
    parser.add_argument('--summary_file', type=str, default=None,
                      help='Path to save evaluation summary JSON file (default: model_dir/evaluation_summary.json)')
    parser.add_argument('--verbose', action='store_true',
                      help='Enable verbose debug output for answer matching')
    parser.add_argument('--enable-point-patterns', action='store_true',
                      help='Temporarily enable additional point (A,B) pattern matching')
    
    args = parser.parse_args()
    
    # Use separate GT filenames if provided, otherwise use same as model files
    gt_files = args.gt_files if args.gt_files else args.files
    
    if len(gt_files) != len(args.files):
        raise ValueError(f"Number of GT files ({len(gt_files)}) must match number of model files ({len(args.files)})")
    
    print("Starting evaluation...")
    
    all_metrics = {
        'total': 0,
        'correct': 0,
        'missing': 0,
        'extra': 0,
        'total_loss': 0.0,
        'total_mse_loss': 0.0,
        'loss_count': 0
    }
    
    # Store per-file results for summary
    file_results = {}
    
    for i, model_file in enumerate(args.files):
        gt_file = gt_files[i]
        print(f"\nEvaluating {model_file} against {gt_file}...")
        
        # Load ground truth and model answers
        gt_path = os.path.join(args.gt_dir, gt_file)
        model_path = os.path.join(args.model_dir, model_file)
        
        metrics = evaluate_file(gt_path, model_path, args.verbose, args.enable_point_patterns)
        if metrics is None:
            continue
            
        # Print individual file metrics
        print_metrics(metrics, model_file)
        
        # Store results for summary
        file_results[model_file] = {
            'gt_file': gt_file,
            'total': metrics['total'],
            'correct': metrics['correct'],
            'missing': metrics['missing'],
            'extra': metrics['extra'],
            'accuracy': metrics['accuracy'],
            'avg_loss': metrics['avg_loss'],
            'avg_mse_loss': metrics['avg_mse_loss'],
            'loss_count': metrics['loss_count']
        }
        
        # Accumulate metrics for overall statistics
        for key in ['total', 'correct', 'missing', 'extra', 'total_loss', 'total_mse_loss', 'loss_count']:
            all_metrics[key] += metrics[key]
    
    # Calculate and print overall metrics
    if all_metrics['total'] > 0:
        all_metrics['accuracy'] = all_metrics['correct'] / all_metrics['total']
    else:
        all_metrics['accuracy'] = 0.0
        
    if all_metrics['loss_count'] > 0:
        all_metrics['avg_loss'] = all_metrics['total_loss'] / all_metrics['loss_count']
        all_metrics['avg_mse_loss'] = all_metrics['total_mse_loss'] / all_metrics['loss_count']
    else:
        all_metrics['avg_loss'] = 0.0
        all_metrics['avg_mse_loss'] = 0.0
    
    print("\nOverall Results:")
    print_metrics(all_metrics, "All Files")
    
    # Create evaluation summary
    summary = {
        'model_dir': args.model_dir,
        'overall': {
            'total': all_metrics['total'],
            'correct': all_metrics['correct'],
            'missing': all_metrics['missing'],
            'extra': all_metrics['extra'],
            'accuracy': all_metrics['accuracy'],
            'avg_loss': all_metrics['avg_loss'],
            'avg_mse_loss': all_metrics['avg_mse_loss'],
            'loss_count': all_metrics['loss_count']
        },
        'per_file': file_results,
        'evaluation_config': {
            'gt_dir': args.gt_dir,
            'model_dir': args.model_dir,
            'gt_files': gt_files,
            'model_files': args.files
        }
    }
    
    # Save summary to JSON file
    if args.summary_file:
        summary_path = args.summary_file
    else:
        summary_path = os.path.join(args.model_dir, 'evaluation_summary.json')
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nEvaluation summary saved to: {summary_path}")

if __name__ == "__main__":
    main() 