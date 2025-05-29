import os
import json
import time
import base64
import logging
import traceback
import asyncio
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
from datasets import load_dataset
from vllm import LLM, SamplingParams

# hf login
from huggingface_hub import login
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    # Will prompt for token if not logged in already
    login()

# Set up ArgumentParser
def parse_args():
    parser = ArgumentParser(description="Run batch inference with vLLM and save results to JSONL")
    parser.add_argument("--dataset_path", type=str, default="/project/afakhrad/tybih/eval/pearl_annotated.py",
                        help="Path to the dataset")
    parser.add_argument("--split", type=str, default="test", 
                        help="Dataset split to use")
    parser.add_argument("--cache_dir", type=str, default="cache",
                        help="Cache directory for datasets")
    parser.add_argument("--model_path", type=str, default="/project/afakhrad/models/CohereForAI/aya-vision-8b",
                        help="Path to the model")
    parser.add_argument("--model_name", type=str, default="",
                        help="Optional friendly name for the model (defaults to last part of model path if not specified)")
    parser.add_argument("--output_file", type=str, default="results.jsonl",
                        help="Path to output JSONL file")
    parser.add_argument("--error_file", type=str, default="errors.txt",
                        help="Path to error log file")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for inference")
    parser.add_argument("--tensor_parallel_size", type=int, default=2,
                        help="Tensor parallel size for vLLM")
    parser.add_argument("--max_tokens", type=int, default=2048,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--max_model_len", type=int, default=16384,
                        help="Maximum model length")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter")
    parser.add_argument("--repetition_penalty", type=float, default=1.1,
                        help="Repetition penalty for sampling")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7,
                        help="GPU memory utilization for vLLM")
    parser.add_argument("--max_image_size", type=int, default=980,
                        help="Maximum image size for the model")
    return parser.parse_args()

def get_content_for_multiple_images(question, question_type, choices=None, available_images=None):
    """Create content structure for multiple images with the question."""
    content = []
    
    # Only add placeholders for available images
    if available_images:
        for i, img in enumerate(available_images):
            content.extend([
                {"text": f"Image {i+1}: ", "type": "text"},
                {"text": None, "type": "image"},
                {"text": "\n", "type": "text"}
            ])
    
    # Add the question text
    if question_type == "multiple_choice" and choices:
        choices_text = "\n".join(choices)
        prompt_text = (
            f"{question}\n{choices_text}\n\n"
            "For the given Multiple Choice Question, analyze the question and answer strictly "
            "from one of the options below. Strictly answer the choice only. No additional text."
        )
    else:
        prompt_text = question
        
    content.append({"text": prompt_text, "type": "text"})
    
    return content

def create_prompt(question, question_type, choices=None, images=None):
    """Create a prompt based on the question type and language."""
    # Create content structure for chat template
    content = get_content_for_multiple_images(question, question_type, choices, images)
    
    messages = [
        {
            "role": "user",
            "content": content,
        }
    ]
    
    # This returns token IDs ready for vLLM
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True)

def get_images_from_data(data):
    """Extract all available images from the data."""
    images = []
    for i in range(1, 7):  # For image1 through image6
        img_key = f'image{i}'
        if img_key in data and data[img_key] is not None:
            try:
                # Convert to RGB to ensure compatibility (handles RGBA)
                img = data[img_key]
                if hasattr(img, 'mode') and img.mode == 'RGBA':
                    images.append(img.convert('RGB'))
                elif hasattr(img, 'mode'):  # It's a PIL image
                    images.append(img)
                else:
                    logging.warning(f"Skipping {img_key}: not a valid PIL image")
            except Exception as e:
                logging.warning(f"Error processing {img_key}: {str(e)}")
    return images

def batch_generate(model, tokenizer, requests, sampling_params):
    """
    Send a batch of requests to vLLM and decode their outputs.
    """
    results = []
    try:
        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            batch_outputs = model.generate(requests, sampling_params=sampling_params)
            for out in batch_outputs:
                # take the first output stream
                token_ids = out.outputs[0].token_ids
                text = tokenizer.decode(token_ids, skip_special_tokens=True)
                results.append(text)
    except Exception as e:
        logging.error(f"Error in batch_generate: {str(e)}")
        raise
    return results

def write_to_jsonl(file_path, data):
    """Write data to a JSONL file (appending if file exists)"""
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')

def write_error_to_file(file_path, data):
    """Write error information to a file"""
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(f"{data}\n")

def main():
    # Parse arguments
    args = parse_args()
    
    # Set model name if not specified
    if not args.model_name:
        args.model_name = os.path.basename(args.model_path.rstrip('/'))
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.error_file)), exist_ok=True)
    
    # Clear previous results if files exist
    if os.path.exists(args.output_file):
        open(args.output_file, 'w').close()
    if os.path.exists(args.error_file):
        open(args.error_file, 'w').close()
    
    logging.info(f"Loading dataset from {args.dataset_path}")
    # Load dataset
    dataset = load_dataset(
        args.dataset_path,
        split=args.split,
        cache_dir=args.cache_dir,
        trust_remote_code=True
    )
    
    logging.info(f"Loading model from {args.model_path} (name: {args.model_name})")
    # Load model
    model = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=torch.bfloat16,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True
    )
    
    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
        stop=["<|im_end|>", '<end_of_turn>']
    )
    
    # Load tokenizer
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    
    n = len(dataset)
    logging.info(f"Processing {n} examples with batch size {args.batch_size}")
    
    error_ids = []
    
    for start in tqdm(range(0, n, args.batch_size)):
        end = min(start + args.batch_size, n)
        indices = list(range(start, end))
        
        # Get batch of examples
        batch = dataset.select(indices)
        
        requests = []
        batch_data = []
        
        for idx, data in enumerate(batch):
            absolute_idx = start + idx
            try:
                # Extract all available images
                images = get_images_from_data(data)
                
                if not images:
                    error_msg = f"No images found for example at index {absolute_idx}"
                    logging.error(error_msg)
                    error_ids.append(absolute_idx)
                    write_error_to_file(args.error_file, error_msg)
                    continue
                
                # Prepare prompt for this example with exact images we have
                prompt_ids = create_prompt(
                    data['question'],
                    data['question_type'],
                    data.get('choices'),
                    images
                )
                
                # Create request with multiple images
                requests.append({
                    "prompt_token_ids": prompt_ids,
                    "multi_modal_data": {
                        "image": images,
                    }
                })
                
                # Prepare data for output (excluding images)
                output_data = {k: v for k, v in data.items() if not k.startswith('image')}
                batch_data.append((absolute_idx, output_data))
            except Exception as e:
                error_msg = f"Failed to prepare prompt for example at index {absolute_idx}: {str(e)}"
                logging.error(error_msg)
                error_ids.append(absolute_idx)
                write_error_to_file(args.error_file, error_msg)
        
        # Filter out any invalid requests
        valid_indices = []
        valid_requests = []
        valid_batch_data = []
        
        for i, (abs_idx, data_dict) in enumerate(batch_data):
            if i < len(requests) and requests[i] is not None:
                valid_indices.append(i)
                valid_requests.append(requests[i])
                valid_batch_data.append((abs_idx, data_dict))
        
        if not valid_requests:
            continue
        
        # Run batch inference
        try:
            batch_outputs = batch_generate(model, tokenizer, valid_requests, sampling_params)
            
            # Process and save results
            for (abs_idx, data_dict), output_text in zip(valid_batch_data, batch_outputs):
                # Add model output, model path, and model name to the data
                result = {
                    "id": abs_idx,
                    "model_output": output_text,
                    "model_path": args.model_path,
                    "model_name": args.model_name,
                    **data_dict  # Include all original data except images
                }
                
                # Write this result to JSONL file
                write_to_jsonl(args.output_file, result)
                
        except Exception as e:
            batch_error_msg = f"Error during batch generation at batch starting {start}: {str(e)}"
            logging.error(batch_error_msg)
            traceback.print_exc()
            
            # Mark all examples in this batch as failed
            for abs_idx, _ in valid_batch_data:
                error_ids.append(abs_idx)
                write_error_to_file(args.error_file, f"Batch error for example {abs_idx}")
    
    # Write summary of errors
    if error_ids:
        summary = f"Total errors: {len(error_ids)}, Error indices: {sorted(error_ids)}"
        write_error_to_file(args.error_file, summary)
        logging.info(summary)
    
    logging.info(f"Processing complete. Results written to {args.output_file}")
    logging.info(f"Errors (if any) written to {args.error_file}")

if __name__ == "__main__":
    # Import tokenizer here to avoid circular import
    from transformers import AutoTokenizer
    main()