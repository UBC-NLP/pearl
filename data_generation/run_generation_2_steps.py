import pandas as pd
import os
from IPython.display import display, HTML, Markdown
from PIL import Image

import base64
import requests
from openai import OpenAI

import json
import re
import numpy as np

from prompts.all_prompts_from_article import get_prompt, QUESTION_TYPES
from prompts.quality_rating import qr_prompt


import asyncio
import json
import aiofiles
import traceback
from concurrent.futures import ThreadPoolExecutor
from tqdm.asyncio import tqdm_asyncio

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id


# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def encode_image_base64_from_path(image_path: str) -> str:
    """Encode an image from a local file path to base64 format."""
    
    with open(image_path, 'rb') as image_file:
        result = base64.b64encode(image_file.read()).decode('utf-8')
    
    return result

def prompt_vlm(image_path, prompt):
    image_base64 = encode_image_base64_from_path(image_path=image_path)
    
    messages = [{
            "role":
            "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{prompt}"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{image_base64}"
                    },
                }
            ],
            
        }]
    
    chat_completion_from_base64 = client.chat.completions.create(
        messages=messages,
        model=model,
        max_tokens=2048,
        stop=["<|im_end|>"],
        temperature=0
    )
    result = chat_completion_from_base64.choices[0].message.content
    messages.append({"role": "assistant", "content": result})
    return result, messages

import re
import json
def parse_json_string(text):
    # First, check if the text is wrapped in code blocks
    code_block_pattern = r'(?:\s*(\w+)\s*)?\n([\s\S]*?)\n'
    matches = re.finditer(code_block_pattern, text)

    for match in matches:
        block_language = match.group(1)
        block_content = match.group(2)

        # If it's a JSON code block, parse it
        if block_language and block_language.lower() == 'json':
            try:
                return json.loads(block_content)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON from code block: {e}")
                # Continue to try parsing as raw JSON

    # If no valid code blocks found or parsing failed, try parsing the text directly as JSON
    # First, try parsing the entire text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # If that fails, try to find JSON-like patterns (text enclosed in curly braces)
        json_pattern = r'(\{[\s\S]*\})'
        json_matches = re.search(json_pattern, text)

        if json_matches:
            potential_json = json_matches.group(1)
            try:
                return json.loads(potential_json)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON pattern: {e}")

    # If all parsing attempts fail
    print("No valid JSON found in the text")
    return None

import json
import re
from transformers import AutoTokenizer

def parse_markdown_json(python_string):
    """
    Parse a JSON string that is enclosed in markdown code block with Arabic content.
    
    Args:
        python_string (str): A string containing JSON wrapped in markdown code block
        
    Returns:
        dict: The parsed JSON data as a Python dictionary
    """
    # Remove outer quotes if present
    if python_string.startswith("'") and python_string.endswith("'"):
        python_string = python_string[1:-1]
    elif python_string.startswith('"') and python_string.endswith('"'):
        python_string = python_string[1:-1]
    
    # Replace escaped newlines with actual newlines
    python_string = python_string.replace('\\n', '\n')
    
    # Extract the JSON content from the markdown code block
    match = re.search(r'```json\n([\s\S]*?)\n```', python_string)
    
    if match:
        json_content = match.group(1)
    else:
        json_content = python_string
    
    # Manually fix the problematic parts of the JSON
    # Process the JSON line by line to handle the Arabic text properly
    lines = json_content.split('\n')
    fixed_lines = []
    
    in_answer = False
    current_answer = ""
    
    for line in lines:
        # Check if this line starts an answer field
        if '"answer":' in line and not line.strip().endswith('"') and not line.strip().endswith('",'):
            in_answer = True
            # Get the part before "answer":
            parts = line.split('"answer":', 1)
            prefix = parts[0] + '"answer": "'
            # Start collecting the answer text
            current_answer = parts[1].strip()
            # Don't add this line yet
        
        # Check if this line ends an answer field
        elif in_answer and (line.strip().endswith(',') or line.strip().endswith('}')):
            # End of the answer
            in_answer = False
            # Remove trailing comma or bracket
            if line.strip().endswith(','):
                line_content = line.rstrip(',').strip()
                ending = '",'
            else:
                line_content = line.rstrip('}').strip()
                ending = '"}'
            
            # Add the complete answer
            current_answer += " " + line_content
            fixed_lines.append(prefix + current_answer + ending)
            current_answer = ""
        
        # Continuation of an answer field
        elif in_answer:
            current_answer += " " + line.strip()
        
        # Regular line, add as-is
        else:
            fixed_lines.append(line)
    
    fixed_content = '\n'.join(fixed_lines)
    
    # Try parsing the fixed JSON
    try:
        return json.loads(fixed_content)
    except json.JSONDecodeError as e:
        # If still having issues, we'll use a more aggressive approach
        # Find the problematic unquoted Arabic sections
        pattern = r'"answer":\s*([^"][^,}]*?)(\s*[,}])'
        
        # Function to escape and add quotes around the matched value
        def add_quotes(match):
            value = match.group(1).strip()
            ending = match.group(2)
            # Escape any quote characters in the text
            value = value.replace('"', '\\"')
            return f'"answer": "{value}"{ending}'
        
        # Replace all instances of unquoted values
        doubly_fixed_content = re.sub(pattern, add_quotes, fixed_content)
        
        try:
            return json.loads(doubly_fixed_content)
        except json.JSONDecodeError as e:
            print(f"JSON parsing still failed: {e}")
            # Last resort - manual parsing to extract the key fields
            result = {}
            try:
                # Extract augmented_caption
                caption_match = re.search(r'"augmented_caption":\s*"([^"]*)"', json_content)
                if caption_match:
                    result["augmented_caption"] = caption_match.group(1)
                
                # Try to extract the QA pairs manually
                result["generated_QAs"] = []
                
                # Find all question blocks
                question_blocks = re.finditer(r'"question":\s*"([^"]*)"', json_content)
                
                for q_match in question_blocks:
                    question = q_match.group(1)
                    # Find answer text that follows this question
                    pos = q_match.end()
                    answer_start = json_content.find('"answer":', pos)
                    if answer_start > 0:
                        answer_end = json_content.find('",', answer_start)
                        if answer_end < 0:
                            answer_end = json_content.find('"}', answer_start)
                        
                        if answer_end > 0:
                            answer_text = json_content[answer_start+9:answer_end].strip()
                            # Add this QA pair
                            result["generated_QAs"].append({
                                "question": question,
                                "answer": answer_text
                            })
                
                return result
            except Exception as ex:
                print(f"Manual parsing failed too: {ex}")
                raise


def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


def get_data_sample(sample_idx, display_meta = False):
    caption = df.iloc[sample_idx]['caption'] if not pd.isna(df.iloc[sample_idx]['caption']) else "N/A"
    country = df.iloc[sample_idx]['country']
    category = df.iloc[sample_idx]['type']
    title = df.iloc[sample_idx]['title']
    image_path = df.iloc[sample_idx]['image']
    article = df.iloc[sample_idx]['article']
    index = int(df.iloc[sample_idx]['id'])  # Convert NumPy int64 to Python int

    if display_meta:
        img = Image.open(df.iloc[sample_idx]['image'])
        display(img)
        print(df.iloc[sample_idx]['title'])
    return caption, country, category, title, image_path, article, index

# Create a semaphore to limit concurrent API calls
MAX_CONCURRENT_CALLS = 50
semaphore = asyncio.Semaphore(MAX_CONCURRENT_CALLS)

# Create a thread pool for CPU-bound operations
thread_pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_CALLS)

# Create a lock for file access
file_lock = asyncio.Lock()

# Correctly define the async wrapper for prompt_vlm
async def prompt_vlm_async(image_path, prompt):
    """Async wrapper for the synchronous prompt_vlm function"""
    def _run_prompt():
        return prompt_vlm(image_path, prompt)
    
    # Run the synchronous function in a thread and return its result directly
    return await asyncio.get_event_loop().run_in_executor(thread_pool, _run_prompt)

async def process_single_item(i):
    """Process a single dataframe item and return the results"""
    try:
        caption, country, category, title, image_path, article, index = get_data_sample(i, display_meta=False)
        
        metadata = {
            "category": category,
            "title": title,
            "country": country,
            "image_caption": caption,
            'wikipedia_article': article
        }
        # Process all question types in parallel
        async def process_question_type(question_type):
            try:
                qa_prompt = get_prompt(question_type, metadata, num_pairs=2)
                
                async with semaphore:
                    QAs, messages = await prompt_vlm_async(image_path, qa_prompt)
                    
                # # Try to parse the response as JSON
                # parsed_QAs = parse_markdown_json(QAs)
                # if not parsed_QAs:
                #     parsed_QAs = parse_json_string(QAs)
                
                # # Add question_type to each item
                # for item in parsed_QAs['generated_QAs']:
                #     item["question_type"] = question_type
                
                return QAs
            except Exception as e:
                error_msg = f"Error processing question type {question_type}: {str(e)}"
                print(error_msg)
                return []
        
        # Run all question types in parallel
        question_tasks = [process_question_type(qt) for qt in list(QUESTION_TYPES.keys())]
        question_results = await asyncio.gather(*question_tasks)
        
        # Flatten the results
        #generated_QAs = [item for sublist in question_results for item in sublist]
        
        # Check if we got any QAs
        if not question_results:
            return {
                'index': index,
                'processed': False,
                'error': "No QA pairs were generated"
            }
        
        export_data = {
            'index': index,
            'processed': True,
            'generated_QAs': question_results
        }
        
        return export_data
        
    except Exception as e:
        error_message = str(e)
        error_type = type(e).__name__
        traceback_str = traceback.format_exc()
        
        print(f"Error processing item {i}:")
        traceback.print_exc()
        
        # Return error information
        return {
            'index': index if 'index' in locals() else i,
            'processed': False,
            'error': error_type,
            'error_message': error_message,
            'traceback': traceback_str
        }

async def write_to_file(data, output_file):
    """Write a single result to the output file and flush immediately"""
    if data is None:
        return
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Use a lock to prevent multiple processes from writing simultaneously
    async with file_lock:    
        async with aiofiles.open(output_file, mode='a', encoding="utf-8") as f:
            # Use the custom encoder to handle NumPy types
            json_string = json.dumps(data, ensure_ascii=False, cls=NumpyEncoder)
            await f.write(json_string + '\n')
            # Explicitly flush to disk after each write
            await f.flush()
            
            # For extra certainty on some systems, you can use fsync
            # This ensures data is written to the physical disk
            if hasattr(f, 'fileno'):
                try:
                    os.fsync(f.fileno())
                except:
                    # Some file-like objects might not support fsync even if they have fileno
                    pass

async def process_and_write(i, output_file):
    """Process an item and immediately write it to the file"""
    result = await process_single_item(i)
    if result:
        await write_to_file(result, output_file)
    return result

async def main():
    output_file = 'output/output_data_qwen72VL_2steps_V2_batch_2_11.jsonl'
    batch_size = 20  # Process this many items at once
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Create tasks for all items
    total_items = len(df)
    
    print(f"Processing {total_items} items in batches of {batch_size}...")
    
    # Process in batches to avoid memory issues
    for start_idx in range(0, total_items, batch_size):
        end_idx = min(start_idx + batch_size, total_items)
        current_batch = list(range(start_idx, end_idx))
        
        print(f"Processing batch {start_idx//batch_size + 1}: items {start_idx} to {end_idx-1}")
        
        # Process the current batch in parallel and write results immediately
        tasks = [process_and_write(i, output_file) for i in current_batch]
        
        # Use a try-except block to handle errors during processing
        try:
            results = await tqdm_asyncio.gather(*tasks, desc="Processing items")
            
            # Report batch progress
            success_count = sum(1 for r in results if r and r.get('processed', False))
            error_count = sum(1 for r in results if r and not r.get('processed', False))
            missing_count = sum(1 for r in results if r is None)
            
            print(f"Batch completed: {success_count} successful, {error_count} failed, {missing_count} missing")
        except Exception as e:
            print(f"Error in batch processing: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    # Load the dataset

    input_file = 'wikipedia_data_w_images_filtered.jsonl'
    selected_cols = ['id', 'title', 'article', 'country', 'type', 'image', 'caption']
    
    df = pd.read_json(input_file, lines=True, orient='records')
    
    # Run the main async function
    asyncio.run(main())
   