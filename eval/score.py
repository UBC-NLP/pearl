import openai
import json
import pandas as pd
from argparse import ArgumentParser
import os
from dotenv import load_dotenv
import time
import base64
from openai import OpenAI
from io import BytesIO
import asyncio
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from tqdm.asyncio import tqdm_asyncio
import logging
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# Get the model ID
def get_model_id():
    try:
        models = client.models.list()
        return models.data[0].id
    except Exception as e:
        logger.error(f"Error getting model ID: {e}")
        raise

async def score_fun_mllm_async(prompt, semaphore, model_id, image_path):
    """Async function to score prompts with images"""
    async with semaphore:
        try:
            # Convert the Hugging Face image (PIL or array) to Base64
            image = Image.open(image_path)
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # Execute API call in a separate thread to not block the event loop
            with ThreadPoolExecutor() as executor:
                response = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    lambda: client.chat.completions.create(
                        model=model_id,
                        messages=[
                            {"role": "user", "content": [
                                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}},
                                {"type": "text", "text": prompt}
                            ]}
                        ],
                    )
                )

            # Extract the score text from the response
            text = response.choices[0].message.content.strip()
            return text
        except Exception as e:
            logger.error(f"Error in model call with image: {e}")
            return 0

async def score_fn_async(prompt, semaphore, model_id):
    """Async function to score text-only prompts"""
    async with semaphore:
        try:
            # Execute API call in a separate thread to not block the event loop
            with ThreadPoolExecutor() as executor:
                response = await asyncio.get_event_loop().run_in_executor(
                    executor,
                    lambda: client.chat.completions.create(
                        model=model_id,
                        messages=[
                            {"role": "user", "content": [
                                {"type": "text", "text": prompt}
                            ]}
                        ],
                    )
                )

            # Extract the score text from the response
            text = response.choices[0].message.content.strip()
            return text
        except Exception as e:
            logger.error(f"Error in model call: {e}")
            return 0

def prepare_score_prompt(question_type: str,
                         question: str,
                         ground_truth: str,
                         predicted_answer: str,
                         image_description:str):
    """
    Build evaluation prompts for the GPT-judge.
    • Closed-form (MCQ / T-F / short factual)  →  relaxed 0/1 score
    • Open-generation                           →  JudgeScore (0-10)  &  CAS (0/1)
    """
    if question_type == "multiple_choice":
        # expected: ground_truth == correct option text *or* option letter
        score_prompt = f"""
                        You are an impartial evaluator.
                        
                        TASK: Decide if the candidate's choice is correct **ignoring surface form** (letter, synonym,
                        capitalisation).  Return ONLY "1" for correct, "0" for incorrect.
                        
                        Question (with options):
                        {question}
                        
                        Gold correct answer:
                        {ground_truth}
                        
                        Candidate's chosen answer:
                        {predicted_answer}
                        
                        Reply with 1 or 0 – nothing else."""
        return score_prompt.strip()
    elif question_type == "true_false":
        score_prompt = f"""
                        You are an impartial evaluator.
                        
                        TASK: Decide if the candidate's TRUE/FALSE judgment matches the gold label.
                        Return ONLY "1" for match, "0" for mismatch.
                        
                        Statement:
                        {question}
                        
                        Gold label (True/False):
                        {ground_truth}
                        
                        Candidate label:
                        {predicted_answer}
                        
                        Reply with 1 or 0 – nothing else."""
        return score_prompt.strip()
    elif question_type == "short_answer":
        # short factual Q-A where synonyms are allowed
        score_prompt = f"""
                        You are an impartial evaluator.
                        
                        TASK: Compare the candidate's short answer with the gold answer.
                        If they express the **same fact** (allowing synonyms, paraphrase, spelling variants) return "1".
                        Otherwise return "0".
                        Give no explanation.
                        
                        Question:
                        {question}
                        
                        Gold answer:
                        {ground_truth}
                        
                        Candidate answer:
                        {predicted_answer}
                        
                        Reply with 1 or 0 – nothing else."""
        return score_prompt.strip()
    else:
        # -------- OPEN‑GENERATION QUESTIONS --------
        # ========= Judge‑score prompt =========
        judge_score_prompt = f"""
        You are an **impartial multimodal evaluator** for Arabic cultural benchmark.  
        Your task is to grade the candidate’s Arabic answer to a culture‑focused question,
        using **both** the reference text and the attached image.
        
        ***Output ONE JSON object only*** in this schema:
        {{
          "correctness": <0‑5>,
          "coherence":  <0‑5>,
          "detail":     <0‑5>,
          "fluency":    <0‑5>
        }}
        
        ### Scoring rubric (0 = very poor, 5 = excellent)
        • **Correctness 40 %** – factual agreement with the gold reference *and* visual evidence.  
        • **Coherence 20 %**  – logical, no contradictions, clear progression.  
        • **Detail 20 %**     – culturally specific names, dates, artefacts, dialect terms (text or image).  
        • **Fluency 20 %**    – smooth Arabic (or requested dialect), correct grammar & style.
        
        
        *Image description:* {image_description}
        
        ### Textual inputs  
        Question: {question}
        
        Gold reference answer:
        {ground_truth}
        
        Candidate answer:
        {predicted_answer}
        
        Respond with the JSON object only — **no additional text**.
        """.strip()
    
        # ========= Cultural‑Awareness (CAS) prompt =========
        cas_prompt = f"""
        You are checking **Cultural Awareness** for Arabic cultural benchmark item.
        
        Return **"1"** if the candidate answer explicitly mentions at least one culturally specific
        element (e.g., festival, landmark, dialect term) that is central to the gold reference
        **or clearly visible/implied in the image**.  
        Return **"0"** otherwise.
        
        
        *Image description:* {image_description}
        
        ### Textual inputs  
        Gold reference:
        {ground_truth}
        
        Candidate answer:
        {predicted_answer}
        
        Reply with **1** or **0** — nothing else.
        """.strip()
        

        return judge_score_prompt.strip(), cas_prompt.strip()
def format_question(question, question_type, choices=None):
    """Create a prompt based on the question type and language."""
    if question_type == "multiple_choice" and choices:
        choices_text = "\n".join(choices)
        prompt_text = (
            "For the given Multiple Choice Question, analyze the question and answer strictly "
            "from one of the options below. Strictly answer the choice only. No additional text.\n"
            f"{question}\n{choices_text}"
        )
    elif question_type == "true_false":
        prompt_text = (
            f"{question}\nThe above question is a True/False question. "
            "Please provide the answer as one word (True or False)"
        )
    elif question_type == "long_answer":
        prompt_text = f"{question}\nAnswer the question in detail in Arabic language."
    else:
        prompt_text = f"{question}\nPlease provide brief, clear responses in Arabic language."
    return prompt_text
async def process_row(idx, row, semaphore, model_id, output_file, error_file):
    """Process a single row from the dataframe asynchronously and write results to file"""
    try:
        # Create a copy of the row data to store results
        result_row = row.copy()

        caption = row["augmented_caption"]
        image_path = row["image_path"]
        qtype = row["question_type"]
        question = row["question"]
        choices = row["choices"]

        question = format_question(question,qtype,choices)       
        ground_truth = row["answer"]
        if qtype == "multiple_choice":  # prepend letter for exact match
            ground_truth = f"{row['answer_letter']}. {ground_truth}"
        pred_answer = row["model_output"]

        prompt_pack = prepare_score_prompt(qtype, question, ground_truth, pred_answer, caption)

        # ---------------- closed-form (single prompt) ----------------
        if not isinstance(prompt_pack, tuple):
            raw = await score_fn_async(prompt_pack, semaphore, model_id)
            raw_stripped = ''.join(c for c in raw if c in '01')
            result_row["score"] = int(raw_stripped) if raw_stripped else 0
            result_row["cas"] = pd.NA
            result_row["correctness"] = pd.NA
            result_row["coherence"] = pd.NA
            result_row["detail"] = pd.NA
            result_row["fluency"] = pd.NA

        # ---------------- open-generation (two prompts) --------------
        else:
            judge_prompt, cas_prompt = prompt_pack

            judge_reply = await score_fun_mllm_async(judge_prompt, semaphore, model_id, image_path)
            # reply is JSON like {"accuracy":4,"reasoning":3,...}
            try:
                judge_reply = (judge_reply.replace("```json", "").replace("```", "")).strip()
                scores = json.loads(judge_reply)
            except json.JSONDecodeError:
                logger.error(f"Error parsing JSON from judge reply: {judge_reply}")
                scores = {"correctness": -1, "coherence": -1, "detail": -1, "fluency": -1}

            result_row["correctness"] = scores.get("correctness", -1)
            result_row["coherence"] = scores.get("coherence",-1)
            result_row["detail"] = scores.get("detail", -1)
            result_row["fluency"] = scores.get("fluency", -1)
            # result_row["score"] = sum([
            #     scores.get("correctness", 0) * 0.4,
            #     scores.get("coherence", 0) * 0.2,
            #     scores.get("detail", 0) * 0.2,
            #     scores.get("fluency", 0) * 0.2
            # ]) * 2  # Scale to 0-10

            cas_reply = await score_fun_mllm_async(cas_prompt, semaphore, model_id, image_path)
            cas_reply = (cas_reply.replace("```json", "").replace("```", "")).strip()
            cas_reply_stripped = ''.join(c for c in cas_reply if c in '01')
            result_row["cas"] = int(cas_reply_stripped) if cas_reply_stripped else 0
         

        # Write the result to the output file immediately
        async with output_file["lock"]:
            with open(output_file["path"], 'a') as f:
                f.write(json.dumps(result_row.to_dict()) + '\n')
                
        logger.info(f"Processed row {idx} successfully")
        return {"idx": idx, "status": "success", "data": result_row}
        
    except Exception as e:
        error_info = {
            "idx": idx,
            "error": str(e),
            "row_data": row.to_dict()
        }
        
        # Write error information to error file
        async with error_file["lock"]:
            with open(error_file["path"], 'a') as f:
                f.write(json.dumps(error_info) + '\n')
                
        logger.error(f"Error processing row {idx}: {e}")
        return {"idx": idx, "status": "error", "error": str(e)}

async def evaluate_with_gpt_judge_async(
    model_result_df: pd.DataFrame, 
    model_id, 
    output_file_path, 
    error_file_path,
    max_concurrent=5
):
    """
    Asynchronously evaluate model predictions using GPT judge and write results to file
    
    Parameters
    ----------
    model_result_df : DataFrame
        Must contain columns
        ['question_type', 'question', 'answer', 'answer_letter',
         'predicted_answer', ...]
    model_id : str
        Model ID to use for evaluation
    output_file_path : str
        Path to the output file to write results
    error_file_path : str
        Path to the error file to log failures
    max_concurrent : int
        Maximum number of concurrent API calls
        
    Returns
    -------
    tuple
        (success_count, error_count) - counts of successful and failed evaluations
    """
    # Create or clear the output and error files
    with open(output_file_path, 'w') as f:
        pass  # Just create/clear the file
    
    with open(error_file_path, 'w') as f:
        pass  # Just create/clear the file
    
    logger.info(f"Output will be written to: {output_file_path}")
    logger.info(f"Errors will be logged to: {error_file_path}")
    
    # Create file locks for thread-safe writing
    output_file = {
        "path": output_file_path,
        "lock": asyncio.Lock()
    }
    
    error_file = {
        "path": error_file_path,
        "lock": asyncio.Lock()
    }
    
    # Create a semaphore to limit concurrent API calls
    semaphore = asyncio.Semaphore(max_concurrent)
    
    # Create tasks for all rows
    tasks = [
        process_row(idx, row, semaphore, model_id, output_file, error_file) 
        for idx, row in model_result_df.iterrows()
    ]
    
    # Execute all tasks concurrently with progress bar
    results = await tqdm_asyncio.gather(*tasks, desc="Evaluating")
    
    # Count successes and errors
    success_count = sum(1 for r in results if r["status"] == "success")
    error_count = sum(1 for r in results if r["status"] == "error")
    
    logger.info(f"Evaluation complete. Successes: {success_count}, Errors: {error_count}")
    
    return success_count, error_count

async def main_async(file_name, max_concurrent=5):
    """Main async function to process a model result file"""
    try:
        # Get model ID
        model_id = get_model_id()
        logger.info(f"Using model: {model_id}")
        
        # Get output file names
        base_name = os.path.splitext(file_name)[0]
        ext = os.path.splitext(file_name)[1]
        
        output_file = f"{base_name}_scored{ext if ext else '.jsonl'}"
        error_file = f"{base_name}_errors.jsonl"
            
        logger.info(f"Processing file: {file_name}")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Read input file
        model_result = pd.read_json(file_name, lines=True, orient='records')
        logger.info(f"Loaded {len(model_result)} records from {file_name}")
        
        # Evaluate with GPT judge and write results as they're processed
        success_count, error_count = await evaluate_with_gpt_judge_async(
            model_result, 
            model_id, 
            output_file, 
            error_file,
            max_concurrent
        )
        
        # Report results
        total_count = len(model_result)
        logger.info(f"Evaluation complete:")
        logger.info(f"  - Total records: {total_count}")
        logger.info(f"  - Successfully processed: {success_count} ({success_count/total_count*100:.1f}%)")
        logger.info(f"  - Errors: {error_count} ({error_count/total_count*100:.1f}%)")
        logger.info(f"  - Results saved to: {output_file}")
        logger.info(f"  - Errors logged to: {error_file}")
        
        return success_count, error_count
    except Exception as e:
        logger.error(f"Error in main_async: {e}")
        raise

def main():
    """Command line entry point"""
    parser = ArgumentParser(description="Evaluate model predictions with GPT judge")
    parser.add_argument("file_name", help="Path to the input JSONL file with model predictions")
    parser.add_argument("--max_concurrent", type=int, default=5, 
                        help="Maximum number of concurrent API calls")
    parser.add_argument("--retry", action="store_true", 
                        help="Process only the failed entries from a previous run")
    args = parser.parse_args()
    
    try:
        # Run the async main function
        start_time = time.time()
        asyncio.run(main_async(args.file_name, args.max_concurrent))
        end_time = time.time()
        
        logger.info(f"Total execution time: {end_time - start_time:.2f} seconds")
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user. Partial results have been saved.")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    import sys
    main()