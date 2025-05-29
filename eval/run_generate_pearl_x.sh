#!/bin/bash

module load CUDA/12.4.0 GCCcore/.12.2.0 CMake/3.24.3 Python/3.10.8
source ~/env_vllm/bin/activate

# Run VLLM Vision Predictor script for multiple models
cd /pearl/eval
export HF_TOKEN="hf_"
# Define common parameters
DATASET_PATH="UBC-NLP/PEARL-X"
SPLIT="test"
CACHE_DIR="cache"
BATCH_SIZE=16
TENSOR_PARALLEL_SIZE=4
MAX_TOKENS=4048
MAX_MODEL_LEN=30000
TEMPERATURE=0.0
REPETITION_PENALTY=1.1
TOP_P=0.9
GPU_MEMORY_UTILIZATION=0.90

# Create results directory if it doesn't exist
mkdir -p results

# Define all models to evaluate
declare -A MODELS

MODELS[0]="Qwen/Qwen2.5-VL-7B-Instruct|Qwen2.5-VL-7B-Instruct"
MODELS[1]="CohereForAI/aya-vision-32b|aya-vision-32b"
MODELS[2]="google/gemma-3-4b-it|gemma-3-4b-it"
MODELS[3]="google/gemma-3-27b-it|gemma-3-27b-it"
MODELS[1]="Qwen/Qwen2.5-VL-3B-Instruct|Qwen2.5-VL-3B-Instruct"
MODELS[5]="CohereForAI/aya-vision-8b|aya-vision-8b"
MODELS[2]="Qwen/Qwen2.5-VL-32B-Instruct|Qwen2.5-VL-32B-Instruct"
MODELS[7]="google/gemma-3-12b-it|gemma-3-12b-it"
MODELS[8]="Qwen/Qwen2.5-VL-72B-Instruct|Qwen2.5-VL-72B-Instruct"

# Run evaluation for each model
for model_info in "${MODELS[@]}"; do
        # Split the model_info into MODEL_PATH and MODEL_NAME
        IFS="|" read -r MODEL_PATH MODEL_NAME <<< "$model_info"
        
        # Define the output files with timestamp
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        OUTPUT_FILE="results/output_${MODEL_NAME}_${TIMESTAMP}.jsonl"
        ERROR_FILE="results/errors_${MODEL_NAME}_${TIMESTAMP}.txt"
        
        # Print parameters being used
        echo "=========================================================="
        echo "Starting evaluation for $MODEL_NAME at $(date)"
        echo "Dataset: $DATASET_PATH"
        echo "Model: $MODEL_PATH ($MODEL_NAME)"
        echo "Batch size: $BATCH_SIZE"
        echo "Results will be saved to $OUTPUT_FILE"
        
        # Run the Python script with all parameters
        CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="0,1,2,4" python generate_pearl_x.py \
                --dataset_path "$DATASET_PATH" \
                --split "$SPLIT" \
                --cache_dir "$CACHE_DIR" \
                --model_path "$MODEL_PATH" \
                --model_name "$MODEL_NAME" \
                --output_file "$OUTPUT_FILE" \
                --error_file "$ERROR_FILE" \
                --batch_size "$BATCH_SIZE" \
                --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
                --max_tokens "$MAX_TOKENS" \
                --temperature "$TEMPERATURE" \
                --repetition_penalty "$REPETITION_PENALTY" \
                --max_model_len "$MAX_MODEL_LEN" \
                --top_p "$TOP_P" \
                --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION"
        
        # Check if the script completed successfully
        if [ $? -eq 0 ]; then
                echo "Evaluation data generation completed successfully for $MODEL_NAME at $(date)"
                echo "Results saved to $OUTPUT_FILE"
        else
                echo "Evaluation failed for $MODEL_NAME at $(date)"
        fi
        echo "=========================================================="
        echo ""
done

echo "All evaluations completed at $(date)"



