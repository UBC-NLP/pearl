# PEARL Benchmark Evaluation

This directory contains scripts and tools for evaluating models on the PEARL benchmark. Follow the steps below to run a complete evaluation.

## Overview

The evaluation process consists of 4 main steps:
1. Generate model responses
2. Start the judge model server
3. Score the responses
4. Collect and analyze results

## Prerequisites

- CUDA-enabled GPUs
- Python environment with VLLM installed
- Access to HuggingFace models
- Required modules loaded (CUDA, GCC, CMake, Python)

## Step-by-Step Evaluation Process

### Step 1: Generate Model Responses

Run the generation script to evaluate 9 models on the PEARL benchmark:

```bash
./run_generate_pearl.sh
```

**Alternative:** For a different evaluation setup, you can also use:
```bash
./run_generate_pearl_x.sh
```

This script will:
- Load the PEARL benchmark dataset
- Evaluate multiple models including:
  - Qwen2.5-VL models (3B, 7B, 32B, 72B)
  - Aya-vision models (8B, 32B)
  - Gemma-3 models (4B, 12B, 27B)
- Generate JSONL output files in the `results/` directory
- Each output file is timestamped for tracking

### Step 2: Start the Judge Model Server

Before scoring, start the VLLM server for the judge model (Qwen2.5-VL-32B-Instruct):

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="0,1,2,4" python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-32B-Instruct \
    --tensor-parallel-size 4 \
    --dtype auto \
    --served-model-name qwenvl2_5_32b \
    --max-model-len 30000 \
    --disable-log-requests \
    --port 8000 \
    --trust_remote_code \
    --gpu-memory-utilization 0.95
```

**Important:** Keep this server running during the scoring step. The server will be available at `http://localhost:8000`.

### Step 3: Score the Generated Responses

Run the scoring script to evaluate all generated responses:

```bash
./run_scoring.sh
```

This script will:
- Process each JSONL file generated in Step 1
- Use the judge model to score responses on multiple criteria:
  - Correctness
  - Coherence
  - Detail
  - Fluency
  - CAS (Context-Aware Scoring)
- Generate `*_scored.jsonl` files for each model
- Handle both multiple choice/true-false questions and open-ended questions

### Step 4: Collect and Analyze Results

Run the results collection script to generate a summary:

```bash
python ../collect_results.py
```

This script will:
- Aggregate all scored JSONL files
- Compute an overall score using weighted averages:
  - Correctness: 40%
  - Coherence: 20%
  - Detail: 20%
  - Fluency: 20%
- Calculate accuracy for MCQ/True-False questions
- Generate averages for open-ended questions
- Save results to `summary.csv`

## Output Files

- **Generation Phase:** `results/output_{model_name}_{timestamp}.jsonl`
- **Scoring Phase:** `results/output_{model_name}_{timestamp}_scored.jsonl`
- **Final Summary:** `summary.csv`

## Configuration

### Model Parameters (Generation)
- Batch size: 16
- Tensor parallel size: 4
- Max tokens: 4048
- Max model length: 16384
- Temperature: 0.0
- Top-p: 0.9
- GPU memory utilization: 0.90

### Scoring Parameters
- Max concurrent requests: 50
- Judge model: Qwen2.5-VL-32B-Instruct
- Server port: 8000

## Troubleshooting

1. **CUDA Memory Issues:** Adjust `--gpu-memory-utilization` parameter
2. **Generation Failures:** Check error files in `results/errors_{model_name}_{timestamp}.txt`
3. **Scoring Issues:** Ensure the judge model server is running and accessible
4. **Missing Dependencies:** Verify all required modules are loaded

## File Structure

```
eval/
├── README.md                    # This file
├── generate_pearl.py           # Main generation script
├── generate_pearl_x.py         # Alternative generation script
├── run_generate_pearl.sh       # Generation runner script
├── run_generate_pearl_x.sh     # Alternative generation runner
├── run_scoring.sh              # Scoring runner script
├── score.py                    # Scoring script
├── results/                    # Generated outputs
└── cache/                      # Model cache directory
```

## Notes

- The evaluation process can take several hours depending on model sizes and hardware
- Monitor GPU memory usage during generation and scoring
- Results are automatically timestamped to avoid conflicts
- The PEARL benchmark includes both full and lite versions
