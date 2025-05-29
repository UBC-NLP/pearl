#!/bin/bash

module load CUDA/12.4.0 GCCcore/.12.2.0 CMake/3.24.3 Python/3.10.8
source ~/env_vllm/bin/activate

# 2. Change to your evaluation directory
cd /pearl/eval


# 3. Loop over each result file and score it pearl benchmark
for file in \
    results/output_gemma-3-12b-it_20250515_182306.jsonl \
    results/output_Qwen2.5-VL-32B-Instruct_20250515_000653.jsonl \
    results/output_gemma-3-27b-it_20250515_184227.jsonl \
    results/output_Qwen2.5-VL-3B-Instruct_20250514_143427.jsonl \
    results/output_gemma-3-4b-it_20250515_190602.jsonl \
    results/output_Qwen2.5-VL-72B-Instruct_20250514_094747.jsonl \
    results/output_Qwen2.5-VL-7B-Instruct_20250515_004442.jsonl \
    results/output_aya-vision-32b_20250514_233427.jsonl \
    results/output_aya-vision-8b_20250514_231009.jsonl
do
    echo "Scoring $file..."
    python score.py "$file" --max_concurrent 50
done

# # 3. Loop over each result file and score it lite benchmark
# for file in \
#     results_lite/output_aya-vision-8b_20250517_020352.jsonl \
#     results_lite/output_aya-vision-32b_20250517_022427.jsonl \
#     results_lite/output_gemma-3-4b-it_20250517_022019.jsonl \
#     results_lite/output_gemma-3-12b-it_20250517_015218.jsonl \
#     results_lite/output_gemma-3-27b-it_20250517_021423.jsonl \
#     results_lite/output_Qwen2.5-VL-3B-Instruct_20250517_020920.jsonl \
#     results_lite/output_Qwen2.5-VL-7B-Instruct_20250517_023745.jsonl \
#     results_lite/output_Qwen2.5-VL-32B-Instruct_20250517_015703.jsonl \
#     results_lite/output_Qwen2.5-VL-72B-Instruct_20250517_024639.jsonl
# do
#     echo "Scoring $file..."
#     python score.py "$file" --max_concurrent 50
# done



