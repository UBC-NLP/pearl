# PEARL Dataset Generation Pipeline

This directory contains the data generation pipeline for creating the PEARL (Perception of Arab Language and Visual Understanding) benchmark dataset. The pipeline uses large vision-language models to generate culturally-aware question-answer pairs about Arab cultural content.

## Overview

The PEARL dataset generation follows a **two-step process** that creates high-quality, culturally-grounded question-answer pairs based on Wikipedia articles and images related to Arab culture.

### Pipeline Components

1. **Prompt Generation System** (`prompts/`)
   - Multi-faceted question type definitions
   - Cultural examples and templates
   - Two-step generation prompts

2. **Generation Engine** (`run_generation_2_steps.py`)
   - Async processing with concurrent API calls
   - Image encoding and VLM interaction
   - Batch processing with error handling

3. **Input Data** (`wikipedia_data_w_images_filtered.jsonl`)
   - Curated Wikipedia articles about Arab culture
   - Associated images and captions
   - Metadata (country, category, title)

## Question Types

The pipeline generates 12 different types of culturally-aware questions:

| Question Type | Focus Area | Description |
|---------------|------------|-------------|
| **Cause and Effect** | Causal relationships | Explores why cultural elements exist and their impacts |
| **Chronological Sequence** | Temporal ordering | Examines historical progression of cultural elements |
| **Comparative Analysis** | Comparisons/contrasts | Compares cultural elements across regions or contexts |
| **Modern Context** | Contemporary relevance | How traditional elements adapt to modern times |
| **General Q&A** | Factual information | Straightforward questions about cultural elements |
| **Hypothesis Formation** | Theory development | Encourages forming theories about cultural phenomena |
| **Problem Solving** | Cultural challenges | Presents scenarios requiring cultural solutions |
| **Origin Identification** | Historical origins | Identifies sources and beginnings of cultural elements |
| **Perspective Shifting** | Multiple viewpoints | Examines elements from different cultural perspectives |
| **Role Playing** | Professional roles | Adopts specific cultural or professional perspectives |
| **Scenario Completion** | Contextual completion | Completes cultural scenarios or situations |

## Cultural Categories

The dataset covers diverse aspects of Arab culture:

- **Architecture** - Traditional buildings, design elements
- **Food** - Traditional cuisine and cooking methods
- **Clothing** - Traditional garments and textiles
- **Handicrafts** - Traditional arts and crafts
- **Music** - Musical instruments and traditions
- **Festivals & Celebrations** - Cultural events and ceremonies
- **Geography** - Landscapes and natural features
- **Flora & Fauna** - Plants and animals in Arab culture
- **Landmarks** - Historical and cultural sites

## Two-Step Generation Process

### Step 1: Augmented Caption Generation
The system first creates a detailed, contextual description of the image by:
- Starting with "تظهر الصورة" (The image shows...)
- Incorporating visual details from the image caption
- Adding relevant cultural context from Wikipedia articles
- Focusing on elements relevant to the target question type

### Step 2: Question-Answer Pair Generation
Using only the augmented caption, the system generates:
- **Questions** that reference cultural elements without naming them specifically
- **Answers** that explicitly identify and explain the cultural elements
- **Multiple perspectives** based on the question type requirements

## Technical Architecture

### Core Components

```
data_generation/
├── run_generation_2_steps.py          # Main generation script
├── prompts/
│   ├── all_prompts_from_article.py    # Question type definitions and templates
│   ├── examples.py                    # Cultural examples for each question type
│   └── few_shots_examples.json        # Additional example questions
└── output/                            # Generated dataset files
```



## Setup and Requirements

### Prerequisites
```bash
# Python packages
pandas
PIL (Pillow)
openai
aiofiles
tqdm
numpy
asyncio
```

### VLM Server Setup
The pipeline requires a running vision-language model server:

```bash
# Start the VLM server (example with Qwen2.5-VL-72B)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="0,1,2,4" python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-72B-Instruct \
    --tensor-parallel-size 4 \
    --dtype auto \
    --served-model-name qwen72vl \
    --max-model-len 30000 \
    --disable-log-requests \
    --port 8000 \
    --trust_remote_code \
    --gpu-memory-utilization 0.95
```

### Configuration
Update the server configuration in `run_generation_2_steps.py`:
```python
openai_api_base = "http://localhost:8000/v1"  # Update port if needed
MAX_CONCURRENT_CALLS = 50                     # Adjust based on server capacity
```

## Usage

### Basic Usage
```python
# Load and process the dataset
python run_generation_2_steps.py
```

### Key Parameters
- **Batch Size**: `batch_size = 20` (adjustable based on memory)
- **Concurrent Calls**: `MAX_CONCURRENT_CALLS = 50`
- **Output File**: Configurable in `main()` function
- **Question Pairs**: 2 Q&A pairs per question type per item

### Input Data Format
The pipeline expects JSONL input with the following structure:
```json
{
  "id": "unique_identifier",
  "title": "Article title in Arabic",
  "article": "Full Wikipedia article text",
  "country": "Country name",
  "type": "Cultural category",
  "image": "path/to/image.jpg",
  "caption": "Image description from wikipedia"
}
```

### Output Format
Generated data is saved as JSONL with this structure:
```json
{
  "index": 123,
  "processed": true,
  "generated_QAs": [
    ["Question type 1 raw response"],
    ["Question type 2 raw response"],
    // ... for all 12 question types
  ]
}
```

