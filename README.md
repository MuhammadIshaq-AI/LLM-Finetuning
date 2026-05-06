# 🧠 LLM Fine-Tuning

> What if we could take a language model… and teach it something new that it doesn't know yet?

This repository demonstrates **end-to-end fine-tuning** of Large Language Models on custom datasets — from sentiment classification with GPT-2 to domain-specific knowledge injection with Qwen2.5 using LoRA. Includes production-ready scripts for dataset augmentation, model evaluation, merging, and export.

---

## 📂 Repository Structure

```
LLM-Finetuning/
├── notebooks/
│   ├── fine-tuning-gpt-2.ipynb       # GPT-2 sentiment analysis fine-tuning
│   ├── fine-tuning-qwen.ipynb        # Qwen2.5-3B LoRA fine-tuning on custom data
│   └── aicortexo_dataset.json        # Custom prompt-completion dataset (15 samples)
├── scripts/
│   ├── dataset_augmentor.py          # Expand small datasets with NLP augmentation
│   ├── convert_dataset.py            # Convert between fine-tuning dataset formats
│   ├── evaluate_finetuned.py         # Benchmark fine-tuned vs base model
│   └── merge_and_export.py           # Merge LoRA adapters & export for production
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔬 Notebooks

### 1. Fine-Tuning GPT-2 — Sentiment Analysis

**Notebook:** [`fine-tuning-gpt-2.ipynb`](notebooks/fine-tuning-gpt-2.ipynb)

Fine-tunes **GPT-2** for binary sentiment classification on the **IMDB Movie Reviews** dataset.

**Pipeline:**
- Load the IMDB dataset (50K reviews) via HuggingFace `datasets`
- Tokenize using `GPT2Tokenizer` with custom padding (GPT-2 has no native pad token)
- Fine-tune `GPT2ForSequenceClassification` with HuggingFace `Trainer`
- Evaluate using accuracy, precision, recall, and F1 metrics
- Compare model predictions before vs. after fine-tuning

**Key Config:**
| Parameter | Value |
|---|---|
| Model | `gpt2` (124M params) |
| Dataset | IMDB (2K train / 500 test subset) |
| Learning Rate | `2e-5` |
| Epochs | `3` |
| Batch Size | `8` |
| Task | Sequence Classification (pos/neg) |

---

### 2. Fine-Tuning Qwen2.5 — Custom Knowledge with LoRA

**Notebook:** [`fine-tuning-qwen.ipynb`](notebooks/fine-tuning-qwen.ipynb)

Fine-tunes **Qwen2.5-3B-Instruct** on a custom dataset to teach the model domain-specific knowledge about [AICortexo](https://aicortexo.com) — using **LoRA** (Low-Rank Adaptation) for parameter-efficient training on a consumer GPU.

**Pipeline:**
1. Query the base model to show it doesn't know about AICortexo
2. Load custom `aicortexo_dataset.json` (15 prompt-completion pairs)
3. Preprocess data into `User: {prompt}\nAssistant: {completion}` format
4. Apply **4-bit quantization** via BitsAndBytes (NF4)
5. Attach **LoRA adapters** to attention layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`)
6. Train with HuggingFace `Trainer`
7. Save & reload LoRA adapter for inference

**Key Config:**
| Parameter | Value |
|---|---|
| Base Model | `Qwen/Qwen2.5-3B-Instruct` (3B params) |
| Quantization | 4-bit NF4 with double quantization |
| LoRA Rank (r) | `8` |
| LoRA Alpha | `16` |
| LoRA Dropout | `0.05` |
| Target Modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Trainable Params | **3.7M / 3.1B** (0.12%) |
| Optimizer | `paged_adamw_8bit` |
| Learning Rate | `2e-4` |
| Epochs | `3` |
| Batch Size | `1` (with 8-step gradient accumulation) |
| Precision | FP16 |

**Training Result:**
- Completed in **~49 seconds** (6 steps)
- Training Loss: `7.11`

---

## 🛠️ Scripts

### 1. Dataset Augmentor

**Script:** [`scripts/dataset_augmentor.py`](scripts/dataset_augmentor.py)

Expands small fine-tuning datasets using NLP augmentation techniques — critical for improving training quality when you only have a handful of examples.

**Techniques:**
- **Prompt Paraphrasing** — Rewrites questions in multiple styles
- **Completion Variation** — Adds stylistic diversity to answers
- **Instruction Format Mixing** — Generates chat-style, instruction-style, and direct formats
- **Follow-up Generation** — Creates contextual follow-up Q&A pairs

```bash
# 5x expansion (default)
python scripts/dataset_augmentor.py \
  --input notebooks/aicortexo_dataset.json \
  --output notebooks/aicortexo_augmented.json

# 10x expansion
python scripts/dataset_augmentor.py \
  --input notebooks/aicortexo_dataset.json \
  --output notebooks/aicortexo_augmented.json \
  --multiplier 10
```

---

### 2. Dataset Format Converter

**Script:** [`scripts/convert_dataset.py`](scripts/convert_dataset.py)

Converts between common LLM fine-tuning dataset formats so you can use your data with any framework.

**Supported Formats:**

| Format | Structure | Used By |
|---|---|---|
| Custom | `{"prompt", "completion"}` | This repo |
| Alpaca | `{"instruction", "input", "output"}` | Stanford Alpaca, Axolotl |
| ChatML | `{"messages": [{"role", "content"}]}` | OpenAI, Qwen, Llama 3 |
| ShareGPT | `{"conversations": [{"from", "value"}]}` | ShareGPT, FastChat |
| CSV | `instruction,input,output` | Spreadsheet workflows |
| Text | `User: ...\nAssistant: ...` | Causal LM training |

```bash
# Convert to ChatML format (for Qwen/Llama chat fine-tuning)
python scripts/convert_dataset.py \
  --input notebooks/aicortexo_dataset.json \
  --input-format custom \
  --output-format chatml \
  --output notebooks/aicortexo_chatml.json

# Convert to Alpaca format (for Axolotl/LLaMA-Factory)
python scripts/convert_dataset.py \
  --input notebooks/aicortexo_dataset.json \
  --input-format custom \
  --output-format alpaca \
  --output notebooks/aicortexo_alpaca.json
```

---

### 3. Model Evaluation

**Script:** [`scripts/evaluate_finetuned.py`](scripts/evaluate_finetuned.py)

Comprehensive benchmarking that compares fine-tuned models against the base model to prove training actually worked.

**Metrics:**
- **Perplexity** — How confident the model is on your data (lower = better)
- **Exact Match** — Does the response contain the expected answer?
- **Fuzzy Match** — Keyword overlap scoring for partial matches
- **BLEU Score** — N-gram precision for generation quality
- **Latency** — Response time and tokens/second throughput

```bash
# Evaluate LoRA adapter
python scripts/evaluate_finetuned.py \
  --model-path ./notebooks/qwen_lora_adapter \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --dataset notebooks/aicortexo_dataset.json \
  --mode lora \
  --compare

# Evaluate merged model
python scripts/evaluate_finetuned.py \
  --model-path ./merged_model \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --dataset notebooks/aicortexo_dataset.json \
  --mode merged \
  --compare \
  --output eval_results.json
```

---

### 4. Merge & Export

**Script:** [`scripts/merge_and_export.py`](scripts/merge_and_export.py)

Merges LoRA adapter weights back into the base model for production deployment — eliminating the adapter dependency and producing a standalone model.

**Capabilities:**
- Merge LoRA → Full model weights
- Push to HuggingFace Hub
- Quick inference validation

```bash
# Merge and save locally
python scripts/merge_and_export.py \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --adapter-path ./notebooks/qwen_lora_adapter \
  --output-dir ./merged_model

# Merge and push to HuggingFace Hub
python scripts/merge_and_export.py \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --adapter-path ./notebooks/qwen_lora_adapter \
  --output-dir ./merged_model \
  --push-to-hub MuhammadIshaq-AI/qwen-aicortexo-finetuned \
  --test
```

---

## 📊 Custom Dataset

The [`aicortexo_dataset.json`](notebooks/aicortexo_dataset.json) contains **15 prompt-completion pairs** covering:

- Company information (location, website, services)
- Founder details (Muhammad Ishaq — AI Engineer)
- Technical stack (Python, PyTorch, Transformers, Docker, AWS, Pinecone)
- Service offerings (chatbots, RAG systems, workflow automation)
- Industry verticals (e-commerce, healthcare, enterprise)

**Format:**
```json
{
  "prompt": "What is Aicortexo?",
  "completion": "Aicortexo is an AI services company focused on building automation systems, AI agents, and intelligent SaaS solutions."
}
```

---

## 🛠️ Tech Stack

- **PyTorch** 2.6+ with CUDA 12.4
- **Transformers** (HuggingFace)
- **PEFT** — Parameter-Efficient Fine-Tuning (LoRA)
- **BitsAndBytes** — 4-bit quantization
- **Accelerate** — Distributed training utilities
- **Datasets** — HuggingFace data loading
- **Evaluate** — Metrics (accuracy, precision, recall, F1)

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (tested on **RTX 3050 6GB**)

### Installation

```bash
# Clone the repository
git clone https://github.com/MuhammadIshaq-AI/LLM-Finetuning.git
cd LLM-Finetuning

# Install dependencies
pip install -r requirements.txt
```

### Running the Notebooks

```bash
# Launch Jupyter
jupyter notebook

# Open either notebook from the notebooks/ directory
```

> **Note:** The Qwen notebook requires a [Hugging Face access token](https://huggingface.co/settings/tokens) for gated model access. Use environment variables instead of hardcoding:
> ```python
> import os
> from huggingface_hub import login
> login(os.environ["HF_TOKEN"])
> ```

### Full Workflow Example

```bash
# 1. Augment your dataset (15 → 75+ samples)
python scripts/dataset_augmentor.py \
  --input notebooks/aicortexo_dataset.json \
  --output notebooks/aicortexo_augmented.json

# 2. Convert to ChatML format for better chat training
python scripts/convert_dataset.py \
  --input notebooks/aicortexo_augmented.json \
  --input-format custom \
  --output-format chatml \
  --output notebooks/aicortexo_chatml.json

# 3. Fine-tune using the notebook (fine-tuning-qwen.ipynb)

# 4. Evaluate the fine-tuned model
python scripts/evaluate_finetuned.py \
  --model-path ./notebooks/qwen_lora_adapter \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --dataset notebooks/aicortexo_dataset.json \
  --mode lora --compare

# 5. Merge and export for production
python scripts/merge_and_export.py \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --adapter-path ./notebooks/qwen_lora_adapter \
  --output-dir ./merged_model --test
```

---

## 📝 Key Concepts

| Concept | Description |
|---|---|
| **Fine-Tuning** | Adapting a pre-trained model to a specific task or domain |
| **LoRA** | Freezes the base model and trains small low-rank matrices — drastically reducing trainable parameters |
| **4-bit Quantization** | Compresses model weights from FP32 → 4-bit, enabling 3B+ models on consumer GPUs |
| **Gradient Accumulation** | Simulates larger batch sizes by accumulating gradients over multiple forward passes |
| **Dataset Augmentation** | Expanding small datasets with paraphrasing and format variation to improve training quality |
| **Model Merging** | Combining LoRA adapter weights back into the base model for standalone deployment |

---

## 👤 Author

**Muhammad Ishaq** — AI Engineer & Data Science Lead at [AICortexo](https://aicortexo.com)

---

## 📄 License

This project is open source and available for educational purposes.
