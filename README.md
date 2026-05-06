# 🧠 LLM Fine-Tuning

> What if we could take a language model… and teach it something new that it doesn't know yet?

This repository demonstrates **end-to-end fine-tuning** of Large Language Models on custom datasets — from sentiment classification with GPT-2 to domain-specific knowledge injection with Qwen2.5 using LoRA.

---

## 📂 Repository Structure

```
LLM-Finetuning/
├── notebooks/
│   ├── fine-tuning-gpt-2.ipynb       # GPT-2 sentiment analysis fine-tuning
│   ├── fine-tuning-qwen.ipynb        # Qwen2.5-3B LoRA fine-tuning on custom data
│   └── aicortexo_dataset.json        # Custom prompt-completion dataset (15 samples)
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
pip install torch transformers datasets accelerate peft bitsandbytes evaluate scikit-learn
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

---

## 📝 Key Concepts

| Concept | Description |
|---|---|
| **Fine-Tuning** | Adapting a pre-trained model to a specific task or domain |
| **LoRA** | Freezes the base model and trains small low-rank matrices — drastically reducing trainable parameters |
| **4-bit Quantization** | Compresses model weights from FP32 → 4-bit, enabling 3B+ models on consumer GPUs |
| **Gradient Accumulation** | Simulates larger batch sizes by accumulating gradients over multiple forward passes |

---

## 👤 Author

**Muhammad Ishaq** — AI Engineer & Data Science Lead at [AICortexo](https://aicortexo.com)

---

## 📄 License

This project is open source and available for educational purposes.
