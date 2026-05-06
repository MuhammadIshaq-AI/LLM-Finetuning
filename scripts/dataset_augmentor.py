"""
📊 Dataset Augmentor for LLM Fine-Tuning
=========================================
Expands small prompt-completion datasets using NLP augmentation techniques.
Designed for datasets like aicortexo_dataset.json (15 samples → 100+ samples).

Techniques:
  1. Prompt Paraphrasing — Rewrites questions in different styles
  2. Completion Variation — Adds stylistic diversity to answers
  3. Role-Based Formatting — Generates multi-format training samples
  4. Contextual Expansion — Creates follow-up Q&A from existing pairs

Usage:
  python scripts/dataset_augmentor.py \
    --input notebooks/aicortexo_dataset.json \
    --output notebooks/aicortexo_dataset_augmented.json \
    --multiplier 5

Author: Muhammad Ishaq — AI Engineer at AICortexo
"""

import json
import argparse
import random
import re
import copy
from pathlib import Path
from typing import List, Dict


# ─── Prompt Paraphrasing Templates ────────────────────────────────────────────
# Each pattern maps a detected intent to alternative phrasings
QUESTION_TEMPLATES = {
    "what_is": [
        "Can you tell me about {topic}?",
        "Explain {topic} to me.",
        "I'd like to know about {topic}.",
        "Give me an overview of {topic}.",
        "Describe {topic}.",
        "What do you know about {topic}?",
        "Tell me everything about {topic}.",
        "What exactly is {topic}?",
    ],
    "where_is": [
        "Where can I find {topic}?",
        "What is the location of {topic}?",
        "In which city is {topic} based?",
        "Where is {topic} headquartered?",
        "Tell me the location of {topic}.",
    ],
    "who_is": [
        "Tell me about {person}.",
        "Who exactly is {person}?",
        "Can you describe {person}?",
        "What does {person} do?",
        "Give me information about {person}.",
        "I want to learn about {person}.",
    ],
    "what_does": [
        "What are the responsibilities of {topic}?",
        "Describe what {topic} does.",
        "Can you explain the role of {topic}?",
        "Tell me about the activities of {topic}.",
        "What is {topic} involved in?",
    ],
    "does_it": [
        "Is it true that {topic}?",
        "Can you confirm if {topic}?",
        "Tell me whether {topic}.",
        "I want to know if {topic}.",
    ],
}

# ─── Completion Variation Prefixes ─────────────────────────────────────────────
ANSWER_PREFIXES = [
    "",  # Keep original
    "Sure! ",
    "Great question! ",
    "Absolutely. ",
    "Here's what I know: ",
    "Of course. ",
    "Let me explain. ",
]

# ─── Instruction-Style Format Templates ────────────────────────────────────────
INSTRUCTION_FORMATS = [
    # Standard User/Assistant (original format)
    lambda p, c: {"prompt": p, "completion": c},
    # Chat-style with system context
    lambda p, c: {
        "prompt": f"You are a knowledgeable AI assistant. Answer the following question accurately.\n\nQuestion: {p}",
        "completion": c,
    },
    # Direct instruction style
    lambda p, c: {
        "prompt": f"Instruction: Answer the following question.\nInput: {p}",
        "completion": f"Output: {c}",
    },
]


def detect_question_type(prompt: str) -> tuple:
    """Detect the type and extract the topic/subject from a prompt."""
    prompt_lower = prompt.lower().strip()

    # What is / What are
    match = re.match(r"what (?:is|are) (?:the )?(.*?)[\?]?$", prompt_lower)
    if match:
        return "what_is", match.group(1).strip().rstrip("?")

    # Where is
    match = re.match(r"where (?:is|are) (.*?)[\?]?$", prompt_lower)
    if match:
        return "where_is", match.group(1).strip().rstrip("?")

    # Who is
    match = re.match(r"who (?:is|are) (?:the )?(.*?)[\?]?$", prompt_lower)
    if match:
        return "who_is", match.group(1).strip().rstrip("?")

    # What does X do
    match = re.match(r"what does (.*?) do[\?]?$", prompt_lower)
    if match:
        return "what_does", match.group(1).strip().rstrip("?")

    # Does X ...
    match = re.match(r"does (.*?)[\?]?$", prompt_lower)
    if match:
        return "does_it", match.group(1).strip().rstrip("?")

    # What kind / What type
    match = re.match(r"what (?:kind|type) of (.*?)[\?]?$", prompt_lower)
    if match:
        return "what_is", match.group(1).strip().rstrip("?")

    return None, prompt.rstrip("?")


def paraphrase_prompt(prompt: str) -> List[str]:
    """Generate paraphrased versions of a prompt."""
    q_type, topic = detect_question_type(prompt)

    paraphrases = [prompt]  # Always include original

    if q_type and q_type in QUESTION_TEMPLATES:
        templates = QUESTION_TEMPLATES[q_type]
        key = "person" if q_type == "who_is" else "topic"
        for template in random.sample(templates, min(3, len(templates))):
            paraphrases.append(template.format(**{key: topic}))

    return paraphrases


def vary_completion(completion: str) -> List[str]:
    """Generate completion variations with different prefixes/styles."""
    variations = []
    for prefix in random.sample(ANSWER_PREFIXES, min(3, len(ANSWER_PREFIXES))):
        if prefix:
            # Lowercase the first char of completion when adding prefix
            varied = prefix + completion[0].lower() + completion[1:]
        else:
            varied = completion
        variations.append(varied)
    return variations


def generate_followup_qa(sample: Dict) -> List[Dict]:
    """Generate follow-up questions from existing Q&A pairs."""
    followups = []
    completion = sample["completion"]

    # If completion mentions a location, generate location-related followup
    if any(word in completion.lower() for word in ["islamabad", "pakistan", "based in"]):
        followups.append({
            "prompt": f"Tell me more about where {extract_subject(sample['prompt'])} operates from.",
            "completion": completion,
        })

    # If completion mentions technologies, generate tech-focused followup
    tech_keywords = ["python", "pytorch", "docker", "aws", "transformers", "pinecone"]
    mentioned_tech = [t for t in tech_keywords if t.lower() in completion.lower()]
    if mentioned_tech:
        followups.append({
            "prompt": f"What tech stack does {extract_subject(sample['prompt'])} use?",
            "completion": completion,
        })

    # If completion mentions a person, create a biographical followup
    if any(word in completion.lower() for word in ["founder", "led by", "engineer"]):
        followups.append({
            "prompt": f"Who leads {extract_subject(sample['prompt'])}?",
            "completion": completion,
        })

    return followups


def extract_subject(prompt: str) -> str:
    """Extract the main subject/entity from a prompt."""
    # Look for proper nouns (capitalized words that aren't at sentence start)
    words = prompt.split()
    for i, word in enumerate(words):
        clean = word.strip("?.,!")
        if clean and clean[0].isupper() and i > 0:
            # Collect consecutive capitalized words
            subject = [clean]
            for j in range(i + 1, len(words)):
                next_clean = words[j].strip("?.,!")
                if next_clean and next_clean[0].isupper():
                    subject.append(next_clean)
                else:
                    break
            return " ".join(subject)

    return "the subject"


def augment_dataset(
    data: List[Dict],
    multiplier: int = 5,
    use_paraphrasing: bool = True,
    use_completion_variation: bool = True,
    use_format_variation: bool = True,
    use_followups: bool = True,
) -> List[Dict]:
    """
    Augment a prompt-completion dataset.

    Args:
        data: List of {"prompt": ..., "completion": ...} dicts
        multiplier: Target multiplier for dataset size
        use_paraphrasing: Enable prompt paraphrasing
        use_completion_variation: Enable completion style variation
        use_format_variation: Enable instruction format variation
        use_followups: Enable follow-up Q&A generation

    Returns:
        Augmented list of prompt-completion pairs
    """
    augmented = []
    seen_prompts = set()

    for sample in data:
        prompt = sample["prompt"]
        completion = sample["completion"]

        # 1. Original sample (always include)
        augmented.append(copy.deepcopy(sample))
        seen_prompts.add(prompt.lower().strip())

        # 2. Paraphrased prompts
        if use_paraphrasing:
            paraphrases = paraphrase_prompt(prompt)
            for para_prompt in paraphrases:
                key = para_prompt.lower().strip()
                if key not in seen_prompts:
                    augmented.append({"prompt": para_prompt, "completion": completion})
                    seen_prompts.add(key)

        # 3. Completion variations (pair with original prompt)
        if use_completion_variation:
            variations = vary_completion(completion)
            for varied_completion in variations[1:]:  # Skip first (original)
                augmented.append({"prompt": prompt, "completion": varied_completion})

        # 4. Format variations
        if use_format_variation:
            for fmt_fn in INSTRUCTION_FORMATS[1:]:  # Skip standard format
                formatted = fmt_fn(prompt, completion)
                key = formatted["prompt"].lower().strip()
                if key not in seen_prompts:
                    augmented.append(formatted)
                    seen_prompts.add(key)

        # 5. Follow-up Q&A
        if use_followups:
            followups = generate_followup_qa(sample)
            for fq in followups:
                key = fq["prompt"].lower().strip()
                if key not in seen_prompts:
                    augmented.append(fq)
                    seen_prompts.add(key)

    # If we've exceeded the target, trim randomly (but always keep originals)
    target_size = len(data) * multiplier
    if len(augmented) > target_size:
        # Keep all originals, randomly sample from augmented
        originals = augmented[: len(data)]
        extras = augmented[len(data) :]
        random.shuffle(extras)
        augmented = originals + extras[: target_size - len(data)]

    random.shuffle(augmented)

    return augmented


def main():
    parser = argparse.ArgumentParser(
        description="📊 Dataset Augmentor — Expand small fine-tuning datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic augmentation (5x expansion)
  python scripts/dataset_augmentor.py --input notebooks/aicortexo_dataset.json --output notebooks/augmented.json

  # Aggressive augmentation (10x expansion)
  python scripts/dataset_augmentor.py --input notebooks/aicortexo_dataset.json --output notebooks/augmented.json --multiplier 10

  # Only paraphrasing (no format/completion variation)
  python scripts/dataset_augmentor.py --input notebooks/aicortexo_dataset.json --output notebooks/augmented.json --no-formats --no-variations
        """,
    )
    parser.add_argument("--input", "-i", required=True, help="Path to input JSON dataset")
    parser.add_argument("--output", "-o", required=True, help="Path to save augmented dataset")
    parser.add_argument("--multiplier", "-m", type=int, default=5, help="Target size multiplier (default: 5)")
    parser.add_argument("--no-paraphrase", action="store_true", help="Disable prompt paraphrasing")
    parser.add_argument("--no-variations", action="store_true", help="Disable completion variations")
    parser.add_argument("--no-formats", action="store_true", help="Disable instruction format variation")
    parser.add_argument("--no-followups", action="store_true", help="Disable follow-up Q&A generation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()
    random.seed(args.seed)

    # Load dataset
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"[LOAD] Loaded {len(data)} samples from {input_path}")

    # Augment
    augmented = augment_dataset(
        data,
        multiplier=args.multiplier,
        use_paraphrasing=not args.no_paraphrase,
        use_completion_variation=not args.no_variations,
        use_format_variation=not args.no_formats,
        use_followups=not args.no_followups,
    )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(augmented, f, indent=2, ensure_ascii=False)

    print(f"[DONE] Augmented dataset: {len(data)} -> {len(augmented)} samples ({len(augmented)/len(data):.1f}x)")
    print(f"[SAVE] Saved to {output_path}")

    # Show sample
    print(f"\n[PREVIEW] Sample augmented entries:")
    for i, sample in enumerate(augmented[:3]):
        print(f"\n  [{i+1}] Prompt:     {sample['prompt'][:80]}...")
        print(f"      Completion: {sample['completion'][:80]}...")


if __name__ == "__main__":
    main()
