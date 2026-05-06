"""
Evaluate Fine-Tuned Model — Comprehensive benchmarking and quality assessment.

Measures:
  1. Perplexity (lower = better)
  2. Exact/Fuzzy Match accuracy against ground truth
  3. BLEU score for generation quality
  4. Response latency and throughput
  5. Side-by-side comparison: base model vs fine-tuned model

Usage:
  python scripts/evaluate_finetuned.py \
    --model-path ./notebooks/qwen_lora_adapter \
    --base-model Qwen/Qwen2.5-3B-Instruct \
    --dataset notebooks/aicortexo_dataset.json \
    --mode lora

  python scripts/evaluate_finetuned.py \
    --model-path ./merged_model \
    --base-model Qwen/Qwen2.5-3B-Instruct \
    --dataset notebooks/aicortexo_dataset.json \
    --mode merged

Author: Muhammad Ishaq
"""

import argparse
import json
import time
import math
import sys
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_model(model_path, base_model=None, mode="lora", use_4bit=True):
    """Load model — supports LoRA adapter or merged model."""
    print(f"\nLoading model (mode={mode})...")

    if mode == "lora":
        from peft import PeftModel

        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            )
            base = AutoModelForCausalLM.from_pretrained(
                base_model, device_map="auto", quantization_config=bnb_config, trust_remote_code=True,
            )
        else:
            base = AutoModelForCausalLM.from_pretrained(
                base_model, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True,
            )
        model = PeftModel.from_pretrained(base, model_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def load_base_model(base_model_name, use_4bit=True):
    """Load the unmodified base model for comparison."""
    print(f"\nLoading base model for comparison: {base_model_name}")
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name, device_map="auto", quantization_config=bnb_config, trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True,
        )
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_new_tokens=150):
    """Generate a response and measure latency."""
    formatted = f"User: {prompt}\nAssistant:"
    inputs = tokenizer(formatted, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    start = time.time()
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=0.7, top_p=0.9, do_sample=True,
            repetition_penalty=1.2,
        )
    latency = time.time() - start

    full_response = tokenizer.decode(output[0], skip_special_tokens=True)
    if "Assistant:" in full_response:
        response = full_response.split("Assistant:")[-1].strip()
    else:
        response = full_response[len(formatted):].strip()

    tokens_generated = output.shape[1] - inputs["input_ids"].shape[1]
    return response, latency, tokens_generated


def compute_perplexity(model, tokenizer, texts, max_length=256):
    """Compute perplexity on a list of texts."""
    print("\nComputing perplexity...")
    total_loss = 0.0
    total_tokens = 0

    for text in texts:
        encodings = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
        input_ids = encodings.input_ids.to(model.device)

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

        total_loss += loss.item() * input_ids.shape[1]
        total_tokens += input_ids.shape[1]

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity


def fuzzy_match(response, expected, threshold=0.5):
    """Check if response contains enough keywords from expected answer."""
    expected_words = set(expected.lower().split())
    response_words = set(response.lower().split())
    # Remove common stop words
    stop_words = {"is", "the", "a", "an", "in", "of", "and", "to", "for", "on", "at", "by", "with"}
    expected_keywords = expected_words - stop_words
    if not expected_keywords:
        return True
    overlap = expected_keywords & response_words
    return len(overlap) / len(expected_keywords) >= threshold


def simple_bleu(reference, hypothesis, n=4):
    """Compute a simple BLEU-like score (unigram to n-gram precision)."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()

    if len(hyp_tokens) == 0:
        return 0.0

    precisions = []
    for i in range(1, n + 1):
        ref_ngrams = defaultdict(int)
        hyp_ngrams = defaultdict(int)

        for j in range(len(ref_tokens) - i + 1):
            ngram = tuple(ref_tokens[j:j + i])
            ref_ngrams[ngram] += 1

        for j in range(len(hyp_tokens) - i + 1):
            ngram = tuple(hyp_tokens[j:j + i])
            hyp_ngrams[ngram] += 1

        matches = 0
        total = 0
        for ngram, count in hyp_ngrams.items():
            matches += min(count, ref_ngrams.get(ngram, 0))
            total += count

        if total == 0:
            precisions.append(0.0)
        else:
            precisions.append(matches / total)

    # Geometric mean of precisions
    if any(p == 0 for p in precisions):
        return 0.0

    log_avg = sum(math.log(p) for p in precisions) / len(precisions)

    # Brevity penalty
    bp = min(1.0, math.exp(1 - len(ref_tokens) / max(len(hyp_tokens), 1)))

    return bp * math.exp(log_avg)


def evaluate(model, tokenizer, dataset, label="Model"):
    """Run full evaluation on a dataset."""
    print(f"\n{'='*60}")
    print(f"  Evaluating: {label}")
    print(f"  Samples: {len(dataset)}")
    print(f"{'='*60}")

    results = {
        "exact_matches": 0,
        "fuzzy_matches": 0,
        "bleu_scores": [],
        "latencies": [],
        "tokens_per_second": [],
        "responses": [],
    }

    for i, sample in enumerate(dataset):
        prompt = sample["prompt"]
        expected = sample["completion"]

        response, latency, tokens_gen = generate_response(model, tokenizer, prompt)

        # Metrics
        exact = expected.lower().strip() in response.lower().strip()
        fuzzy = fuzzy_match(response, expected)
        bleu = simple_bleu(expected, response)
        tps = tokens_gen / latency if latency > 0 else 0

        results["exact_matches"] += int(exact)
        results["fuzzy_matches"] += int(fuzzy)
        results["bleu_scores"].append(bleu)
        results["latencies"].append(latency)
        results["tokens_per_second"].append(tps)
        results["responses"].append({
            "prompt": prompt,
            "expected": expected,
            "generated": response,
            "exact_match": exact,
            "fuzzy_match": fuzzy,
            "bleu": bleu,
            "latency_ms": latency * 1000,
        })

        status = "EXACT" if exact else ("FUZZY" if fuzzy else "MISS")
        print(f"  [{i+1}/{len(dataset)}] [{status}] BLEU={bleu:.3f} | {latency*1000:.0f}ms | {prompt[:50]}...")

    n = len(dataset)
    summary = {
        "exact_match_rate": results["exact_matches"] / n,
        "fuzzy_match_rate": results["fuzzy_matches"] / n,
        "avg_bleu": sum(results["bleu_scores"]) / n,
        "avg_latency_ms": sum(results["latencies"]) / n * 1000,
        "avg_tokens_per_second": sum(results["tokens_per_second"]) / n,
    }

    print(f"\n{'─'*40}")
    print(f"  Results for: {label}")
    print(f"{'─'*40}")
    print(f"  Exact Match:   {summary['exact_match_rate']:.1%} ({results['exact_matches']}/{n})")
    print(f"  Fuzzy Match:   {summary['fuzzy_match_rate']:.1%} ({results['fuzzy_matches']}/{n})")
    print(f"  Avg BLEU:      {summary['avg_bleu']:.4f}")
    print(f"  Avg Latency:   {summary['avg_latency_ms']:.0f}ms")
    print(f"  Avg Tokens/s:  {summary['avg_tokens_per_second']:.1f}")

    return summary, results


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned LLM against ground truth")
    parser.add_argument("--model-path", required=True, help="Path to fine-tuned model or adapter")
    parser.add_argument("--base-model", required=True, help="Base model name for comparison")
    parser.add_argument("--dataset", required=True, help="Path to evaluation dataset (JSON)")
    parser.add_argument("--mode", default="lora", choices=["lora", "merged"], help="Model loading mode")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--compare", action="store_true", help="Also evaluate base model for comparison")
    parser.add_argument("--output", type=str, default=None, help="Save detailed results to JSON")
    args = parser.parse_args()

    # Load dataset
    with open(args.dataset, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"Loaded {len(dataset)} evaluation samples")

    # Evaluate fine-tuned model
    ft_model, ft_tokenizer = load_model(args.model_path, args.base_model, args.mode, not args.no_4bit)

    # Compute perplexity
    texts = [f"User: {s['prompt']}\nAssistant: {s['completion']}" for s in dataset]
    ft_ppl = compute_perplexity(ft_model, ft_tokenizer, texts)
    print(f"\n  Fine-tuned Perplexity: {ft_ppl:.2f}")

    ft_summary, ft_results = evaluate(ft_model, ft_tokenizer, dataset, "Fine-Tuned Model")
    ft_summary["perplexity"] = ft_ppl

    # Compare with base model
    if args.compare:
        del ft_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        base_model, base_tokenizer = load_base_model(args.base_model, not args.no_4bit)
        base_ppl = compute_perplexity(base_model, base_tokenizer, texts)
        print(f"\n  Base Perplexity: {base_ppl:.2f}")

        base_summary, base_results = evaluate(base_model, base_tokenizer, dataset, "Base Model (No Fine-Tuning)")
        base_summary["perplexity"] = base_ppl

        # Side-by-side comparison
        print(f"\n{'='*60}")
        print(f"  COMPARISON: Base vs Fine-Tuned")
        print(f"{'='*60}")
        print(f"  {'Metric':<25} {'Base':>12} {'Fine-Tuned':>12} {'Delta':>12}")
        print(f"  {'─'*61}")

        metrics = [
            ("Perplexity", base_ppl, ft_ppl, True),
            ("Exact Match", base_summary["exact_match_rate"], ft_summary["exact_match_rate"], False),
            ("Fuzzy Match", base_summary["fuzzy_match_rate"], ft_summary["fuzzy_match_rate"], False),
            ("Avg BLEU", base_summary["avg_bleu"], ft_summary["avg_bleu"], False),
            ("Avg Latency (ms)", base_summary["avg_latency_ms"], ft_summary["avg_latency_ms"], True),
        ]

        for name, base_val, ft_val, lower_better in metrics:
            delta = ft_val - base_val
            if name in ("Exact Match", "Fuzzy Match"):
                print(f"  {name:<25} {base_val:>11.1%} {ft_val:>11.1%} {delta:>+11.1%}")
            elif name == "Perplexity":
                print(f"  {name:<25} {base_val:>11.2f} {ft_val:>11.2f} {delta:>+11.2f}")
            else:
                print(f"  {name:<25} {base_val:>11.4f} {ft_val:>11.4f} {delta:>+11.4f}")

    # Save results
    if args.output:
        output_data = {"fine_tuned": {"summary": ft_summary, "details": ft_results["responses"]}}
        if args.compare:
            output_data["base"] = {"summary": base_summary, "details": base_results["responses"]}

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to {args.output}")

    print(f"\n{'='*60}\nEvaluation complete!\n{'='*60}")


if __name__ == "__main__":
    main()
