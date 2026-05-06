"""
Merge LoRA & Export — Merges adapters into base model for production use.

Usage:
  python scripts/merge_and_export.py \
    --base-model Qwen/Qwen2.5-3B-Instruct \
    --adapter-path ./notebooks/qwen_lora_adapter \
    --output-dir ./merged_model

  # Push to HuggingFace Hub
  python scripts/merge_and_export.py \
    --base-model Qwen/Qwen2.5-3B-Instruct \
    --adapter-path ./notebooks/qwen_lora_adapter \
    --output-dir ./merged_model \
    --push-to-hub your-username/model-name

Author: Muhammad Ishaq
"""

import argparse
import sys
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_lora(base_model_name, adapter_path, output_dir, dtype="float16"):
    print(f"\n{'='*60}")
    print(f"  Base Model:   {base_model_name}")
    print(f"  Adapter:      {adapter_path}")
    print(f"  Output:       {output_dir}")
    print(f"{'='*60}\n")

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map.get(dtype, torch.float16)

    # Step 1: Load base model
    print("[1/4] Loading base model...")
    start = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch_dtype, device_map="cpu", trust_remote_code=True,
    )
    print(f"  Done in {time.time() - start:.1f}s | Params: {sum(p.numel() for p in base_model.parameters()):,}")

    # Step 2: Load LoRA adapter
    print("\n[2/4] Loading LoRA adapter...")
    start = time.time()
    model = PeftModel.from_pretrained(base_model, adapter_path)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Done in {time.time() - start:.1f}s | Trainable: {trainable:,}/{total:,} ({trainable/total*100:.4f}%)")

    # Step 3: Merge
    print("\n[3/4] Merging LoRA weights into base model...")
    start = time.time()
    merged_model = model.merge_and_unload()
    print(f"  Done in {time.time() - start:.1f}s")

    # Step 4: Save
    print(f"\n[4/4] Saving merged model to {output_dir}...")
    start = time.time()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.save_pretrained(output_dir)
    model_size = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())
    print(f"  Done in {time.time() - start:.1f}s | Size: {model_size / (1024**3):.2f} GB")

    return merged_model, tokenizer


def push_to_hub(output_dir, repo_id, private=True):
    from huggingface_hub import HfApi
    print(f"\nPushing to HuggingFace Hub: {repo_id} (private={private})")
    api = HfApi()
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(folder_path=output_dir, repo_id=repo_id, commit_message="Upload merged model")
    print(f"  Done! https://huggingface.co/{repo_id}")


def quick_test(model_dir, prompts=None):
    print(f"\nQuick Inference Test")
    if prompts is None:
        prompts = ["What is AICortexo?", "Who founded AICortexo?", "What technologies does AICortexo use?"]

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
    model.eval()

    for prompt_text in prompts:
        formatted = f"User: {prompt_text}\nAssistant:"
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=100, temperature=0.7, top_p=0.9, do_sample=True)
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        print(f"\n  Q: {prompt_text}")
        print(f"  A: {response[:200]}")
    print("\nInference test complete!")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters & export for production")
    parser.add_argument("--base-model", required=True, help="Base model name (e.g., Qwen/Qwen2.5-3B-Instruct)")
    parser.add_argument("--adapter-path", required=True, help="Path to LoRA adapter directory")
    parser.add_argument("--output-dir", required=True, help="Directory to save merged model")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--push-to-hub", type=str, default=None, help="HuggingFace Hub repo ID")
    parser.add_argument("--private", action="store_true", help="Make Hub repo private")
    parser.add_argument("--test", action="store_true", help="Run quick inference test after merge")
    parser.add_argument("--test-prompts", nargs="+", default=None, help="Custom test prompts")
    args = parser.parse_args()

    if not Path(args.adapter_path).exists():
        print(f"Adapter path not found: {args.adapter_path}")
        sys.exit(1)

    merged_model, tokenizer = merge_lora(args.base_model, str(Path(args.adapter_path)), args.output_dir, args.dtype)
    del merged_model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.push_to_hub:
        push_to_hub(args.output_dir, args.push_to_hub, args.private)
    if args.test:
        quick_test(args.output_dir, args.test_prompts)

    print(f"\n{'='*60}\nAll done!\n{'='*60}")


if __name__ == "__main__":
    main()
