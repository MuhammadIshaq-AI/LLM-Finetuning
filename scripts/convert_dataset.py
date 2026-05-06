"""
Convert Dataset Formats — Converts between common LLM fine-tuning dataset formats.

Supports:
  - prompt/completion (custom JSON) ↔ Alpaca format ↔ ChatML ↔ ShareGPT ↔ CSV

Usage:
  # Custom JSON → Alpaca format
  python scripts/convert_dataset.py \
    --input notebooks/aicortexo_dataset.json \
    --input-format custom \
    --output-format alpaca \
    --output notebooks/aicortexo_alpaca.json

  # Custom JSON → ChatML (for Qwen/Llama chat models)
  python scripts/convert_dataset.py \
    --input notebooks/aicortexo_dataset.json \
    --input-format custom \
    --output-format chatml \
    --output notebooks/aicortexo_chatml.json

Author: Muhammad Ishaq
"""

import argparse
import json
import csv
from pathlib import Path


# ─── Format Parsers ───────────────────────────────────────────────────────────

def parse_custom(data):
    """Parse {"prompt": ..., "completion": ...} format."""
    return [{"instruction": d["prompt"], "output": d["completion"], "input": ""} for d in data]


def parse_alpaca(data):
    """Parse Alpaca format: {"instruction", "input", "output"}."""
    return [{"instruction": d["instruction"], "output": d["output"], "input": d.get("input", "")} for d in data]


def parse_sharegpt(data):
    """Parse ShareGPT format: {"conversations": [{"from": "human/gpt", "value": ...}]}."""
    results = []
    for item in data:
        convs = item.get("conversations", [])
        human_msgs = [c["value"] for c in convs if c["from"] == "human"]
        gpt_msgs = [c["value"] for c in convs if c["from"] == "gpt"]
        for h, g in zip(human_msgs, gpt_msgs):
            results.append({"instruction": h, "output": g, "input": ""})
    return results


def parse_chatml(data):
    """Parse ChatML format: {"messages": [{"role": ..., "content": ...}]}."""
    results = []
    for item in data:
        msgs = item.get("messages", [])
        user_msgs = [m["content"] for m in msgs if m["role"] == "user"]
        asst_msgs = [m["content"] for m in msgs if m["role"] == "assistant"]
        for u, a in zip(user_msgs, asst_msgs):
            results.append({"instruction": u, "output": a, "input": ""})
    return results


# ─── Format Exporters ─────────────────────────────────────────────────────────

def export_custom(records):
    """Export to {"prompt": ..., "completion": ...} format."""
    return [{"prompt": r["instruction"], "completion": r["output"]} for r in records]


def export_alpaca(records):
    """Export to Alpaca format."""
    return [{"instruction": r["instruction"], "input": r["input"], "output": r["output"]} for r in records]


def export_sharegpt(records):
    """Export to ShareGPT format."""
    return [
        {
            "conversations": [
                {"from": "human", "value": r["instruction"]},
                {"from": "gpt", "value": r["output"]},
            ]
        }
        for r in records
    ]


def export_chatml(records, system_prompt="You are a helpful AI assistant."):
    """Export to ChatML/OpenAI format."""
    return [
        {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": r["instruction"]},
                {"role": "assistant", "content": r["output"]},
            ]
        }
        for r in records
    ]


def export_csv(records, output_path):
    """Export to CSV format."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["instruction", "input", "output"])
        writer.writeheader()
        writer.writerows(records)
    return None  # Already saved


def export_text(records):
    """Export to plain text format (for simple causal LM training)."""
    lines = []
    for r in records:
        lines.append(f"User: {r['instruction']}\nAssistant: {r['output']}\n")
    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

PARSERS = {
    "custom": parse_custom,
    "alpaca": parse_alpaca,
    "sharegpt": parse_sharegpt,
    "chatml": parse_chatml,
}

EXPORTERS = {
    "custom": export_custom,
    "alpaca": export_alpaca,
    "sharegpt": export_sharegpt,
    "chatml": export_chatml,
    "csv": export_csv,
    "text": export_text,
}


def main():
    parser = argparse.ArgumentParser(
        description="Convert between LLM fine-tuning dataset formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported formats:
  custom    - {"prompt": ..., "completion": ...}
  alpaca    - {"instruction": ..., "input": ..., "output": ...}
  sharegpt  - {"conversations": [{"from": "human/gpt", "value": ...}]}
  chatml    - {"messages": [{"role": ..., "content": ...}]}
  csv       - instruction,input,output (export only)
  text      - Plain text User/Assistant pairs (export only)

Examples:
  python scripts/convert_dataset.py -i data.json -if custom -of alpaca -o alpaca_data.json
  python scripts/convert_dataset.py -i data.json -if custom -of chatml -o chatml_data.json
        """,
    )
    parser.add_argument("--input", "-i", required=True, help="Input file path")
    parser.add_argument("--output", "-o", required=True, help="Output file path")
    parser.add_argument("--input-format", "-if", required=True, choices=list(PARSERS.keys()), help="Input format")
    parser.add_argument("--output-format", "-of", required=True, choices=list(EXPORTERS.keys()), help="Output format")
    parser.add_argument("--system-prompt", default="You are a helpful AI assistant.", help="System prompt for ChatML")

    args = parser.parse_args()

    # Load input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    print(f"Loaded {len(raw_data)} samples from {input_path} (format: {args.input_format})")

    # Parse
    records = PARSERS[args.input_format](raw_data)
    print(f"Parsed {len(records)} records")

    # Export
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.output_format == "csv":
        export_csv(records, str(output_path))
    elif args.output_format == "text":
        text_output = export_text(records)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text_output)
    elif args.output_format == "chatml":
        exported = export_chatml(records, args.system_prompt)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(exported, f, indent=2, ensure_ascii=False)
    else:
        exported = EXPORTERS[args.output_format](records)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(exported, f, indent=2, ensure_ascii=False)

    print(f"Exported {len(records)} records to {output_path} (format: {args.output_format})")

    # Show preview
    print(f"\nPreview (first 2 entries):")
    if args.output_format not in ("csv", "text"):
        with open(output_path, "r", encoding="utf-8") as f:
            preview = json.load(f)
        for i, entry in enumerate(preview[:2]):
            print(f"  [{i+1}] {json.dumps(entry, ensure_ascii=False)[:150]}...")


if __name__ == "__main__":
    main()
