#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a saved Hugging Face student RM directory to safetensors format."
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="Input directory, e.g. ./student_rm_regression_trial/final_model",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: <model-dir>_safetensors",
    )
    parser.add_argument(
        "--max-shard-size",
        type=str,
        default="5GB",
        help="Shard size passed to save_pretrained().",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to from_pretrained().",
    )
    return parser.parse_args()


def copy_sidecar_files(src_dir: Path, dst_dir: Path) -> List[str]:
    copied: List[str] = []
    for name in [
        "score_normalization.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "sentencepiece.bpe.model",
        "spiece.model",
        "added_tokens.json",
        "chat_template.jinja",
    ]:
        src = src_dir / name
        dst = dst_dir / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            copied.append(name)
    return copied


def save_as_safetensors(model, output_dir: Path, max_shard_size: str) -> str:
    """Support both transformers v4 and v5 style save_pretrained()."""
    try:
        model.save_pretrained(
            str(output_dir),
            safe_serialization=True,
            max_shard_size=max_shard_size,
        )
        return "save_pretrained(safe_serialization=True)"
    except TypeError as exc:
        # transformers v5 removed the safe_serialization kwarg and defaults to safetensors.
        if "safe_serialization" not in str(exc):
            raise
        model.save_pretrained(
            str(output_dir),
            max_shard_size=max_shard_size,
        )
        return "save_pretrained(default_safetensors)"


def main() -> None:
    args = parse_args()

    model_dir = Path(args.model_dir).expanduser().resolve()
    if not model_dir.exists():
        raise SystemExit(f"model-dir does not exist: {model_dir}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else model_dir.parent / f"{model_dir.name}_safetensors"
    )
    if output_dir.exists():
        raise SystemExit(f"output-dir already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "transformers is not installed. Run: pip install -U transformers"
        ) from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=args.trust_remote_code,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_dir,
            trust_remote_code=args.trust_remote_code,
            low_cpu_mem_usage=True,
        )
    except Exception:
        # clean up empty output dir on failure
        shutil.rmtree(output_dir, ignore_errors=True)
        raise

    try:
        save_mode = save_as_safetensors(model, output_dir, args.max_shard_size)
    except Exception as exc:
        shutil.rmtree(output_dir, ignore_errors=True)
        message = str(exc)
        if "safetensors" in message.lower():
            raise SystemExit(
                "Failed while saving as safetensors. You may need:\n"
                "  pip install -U safetensors\n\n"
                f"Original error: {exc}"
            ) from exc
        raise

    tokenizer.save_pretrained(str(output_dir))
    copied = copy_sidecar_files(model_dir, output_dir)

    safetensor_files = sorted(
        [p.name for p in output_dir.glob("*.safetensors")]
        + [p.name for p in output_dir.glob("*.safetensors.index.json")]
    )
    if not safetensor_files:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise SystemExit(
            "Conversion finished without any .safetensors files. "
            "Please check your transformers version and model directory."
        )

    summary = {
        "input_model_dir": str(model_dir),
        "output_dir": str(output_dir),
        "save_mode": save_mode,
        "copied_sidecar_files": copied,
        "safetensors_files": safetensor_files,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
