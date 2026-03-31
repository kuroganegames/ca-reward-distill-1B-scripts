#!/usr/bin/env python3
"""学習済み 生徒RM を再読み込みして、教師データセット上で簡易スコア確認を行う。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reload and sanity-check a trained student RM.")
    parser.add_argument("--model-dir", type=str, required=True, help="train_student_rm_regression.py が保存した final_model")
    parser.add_argument("--teacher-dataset-dir", type=str, default=None, help="必要なら教師データセットから数件読み込んで確認")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--train-split", type=str, default="train")
    parser.add_argument("--messages-column", type=str, default="messages")
    parser.add_argument("--response-column", type=str, default="response")
    parser.add_argument("--score-column", type=str, default="teacher_score")
    parser.add_argument("--text-column", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def configure_hf_cache(cache_dir: Optional[str]) -> None:
    if not cache_dir:
        return
    root = Path(cache_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "hub").mkdir(parents=True, exist_ok=True)
    (root / "datasets").mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(root)
    os.environ["HF_HUB_CACHE"] = str(root / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(root / "datasets")


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return str(text).strip()


def build_full_chat(messages: Sequence[Mapping[str, Any]], response: str) -> List[Dict[str, str]]:
    response = normalize_text(response)
    built: List[Dict[str, str]] = [
        {"role": str(m.get("role", "")), "content": normalize_text(m.get("content", ""))}
        for m in messages
    ]
    if built and built[-1].get("role") == "assistant" and normalize_text(built[-1].get("content")) == response:
        return built
    built.append({"role": "assistant", "content": response})
    return built


def build_text(example: Mapping[str, Any], tokenizer, args: argparse.Namespace) -> str:
    if args.text_column and args.text_column in example and example[args.text_column] is not None:
        return normalize_text(example[args.text_column])
    messages = example.get(args.messages_column)
    response = example.get(args.response_column)
    if messages is None or response is None:
        raise KeyError(f"{args.messages_column!r} / {args.response_column!r} が必要です。")
    full_chat = build_full_chat(messages, response)
    return tokenizer.apply_chat_template(full_chat, tokenize=False, add_generation_prompt=False)


def load_dataset_as_train_split(path: str, train_split: str) -> Dataset:
    loaded = load_from_disk(path)
    if isinstance(loaded, DatasetDict):
        if train_split not in loaded:
            raise SystemExit(f"split {train_split!r} がありません。利用可能: {list(loaded.keys())}")
        return loaded[train_split]
    if isinstance(loaded, Dataset):
        return loaded
    raise SystemExit(f"Unsupported dataset type: {type(loaded)!r}")


def main() -> None:
    args = parse_args()
    configure_hf_cache(args.cache_dir)

    model_dir = Path(args.model_dir).expanduser().resolve()
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    score_stats = None
    score_stats_path = model_dir / "score_normalization.json"
    if score_stats_path.exists():
        with score_stats_path.open("r", encoding="utf-8") as f:
            score_stats = json.load(f)

    print(f"Loaded model: {model_dir}")
    print(f"pad_token_id={model.config.pad_token_id}, num_labels={model.config.num_labels}, problem_type={model.config.problem_type}")
    if score_stats is not None:
        print(f"score_normalization={json.dumps(score_stats, ensure_ascii=False)}")

    if args.teacher_dataset_dir is None:
        return

    ds = load_dataset_as_train_split(args.teacher_dataset_dir, args.train_split)
    n = min(args.max_samples, len(ds))
    ds = ds.select(range(n))

    texts: List[str] = []
    gold_scores: List[float] = []
    for ex in ds:
        texts.append(build_text(ex, tokenizer=tokenizer, args=args))
        gold_scores.append(float(ex[args.score_column]))

    preds: List[float] = []
    for start in range(0, len(texts), args.batch_size):
        batch_texts = texts[start:start + args.batch_size]
        tokenized = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
            add_special_tokens=False,
            padding=True,
        ).to(device)
        with torch.no_grad():
            scores = model(**tokenized).logits.squeeze(-1).float().cpu().tolist()
        preds.extend(scores)

    if score_stats is not None and score_stats.get("mode") == "train_zscore":
        mean = float(score_stats.get("mean", 0.0))
        std = float(score_stats.get("std", 1.0))
        preds_denorm = [p * std + mean for p in preds]
    else:
        preds_denorm = preds

    print("\nSample predictions:")
    for idx, (pred, pred_denorm, gold, ex) in enumerate(zip(preds, preds_denorm, gold_scores, ds), start=1):
        sample_id = ex.get("sample_id", f"row{idx-1}")
        prompt_preview = ""
        messages = ex.get(args.messages_column)
        if messages:
            for message in reversed(messages):
                if str(message.get("role", "")) == "user":
                    prompt_preview = normalize_text(message.get("content", ""))
                    break
        prompt_preview = prompt_preview[:80].replace("\n", " ")
        print(f"[{idx}] sample_id={sample_id} pred_norm={pred:.4f} pred_denorm={pred_denorm:.4f} gold={gold:.4f} prompt={prompt_preview}")

    gold = np.asarray(gold_scores, dtype=np.float64)
    pred_denorm_arr = np.asarray(preds_denorm, dtype=np.float64)
    mse = float(np.mean((pred_denorm_arr - gold) ** 2))
    mae = float(np.mean(np.abs(pred_denorm_arr - gold)))
    if gold.size > 1 and np.std(gold) > 0 and np.std(pred_denorm_arr) > 0:
        pearson = float(np.corrcoef(pred_denorm_arr, gold)[0, 1])
    else:
        pearson = 0.0
    print("\nSubset metrics:")
    print(json.dumps({"n": len(gold_scores), "mse": mse, "mae": mae, "pearson": pearson}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
