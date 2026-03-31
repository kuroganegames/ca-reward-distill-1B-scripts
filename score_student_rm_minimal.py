#!/usr/bin/env python3
"""保存済みの生徒 Reward Model で推論スコアを出す最小コード。

主な使い方:
1) 単発サンプルをスコア
    python score_student_rm_minimal.py \
      --model-dir ./student_rm_regression_trial/final_model \
      --prompt "富士山について短く説明して" \
      --response "富士山は日本で最も高い山です。"

2) messages(JSON) + response でスコア
    python score_student_rm_minimal.py \
      --model-dir ./student_rm_regression_trial/final_model \
      --messages-json '[{"role":"user","content":"こんにちは"}]' \
      --response "こんにちは。どうしましたか？"

3) JSONL をまとめてスコア
    python score_student_rm_minimal.py \
      --model-dir ./student_rm_regression_trial/final_model \
      --input-jsonl ./records.jsonl \
      --output-jsonl ./records_scored.jsonl \
      --batch-size 8

JSONL の各行は以下のどれかを想定:
- {"messages": [...], "response": "..."}
- {"prompt": "...", "response": "..."}
- {"text": "<already formatted chat text>"}

出力スコア:
- student_score_norm: 学習時の正規化空間での出力
- student_score_denorm: score_normalization.json があれば教師RMスケールへ戻した値
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score records with a trained student reward model.")
    parser.add_argument("--model-dir", type=str, required=True, help="train_student_rm_regression.py の final_model")
    parser.add_argument("--cache-dir", type=str, default=None, help="HF cache root を手動指定")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="推論デバイス")
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="モデル読み込み時の dtype。auto は GPU なら bf16 優先",
    )
    parser.add_argument("--max-length", type=int, default=2048, help="tokenize 時の最大長")
    parser.add_argument("--batch-size", type=int, default=8, help="JSONL モード時のバッチサイズ")

    parser.add_argument("--prompt", type=str, default=None, help="単発モード: user prompt")
    parser.add_argument("--response", type=str, default=None, help="単発モード: assistant response")
    parser.add_argument(
        "--messages-json",
        type=str,
        default=None,
        help="単発モード: messages JSON 文字列または .json ファイルパス",
    )

    parser.add_argument("--input-jsonl", type=str, default=None, help="JSONL 一括入力。'-' で stdin")
    parser.add_argument("--output-jsonl", type=str, default=None, help="JSONL 一括出力。未指定なら stdout")
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="JSONL に既に chat text がある場合の列名 (既定: text)",
    )
    parser.add_argument("--messages-column", type=str, default="messages")
    parser.add_argument("--prompt-column", type=str, default="prompt")
    parser.add_argument("--response-column", type=str, default="response")
    parser.add_argument("--pretty", action="store_true", help="単発モードの JSON を整形して表示")
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


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda が指定されましたが CUDA が利用できません。")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_torch_dtype(dtype_arg: str, device: torch.device):
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if device.type == "cpu":
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def maybe_set_pad_token(tokenizer) -> None:
    if tokenizer.pad_token_id is not None:
        return
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        return
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})


def load_score_normalization(model_dir: Path) -> Optional[Dict[str, Any]]:
    path = model_dir / "score_normalization.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def denormalize_score(score_norm: float, score_stats: Optional[Mapping[str, Any]]) -> float:
    if not score_stats:
        return float(score_norm)
    mode = str(score_stats.get("mode", "none"))
    if mode != "train_zscore":
        return float(score_norm)
    mean = float(score_stats.get("mean", 0.0))
    std = float(score_stats.get("std", 1.0))
    return float(score_norm) * std + mean


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return str(text).strip()


def load_json_or_file(value: str) -> Any:
    maybe_path = Path(value)
    if maybe_path.exists() and maybe_path.is_file():
        with maybe_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(value)


def build_full_chat(messages: Sequence[Mapping[str, Any]], response: str) -> List[Dict[str, str]]:
    normalized_response = normalize_text(response)
    built: List[Dict[str, str]] = [
        {"role": str(message.get("role", "")), "content": normalize_text(message.get("content", ""))}
        for message in messages
    ]
    if built and built[-1].get("role") == "assistant" and normalize_text(built[-1].get("content")) == normalized_response:
        return built
    built.append({"role": "assistant", "content": normalized_response})
    return built


def build_text_from_record(record: Mapping[str, Any], tokenizer, args: argparse.Namespace) -> str:
    text_column = args.text_column
    if text_column and record.get(text_column) is not None and normalize_text(record.get(text_column)):
        return normalize_text(record.get(text_column))

    messages = record.get(args.messages_column)
    response = record.get(args.response_column)

    if messages is None and record.get(args.prompt_column) is not None:
        messages = [{"role": "user", "content": normalize_text(record.get(args.prompt_column))}]
    if messages is None or response is None:
        raise KeyError(
            f"record には {args.text_column!r} か、{args.messages_column!r}+{args.response_column!r}、"
            f"または {args.prompt_column!r}+{args.response_column!r} が必要です。"
        )

    full_chat = build_full_chat(messages=messages, response=str(response))
    return tokenizer.apply_chat_template(full_chat, tokenize=False, add_generation_prompt=False)


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    handle = sys.stdin if path == "-" else open(path, "r", encoding="utf-8")
    try:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"JSONL parse error at line {line_no}: {exc}") from exc
            if not isinstance(payload, dict):
                raise SystemExit(f"JSONL line {line_no} is not an object.")
            yield payload
    finally:
        if handle is not sys.stdin:
            handle.close()


class JsonlWriter:
    def __init__(self, path: Optional[str]):
        self.path = path
        self.handle = sys.stdout if path is None else open(path, "w", encoding="utf-8")

    def write(self, record: Mapping[str, Any]) -> None:
        self.handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def close(self) -> None:
        if self.handle is not sys.stdout:
            self.handle.close()


@torch.inference_mode()
def score_texts(
    model,
    tokenizer,
    texts: Sequence[str],
    device: torch.device,
    max_length: int,
) -> List[float]:
    batch = tokenizer(
        list(texts),
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
        padding=True,
    )
    batch = {k: v.to(device) for k, v in batch.items()}
    logits = model(**batch).logits.squeeze(-1).float().cpu().tolist()
    if isinstance(logits, float):
        return [float(logits)]
    return [float(x) for x in logits]


def run_single_example(model, tokenizer, device: torch.device, score_stats: Optional[Mapping[str, Any]], args: argparse.Namespace) -> None:
    if args.messages_json is not None:
        messages = load_json_or_file(args.messages_json)
        if not isinstance(messages, list):
            raise SystemExit("--messages-json は messages の配列である必要があります。")
    elif args.prompt is not None:
        messages = [{"role": "user", "content": normalize_text(args.prompt)}]
    else:
        raise SystemExit("単発モードでは --prompt か --messages-json のどちらかが必要です。")

    if args.response is None:
        raise SystemExit("単発モードでは --response が必要です。")

    record = {
        args.messages_column: messages,
        args.response_column: args.response,
    }
    text = build_text_from_record(record, tokenizer=tokenizer, args=args)
    score_norm = score_texts(model, tokenizer, texts=[text], device=device, max_length=args.max_length)[0]
    score_denorm = denormalize_score(score_norm, score_stats)

    payload = {
        "student_score_norm": float(score_norm),
        "student_score_denorm": float(score_denorm),
        "messages": messages,
        "response": args.response,
    }
    if args.pretty:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False))


def run_jsonl_batch(model, tokenizer, device: torch.device, score_stats: Optional[Mapping[str, Any]], args: argparse.Namespace) -> None:
    writer = JsonlWriter(args.output_jsonl)
    records: List[Dict[str, Any]] = []
    texts: List[str] = []
    try:
        for record in iter_jsonl(args.input_jsonl):
            records.append(record)
            texts.append(build_text_from_record(record, tokenizer=tokenizer, args=args))
            if len(records) >= args.batch_size:
                score_and_flush(records, texts, writer, model, tokenizer, device, score_stats, args.max_length)
                records, texts = [], []
        if records:
            score_and_flush(records, texts, writer, model, tokenizer, device, score_stats, args.max_length)
    finally:
        writer.close()


def score_and_flush(
    records: List[Dict[str, Any]],
    texts: List[str],
    writer: JsonlWriter,
    model,
    tokenizer,
    device: torch.device,
    score_stats: Optional[Mapping[str, Any]],
    max_length: int,
) -> None:
    scores_norm = score_texts(model, tokenizer, texts=texts, device=device, max_length=max_length)
    for record, score_norm in zip(records, scores_norm):
        out = dict(record)
        out["student_score_norm"] = float(score_norm)
        out["student_score_denorm"] = float(denormalize_score(score_norm, score_stats))
        writer.write(out)


def main() -> None:
    args = parse_args()
    configure_hf_cache(args.cache_dir)

    model_dir = Path(args.model_dir).expanduser().resolve()
    device = resolve_device(args.device)
    torch_dtype = resolve_torch_dtype(args.torch_dtype, device)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    maybe_set_pad_token(tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, torch_dtype=torch_dtype)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    model.to(device)

    score_stats = load_score_normalization(model_dir)

    if args.input_jsonl is not None:
        run_jsonl_batch(model=model, tokenizer=tokenizer, device=device, score_stats=score_stats, args=args)
    else:
        run_single_example(model=model, tokenizer=tokenizer, device=device, score_stats=score_stats, args=args)


if __name__ == "__main__":
    main()
