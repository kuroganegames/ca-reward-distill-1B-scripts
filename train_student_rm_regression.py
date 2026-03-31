#!/usr/bin/env python3
"""教師スコア付きデータセットから、生徒 Reward Model をスコア回帰蒸留で学習する。

想定入力:
- これまでに作成した教師データセット (`datasets.save_to_disk()` 済み)
- 基本列: `messages`, `response`, `teacher_score`

主な機能:
- `sbintuitions/sarashina2.2-1b-instruct-v0.1` に SequenceClassification head を載せて初期化
- `teacher_score` を回帰ラベルとして学習 (既定: train split 上の z-score 正規化 + MSE)
- 必要に応じて train/validation を prompt_hash ベースで分割
- tokenize 済み DatasetDict を `save_to_disk()` で保存して再利用可能
- `Trainer` で checkpoint 保存 / resume / best model restore
- 最終モデルと score 正規化統計を保存

起動例 (単GPU):
    CUDA_VISIBLE_DEVICES=0 python train_student_rm_regression.py \
      --teacher-dataset-dir ./teacher_data_trial/final_dataset \
      --output-dir ./student_rm_regression_trial \
      --cache-dir /data/hf_cache \
      --max-length 2048 \
      --per-device-train-batch-size 2 \
      --gradient-accumulation-steps 16 \
      --learning-rate 1e-5 \
      --num-train-epochs 1 \
      --gradient-checkpointing

起動例 (multi-GPU):
    accelerate launch --multi_gpu --num_processes 2 train_student_rm_regression.py \
      --teacher-dataset-dir ./teacher_data_trial/final_dataset \
      --output-dir ./student_rm_regression_trial \
      --cache-dir /data/hf_cache \
      --max-length 2048 \
      --per-device-train-batch-size 2 \
      --gradient-accumulation-steps 16 \
      --learning-rate 1e-5 \
      --num-train-epochs 1 \
      --gradient-checkpointing

出力ディレクトリ構成:
    output_dir/
      checkpoints/                # Trainer checkpoints
      tokenized_dataset/          # save_to_disk 済みの tokenize 済み dataset
      final_model/                # save_pretrained 済み生徒RM
      preprocess_summary.json
      training_summary.json
      score_normalization.json
      train_args.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import PartialState
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint


DEFAULT_STUDENT_MODEL = "sbintuitions/sarashina2.2-1b-instruct-v0.1"
DEFAULT_FINAL_MODEL_SUBDIR = "final_model"
DEFAULT_TOKENIZED_SUBDIR = "tokenized_dataset"
DEFAULT_CHECKPOINT_SUBDIR = "checkpoints"


@dataclass(frozen=True)
class ScoreStats:
    mode: str
    mean: float
    std: float
    clip_abs: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "mean": self.mean,
            "std": self.std,
            "clip_abs": self.clip_abs,
        }


class RewardRegressionCollator:
    """float labels 用のシンプルな collator。"""

    def __init__(self, tokenizer, pad_to_multiple_of: Optional[int] = 8):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.inner = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt",
        )

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        labels = [float(f["labels"]) for f in features]
        batch_features: List[Dict[str, Any]] = []
        for f in features:
            batch_features.append({
                "input_ids": f["input_ids"],
                "attention_mask": f["attention_mask"],
            })
        batch = self.inner(batch_features)
        batch["labels"] = torch.tensor(labels, dtype=torch.float32)
        return batch


class RegressionDistillationTrainer(Trainer):
    """SequenceClassification の logits と教師スコアの MSE/Huber を明示的に計算する。"""

    def __init__(self, *args, loss_type: str = "mse", huber_delta: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_type = loss_type
        self.huber_delta = float(huber_delta)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits.squeeze(-1)
        labels = labels.to(logits.device).float().view_as(logits)

        if self.loss_type == "mse":
            loss = F.mse_loss(logits.float(), labels.float())
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(logits.float(), labels.float(), beta=self.huber_delta)
        else:  # pragma: no cover
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return (loss, outputs) if return_outputs else loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a student reward model with score regression distillation.")

    parser.add_argument("--teacher-dataset-dir", type=str, required=True, help="教師スコア付き dataset の save_to_disk ディレクトリ")
    parser.add_argument("--output-dir", type=str, required=True, help="作業用 root ディレクトリ")
    parser.add_argument("--cache-dir", type=str, default=None, help="HF cache root を手動指定")

    parser.add_argument("--student-model-id", type=str, default=DEFAULT_STUDENT_MODEL, help="初期化に使う生徒 base model")
    parser.add_argument("--final-model-subdir", type=str, default=DEFAULT_FINAL_MODEL_SUBDIR, help="最終モデルの保存先 subdir 名")
    parser.add_argument("--tokenized-subdir", type=str, default=DEFAULT_TOKENIZED_SUBDIR, help="tokenize 済み dataset 保存 subdir 名")
    parser.add_argument("--checkpoint-subdir", type=str, default=DEFAULT_CHECKPOINT_SUBDIR, help="Trainer checkpoints 保存 subdir 名")
    parser.add_argument("--trust-remote-code", action="store_true", help="from_pretrained(..., trust_remote_code=True) を許可")
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="auto",
        choices=["auto", "sdpa", "eager", "flash_attention_2"],
        help="必要なら attention 実装を明示する",
    )

    parser.add_argument("--train-split", type=str, default="train", help="入力 DatasetDict に train がある場合の split 名")
    parser.add_argument("--validation-split", type=str, default="validation", help="入力 DatasetDict に validation がある場合の split 名")
    parser.add_argument("--messages-column", type=str, default="messages", help="prompt messages 列名")
    parser.add_argument("--response-column", type=str, default="response", help="assistant 応答列名")
    parser.add_argument("--score-column", type=str, default="teacher_score", help="教師スコア列名")
    parser.add_argument("--prompt-hash-column", type=str, default="prompt_hash", help="同一 prompt 判定に使う列名")
    parser.add_argument("--sample-id-column", type=str, default="sample_id", help="fallback 用の一意ID列")
    parser.add_argument("--text-column", type=str, default=None, help="既に全文文字列がある場合の列名。指定時は messages+response より優先")

    parser.add_argument("--validation-ratio", type=float, default=0.02, help="validation split が無いときの holdout 比率")
    parser.add_argument("--split-mod-base", type=int, default=10000, help="hash split の分母")
    parser.add_argument("--max-train-samples", type=int, default=None, help="試運転用に train を先頭N件に制限")
    parser.add_argument("--max-eval-samples", type=int, default=None, help="試運転用に eval を先頭N件に制限")
    parser.add_argument("--shuffle-before-select", action="store_true", help="max_*_samples を切る前に shuffle する")

    parser.add_argument("--max-length", type=int, default=2048, help="tokenize 時の最大長")
    parser.add_argument("--preprocessing-num-workers", type=int, default=8, help="dataset.map の num_proc")
    parser.add_argument("--overwrite-tokenized-dataset", action="store_true", help="既存 tokenized_dataset を作り直す")
    parser.add_argument("--keep-raw-columns", action="store_true", help="tokenized dataset に元列も残す")

    parser.add_argument(
        "--score-normalization",
        type=str,
        default="train_zscore",
        choices=["none", "train_zscore"],
        help="教師スコアの正規化方式",
    )
    parser.add_argument("--score-clip-abs", type=float, default=None, help="正規化後ラベルを ±この値で clip")
    parser.add_argument("--loss-type", type=str, default="mse", choices=["mse", "huber"], help="回帰損失")
    parser.add_argument("--huber-delta", type=float, default=1.0, help="Huber 損失の delta")

    parser.add_argument("--per-device-train-batch-size", type=int, default=2)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1, help="正の値なら epoch より優先")
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", type=str, default="linear")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--optim", type=str, default="adamw_torch_fused")

    parser.add_argument("--eval-strategy", type=str, default="steps", choices=["no", "steps", "epoch"])
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-strategy", type=str, default="steps", choices=["no", "steps", "epoch"])
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-accumulation-steps", type=int, default=16)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--load-best-model-at-end", action="store_true", help="評価指標に基づく best model を最終採用")
    parser.add_argument("--metric-for-best-model", type=str, default=None, help="例: eval_mae_denorm / eval_loss")
    parser.add_argument("--greater-is-better", action="store_true", help="metric_for_best_model が大きいほど良い場合")

    parser.add_argument("--gradient-checkpointing", action="store_true", help="勾配チェックポイントでメモリ節約")
    parser.add_argument("--torch-compile", action="store_true", help="torch_compile を有効化")
    parser.add_argument("--tf32", action="store_true", help="TF32 を許可 (CUDA のみ)")
    parser.add_argument("--bf16", action="store_true", help="bf16 学習を有効化")
    parser.add_argument("--fp16", action="store_true", help="fp16 学習を有効化")

    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="checkpoint path または 'last'")
    parser.add_argument("--report-to", type=str, default="none", help="Trainer report_to")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-safetensors", action="store_true", help="checkpoint/model を safetensors で保存")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.validation_ratio < 0 or args.validation_ratio >= 1:
        raise SystemExit("--validation-ratio は [0,1) の範囲で指定してください。")
    if args.max_length <= 0:
        raise SystemExit("--max-length は 1 以上にしてください。")
    if args.per_device_train_batch_size <= 0 or args.per_device_eval_batch_size <= 0:
        raise SystemExit("batch size は 1 以上にしてください。")
    if args.gradient_accumulation_steps <= 0:
        raise SystemExit("--gradient-accumulation-steps は 1 以上にしてください。")
    if args.fp16 and args.bf16:
        raise SystemExit("--bf16 と --fp16 は同時に指定できません。")
    if args.eval_strategy == "no" and args.load_best_model_at_end:
        raise SystemExit("--eval-strategy no では --load-best-model-at-end を使えません。")
    if args.eval_strategy == "no" and args.validation_ratio > 0 and args.validation_split:
        pass
    if args.save_strategy == "steps" and args.save_steps <= 0:
        raise SystemExit("--save-steps は 1 以上にしてください。")
    if args.eval_strategy == "steps" and args.eval_steps <= 0:
        raise SystemExit("--eval-steps は 1 以上にしてください。")


def configure_hf_cache(cache_dir: Optional[str]) -> Optional[Path]:
    if not cache_dir:
        return None
    root = Path(cache_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "hub").mkdir(parents=True, exist_ok=True)
    (root / "datasets").mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(root)
    os.environ["HF_HUB_CACHE"] = str(root / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(root / "datasets")
    return root


def safe_json_dump(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def maybe_enable_tf32(enabled: bool) -> None:
    if not enabled or not torch.cuda.is_available():
        return
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return str(text).strip()


def canonicalize_messages(messages: Sequence[Mapping[str, Any]]) -> str:
    parts: List[str] = []
    for message in messages:
        role = str(message.get("role", ""))
        content = normalize_text(message.get("content", ""))
        parts.append(f"<{role}>\n{content}")
    return "\n".join(parts)


def stable_example_key(example: Mapping[str, Any], args: argparse.Namespace) -> str:
    for col in [args.prompt_hash_column, args.sample_id_column]:
        value = example.get(col)
        if value is not None and str(value) != "":
            return str(value)
    if args.text_column and example.get(args.text_column) is not None:
        return hashlib.sha1(str(example[args.text_column]).encode("utf-8")).hexdigest()
    messages = example.get(args.messages_column) or []
    response = example.get(args.response_column) or ""
    basis = canonicalize_messages(messages) + "\n<assistant>\n" + normalize_text(response)
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()


def bucket_from_key(key: str, mod_base: int) -> int:
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % mod_base


def ensure_datasetdict(obj: Any, args: argparse.Namespace) -> DatasetDict:
    if isinstance(obj, DatasetDict):
        return obj
    if isinstance(obj, Dataset):
        return DatasetDict({args.train_split: obj})
    raise TypeError(f"Unsupported dataset type: {type(obj)!r}")


def load_teacher_dataset(args: argparse.Namespace) -> DatasetDict:
    loaded = load_from_disk(args.teacher_dataset_dir)
    dataset_dict = ensure_datasetdict(loaded, args)
    if args.train_split not in dataset_dict:
        raise SystemExit(f"入力 dataset に train split {args.train_split!r} がありません。利用可能: {list(dataset_dict.keys())}")
    return dataset_dict


def split_dataset_if_needed(dataset_dict: DatasetDict, args: argparse.Namespace) -> DatasetDict:
    if args.validation_split in dataset_dict:
        return DatasetDict({
            "train": dataset_dict[args.train_split],
            "validation": dataset_dict[args.validation_split],
        })

    train_ds = dataset_dict[args.train_split]
    if args.validation_ratio <= 0:
        return DatasetDict({"train": train_ds})

    threshold = int(args.validation_ratio * args.split_mod_base)

    def is_validation(example: Mapping[str, Any]) -> bool:
        key = stable_example_key(example, args)
        return bucket_from_key(key, args.split_mod_base) < threshold

    val_ds = train_ds.filter(is_validation, desc="Create validation split by stable hash")
    trn_ds = train_ds.filter(lambda ex: not is_validation(ex), desc="Create train split by stable hash")
    if len(trn_ds) == 0:
        raise SystemExit("train split が空になりました。--validation-ratio を下げてください。")
    if len(val_ds) == 0:
        return DatasetDict({"train": trn_ds})
    return DatasetDict({"train": trn_ds, "validation": val_ds})


def maybe_shuffle_and_select(ds: Dataset, max_samples: Optional[int], seed: int, shuffle_before_select: bool) -> Dataset:
    if max_samples is None:
        return ds
    if max_samples <= 0:
        raise SystemExit("max_*_samples は 1 以上にしてください。")
    if shuffle_before_select:
        ds = ds.shuffle(seed=seed)
    max_samples = min(max_samples, len(ds))
    return ds.select(range(max_samples))


def compute_score_stats(train_ds: Dataset, score_column: str, mode: str, clip_abs: Optional[float]) -> ScoreStats:
    values = np.asarray(train_ds[score_column], dtype=np.float64)
    if values.size == 0:
        raise SystemExit("train split が空です。")
    if mode == "none":
        mean = 0.0
        std = 1.0
    elif mode == "train_zscore":
        mean = float(values.mean())
        std = float(values.std())
        if not np.isfinite(std) or std < 1e-6:
            std = 1.0
    else:  # pragma: no cover
        raise ValueError(f"Unknown normalization mode: {mode}")
    return ScoreStats(mode=mode, mean=mean, std=std, clip_abs=clip_abs)


def apply_score_transform(score: float, stats: ScoreStats) -> float:
    value = float(score)
    if stats.mode == "train_zscore":
        value = (value - stats.mean) / stats.std
    if stats.clip_abs is not None:
        clip = float(stats.clip_abs)
        value = max(-clip, min(clip, value))
    return float(value)


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


def build_text_from_example(example: Mapping[str, Any], tokenizer, args: argparse.Namespace) -> str:
    if args.text_column and args.text_column in example and example[args.text_column] is not None:
        return normalize_text(example[args.text_column])
    messages = example.get(args.messages_column)
    response = example.get(args.response_column)
    if messages is None or response is None:
        raise KeyError(
            f"text_column 未指定時は {args.messages_column!r} と {args.response_column!r} が必要です。"
        )
    full_chat = build_full_chat(messages, response)
    return tokenizer.apply_chat_template(full_chat, tokenize=False, add_generation_prompt=False)


def tokenize_batch(examples: MutableMapping[str, List[Any]], tokenizer, args: argparse.Namespace, stats: ScoreStats) -> Dict[str, Any]:
    size = len(examples[args.score_column])
    texts: List[str] = []
    labels: List[float] = []
    raw_scores: List[float] = []

    for i in range(size):
        example = {key: value[i] for key, value in examples.items()}
        text = build_text_from_example(example, tokenizer, args)
        raw_score = float(example[args.score_column])
        texts.append(text)
        raw_scores.append(raw_score)
        labels.append(apply_score_transform(raw_score, stats))

    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=args.max_length,
        add_special_tokens=False,
    )
    tokenized["labels"] = labels
    tokenized["teacher_score_raw"] = raw_scores
    tokenized["length"] = [len(ids) for ids in tokenized["input_ids"]]
    return tokenized


def tokenize_batch_for_map(examples: MutableMapping[str, List[Any]], tokenizer, args_dict: Dict[str, Any], score_stats_dict: Dict[str, Any]) -> Dict[str, Any]:
    namespace = argparse.Namespace(**args_dict)
    score_stats = ScoreStats(**score_stats_dict)
    return tokenize_batch(examples=examples, tokenizer=tokenizer, args=namespace, stats=score_stats)


def prepare_tokenized_dataset(
    raw_dataset_dict: DatasetDict,
    tokenizer,
    args: argparse.Namespace,
    tokenized_dir: Path,
    preprocess_summary_path: Path,
    score_stats_path: Path,
    state: PartialState,
) -> Tuple[DatasetDict, ScoreStats, Dict[str, Any]]:
    if tokenized_dir.exists() and not args.overwrite_tokenized_dataset:
        tokenized = load_from_disk(str(tokenized_dir))
        with score_stats_path.open("r", encoding="utf-8") as f:
            score_stats = ScoreStats(**json.load(f))
        with preprocess_summary_path.open("r", encoding="utf-8") as f:
            preprocess_summary = json.load(f)
        return tokenized, score_stats, preprocess_summary

    if state.is_main_process:
        if tokenized_dir.exists() and args.overwrite_tokenized_dataset:
            import shutil
            shutil.rmtree(tokenized_dir)

        tokenized_dir.parent.mkdir(parents=True, exist_ok=True)
        preprocess_summary_path.parent.mkdir(parents=True, exist_ok=True)

        train_ds = raw_dataset_dict["train"]
        train_ds = maybe_shuffle_and_select(train_ds, args.max_train_samples, args.seed, args.shuffle_before_select)

        work = DatasetDict({"train": train_ds})
        if "validation" in raw_dataset_dict:
            val_ds = raw_dataset_dict["validation"]
            val_ds = maybe_shuffle_and_select(val_ds, args.max_eval_samples, args.seed + 1, args.shuffle_before_select)
            work["validation"] = val_ds

        score_stats = compute_score_stats(work["train"], args.score_column, args.score_normalization, args.score_clip_abs)

        remove_columns: Optional[List[str]] = None
        if not args.keep_raw_columns:
            remove_columns = list(work["train"].column_names)

        tokenized_splits: Dict[str, Dataset] = {}
        for split_name, split_ds in work.items():
            tokenized_split = split_ds.map(
                tokenize_batch_for_map,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=remove_columns,
                desc=f"Tokenize split={split_name}",
                fn_kwargs={
                    "tokenizer": tokenizer,
                    "args_dict": vars(args),
                    "score_stats_dict": score_stats.to_dict(),
                },
            )
            tokenized_splits[split_name] = tokenized_split

        tokenized = DatasetDict(tokenized_splits)
        tokenized.save_to_disk(str(tokenized_dir))

        preprocess_summary = {
            "teacher_dataset_dir": str(Path(args.teacher_dataset_dir).resolve()),
            "tokenized_dataset_dir": str(tokenized_dir.resolve()),
            "student_model_id": args.student_model_id,
            "max_length": args.max_length,
            "score_column": args.score_column,
            "score_normalization": score_stats.to_dict(),
            "splits": {split: len(ds) for split, ds in tokenized.items()},
            "keep_raw_columns": bool(args.keep_raw_columns),
        }
        safe_json_dump(score_stats.to_dict(), score_stats_path)
        safe_json_dump(preprocess_summary, preprocess_summary_path)

    state.wait_for_everyone()
    tokenized = load_from_disk(str(tokenized_dir))
    with score_stats_path.open("r", encoding="utf-8") as f:
        score_stats = ScoreStats(**json.load(f))
    with preprocess_summary_path.open("r", encoding="utf-8") as f:
        preprocess_summary = json.load(f)
    return tokenized, score_stats, preprocess_summary


def maybe_set_pad_token(tokenizer) -> bool:
    if tokenizer.pad_token_id is not None:
        return False
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        return False
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    return True


def resolve_attn_implementation(attn_impl: str) -> Optional[str]:
    return None if attn_impl == "auto" else attn_impl


def create_student_model_and_tokenizer(args: argparse.Namespace):
    tokenizer = AutoTokenizer.from_pretrained(
        args.student_model_id,
        trust_remote_code=args.trust_remote_code,
    )
    resize_needed = maybe_set_pad_token(tokenizer)

    config = AutoConfig.from_pretrained(
        args.student_model_id,
        trust_remote_code=args.trust_remote_code,
    )
    config.num_labels = 1
    config.pad_token_id = tokenizer.pad_token_id
    config.problem_type = "regression"

    model_kwargs: Dict[str, Any] = {
        "config": config,
        "trust_remote_code": args.trust_remote_code,
    }
    attn_impl = resolve_attn_implementation(args.attn_implementation)
    if attn_impl is not None:
        model_kwargs["attn_implementation"] = attn_impl

    model = AutoModelForSequenceClassification.from_pretrained(
        args.student_model_id,
        **model_kwargs,
    )

    if resize_needed:
        model.resize_token_embeddings(len(tokenizer))

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.problem_type = "regression"
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    return model, tokenizer


def maybe_enable_bf16_fp16(args: argparse.Namespace) -> Tuple[bool, bool]:
    if not torch.cuda.is_available():
        return False, False
    bf16 = bool(args.bf16)
    fp16 = bool(args.fp16)
    if bf16 and not torch.cuda.is_bf16_supported():
        bf16 = False
        fp16 = True if not fp16 else fp16
    return bf16, fp16


def make_compute_metrics(score_stats: ScoreStats):
    def compute_metrics(eval_pred) -> Dict[str, float]:
        predictions, labels = eval_pred
        preds = np.asarray(predictions)
        labs = np.asarray(labels)
        if preds.ndim > 1:
            preds = preds.squeeze(-1)
        preds = preds.astype(np.float64)
        labs = labs.astype(np.float64)

        metrics: Dict[str, float] = {}
        metrics["mse"] = float(np.mean((preds - labs) ** 2))
        metrics["mae"] = float(np.mean(np.abs(preds - labs)))
        if preds.size > 1 and np.std(preds) > 0 and np.std(labs) > 0:
            metrics["pearson"] = float(np.corrcoef(preds, labs)[0, 1])
        else:
            metrics["pearson"] = 0.0

        if score_stats.mode == "train_zscore":
            preds_denorm = preds * score_stats.std + score_stats.mean
            labs_denorm = labs * score_stats.std + score_stats.mean
        else:
            preds_denorm = preds
            labs_denorm = labs
        metrics["mse_denorm"] = float(np.mean((preds_denorm - labs_denorm) ** 2))
        metrics["mae_denorm"] = float(np.mean(np.abs(preds_denorm - labs_denorm)))
        if preds_denorm.size > 1 and np.std(preds_denorm) > 0 and np.std(labs_denorm) > 0:
            metrics["pearson_denorm"] = float(np.corrcoef(preds_denorm, labs_denorm)[0, 1])
        else:
            metrics["pearson_denorm"] = 0.0
        return metrics

    return compute_metrics


def build_training_arguments(args: argparse.Namespace, checkpoint_dir: Path) -> TrainingArguments:
    bf16, fp16 = maybe_enable_bf16_fp16(args)
    eval_enabled = args.eval_strategy != "no"
    load_best = bool(args.load_best_model_at_end and eval_enabled)

    metric_for_best_model = args.metric_for_best_model
    if load_best and not metric_for_best_model:
        metric_for_best_model = "eval_mae_denorm"

    return TrainingArguments(
        output_dir=str(checkpoint_dir),
        do_train=True,
        do_eval=eval_enabled,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        optim=args.optim,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps if args.eval_strategy == "steps" else None,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps if args.save_strategy == "steps" else None,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=load_best,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=args.greater_is_better if args.metric_for_best_model else False,
        bf16=bf16,
        fp16=fp16,
        tf32=args.tf32 if torch.cuda.is_available() else False,
        gradient_checkpointing=args.gradient_checkpointing,
        torch_compile=args.torch_compile,
        report_to=args.report_to,
        seed=args.seed,
        data_seed=args.seed,
        eval_accumulation_steps=args.eval_accumulation_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        ddp_find_unused_parameters=False,
        remove_unused_columns=True,
        save_safetensors=args.save_safetensors,
        label_names=["labels"],
    )


def resolve_resume_checkpoint(args: argparse.Namespace, checkpoint_dir: Path) -> Optional[str]:
    if args.resume_from_checkpoint is None:
        return None
    if args.resume_from_checkpoint == "last":
        last_ckpt = get_last_checkpoint(str(checkpoint_dir))
        if last_ckpt is None:
            raise SystemExit(f"checkpoint が見つかりませんでした: {checkpoint_dir}")
        return last_ckpt
    return args.resume_from_checkpoint


def write_training_summary(
    path: Path,
    args: argparse.Namespace,
    preprocess_summary: Mapping[str, Any],
    score_stats: ScoreStats,
    training_args: TrainingArguments,
    trainer: Trainer,
    tokenized: DatasetDict,
    resume_checkpoint: Optional[str],
) -> None:
    metrics: Dict[str, Any] = {}
    if trainer.state.log_history:
        metrics["log_history_tail"] = trainer.state.log_history[-20:]
    metrics["best_model_checkpoint"] = trainer.state.best_model_checkpoint
    metrics["best_metric"] = trainer.state.best_metric
    metrics["global_step"] = trainer.state.global_step
    metrics["epoch"] = trainer.state.epoch

    payload = {
        "student_model_id": args.student_model_id,
        "teacher_dataset_dir": str(Path(args.teacher_dataset_dir).resolve()),
        "preprocess_summary": dict(preprocess_summary),
        "score_normalization": score_stats.to_dict(),
        "tokenized_splits": {split: len(ds) for split, ds in tokenized.items()},
        "training_arguments": training_args.to_dict(),
        "resume_from_checkpoint": resume_checkpoint,
        "trainer_state": metrics,
    }
    safe_json_dump(payload, path)


def main() -> None:
    args = parse_args()
    validate_args(args)
    configure_hf_cache(args.cache_dir)
    maybe_enable_tf32(args.tf32)
    set_seed(args.seed)

    output_dir = Path(args.output_dir).expanduser().resolve()
    checkpoint_dir = output_dir / args.checkpoint_subdir
    tokenized_dir = output_dir / args.tokenized_subdir
    final_model_dir = output_dir / args.final_model_subdir
    preprocess_summary_path = output_dir / "preprocess_summary.json"
    score_stats_path = output_dir / "score_normalization.json"
    training_summary_path = output_dir / "training_summary.json"
    train_args_path = output_dir / "train_args.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_json_dump(vars(args), train_args_path)

    state = PartialState()

    raw_dataset_dict = load_teacher_dataset(args)
    raw_dataset_dict = split_dataset_if_needed(raw_dataset_dict, args)

    # tokenize 済み dataset は main process のみが作り、他 rank は barrier 後に読む。
    tokenizer_for_prep = AutoTokenizer.from_pretrained(
        args.student_model_id,
        trust_remote_code=args.trust_remote_code,
    )
    maybe_set_pad_token(tokenizer_for_prep)

    tokenized, score_stats, preprocess_summary = prepare_tokenized_dataset(
        raw_dataset_dict=raw_dataset_dict,
        tokenizer=tokenizer_for_prep,
        args=args,
        tokenized_dir=tokenized_dir,
        preprocess_summary_path=preprocess_summary_path,
        score_stats_path=score_stats_path,
        state=state,
    )

    model, tokenizer = create_student_model_and_tokenizer(args)
    collator = RewardRegressionCollator(tokenizer=tokenizer, pad_to_multiple_of=8 if torch.cuda.is_available() else None)
    training_args = build_training_arguments(args, checkpoint_dir=checkpoint_dir)
    compute_metrics = make_compute_metrics(score_stats)

    train_dataset = tokenized["train"]
    eval_dataset = tokenized.get("validation") if training_args.do_eval and "validation" in tokenized else None

    trainer = RegressionDistillationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics if eval_dataset is not None else None,
        loss_type=args.loss_type,
        huber_delta=args.huber_delta,
    )

    resume_checkpoint = resolve_resume_checkpoint(args, checkpoint_dir)
    trainer.train(resume_from_checkpoint=resume_checkpoint)

    if eval_dataset is not None:
        eval_metrics = trainer.evaluate(eval_dataset=eval_dataset)
        if trainer.is_world_process_zero():
            safe_json_dump({k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in eval_metrics.items()}, output_dir / "final_eval_metrics.json")

    if trainer.is_world_process_zero():
        final_model_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(final_model_dir))
        tokenizer.save_pretrained(str(final_model_dir))
        safe_json_dump(score_stats.to_dict(), final_model_dir / "score_normalization.json")
        write_training_summary(
            path=training_summary_path,
            args=args,
            preprocess_summary=preprocess_summary,
            score_stats=score_stats,
            training_args=training_args,
            trainer=trainer,
            tokenized=tokenized,
            resume_checkpoint=resume_checkpoint,
        )


if __name__ == "__main__":
    main()
