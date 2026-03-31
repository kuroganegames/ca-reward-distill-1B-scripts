#!/usr/bin/env python3
"""生徒 Reward Model の予測を教師RMスコア付きデータセット上で評価する。

想定入力:
- 学習済み生徒RM (`train_student_rm_regression.py` の final_model)
- 教師スコア付き dataset (`generate_teacher_dataset.py` の final_dataset)

主な機能:
- 保存済み生徒RMを読み込み、teacher dataset の各行をスコア
- 予測値を `DatasetDict.save_to_disk()` 形式で保存可能
- 全体相関 (Pearson / Spearman / MSE / MAE)
- prompt 単位の順位指標
    - top1 agreement
    - pairwise accuracy (micro / macro)
    - prompt-wise Spearman 平均
- `accelerate launch --multi_gpu ...` で rank ごとに推論し、最後に main process で集約

起動例 (単GPU):
    python evaluate_student_rm_against_teacher.py \
      --model-dir ./student_rm_regression_trial/final_model \
      --teacher-dataset-dir ./teacher_data_trial/final_dataset \
      --output-dir ./student_rm_eval \
      --batch-size 8 \
      --max-length 2048

起動例 (multi-GPU):
    accelerate launch --multi_gpu --num_processes 2 evaluate_student_rm_against_teacher.py \
      --model-dir ./student_rm_regression_trial/final_model \
      --teacher-dataset-dir ./teacher_data_trial/final_dataset \
      --output-dir ./student_rm_eval \
      --batch-size 8 \
      --max-length 2048
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
from accelerate import PartialState
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained student reward model against teacher scores.")
    parser.add_argument("--model-dir", type=str, required=True, help="train_student_rm_regression.py の final_model")
    parser.add_argument("--teacher-dataset-dir", type=str, required=True, help="教師スコア付き dataset の save_to_disk ディレクトリ")
    parser.add_argument("--output-dir", type=str, required=True, help="評価結果の保存先")
    parser.add_argument("--cache-dir", type=str, default=None, help="HF cache root を手動指定")

    parser.add_argument("--split", type=str, default="train", help="DatasetDict の split 名")
    parser.add_argument("--max-samples", type=int, default=None, help="試運転用に先頭N件だけ使う")
    parser.add_argument("--batch-size", type=int, default=8, help="推論バッチサイズ")
    parser.add_argument("--max-length", type=int, default=2048, help="tokenize 時の最大長")
    parser.add_argument("--text-column", type=str, default=None, help="既に全文テキスト列がある場合の列名")
    parser.add_argument("--messages-column", type=str, default="messages")
    parser.add_argument("--response-column", type=str, default="response")
    parser.add_argument("--score-column", type=str, default="teacher_score")
    parser.add_argument(
        "--group-column",
        type=str,
        default="global_prompt_index",
        help="同一 prompt グループの列名。無ければ prompt_hash を試す",
    )
    parser.add_argument("--fallback-group-column", type=str, default="prompt_hash", help="group-column が無いときの代替列")
    parser.add_argument("--sample-id-column", type=str, default="sample_id")

    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="モデル読み込み時の dtype。auto は GPU なら bf16 優先",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="auto",
        choices=["auto", "sdpa", "eager", "flash_attention_2"],
        help="必要なら attention 実装を明示",
    )
    parser.add_argument("--tf32", action="store_true", help="CUDA 環境で TF32 を有効化")

    parser.add_argument(
        "--pairwise-margin",
        type=float,
        default=0.0,
        help="teacher score 差分の絶対値がこの閾値以下のペアは pairwise accuracy から除外",
    )
    parser.add_argument(
        "--save-predictions-dataset",
        action="store_true",
        help="student 予測を加えた dataset を save_to_disk で保存",
    )
    parser.add_argument(
        "--keep-original-columns",
        action="store_true",
        help="予測 dataset 保存時に元列をすべて保持。未指定なら軽量列のみ保存",
    )
    parser.add_argument(
        "--predictions-subdir",
        type=str,
        default="predictions_dataset",
        help="save_to_disk する予測 dataset の subdir 名",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="既存 output-dir があっても上書き続行する",
    )
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


def maybe_set_pad_token(tokenizer) -> None:
    if tokenizer.pad_token_id is not None:
        return
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        return
    tokenizer.add_special_tokens({"pad_token": "<|pad|>"})


def resolve_torch_dtype(dtype_arg: str):
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float16":
        return torch.float16
    if dtype_arg == "bfloat16":
        return torch.bfloat16
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def resolve_attn_implementation(attn_impl: str) -> Optional[str]:
    return None if attn_impl == "auto" else attn_impl


def load_score_normalization(model_dir: Path) -> Optional[Dict[str, Any]]:
    path = model_dir / "score_normalization.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def denormalize_scores(scores_norm: np.ndarray, score_stats: Optional[Mapping[str, Any]]) -> np.ndarray:
    scores_norm = np.asarray(scores_norm, dtype=np.float64)
    if not score_stats:
        return scores_norm.copy()
    if str(score_stats.get("mode", "none")) != "train_zscore":
        return scores_norm.copy()
    mean = float(score_stats.get("mean", 0.0))
    std = float(score_stats.get("std", 1.0))
    return scores_norm * std + mean


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
            f"{args.text_column!r} が無い場合は {args.messages_column!r} と {args.response_column!r} が必要です。"
        )
    full_chat = build_full_chat(messages, response)
    return tokenizer.apply_chat_template(full_chat, tokenize=False, add_generation_prompt=False)


def ensure_dataset(obj: Any, split: str) -> Dataset:
    if isinstance(obj, DatasetDict):
        if split not in obj:
            raise SystemExit(f"入力 dataset に split={split!r} がありません。利用可能: {list(obj.keys())}")
        return obj[split]
    if isinstance(obj, Dataset):
        return obj
    raise SystemExit(f"Unsupported dataset type: {type(obj)!r}")


def load_teacher_dataset(path: str, split: str, max_samples: Optional[int]) -> Dataset:
    ds = ensure_dataset(load_from_disk(path), split)
    if max_samples is not None:
        if max_samples <= 0:
            raise SystemExit("--max-samples は 1 以上にしてください。")
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def select_group_column(ds: Dataset, primary: str, fallback: str) -> str:
    if primary in ds.column_names:
        return primary
    if fallback in ds.column_names:
        return fallback
    raise SystemExit(
        f"group 用の列が見つかりません。{primary!r} も {fallback!r} も存在しません。利用可能: {ds.column_names}"
    )


def shard_bounds(total: int, world_size: int, rank: int) -> Tuple[int, int]:
    start = (total * rank) // world_size
    end = (total * (rank + 1)) // world_size
    return start, end


def iter_batch_slices(total: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, total, batch_size):
        yield start, min(start + batch_size, total)


def dict_of_lists_to_examples(batch: Mapping[str, List[Any]]) -> List[Dict[str, Any]]:
    if not batch:
        return []
    keys = list(batch.keys())
    n = len(batch[keys[0]])
    return [{key: batch[key][i] for key in keys} for i in range(n)]


@torch.inference_mode()
def score_local_dataset(
    ds: Dataset,
    model,
    tokenizer,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray]:
    preds_norm: List[float] = []
    for start, end in tqdm(iter_batch_slices(len(ds), args.batch_size)):
        batch = ds[start:end]
        examples = dict_of_lists_to_examples(batch)
        texts = [build_text_from_example(example, tokenizer=tokenizer, args=args) for example in examples]
        tokenized = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
            add_special_tokens=False,
            padding=True,
        )
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        logits = model(**tokenized).logits.squeeze(-1).float().cpu().numpy()
        preds_norm.extend(float(x) for x in np.atleast_1d(logits))
    pred_norm_arr = np.asarray(preds_norm, dtype=np.float64)
    return pred_norm_arr, pred_norm_arr.copy()


def rankdata_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("rankdata_average expects a 1D array")
    n = values.size
    if n == 0:
        return np.asarray([], dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_values[j] == sorted_values[i]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0  # 1-based ranks
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size <= 1 or y.size <= 1:
        return 0.0
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return 0.0
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx <= 0.0 or sy <= 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size <= 1 or y.size <= 1:
        return 0.0
    return pearson_corr(rankdata_average(x), rankdata_average(y))


def compute_global_metrics(teacher_scores: np.ndarray, student_scores_denorm: np.ndarray, student_scores_norm: np.ndarray) -> Dict[str, Any]:
    teacher_scores = np.asarray(teacher_scores, dtype=np.float64)
    student_scores_denorm = np.asarray(student_scores_denorm, dtype=np.float64)
    student_scores_norm = np.asarray(student_scores_norm, dtype=np.float64)

    return {
        "n": int(teacher_scores.size),
        "teacher_score_mean": float(np.mean(teacher_scores)) if teacher_scores.size else 0.0,
        "teacher_score_std": float(np.std(teacher_scores)) if teacher_scores.size else 0.0,
        "student_score_denorm_mean": float(np.mean(student_scores_denorm)) if student_scores_denorm.size else 0.0,
        "student_score_denorm_std": float(np.std(student_scores_denorm)) if student_scores_denorm.size else 0.0,
        "student_score_norm_mean": float(np.mean(student_scores_norm)) if student_scores_norm.size else 0.0,
        "student_score_norm_std": float(np.std(student_scores_norm)) if student_scores_norm.size else 0.0,
        "mse_denorm": float(np.mean((student_scores_denorm - teacher_scores) ** 2)) if teacher_scores.size else 0.0,
        "mae_denorm": float(np.mean(np.abs(student_scores_denorm - teacher_scores))) if teacher_scores.size else 0.0,
        "pearson_denorm": pearson_corr(student_scores_denorm, teacher_scores),
        "spearman_denorm": spearman_corr(student_scores_denorm, teacher_scores),
    }


def compute_group_metrics(
    group_keys: Sequence[Any],
    teacher_scores: np.ndarray,
    student_scores_denorm: np.ndarray,
    pairwise_margin: float,
) -> Dict[str, Any]:
    groups: Dict[str, List[int]] = {}
    for idx, key in enumerate(group_keys):
        groups.setdefault(str(key), []).append(idx)

    prompt_sizes = [len(v) for v in groups.values()]
    prompts_multi = 0
    top1_total = 0
    top1_correct = 0
    pair_correct_micro = 0.0
    pair_total_micro = 0
    pair_acc_macro_values: List[float] = []
    prompt_spearman_values: List[float] = []

    for _, indices in groups.items():
        if len(indices) < 2:
            continue
        prompts_multi += 1
        t = np.asarray([teacher_scores[i] for i in indices], dtype=np.float64)
        s = np.asarray([student_scores_denorm[i] for i in indices], dtype=np.float64)

        top1_total += 1
        if int(np.argmax(t)) == int(np.argmax(s)):
            top1_correct += 1

        prompt_spearman_values.append(spearman_corr(t, s))

        local_correct = 0.0
        local_total = 0
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                diff_t = float(t[i] - t[j])
                if abs(diff_t) <= pairwise_margin:
                    continue
                local_total += 1
                diff_s = float(s[i] - s[j])
                if diff_t > 0 and diff_s > 0:
                    local_correct += 1.0
                elif diff_t < 0 and diff_s < 0:
                    local_correct += 1.0
                elif diff_s == 0.0:
                    local_correct += 0.5
        if local_total > 0:
            pair_acc_macro_values.append(local_correct / local_total)
            pair_correct_micro += local_correct
            pair_total_micro += local_total

    return {
        "num_prompt_groups": int(len(groups)),
        "num_prompt_groups_multi_candidate": int(prompts_multi),
        "candidate_count_per_prompt_mean": float(np.mean(prompt_sizes)) if prompt_sizes else 0.0,
        "candidate_count_per_prompt_median": float(np.median(prompt_sizes)) if prompt_sizes else 0.0,
        "candidate_count_per_prompt_max": int(max(prompt_sizes)) if prompt_sizes else 0,
        "top1_agreement": float(top1_correct / top1_total) if top1_total > 0 else 0.0,
        "top1_agreement_num_prompts": int(top1_total),
        "pairwise_accuracy_micro": float(pair_correct_micro / pair_total_micro) if pair_total_micro > 0 else 0.0,
        "pairwise_accuracy_micro_num_pairs": int(pair_total_micro),
        "pairwise_accuracy_macro": float(np.mean(pair_acc_macro_values)) if pair_acc_macro_values else 0.0,
        "pairwise_accuracy_macro_num_prompts": int(len(pair_acc_macro_values)),
        "prompt_spearman_mean": float(np.mean(prompt_spearman_values)) if prompt_spearman_values else 0.0,
        "prompt_spearman_median": float(np.median(prompt_spearman_values)) if prompt_spearman_values else 0.0,
        "prompt_spearman_num_prompts": int(len(prompt_spearman_values)),
        "pairwise_margin": float(pairwise_margin),
    }


def save_local_predictions(part_path: Path, start_index: int, pred_norm: np.ndarray) -> None:
    part_path.parent.mkdir(parents=True, exist_ok=True)
    indices = np.arange(start_index, start_index + pred_norm.size, dtype=np.int64)
    np.savez_compressed(
        part_path,
        orig_index=indices,
        student_score_norm=np.asarray(pred_norm, dtype=np.float32),
    )


def load_merged_predictions(parts_dir: Path, total_size: int, world_size: int) -> np.ndarray:
    merged = np.empty(total_size, dtype=np.float64)
    seen = np.zeros(total_size, dtype=np.bool_)
    for rank in range(world_size):
        path = parts_dir / f"rank{rank:05d}.npz"
        if not path.exists():
            raise SystemExit(f"Missing rank output: {path}")
        payload = np.load(path)
        indices = payload["orig_index"].astype(np.int64)
        scores = payload["student_score_norm"].astype(np.float64)
        if indices.size != scores.size:
            raise SystemExit(f"Corrupted part file (size mismatch): {path}")
        merged[indices] = scores
        seen[indices] = True
    if not bool(np.all(seen)):
        missing = np.where(~seen)[0][:10].tolist()
        raise SystemExit(f"Some predictions are missing after merge. missing_example_indices={missing}")
    return merged


def build_predictions_dataset(
    ds: Dataset,
    split: str,
    student_scores_norm: np.ndarray,
    student_scores_denorm: np.ndarray,
    score_column: str,
    group_column: str,
    sample_id_column: str,
    keep_original_columns: bool,
) -> DatasetDict:
    if keep_original_columns:
        out = ds.add_column("student_score_norm", student_scores_norm.tolist())
        out = out.add_column("student_score_denorm", student_scores_denorm.tolist())
        return DatasetDict({split: out})

    keep_columns: List[str] = []
    for column in [sample_id_column, group_column, "prompt_hash", "generator_model", "candidate_index", score_column, "response"]:
        if column in ds.column_names and column not in keep_columns:
            keep_columns.append(column)

    data: Dict[str, List[Any]] = {column: list(ds[column]) for column in keep_columns}
    data["student_score_norm"] = student_scores_norm.tolist()
    data["student_score_denorm"] = student_scores_denorm.tolist()
    out = Dataset.from_dict(data)
    return DatasetDict({split: out})


def main() -> None:
    args = parse_args()
    configure_hf_cache(args.cache_dir)
    maybe_enable_tf32(args.tf32)

    output_dir = Path(args.output_dir).expanduser().resolve()
    parts_dir = output_dir / "parts"
    summary_path = output_dir / "evaluation_summary.json"
    predictions_dir = output_dir / args.predictions_subdir
    run_config_path = output_dir / "eval_args.json"

    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite_output:
        # 既存 parts への誤上書きを防ぐ。空でない場合のみ止める。
        existing = {p.name for p in output_dir.iterdir()}
        allowed = {"parts", "evaluation_summary.json", "eval_args.json", args.predictions_subdir}
        if existing - allowed:
            raise SystemExit(f"output_dir が既に存在しており空ではありません: {output_dir}. --overwrite-output を付けてください。")
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_json_dump(vars(args), run_config_path)

    state = PartialState()
    model_dir = Path(args.model_dir).expanduser().resolve()
    teacher_ds = load_teacher_dataset(args.teacher_dataset_dir, split=args.split, max_samples=args.max_samples)
    group_column = select_group_column(teacher_ds, args.group_column, args.fallback_group_column)

    rank_start, rank_end = shard_bounds(len(teacher_ds), state.num_processes, state.process_index)
    local_ds = teacher_ds.select(range(rank_start, rank_end)) if rank_end > rank_start else teacher_ds.select([])

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    maybe_set_pad_token(tokenizer)

    model_kwargs: Dict[str, Any] = {
        "torch_dtype": resolve_torch_dtype(args.torch_dtype),
    }
    attn_impl = resolve_attn_implementation(args.attn_implementation)
    if attn_impl is not None:
        model_kwargs["attn_implementation"] = attn_impl

    model = AutoModelForSequenceClassification.from_pretrained(model_dir, **model_kwargs)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    model.to(state.device)

    local_pred_norm, _ = score_local_dataset(
        ds=local_ds,
        model=model,
        tokenizer=tokenizer,
        device=state.device,
        args=args,
    )
    save_local_predictions(parts_dir / f"rank{state.process_index:05d}.npz", start_index=rank_start, pred_norm=local_pred_norm)

    state.wait_for_everyone()

    if state.is_main_process:
        score_stats = load_score_normalization(model_dir)
        pred_norm = load_merged_predictions(parts_dir=parts_dir, total_size=len(teacher_ds), world_size=state.num_processes)
        pred_denorm = denormalize_scores(pred_norm, score_stats)
        teacher_scores = np.asarray(teacher_ds[args.score_column], dtype=np.float64)
        group_keys = list(teacher_ds[group_column])

        global_metrics = compute_global_metrics(
            teacher_scores=teacher_scores,
            student_scores_denorm=pred_denorm,
            student_scores_norm=pred_norm,
        )
        group_metrics = compute_group_metrics(
            group_keys=group_keys,
            teacher_scores=teacher_scores,
            student_scores_denorm=pred_denorm,
            pairwise_margin=args.pairwise_margin,
        )

        summary = {
            "model_dir": str(model_dir),
            "teacher_dataset_dir": str(Path(args.teacher_dataset_dir).expanduser().resolve()),
            "split": args.split,
            "num_processes": int(state.num_processes),
            "group_column": group_column,
            "score_column": args.score_column,
            "score_normalization": score_stats,
            "global_metrics": global_metrics,
            "group_metrics": group_metrics,
        }
        safe_json_dump(summary, summary_path)

        if args.save_predictions_dataset:
            predictions = build_predictions_dataset(
                ds=teacher_ds,
                split=args.split,
                student_scores_norm=pred_norm,
                student_scores_denorm=pred_denorm,
                score_column=args.score_column,
                group_column=group_column,
                sample_id_column=args.sample_id_column,
                keep_original_columns=args.keep_original_columns,
            )
            if predictions_dir.exists():
                import shutil
                shutil.rmtree(predictions_dir)
            predictions.save_to_disk(str(predictions_dir))

        print(json.dumps(summary, ensure_ascii=False, indent=2))

    state.wait_for_everyone()


if __name__ == "__main__":
    main()
