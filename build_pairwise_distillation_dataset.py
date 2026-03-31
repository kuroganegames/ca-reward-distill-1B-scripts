#!/usr/bin/env python3
"""保存済み教師データセットから pairwise 蒸留用の (chosen, rejected) データセットを作る。

主な要件:
- 入力: `generate_teacher_dataset.py` が `save_to_disk()` した教師データセット
- 出力: `datasets.load_from_disk()` で再読込できる pairwise Dataset / DatasetDict
- 中断再開: parquet の part 保存 + progress JSON + manifest JSONL
- 省ストレージ: 既定では prompt は `messages` 1本、chosen/rejected は応答文字列のみ
  （必要なら `--store-conversation-columns` で full conversation も追加保存できる）

想定起動例:
    python build_pairwise_distillation_dataset.py \
      --teacher-dataset-dir ./teacher_data_trial/final_dataset \
      --output-dir ./pairwise_trial \
      --cache-dir /data/hf_cache \
      --pairing-strategy best_vs_rest \
      --min-score-margin 0.0 \
      --max-pairs-per-prompt 4 \
      --val-ratio 0.02
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import re
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
except ImportError as exc:  # pragma: no cover
    raise SystemExit("datasets が見つかりません。`pip install -U datasets pyarrow` を実行してください。") from exc

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pa = None
    pq = None


DEFAULT_FINAL_SUBDIR = "final_dataset"
PART_RE = re.compile(r"^part_(?P<part_id>\d{6})\.parquet$")

REQUIRED_COLUMNS = {
    "messages",
    "response",
    "response_hash",
    "teacher_score",
    "teacher_model",
    "generator_model",
    "generator_key",
    "generator_model_index",
    "candidate_index",
    "sample_id",
    "prompt_hash",
    "prompt_raw",
    "global_prompt_index",
}

PROMPT_METADATA_CANDIDATES = [
    "prompt_raw",
    "prompt_language",
    "answer_language",
    "source_alias",
    "source_dataset",
    "source_split",
    "source_config_name",
    "source_data_dir",
    "source_example_id",
    "source_row_number",
    "wrapped_for_japanese_answer",
    "num_prompt_messages",
    "prompt_char_len",
    "local_prompt_index",
    "rank",
]


@dataclass(frozen=True)
class RunPaths:
    root: Path
    state_dir: Path
    parts_dir: Path
    train_parts_dir: Path
    validation_parts_dir: Path
    final_dataset_dir: Path
    config_path: Path
    summary_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build pairwise (chosen, rejected) dataset from teacher-scored rows.")
    parser.add_argument("--teacher-dataset-dir", type=str, required=True, help="教師データセットの save_to_disk ディレクトリ")
    parser.add_argument("--output-dir", type=str, required=True, help="作業用 root ディレクトリ")
    parser.add_argument("--cache-dir", type=str, default=None, help="Hugging Face cache root。手動指定可")
    parser.add_argument("--train-split", type=str, default="train", help="入力 DatasetDict の split 名")
    parser.add_argument("--final-dataset-subdir", type=str, default=DEFAULT_FINAL_SUBDIR, help="output-dir 配下の最終 save_to_disk ディレクトリ名")

    parser.add_argument(
        "--pairing-strategy",
        type=str,
        default="best_vs_rest",
        choices=["best_vs_rest", "top_bottom", "all_pairs", "adjacent"],
        help="同一 prompt 内の候補から pair を作る方式",
    )
    parser.add_argument("--min-score-margin", type=float, default=0.0, help="chosen_score - rejected_score の最小差")
    parser.add_argument(
        "--max-pairs-per-prompt",
        type=int,
        default=None,
        help="1 prompt あたり最大何 pair まで保存するか。None なら無制限",
    )
    parser.add_argument(
        "--min-candidates-per-prompt",
        type=int,
        default=2,
        help="この数以上のユニーク候補がある prompt のみ pair 化する",
    )
    parser.add_argument("--drop-empty-responses", dest="drop_empty_responses", action="store_true", help="空応答を落とす (既定)")
    parser.add_argument("--keep-empty-responses", dest="drop_empty_responses", action="store_false", help="空応答も残す")
    parser.set_defaults(drop_empty_responses=True)
    parser.add_argument("--dedupe-response-hash", dest="dedupe_response_hash", action="store_true", help="同一 prompt 内の重複 response_hash を落とす (既定)")
    parser.add_argument("--keep-duplicate-responses", dest="dedupe_response_hash", action="store_false", help="重複応答も残す")
    parser.set_defaults(dedupe_response_hash=True)
    parser.add_argument(
        "--store-conversation-columns",
        action="store_true",
        help="`chosen_messages` / `rejected_messages` の full conversation 列も保存する",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.0,
        help="prompt 単位で validation に振り分ける比率。0.0 なら train のみ",
    )
    parser.add_argument("--split-seed", type=int, default=42, help="train/validation のハッシュ分割用 seed")

    parser.add_argument("--read-batch-size", type=int, default=50000, help="ソート後に何行ずつ読み出すか")
    parser.add_argument("--flush-pair-count", type=int, default=50000, help="何 pair たまったら parquet part として flush するか")
    parser.add_argument("--parquet-compression", type=str, default="zstd", help="中間 parquet の圧縮方式")
    parser.add_argument("--save-max-shard-size", type=str, default="2GB", help="final_dataset.save_to_disk の max_shard_size")
    parser.add_argument(
        "--sort-indices-cache-file",
        type=str,
        default=None,
        help="Dataset.sort() の indices cache file。未指定なら output-dir/state/sorted_indices.arrow",
    )
    parser.add_argument("--max-prompt-groups", type=int, default=None, help="先頭 N prompt group のみ処理。試運転用")
    parser.add_argument("--finalize-only", action="store_true", help="pair 生成は行わず、既存 parquet part を最終 dataset にまとめ直す")
    parser.add_argument("--allow-config-mismatch", action="store_true", help="既存 config と引数が違っても resume を強行する")
    parser.add_argument("--cleanup-sort-cache", action="store_true", help="完了後に sort 用 index cache を削除する")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not (0.0 <= args.val_ratio < 1.0):
        raise SystemExit("--val-ratio は 0.0 以上 1.0 未満にしてください。")
    if args.min_candidates_per_prompt < 2:
        raise SystemExit("--min-candidates-per-prompt は 2 以上にしてください。")
    if args.max_pairs_per_prompt is not None and args.max_pairs_per_prompt <= 0:
        raise SystemExit("--max-pairs-per-prompt は 1 以上にしてください。")
    if args.read_batch_size <= 0:
        raise SystemExit("--read-batch-size は 1 以上にしてください。")
    if args.flush_pair_count <= 0:
        raise SystemExit("--flush-pair-count は 1 以上にしてください。")
    if args.max_prompt_groups is not None and args.max_prompt_groups <= 0:
        raise SystemExit("--max-prompt-groups は 1 以上にしてください。")



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



def build_paths(output_dir: str, final_dataset_subdir: str) -> RunPaths:
    root = Path(output_dir).expanduser().resolve()
    parts_dir = root / "parts"
    return RunPaths(
        root=root,
        state_dir=root / "state",
        parts_dir=parts_dir,
        train_parts_dir=parts_dir / "train",
        validation_parts_dir=parts_dir / "validation",
        final_dataset_dir=root / final_dataset_subdir,
        config_path=root / "config.json",
        summary_path=root / "final_summary.json",
    )



def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())



def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding=encoding) as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)



def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")



def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())



def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def load_teacher_dataset(teacher_dataset_dir: str, train_split: str):
    ds_any = load_from_disk(str(Path(teacher_dataset_dir).expanduser().resolve()))
    if isinstance(ds_any, DatasetDict):
        if train_split not in ds_any:
            raise SystemExit(f"入力 DatasetDict に split={train_split!r} がありません。利用可能: {list(ds_any.keys())}")
        ds = ds_any[train_split]
    else:
        ds = ds_any

    missing = REQUIRED_COLUMNS - set(ds.column_names)
    if missing:
        raise SystemExit(f"入力教師データに必須列が足りません: {sorted(missing)}")
    return ds



def gather_prompt_metadata_columns(ds: Any) -> List[str]:
    return [c for c in PROMPT_METADATA_CANDIDATES if c in ds.column_names]



def prepare_run_config(args: argparse.Namespace, input_rows: int) -> Dict[str, Any]:
    return {
        "created_at": now_iso(),
        "teacher_dataset_dir": str(Path(args.teacher_dataset_dir).expanduser().resolve()),
        "output_dir": str(Path(args.output_dir).expanduser().resolve()),
        "cache_dir": str(Path(args.cache_dir).expanduser().resolve()) if args.cache_dir else None,
        "train_split": args.train_split,
        "final_dataset_subdir": args.final_dataset_subdir,
        "pairing_strategy": args.pairing_strategy,
        "min_score_margin": args.min_score_margin,
        "max_pairs_per_prompt": args.max_pairs_per_prompt,
        "min_candidates_per_prompt": args.min_candidates_per_prompt,
        "drop_empty_responses": bool(args.drop_empty_responses),
        "dedupe_response_hash": bool(args.dedupe_response_hash),
        "store_conversation_columns": bool(args.store_conversation_columns),
        "val_ratio": args.val_ratio,
        "split_seed": args.split_seed,
        "read_batch_size": args.read_batch_size,
        "flush_pair_count": args.flush_pair_count,
        "parquet_compression": args.parquet_compression,
        "save_max_shard_size": args.save_max_shard_size,
        "sort_indices_cache_file": args.sort_indices_cache_file,
        "max_prompt_groups": args.max_prompt_groups,
        "input_rows": input_rows,
        "command": sys.argv,
        "script_name": Path(sys.argv[0]).name,
    }



def init_or_validate_run_config(
    paths: RunPaths,
    args: argparse.Namespace,
    input_rows: int,
    allow_config_mismatch: bool,
) -> Dict[str, Any]:
    current = prepare_run_config(args=args, input_rows=input_rows)
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.state_dir.mkdir(parents=True, exist_ok=True)
    paths.train_parts_dir.mkdir(parents=True, exist_ok=True)
    paths.validation_parts_dir.mkdir(parents=True, exist_ok=True)

    if not paths.config_path.exists():
        atomic_write_json(paths.config_path, current)
        return current

    existing = read_json(paths.config_path)
    compare_keys = [
        "teacher_dataset_dir",
        "train_split",
        "pairing_strategy",
        "min_score_margin",
        "max_pairs_per_prompt",
        "min_candidates_per_prompt",
        "drop_empty_responses",
        "dedupe_response_hash",
        "store_conversation_columns",
        "val_ratio",
        "split_seed",
        "max_prompt_groups",
        "input_rows",
    ]
    mismatches = []
    for key in compare_keys:
        if existing.get(key) != current.get(key):
            mismatches.append(f"{key}: existing={existing.get(key)!r} current={current.get(key)!r}")

    if mismatches and not allow_config_mismatch:
        mismatch_text = "\n".join(mismatches[:20])
        raise SystemExit(
            "既存 output-dir の config.json と今回の引数が一致しません。\n"
            f"{mismatch_text}\n"
            "別の output-dir を使うか、既存ディレクトリを削除するか、"
            "整合性を理解した上で --allow-config-mismatch を付けてください。"
        )
    if mismatches and allow_config_mismatch:
        print("[WARN] config mismatch を許可して続行します。", file=sys.stderr)
    return existing



def progress_path(paths: RunPaths) -> Path:
    return paths.state_dir / "progress.json"



def manifest_path(paths: RunPaths, split_name: str) -> Path:
    return paths.state_dir / f"manifest_{split_name}.jsonl"



def commit_log_path(paths: RunPaths) -> Path:
    return paths.state_dir / "commit_log.jsonl"



def error_log_path(paths: RunPaths) -> Path:
    return paths.state_dir / "last_error.txt"



def write_progress(paths: RunPaths, payload: Dict[str, Any]) -> None:
    payload = dict(payload)
    payload["updated_at"] = now_iso()
    atomic_write_json(progress_path(paths), payload)



def write_error(paths: RunPaths, exc: BaseException) -> None:
    text = f"[{now_iso()}] {type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
    atomic_write_text(error_log_path(paths), text)



def write_rows_to_parquet(rows: List[Dict[str, Any]], path: Path, compression: str) -> int:
    if not rows:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")

    ds = Dataset.from_list(rows)
    if hasattr(ds, "to_parquet"):
        try:
            bytes_written = ds.to_parquet(str(tmp), compression=compression)
            os.replace(tmp, path)
            return int(bytes_written)
        except Exception:
            if pa is None or pq is None:
                raise

    if pa is None or pq is None:
        raise RuntimeError(
            "datasets.Dataset.to_parquet が利用できず、pyarrow も見つかりません。"
            " `pip install -U pyarrow` を実行してください。"
        )

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, tmp, compression=compression)
    os.replace(tmp, path)
    return int(path.stat().st_size)



def scan_existing_parts(parts_dir: Path) -> List[Path]:
    if not parts_dir.exists():
        return []
    paths = []
    for path in sorted(parts_dir.glob("part_*.parquet")):
        if PART_RE.match(path.name):
            paths.append(path)
    return paths



def next_part_id(parts_dir: Path) -> int:
    part_paths = scan_existing_parts(parts_dir)
    if not part_paths:
        return 0
    last = part_paths[-1]
    match = PART_RE.match(last.name)
    if not match:
        return 0
    return int(match.group("part_id")) + 1



def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items



def load_manifest_entries(paths: RunPaths, split_name: str) -> List[Dict[str, Any]]:
    entries = read_jsonl(manifest_path(paths, split_name))
    valid_entries: List[Dict[str, Any]] = []
    for entry in entries:
        part_path = Path(entry["path"])
        if part_path.exists():
            valid_entries.append(entry)
    valid_entries.sort(key=lambda x: int(x.get("part_id", 0)))
    return valid_entries



def split_name_for_prompt(prompt_hash: str, val_ratio: float, split_seed: int) -> str:
    if val_ratio <= 0.0:
        return "train"
    digest = hashlib.sha1(f"{split_seed}:{prompt_hash}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big") / float(2**64)
    return "validation" if value < val_ratio else "train"



def deep_copy_messages(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    copied: List[Dict[str, Any]] = []
    for message in messages:
        copied.append({
            "role": str(message.get("role", "user")),
            "content": message.get("content", ""),
        })
    return copied



def build_full_conversation(messages: Sequence[Dict[str, Any]], response: str) -> List[Dict[str, Any]]:
    conversation = deep_copy_messages(messages)
    conversation.append({"role": "assistant", "content": response})
    return conversation



def normalize_response(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, list):
        return "\n".join(str(x) for x in text if x is not None).strip()
    return str(text).strip()



def dedupe_group_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    dedupe_response_hash: bool,
    drop_empty_responses: bool,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen_response_hashes = set()
    for row in rows:
        response = normalize_response(row.get("response"))
        if drop_empty_responses and not response:
            continue
        if dedupe_response_hash:
            response_hash = str(row.get("response_hash", ""))
            if response_hash in seen_response_hashes:
                continue
            seen_response_hashes.add(response_hash)
        normalized_row = dict(row)
        normalized_row["response"] = response
        out.append(normalized_row)
    return out



def enumerate_candidate_pairs(
    rows_sorted_desc: Sequence[Dict[str, Any]],
    strategy: str,
    min_score_margin: float,
    max_pairs_per_prompt: Optional[int],
) -> List[Tuple[Dict[str, Any], Dict[str, Any], float]]:
    n = len(rows_sorted_desc)
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any], float]] = []
    if n < 2:
        return pairs

    def maybe_add(chosen_idx: int, rejected_idx: int) -> None:
        if not (0 <= chosen_idx < n and 0 <= rejected_idx < n):
            return
        if chosen_idx == rejected_idx:
            return
        chosen = rows_sorted_desc[chosen_idx]
        rejected = rows_sorted_desc[rejected_idx]
        margin = float(chosen["teacher_score"]) - float(rejected["teacher_score"])
        if margin <= 0:
            return
        if margin < min_score_margin:
            return
        pairs.append((chosen, rejected, margin))

    if strategy == "top_bottom":
        maybe_add(0, n - 1)
    elif strategy == "adjacent":
        for i in range(n - 1):
            maybe_add(i, i + 1)
    elif strategy == "best_vs_rest":
        # スコア差が大きい順で top を各下位候補と比較する。
        for j in range(n - 1, 0, -1):
            maybe_add(0, j)
    elif strategy == "all_pairs":
        temp_pairs: List[Tuple[Dict[str, Any], Dict[str, Any], float]] = []
        for i in range(n - 1):
            for j in range(i + 1, n):
                chosen = rows_sorted_desc[i]
                rejected = rows_sorted_desc[j]
                margin = float(chosen["teacher_score"]) - float(rejected["teacher_score"])
                if margin <= 0 or margin < min_score_margin:
                    continue
                temp_pairs.append((chosen, rejected, margin))
        temp_pairs.sort(key=lambda x: (-x[2], x[0]["sample_id"], x[1]["sample_id"]))
        pairs = temp_pairs
    else:
        raise ValueError(f"Unknown pairing strategy: {strategy}")

    if strategy != "all_pairs":
        pairs.sort(key=lambda x: (-x[2], x[0]["sample_id"], x[1]["sample_id"]))

    if max_pairs_per_prompt is not None:
        pairs = pairs[:max_pairs_per_prompt]
    return pairs



def build_pair_row(
    *,
    prompt_rows_sorted_desc: Sequence[Dict[str, Any]],
    chosen_row: Dict[str, Any],
    rejected_row: Dict[str, Any],
    score_margin: float,
    pair_rank_within_prompt: int,
    strategy: str,
    split_name: str,
    prompt_metadata_columns: Sequence[str],
    store_conversation_columns: bool,
    num_candidates_raw: int,
) -> Dict[str, Any]:
    prompt_row = prompt_rows_sorted_desc[0]
    messages = deep_copy_messages(prompt_row["messages"])
    prompt_hash = str(prompt_row["prompt_hash"])
    pair_basis = f"{prompt_hash}|{chosen_row['sample_id']}|{rejected_row['sample_id']}|{strategy}"
    pair_id = hashlib.sha1(pair_basis.encode("utf-8")).hexdigest()

    row: Dict[str, Any] = {
        "pair_id": pair_id,
        "split": split_name,
        "pair_strategy": strategy,
        "pair_rank_within_prompt": int(pair_rank_within_prompt),
        "messages": messages,
        "chosen": str(chosen_row["response"]),
        "rejected": str(rejected_row["response"]),
        "prompt_hash": prompt_hash,
        "global_prompt_index": int(prompt_row["global_prompt_index"]),
        "teacher_model": str(prompt_row["teacher_model"]),
        "chosen_sample_id": str(chosen_row["sample_id"]),
        "rejected_sample_id": str(rejected_row["sample_id"]),
        "chosen_score": float(chosen_row["teacher_score"]),
        "rejected_score": float(rejected_row["teacher_score"]),
        "score_margin": float(score_margin),
        "chosen_response_hash": str(chosen_row["response_hash"]),
        "rejected_response_hash": str(rejected_row["response_hash"]),
        "chosen_generator_model": str(chosen_row["generator_model"]),
        "rejected_generator_model": str(rejected_row["generator_model"]),
        "chosen_generator_key": str(chosen_row["generator_key"]),
        "rejected_generator_key": str(rejected_row["generator_key"]),
        "chosen_generator_model_index": int(chosen_row["generator_model_index"]),
        "rejected_generator_model_index": int(rejected_row["generator_model_index"]),
        "chosen_candidate_index": int(chosen_row["candidate_index"]),
        "rejected_candidate_index": int(rejected_row["candidate_index"]),
        "num_candidates_raw": int(num_candidates_raw),
        "num_candidates_after_filter": int(len(prompt_rows_sorted_desc)),
    }
    for col in prompt_metadata_columns:
        if col == "prompt_raw":
            row[col] = str(prompt_row.get(col, ""))
        else:
            row[col] = prompt_row.get(col)

    if store_conversation_columns:
        row["chosen_messages"] = build_full_conversation(messages, str(chosen_row["response"]))
        row["rejected_messages"] = build_full_conversation(messages, str(rejected_row["response"]))

    return row



def summarize_group_stats(unique_candidate_count: int, num_pairs: int) -> Dict[str, int]:
    return {
        "unique_candidate_count": int(unique_candidate_count),
        "num_pairs": int(num_pairs),
    }



def sort_teacher_dataset(ds: Any, sort_indices_cache_file: Path):
    sort_indices_cache_file.parent.mkdir(parents=True, exist_ok=True)
    return ds.sort(
        ["global_prompt_index", "teacher_score", "sample_id"],
        reverse=[False, True, False],
        indices_cache_file_name=str(sort_indices_cache_file),
        writer_batch_size=1000,
    )



def load_or_init_progress(paths: RunPaths) -> Dict[str, Any]:
    base = {
        "status": "initialized",
        "committed_sorted_rows": 0,
        "processed_prompt_groups": 0,
        "skipped_prompt_groups": 0,
        "train_pairs": 0,
        "validation_pairs": 0,
        "next_part_id": {
            "train": 0,
            "validation": 0,
        },
        "last_global_prompt_index": None,
        "finalized": False,
        "started_at": now_iso(),
    }

    if progress_path(paths).exists():
        base.update(read_json(progress_path(paths)))

    commit_entries = read_jsonl(commit_log_path(paths))
    if commit_entries:
        # commit_log を source of truth とする。part file が存在しても commit されていないものは orphan とみなす。
        base.update(commit_entries[-1])
    return base



def finalize_dataset(paths: RunPaths, save_max_shard_size: str) -> Dict[str, Any]:
    data_files: Dict[str, List[str]] = {}
    commit_entries = read_jsonl(commit_log_path(paths))
    if not commit_entries:
        raise SystemExit("finalize を中止しました。commit_log.jsonl が空で、コミット済み part を特定できません。")

    committed_parts: Dict[str, Dict[int, str]] = {"train": {}, "validation": {}}
    for entry in commit_entries:
        written_parts = entry.get("written_parts", {})
        for split_name in ("train", "validation"):
            for part in written_parts.get(split_name, []):
                part_id = int(part["part_id"])
                part_path = str(part["path"])
                if Path(part_path).exists():
                    committed_parts[split_name][part_id] = part_path

    train_parts = [Path(committed_parts["train"][k]) for k in sorted(committed_parts["train"])]
    validation_parts = [Path(committed_parts["validation"][k]) for k in sorted(committed_parts["validation"])]
    if train_parts:
        data_files["train"] = [str(p) for p in train_parts]
    if validation_parts:
        data_files["validation"] = [str(p) for p in validation_parts]
    if not data_files:
        raise SystemExit("finalize を中止しました。コミット済み parquet part が 1 つもありません。")

    ds_dict = load_dataset("parquet", data_files=data_files)

    if paths.final_dataset_dir.exists():
        shutil.rmtree(paths.final_dataset_dir)
    ds_dict.save_to_disk(str(paths.final_dataset_dir), max_shard_size=save_max_shard_size)

    summary = {
        "created_at": now_iso(),
        "final_dataset_dir": str(paths.final_dataset_dir),
        "splits": {},
        "total_rows": 0,
        "num_train_parts": len(train_parts),
        "num_validation_parts": len(validation_parts),
    }
    total_rows = 0
    for split_name, ds in ds_dict.items():
        split_rows = int(len(ds))
        total_rows += split_rows
        summary["splits"][split_name] = {
            "rows": split_rows,
            "columns": list(ds.column_names),
        }
    summary["total_rows"] = total_rows
    atomic_write_json(paths.summary_path, summary)
    return summary



def main() -> None:
    args = parse_args()
    validate_args(args)
    cache_dir = configure_hf_cache(args.cache_dir)
    paths = build_paths(args.output_dir, args.final_dataset_subdir)

    try:
        teacher_ds = load_teacher_dataset(args.teacher_dataset_dir, args.train_split)
        prompt_metadata_columns = gather_prompt_metadata_columns(teacher_ds)
        run_config = init_or_validate_run_config(
            paths=paths,
            args=args,
            input_rows=len(teacher_ds),
            allow_config_mismatch=args.allow_config_mismatch,
        )

        sort_indices_cache_file = (
            Path(args.sort_indices_cache_file).expanduser().resolve()
            if args.sort_indices_cache_file
            else paths.state_dir / "sorted_indices.arrow"
        )

        if args.finalize_only:
            summary = finalize_dataset(paths=paths, save_max_shard_size=args.save_max_shard_size)
            progress = load_or_init_progress(paths)
            progress["status"] = "finalized"
            progress["finalized"] = True
            progress["finalized_at"] = now_iso()
            write_progress(paths, progress)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return

        print(
            f"[INFO] sorting teacher dataset by global_prompt_index, teacher_score -> {sort_indices_cache_file}",
            file=sys.stderr,
        )
        sorted_ds = sort_teacher_dataset(teacher_ds, sort_indices_cache_file)
        total_sorted_rows = len(sorted_ds)
        del teacher_ds
        gc.collect()

        progress = load_or_init_progress(paths)
        committed_sorted_rows = int(progress.get("committed_sorted_rows", 0))
        processed_prompt_groups = int(progress.get("processed_prompt_groups", 0))
        skipped_prompt_groups = int(progress.get("skipped_prompt_groups", 0))
        total_pairs = {
            "train": int(progress.get("train_pairs", 0)),
            "validation": int(progress.get("validation_pairs", 0)),
        }
        next_part_ids = dict(progress.get("next_part_id", {}))
        next_part_ids.setdefault("train", next_part_id(paths.train_parts_dir))
        next_part_ids.setdefault("validation", next_part_id(paths.validation_parts_dir))

        buffers: Dict[str, List[Dict[str, Any]]] = {"train": [], "validation": []}
        current_group_rows: List[Dict[str, Any]] = []
        current_global_prompt_index: Optional[int] = None
        current_prompt_hash: Optional[str] = None

        uncommitted_sorted_rows = committed_sorted_rows
        uncommitted_processed_groups = processed_prompt_groups
        uncommitted_skipped_groups = skipped_prompt_groups
        last_global_prompt_index = progress.get("last_global_prompt_index")

        def flush_buffers(force: bool = False) -> None:
            nonlocal committed_sorted_rows
            nonlocal processed_prompt_groups
            nonlocal skipped_prompt_groups
            nonlocal total_pairs
            nonlocal next_part_ids
            total_buffered = len(buffers["train"]) + len(buffers["validation"])
            if (not force) and total_buffered < args.flush_pair_count:
                return
            if total_buffered == 0:
                # pair を伴わないスキップ group だけ進んだ場合でも progress は更新する。
                if committed_sorted_rows != uncommitted_sorted_rows or processed_prompt_groups != uncommitted_processed_groups or skipped_prompt_groups != uncommitted_skipped_groups:
                    committed_sorted_rows = uncommitted_sorted_rows
                    processed_prompt_groups = uncommitted_processed_groups
                    skipped_prompt_groups = uncommitted_skipped_groups
                    progress_payload = {
                        "status": "running",
                        "committed_sorted_rows": committed_sorted_rows,
                        "processed_prompt_groups": processed_prompt_groups,
                        "skipped_prompt_groups": skipped_prompt_groups,
                        "train_pairs": total_pairs["train"],
                        "validation_pairs": total_pairs["validation"],
                        "next_part_id": next_part_ids,
                        "last_global_prompt_index": last_global_prompt_index,
                        "finalized": False,
                        "sort_indices_cache_file": str(sort_indices_cache_file),
                        "total_sorted_rows": total_sorted_rows,
                    }
                    append_jsonl(commit_log_path(paths), dict(progress_payload, written_parts={"train": [], "validation": []}, committed_at=now_iso()))
                    write_progress(paths, progress_payload)
                return

            written_parts: Dict[str, List[Dict[str, Any]]] = {"train": [], "validation": []}
            for split_name, part_dir in (("train", paths.train_parts_dir), ("validation", paths.validation_parts_dir)):
                rows = buffers[split_name]
                if not rows:
                    continue
                part_id = int(next_part_ids[split_name])
                part_path = part_dir / f"part_{part_id:06d}.parquet"
                size_bytes = write_rows_to_parquet(rows=rows, path=part_path, compression=args.parquet_compression)
                part_record = {
                    "part_id": part_id,
                    "path": str(part_path),
                    "split": split_name,
                    "rows": len(rows),
                    "size_bytes": size_bytes,
                    "created_at": now_iso(),
                }
                append_jsonl(manifest_path(paths, split_name), part_record)
                written_parts[split_name].append(part_record)
                total_pairs[split_name] += len(rows)
                next_part_ids[split_name] = part_id + 1
                buffers[split_name] = []

            committed_sorted_rows = uncommitted_sorted_rows
            processed_prompt_groups = uncommitted_processed_groups
            skipped_prompt_groups = uncommitted_skipped_groups
            progress_payload = {
                "status": "running",
                "committed_sorted_rows": committed_sorted_rows,
                "processed_prompt_groups": processed_prompt_groups,
                "skipped_prompt_groups": skipped_prompt_groups,
                "train_pairs": total_pairs["train"],
                "validation_pairs": total_pairs["validation"],
                "next_part_id": next_part_ids,
                "last_global_prompt_index": last_global_prompt_index,
                "finalized": False,
                "sort_indices_cache_file": str(sort_indices_cache_file),
                "total_sorted_rows": total_sorted_rows,
            }
            append_jsonl(commit_log_path(paths), dict(progress_payload, written_parts=written_parts, committed_at=now_iso()))
            write_progress(paths, progress_payload)

        def process_completed_group(group_rows: Sequence[Dict[str, Any]]) -> None:
            nonlocal uncommitted_processed_groups
            nonlocal uncommitted_skipped_groups
            nonlocal last_global_prompt_index
            if not group_rows:
                return
            deduped_rows = dedupe_group_rows(
                group_rows,
                dedupe_response_hash=args.dedupe_response_hash,
                drop_empty_responses=args.drop_empty_responses,
            )
            if len(deduped_rows) < args.min_candidates_per_prompt:
                uncommitted_skipped_groups += 1
                last_global_prompt_index = int(group_rows[0]["global_prompt_index"])
                return
            pairs = enumerate_candidate_pairs(
                rows_sorted_desc=deduped_rows,
                strategy=args.pairing_strategy,
                min_score_margin=args.min_score_margin,
                max_pairs_per_prompt=args.max_pairs_per_prompt,
            )
            if not pairs:
                uncommitted_skipped_groups += 1
                last_global_prompt_index = int(group_rows[0]["global_prompt_index"])
                return
            split_name = split_name_for_prompt(
                prompt_hash=str(group_rows[0]["prompt_hash"]),
                val_ratio=args.val_ratio,
                split_seed=args.split_seed,
            )
            pair_rows = []
            for pair_rank, (chosen_row, rejected_row, margin) in enumerate(pairs):
                pair_rows.append(
                    build_pair_row(
                        prompt_rows_sorted_desc=deduped_rows,
                        chosen_row=chosen_row,
                        rejected_row=rejected_row,
                        score_margin=margin,
                        pair_rank_within_prompt=pair_rank,
                        strategy=args.pairing_strategy,
                        split_name=split_name,
                        prompt_metadata_columns=prompt_metadata_columns,
                        store_conversation_columns=args.store_conversation_columns,
                        num_candidates_raw=len(group_rows),
                    )
                )
            buffers[split_name].extend(pair_rows)
            uncommitted_processed_groups += 1
            last_global_prompt_index = int(group_rows[0]["global_prompt_index"])

        if committed_sorted_rows > 0:
            print(
                f"[INFO] resuming from sorted row {committed_sorted_rows}/{total_sorted_rows}",
                file=sys.stderr,
            )

        if args.max_prompt_groups is not None and (processed_prompt_groups + skipped_prompt_groups) >= args.max_prompt_groups:
            print(
                f"[INFO] already reached max_prompt_groups={args.max_prompt_groups}; moving to finalize",
                file=sys.stderr,
            )
        else:
            for batch_start in range(committed_sorted_rows, total_sorted_rows, args.read_batch_size):
                if args.max_prompt_groups is not None and (uncommitted_processed_groups + uncommitted_skipped_groups) >= args.max_prompt_groups:
                    break
                batch_end = min(batch_start + args.read_batch_size, total_sorted_rows)
                batch = sorted_ds[batch_start:batch_end]
                batch_len = len(batch["global_prompt_index"])
                columns = list(batch.keys())

                for offset in range(batch_len):
                    row = {col: batch[col][offset] for col in columns}
                    row_global_prompt_index = int(row["global_prompt_index"])
                    row_prompt_hash = str(row["prompt_hash"])

                    if current_global_prompt_index is None:
                        current_global_prompt_index = row_global_prompt_index
                        current_prompt_hash = row_prompt_hash
                        current_group_rows = [row]
                        continue

                    if row_global_prompt_index == current_global_prompt_index:
                        current_group_rows.append(row)
                        continue

                    # 前 group を確定してから新しい group を始める。
                    process_completed_group(current_group_rows)
                    current_group_rows = [row]
                    current_global_prompt_index = row_global_prompt_index
                    current_prompt_hash = row_prompt_hash
                    uncommitted_sorted_rows = batch_start + offset
                    if args.max_prompt_groups is not None and (uncommitted_processed_groups + uncommitted_skipped_groups) >= args.max_prompt_groups:
                        break
                    flush_buffers(force=False)

                if args.max_prompt_groups is not None and (uncommitted_processed_groups + uncommitted_skipped_groups) >= args.max_prompt_groups:
                    break

            # 最後の group を忘れずに処理
            if current_group_rows and (args.max_prompt_groups is None or (uncommitted_processed_groups + uncommitted_skipped_groups) < args.max_prompt_groups):
                process_completed_group(current_group_rows)
                uncommitted_sorted_rows = total_sorted_rows

            flush_buffers(force=True)

        summary = finalize_dataset(paths=paths, save_max_shard_size=args.save_max_shard_size)
        progress = load_or_init_progress(paths)
        progress["status"] = "finalized"
        progress["finalized"] = True
        progress["finalized_at"] = now_iso()
        progress["committed_sorted_rows"] = committed_sorted_rows
        progress["processed_prompt_groups"] = processed_prompt_groups
        progress["skipped_prompt_groups"] = skipped_prompt_groups
        progress["train_pairs"] = total_pairs["train"]
        progress["validation_pairs"] = total_pairs["validation"]
        progress["next_part_id"] = next_part_ids
        progress["last_global_prompt_index"] = last_global_prompt_index
        progress["sort_indices_cache_file"] = str(sort_indices_cache_file)
        progress["total_sorted_rows"] = total_sorted_rows
        write_progress(paths, progress)

        if args.cleanup_sort_cache and sort_indices_cache_file.exists():
            try:
                sort_indices_cache_file.unlink()
            except OSError:
                pass

        print(json.dumps(summary, ensure_ascii=False, indent=2))

    except Exception as exc:
        write_error(paths, exc)
        raise


if __name__ == "__main__":
    main()
