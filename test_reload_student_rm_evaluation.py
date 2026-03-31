#!/usr/bin/env python3
"""生徒RMの評価出力 (predictions dataset / summary JSON) を再読み込みして簡易確認する。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
from datasets import Dataset, DatasetDict, load_from_disk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reload evaluation outputs and run basic sanity checks.")
    parser.add_argument("--predictions-dataset-dir", type=str, required=True, help="evaluate_student_rm_against_teacher.py が保存した predictions_dataset")
    parser.add_argument("--summary-json", type=str, default=None, help="evaluation_summary.json")
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--score-column", type=str, default="teacher_score")
    parser.add_argument("--student-score-column", type=str, default="student_score_denorm")
    parser.add_argument("--group-column", type=str, default="global_prompt_index")
    parser.add_argument("--max-samples", type=int, default=2000, help="先頭N件だけで再計算")
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


def ensure_dataset(obj, split: str) -> Dataset:
    if isinstance(obj, DatasetDict):
        if split not in obj:
            raise SystemExit(f"split {split!r} がありません。利用可能: {list(obj.keys())}")
        return obj[split]
    if isinstance(obj, Dataset):
        return obj
    raise SystemExit(f"Unsupported dataset type: {type(obj)!r}")


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size <= 1 or y.size <= 1:
        return 0.0
    if np.std(x) <= 0 or np.std(y) <= 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def main() -> None:
    args = parse_args()
    configure_hf_cache(args.cache_dir)

    ds = ensure_dataset(load_from_disk(args.predictions_dataset_dir), args.split)
    required_columns = [args.score_column, args.student_score_column]
    for column in required_columns:
        if column not in ds.column_names:
            raise SystemExit(f"必要列 {column!r} が見つかりません。利用可能: {ds.column_names}")

    n_total = len(ds)
    n = min(args.max_samples, n_total)
    if n <= 0:
        raise SystemExit("dataset が空です。")
    sub = ds.select(range(n))

    teacher = np.asarray(sub[args.score_column], dtype=np.float64)
    student = np.asarray(sub[args.student_score_column], dtype=np.float64)
    mae = float(np.mean(np.abs(student - teacher)))
    mse = float(np.mean((student - teacher) ** 2))
    pearson = pearson_corr(student, teacher)

    payload = {
        "num_rows_total": int(n_total),
        "num_rows_checked": int(n),
        "columns": list(ds.column_names),
        "subset_metrics": {
            "mae": mae,
            "mse": mse,
            "pearson": pearson,
        },
    }
    if args.group_column in ds.column_names:
        payload["num_prompt_groups_checked"] = int(len(set(map(str, sub[args.group_column]))))

    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.summary_json is not None:
        summary_path = Path(args.summary_json).expanduser().resolve()
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        print("\nSummary file preview:")
        print(json.dumps({
            "global_metrics": summary.get("global_metrics", {}),
            "group_metrics": summary.get("group_metrics", {}),
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
