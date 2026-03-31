#!/usr/bin/env python3
"""生成済み教師データを load_from_disk() で再読込し、簡易検証する。"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Dict

try:
    from datasets import DatasetDict, load_from_disk
except ImportError as exc:  # pragma: no cover
    raise SystemExit("datasets が見つかりません。`pip install -U datasets pyarrow` を実行してください。") from exc


REQUIRED_COLUMNS = {
    "messages",
    "response",
    "teacher_score",
    "teacher_model",
    "generator_model",
    "generator_key",
    "generator_model_index",
    "candidate_index",
    "sample_id",
    "prompt_hash",
    "response_hash",
    "prompt_raw",
    "rank",
    "local_prompt_index",
    "global_prompt_index",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reload and validate a saved teacher dataset.")
    parser.add_argument("--dataset-dir", type=str, required=True, help="save_to_disk 済みの final_dataset ディレクトリ")
    parser.add_argument("--head", type=int, default=3, help="表示する先頭サンプル数")
    parser.add_argument("--summary-json", type=str, default=None, help="final_summary.json があれば件数照合する")
    parser.add_argument("--check-unique-sample-id", action="store_true", help="sample_id の重複が無いことも確認する")
    return parser.parse_args()


def validate_example(row: Dict[str, Any], idx: int) -> None:
    assert isinstance(row["messages"], list) and row["messages"], f"row {idx}: messages が空または不正"
    assert row["messages"][-1]["role"] == "user", f"row {idx}: prompt messages の末尾は user のはず"
    assert isinstance(row["response"], str), f"row {idx}: response が str ではありません"
    assert isinstance(row["teacher_score"], (int, float)), f"row {idx}: teacher_score が数値ではありません"
    assert isinstance(row["sample_id"], str) and row["sample_id"], f"row {idx}: sample_id が空です"
    assert isinstance(row["prompt_hash"], str) and row["prompt_hash"], f"row {idx}: prompt_hash が空です"
    assert isinstance(row["response_hash"], str) and row["response_hash"], f"row {idx}: response_hash が空です"


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    ds_any = load_from_disk(str(dataset_dir))
    if isinstance(ds_any, DatasetDict):
        assert "train" in ds_any, "DatasetDict に train split がありません"
        ds = ds_any["train"]
    else:
        ds = ds_any

    missing = REQUIRED_COLUMNS - set(ds.column_names)
    assert not missing, f"必須列が足りません: {sorted(missing)}"
    assert len(ds) > 0, "データセットが空です"

    generator_counter = collections.Counter(ds["generator_model"])
    rank_counter = collections.Counter(ds["rank"])
    candidate_counter = collections.Counter(ds["candidate_index"])

    print(f"dataset_dir: {dataset_dir}")
    print(f"rows: {len(ds)}")
    print(f"columns: {ds.column_names}")
    print("generator counts:")
    for key, value in generator_counter.most_common():
        print(f"  - {key}: {value}")
    print("rank counts:")
    for key, value in rank_counter.most_common():
        print(f"  - rank {key}: {value}")
    print("candidate_index counts:")
    for key, value in sorted(candidate_counter.items()):
        print(f"  - {key}: {value}")

    for i in range(min(args.head, len(ds))):
        row = ds[i]
        validate_example(row, i)
        prompt_preview = row["prompt_raw"].replace("\n", " ")[:100]
        response_preview = row["response"].replace("\n", " ")[:100]
        print(f"\n[example {i}]")
        print(f"generator_model: {row['generator_model']}")
        print(f"candidate_index: {row['candidate_index']}")
        print(f"teacher_score: {row['teacher_score']:.6f}")
        print(f"prompt_preview: {prompt_preview}")
        print(f"response_preview: {response_preview}")

    if args.check_unique_sample_id:
        unique_ids = set(ds["sample_id"])
        assert len(unique_ids) == len(ds), f"sample_id に重複があります: unique={len(unique_ids)} rows={len(ds)}"
        print("\nunique sample_id check: OK")

    if args.summary_json:
        summary_path = Path(args.summary_json).expanduser().resolve()
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        assert int(summary["total_rows"]) == len(ds), (
            f"summary total_rows mismatch: summary={summary['total_rows']} dataset={len(ds)}"
        )
        print("summary total_rows check: OK")

    print("\n[OK] teacher dataset reload test passed.")


if __name__ == "__main__":
    main()
