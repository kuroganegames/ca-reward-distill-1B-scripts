#!/usr/bin/env python3
"""save_to_disk 済みの mixed prompt dataset を再読み込みして簡易検証する。"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Dict, List

try:
    from datasets import DatasetDict, load_from_disk
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "datasets が見つかりません。`pip install -U datasets pyarrow` を実行してください。"
    ) from exc


REQUIRED_COLUMNS = {
    "messages",
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
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reload and sanity-check a saved mixed prompt dataset.")
    parser.add_argument("--dataset-dir", type=str, required=True, help="save_to_disk 済みのディレクトリ")
    parser.add_argument("--head", type=int, default=3, help="表示する先頭サンプル数")
    parser.add_argument("--summary-json", type=str, default=None, help="build summary JSON があれば追加検証する")
    return parser.parse_args()


def validate_example(example: Dict[str, Any], idx: int) -> None:
    assert isinstance(example["messages"], list), f"row {idx}: messages is not a list"
    assert example["messages"], f"row {idx}: messages is empty"
    assert example["messages"][-1]["role"] == "user", f"row {idx}: last role must be user"
    assert isinstance(example["prompt_raw"], str) and example["prompt_raw"].strip(), f"row {idx}: prompt_raw is empty"
    assert example["prompt_char_len"] == len(example["prompt_raw"]), f"row {idx}: prompt_char_len mismatch"
    assert example["num_prompt_messages"] == len(example["messages"]), f"row {idx}: num_prompt_messages mismatch"


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    ds = load_from_disk(str(dataset_dir))

    if isinstance(ds, DatasetDict):
        assert "train" in ds, "DatasetDict に train split がありません"
        train = ds["train"]
    else:
        train = ds

    missing = REQUIRED_COLUMNS - set(train.column_names)
    assert not missing, f"必須列が足りません: {sorted(missing)}"
    assert len(train) > 0, "データセットが空です"

    source_counter = collections.Counter(train["source_alias"])
    lang_counter = collections.Counter(train["prompt_language"])

    print(f"dataset_dir: {dataset_dir}")
    print(f"rows: {len(train)}")
    print(f"columns: {train.column_names}")
    print("source counts:")
    for source_alias, count in source_counter.most_common():
        print(f"  - {source_alias}: {count}")
    print("prompt_language counts:")
    for lang, count in lang_counter.most_common():
        print(f"  - {lang}: {count}")

    for i in range(min(args.head, len(train))):
        example = train[i]
        validate_example(example, i)
        preview = example["prompt_raw"].replace("\n", " ")[:120]
        print(f"\n[example {i}]")
        print(f"source_alias: {example['source_alias']}")
        print(f"prompt_language: {example['prompt_language']}")
        print(f"num_prompt_messages: {example['num_prompt_messages']}")
        print(f"wrapped_for_japanese_answer: {example['wrapped_for_japanese_answer']}")
        print(f"prompt_preview: {preview}")

    if args.summary_json:
        summary_path = Path(args.summary_json).expanduser().resolve()
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        assert summary["total_rows"] == len(train), (
            f"summary total_rows mismatch: summary={summary['total_rows']} dataset={len(train)}"
        )
        requested_sources = set(summary["counts"].keys())
        actual_sources = set(source_counter.keys())
        print("\nsummary check:")
        print(f"  requested sources: {sorted(requested_sources)}")
        print(f"  actual sources   : {sorted(actual_sources)}")

    print("\n[OK] reload test passed.")


if __name__ == "__main__":
    main()
