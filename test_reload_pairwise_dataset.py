#!/usr/bin/env python3
"""生成済み pairwise データセットを load_from_disk() で再読込し、簡易検証する。"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Set

try:
    from datasets import Dataset, DatasetDict, load_from_disk
except ImportError as exc:  # pragma: no cover
    raise SystemExit("datasets が見つかりません。`pip install -U datasets pyarrow` を実行してください。") from exc


BASE_REQUIRED_COLUMNS = {
    "pair_id",
    "split",
    "pair_strategy",
    "pair_rank_within_prompt",
    "messages",
    "chosen",
    "rejected",
    "prompt_hash",
    "global_prompt_index",
    "teacher_model",
    "chosen_sample_id",
    "rejected_sample_id",
    "chosen_score",
    "rejected_score",
    "score_margin",
    "chosen_response_hash",
    "rejected_response_hash",
    "chosen_generator_model",
    "rejected_generator_model",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reload and validate a saved pairwise dataset.")
    parser.add_argument("--dataset-dir", type=str, required=True, help="save_to_disk 済みの final_dataset ディレクトリ")
    parser.add_argument("--summary-json", type=str, default=None, help="final_summary.json があれば件数照合する")
    parser.add_argument("--head", type=int, default=3, help="表示する先頭サンプル数")
    parser.add_argument("--check-unique-pair-id", action="store_true", help="pair_id の重複が無いことを確認する")
    parser.add_argument("--check-no-prompt-leak", action="store_true", help="train/validation の prompt_hash 重複が無いことを確認する")
    return parser.parse_args()



def validate_row(row: Dict[str, Any], idx: int) -> None:
    assert isinstance(row["messages"], list) and row["messages"], f"row {idx}: messages が空または不正"
    assert row["messages"][-1]["role"] == "user", f"row {idx}: prompt messages の末尾は user のはず"
    assert isinstance(row["chosen"], str), f"row {idx}: chosen が str ではありません"
    assert isinstance(row["rejected"], str), f"row {idx}: rejected が str ではありません"
    assert float(row["chosen_score"]) > float(row["rejected_score"]), f"row {idx}: chosen_score <= rejected_score"
    assert float(row["score_margin"]) > 0.0, f"row {idx}: score_margin <= 0"
    assert isinstance(row["pair_id"], str) and row["pair_id"], f"row {idx}: pair_id が空です"
    assert row["chosen_sample_id"] != row["rejected_sample_id"], f"row {idx}: chosen と rejected の sample_id が同一です"
    if "chosen_messages" in row:
        assert isinstance(row["chosen_messages"], list) and row["chosen_messages"], f"row {idx}: chosen_messages が不正"
        assert row["chosen_messages"][-1]["role"] == "assistant", f"row {idx}: chosen_messages の末尾は assistant のはず"
    if "rejected_messages" in row:
        assert isinstance(row["rejected_messages"], list) and row["rejected_messages"], f"row {idx}: rejected_messages が不正"
        assert row["rejected_messages"][-1]["role"] == "assistant", f"row {idx}: rejected_messages の末尾は assistant のはず"



def iter_splits(ds_any: Any) -> Iterable[tuple[str, Dataset]]:
    if isinstance(ds_any, DatasetDict):
        for split_name, ds in ds_any.items():
            yield split_name, ds
    else:
        yield "train", ds_any



def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    ds_any = load_from_disk(str(dataset_dir))

    total_rows = 0
    pair_ids: Set[str] = set()
    prompt_hash_by_split: Dict[str, Set[str]] = {}
    summary_counts: Dict[str, int] = {}

    for split_name, ds in iter_splits(ds_any):
        missing = BASE_REQUIRED_COLUMNS - set(ds.column_names)
        assert not missing, f"split={split_name}: 必須列が足りません: {sorted(missing)}"
        assert len(ds) > 0, f"split={split_name}: データセットが空です"

        total_rows += len(ds)
        summary_counts[split_name] = len(ds)
        prompt_hash_by_split[split_name] = set(ds["prompt_hash"])

        strategy_counter = collections.Counter(ds["pair_strategy"])
        split_counter = collections.Counter(ds["split"])
        margin_values = ds["score_margin"]
        print(f"\n[{split_name}]")
        print(f"rows: {len(ds)}")
        print(f"columns: {ds.column_names}")
        print("pair_strategy counts:")
        for key, value in strategy_counter.most_common():
            print(f"  - {key}: {value}")
        print("split column counts:")
        for key, value in split_counter.items():
            print(f"  - {key}: {value}")
        print(f"score_margin: min={min(margin_values):.6f} max={max(margin_values):.6f} avg={sum(margin_values)/len(margin_values):.6f}")

        for i in range(min(args.head, len(ds))):
            row = ds[i]
            validate_row(row, i)
            prompt_preview = str(row.get("prompt_raw", "")).replace("\n", " ")[:80]
            chosen_preview = row["chosen"].replace("\n", " ")[:80]
            rejected_preview = row["rejected"].replace("\n", " ")[:80]
            print(f"\n[example {split_name}:{i}]")
            print(f"pair_id: {row['pair_id']}")
            print(f"score_margin: {float(row['score_margin']):.6f}")
            print(f"prompt_preview: {prompt_preview}")
            print(f"chosen_preview: {chosen_preview}")
            print(f"rejected_preview: {rejected_preview}")

        if args.check_unique_pair_id:
            current_pair_ids = set(ds["pair_id"])
            assert len(current_pair_ids) == len(ds), f"split={split_name}: pair_id に重複があります"
            overlap = pair_ids & current_pair_ids
            assert not overlap, f"split 間で pair_id が重複しています: {list(sorted(overlap))[:5]}"
            pair_ids |= current_pair_ids
            print(f"unique pair_id check ({split_name}): OK")

    if args.check_no_prompt_leak and {"train", "validation"}.issubset(prompt_hash_by_split):
        overlap = prompt_hash_by_split["train"] & prompt_hash_by_split["validation"]
        assert not overlap, f"train/validation で prompt_hash が重複しています: {list(sorted(overlap))[:5]}"
        print("train/validation prompt leakage check: OK")

    if args.summary_json:
        summary_path = Path(args.summary_json).expanduser().resolve()
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        assert int(summary["total_rows"]) == total_rows, (
            f"summary total_rows mismatch: summary={summary['total_rows']} dataset={total_rows}"
        )
        for split_name, expected in summary.get("splits", {}).items():
            if split_name in summary_counts:
                assert int(expected["rows"]) == summary_counts[split_name], (
                    f"summary rows mismatch for {split_name}: summary={expected['rows']} dataset={summary_counts[split_name]}"
                )
        print("summary total_rows check: OK")

    print("\n[OK] pairwise dataset reload test passed.")


if __name__ == "__main__":
    main()
