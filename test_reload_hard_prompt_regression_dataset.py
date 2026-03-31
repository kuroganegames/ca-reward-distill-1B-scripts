#!/usr/bin/env python3
"""hard prompt 追加学習データ出力を再読み込みして簡易検証する。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Tuple

try:
    from datasets import Dataset, DatasetDict, load_from_disk
except ImportError as exc:  # pragma: no cover
    raise SystemExit("datasets が見つかりません。`pip install -U datasets pyarrow` を実行してください。") from exc


ALL_PROMPT_REQUIRED = {
    "group_key",
    "hard_prompt_score",
    "selected",
    "top1_match",
    "pairwise_accuracy",
    "prompt_spearman",
    "mean_abs_error",
}

HARD_PROMPT_REQUIRED = {
    "messages",
    "group_key",
    "hard_prompt_score",
    "hard_prompt_rank",
    "selection_reason",
}

HARD_TRAIN_REQUIRED = {
    "messages",
    "response",
    "teacher_score",
    "hard_prompt_score",
    "hard_prompt_rank",
    "hard_selection_reason",
    "prompt_pairwise_accuracy",
    "prompt_spearman",
    "score_abs_error_denorm",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reload and validate hard-prompt regression datasets.")
    parser.add_argument("--output-dir", type=str, required=True, help="build_hard_prompt_regression_dataset.py の output-dir")
    parser.add_argument("--summary-json", type=str, default=None, help="summary.json")
    parser.add_argument("--all-prompt-metrics-subdir", type=str, default="all_prompt_metrics_dataset")
    parser.add_argument("--hard-prompt-subdir", type=str, default="hard_prompt_dataset")
    parser.add_argument("--hard-training-subdir", type=str, default="hard_training_rows")
    parser.add_argument("--augmented-subdir", type=str, default="augmented_training_dataset")
    parser.add_argument("--all-split", type=str, default="train", help="all_prompt_metrics_dataset の split")
    parser.add_argument("--train-split", type=str, default="train", help="hard_prompt_dataset / hard_training_rows / augmented dataset の split")
    parser.add_argument("--head", type=int, default=3)
    return parser.parse_args()



def ensure_dataset(obj: Any, split: str) -> Dataset:
    if isinstance(obj, DatasetDict):
        if split not in obj:
            raise SystemExit(f"split {split!r} がありません。利用可能: {list(obj.keys())}")
        return obj[split]
    if isinstance(obj, Dataset):
        return obj
    raise SystemExit(f"Unsupported dataset type: {type(obj)!r}")



def maybe_load_dataset(path: Path, split: str) -> Tuple[bool, Dataset | None]:
    if not path.exists():
        return False, None
    return True, ensure_dataset(load_from_disk(str(path)), split)



def preview_messages(messages: Any) -> str:
    if not isinstance(messages, list) or not messages:
        return ""
    last = messages[-1]
    if isinstance(last, dict):
        return str(last.get("content", "")).replace("\n", " ")[:80]
    return str(last).replace("\n", " ")[:80]



def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()

    all_path = output_dir / args.all_prompt_metrics_subdir
    hard_prompt_path = output_dir / args.hard_prompt_subdir
    hard_train_path = output_dir / args.hard_training_subdir
    aug_path = output_dir / args.augmented_subdir

    all_ds = ensure_dataset(load_from_disk(str(all_path)), args.all_split)
    hard_prompt_ds = ensure_dataset(load_from_disk(str(hard_prompt_path)), args.train_split)
    hard_train_ds = ensure_dataset(load_from_disk(str(hard_train_path)), args.train_split)
    aug_exists, aug_ds = maybe_load_dataset(aug_path, args.train_split)

    missing = ALL_PROMPT_REQUIRED - set(all_ds.column_names)
    assert not missing, f"all_prompt_metrics に必須列が足りません: {sorted(missing)}"
    missing = HARD_PROMPT_REQUIRED - set(hard_prompt_ds.column_names)
    assert not missing, f"hard_prompt_dataset に必須列が足りません: {sorted(missing)}"
    missing = HARD_TRAIN_REQUIRED - set(hard_train_ds.column_names)
    assert not missing, f"hard_training_rows に必須列が足りません: {sorted(missing)}"

    if len(hard_prompt_ds) > 0:
        ranks = list(hard_prompt_ds["hard_prompt_rank"])
        assert all(int(rank) >= 1 for rank in ranks), "hard_prompt_rank に 1 未満が含まれています"

    hard_group_keys = set(map(str, hard_prompt_ds["group_key"]))
    train_group_keys = set(map(str, hard_train_ds["group_key"])) if "group_key" in hard_train_ds.column_names else set()
    assert train_group_keys.issubset(hard_group_keys), "hard_training_rows に hard_prompt_dataset に無い group_key が含まれています"

    print("[all_prompt_metrics]")
    print(f"rows: {len(all_ds)}")
    print(f"columns: {all_ds.column_names}")
    for i in range(min(args.head, len(all_ds))):
        row = all_ds[i]
        print(f"  - idx={i} group_key={row['group_key']} score={float(row['hard_prompt_score']):.4f} selected={bool(row['selected'])}")

    print("\n[hard_prompt_dataset]")
    print(f"rows: {len(hard_prompt_ds)}")
    print(f"columns: {hard_prompt_ds.column_names}")
    for i in range(min(args.head, len(hard_prompt_ds))):
        row = hard_prompt_ds[i]
        assert isinstance(row["messages"], list), f"row {i}: messages が list ではありません"
        print(
            f"  - rank={int(row['hard_prompt_rank'])} score={float(row['hard_prompt_score']):.4f} "
            f"reason={row['selection_reason']} preview={preview_messages(row['messages'])}"
        )

    print("\n[hard_training_rows]")
    print(f"rows: {len(hard_train_ds)}")
    print(f"columns: {hard_train_ds.column_names}")
    for i in range(min(args.head, len(hard_train_ds))):
        row = hard_train_ds[i]
        assert isinstance(row["messages"], list), f"row {i}: messages が list ではありません"
        assert isinstance(row["response"], str), f"row {i}: response が str ではありません"
        print(
            f"  - rank={int(row['hard_prompt_rank'])} sample_id={row.get('sample_id', '')} "
            f"teacher={float(row['teacher_score']):.4f} abs_err={float(row['score_abs_error_denorm']):.4f}"
        )

    if aug_exists and aug_ds is not None:
        print("\n[augmented_training_dataset]")
        print(f"rows: {len(aug_ds)}")
        print(f"columns: {aug_ds.column_names}")

    if args.summary_json:
        summary_path = Path(args.summary_json).expanduser().resolve()
        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
        counts = summary.get("counts", {})
        assert int(counts.get("prompt_groups_total", len(all_ds))) == len(all_ds), "summary の prompt_groups_total と all_prompt_metrics rows が一致しません"
        assert int(counts.get("prompt_groups_selected", len(hard_prompt_ds))) == len(hard_prompt_ds), "summary の prompt_groups_selected と hard_prompt_dataset rows が一致しません"
        assert int(counts.get("hard_training_rows", len(hard_train_ds))) == len(hard_train_ds), "summary の hard_training_rows と dataset rows が一致しません"
        if bool(counts.get("augmented_dataset_saved", False)):
            assert aug_exists and aug_ds is not None, "summary では augmented_dataset_saved=True なのに dataset がありません"
            assert int(counts.get("augmented_train_rows", len(aug_ds))) == len(aug_ds), "summary の augmented_train_rows と dataset rows が一致しません"
        print("\nsummary count checks: OK")

    print("\n[OK] hard prompt regression dataset reload test passed.")


if __name__ == "__main__":
    main()
