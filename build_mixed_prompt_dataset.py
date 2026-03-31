#!/usr/bin/env python3
"""日本語中心のミックス・プロンプトデータセットを Hugging Face Datasets 形式で保存する。

目的:
- 複数の公開 instruction / chat データセットから「プロンプト側」だけを抽出する
- 英語データは必要に応じて「日本語で回答してください」を付加する
- 生成用の入力母集団を 1 つの DatasetDict(train) にまとめて save_to_disk する
- 後から件数だけ差し替えやすいように dict / JSON で管理できるようにする

保存形式:
- datasets.DatasetDict({"train": Dataset}) を save_to_disk(output_dir)
- そのため、後で datasets.load_from_disk(output_dir) で再読み込みできる

主な列:
- messages: 生成入力としてそのまま使える prompt-side の chat messages
- prompt_raw: 最後の user 発話の生テキスト（ラッピング前）
- prompt_language: 推定または既知の入力言語
- answer_language: 期待する出力言語（既定は ja）
- source_*: 元データセットのメタ情報

例:
    python build_mixed_prompt_dataset.py \
        --output-dir ./mixed_prompt_pool \
        --cache-dir /data/hf_cache \
        --counts-preset trial \
        --streaming

    python build_mixed_prompt_dataset.py \
        --output-dir ./mixed_prompt_pool \
        --cache-dir /data/hf_cache \
        --counts-preset trial \
        --counts-json ./counts_override.json \
        --streaming
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

try:
    from datasets import Dataset, DatasetDict, load_dataset
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "datasets が見つかりません。`pip install -U datasets pyarrow` を実行してください。"
    ) from exc

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, *args, **kwargs):
        return iterable


EN_TO_JA_SUFFIX = "\n\nPlease answer in Japanese.\n日本語で回答してください。"


@dataclass(frozen=True)
class SourceSpec:
    alias: str
    path: str
    split: str = "train"
    config_name: Optional[str] = None
    data_dir: Optional[str] = None
    default_lang: str = "ja"  # ja / en / auto
    allowed_prompt_languages: Optional[Tuple[str, ...]] = None
    wrap_english_prompt_to_japanese: bool = False
    notes: str = ""


# --- 対象データセット定義 -------------------------------------------------
# ここを直接編集してもよいし、counts だけ JSON で上書きしてもよい。
SOURCE_SPECS: Dict[str, SourceSpec] = {
    "magpie_ja": SourceSpec(
        alias="magpie_ja",
        path="llm-jp/magpie-sft-v1.0",
        split="train",
        default_lang="ja",
        notes="Japanese Magpie instruction dataset.",
    ),
    "extraction_wiki_ja": SourceSpec(
        alias="extraction_wiki_ja",
        path="llm-jp/extraction-wiki-ja",
        split="train",
        # subset 名は v0.1 / v0.2 / v0.3。まずは richest な v0.3 を採用。
        config_name="v0.3",
        default_lang="ja",
        notes="Japanese extraction / structuring dataset from Wikipedia.",
    ),
    "wizard_math_code_ja": SourceSpec(
        alias="wizard_math_code_ja",
        path="llm-jp/wizardlm8x22b-logical-math-coding-sft-ja",
        split="train",
        default_lang="ja",
        notes="Logical / math / coding heavy Japanese dataset.",
    ),
    "oasst2_ja": SourceSpec(
        alias="oasst2_ja",
        path="llm-jp/oasst2-33k-ja",
        split="train",
        default_lang="ja",
        notes="Japanese OASST2 translation.",
    ),
    "oasst1_ja": SourceSpec(
        alias="oasst1_ja",
        path="llm-jp/oasst1-21k-ja",
        split="train",
        default_lang="ja",
        notes="Japanese OASST1 translation.",
    ),
    "synthetic_jp_en_coding_ja_only": SourceSpec(
        alias="synthetic_jp_en_coding_ja_only",
        path="llm-jp/Synthetic-JP-EN-Coding-Dataset",
        split="train",
        default_lang="auto",
        allowed_prompt_languages=("ja",),
        notes="Mixed JP/EN coding dataset; keep Japanese prompts only by default.",
    ),
    "dolly_ja": SourceSpec(
        alias="dolly_ja",
        path="llm-jp/databricks-dolly-15k-ja",
        split="train",
        default_lang="ja",
        notes="Japanese Dolly translation.",
    ),
    "llm_jp_instructions_v1_train": SourceSpec(
        alias="llm_jp_instructions_v1_train",
        path="llm-jp/llm-jp-instructions",
        split="train",
        data_dir="v1.0",
        default_lang="ja",
        notes="Manually created Japanese instruction dataset (v1.0/train).",
    ),
    "oasst2_en_to_ja": SourceSpec(
        alias="oasst2_en_to_ja",
        path="llm-jp/oasst2-33k-en",
        split="train",
        default_lang="en",
        wrap_english_prompt_to_japanese=True,
        notes="English OASST2 prompts; append request to answer in Japanese.",
    ),
    "oasst1_en_to_ja": SourceSpec(
        alias="oasst1_en_to_ja",
        path="llm-jp/oasst1-21k-en",
        split="train",
        default_lang="en",
        wrap_english_prompt_to_japanese=True,
        notes="English OASST1 prompts; append request to answer in Japanese.",
    ),
    "daring_anteater_en_to_ja": SourceSpec(
        alias="daring_anteater_en_to_ja",
        path="nvidia/Daring-Anteater",
        split="train",
        default_lang="en",
        wrap_english_prompt_to_japanese=True,
        notes="Optional larger English source; disabled by default in trial preset.",
    ),
}


# 以前の提案に沿った試運転用（約 52k）
TRIAL_COUNTS: Dict[str, int] = {
    "magpie_ja": 15_000,
    "extraction_wiki_ja": 10_000,
    "wizard_math_code_ja": 8_000,
    "oasst2_ja": 6_000,
    "oasst1_ja": 3_000,
    "synthetic_jp_en_coding_ja_only": 3_000,
    "dolly_ja": 2_000,
    "llm_jp_instructions_v1_train": 1_000,
    "oasst2_en_to_ja": 2_000,
    "oasst1_en_to_ja": 2_000,
    "daring_anteater_en_to_ja": 0,
}


# 本番寄りの例（適宜調整）
PROD_LIKE_COUNTS: Dict[str, int] = {
    "magpie_ja": 30_000,
    "extraction_wiki_ja": 20_000,
    "wizard_math_code_ja": 15_000,
    "oasst2_ja": 10_000,
    "oasst1_ja": 6_000,
    "synthetic_jp_en_coding_ja_only": 6_000,
    "dolly_ja": 4_000,
    "llm_jp_instructions_v1_train": 1_000,
    "oasst2_en_to_ja": 4_000,
    "oasst1_en_to_ja": 4_000,
    "daring_anteater_en_to_ja": 0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build mixed prompt dataset for Japanese reward-model data generation.")
    parser.add_argument("--output-dir", type=str, required=True, help="save_to_disk する出力先ディレクトリ")
    parser.add_argument("--cache-dir", type=str, default=None, help="Hugging Face cache root。ストレージ節約のため手動指定可能")
    parser.add_argument(
        "--counts-preset",
        type=str,
        default="trial",
        choices=["trial", "prod_like"],
        help="件数の既定セット。counts-json で部分上書き可能",
    )
    parser.add_argument(
        "--counts-json",
        type=str,
        default=None,
        help="件数 override 用 JSON。例: {\"magpie_ja\": 5000, \"oasst2_en_to_ja\": 500}",
    )
    parser.add_argument("--seed", type=int, default=42, help="shuffle / sampling 用 seed")
    parser.add_argument("--shuffle-buffer-size", type=int, default=10_000, help="streaming shuffle の buffer size")
    parser.add_argument("--streaming", dest="streaming", action="store_true", help="可能なら streaming で読み込む（既定）")
    parser.add_argument("--no-streaming", dest="streaming", action="store_false", help="通常の load_dataset を使う")
    parser.set_defaults(streaming=True)
    parser.add_argument("--dedup", dest="dedup", action="store_true", help="正規化 messages ベースで重複除去する（既定）")
    parser.add_argument("--no-dedup", dest="dedup", action="store_false", help="重複除去しない")
    parser.set_defaults(dedup=True)
    parser.add_argument(
        "--keep-existing-output",
        action="store_true",
        help="出力先が既に存在しても削除せず上書きエラーをそのまま出す。既定は存在時に終了。",
    )
    return parser.parse_args()


def configure_hf_cache(cache_dir: Optional[str]) -> Optional[Path]:
    if not cache_dir:
        return None
    root = Path(cache_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    (root / "hub").mkdir(parents=True, exist_ok=True)
    (root / "datasets").mkdir(parents=True, exist_ok=True)

    # 環境変数にも設定しておくと huggingface_hub / datasets の両方で効きやすい。
    os.environ["HF_HOME"] = str(root)
    os.environ["HF_HUB_CACHE"] = str(root / "hub")
    os.environ["HF_DATASETS_CACHE"] = str(root / "datasets")
    return root


def load_effective_counts(preset: str, counts_json: Optional[str]) -> Dict[str, int]:
    if preset == "trial":
        counts = dict(TRIAL_COUNTS)
    elif preset == "prod_like":
        counts = dict(PROD_LIKE_COUNTS)
    else:  # pragma: no cover
        raise ValueError(f"Unknown preset: {preset}")

    if counts_json:
        with open(counts_json, "r", encoding="utf-8") as f:
            override = json.load(f)
        if not isinstance(override, dict):
            raise ValueError("counts-json は dict 形式の JSON である必要があります。")
        for key, value in override.items():
            if key not in SOURCE_SPECS:
                raise KeyError(f"counts-json に未知の source alias が含まれています: {key}")
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"counts-json の値は 0 以上の int にしてください: {key}={value}")
            counts[key] = value

    # SOURCE_SPECS に存在しない key を弾く
    for key in counts:
        if key not in SOURCE_SPECS:
            raise KeyError(f"Unknown source alias in counts: {key}")
    return counts


def normalize_role(role: Any) -> Optional[str]:
    if role is None:
        return None
    text = str(role).strip().lower()
    mapping = {
        "system": "system",
        "sys": "system",
        "user": "user",
        "human": "user",
        "prompter": "user",
        "instruction": "user",
        "question": "user",
        "assistant": "assistant",
        "gpt": "assistant",
        "bot": "assistant",
        "model": "assistant",
        "responder": "assistant",
    }
    return mapping.get(text, text if text in {"system", "user", "assistant"} else None)


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, list):
        return "\n".join(str(x) for x in text if x is not None).strip()
    return str(text).strip()


def extract_messages_from_list(items: Any) -> Optional[List[Dict[str, str]]]:
    if not isinstance(items, list):
        return None

    messages: List[Dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        role = normalize_role(
            item.get("role")
            or item.get("from")
            or item.get("speaker")
            or item.get("author")
            or item.get("turn_role")
        )
        content = normalize_text(
            item.get("content")
            or item.get("value")
            or item.get("text")
            or item.get("message")
            or item.get("prompt")
        )
        if role and content:
            messages.append({"role": role, "content": content})
    return messages or None


def extract_system_message(record: Dict[str, Any]) -> List[Dict[str, str]]:
    for key in ["system", "system_prompt", "system_message", "instruction_system"]:
        value = normalize_text(record.get(key))
        if value:
            return [{"role": "system", "content": value}]
    return []


def combine_instruction_and_context(instruction: str, context: str) -> str:
    instruction = instruction.strip()
    context = context.strip()
    if instruction and context:
        if context in instruction:
            return instruction
        return f"{instruction}\n\n### 追加情報\n{context}"
    return instruction or context


def extract_messages_from_fields(record: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    system_messages = extract_system_message(record)

    # よくある instruction-style スキーマ
    question = normalize_text(record.get("question"))
    instruction = normalize_text(record.get("instruction"))
    prompt = normalize_text(record.get("prompt"))
    query = normalize_text(record.get("query"))
    context = normalize_text(record.get("context")) or normalize_text(record.get("input"))
    text = normalize_text(record.get("text"))
    content = normalize_text(record.get("content"))
    task = normalize_text(record.get("task"))

    user_text = ""
    if question:
        user_text = question
    elif instruction or context:
        user_text = combine_instruction_and_context(instruction, context)
    elif prompt:
        user_text = prompt
    elif query:
        user_text = query
    elif content and len(content) <= 10_000:
        # content は assistant 側である可能性もあるが、他の有力候補がない時だけ使う。
        user_text = content
    elif text and len(text) <= 10_000:
        user_text = text
    elif task and len(task) <= 10_000:
        user_text = task

    user_text = user_text.strip()
    if not user_text:
        return None
    return system_messages + [{"role": "user", "content": user_text}]


def extract_messages(record: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    # chat / conversation 系を優先
    for key in ["messages", "conversations", "conversation", "dialog", "dialogue", "chat"]:
        messages = extract_messages_from_list(record.get(key))
        if messages:
            system_messages = extract_system_message(record)
            # 既存 messages が system で始まっているなら top-level system は重複追加しない
            if system_messages and not (messages and messages[0]["role"] == "system"):
                return system_messages + messages
            return messages

    return extract_messages_from_fields(record)


def trim_to_prompt_side(messages: Sequence[Dict[str, str]]) -> Optional[List[Dict[str, str]]]:
    trimmed = [
        {"role": normalize_role(m.get("role")) or "user", "content": normalize_text(m.get("content"))}
        for m in messages
        if normalize_text(m.get("content"))
    ]
    trimmed = [m for m in trimmed if m["role"] in {"system", "user", "assistant"}]
    if not trimmed:
        return None

    # 学習済み応答が末尾にある場合は落として「次に assistant を生成するための入力」にする
    while trimmed and trimmed[-1]["role"] == "assistant":
        trimmed.pop()
    if not trimmed:
        return None

    # 末尾が user になるように切り詰める
    if trimmed[-1]["role"] != "user":
        last_user_idx = None
        for idx in range(len(trimmed) - 1, -1, -1):
            if trimmed[idx]["role"] == "user":
                last_user_idx = idx
                break
        if last_user_idx is None:
            return None
        trimmed = trimmed[: last_user_idx + 1]

    if not trimmed or trimmed[-1]["role"] != "user":
        return None
    return trimmed


def get_last_user_content(messages: Sequence[Dict[str, str]]) -> str:
    for message in reversed(messages):
        if message["role"] == "user":
            return normalize_text(message["content"])
    return ""


_JA_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]")
_EN_RE = re.compile(r"[A-Za-z]")
_WS_RE = re.compile(r"\s+")


def guess_language(text: str) -> str:
    ja_count = len(_JA_RE.findall(text))
    en_count = len(_EN_RE.findall(text))
    if ja_count >= 2 and ja_count >= max(1, int(en_count * 0.05)):
        return "ja"
    if en_count >= 8 and ja_count == 0:
        return "en"
    if ja_count > en_count:
        return "ja"
    if en_count > ja_count:
        return "en"
    return "unknown"


def append_english_to_japanese_request(messages: Sequence[Dict[str, str]]) -> Tuple[List[Dict[str, str]], bool]:
    wrapped = copy.deepcopy(list(messages))
    for idx in range(len(wrapped) - 1, -1, -1):
        if wrapped[idx]["role"] == "user":
            wrapped[idx]["content"] = wrapped[idx]["content"].rstrip() + EN_TO_JA_SUFFIX
            return wrapped, True
    return wrapped, False


def canonicalize_messages(messages: Sequence[Dict[str, str]]) -> str:
    normalized = []
    for m in messages:
        content = _WS_RE.sub(" ", normalize_text(m.get("content"))).strip()
        normalized.append({"role": m.get("role", "user"), "content": content})
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))


def load_source_dataset(spec: SourceSpec, cache_dir: Optional[Path], streaming: bool):
    attempts: List[Dict[str, Any]] = []

    # config_name / data_dir の指定があるデータセットは、誤って default subset を取らないよう
    # 明示指定を優先し、必要な場合だけ相互に代替候補を試す。
    base_kwargs: Dict[str, Any] = {
        "split": spec.split,
        "streaming": streaming,
    }
    if cache_dir is not None:
        base_kwargs["cache_dir"] = str(cache_dir)

    has_explicit_subset = spec.config_name is not None or spec.data_dir is not None
    if spec.config_name is not None and spec.data_dir is not None:
        attempts.append({**base_kwargs, "name": spec.config_name, "data_dir": spec.data_dir})
    if spec.config_name is not None:
        attempts.append({**base_kwargs, "name": spec.config_name})
        # 一部データセットは viewer 上の subset 名を data_dir として要求することがある
        attempts.append({**base_kwargs, "data_dir": spec.config_name})
    if spec.data_dir is not None:
        attempts.append({**base_kwargs, "data_dir": spec.data_dir})
        # 逆パターンの互換候補
        attempts.append({**base_kwargs, "name": spec.data_dir})
    if not has_explicit_subset:
        attempts.append(dict(base_kwargs))

    last_error: Optional[Exception] = None
    for kwargs in attempts:
        try:
            return load_dataset(spec.path, **kwargs)
        except Exception as exc:  # noqa: BLE001 - 利便性優先で集約
            last_error = exc
            continue

    if streaming:
        # 一部データセットは streaming がうまく動かない場合があるため、同じ候補群で非 streaming を再試行する。
        print(f"[WARN] streaming load に失敗したため {spec.alias} は非 streaming で再試行します。", file=sys.stderr)
        for kwargs in attempts:
            fallback_kwargs = dict(kwargs)
            fallback_kwargs["streaming"] = False
            try:
                return load_dataset(spec.path, **fallback_kwargs)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue

    raise RuntimeError(f"Failed to load dataset for {spec.alias} ({spec.path}). Last error: {last_error}")


def normalize_record(
    record: Dict[str, Any],
    spec: SourceSpec,
    row_number: int,
) -> Optional[Dict[str, Any]]:
    raw_messages = extract_messages(record)
    if not raw_messages:
        return None

    prompt_messages = trim_to_prompt_side(raw_messages)
    if not prompt_messages:
        return None

    prompt_raw = get_last_user_content(prompt_messages)
    if not prompt_raw:
        return None

    if spec.default_lang in {"ja", "en"}:
        prompt_language = spec.default_lang
    else:
        prompt_language = guess_language(prompt_raw)

    if spec.allowed_prompt_languages is not None and prompt_language not in set(spec.allowed_prompt_languages):
        return None

    final_messages = copy.deepcopy(prompt_messages)
    wrapped_for_japanese_answer = False
    if spec.wrap_english_prompt_to_japanese and prompt_language == "en":
        final_messages, wrapped_for_japanese_answer = append_english_to_japanese_request(final_messages)

    example_id = normalize_text(record.get("id") or record.get("sample_id") or record.get("uuid") or record.get("idx"))

    return {
        "messages": final_messages,
        "prompt_raw": prompt_raw,
        "prompt_language": prompt_language,
        "answer_language": "ja",
        "source_alias": spec.alias,
        "source_dataset": spec.path,
        "source_split": spec.split,
        "source_config_name": spec.config_name or "",
        "source_data_dir": spec.data_dir or "",
        "source_example_id": example_id,
        "source_row_number": row_number,
        "wrapped_for_japanese_answer": wrapped_for_japanese_answer,
        "num_prompt_messages": len(final_messages),
        "prompt_char_len": len(prompt_raw),
    }


def maybe_shuffle_dataset(ds: Any, seed: int, buffer_size: int) -> Any:
    if not hasattr(ds, "shuffle"):
        return ds
    try:
        # IterableDataset は buffer_size を使い、通常 Dataset は使わない
        return ds.shuffle(seed=seed, buffer_size=buffer_size)
    except TypeError:
        return ds.shuffle(seed=seed)


def collect_rows_from_source(
    spec: SourceSpec,
    target_count: int,
    cache_dir: Optional[Path],
    streaming: bool,
    seed: int,
    shuffle_buffer_size: int,
    dedup: bool,
    seen_keys: Optional[set[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if target_count <= 0:
        return [], {
            "alias": spec.alias,
            "requested": target_count,
            "collected": 0,
            "scanned": 0,
            "skipped_dedup": 0,
            "skipped_parse": 0,
            "streaming_used": streaming,
        }

    ds = load_source_dataset(spec=spec, cache_dir=cache_dir, streaming=streaming)
    ds = maybe_shuffle_dataset(ds, seed=seed, buffer_size=shuffle_buffer_size)

    rows: List[Dict[str, Any]] = []
    scanned = 0
    skipped_dedup = 0
    skipped_parse = 0

    iterator = enumerate(ds)
    pbar = tqdm(total=target_count, desc=f"collect:{spec.alias}")
    for row_number, record in iterator:
        scanned += 1
        normalized = normalize_record(record=record, spec=spec, row_number=row_number)
        if normalized is None:
            skipped_parse += 1
            continue

        if dedup:
            assert seen_keys is not None
            key = canonicalize_messages(normalized["messages"])
            if key in seen_keys:
                skipped_dedup += 1
                continue
            seen_keys.add(key)

        rows.append(normalized)
        pbar.update(1)
        if len(rows) >= target_count:
            break
    pbar.close()

    summary = {
        "alias": spec.alias,
        "requested": target_count,
        "collected": len(rows),
        "scanned": scanned,
        "skipped_dedup": skipped_dedup,
        "skipped_parse": skipped_parse,
        "streaming_used": streaming,
    }
    return rows, summary


def build_dataset_dict(records: List[Dict[str, Any]], seed: int) -> DatasetDict:
    rng = random.Random(seed)
    rng.shuffle(records)
    train = Dataset.from_list(records)
    return DatasetDict({"train": train})


def save_summary(
    output_dir: Path,
    args: argparse.Namespace,
    counts: Dict[str, int],
    source_summaries: List[Dict[str, Any]],
    total_rows: int,
) -> Path:
    summary_path = output_dir.parent / f"{output_dir.name}_build_summary.json"
    payload = {
        "output_dir": str(output_dir),
        "cache_dir": args.cache_dir,
        "counts_preset": args.counts_preset,
        "counts": counts,
        "seed": args.seed,
        "streaming": args.streaming,
        "dedup": args.dedup,
        "shuffle_buffer_size": args.shuffle_buffer_size,
        "total_rows": total_rows,
        "sources": source_summaries,
        "source_specs": {alias: spec.__dict__ for alias, spec in SOURCE_SPECS.items()},
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return summary_path


def main() -> None:
    args = parse_args()
    cache_dir = configure_hf_cache(args.cache_dir)
    counts = load_effective_counts(args.counts_preset, args.counts_json)

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and not args.keep_existing_output:
        raise SystemExit(
            f"出力先が既に存在します: {output_dir}\n"
            "上書き事故を避けるため停止しました。別ディレクトリを指定するか、既存ディレクトリを削除してください。"
        )

    seen_keys: Optional[set[str]] = set() if args.dedup else None
    all_records: List[Dict[str, Any]] = []
    source_summaries: List[Dict[str, Any]] = []

    # counts dict の順序で優先順位が決まる
    for alias, target_count in counts.items():
        spec = SOURCE_SPECS[alias]
        rows, summary = collect_rows_from_source(
            spec=spec,
            target_count=target_count,
            cache_dir=cache_dir,
            streaming=args.streaming,
            seed=args.seed,
            shuffle_buffer_size=args.shuffle_buffer_size,
            dedup=args.dedup,
            seen_keys=seen_keys,
        )
        all_records.extend(rows)
        source_summaries.append(summary)
        print(
            f"[INFO] {alias}: requested={summary['requested']} collected={summary['collected']} "
            f"scanned={summary['scanned']} skipped_parse={summary['skipped_parse']} "
            f"skipped_dedup={summary['skipped_dedup']}",
            file=sys.stderr,
        )

    dataset_dict = build_dataset_dict(all_records, seed=args.seed)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_dir))
    summary_path = save_summary(
        output_dir=output_dir,
        args=args,
        counts=counts,
        source_summaries=source_summaries,
        total_rows=len(all_records),
    )

    print(f"[DONE] saved dataset to: {output_dir}")
    print(f"[DONE] saved build summary to: {summary_path}")
    print(dataset_dict)


if __name__ == "__main__":
    main()
