#!/usr/bin/env python3
"""保存済み prompt dataset から候補応答を生成し、ca-reward-3b-ja でスコア付けした教師データを作る。

主な要件:
- 入力: `datasets.load_from_disk()` で読み込める prompt-only dataset
- 生成: `transformers` の CausalLM + chat template
- 教師スコア: `cyberagent/ca-reward-3b-ja` (SequenceClassification)
- 並列化: `accelerate` の multi-process / multi-GPU
- 中断再開: rank ごとの parquet part + progress JSON + manifest JSONL
- 出力: 最終的に `DatasetDict.save_to_disk()` した教師データ

想定起動例:
    accelerate launch --multi_gpu --num_processes 2 generate_teacher_dataset.py \
        --prompt-dataset-dir ./mixed_prompt_pool_trial \
        --output-dir ./teacher_data_trial \
        --cache-dir /data/hf_cache \
        --generator-models sbintuitions/sarashina2.2-1b-instruct-v0.1 \
        --num-candidates-per-prompt 4 \
        --prompt-micro-batch-size 8 \
        --flush-prompt-count 64 \
        --teacher-batch-size 32 \
        --max-new-tokens 512

出力ディレクトリ構成:
    output_dir/
      config.json                       # 実行設定 (resume 時に整合性チェック)
      state/
        progress_rank00.json            # rank ごとの進捗
        progress_rank01.json
        manifest_rank00.jsonl           # flush 済み part の記録
        manifest_rank01.jsonl
        rank00_last_error.txt           # 直近エラー
      parts/
        gen_000_<model_key>/
          rank00/part_l00000000_l00000064.parquet
          rank01/...
      final_dataset/                    # load_from_disk 可能な最終成果物
      final_summary.json
"""

from __future__ import annotations

import argparse
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
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("torch が見つかりません。`pip install -U torch` を実行してください。") from exc

try:
    from accelerate import Accelerator
except ImportError as exc:  # pragma: no cover
    raise SystemExit("accelerate が見つかりません。`pip install -U accelerate` を実行してください。") from exc

try:
    from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
except ImportError as exc:  # pragma: no cover
    raise SystemExit("datasets が見つかりません。`pip install -U datasets pyarrow` を実行してください。") from exc

try:
    from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, set_seed
except ImportError as exc:  # pragma: no cover
    raise SystemExit("transformers が見つかりません。`pip install -U transformers sentencepiece` を実行してください。") from exc

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pa = None
    pq = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(iterable=None, *args, **kwargs):
        return iterable


DEFAULT_TEACHER_MODEL = "cyberagent/ca-reward-3b-ja"
DEFAULT_GENERATOR_MODEL = "sbintuitions/sarashina2.2-1b-instruct-v0.1"
DEFAULT_FINAL_SUBDIR = "final_dataset"

PART_FILENAME_RE = re.compile(r"^part_l(?P<start>\d{8})_l(?P<end>\d{8})\.parquet$")
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")

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
]



@dataclass(frozen=True)
class RunPaths:
    root: Path
    state_dir: Path
    parts_dir: Path
    final_dataset_dir: Path
    config_path: Path
    summary_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate teacher-labeled dataset with transformers + accelerate + datasets.")
    parser.add_argument("--prompt-dataset-dir", type=str, required=True, help="前段の save_to_disk 済み prompt dataset ディレクトリ")
    parser.add_argument("--output-dir", type=str, required=True, help="作業用 root ディレクトリ。parts/state/final_dataset をこの下に保存")
    parser.add_argument("--cache-dir", type=str, default=None, help="Hugging Face cache root。手動指定可")
    parser.add_argument("--final-dataset-subdir", type=str, default=DEFAULT_FINAL_SUBDIR, help="output-dir 配下の最終 save_to_disk ディレクトリ名")
    parser.add_argument("--train-split", type=str, default="train", help="DatasetDict 入力時に使う split 名")
    parser.add_argument("--messages-column", type=str, default="messages", help="prompt chat messages 列名")
    parser.add_argument("--max-prompts", type=int, default=None, help="先頭 N prompt のみ使う。試運転用")

    parser.add_argument(
        "--generator-models",
        nargs="+",
        default=[DEFAULT_GENERATOR_MODEL],
        help="候補応答を生成するモデルIDを 1 個以上指定。順番に処理する",
    )
    parser.add_argument("--teacher-model", type=str, default=DEFAULT_TEACHER_MODEL, help="報酬モデルID")

    parser.add_argument("--num-candidates-per-prompt", type=int, default=4, help="各 prompt あたり生成する候補数")
    parser.add_argument("--prompt-micro-batch-size", type=int, default=8, help="1 GPU あたりの生成 micro-batch prompt 数")
    parser.add_argument(
        "--flush-prompt-count",
        type=int,
        default=64,
        help="何 prompt 分たまったら parquet part として flush するか。中断時に失う最大作業量を制御",
    )
    parser.add_argument("--teacher-batch-size", type=int, default=32, help="教師RMのスコアリング batch size (prompt-response pair 数)")

    parser.add_argument("--generator-max-input-length", type=int, default=3072, help="生成モデルへの入力最大長")
    parser.add_argument("--teacher-max-length", type=int, default=4096, help="教師RMの最大長")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="各候補の最大生成長")
    parser.add_argument("--temperature", type=float, default=0.8, help="sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="sampling top-p")
    parser.add_argument("--repetition-penalty", type=float, default=1.05, help="generation repetition penalty")

    parser.add_argument("--do-sample", dest="do_sample", action="store_true", help="sampling を有効化 (既定)")
    parser.add_argument("--no-sample", dest="do_sample", action="store_false", help="sampling を無効化")
    parser.set_defaults(do_sample=True)

    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="モデル読込 dtype",
    )
    parser.add_argument("--seed", type=int, default=42, help="base seed。batch ごとに派生 seed を作る")
    parser.add_argument("--parquet-compression", type=str, default="zstd", help="中間 parquet の圧縮方式")
    parser.add_argument("--save-max-shard-size", type=str, default="2GB", help="final_dataset.save_to_disk の max_shard_size")

    parser.add_argument("--finalize-only", action="store_true", help="生成は行わず、既存 parquet part を最終 dataset にまとめ直す")
    parser.add_argument("--allow-config-mismatch", action="store_true", help="既存 config と引数が違っても resume を強行する")
    parser.add_argument("--trust-remote-code", action="store_true", help="from_pretrained(..., trust_remote_code=True) を許可する")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_candidates_per_prompt <= 0:
        raise SystemExit("--num-candidates-per-prompt は 1 以上にしてください。")
    if args.prompt_micro_batch_size <= 0:
        raise SystemExit("--prompt-micro-batch-size は 1 以上にしてください。")
    if args.flush_prompt_count <= 0:
        raise SystemExit("--flush-prompt-count は 1 以上にしてください。")
    if args.teacher_batch_size <= 0:
        raise SystemExit("--teacher-batch-size は 1 以上にしてください。")
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens は 1 以上にしてください。")
    if args.max_prompts is not None and args.max_prompts <= 0:
        raise SystemExit("--max-prompts は 1 以上にしてください。")
    if (not args.do_sample) and args.num_candidates_per_prompt > 1:
        raise SystemExit(
            "--no-sample で --num-candidates-per-prompt > 1 は使えません。"
            "複数候補が必要なら sampling を有効化してください。"
        )


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


def set_torch_perf_flags() -> None:
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass


def resolve_torch_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if dtype_name == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if dtype_name == "bfloat16":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    if dtype_name == "float16":
        return torch.float16 if device.type == "cuda" else torch.float32
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype_name}")


def build_paths(output_dir: str, final_dataset_subdir: str) -> RunPaths:
    root = Path(output_dir).expanduser().resolve()
    return RunPaths(
        root=root,
        state_dir=root / "state",
        parts_dir=root / "parts",
        final_dataset_dir=root / final_dataset_subdir,
        config_path=root / "config.json",
        summary_path=root / "final_summary.json",
    )


def safe_model_key(model_id: str) -> str:
    model_id = model_id.replace("/", "__")
    return SAFE_NAME_RE.sub("_", model_id)


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
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_prompt_dataset(prompt_dataset_dir: str, train_split: str, max_prompts: Optional[int], messages_column: str):
    ds_any = load_from_disk(str(Path(prompt_dataset_dir).expanduser().resolve()))
    if isinstance(ds_any, DatasetDict):
        if train_split not in ds_any:
            raise SystemExit(f"入力 DatasetDict に split={train_split!r} がありません。利用可能: {list(ds_any.keys())}")
        ds = ds_any[train_split]
    else:
        ds = ds_any

    if messages_column not in ds.column_names:
        raise SystemExit(f"入力 dataset に必須列 {messages_column!r} がありません。利用可能: {ds.column_names}")

    if max_prompts is not None:
        max_prompts = min(max_prompts, len(ds))
        ds = ds.select(range(max_prompts))
    return ds


def contiguous_shard_bounds(total: int, world_size: int, rank: int) -> Tuple[int, int]:
    if world_size <= 0:
        raise ValueError(f"world_size must be positive: {world_size}")
    if not (0 <= rank < world_size):
        raise ValueError(f"invalid rank/world_size: rank={rank}, world_size={world_size}")
    base = total // world_size
    remainder = total % world_size
    if rank < remainder:
        start = rank * (base + 1)
        end = start + base + 1
    else:
        start = remainder * (base + 1) + (rank - remainder) * base
        end = start + base
    return start, end


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, list):
        return "\n".join(str(x) for x in text if x is not None).strip()
    return str(text).strip()


def canonicalize_messages(messages: Sequence[Dict[str, Any]]) -> str:
    normalized = []
    for m in messages:
        normalized.append({
            "role": str(m.get("role", "user")),
            "content": normalize_text(m.get("content")),
        })
    return json.dumps(normalized, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def compute_prompt_hash(messages: Sequence[Dict[str, Any]]) -> str:
    return hashlib.sha1(canonicalize_messages(messages).encode("utf-8")).hexdigest()


def compute_response_hash(response: str) -> str:
    return hashlib.sha1(response.encode("utf-8")).hexdigest()


def extract_last_user_text(messages: Sequence[Dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return normalize_text(message.get("content"))
    return ""


def prepare_run_config(args: argparse.Namespace, total_prompts: int, world_size: int) -> Dict[str, Any]:
    return {
        "created_at": now_iso(),
        "prompt_dataset_dir": str(Path(args.prompt_dataset_dir).expanduser().resolve()),
        "output_dir": str(Path(args.output_dir).expanduser().resolve()),
        "final_dataset_subdir": args.final_dataset_subdir,
        "cache_dir": str(Path(args.cache_dir).expanduser().resolve()) if args.cache_dir else None,
        "train_split": args.train_split,
        "messages_column": args.messages_column,
        "max_prompts": args.max_prompts,
        "generator_models": list(args.generator_models),
        "teacher_model": args.teacher_model,
        "num_candidates_per_prompt": args.num_candidates_per_prompt,
        "prompt_micro_batch_size": args.prompt_micro_batch_size,
        "flush_prompt_count": args.flush_prompt_count,
        "teacher_batch_size": args.teacher_batch_size,
        "generator_max_input_length": args.generator_max_input_length,
        "teacher_max_length": args.teacher_max_length,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "do_sample": args.do_sample,
        "dtype": args.dtype,
        "seed": args.seed,
        "parquet_compression": args.parquet_compression,
        "save_max_shard_size": args.save_max_shard_size,
        "trust_remote_code": args.trust_remote_code,
        "world_size": world_size,
        "total_prompts": total_prompts,
        "command": sys.argv,
        "script_name": Path(sys.argv[0]).name,
    }


def init_or_validate_run_config(
    paths: RunPaths,
    args: argparse.Namespace,
    total_prompts: int,
    world_size: int,
    allow_config_mismatch: bool,
) -> Dict[str, Any]:
    current = prepare_run_config(args=args, total_prompts=total_prompts, world_size=world_size)

    paths.root.mkdir(parents=True, exist_ok=True)
    paths.state_dir.mkdir(parents=True, exist_ok=True)
    paths.parts_dir.mkdir(parents=True, exist_ok=True)

    if not paths.config_path.exists():
        atomic_write_json(paths.config_path, current)
        return current

    existing = read_json(paths.config_path)
    compare_keys = [
        "prompt_dataset_dir",
        "output_dir",
        "final_dataset_subdir",
        "train_split",
        "messages_column",
        "max_prompts",
        "generator_models",
        "teacher_model",
        "num_candidates_per_prompt",
        "prompt_micro_batch_size",
        "generator_max_input_length",
        "teacher_max_length",
        "max_new_tokens",
        "temperature",
        "top_p",
        "repetition_penalty",
        "do_sample",
        "dtype",
        "seed",
        "trust_remote_code",
        "world_size",
        "total_prompts",
    ]
    mismatches: List[str] = []
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


def progress_path(paths: RunPaths, rank: int) -> Path:
    return paths.state_dir / f"progress_rank{rank:02d}.json"


def manifest_path(paths: RunPaths, rank: int) -> Path:
    return paths.state_dir / f"manifest_rank{rank:02d}.jsonl"


def error_log_path(paths: RunPaths, rank: int) -> Path:
    return paths.state_dir / f"rank{rank:02d}_last_error.txt"


def write_progress(paths: RunPaths, rank: int, payload: Dict[str, Any]) -> None:
    payload = dict(payload)
    payload["rank"] = rank
    payload["updated_at"] = now_iso()
    atomic_write_json(progress_path(paths, rank), payload)


def write_error(paths: RunPaths, rank: int, exc: BaseException) -> None:
    text = f"[{now_iso()}] {type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
    atomic_write_text(error_log_path(paths, rank), text)


def generator_parts_dir(paths: RunPaths, generator_index: int, generator_key: str, rank: int) -> Path:
    return paths.parts_dir / f"gen_{generator_index:03d}_{generator_key}" / f"rank{rank:02d}"


def part_file_path(parts_dir: Path, start_local_idx: int, end_local_idx: int) -> Path:
    return parts_dir / f"part_l{start_local_idx:08d}_l{end_local_idx:08d}.parquet"


def scan_committed_end(parts_dir: Path, local_total_prompts: int) -> Tuple[int, List[Dict[str, int]]]:
    if not parts_dir.exists():
        return 0, []

    parts: List[Dict[str, int]] = []
    for path in sorted(parts_dir.glob("part_l*.parquet")):
        match = PART_FILENAME_RE.match(path.name)
        if not match:
            continue
        start = int(match.group("start"))
        end = int(match.group("end"))
        if not (0 <= start <= end <= local_total_prompts):
            continue
        parts.append({"start": start, "end": end})

    contiguous_end = 0
    contiguous_parts: List[Dict[str, int]] = []
    for part in sorted(parts, key=lambda x: (x["start"], x["end"])):
        if part["start"] != contiguous_end:
            break
        contiguous_end = part["end"]
        contiguous_parts.append(part)
    return contiguous_end, contiguous_parts


def load_generator_tokenizer(model_id: str, cache_dir: Optional[Path], trust_remote_code: bool):
    kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
    if getattr(tokenizer, "pad_token_id", None) is None:
        if getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise SystemExit(f"generator tokenizer に pad_token / eos_token がありません: {model_id}")
    tokenizer.padding_side = "left"
    return tokenizer


def load_teacher_tokenizer(model_id: str, cache_dir: Optional[Path], trust_remote_code: bool):
    kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
    if getattr(tokenizer, "pad_token_id", None) is None:
        if getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise SystemExit(f"teacher tokenizer に pad_token / eos_token がありません: {model_id}")
    tokenizer.padding_side = "right"
    return tokenizer


def load_generator_model(
    model_id: str,
    device: torch.device,
    torch_dtype: torch.dtype,
    cache_dir: Optional[Path],
    trust_remote_code: bool,
):
    kwargs: Dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": trust_remote_code,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.to(device)
    model.eval()
    return model


def load_teacher_model(
    model_id: str,
    device: torch.device,
    torch_dtype: torch.dtype,
    cache_dir: Optional[Path],
    trust_remote_code: bool,
):
    kwargs: Dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": trust_remote_code,
        "num_labels": 1,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, **kwargs)
    model.to(device)
    model.eval()
    return model


def batch_seed(base_seed: int, generator_index: int, global_prompt_start: int) -> int:
    return int(base_seed + generator_index * 1_000_003 + global_prompt_start * 97)


def build_generation_texts(
    tokenizer: Any,
    prompt_messages_list: Sequence[Sequence[Dict[str, Any]]],
) -> List[str]:
    texts: List[str] = []
    for messages in prompt_messages_list:
        text = tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=True,
        )
        texts.append(text)
    return texts


def generate_candidate_responses(
    model: Any,
    tokenizer: Any,
    prompt_messages_list: Sequence[Sequence[Dict[str, Any]]],
    device: torch.device,
    max_input_length: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    num_candidates_per_prompt: int,
    seed: int,
) -> List[List[Dict[str, Any]]]:
    generation_texts = build_generation_texts(tokenizer=tokenizer, prompt_messages_list=prompt_messages_list)
    encoded = tokenizer(
        generation_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_input_length,
        add_special_tokens=False,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    prompt_padded_len = int(encoded["input_ids"].shape[1])

    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "num_return_sequences": num_candidates_per_prompt,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": repetition_penalty,
        "use_cache": True,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with torch.inference_mode():
        if do_sample:
            set_seed(seed)
        outputs = model.generate(**encoded, **generation_kwargs)

    generated_only = outputs[:, prompt_padded_len:]
    decoded = tokenizer.batch_decode(generated_only, skip_special_tokens=True)

    candidates_per_prompt: List[List[Dict[str, Any]]] = []
    batch_size = len(prompt_messages_list)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1
    for prompt_idx in range(batch_size):
        group: List[Dict[str, Any]] = []
        for candidate_idx in range(num_candidates_per_prompt):
            flat_idx = prompt_idx * num_candidates_per_prompt + candidate_idx
            token_ids = generated_only[flat_idx]
            token_len = int((token_ids != pad_id).sum().item()) if pad_id >= 0 else int(token_ids.numel())
            response = decoded[flat_idx].strip()
            group.append({
                "candidate_index": candidate_idx,
                "response": response,
                "response_token_len": token_len,
                "response_char_len": len(response),
            })
        candidates_per_prompt.append(group)
    return candidates_per_prompt


def build_teacher_texts(
    tokenizer: Any,
    prompt_messages_list: Sequence[Sequence[Dict[str, Any]]],
    candidates_per_prompt: Sequence[Sequence[Dict[str, Any]]],
) -> List[str]:
    texts: List[str] = []
    for messages, candidates in zip(prompt_messages_list, candidates_per_prompt):
        for candidate in candidates:
            full_chat = list(messages) + [{"role": "assistant", "content": candidate["response"]}]
            text = tokenizer.apply_chat_template(full_chat, tokenize=False, add_generation_prompt=False)
            texts.append(text)
    return texts


def score_teacher_texts(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> List[float]:
    scores: List[float] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            batch_texts = list(texts[start:end])
            batch = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            )
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits.squeeze(-1)
            scores.extend([float(x) for x in logits.detach().float().cpu().tolist()])
    return scores


def prompt_batch_to_rows(
    prompt_batch: Dict[str, List[Any]],
    prompt_messages_list: Sequence[Sequence[Dict[str, Any]]],
    candidates_per_prompt: Sequence[Sequence[Dict[str, Any]]],
    teacher_scores: Sequence[float],
    prompt_metadata_columns: Sequence[str],
    generator_model: str,
    generator_key: str,
    generator_index: int,
    teacher_model: str,
    local_prompt_start: int,
    global_prompt_start: int,
    generation_seed: int,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    flat_score_idx = 0
    batch_size = len(prompt_messages_list)

    for prompt_offset in range(batch_size):
        messages = list(prompt_messages_list[prompt_offset])
        prompt_hash = compute_prompt_hash(messages)
        prompt_raw = ""
        if "prompt_raw" in prompt_batch:
            prompt_raw = normalize_text(prompt_batch["prompt_raw"][prompt_offset])
        if not prompt_raw:
            prompt_raw = extract_last_user_text(messages)

        base_row: Dict[str, Any] = {
            "messages": messages,
            "prompt_hash": prompt_hash,
            "prompt_raw": prompt_raw,
            "generator_model": generator_model,
            "generator_key": generator_key,
            "generator_model_index": generator_index,
            "teacher_model": teacher_model,
            "rank": None,  # main loop で埋める
            "local_prompt_index": local_prompt_start + prompt_offset,
            "global_prompt_index": global_prompt_start + prompt_offset,
            "generation_seed": generation_seed,
            "generation_do_sample": bool(args.do_sample),
            "generation_temperature": float(args.temperature),
            "generation_top_p": float(args.top_p),
            "generation_repetition_penalty": float(args.repetition_penalty),
            "generation_max_new_tokens": int(args.max_new_tokens),
            "generator_max_input_length": int(args.generator_max_input_length),
            "teacher_max_length": int(args.teacher_max_length),
        }
        for col in prompt_metadata_columns:
            if col == "prompt_raw":
                continue
            if col in prompt_batch:
                base_row[col] = prompt_batch[col][prompt_offset]

        for candidate in candidates_per_prompt[prompt_offset]:
            response = candidate["response"]
            response_hash = compute_response_hash(response)
            row = dict(base_row)
            row.update({
                "candidate_index": int(candidate["candidate_index"]),
                "response": response,
                "response_hash": response_hash,
                "response_token_len": int(candidate["response_token_len"]),
                "response_char_len": int(candidate["response_char_len"]),
                "teacher_score": float(teacher_scores[flat_score_idx]),
            })
            sample_basis = f"{prompt_hash}|{generator_model}|{candidate['candidate_index']}|{response_hash}"
            row["sample_id"] = hashlib.sha1(sample_basis.encode("utf-8")).hexdigest()
            rows.append(row)
            flat_score_idx += 1

    if flat_score_idx != len(teacher_scores):
        raise RuntimeError(f"teacher_scores の数が合いません: used={flat_score_idx}, total={len(teacher_scores)}")
    return rows


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


def gather_prompt_metadata_columns(prompt_ds: Any, messages_column: str) -> List[str]:
    columns = [c for c in PROMPT_METADATA_CANDIDATES if c in prompt_ds.column_names and c != messages_column]
    return columns


def finalize_dataset(
    paths: RunPaths,
    run_config: Dict[str, Any],
    cache_dir: Optional[Path],
    total_prompts: int,
) -> Dict[str, Any]:
    world_size = int(run_config["world_size"])
    generator_models: List[str] = list(run_config["generator_models"])

    # まず「すべての rank/generator がローカル shard を最後まで書けているか」を確認する。
    generator_completion: List[Dict[str, Any]] = []
    for generator_index, generator_model in enumerate(generator_models):
        generator_key = safe_model_key(generator_model)
        generator_payload = {
            "generator_index": generator_index,
            "generator_model": generator_model,
            "generator_key": generator_key,
            "ranks": [],
        }
        for rank in range(world_size):
            shard_start, shard_end = contiguous_shard_bounds(total=total_prompts, world_size=world_size, rank=rank)
            local_total = shard_end - shard_start
            rank_parts_dir = generator_parts_dir(paths, generator_index, generator_key, rank)
            committed_end, contiguous_parts = scan_committed_end(rank_parts_dir, local_total_prompts=local_total)
            generator_payload["ranks"].append({
                "rank": rank,
                "local_total_prompts": local_total,
                "committed_end": committed_end,
                "num_parts": len(contiguous_parts),
            })
            if committed_end != local_total:
                raise SystemExit(
                    "finalize を中止しました。まだ完了していない rank があります: "
                    f"generator={generator_model}, rank={rank}, committed_end={committed_end}, local_total={local_total}"
                )
        generator_completion.append(generator_payload)

    part_files = sorted(str(p) for p in paths.parts_dir.glob("gen_*/*/part_l*.parquet"))
    if not part_files:
        raise SystemExit(f"parquet part が見つかりません: {paths.parts_dir}")

    dataset_cache_dir = str(cache_dir / "datasets") if cache_dir is not None else None
    merged = load_dataset(
        "parquet",
        data_files={"train": part_files},
        cache_dir=dataset_cache_dir,
    )
    if not isinstance(merged, DatasetDict):
        merged = DatasetDict({"train": merged})

    if paths.final_dataset_dir.exists():
        shutil.rmtree(paths.final_dataset_dir)
    merged.save_to_disk(str(paths.final_dataset_dir), max_shard_size=run_config.get("save_max_shard_size", "2GB"))

    summary = {
        "finalized_at": now_iso(),
        "output_dir": str(paths.root),
        "final_dataset_dir": str(paths.final_dataset_dir),
        "total_rows": int(len(merged["train"])),
        "total_part_files": len(part_files),
        "world_size": world_size,
        "total_prompts": total_prompts,
        "generator_completion": generator_completion,
        "columns": list(merged["train"].column_names),
    }
    atomic_write_json(paths.summary_path, summary)
    return summary


def print_rank_header(accelerator: Accelerator, text: str) -> None:
    accelerator.print(text)


def main() -> None:
    args = parse_args()
    validate_args(args)
    cache_dir = configure_hf_cache(args.cache_dir)
    set_torch_perf_flags()

    accelerator = Accelerator()
    device = accelerator.device
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    torch_dtype = resolve_torch_dtype(args.dtype, device)

    paths = build_paths(args.output_dir, args.final_dataset_subdir)

    prompt_ds = load_prompt_dataset(
        prompt_dataset_dir=args.prompt_dataset_dir,
        train_split=args.train_split,
        max_prompts=args.max_prompts,
        messages_column=args.messages_column,
    )
    total_prompts = len(prompt_ds)

    if accelerator.is_main_process:
        run_config = init_or_validate_run_config(
            paths=paths,
            args=args,
            total_prompts=total_prompts,
            world_size=world_size,
            allow_config_mismatch=args.allow_config_mismatch,
        )
        print_rank_header(
            accelerator,
            f"[INFO] total_prompts={total_prompts} world_size={world_size} output_dir={paths.root}",
        )
    accelerator.wait_for_everyone()

    run_config = read_json(paths.config_path)

    if args.finalize_only:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            summary = finalize_dataset(paths=paths, run_config=run_config, cache_dir=cache_dir, total_prompts=total_prompts)
            print(f"[DONE] finalized dataset to: {paths.final_dataset_dir}")
            print(f"[DONE] summary: {paths.summary_path}")
            print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    local_global_start, local_global_end = contiguous_shard_bounds(total_prompts, world_size, rank)
    local_total_prompts = local_global_end - local_global_start
    prompt_metadata_columns = gather_prompt_metadata_columns(prompt_ds, messages_column=args.messages_column)

    progress_payload: Dict[str, Any] = {
        "status": "initializing",
        "prompt_dataset_dir": str(Path(args.prompt_dataset_dir).expanduser().resolve()),
        "total_prompts": total_prompts,
        "local_global_start": local_global_start,
        "local_global_end": local_global_end,
        "local_total_prompts": local_total_prompts,
        "world_size": world_size,
        "device": str(device),
        "generator_models": list(args.generator_models),
        "teacher_model": args.teacher_model,
        "generators": {},
    }
    write_progress(paths, rank, progress_payload)

    teacher_tokenizer = load_teacher_tokenizer(
        model_id=args.teacher_model,
        cache_dir=cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    teacher_model = load_teacher_model(
        model_id=args.teacher_model,
        device=device,
        torch_dtype=torch_dtype,
        cache_dir=cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    teacher_model.config.pad_token_id = teacher_tokenizer.pad_token_id

    try:
        for generator_index, generator_model_id in enumerate(args.generator_models):
            generator_key = safe_model_key(generator_model_id)
            rank_parts_dir = generator_parts_dir(paths, generator_index, generator_key, rank)
            committed_end, contiguous_parts = scan_committed_end(rank_parts_dir, local_total_prompts=local_total_prompts)

            progress_payload["status"] = "running"
            progress_payload["active_generator_index"] = generator_index
            progress_payload["active_generator_model"] = generator_model_id
            progress_payload["active_generator_key"] = generator_key
            progress_payload["generators"][generator_key] = {
                "generator_index": generator_index,
                "generator_model": generator_model_id,
                "local_total_prompts": local_total_prompts,
                "committed_local_prompt_index": committed_end,
                "buffer_start_local_prompt_index": None,
                "next_local_prompt_index": committed_end,
                "num_existing_parts": len(contiguous_parts),
            }
            write_progress(paths, rank, progress_payload)

            if committed_end >= local_total_prompts:
                progress_payload["generators"][generator_key]["finished"] = True
                write_progress(paths, rank, progress_payload)
                if rank == 0:
                    print(f"[INFO] generator already complete, skipping: {generator_model_id}")
                continue

            generator_tokenizer = load_generator_tokenizer(
                model_id=generator_model_id,
                cache_dir=cache_dir,
                trust_remote_code=args.trust_remote_code,
            )
            generator_model = load_generator_model(
                model_id=generator_model_id,
                device=device,
                torch_dtype=torch_dtype,
                cache_dir=cache_dir,
                trust_remote_code=args.trust_remote_code,
            )
            try:
                if hasattr(generator_model, "generation_config"):
                    generator_model.generation_config.pad_token_id = generator_tokenizer.pad_token_id
                    if getattr(generator_tokenizer, "eos_token_id", None) is not None:
                        generator_model.generation_config.eos_token_id = generator_tokenizer.eos_token_id

                buffer_rows: List[Dict[str, Any]] = []
                buffer_start_local: Optional[int] = None
                next_local_idx = committed_end

                pbar = tqdm(
                    total=local_total_prompts,
                    initial=committed_end,
                    desc=f"rank{rank}:{generator_key}",
                    disable=(rank != 0),
                )
                while next_local_idx < local_total_prompts:
                    batch_local_start = next_local_idx
                    batch_local_end = min(batch_local_start + args.prompt_micro_batch_size, local_total_prompts)
                    batch_global_start = local_global_start + batch_local_start
                    batch_global_end = local_global_start + batch_local_end

                    prompt_batch = prompt_ds[batch_global_start:batch_global_end]
                    prompt_messages_list = prompt_batch[args.messages_column]

                    generation_seed_value = batch_seed(
                        base_seed=args.seed,
                        generator_index=generator_index,
                        global_prompt_start=batch_global_start,
                    )

                    try:
                        candidates_per_prompt = generate_candidate_responses(
                            model=generator_model,
                            tokenizer=generator_tokenizer,
                            prompt_messages_list=prompt_messages_list,
                            device=device,
                            max_input_length=args.generator_max_input_length,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=args.do_sample,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            repetition_penalty=args.repetition_penalty,
                            num_candidates_per_prompt=args.num_candidates_per_prompt,
                            seed=generation_seed_value,
                        )
                        teacher_texts = build_teacher_texts(
                            tokenizer=teacher_tokenizer,
                            prompt_messages_list=prompt_messages_list,
                            candidates_per_prompt=candidates_per_prompt,
                        )
                        teacher_scores = score_teacher_texts(
                            model=teacher_model,
                            tokenizer=teacher_tokenizer,
                            texts=teacher_texts,
                            device=device,
                            max_length=args.teacher_max_length,
                            batch_size=args.teacher_batch_size,
                        )
                        batch_rows = prompt_batch_to_rows(
                            prompt_batch=prompt_batch,
                            prompt_messages_list=prompt_messages_list,
                            candidates_per_prompt=candidates_per_prompt,
                            teacher_scores=teacher_scores,
                            prompt_metadata_columns=prompt_metadata_columns,
                            generator_model=generator_model_id,
                            generator_key=generator_key,
                            generator_index=generator_index,
                            teacher_model=args.teacher_model,
                            local_prompt_start=batch_local_start,
                            global_prompt_start=batch_global_start,
                            generation_seed=generation_seed_value,
                            args=args,
                        )
                        for row in batch_rows:
                            row["rank"] = rank
                    except Exception as exc:  # noqa: BLE001
                        progress_payload["status"] = "error"
                        progress_payload["error_generator_key"] = generator_key
                        progress_payload["error_local_prompt_start"] = batch_local_start
                        progress_payload["error_local_prompt_end"] = batch_local_end
                        write_progress(paths, rank, progress_payload)
                        write_error(paths, rank, exc)
                        raise

                    if buffer_start_local is None:
                        buffer_start_local = batch_local_start
                    buffer_rows.extend(batch_rows)
                    next_local_idx = batch_local_end

                    progress_payload["generators"][generator_key].update({
                        "buffer_start_local_prompt_index": buffer_start_local,
                        "next_local_prompt_index": next_local_idx,
                        "committed_local_prompt_index": committed_end,
                        "buffer_prompt_count": 0 if buffer_start_local is None else next_local_idx - buffer_start_local,
                        "buffer_row_count": len(buffer_rows),
                    })
                    write_progress(paths, rank, progress_payload)

                    if rank == 0:
                        pbar.update(batch_local_end - batch_local_start)

                    need_flush = (
                        buffer_start_local is not None
                        and (
                            (next_local_idx - buffer_start_local) >= args.flush_prompt_count
                            or next_local_idx >= local_total_prompts
                        )
                    )
                    if need_flush:
                        assert buffer_start_local is not None
                        part_path = part_file_path(
                            parts_dir=rank_parts_dir,
                            start_local_idx=buffer_start_local,
                            end_local_idx=next_local_idx,
                        )
                        if part_path.exists():
                            raise RuntimeError(f"既存 part と衝突しました: {part_path}")

                        bytes_written = write_rows_to_parquet(
                            rows=buffer_rows,
                            path=part_path,
                            compression=args.parquet_compression,
                        )
                        append_jsonl(
                            manifest_path(paths, rank),
                            {
                                "written_at": now_iso(),
                                "generator_index": generator_index,
                                "generator_model": generator_model_id,
                                "generator_key": generator_key,
                                "rank": rank,
                                "part_path": str(part_path),
                                "local_prompt_start": buffer_start_local,
                                "local_prompt_end": next_local_idx,
                                "num_rows": len(buffer_rows),
                                "bytes_written": bytes_written,
                            },
                        )

                        committed_end = next_local_idx
                        buffer_rows = []
                        buffer_start_local = None
                        progress_payload["generators"][generator_key].update({
                            "committed_local_prompt_index": committed_end,
                            "buffer_start_local_prompt_index": None,
                            "next_local_prompt_index": committed_end,
                            "buffer_prompt_count": 0,
                            "buffer_row_count": 0,
                        })
                        write_progress(paths, rank, progress_payload)

                if rank == 0:
                    pbar.close()

                progress_payload["generators"][generator_key].update({
                    "committed_local_prompt_index": committed_end,
                    "buffer_start_local_prompt_index": None,
                    "next_local_prompt_index": committed_end,
                    "finished": True,
                })
                write_progress(paths, rank, progress_payload)

            finally:
                try:
                    del generator_model
                except Exception:
                    pass
                try:
                    del generator_tokenizer
                except Exception:
                    pass
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        progress_payload["status"] = "waiting_for_finalize"
        write_progress(paths, rank, progress_payload)
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            summary = finalize_dataset(paths=paths, run_config=run_config, cache_dir=cache_dir, total_prompts=total_prompts)
            print(f"[DONE] final dataset saved to: {paths.final_dataset_dir}")
            print(f"[DONE] summary saved to: {paths.summary_path}")
            print(json.dumps(summary, ensure_ascii=False, indent=2))

        accelerator.wait_for_everyone()
        progress_payload["status"] = "done"
        write_progress(paths, rank, progress_payload)

    finally:
        try:
            del teacher_model
        except Exception:
            pass
        try:
            del teacher_tokenizer
        except Exception:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
