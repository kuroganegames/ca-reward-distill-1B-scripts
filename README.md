# ca-reward-distill-1B scripts

**※ 本コード、モデルはすべてChatGPTのみを使用して作成されました。よって、全内容を公開します。**

この README は、添付されているスクリプト群から読み取れる使い方を整理したものです。主な目的は、

1. 日本語中心の prompt-only データセットを作る  
2. 生成モデルで候補応答を作り、教師 Reward Model で採点する  
3. 教師スコアを使って生徒 Reward Model を回帰蒸留で学習する  
4. 学習済み生徒 Reward Model を評価・推論する  
5. 必要に応じて pairwise 蒸留用データも作る  

という一連の流れを実行できるようにすることです。

---

## 1. 全体像

このコード群には、大きく分けて次の 2 系統があります。

### A. 回帰蒸留の本線

もっとも完成している主経路です。

- `build_mixed_prompt_dataset.py`  
  公開データセットから **prompt-only の mixed dataset** を作る
- `generate_teacher_dataset.py`  
  prompt から複数候補応答を生成し、**教師RMでスコア付けした dataset** を作る
- `train_student_rm_regression.py`  
  教師スコアを回帰ラベルとして **生徒RMを学習** する
- `evaluate_student_rm_against_teacher.py`  
  生徒RMの予測と教師RMスコアを比較評価する
- `score_student_rm_minimal.py`  
  学習済み生徒RMで単発/JSONL推論する

### B. pairwise 蒸留用の補助経路

- `build_pairwise_distillation_dataset.py`  
  教師スコア付きデータから `(chosen, rejected)` ペアを作る

この pairwise データセット生成器は入っていますが、**添付コード内に pairwise 学習本体は含まれていません**。したがって、すぐに end-to-end で回せるのは回帰蒸留のほうです。

---

## 2. 含まれるファイル一覧

| ファイル | 役割 |
|---|---|
| `build_mixed_prompt_dataset.py` | 公開 instruction/chat データから prompt-only dataset を構築 |
| `generate_teacher_dataset.py` | 候補応答生成 + 教師RM採点 |
| `build_pairwise_distillation_dataset.py` | 教師データから pairwise dataset を構築 |
| `train_student_rm_regression.py` | 教師スコア回帰で生徒RMを学習 |
| `evaluate_student_rm_against_teacher.py` | 生徒RMを教師スコア付きデータ上で評価 |
| `score_student_rm_minimal.py` | 学習済み生徒RMの最小推論スクリプト |
| `trial_counts_example.json` | mixed prompt dataset の件数例（小さめ） |
| `prod_like_counts_example.json` | mixed prompt dataset の件数例（大きめ） |
| `test_reload_mixed_dataset.py` | mixed prompt dataset の再読込テスト |
| `test_reload_teacher_dataset.py` | teacher dataset の再読込テスト |
| `test_reload_pairwise_dataset.py` | pairwise dataset の再読込テスト |
| `test_reload_student_rm_regression.py` | 学習済み生徒RMの簡易再確認 |
| `test_reload_student_rm_evaluation.py` | 評価出力の再確認 |
| `test_reload_hard_prompt_regression_dataset.py` | hard-prompt系出力の再読込テスト |

補足:

- `test_reload_hard_prompt_regression_dataset.py` は、`build_hard_prompt_regression_dataset.py` の出力を検証する前提ですが、その本体スクリプトは今回の添付には含まれていません。
- `test_reload_hard_prompt_regression_dataset.py` が 2 つありますが、内容は同一です。

---

## 3. 依存ライブラリ

最低限、以下が必要です。

```bash
pip install -U torch transformers accelerate datasets pyarrow sentencepiece numpy tqdm
```

マルチGPUで実行するなら、事前に `accelerate config` を済ませておくと楽です。

```bash
accelerate config
```

また、公開データセットやモデルを Hugging Face Hub から取得するので、ネットワーク接続と十分なディスク容量を前提にしてください。

---

## 4. まず押さえるべきワークフロー

最短の実行順は以下です。

```text
build_mixed_prompt_dataset.py
        ↓
generate_teacher_dataset.py
        ↓
train_student_rm_regression.py
        ↓
evaluate_student_rm_against_teacher.py
        ↓
score_student_rm_minimal.py
```

pairwise データが欲しいときだけ、教師データ生成後に

```text
generate_teacher_dataset.py
        ↓
build_pairwise_distillation_dataset.py
```

を追加します。

---

## 5. ステップ別の使い方

## 5.1 mixed prompt dataset を作る

### 目的

複数の公開データセットから **prompt 側だけ** を取り出し、日本語回答を期待する 1 つの prompt pool にまとめます。

### 入力

Hugging Face Hub 上の公開データセット。

### 出力

`datasets.DatasetDict({"train": Dataset})` を `save_to_disk()` したディレクトリ。

### 代表コマンド

```bash
python build_mixed_prompt_dataset.py \
  --output-dir ./mixed_prompt_pool_trial \
  --cache-dir /data/hf_cache \
  --counts-preset trial \
  --streaming
```

件数を JSON で上書きしたい場合:

```bash
python build_mixed_prompt_dataset.py \
  --output-dir ./mixed_prompt_pool_custom \
  --cache-dir /data/hf_cache \
  --counts-preset trial \
  --counts-json ./trial_counts_example.json \
  --streaming
```

### 主なオプション

- `--counts-preset {trial,prod_like}`  
  既定の件数セットを選ぶ
- `--counts-json path.json`  
  source ごとの件数を上書きする
- `--streaming / --no-streaming`  
  `load_dataset(..., streaming=True)` を使うかどうか
- `--dedup / --no-dedup`  
  正規化した `messages` ベースで重複除去するか
- `--shuffle-buffer-size`  
  streaming shuffle の buffer サイズ
- `--keep-existing-output`  
  出力先が存在しても停止しない

### 生成される主な列

- `messages`: prompt-side の chat messages
- `prompt_raw`: 最後の user 発話の生テキスト
- `prompt_language`: 推定/既知の入力言語
- `answer_language`: 期待する出力言語（既定 `ja`）
- `source_alias`, `source_dataset` などの source メタ情報

### 補足

- 英語データセット由来の prompt には、必要に応じて `Please answer in Japanese. / 日本語で回答してください。` が追記されます。
- `trial_counts_example.json` と `prod_like_counts_example.json` は、そのままコピーして編集できるサンプルです。

### 動作確認

```bash
python test_reload_mixed_dataset.py \
  --dataset-dir ./mixed_prompt_pool_trial \
  --summary-json ./mixed_prompt_pool_trial_build_summary.json
```

---

## 5.2 教師スコア付き dataset を作る

### 目的

prompt-only dataset から複数候補応答を生成し、教師 Reward Model でスコア付けした dataset を作ります。

### 既定モデル

- 教師RM: `cyberagent/ca-reward-3b-ja`
- 生成モデル: `sbintuitions/sarashina2.2-1b-instruct-v0.1`

### 入力

`build_mixed_prompt_dataset.py` の出力ディレクトリ。

### 出力

`output_dir/final_dataset/` に `save_to_disk()` 形式で保存されます。中間生成物として `parts/` と `state/` も作られます。

### 単GPU例

```bash
python generate_teacher_dataset.py \
  --prompt-dataset-dir ./mixed_prompt_pool_trial \
  --output-dir ./teacher_data_trial \
  --cache-dir /data/hf_cache \
  --generator-models sbintuitions/sarashina2.2-1b-instruct-v0.1 \
  --num-candidates-per-prompt 4 \
  --prompt-micro-batch-size 8 \
  --flush-prompt-count 64 \
  --teacher-batch-size 32 \
  --max-new-tokens 512
```

### マルチGPU例

```bash
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
```

### 主なオプション

- `--generator-models model1 model2 ...`  
  候補応答生成に使うモデルを複数指定可能
- `--num-candidates-per-prompt`  
  各 prompt あたりの候補数
- `--prompt-micro-batch-size`  
  生成時の prompt バッチサイズ
- `--teacher-batch-size`  
  教師RM採点時のバッチサイズ
- `--generator-max-input-length`  
  生成モデルの入力長上限
- `--teacher-max-length`  
  教師RMの入力長上限
- `--max-new-tokens`  
  1候補あたりの最大生成長
- `--temperature`, `--top-p`, `--repetition-penalty`  
  sampling 設定
- `--do-sample / --no-sample`  
  sampling の有無
- `--finalize-only`  
  既存 parquet parts を最終 dataset にまとめ直すだけにする
- `--allow-config-mismatch`  
  再開時の config 不一致を許容する

### 注意点

- `--no-sample` と `--num-candidates-per-prompt > 1` は同時に使えません。
- 中断再開に対応しており、rank ごと・generator ごとに parquet part を積み上げます。
- `finalize-only` は、すでに parts が揃っているときに最終 `save_to_disk()` を作り直す用途です。

### 主な出力列

- `messages`: prompt 側 messages
- `response`: 生成された候補応答
- `teacher_score`: 教師RMスコア
- `teacher_model`
- `generator_model`, `generator_key`, `generator_model_index`
- `candidate_index`
- `sample_id`
- `prompt_hash`, `response_hash`
- `prompt_raw`
- `rank`, `local_prompt_index`, `global_prompt_index`

### 動作確認

```bash
python test_reload_teacher_dataset.py \
  --dataset-dir ./teacher_data_trial/final_dataset \
  --summary-json ./teacher_data_trial/final_summary.json \
  --check-unique-sample-id
```

---

## 5.3 pairwise 蒸留用 dataset を作る（任意）

### 目的

教師スコア付き dataset を prompt ごとにまとめ、`chosen > rejected` となる pair を作ります。

### 入力

`generate_teacher_dataset.py` の `final_dataset`。

### 出力

`output_dir/final_dataset/` に `save_to_disk()` 形式で保存されます。

### 代表コマンド

```bash
python build_pairwise_distillation_dataset.py \
  --teacher-dataset-dir ./teacher_data_trial/final_dataset \
  --output-dir ./pairwise_trial \
  --cache-dir /data/hf_cache \
  --pairing-strategy best_vs_rest \
  --min-score-margin 0.0 \
  --max-pairs-per-prompt 4 \
  --val-ratio 0.02
```

### 主なペア戦略

- `best_vs_rest`: 最高得点候補 vs それ以外
- `top_bottom`: 最上位 vs 最下位
- `adjacent`: スコア順で隣接候補同士
- `all_pairs`: 全組み合わせ

### 主なオプション

- `--min-score-margin`  
  `chosen_score - rejected_score` の最小閾値
- `--max-pairs-per-prompt`  
  1 prompt あたりの最大ペア数
- `--min-candidates-per-prompt`  
  ペア化に必要な最低候補数
- `--drop-empty-responses / --keep-empty-responses`
- `--dedupe-response-hash / --keep-duplicate-responses`
- `--store-conversation-columns`  
  `chosen_messages` / `rejected_messages` も保存する
- `--val-ratio`  
  prompt 単位で validation を切る比率
- `--split-seed`  
  prompt hash に対する split seed
- `--finalize-only`  
  既存 parts を dataset にまとめ直すだけにする

### 主な出力列

- `messages`: prompt 側 messages
- `chosen`, `rejected`: 応答文字列
- `chosen_score`, `rejected_score`, `score_margin`
- `pair_id`, `pair_strategy`, `pair_rank_within_prompt`
- `chosen_sample_id`, `rejected_sample_id`
- `prompt_hash`, `global_prompt_index`
- `chosen_generator_model`, `rejected_generator_model`
- 必要に応じて `chosen_messages`, `rejected_messages`

### 動作確認

```bash
python test_reload_pairwise_dataset.py \
  --dataset-dir ./pairwise_trial/final_dataset \
  --summary-json ./pairwise_trial/final_summary.json \
  --check-unique-pair-id \
  --check-no-prompt-leak
```

### 重要

添付コードでは、この pairwise dataset を直接学習に使うスクリプトは入っていません。  
すぐに生徒RMを学習したい場合は、次節の **回帰蒸留** を使ってください。

---

## 5.4 生徒 Reward Model を回帰蒸留で学習する

### 目的

教師RMの `teacher_score` を回帰ラベルとして、生徒 Reward Model を学習します。

### 既定の生徒初期モデル

- `sbintuitions/sarashina2.2-1b-instruct-v0.1`

### 入力

`generate_teacher_dataset.py` の `final_dataset`。

### 出力

```text
output_dir/
  checkpoints/
  tokenized_dataset/
  final_model/
  preprocess_summary.json
  training_summary.json
  score_normalization.json
  train_args.json
```

### 単GPU例

```bash
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
```

### マルチGPU例

```bash
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
```

### 学習の中身

- `messages + response` を 1 本の chat text にして tokenizer の chat template で整形
- `teacher_score` をラベルにする
- 既定では `train` split のスコアに対する **z-score 正規化** をかけて学習
- 損失は既定 `MSE`。`Huber` も選択可能
- `Trainer` ベースで checkpoint 保存、resume、best model restore に対応

### 主なオプション

#### データ関連

- `--validation-ratio`  
  validation split が無いときの holdout 比率
- `--prompt-hash-column`, `--sample-id-column`  
  stable hash split に使う列
- `--text-column`  
  すでに全文 text 列がある場合に使用
- `--max-train-samples`, `--max-eval-samples`  
  試運転用
- `--shuffle-before-select`  
  sample 制限前に shuffle

#### 前処理関連

- `--max-length`
- `--preprocessing-num-workers`
- `--overwrite-tokenized-dataset`
- `--keep-raw-columns`

#### ラベル変換関連

- `--score-normalization {none,train_zscore}`
- `--score-clip-abs`
- `--loss-type {mse,huber}`
- `--huber-delta`

#### 学習関連

- `--per-device-train-batch-size`
- `--per-device-eval-batch-size`
- `--gradient-accumulation-steps`
- `--learning-rate`
- `--weight-decay`
- `--num-train-epochs`
- `--max-steps`
- `--warmup-ratio`
- `--lr-scheduler-type`
- `--gradient-checkpointing`
- `--bf16`, `--fp16`, `--tf32`
- `--torch-compile`
- `--resume-from-checkpoint last`

### 動作確認

```bash
python test_reload_student_rm_regression.py \
  --model-dir ./student_rm_regression_trial/final_model \
  --teacher-dataset-dir ./teacher_data_trial/final_dataset
```

---

## 5.5 生徒RMを教師データ上で評価する

### 目的

学習済み生徒RMの予測値と教師RMスコアを比較し、相関や順位再現性を見ます。

### 入力

- 学習済み生徒RM (`final_model`)
- 教師スコア付き dataset (`final_dataset`)

### 出力

- `evaluation_summary.json`
- 必要なら `predictions_dataset/`
- rank ごとの一時ファイル `parts/rankXXXXX.npz`

### 単GPU例

```bash
python evaluate_student_rm_against_teacher.py \
  --model-dir ./student_rm_regression_trial/final_model \
  --teacher-dataset-dir ./teacher_data_trial/final_dataset \
  --output-dir ./student_rm_eval \
  --batch-size 8 \
  --max-length 2048
```

### マルチGPU例

```bash
accelerate launch --multi_gpu --num_processes 2 evaluate_student_rm_against_teacher.py \
  --model-dir ./student_rm_regression_trial/final_model \
  --teacher-dataset-dir ./teacher_data_trial/final_dataset \
  --output-dir ./student_rm_eval \
  --batch-size 8 \
  --max-length 2048
```

### 主な評価指標

#### 全体指標

- Pearson 相関
- Spearman 相関
- MSE
- MAE

#### prompt 単位の順位指標

- top1 agreement
- pairwise accuracy (micro / macro)
- prompt-wise Spearman 平均

### 主なオプション

- `--pairwise-margin`  
  教師スコア差が小さいペアを pairwise accuracy 集計から除外
- `--save-predictions-dataset`  
  生徒予測を付けた dataset を保存
- `--keep-original-columns`  
  predictions dataset に元列を全部残す
- `--overwrite-output`  
  既存出力の上書き許可

### 予測 dataset 保存例

```bash
python evaluate_student_rm_against_teacher.py \
  --model-dir ./student_rm_regression_trial/final_model \
  --teacher-dataset-dir ./teacher_data_trial/final_dataset \
  --output-dir ./student_rm_eval \
  --batch-size 8 \
  --max-length 2048 \
  --save-predictions-dataset
```

### 動作確認

```bash
python test_reload_student_rm_evaluation.py \
  --predictions-dataset-dir ./student_rm_eval/predictions_dataset \
  --summary-json ./student_rm_eval/evaluation_summary.json
```

---

## 5.6 学習済み生徒RMで最小推論する

### 目的

手元の prompt/response をさっと採点したいときの最小スクリプトです。

### 単発: prompt + response

```bash
python score_student_rm_minimal.py \
  --model-dir ./student_rm_regression_trial/final_model \
  --prompt "富士山について短く説明して" \
  --response "富士山は日本で最も高い山です。"
```

### 単発: messages(JSON) + response

```bash
python score_student_rm_minimal.py \
  --model-dir ./student_rm_regression_trial/final_model \
  --messages-json '[{"role":"user","content":"こんにちは"}]' \
  --response "こんにちは。どうしましたか？"
```

### JSONL 一括推論

```bash
python score_student_rm_minimal.py \
  --model-dir ./student_rm_regression_trial/final_model \
  --input-jsonl ./records.jsonl \
  --output-jsonl ./records_scored.jsonl \
  --batch-size 8
```

### JSONL の想定フォーマット

各行は次のいずれかです。

```json
{"messages": [...], "response": "..."}
```

```json
{"prompt": "...", "response": "..."}
```

```json
{"text": "<already formatted chat text>"}
```

### 出力列

- `student_score_norm`: 学習時の正規化空間でのスコア
- `student_score_denorm`: `score_normalization.json` があれば教師RMスケールへ戻した値

---

## 6. 各段階の入出力対応表

| 段階 | 入力 | 出力 |
|---|---|---|
| mixed prompt 構築 | 公開HF dataset 群 | prompt-only DatasetDict |
| teacher dataset 生成 | prompt-only DatasetDict | teacher-score付き DatasetDict |
| pairwise dataset 構築 | teacher-score付き DatasetDict | chosen/rejected DatasetDict |
| student 回帰学習 | teacher-score付き DatasetDict | 学習済み `final_model/` |
| student 評価 | `final_model/` + teacher-score付き DatasetDict | summary JSON + 任意の predictions dataset |
| 最小推論 | `final_model/` + 任意入力 | score JSON / JSONL |

---

## 7. 出力ディレクトリの見方

## 7.1 `generate_teacher_dataset.py`

```text
teacher_data_trial/
  config.json
  state/
    progress_rank00.json
    manifest_rank00.jsonl
    rank00_last_error.txt
  parts/
    gen_000_<model_key>/
      rank00/
        part_l00000000_l00000064.parquet
  final_dataset/
  final_summary.json
```

## 7.2 `build_pairwise_distillation_dataset.py`

```text
pairwise_trial/
  config.json
  state/
    progress.json
    manifest_train.jsonl
    manifest_validation.jsonl
    commit_log.jsonl
    last_error.txt
  parts/
    train/
      part_000000.parquet
    validation/
      part_000000.parquet
  final_dataset/
  final_summary.json
```

## 7.3 `train_student_rm_regression.py`

```text
student_rm_regression_trial/
  checkpoints/
  tokenized_dataset/
  final_model/
  preprocess_summary.json
  training_summary.json
  score_normalization.json
  train_args.json
```

## 7.4 `evaluate_student_rm_against_teacher.py`

```text
student_rm_eval/
  parts/
    rank00000.npz
    rank00001.npz
  evaluation_summary.json
  eval_args.json
  predictions_dataset/   # --save-predictions-dataset 時のみ
```

---

## 8. 再開・安全装置まわり

このコード群は、長時間処理の中断再開をかなり意識して作られています。

### `build_mixed_prompt_dataset.py`

- 既存 `output-dir` があると既定では停止
- 誤上書きを避けたいときに安全

### `generate_teacher_dataset.py`

- rank ごと / generator ごとに parquet part を保存
- 再実行すると、連続して完成済みの local prompt 範囲を検出して再開
- config が違うと停止
- 強制再開したいときだけ `--allow-config-mismatch`
- dataset まとめ直しだけしたいときは `--finalize-only`

### `build_pairwise_distillation_dataset.py`

- 中間 part と progress を残しながら再開可能
- `commit_log.jsonl` を source of truth として使う
- `--finalize-only` あり

### `train_student_rm_regression.py`

- `Trainer` の checkpoint を保存
- `--resume-from-checkpoint last` で再開可能
- tokenize 済み dataset も別途 `save_to_disk()` して再利用可能

### `evaluate_student_rm_against_teacher.py`

- `output-dir` が空でないと既定では停止
- 上書きしたいときは `--overwrite-output`

---

## 9. よくある使い分け

## 9.1 まず最小で一通り確認したい

1. `build_mixed_prompt_dataset.py --counts-preset trial`
2. `generate_teacher_dataset.py --max-new-tokens 256 --num-candidates-per-prompt 2`
3. `train_student_rm_regression.py --num-train-epochs 1 --max-train-samples 1000`
4. `evaluate_student_rm_against_teacher.py --max-samples 1000`

## 9.2 生成品質の多様性を上げたい

- `generate_teacher_dataset.py` の `--generator-models` を複数指定する
- `--num-candidates-per-prompt` を増やす
- `--temperature` や `--top-p` を調整する

## 9.3 順位学習用データも欲しい

- 教師 dataset 生成後に `build_pairwise_distillation_dataset.py` を実行する
- ただし、添付コードだけでは pairwise 学習本体は実行できない

## 9.4 評価時に教師スケールへ戻した値を見たい

- `train_student_rm_regression.py` が `score_normalization.json` を保存する
- `evaluate_student_rm_against_teacher.py` と `score_student_rm_minimal.py` はそれを読んで `student_score_denorm` を計算する

---

## 10. 推奨の最小 end-to-end 実行例

```bash
# 1) prompt pool
python build_mixed_prompt_dataset.py \
  --output-dir ./mixed_prompt_pool_trial \
  --cache-dir /data/hf_cache \
  --counts-preset trial \
  --streaming

# 2) teacher data
python generate_teacher_dataset.py \
  --prompt-dataset-dir ./mixed_prompt_pool_trial \
  --output-dir ./teacher_data_trial \
  --cache-dir /data/hf_cache \
  --generator-models sbintuitions/sarashina2.2-1b-instruct-v0.1 \
  --num-candidates-per-prompt 4 \
  --prompt-micro-batch-size 8 \
  --flush-prompt-count 64 \
  --teacher-batch-size 32 \
  --max-new-tokens 512

# 3) student RM regression distillation
python train_student_rm_regression.py \
  --teacher-dataset-dir ./teacher_data_trial/final_dataset \
  --output-dir ./student_rm_regression_trial \
  --cache-dir /data/hf_cache \
  --max-length 2048 \
  --per-device-train-batch-size 2 \
  --gradient-accumulation-steps 16 \
  --learning-rate 1e-5 \
  --num-train-epochs 1 \
  --gradient-checkpointing

# 4) evaluation
python evaluate_student_rm_against_teacher.py \
  --model-dir ./student_rm_regression_trial/final_model \
  --teacher-dataset-dir ./teacher_data_trial/final_dataset \
  --output-dir ./student_rm_eval \
  --batch-size 8 \
  --max-length 2048 \
  --save-predictions-dataset

# 5) one-shot scoring
python score_student_rm_minimal.py \
  --model-dir ./student_rm_regression_trial/final_model \
  --prompt "富士山について短く説明して" \
  --response "富士山は日本で最も高い山です。"
```

---

## 11. この添付一式だけでは不足しているもの

今回の添付だけを見る限り、以下は入っていません。

- pairwise dataset を使って生徒RMを学習するスクリプト
- `test_reload_hard_prompt_regression_dataset.py` が前提にしている `build_hard_prompt_regression_dataset.py`

そのため、現時点で迷ったら **teacher dataset → regression distillation → evaluation** の流れを使うのが安全です。

---

## 12. 迷ったときの判断基準

- **prompt pool を作りたい** → `build_mixed_prompt_dataset.py`
- **教師スコア付き候補データが欲しい** → `generate_teacher_dataset.py`
- **生徒RMを今すぐ学習したい** → `train_student_rm_regression.py`
- **順位ペア `(chosen, rejected)` が欲しい** → `build_pairwise_distillation_dataset.py`
- **生徒RMの性能を見たい** → `evaluate_student_rm_against_teacher.py`
- **単発で採点したい** → `score_student_rm_minimal.py`

