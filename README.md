# 2_dict_model

A context-aware glossary translator built from scratch in PyTorch — **no
pre-trained LLM weights anywhere in the pipeline**. Every parameter is
randomly initialised; every tokenizer entry comes from the training corpus.

The task: given an identifier and an English expression, emit the correct
Korean translation. The identifier is what disambiguates the polysemy — the
same English word maps to many Korean glosses depending on context.

| id | eng | ko |
|----|-----|----|
| `abc_form_number` | number | 수자 |
| `abc_table_row_number` | row number | 행번호 |
| `id_shapes_number` | shapes number | 도형개수 |
| `abc_user_name` | name | 이름 |
| `abc_company_name` | name | 상호 |
| `abc_file_name` | name | 파일명 |

The pipeline is small enough to train on CPU in well under a minute on the
sample glossary (115 rows) yet exercises every concept you would scale up
for a real glossary: custom tokenizer, stratified splits with target-token
coverage, encoder-decoder Transformer, teacher-forced training, beam search.

---

## 1. Project goals

- Translate `(id, eng) -> ko` for a closed-domain glossary where the *id*
  carries the disambiguating context.
- Build the model from PyTorch primitives — no pretrained weights, no
  HF Transformers model checkpoints, no external trainer framework.
- Make the **dataset configuration** the centerpiece of the design: the
  tokenizer, the source-encoding scheme, and the splitting strategy are all
  shaped by the polysemy that defines the task.
- Be fully offline and reproducible — fixed seeds, deterministic vocab
  ordering, no network calls anywhere.

---

## 2. Architecture overview

```
                ┌─────────────────────────────────────────────┐
                │  glossary.csv  (id, eng, ko)                │
                └────────────────────┬────────────────────────┘
                                     │
                         scripts/prepare_data.py
                                     │
                ┌────────────────────▼────────────────────────┐
                │  data/processed/                            │
                │    tokenizer.json                           │
                │    train.jsonl / val.jsonl / test.jsonl     │
                └────────────────────┬────────────────────────┘
                                     │
                            scripts/train.py
                                     │
                ┌────────────────────▼────────────────────────┐
                │  runs/exp1/                                 │
                │    best.pt, last.pt, history.json,          │
                │    tokenizer.json                           │
                └────────────────────┬────────────────────────┘
                                     │
                             scripts/infer.py
                                     │
                              greedy / beam
                                     │
                                 Korean output
```

The model is a vanilla Vaswani-style encoder-decoder Transformer:
sinusoidal positional encodings, multi-head scaled-dot-product attention,
position-wise feed-forward blocks, pre-norm residuals, and a tied target
embedding / output projection. Default size on the sample data is
`d_model=128`, 3 encoder + 3 decoder layers, 4 heads — about 1.4M params.

Source sequences carry both halves of the input in one stream:

```
[BOS]  i:abc  i:form  i:number  [SEP]  e:number  [EOS]
```

The `i:` / `e:` namespacing keeps an ID atom (`i:number`) lexically
distinct from the same surface English word (`e:number`) inside a single
source-vocabulary embedding table — without that, the disambiguation signal
collapses.

Target sequences are Korean syllables (single Hangul codepoints):

```
[BOS]  도  형  개  수  [EOS]
```

---

## 3. Dataset configuration (the important bit)

Three properties of the data drive almost every design choice in
[src/dict_model/dataset.py](src/dict_model/dataset.py) and
[src/dict_model/tokenizer.py](src/dict_model/tokenizer.py):

### 3.1. The disambiguator lives in the ID

The same `eng` repeats dozens of times across rows; only the `id` changes.
A tokenizer that pools the ID into a single opaque string would force the
model to memorise specific IDs verbatim. Instead we split snake_case IDs on
`_` into atoms and feed every atom into the encoder. That way
`abc_form_number` and `abc_invoice_number` *share* the atoms `abc` and
`number`, and only their distinguishing atom (`form` vs `invoice`) differs.

### 3.2. ID atoms and English words live in disjoint sub-namespaces

If the source vocab merged ID atoms and English words by surface form,
`number` (the ID atom) and `number` (the English word) would hash to the
same embedding row — and the disambiguation signal would vanish.

We avoid that with tiny string prefixes: every ID atom is stored as
`i:atom` and every English word as `e:word`. Same one embedding table,
disjoint entries. See
[`Tokenizer._id_tok` / `_en_tok`](src/dict_model/tokenizer.py).

### 3.3. Stratified split for closed-vocab targets

The Korean target vocabulary is closed: every syllable the model will ever
predict is in `glossary.csv`. If a syllable lands only in val/test and
never in train, the model cannot possibly produce it.

`split_rows(..., stratify=True)` uses a greedy *coverage* strategy: it
first sends the minimum number of rows into the train split such that
every Korean syllable in the corpus appears at least once at training
time, then divides the remainder into train/val/test by the requested
fractions. See
[`split_rows`](src/dict_model/dataset.py).

### 3.4. Padding & masks

Source lengths vary because IDs have different atom counts. We pad inside
the collate function with `PAD_ID=0` and emit boolean key-padding masks
(`True` = padding) that the encoder/decoder consume directly. The decoder
also builds an additive causal mask on the fly. See
[`collate`](src/dict_model/dataset.py) and
[`DictTransformer.causal_mask`](src/dict_model/model.py).

### 3.5. CSV → JSONL contract

`scripts/prepare_data.py` does all string processing once and writes
encoded integer sequences to `data/processed/{train,val,test}.jsonl`. The
training loop reads JSONL only — no CSV parsing in the hot path.

```jsonc
{
  "id": "abc_form_number",
  "eng": "number",
  "ko":  "수자",
  "src": [2, 5, 7, 9, 4, 10, 3],   // BOS i:abc i:form i:number SEP e:number EOS
  "tgt": [2, 11, 12, 3]            //  BOS 수 자 EOS
}
```

---

## 4. Project layout

```
2_dict_model/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml            # hyperparameters + paths
├── data/
│   ├── raw/glossary.csv        # the only file you edit when adding terms
│   └── processed/              # tokenizer + encoded splits (generated)
├── scripts/
│   ├── prepare_data.py         # CSV -> tokenizer + JSONL splits
│   ├── train.py                # train the model
│   └── infer.py                # run a trained checkpoint
├── src/dict_model/
│   ├── __init__.py
│   ├── tokenizer.py            # id/eng/ko vocabularies + encode/decode
│   ├── dataset.py              # CSV reader, splits, JSONL, Dataset, collate
│   ├── model.py                # Transformer encoder-decoder (from scratch)
│   ├── trainer.py              # training loop, AdamW + Noam schedule
│   ├── decode.py               # greedy + beam-search decoding
│   └── utils.py                # seed, yaml loader, project_root
├── tests/
│   ├── test_tokenizer.py
│   ├── test_dataset.py
│   └── test_model.py           # includes overfit-one-batch sanity test
└── runs/                       # training output (checkpoints, history)
```

---

## 5. Quickstart

### 5.1. Install

```bash
cd 2_dict_model
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies are intentionally minimal: `torch`, `PyYAML`, and `pytest`
(for the test suite). No `transformers`, no `tokenizers`, no `datasets`.

### 5.2. Run the tests

```bash
python -m pytest -q
```

The suite covers tokenizer round-trips, stratified-split coverage, batch
collation, the causal-mask invariant, and a one-batch overfit check that
proves the model can actually drive loss to ~0.

### 5.3. Train

```bash
python scripts/prepare_data.py --config configs/default.yaml
python scripts/train.py        --config configs/default.yaml
```

Training writes checkpoints (`best.pt`, `last.pt`), a JSON history, and a
copy of the tokenizer to `runs/exp1/` (configurable via
`train.output_dir`).

### 5.4. Predict

Single example:

```bash
python scripts/infer.py --ckpt runs/exp1/best.pt \
    --id abc_table_row_number --eng "row number" --mode beam
# → 행번호
```

Batch mode (CSV with `id,eng` columns):

```bash
python scripts/infer.py --ckpt runs/exp1/best.pt --batch my_inputs.csv
```

### 5.5. Sample outputs (default config, beam=4)

After ~30 epochs of training on the sample glossary the model handles the
polysemy correctly on held-out IDs:

| id | eng | predicted |
|----|-----|-----------|
| `abc_form_number` | number | 수자 |
| `abc_table_row_number` | row number | 행번호 |
| `id_shapes_number` | shapes number | 도형개수 |
| `abc_user_name` | name | 이름 |
| `abc_company_name` | name | 상호 |
| `abc_file_name` | name | 파일명 |
| `abc_shirt_size` | size | 사이즈 |
| `abc_paper_size` | size | 규격 |
| `abc_text_color` | color | 색상 |
| `abc_hair_color` | color | 색깔 |
| `abc_user_type` | type | 유형 |
| `abc_file_type` | type | 형식 |

(Some near-misses remain at this dataset scale — e.g. `abc_image_size` /
`size` → 용량 instead of 크기. Adding a handful more rows for the
under-represented sense closes the gap.)

---

## 6. Configuration reference

All knobs live in [configs/default.yaml](configs/default.yaml). The four
sections you are most likely to touch:

| Section | Knob | What it does |
|---------|------|-------------|
| `data.splits` | `train / val / test` | fractions, must sum to 1.0 |
| `data.stratify` | bool | greedy syllable-coverage stratification (recommended `true` for small glossaries) |
| `tokenizer.eng_lower` | bool | lowercase English before tokenising |
| `tokenizer.max_src_len` / `max_tgt_len` | int | truncate the rare overlong examples |
| `model.d_model / n_heads / n_enc_layers / n_dec_layers` | int | Transformer size |
| `model.share_target_embedding` | bool | tie target embedding with the output projection |
| `train.lr / warmup_steps / label_smoothing` | float | optimisation |
| `train.device` | "auto"/"cpu"/"cuda" | runtime device |
| `infer.beam_size / length_penalty` | int / float | decoding |

---

## 7. Library usage

The project deliberately stays inside a tight set of dependencies. What each
library is used for, and what it is *not* used for:

| Library | Used for | Not used for |
|---------|----------|--------------|
| `torch` | tensors, autograd, `nn.Module`, `nn.Linear`, `nn.LayerNorm`, `nn.Embedding`, `nn.Dropout`, `AdamW`, `LambdaLR`, `DataLoader`, `Dataset` | `nn.Transformer*` modules — the encoder/decoder/attention are written by hand from primitives so the data flow is fully exposed |
| `PyYAML` | reading `configs/*.yaml` | anything else |
| `pytest` | the test suite | runtime |

Nothing from the `transformers`, `tokenizers`, `datasets`, or `huggingface_hub`
packages is imported anywhere. Every weight is randomly initialised.

---

## 8. Developer guide

### 8.1. Adding new glossary entries

Append rows to [data/raw/glossary.csv](data/raw/glossary.csv) (UTF-8, header
`id,eng,ko`), then re-run `prepare_data.py` and `train.py`. The tokenizer
will pick up any new ID atoms, English words, and Korean syllables
automatically.

### 8.2. Adding a new translation sense

When you add a new Korean gloss for an existing English word, make sure
the *disambiguating* ID atom is present in enough rows for the model to
learn the pattern. Two or three contrasting rows per sense is usually
enough at this scale — the sample glossary uses about five rows per sense.

### 8.3. Inspecting tokenization

```python
from dict_model.tokenizer import Tokenizer
tok = Tokenizer.load("data/processed/tokenizer.json")

print(tok.encode_source("abc_table_row_number", "row number"))
# [2, ...src ids..., 3]

print([tok.src_vocab.itos[i] for i in tok.encode_source("abc_form_number", "number")])
# ['<bos>', 'i:abc', 'i:form', 'i:number', '<sep>', 'e:number', '<eos>']
```

### 8.4. Inspecting a batch

```python
from torch.utils.data import DataLoader
from dict_model.dataset import DictDataset, collate, read_split_jsonl

ds = DictDataset(read_split_jsonl("data/processed/train.jsonl"))
batch = next(iter(DataLoader(ds, batch_size=4, collate_fn=collate)))
print(batch["src"].shape, batch["tgt_in"].shape, batch["src_kpm"].dtype)
```

### 8.5. Swapping in your own tokenizer / model

The five modules are deliberately small (~100–250 lines each) and have one
public class apiece, so substitutions are local:

- New target tokenisation (jamo-level, BPE, …) → re-implement
  `split_ko` and `decode_target` in `tokenizer.py`.
- Different model (e.g. encoder-only with a classification head) →
  replace `DictTransformer` in `model.py`; everything else stays.
- Different optimisation regime → edit `trainer.py`; the dataset/model
  contracts do not change.

### 8.6. Reproducibility

`utils.set_seed` seeds Python, NumPy (if available), and PyTorch.
Tokenizer vocabularies are constructed with a deterministic
frequency-then-alphabetical ordering, so two runs with the same CSV and
seed produce byte-identical `tokenizer.json` files.

---

## 9. Limits & future work

- **No subword fallback for unknown ID atoms.** A truly new ID at
  inference time (`abc_xyz_number` where `xyz` was never seen at train
  time) maps `xyz` to `<unk>`. A char-level fallback in
  `Tokenizer.encode_source` would address this.
- **Greedy coverage stratification can over-allocate to train.** If the
  requested train fraction is smaller than the minimum number of rows
  needed to cover every syllable, coverage wins and the fractions drift.
  The split routine logs this case but does not fail.
- **Single sense per (id, eng) pair.** The training objective and the
  loss assume one canonical Korean translation per row. Allowing multiple
  acceptable translations would need a `set`-valued target loss.
