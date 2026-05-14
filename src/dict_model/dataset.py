"""Dataset + DataLoader plumbing for the dictionary model.

DATASET CONFIGURATION — design notes
------------------------------------
The disambiguation signal in this task lives entirely in the **(id, eng)
pair**, not in the surface English alone. Three concrete consequences for
how the dataset is built:

1.  *Row uniqueness key*
    A row is identified by the pair ``(id, eng)``. The same ``eng`` repeats
    constantly across rows with different IDs — that is the whole reason the
    task is interesting — so deduplication must not collapse on ``eng``.

2.  *Split strategy*
    A naive random split risks leaking an exact ``(id, eng)`` example into
    both train and val. Worse, since each ``id`` is unique in our glossary,
    a uniform random split is fine — but we still **stratify by Korean
    target token** so each split sees every Korean syllable at training
    time. Without that, a rare syllable in val that the model never saw at
    train time would be impossible to predict.

3.  *Padding & masking*
    Source lengths vary (id atom counts differ). We pad inside the collate
    function with the PAD token (id 0) and emit boolean key-padding masks
    the encoder/decoder consume directly.

The encoded splits and the fitted tokenizer are persisted under
``data/processed/`` so training never has to redo the CSV parse.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset

from .tokenizer import PAD_ID, Tokenizer, split_ko


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

@dataclass
class GlossaryRow:
    id: str
    eng: str
    ko: str


def read_glossary(csv_path: str | Path) -> list[GlossaryRow]:
    """Read a ``id,eng,ko`` CSV (with header) into a list of rows."""
    rows: list[GlossaryRow] = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        required = {"id", "eng", "ko"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"glossary CSV missing columns: {sorted(missing)}")
        for r in reader:
            rows.append(GlossaryRow(id=r["id"].strip(), eng=r["eng"].strip(), ko=r["ko"].strip()))
    return rows


# ---------------------------------------------------------------------------
# Train / val / test splitting
# ---------------------------------------------------------------------------

def split_rows(
    rows: Sequence[GlossaryRow],
    *,
    train: float,
    val: float,
    test: float,
    seed: int,
    stratify: bool,
) -> dict[str, list[GlossaryRow]]:
    """Return train/val/test lists.

    With ``stratify=True`` we use a greedy *coverage* strategy: we first
    funnel the minimum number of rows into train so that every Korean
    syllable in the corpus appears at least once at training time, and only
    then divide the remainder into the requested train/val/test fractions.
    This avoids the failure mode where a rare syllable lands only in val or
    test and is therefore unlearnable.

    Without stratification we just shuffle and slice by the requested
    fractions."""
    total = train + val + test
    if not (0.999 <= total <= 1.001):
        raise ValueError(f"split fractions must sum to 1.0, got {total}")
    rng = random.Random(seed)
    n = len(rows)
    n_train = int(round(n * train))
    n_val = int(round(n * val))
    # n_test absorbs any rounding drift so the splits partition exactly.

    idxs = list(range(n))
    rng.shuffle(idxs)

    if not stratify:
        train_idxs = idxs[:n_train]
        val_idxs = idxs[n_train : n_train + n_val]
        test_idxs = idxs[n_train + n_val :]
    else:
        all_syls = {c for r in rows for c in split_ko(r.ko)}
        covered: set[str] = set()
        coverage_pick: list[int] = []
        rest: list[int] = []
        for i in idxs:
            syls = set(split_ko(rows[i].ko))
            if syls - covered:
                coverage_pick.append(i)
                covered |= syls
            else:
                rest.append(i)
            if covered == all_syls:
                # already fully covered — push the rest into `rest`
                pass
        # `coverage_pick` is guaranteed to live in train; pad train with rest.
        train_idxs = list(coverage_pick)
        while len(train_idxs) < n_train and rest:
            train_idxs.append(rest.pop(0))
        # If coverage required more rows than n_train, spill (this would mean
        # the requested train fraction is too small to be fully stratified).
        if len(train_idxs) > n_train:
            spill = train_idxs[n_train:]
            train_idxs = train_idxs[:n_train]
            rest = spill + rest
        rng.shuffle(rest)
        val_idxs = rest[:n_val]
        test_idxs = rest[n_val:]

    return {
        "train": [rows[i] for i in train_idxs],
        "val":   [rows[i] for i in val_idxs],
        "test":  [rows[i] for i in test_idxs],
    }


# ---------------------------------------------------------------------------
# Encoded artefacts on disk
# ---------------------------------------------------------------------------

def write_split_jsonl(rows: Sequence[GlossaryRow], tokenizer: Tokenizer, path: Path) -> None:
    """Encode each row and append it to a JSONL file under ``processed/``.

    Storing encoded ids (not just raw text) means the training script does
    zero string processing in the hot path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            src = tokenizer.encode_source(r.id, r.eng)
            tgt = tokenizer.encode_target(r.ko)
            f.write(json.dumps({"id": r.id, "eng": r.eng, "ko": r.ko, "src": src, "tgt": tgt}, ensure_ascii=False))
            f.write("\n")


def read_split_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


# ---------------------------------------------------------------------------
# PyTorch Dataset / collate
# ---------------------------------------------------------------------------

class DictDataset(Dataset):
    """Wraps the encoded JSONL split as a torch ``Dataset``.

    Each item is a dict of python lists; tensorisation happens in the
    collate function so we avoid copying tensors that will get re-padded
    anyway."""

    def __init__(self, encoded: Sequence[dict]):
        self.encoded = list(encoded)

    def __len__(self) -> int:
        return len(self.encoded)

    def __getitem__(self, idx: int) -> dict:
        ex = self.encoded[idx]
        return {"src": ex["src"], "tgt": ex["tgt"], "id": ex["id"], "eng": ex["eng"], "ko": ex["ko"]}


def collate(batch: Sequence[dict]) -> dict:
    """Pad src/tgt to the batch max-length and return tensors + masks.

    Returned tensors
    ----------------
    * ``src``      (B, S)   long, padded with PAD_ID
    * ``tgt_in``   (B, T-1) long, decoder input (teacher forcing)
    * ``tgt_out``  (B, T-1) long, decoder labels shifted by one
    * ``src_kpm``  (B, S)   bool, True where the position is *padding*
    * ``tgt_kpm``  (B, T-1) bool, True where the position is *padding*
    """
    B = len(batch)
    s_max = max(len(b["src"]) for b in batch)
    t_max = max(len(b["tgt"]) for b in batch)
    src = torch.full((B, s_max), PAD_ID, dtype=torch.long)
    tgt = torch.full((B, t_max), PAD_ID, dtype=torch.long)
    for i, b in enumerate(batch):
        src[i, : len(b["src"])] = torch.tensor(b["src"], dtype=torch.long)
        tgt[i, : len(b["tgt"])] = torch.tensor(b["tgt"], dtype=torch.long)
    tgt_in = tgt[:, :-1].contiguous()
    tgt_out = tgt[:, 1:].contiguous()
    src_kpm = src == PAD_ID
    tgt_kpm = tgt_in == PAD_ID
    return {
        "src": src,
        "tgt_in": tgt_in,
        "tgt_out": tgt_out,
        "src_kpm": src_kpm,
        "tgt_kpm": tgt_kpm,
        "raw": [{"id": b["id"], "eng": b["eng"], "ko": b["ko"]} for b in batch],
    }
