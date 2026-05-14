#!/usr/bin/env python3
"""Build the tokenizer and encoded train/val/test splits.

Reads the glossary CSV listed in ``configs/default.yaml`` (or whichever
config file is passed via ``--config``), fits the source/target
vocabularies on the *full* corpus, splits the rows into train/val/test,
encodes each split, and writes everything under ``data/processed/``.

Why fit the vocab on the full corpus and not on the train split only?
The glossary task has a closed Korean character set: every syllable that
will ever be predicted is already known at corpus build time. Pretending
otherwise would only manufacture spurious ``<unk>`` targets in val/test.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``src/`` importable when running this script directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dict_model.dataset import read_glossary, split_rows, write_split_jsonl  # noqa: E402
from dict_model.tokenizer import Tokenizer                                   # noqa: E402
from dict_model.utils import load_yaml, set_seed                             # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["data"]["seed"])

    csv_path = ROOT / cfg["data"]["glossary_csv"]
    out_dir = ROOT / cfg["data"]["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_glossary(csv_path)
    print(f"[prepare] loaded {len(rows)} rows from {csv_path.relative_to(ROOT)}")

    tok_cfg = cfg["tokenizer"]
    tokenizer = Tokenizer.build(
        ((r.id, r.eng, r.ko) for r in rows),
        eng_lower=tok_cfg["eng_lower"],
        max_src_len=tok_cfg["max_src_len"],
        max_tgt_len=tok_cfg["max_tgt_len"],
        id_subword_min_freq=tok_cfg["id_subword_min_freq"],
    )
    tok_path = out_dir / "tokenizer.json"
    tokenizer.save(tok_path)
    print(f"[prepare] tokenizer: src_vocab={len(tokenizer.src_vocab)}  tgt_vocab={len(tokenizer.tgt_vocab)}  -> {tok_path.relative_to(ROOT)}")

    sp = cfg["data"]["splits"]
    splits = split_rows(
        rows,
        train=sp["train"],
        val=sp["val"],
        test=sp["test"],
        seed=cfg["data"]["seed"],
        stratify=cfg["data"]["stratify"],
    )
    for name, split in splits.items():
        path = out_dir / f"{name}.jsonl"
        write_split_jsonl(split, tokenizer, path)
        print(f"[prepare] {name:5s} {len(split):>5d} rows -> {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
