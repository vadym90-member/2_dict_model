#!/usr/bin/env python3
"""Train a ``DictTransformer`` on the encoded splits.

Assumes ``scripts/prepare_data.py`` has already run and that
``data/processed/{tokenizer.json,train.jsonl,val.jsonl,test.jsonl}`` exist.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch                                                            # noqa: E402
from torch.utils.data import DataLoader                                 # noqa: E402

from dict_model.dataset import DictDataset, collate, read_split_jsonl   # noqa: E402
from dict_model.model import DictTransformer, ModelConfig               # noqa: E402
from dict_model.tokenizer import Tokenizer                              # noqa: E402
from dict_model.trainer import Trainer, TrainConfig                     # noqa: E402
from dict_model.utils import load_yaml, set_seed                        # noqa: E402


def _make_loader(split_path: Path, batch_size: int, shuffle: bool) -> DataLoader:
    ds = DictDataset(read_split_jsonl(split_path))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["data"]["seed"])

    processed = ROOT / cfg["data"]["processed_dir"]
    tok = Tokenizer.load(processed / "tokenizer.json")

    train_loader = _make_loader(processed / "train.jsonl", cfg["train"]["batch_size"], shuffle=True)
    val_loader = _make_loader(processed / "val.jsonl", cfg["train"]["batch_size"], shuffle=False)

    mcfg = cfg["model"]
    model = DictTransformer(
        ModelConfig(
            src_vocab_size=len(tok.src_vocab),
            tgt_vocab_size=len(tok.tgt_vocab),
            d_model=mcfg["d_model"],
            n_heads=mcfg["n_heads"],
            n_enc_layers=mcfg["n_enc_layers"],
            n_dec_layers=mcfg["n_dec_layers"],
            d_ff=mcfg["d_ff"],
            dropout=mcfg["dropout"],
            max_len=max(tok.max_src_len, tok.max_tgt_len) + 4,
            share_target_embedding=mcfg["share_target_embedding"],
        )
    )

    tcfg = cfg["train"]
    out_dir = ROOT / tcfg["output_dir"]
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=TrainConfig(
            epochs=tcfg["epochs"],
            batch_size=tcfg["batch_size"],
            lr=tcfg["lr"],
            warmup_steps=tcfg["warmup_steps"],
            weight_decay=tcfg["weight_decay"],
            label_smoothing=tcfg["label_smoothing"],
            grad_clip=tcfg["grad_clip"],
            log_every=tcfg["log_every"],
            eval_every_epoch=tcfg["eval_every_epoch"],
            output_dir=str(out_dir),
            device=tcfg["device"],
        ),
    )
    trainer.fit()

    # Copy tokenizer next to checkpoints so inference loads it without the config.
    (out_dir / "tokenizer.json").write_text(
        (processed / "tokenizer.json").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    print(f"[train] artifacts saved under {out_dir}")


if __name__ == "__main__":
    main()
