#!/usr/bin/env python3
"""Run inference against a trained checkpoint.

Usage:

    python scripts/infer.py --ckpt runs/exp1/best.pt --id abc_form_number --eng number
    python scripts/infer.py --ckpt runs/exp1/best.pt --batch some_inputs.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch                                                                 # noqa: E402

from dict_model.decode import DecodeConfig, beam_search_decode, greedy_decode  # noqa: E402
from dict_model.model import DictTransformer, ModelConfig                    # noqa: E402
from dict_model.tokenizer import Tokenizer                                   # noqa: E402


def load_model(ckpt_path: Path) -> tuple[DictTransformer, Tokenizer]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    tok_path = ckpt_path.parent / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(f"expected {tok_path} next to {ckpt_path}")
    tok = Tokenizer.load(tok_path)
    mcfg = ModelConfig(**ckpt["model_cfg"])
    model = DictTransformer(mcfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--id", dest="raw_id", help="single ID")
    ap.add_argument("--eng", dest="raw_eng", help="single English expression")
    ap.add_argument("--batch", type=Path, help="CSV with id,eng columns")
    ap.add_argument("--mode", choices=["greedy", "beam"], default="beam")
    ap.add_argument("--beam-size", type=int, default=4)
    ap.add_argument("--length-penalty", type=float, default=0.6)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    args = ap.parse_args()

    model, tok = load_model(args.ckpt)
    dcfg = DecodeConfig(
        beam_size=args.beam_size,
        length_penalty=args.length_penalty,
        max_new_tokens=args.max_new_tokens,
    )

    def predict(raw_id: str, raw_eng: str) -> str:
        if args.mode == "greedy":
            return greedy_decode(model, tok, raw_id, raw_eng, max_new_tokens=args.max_new_tokens)
        return beam_search_decode(model, tok, raw_id, raw_eng, cfg=dcfg)

    if args.batch:
        with open(args.batch, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pred = predict(row["id"], row["eng"])
                print(f"{row['id']}\t{row['eng']}\t{pred}")
    else:
        if not (args.raw_id and args.raw_eng):
            ap.error("provide --id and --eng (or use --batch)")
        pred = predict(args.raw_id, args.raw_eng)
        print(pred)


if __name__ == "__main__":
    main()
