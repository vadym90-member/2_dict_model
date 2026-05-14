from __future__ import annotations

from pathlib import Path

import torch

from dict_model.dataset import (
    DictDataset,
    GlossaryRow,
    collate,
    read_glossary,
    read_split_jsonl,
    split_rows,
    write_split_jsonl,
)
from dict_model.tokenizer import PAD_ID, Tokenizer


def _rows() -> list[GlossaryRow]:
    return [
        GlossaryRow("abc_form_number", "number", "수자"),
        GlossaryRow("abc_invoice_number", "number", "수자"),
        GlossaryRow("abc_table_row_number", "row number", "행번호"),
        GlossaryRow("abc_grid_row_number", "row number", "행번호"),
        GlossaryRow("id_shapes_number", "shapes number", "도형개수"),
        GlossaryRow("id_pages_number", "pages number", "페이지개수"),
    ]


def test_read_glossary(tmp_path: Path):
    p = tmp_path / "g.csv"
    p.write_text("id,eng,ko\nabc_form_number,number,수자\n", encoding="utf-8")
    rows = read_glossary(p)
    assert len(rows) == 1
    assert rows[0].id == "abc_form_number" and rows[0].ko == "수자"


def test_split_rows_stratified_covers_all_syllables():
    rows = _rows()
    splits = split_rows(rows, train=0.66, val=0.17, test=0.17, seed=0, stratify=True)
    train_syls = {ch for r in splits["train"] for ch in r.ko}
    all_syls = {ch for r in rows for ch in r.ko}
    assert train_syls == all_syls, "every Korean syllable must appear in train"


def test_collate_shapes_and_masks():
    rows = _rows()
    tok = Tokenizer.build([(r.id, r.eng, r.ko) for r in rows])
    encoded = [
        {"id": r.id, "eng": r.eng, "ko": r.ko,
         "src": tok.encode_source(r.id, r.eng),
         "tgt": tok.encode_target(r.ko)}
        for r in rows
    ]
    ds = DictDataset(encoded)
    batch = collate([ds[i] for i in range(4)])
    assert batch["src"].dtype == torch.long
    assert batch["tgt_in"].shape == batch["tgt_out"].shape
    assert batch["tgt_in"].shape[1] == batch["tgt_in"].shape[1]
    # Padding mask consistency: True iff token equals PAD.
    assert torch.equal(batch["src_kpm"], batch["src"] == PAD_ID)
    assert torch.equal(batch["tgt_kpm"], batch["tgt_in"] == PAD_ID)


def test_jsonl_roundtrip(tmp_path: Path):
    rows = _rows()
    tok = Tokenizer.build([(r.id, r.eng, r.ko) for r in rows])
    p = tmp_path / "split.jsonl"
    write_split_jsonl(rows, tok, p)
    loaded = read_split_jsonl(p)
    assert len(loaded) == len(rows)
    assert loaded[0]["src"] == tok.encode_source(rows[0].id, rows[0].eng)
    assert loaded[0]["tgt"] == tok.encode_target(rows[0].ko)
