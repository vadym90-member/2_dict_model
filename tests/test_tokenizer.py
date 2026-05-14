from __future__ import annotations

from pathlib import Path

from dict_model.tokenizer import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    SEP_ID,
    SPECIALS,
    Tokenizer,
    split_eng,
    split_id,
    split_ko,
)


SAMPLE_ROWS = [
    ("abc_form_number", "number", "수자"),
    ("abc_table_row_number", "row number", "행번호"),
    ("id_shapes_number", "shapes number", "도형개수"),
]


def test_splitters():
    assert split_id("abc_form_number") == ["abc", "form", "number"]
    assert split_id("___a__b_") == ["a", "b"]
    assert split_eng("Row Number") == ["row", "number"]
    assert split_eng("Row Number", lower=False) == ["Row", "Number"]
    assert split_ko("도형개수") == ["도", "형", "개", "수"]


def test_specials_are_pinned():
    tok = Tokenizer.build(SAMPLE_ROWS)
    for i, s in enumerate(SPECIALS):
        assert tok.src_vocab.itos[i] == s
        assert tok.tgt_vocab.itos[i] == s
    assert (PAD_ID, BOS_ID, EOS_ID, SEP_ID) == (0, 2, 3, 4)


def test_id_and_english_share_no_indices():
    """An ID atom ``number`` and the English word ``number`` must hash to
    different ids, otherwise the disambiguation signal collapses."""
    tok = Tokenizer.build(SAMPLE_ROWS)
    assert tok.src_vocab.stoi["i:number"] != tok.src_vocab.stoi["e:number"]


def test_encode_shape_and_special_tokens():
    tok = Tokenizer.build(SAMPLE_ROWS)
    src = tok.encode_source("abc_form_number", "number")
    assert src[0] == BOS_ID and src[-1] == EOS_ID
    assert SEP_ID in src
    # The number of ID atoms before SEP equals the number of underscore parts.
    sep_at = src.index(SEP_ID)
    assert sep_at == 1 + len(split_id("abc_form_number"))

    tgt = tok.encode_target("수자")
    assert tgt[0] == BOS_ID and tgt[-1] == EOS_ID
    assert len(tgt) == 4  # BOS, 수, 자, EOS


def test_roundtrip_decode():
    tok = Tokenizer.build(SAMPLE_ROWS)
    for _, _, ko in SAMPLE_ROWS:
        ids = tok.encode_target(ko)
        assert tok.decode_target(ids) == ko


def test_save_load(tmp_path: Path):
    tok = Tokenizer.build(SAMPLE_ROWS)
    p = tmp_path / "t.json"
    tok.save(p)
    tok2 = Tokenizer.load(p)
    assert tok.src_vocab.itos == tok2.src_vocab.itos
    assert tok.tgt_vocab.itos == tok2.tgt_vocab.itos
    assert tok2.encode_source("abc_form_number", "number") == tok.encode_source("abc_form_number", "number")
