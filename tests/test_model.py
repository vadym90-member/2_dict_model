from __future__ import annotations

import torch

from dict_model.dataset import GlossaryRow, collate
from dict_model.decode import greedy_decode
from dict_model.model import DictTransformer, ModelConfig
from dict_model.tokenizer import BOS_ID, PAD_ID, Tokenizer


def _tiny_tokenizer() -> Tokenizer:
    rows = [
        ("abc_form_number", "number", "수자"),
        ("abc_table_row_number", "row number", "행번호"),
        ("id_shapes_number", "shapes number", "도형개수"),
    ]
    return Tokenizer.build(rows)


def _tiny_model(tok: Tokenizer) -> DictTransformer:
    return DictTransformer(
        ModelConfig(
            src_vocab_size=len(tok.src_vocab),
            tgt_vocab_size=len(tok.tgt_vocab),
            d_model=32,
            n_heads=4,
            n_enc_layers=2,
            n_dec_layers=2,
            d_ff=64,
            dropout=0.0,
            max_len=32,
        )
    )


def test_forward_shape():
    tok = _tiny_tokenizer()
    model = _tiny_model(tok)
    rows = [GlossaryRow("abc_form_number", "number", "수자")]
    batch = collate(
        [{"id": r.id, "eng": r.eng, "ko": r.ko,
          "src": tok.encode_source(r.id, r.eng),
          "tgt": tok.encode_target(r.ko)} for r in rows]
    )
    logits = model(batch["src"], batch["tgt_in"], src_kpm=batch["src_kpm"], tgt_kpm=batch["tgt_kpm"])
    assert logits.shape == (1, batch["tgt_in"].size(1), len(tok.tgt_vocab))


def test_causal_mask_blocks_future_tokens():
    L, D = 4, 8
    model = DictTransformer(
        ModelConfig(src_vocab_size=10, tgt_vocab_size=10, d_model=D, n_heads=2, n_enc_layers=1, n_dec_layers=1, d_ff=16, dropout=0.0, max_len=16)
    )
    src = torch.tensor([[BOS_ID, 5, 6, 7]], dtype=torch.long)
    src_kpm = src == PAD_ID
    memory = model.encode(src, src_kpm)
    tgt_a = torch.tensor([[BOS_ID, 5, 6, 7]], dtype=torch.long)
    tgt_b = tgt_a.clone()
    tgt_b[0, -1] = 9  # change the *last* token only
    out_a = model.decode(tgt_a, memory, tgt_kpm=tgt_a == PAD_ID, mem_kpm=src_kpm)
    out_b = model.decode(tgt_b, memory, tgt_kpm=tgt_b == PAD_ID, mem_kpm=src_kpm)
    # Earlier positions must be unaffected by a change at the end (causality).
    assert torch.allclose(out_a[:, :-1], out_b[:, :-1], atol=1e-5)
    assert not torch.allclose(out_a[:, -1], out_b[:, -1], atol=1e-5)


def test_overfit_one_batch():
    """Sanity: the model can drive loss toward zero on a single batch."""
    torch.manual_seed(0)
    tok = _tiny_tokenizer()
    model = _tiny_model(tok)
    rows = [
        GlossaryRow("abc_form_number", "number", "수자"),
        GlossaryRow("abc_table_row_number", "row number", "행번호"),
        GlossaryRow("id_shapes_number", "shapes number", "도형개수"),
    ]
    batch = collate(
        [{"id": r.id, "eng": r.eng, "ko": r.ko,
          "src": tok.encode_source(r.id, r.eng),
          "tgt": tok.encode_target(r.ko)} for r in rows]
    )
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_ID)
    initial = None
    for _ in range(120):
        logits = model(batch["src"], batch["tgt_in"], src_kpm=batch["src_kpm"], tgt_kpm=batch["tgt_kpm"])
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), batch["tgt_out"].reshape(-1))
        if initial is None:
            initial = loss.item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert loss.item() < 0.1, f"failed to overfit tiny batch: {initial=:.3f} -> {loss.item():.3f}"


def test_greedy_decode_runs():
    tok = _tiny_tokenizer()
    model = _tiny_model(tok)
    out = greedy_decode(model, tok, "abc_form_number", "number", max_new_tokens=8)
    assert isinstance(out, str)
