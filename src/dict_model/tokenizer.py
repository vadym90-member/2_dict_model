"""Tokenizer for the (ID, English) -> Korean dictionary model.

The translation task is *context-disambiguated*: the same English word
("number") can map to many Korean glosses ("수자", "행번호", "도형개수", …)
depending on the surrounding ID (`abc_form_number`, `abc_table_row_number`,
`id_shapes_number`, …). The tokenizer therefore exposes the ID structure to
the model explicitly, rather than throwing it away with a generic word
tokenizer.

Encoding scheme
---------------
A row ``(id, eng, ko)`` is encoded into two integer sequences:

    src = [BOS] + id_atoms + [SEP] + eng_words + [EOS]
    tgt = [BOS] + ko_syllables + [EOS]

* The source uses a *single* vocabulary so both halves share one embedding
  table. ID atoms and English words live in disjoint namespaces inside that
  vocab via the ``i:`` / ``e:`` prefixes — that way an English word "number"
  and the ID atom "number" stay distinguishable tokens.
* The target uses a separate Korean-syllable vocabulary.

Special tokens occupy fixed indices so the model code can rely on them:

    PAD = 0   UNK = 1   BOS = 2   EOS = 3   SEP = 4
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

PAD_ID, UNK_ID, BOS_ID, EOS_ID, SEP_ID = 0, 1, 2, 3, 4
SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>", "<sep>"]

# A Hangul syllable is a single codepoint in [U+AC00, U+D7A3]. We tokenize
# Korean at the syllable level — this keeps the target vocabulary small
# (≈ a few hundred entries for a realistic glossary) and avoids any
# pretrained subword tokenizer.
_HANGUL_RE = re.compile(r"[가-힣]")


# ---------------------------------------------------------------------------
# Atom splitters
# ---------------------------------------------------------------------------

def split_id(raw_id: str) -> list[str]:
    """Split a snake_case ID into atoms. Empty atoms are dropped."""
    return [atom for atom in raw_id.strip().split("_") if atom]


def split_eng(raw_eng: str, lower: bool = True) -> list[str]:
    """Whitespace-tokenize the English expression."""
    text = raw_eng.lower() if lower else raw_eng
    return text.strip().split()


def split_ko(raw_ko: str) -> list[str]:
    """Split Korean into syllable tokens; non-Hangul characters survive as-is."""
    out: list[str] = []
    for ch in raw_ko.strip():
        if ch.isspace():
            continue
        out.append(ch)
    return out


# ---------------------------------------------------------------------------
# Vocab
# ---------------------------------------------------------------------------

@dataclass
class Vocab:
    """A minimal symbol↔index table with the standard 5 specials at the front."""

    itos: list[str] = field(default_factory=lambda: list(SPECIALS))
    stoi: dict[str, int] = field(default_factory=lambda: {s: i for i, s in enumerate(SPECIALS)})

    def __len__(self) -> int:
        return len(self.itos)

    def add(self, token: str) -> int:
        if token not in self.stoi:
            self.stoi[token] = len(self.itos)
            self.itos.append(token)
        return self.stoi[token]

    def encode(self, tokens: Iterable[str]) -> list[int]:
        return [self.stoi.get(t, UNK_ID) for t in tokens]

    def decode(self, ids: Iterable[int]) -> list[str]:
        return [self.itos[i] if 0 <= i < len(self.itos) else "<unk>" for i in ids]

    def to_dict(self) -> dict:
        return {"itos": self.itos}

    @classmethod
    def from_dict(cls, d: dict) -> "Vocab":
        v = cls(itos=list(d["itos"]))
        v.stoi = {s: i for i, s in enumerate(v.itos)}
        return v


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

@dataclass
class Tokenizer:
    """Holds the source (id+eng) vocab and the target (ko) vocab.

    The source vocabulary is shared between the ID and English halves of the
    input so we can use a single embedding table on the encoder side. To keep
    the two namespaces disjoint we prefix tokens with ``i:`` or ``e:`` before
    inserting them into the vocab.
    """

    src_vocab: Vocab = field(default_factory=Vocab)
    tgt_vocab: Vocab = field(default_factory=Vocab)
    eng_lower: bool = True
    max_src_len: int = 32
    max_tgt_len: int = 24

    # -- vocabulary construction -------------------------------------------

    @staticmethod
    def _id_tok(atom: str) -> str:
        return "i:" + atom

    @staticmethod
    def _en_tok(word: str) -> str:
        return "e:" + word

    @classmethod
    def build(
        cls,
        rows: Iterable[tuple[str, str, str]],
        *,
        eng_lower: bool = True,
        max_src_len: int = 32,
        max_tgt_len: int = 24,
        id_subword_min_freq: int = 1,
    ) -> "Tokenizer":
        """Fit vocabularies from an iterable of (id, eng, ko) rows."""
        tok = cls(eng_lower=eng_lower, max_src_len=max_src_len, max_tgt_len=max_tgt_len)
        id_counts: Counter[str] = Counter()
        en_counts: Counter[str] = Counter()
        ko_counts: Counter[str] = Counter()
        for raw_id, raw_eng, raw_ko in rows:
            id_counts.update(split_id(raw_id))
            en_counts.update(split_eng(raw_eng, lower=eng_lower))
            ko_counts.update(split_ko(raw_ko))
        # Stable, deterministic ordering: frequency desc, then alphabetical.
        for atom, c in sorted(id_counts.items(), key=lambda x: (-x[1], x[0])):
            if c >= id_subword_min_freq:
                tok.src_vocab.add(cls._id_tok(atom))
        for word, _ in sorted(en_counts.items(), key=lambda x: (-x[1], x[0])):
            tok.src_vocab.add(cls._en_tok(word))
        for syl, _ in sorted(ko_counts.items(), key=lambda x: (-x[1], x[0])):
            tok.tgt_vocab.add(syl)
        return tok

    # -- encoding ----------------------------------------------------------

    def encode_source(self, raw_id: str, raw_eng: str) -> list[int]:
        """Encode the source side: [BOS] <id-atoms> [SEP] <eng-words> [EOS]."""
        id_ids = self.src_vocab.encode(self._id_tok(a) for a in split_id(raw_id))
        en_ids = self.src_vocab.encode(self._en_tok(w) for w in split_eng(raw_eng, self.eng_lower))
        seq = [BOS_ID, *id_ids, SEP_ID, *en_ids, EOS_ID]
        if len(seq) > self.max_src_len:
            # Truncate the English tail first, keep the ID atoms intact since
            # they carry the disambiguating context.
            keep = self.max_src_len - 1  # leave room for EOS
            seq = seq[:keep] + [EOS_ID]
        return seq

    def encode_target(self, raw_ko: str) -> list[int]:
        """Encode the target side: [BOS] <ko-syllables> [EOS]."""
        ko_ids = self.tgt_vocab.encode(split_ko(raw_ko))
        seq = [BOS_ID, *ko_ids, EOS_ID]
        if len(seq) > self.max_tgt_len:
            seq = seq[: self.max_tgt_len - 1] + [EOS_ID]
        return seq

    # -- decoding ----------------------------------------------------------

    def decode_target(self, ids: Iterable[int]) -> str:
        """Decode target ids back to a Korean string, stripping specials."""
        syllables: list[str] = []
        for i in ids:
            if i in (PAD_ID, BOS_ID):
                continue
            if i == EOS_ID:
                break
            tok = self.tgt_vocab.itos[i] if 0 <= i < len(self.tgt_vocab) else ""
            if tok in SPECIALS:
                continue
            syllables.append(tok)
        return "".join(syllables)

    # -- persistence -------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "src_vocab": self.src_vocab.to_dict(),
            "tgt_vocab": self.tgt_vocab.to_dict(),
            "eng_lower": self.eng_lower,
            "max_src_len": self.max_src_len,
            "max_tgt_len": self.max_tgt_len,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            src_vocab=Vocab.from_dict(payload["src_vocab"]),
            tgt_vocab=Vocab.from_dict(payload["tgt_vocab"]),
            eng_lower=payload["eng_lower"],
            max_src_len=payload["max_src_len"],
            max_tgt_len=payload["max_tgt_len"],
        )
