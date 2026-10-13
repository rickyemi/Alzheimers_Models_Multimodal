"""Clinical note cleaning and Hugging Face tokenizer construction."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ABBREV = {r"\bpt\b": "patient", r"\bc/o\b": "complains of", r"\bhx\b": "history", r"\bhtn\b": "hypertension",
          r"\bdm\b": "diabetes", r"\biadls?\b": "instrumental activities of daily living",
          r"\badls?\b": "activities of daily living", r"\bpcp\b": "primary care physician", r"\bscd\b": "subjective cognitive decline",
          r"\bmci\b": "mild cognitive impairment", r"\byrs\b": "years", r"\bh\b": "hours", r"\bnp\b": "nurse practitioner"}
EMPTY_TOKEN = "no clinical note available"


def clean_note(text) -> str:
    if not isinstance(text, str) or not text.strip():
        return EMPTY_TOKEN
    t = text.lower()
    for pat, rep in ABBREV.items():
        t = re.sub(pat, rep, t)
    t = re.sub(r"\d+(\.\d+)?", lambda m: "<num>", t)        # de-identify numbers (ages, dates, doses)
    t = re.sub(r"[^a-z<>\s;:,.\-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def clean_notes_table(notes: pd.DataFrame, id_fn) -> tuple[pd.DataFrame, dict]:
    log = {"rows_raw": len(notes)}
    n = notes.copy()
    n["patient_id"] = n.patient_id.map(id_fn)
    before = len(n)
    n = n.drop_duplicates(subset=["patient_id", "visit_date", "note_text"])
    log["exact_duplicates_removed"] = before - len(n)
    log["empty_notes"] = int(n.note_text.fillna("").str.strip().eq("").sum())
    n["note_clean"] = n.note_text.map(clean_note)
    n = n.sort_values("visit_date").groupby("patient_id", as_index=False).last()   # latest note per patient
    log["rows_clean"] = len(n)
    return n, log


def train_wordpiece_tokenizer(texts: list[str], vocab_size: int, out_dir: Path):
    """Train a WordPiece tokenizer with HF `tokenizers` and wrap it as a `transformers` fast tokenizer.

    In production replace this with a domain model tokenizer, e.g.
    AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT').
    """
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    trainer = trainers.WordPieceTrainer(vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    tok.train_from_iterator(texts, trainer)
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", tok.token_to_id("[CLS]")), ("[SEP]", tok.token_to_id("[SEP]"))])
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
                                   sep_token="[SEP]", mask_token="[MASK]")
    out_dir.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(str(out_dir))
    return fast
