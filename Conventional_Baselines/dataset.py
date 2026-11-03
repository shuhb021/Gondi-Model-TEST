"""
dataset.py — Data loading and batch collation for Gondi Translation Baselines
==============================================================================
"""

import os
from typing import Tuple, List, Optional
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from gondi_bpe_tokenizer import GondiTokenizer


class GondiTranslationDataset(Dataset):
    """
    Parallel Translation Dataset for Gondi -> Target language (Hindi / English).
    Uses the shared Gondi BPE Tokenizer for both source and target.
    """
    def __init__(self, csv_path: str, tokenizer: GondiTokenizer,
                 src_col: str = "Gondi", tgt_col: str = "Hindi",
                 max_samples: Optional[int] = None):
        super().__init__()
        self.tokenizer = tokenizer
        self.pad_idx = tokenizer.token_to_id.get(GondiTokenizer.PAD_TOKEN, 0)
        self.sos_idx = tokenizer.token_to_id.get(GondiTokenizer.START_TOKEN, 2)
        self.eos_idx = tokenizer.token_to_id.get(GondiTokenizer.END_TOKEN, 3)

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Dataset CSV not found at: {csv_path}")

        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        df = df.dropna(subset=[src_col, tgt_col]).reset_index(drop=True)

        # Filter out empty or pure whitespace strings
        df[src_col] = df[src_col].astype(str).str.strip()
        df[tgt_col] = df[tgt_col].astype(str).str.strip()
        df = df[(df[src_col] != "") & (df[tgt_col] != "")].reset_index(drop=True)

        if max_samples and max_samples < len(df):
            df = df.iloc[:max_samples].reset_index(drop=True)

        self.src_texts = df[src_col].tolist()
        self.tgt_texts = df[tgt_col].tolist()

    def __len__(self) -> int:
        return len(self.src_texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        src_text = self.src_texts[idx]
        tgt_text = self.tgt_texts[idx]

        src_ids = self.tokenizer.encode(src_text)
        tgt_ids = self.tokenizer.encode(tgt_text)

        # Prepend SOS and append EOS
        src_tensor = torch.tensor([self.sos_idx] + src_ids + [self.eos_idx], dtype=torch.long)
        tgt_tensor = torch.tensor([self.sos_idx] + tgt_ids + [self.eos_idx], dtype=torch.long)

        return src_tensor, tgt_tensor


def pad_collate_fn(batch, pad_idx: int = 0):
    """
    Pads variable-length sequences to the maximum length in the current batch.
    Returns:
      src_padded:  [batch, max_src_len]
      tgt_padded:  [batch, max_tgt_len]
      src_lengths: [batch]
    """
    src_seqs, tgt_seqs = zip(*batch)

    src_lengths = torch.tensor([len(s) for s in src_seqs], dtype=torch.long)
    src_padded = nn.utils.rnn.pad_sequence(src_seqs, batch_first=True, padding_value=pad_idx)
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_seqs, batch_first=True, padding_value=pad_idx)

    return src_padded, tgt_padded, src_lengths
