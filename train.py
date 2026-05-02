"""
train.py — Full Training Pipeline for Gondi Seq2Seq Translation
================================================================
Reads gondi_training_dataset.csv, tokenizes with GondiBPETokenizer,
trains Seq2SeqWithAttention (Bidirectional Encoder + Bahdanau Attention).

Dependencies: torch, pandas, sklearn  (no extras)
"""

import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split

# ── Local imports ────────────────────────────────────────────────────────────
from gondi_bpe_tokenizer import GondiTokenizer
from gondi_seq2seq import build_model, train_one_epoch, evaluate

# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE       = 32
EPOCHS           = 30
EMBED_DIM        = 256
HIDDEN_DIM       = 512
NUM_LAYERS       = 2
DROPOUT          = 0.5
TEACHER_FORCING  = 0.5
LR               = 1e-3

# Paths
CSV_PATH         = "gondi_training_dataset.csv"
TOKENIZER_PATH   = "gondi_bpe_model.json"
CHECKPOINT_PATH  = "best_gondi_model.pt"

# Reproducibility
RANDOM_SEED      = 42
torch.manual_seed(RANDOM_SEED)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
#  TOKENIZER SETUP
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  Loading tokenizer …")
print("=" * 70)

tokenizer = GondiTokenizer.load(TOKENIZER_PATH)

PAD_IDX = tokenizer.token_to_id[GondiTokenizer.PAD_TOKEN]     # 0
SOS_IDX = tokenizer.token_to_id[GondiTokenizer.START_TOKEN]   # 2
EOS_IDX = tokenizer.token_to_id[GondiTokenizer.END_TOKEN]     # 3

VOCAB_SIZE = tokenizer.vocab_size()

print(f"  Vocab size : {VOCAB_SIZE}")
print(f"  PAD={PAD_IDX}  SOS={SOS_IDX}  EOS={EOS_IDX}")


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────────────────────────────────────
class GondiDataset(Dataset):
    """
    Reads the CSV, tokenizes both Gondi (source) and Hindi (target) columns,
    and wraps each sequence with [START] … [END] tokens.

    The same BPE tokenizer is used for both columns — the GondiBPETokenizer
    was trained on Gondi words but the Devanagari characters present in
    both columns are handled identically through the shared character-level
    vocabulary.
    """

    def __init__(self, csv_path: str, tokenizer: GondiTokenizer,
                 src_col: str = "Gondi", tgt_col: str = "Hindi"):
        super().__init__()
        df = pd.read_csv(csv_path, encoding="utf-8-sig")

        # Drop rows with missing source or target
        df = df.dropna(subset=[src_col, tgt_col]).reset_index(drop=True)
        df[src_col] = df[src_col].astype(str)
        df[tgt_col] = df[tgt_col].astype(str)

        self.src_texts = df[src_col].tolist()
        self.tgt_texts = df[tgt_col].tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src_ids = self.tokenizer.encode(self.src_texts[idx])
        tgt_ids = self.tokenizer.encode(self.tgt_texts[idx])

        # Wrap with SOS / EOS
        src_ids = [SOS_IDX] + src_ids + [EOS_IDX]
        tgt_ids = [SOS_IDX] + tgt_ids + [EOS_IDX]

        return torch.tensor(src_ids, dtype=torch.long), \
               torch.tensor(tgt_ids, dtype=torch.long)


def collate_fn(batch):
    """
    Pad variable-length sequences to the longest in the batch.
    Returns (src, trg) — both [batch, max_len].

    NOTE: train_one_epoch() and evaluate() in gondi_seq2seq.py
    unpack the dataloader as `for src, trg in dataloader:` and
    compute src_lengths internally, so we return exactly 2 tensors.
    """
    src_seqs, tgt_seqs = zip(*batch)

    # Pad source sequences
    src_padded = nn.utils.rnn.pad_sequence(src_seqs, batch_first=True,
                                            padding_value=PAD_IDX)
    # Pad target sequences
    tgt_padded = nn.utils.rnn.pad_sequence(tgt_seqs, batch_first=True,
                                            padding_value=PAD_IDX)

    return src_padded, tgt_padded


# ─────────────────────────────────────────────────────────────────────────────
#  TRAIN / VAL SPLIT
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  Preparing dataset …")
print("=" * 70)

full_dataset = GondiDataset(CSV_PATH, tokenizer)
total        = len(full_dataset)
val_size     = int(total * 0.1)
train_size   = total - val_size

train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(RANDOM_SEED)
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=collate_fn,
                          num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                          shuffle=False, collate_fn=collate_fn,
                          num_workers=0, pin_memory=True)

print(f"  Total samples : {total}")
print(f"  Train         : {train_size}")
print(f"  Validation    : {val_size}")
print(f"  Batches/epoch : {len(train_loader)} train | {len(val_loader)} val")


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  Building model …")
print("=" * 70)

model = build_model(
    gondi_vocab_size  = VOCAB_SIZE,
    target_vocab_size = VOCAB_SIZE,   # same tokenizer for both columns
    device            = device,
    embed_dim         = EMBED_DIM,
    hidden_dim        = HIDDEN_DIM,
    num_layers        = NUM_LAYERS,
    dropout           = DROPOUT,
)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Device             : {device}")
print(f"  Total parameters   : {total_params:,}")
print(f"  Trainable params   : {trainable_params:,}")
print(f"  Encoder vocab      : {VOCAB_SIZE}")
print(f"  Decoder vocab      : {VOCAB_SIZE}")


# ─────────────────────────────────────────────────────────────────────────────
#  OPTIMIZER, LOSS, SCHEDULER
# ─────────────────────────────────────────────────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", patience=2, factor=0.5
)


# ─────────────────────────────────────────────────────────────────────────────
#  TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  Starting training …")
print("=" * 70)
print(f"\n  {'Epoch':>5} | {'Train Loss':>11} | {'Val Loss':>11} | {'LR':>10}")
print("  " + "─" * 50)

best_val_loss = float("inf")

for epoch in range(1, EPOCHS + 1):

    train_loss = train_one_epoch(
        model, train_loader, optimizer, criterion,
        grad_clip=1.0,
        teacher_forcing_ratio=TEACHER_FORCING,
        pad_idx=PAD_IDX,
    )

    val_loss = evaluate(
        model, val_loader, criterion,
        pad_idx=PAD_IDX,
    )

    # Step the scheduler
    current_lr = optimizer.param_groups[0]["lr"]
    scheduler.step(val_loss)

    # Logging
    print(f"  {epoch:>5} | {train_loss:>11.4f} | {val_loss:>11.4f} | {current_lr:>10.6f}")

    # Checkpointing — save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(
            {
                "model_state_dict":   model.state_dict(),
                "encoder_state_dict": model.encoder.state_dict(),
                "decoder_state_dict": model.decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch":             epoch,
                "val_loss":          val_loss,
            },
            CHECKPOINT_PATH,
        )
        print(f"         ↳ ✓ Best model saved → {CHECKPOINT_PATH}  (val_loss={val_loss:.4f})")

print("\n" + "=" * 70)
print(f"  Training complete.  Best val loss: {best_val_loss:.4f}")
print(f"  Checkpoint: {os.path.abspath(CHECKPOINT_PATH)}")
print("=" * 70)
