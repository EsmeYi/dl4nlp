"""
final_train_scaling.py — Data scaling curve experiment.

Trains the baseline model (identical config to final_train.py) using only a
fraction of the training set.  Val and test sets are always full-size.

Usage:
    python final_train_scaling.py --fraction 0.25 --seed 42
    python final_train_scaling.py --fraction 0.50 --seed 42
    python final_train_scaling.py --fraction 0.75 --seed 42
    python final_train_scaling.py --fraction 1.00 --seed 42

Called by run_scaling.sh via SLURM array (array index = fraction * 100).
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent))
from prepare import (
    CACHE_DIR,
    MAX_SEQ_LEN,
    TripletDataset,
    evaluate_probe,
    evaluate_retrieval,
    get_dataloaders,
    load_tokenizers,
    make_collate_fn,
)

# ── Hyperparameters (identical to final_train.py / baseline) ─────────────────

TOKENIZER     = "ia"
EMBED_DIM     = 256
PROJ_DIM      = 128
NUM_LAYERS    = 4
NUM_HEADS     = 4
MLP_RATIO     = 4
DROPOUT       = 0.1
BATCH_SIZE    = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 0.01
TEMPERATURE   = 0.05
WARMUP_STEPS  = 200
LAMBDA_HW     = 0.15
TRAIN_SECONDS = 3600
AMP           = True


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Encoder ───────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.pos_emb = nn.Embedding(MAX_SEQ_LEN, EMBED_DIM)
        self.drop    = nn.Dropout(DROPOUT)
        layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM, nhead=NUM_HEADS,
            dim_feedforward=EMBED_DIM * MLP_RATIO,
            dropout=DROPOUT, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=NUM_LAYERS, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(EMBED_DIM)
        self.proj = nn.Linear(EMBED_DIM, PROJ_DIM, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, L = ids.shape
        pos    = torch.arange(L, device=ids.device).unsqueeze(0)
        x      = self.drop(self.tok_emb(ids) + self.pos_emb(pos))
        mask   = (ids == 0)
        x      = self.transformer(x, src_key_padding_mask=mask)
        mask_f = (~mask).unsqueeze(-1).float()
        x      = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1)
        x      = self.norm(x)
        return self.proj(x)


# ── InfoNCE loss ──────────────────────────────────────────────────────────────

def infonce(emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(emb_a, dim=-1)
    b = F.normalize(emb_b, dim=-1)
    logits = a @ b.T / TEMPERATURE
    labels = torch.arange(logits.shape[0], device=logits.device)
    return (
        F.cross_entropy(logits,   labels, label_smoothing=0.05)
        + F.cross_entropy(logits.T, labels, label_smoothing=0.05)
    ) / 2


# ── Training ──────────────────────────────────────────────────────────────────

def infinite(loader):
    while True:
        yield from loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="Fraction of training data to use (0 < f <= 1.0)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    assert 0 < args.fraction <= 1.0, "fraction must be in (0, 1]"
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  fraction={args.fraction}  seed={args.seed}")

    print("Loading tokenizers ...")
    bpe, ia = load_tokenizers()
    tok = ia if TOKENIZER == "ia" else bpe

    # ── Build subsampled train loader ────────────────────────────────────────
    full_train_ds = TripletDataset(CACHE_DIR / "train.jsonl")
    n_full  = len(full_train_ds)
    n_use   = max(BATCH_SIZE, int(n_full * args.fraction))
    indices = torch.randperm(n_full, generator=torch.Generator().manual_seed(args.seed)).tolist()
    subset  = Subset(full_train_ds, indices[:n_use])
    train_loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=make_collate_fn(tok),
        pin_memory=True,
        persistent_workers=True,
    )

    # Val/test always full-size
    _, val_loader, test_loader = get_dataloaders(tok, batch_size=BATCH_SIZE, num_workers=4)

    print(f"Train subset: {n_use:,} / {n_full:,} samples ({args.fraction*100:.0f}%)")

    src_vocab = bpe.vocab_size
    asm_vocab = ia.asm_vocab_size

    src_enc = Encoder(src_vocab).to(device)
    asm_enc = Encoder(asm_vocab).to(device)
    hw_head = nn.Linear(PROJ_DIM, 1).to(device)

    params = (list(src_enc.parameters())
              + list(asm_enc.parameters())
              + list(hw_head.parameters()))
    num_params_M = sum(p.numel() for p in params) / 1e6

    optimizer = AdamW(params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, 10_000 - WARMUP_STEPS)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    scaler  = torch.cuda.amp.GradScaler(enabled=AMP and device.type == "cuda")
    amp_ctx = torch.cuda.amp.autocast(
        enabled=AMP and device.type == "cuda", dtype=torch.bfloat16
    )

    src_enc.train(); asm_enc.train(); hw_head.train()

    step = 0; total_loss = 0.0; t_start = None
    print(f"Training {num_params_M:.1f}M params for {TRAIN_SECONDS}s ...")

    for batch in infinite(train_loader):
        src_ids = batch["src_ids"].to(device)
        asm_ids = batch["asm_ids"].to(device)
        ipc     = batch["ipc"].float().to(device)

        with amp_ctx:
            src_emb = src_enc(src_ids)
            asm_emb = asm_enc(asm_ids)
            loss    = infonce(src_emb, asm_emb)
            hw_loss = F.mse_loss(hw_head(src_emb).squeeze(-1), ipc)
            loss    = loss + LAMBDA_HW * hw_loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(params, 0.5)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if t_start is None:
            t_start = time.time()
        total_loss += loss.item()
        step       += 1

        if step % 500 == 0:
            print(f"  step {step}  loss {total_loss/step:.4f}  elapsed {time.time()-t_start:.0f}s")

        if time.time() - t_start >= TRAIN_SECONDS:
            break

    training_seconds = time.time() - t_start
    peak_vram_mb = (torch.cuda.max_memory_allocated() / 1024**2
                    if device.type == "cuda" else 0.0)

    print("Evaluating on val set ...")
    val_ret   = evaluate_retrieval(src_enc, asm_enc, val_loader,   device)
    val_probe = evaluate_probe(src_enc, train_loader, val_loader,  device)

    print("Evaluating on test set ...")
    test_ret   = evaluate_retrieval(src_enc, asm_enc, test_loader,  device)
    test_probe = evaluate_probe(src_enc, train_loader, test_loader, device)

    frac_pct = int(args.fraction * 100)
    print(f"\n--- Scaling {frac_pct}% ---")
    print(f"train_samples:        {n_use}")
    print(f"[VAL]  recall@1:      {val_ret['recall@1']:.6f}")
    print(f"[VAL]  recall@5:      {val_ret['recall@5']:.6f}")
    print(f"[VAL]  recall@10:     {val_ret['recall@10']:.6f}")
    print(f"[VAL]  probe_r2_ipc:  {val_probe['probe_r2_ipc']:.6f}")
    print(f"[TEST] recall@1:      {test_ret['recall@1']:.6f}")
    print(f"[TEST] recall@5:      {test_ret['recall@5']:.6f}")
    print(f"[TEST] recall@10:     {test_ret['recall@10']:.6f}")
    print(f"[TEST] probe_r2_ipc:  {test_probe['probe_r2_ipc']:.6f}")
    print(f"training_seconds:     {training_seconds:.1f}")
    print(f"peak_vram_mb:         {peak_vram_mb:.1f}")
    print(f"num_steps:            {step}")

    results = {
        "fraction": args.fraction,
        "train_samples": n_use,
        "seed": args.seed,
        "val_recall1":   val_ret["recall@1"],
        "val_recall5":   val_ret["recall@5"],
        "val_recall10":  val_ret["recall@10"],
        "val_r2":        val_probe["probe_r2_ipc"],
        "test_recall1":  test_ret["recall@1"],
        "test_recall5":  test_ret["recall@5"],
        "test_recall10": test_ret["recall@10"],
        "test_r2":       test_probe["probe_r2_ipc"],
        "avg_train_loss": total_loss / max(1, step),
        "training_seconds": training_seconds,
        "num_steps": step,
    }
    out = Path(__file__).parent / f"scaling_results_{frac_pct}pct_seed{args.seed}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Results saved → {out}")


if __name__ == "__main__":
    main()
