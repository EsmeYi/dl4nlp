"""
final_train_seeded.py — identical to final_train.py but accepts --seed and
saves results to a JSON file for multi-seed aggregation.

Usage:
    python final_train_seeded.py --seed 42
    python final_train_seeded.py --seed 43
    python final_train_seeded.py --seed 44
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

sys.path.insert(0, str(Path(__file__).parent))
from prepare import (
    MAX_SEQ_LEN,
    evaluate_probe,
    evaluate_retrieval,
    get_dataloaders,
    load_tokenizers,
)

# ── Hyperparameters (identical to final_train.py) ────────────────────────────

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
        self.transformer = nn.TransformerEncoder(layer, num_layers=NUM_LAYERS,
                                                  enable_nested_tensor=False)
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


def infonce(emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(emb_a, dim=-1)
    b = F.normalize(emb_b, dim=-1)
    logits = a @ b.T / TEMPERATURE
    labels = torch.arange(logits.shape[0], device=logits.device)
    return (
        F.cross_entropy(logits, labels, label_smoothing=0.05)
        + F.cross_entropy(logits.T, labels, label_smoothing=0.05)
    ) / 2


def infinite(loader):
    while True:
        yield from loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"Seed: {args.seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bpe, ia = load_tokenizers()
    train_loader, val_loader, test_loader = get_dataloaders(
        ia, batch_size=BATCH_SIZE, num_workers=4
    )

    src_enc = Encoder(bpe.vocab_size).to(device)
    asm_enc = Encoder(ia.asm_vocab_size).to(device)
    hw_head = nn.Linear(PROJ_DIM, 1).to(device)

    params = list(src_enc.parameters()) + list(asm_enc.parameters()) + list(hw_head.parameters())
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
        step += 1

        if step % 500 == 0:
            print(f"  step {step}  loss {total_loss/step:.4f}  elapsed {time.time()-t_start:.0f}s")

        if time.time() - t_start >= TRAIN_SECONDS:
            break

    training_seconds = time.time() - t_start

    ckpt_path = Path(__file__).parent / f"final_model_seed{args.seed}.pt"
    torch.save({"src_enc": src_enc.state_dict(),
                "asm_enc": asm_enc.state_dict(),
                "hw_head": hw_head.state_dict()}, ckpt_path)

    print("Evaluating on test set ...")
    test_ret   = evaluate_retrieval(src_enc, asm_enc, test_loader, device)
    test_probe = evaluate_probe(src_enc, train_loader, test_loader, device)

    results = {
        "seed":          args.seed,
        "recall@1":      test_ret["recall@1"],
        "recall@5":      test_ret["recall@5"],
        "recall@10":     test_ret["recall@10"],
        "probe_r2":      test_probe["probe_r2_ipc"],
        "train_seconds": training_seconds,
        "num_steps":     step,
    }

    out_path = Path(__file__).parent / f"results_seed{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[TEST] recall@1:  {results['recall@1']:.4f}")
    print(f"[TEST] recall@5:  {results['recall@5']:.4f}")
    print(f"[TEST] recall@10: {results['recall@10']:.4f}")
    print(f"[TEST] probe_r2:  {results['probe_r2']:.4f}")
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
