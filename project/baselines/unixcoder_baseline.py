"""
unixcoder_baseline.py — UniXcoder as source encoder baseline.

Two evaluations:
  1. Zero-shot probe: UniXcoder (no fine-tuning) → predict IPC via Ridge regression.
     Answers: does UniXcoder already have hardware intuition?
  2. Fine-tuned retrieval: UniXcoder src_enc + custom asm_enc trained with InfoNCE.
     Compares recall@1 directly against our from-scratch model.

Usage: python unixcoder_baseline.py
"""

import json
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

# huggingface_hub 0.23.x removed is_offline_mode and get_full_repo_name;
# patch them back before importing transformers 4.40.2 which expects them.
import huggingface_hub as _hfhub
if not hasattr(_hfhub, "is_offline_mode"):
    _offline_fn = lambda: os.getenv("HF_HUB_OFFLINE", "0") == "1"
    _hfhub.is_offline_mode = _offline_fn
    if hasattr(_hfhub, "utils"):
        _hfhub.utils.is_offline_mode = _offline_fn
if not hasattr(_hfhub, "get_full_repo_name"):
    _hfhub.get_full_repo_name = lambda model_id, **kw: model_id

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaModel, RobertaTokenizerFast

sys.path.insert(0, str(Path(__file__).parent))
from prepare import (
    CACHE_DIR,
    MAX_SEQ_LEN,
    evaluate_probe,
    evaluate_retrieval,
    load_tokenizers,
    make_collate_fn,
)

# ── Config ────────────────────────────────────────────────────────────────────

CB_MODEL      = "microsoft/unixcoder-base"
CB_MAX_LEN    = 512
EMBED_DIM     = 256
PROJ_DIM      = 128
NUM_LAYERS    = 4
NUM_HEADS     = 4
MLP_RATIO     = 4
DROPOUT       = 0.1

BATCH_SIZE    = 48    # smaller: UniXcoder uses more VRAM
LEARNING_RATE = 1e-4
CB_LR         = 2e-5
WEIGHT_DECAY  = 0.01
TEMPERATURE   = 0.05
WARMUP_STEPS  = 200
LAMBDA_HW     = 0.15
TRAIN_SECONDS = 3600
AMP           = True

EVAL_N        = 2000
PROBE_TR_N    = 5000
PROBE_VAL_N   = 2000

# ── Dataset with UniXcoder tokenization in src_ids ─────────────────────────────

class CBTripletDataset(Dataset):
    """Like TripletDataset but tokenizes source with UniXcoder tokenizer."""

    def __init__(self, path: Path, cb_tok: RobertaTokenizerFast, ia_tok, max_len: int = CB_MAX_LEN):
        self.cb_tok = cb_tok
        self.ia_tok = ia_tok
        self.max_len = max_len
        self.records: List[dict] = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        cb_enc = self.cb_tok(
            r["source"],
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        asm_ids = self.ia_tok.encode_asm(r["asm"])
        asm_ids = asm_ids[:MAX_SEQ_LEN]
        asm_ids += [0] * (MAX_SEQ_LEN - len(asm_ids))

        return {
            # src_ids here are UniXcoder token IDs (pad=1 in RoBERTa)
            "src_ids": cb_enc["input_ids"].squeeze(0),
            "asm_ids": torch.tensor(asm_ids, dtype=torch.long),
            "ipc":     float(r.get("ipc", 0.0)),
            "rth":     float(r.get("block_rthroughput", 0.0)),
        }


def cb_collate(batch):
    return {
        "src_ids": torch.stack([b["src_ids"] for b in batch]),
        "asm_ids": torch.stack([b["asm_ids"] for b in batch]),
        "ipc":     torch.tensor([b["ipc"] for b in batch], dtype=torch.float),
        "rth":     torch.tensor([b["rth"] for b in batch], dtype=torch.float),
    }


def get_cb_dataloaders(cb_tok, ia_tok, batch_size: int, num_workers: int = 2):
    loaders = {}
    for split in ("train", "val", "test"):
        ds = CBTripletDataset(CACHE_DIR / f"{split}.jsonl", cb_tok, ia_tok)
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            collate_fn=cb_collate,
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )
    return loaders["train"], loaders["val"], loaders["test"]


# ── Encoders ──────────────────────────────────────────────────────────────────

class CBSrcEncoder(nn.Module):
    """UniXcoder source encoder. forward(ids) accepts RoBERTa token IDs (pad=1)."""

    def __init__(self):
        super().__init__()
        self.bert = RobertaModel.from_pretrained(CB_MODEL)
        self.norm = nn.LayerNorm(768)
        self.proj = nn.Linear(768, PROJ_DIM, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # RoBERTa pad token id = 1
        attn_mask = (ids != 1).long()
        out = self.bert(input_ids=ids, attention_mask=attn_mask)
        # mean pool over non-pad tokens
        mask_f = attn_mask.unsqueeze(-1).float()
        x = (out.last_hidden_state * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        x = self.norm(x)
        return self.proj(x)


class AsmEncoder(nn.Module):
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
        self.transformer = nn.TransformerEncoder(layer, num_layers=NUM_LAYERS, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(EMBED_DIM)
        self.proj = nn.Linear(EMBED_DIM, PROJ_DIM, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, L = ids.shape
        pos    = torch.arange(L, device=ids.device).unsqueeze(0)
        x      = self.drop(self.tok_emb(ids) + self.pos_emb(pos))
        mask   = (ids == 0)
        x      = self.transformer(x, src_key_padding_mask=mask)
        mask_f = (~mask).unsqueeze(-1).float()
        x      = (x * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        x      = self.norm(x)
        return self.proj(x)


# ── Loss ──────────────────────────────────────────────────────────────────────

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


# ── Zero-shot UniXcoder probe ───────────────────────────────────────────────────

@torch.no_grad()
def zero_shot_probe(cb_enc: CBSrcEncoder, cb_train_loader, cb_val_loader, device):
    """Ridge regression on frozen UniXcoder embeddings → predict IPC."""
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    def collect(loader, n):
        cb_enc.eval()
        embs, labels = [], []
        seen = 0
        for batch in loader:
            ids = batch["src_ids"].to(device)
            emb = cb_enc(ids).cpu().numpy()
            embs.append(emb)
            labels.append(batch["ipc"].numpy())
            seen += ids.shape[0]
            if seen >= n:
                break
        return np.vstack(embs)[:n], np.concatenate(labels)[:n]

    print("  Collecting UniXcoder embeddings (train) ...")
    X_tr, y_tr = collect(cb_train_loader, PROBE_TR_N)
    print("  Collecting UniXcoder embeddings (val) ...")
    X_val, y_val = collect(cb_val_loader, PROBE_VAL_N)

    reg = Ridge(alpha=1.0)
    reg.fit(X_tr, np.log1p(y_tr))
    r2 = float(reg.score(X_val, np.log1p(y_val)))
    return r2


# ── Fine-tuned UniXcoder training ───────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading tokenizers ...")
    bpe, ia = load_tokenizers()
    cb_tok = RobertaTokenizerFast.from_pretrained(CB_MODEL)

    # ── 1. Zero-shot probe (no training) ────────────────────────────────────
    print("\n=== Zero-shot UniXcoder probe ===")
    cb_enc_frozen = CBSrcEncoder().to(device)
    print(f"  UniXcoder params: {sum(p.numel() for p in cb_enc_frozen.parameters())/1e6:.1f}M")

    cb_train_loader, cb_val_loader, cb_test_loader = get_cb_dataloaders(
        cb_tok, ia, batch_size=32, num_workers=2
    )

    zs_r2_val  = zero_shot_probe(cb_enc_frozen, cb_train_loader, cb_val_loader,  device)
    zs_r2_test = zero_shot_probe(cb_enc_frozen, cb_train_loader, cb_test_loader, device)
    print(f"[ZERO-SHOT VAL]  probe_r2_ipc: {zs_r2_val:.6f}")
    print(f"[ZERO-SHOT TEST] probe_r2_ipc: {zs_r2_test:.6f}")
    del cb_enc_frozen
    torch.cuda.empty_cache() if device.type == "cuda" else None

    # ── 2. Fine-tuned UniXcoder + AsmEncoder retrieval ───────────────────────
    print("\n=== Fine-tuned UniXcoder retrieval ===")
    src_enc = CBSrcEncoder().to(device)
    asm_enc = AsmEncoder(ia.asm_vocab_size).to(device)
    hw_head = nn.Linear(PROJ_DIM, 1).to(device)

    cb_params    = list(src_enc.bert.parameters())
    other_params = (
        list(src_enc.norm.parameters()) + list(src_enc.proj.parameters())
        + list(asm_enc.parameters()) + list(hw_head.parameters())
    )
    all_params   = cb_params + other_params
    total_M      = sum(p.numel() for p in all_params) / 1e6

    optimizer = AdamW([
        {"params": cb_params,    "lr": CB_LR},
        {"params": other_params, "lr": LEARNING_RATE},
    ], weight_decay=WEIGHT_DECAY)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, 10_000 - WARMUP_STEPS)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    scaler  = torch.cuda.amp.GradScaler(enabled=AMP and device.type == "cuda")
    amp_ctx = torch.cuda.amp.autocast(enabled=AMP and device.type == "cuda", dtype=torch.bfloat16)

    src_enc.train(); asm_enc.train(); hw_head.train()

    step = 0; total_loss = 0.0; t_start = None
    print(f"  Training {total_M:.1f}M params for {TRAIN_SECONDS}s ...")

    for batch in infinite(cb_train_loader):
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
        nn.utils.clip_grad_norm_(all_params, 0.5)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if t_start is None:
            t_start = time.time()
        total_loss += loss.item()
        step       += 1

        if step % 100 == 0:
            print(f"  step {step}  loss {total_loss/step:.4f}  elapsed {time.time()-t_start:.0f}s")

        if time.time() - t_start >= TRAIN_SECONDS:
            break

    training_seconds = time.time() - t_start
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0

    print("  Evaluating on val set ...")
    val_ret   = evaluate_retrieval(src_enc, asm_enc, cb_val_loader,   device, n=EVAL_N)
    val_probe = evaluate_probe(src_enc, cb_train_loader, cb_val_loader, device,
                               train_n=PROBE_TR_N, val_n=PROBE_VAL_N)

    print("  Evaluating on test set ...")
    test_ret   = evaluate_retrieval(src_enc, asm_enc, cb_test_loader,   device, n=EVAL_N)
    test_probe = evaluate_probe(src_enc, cb_train_loader, cb_test_loader, device,
                                train_n=PROBE_TR_N, val_n=PROBE_VAL_N)

    print("\n--- UniXcoder Baseline Results ---")
    print(f"[ZERO-SHOT VAL]  probe_r2_ipc:  {zs_r2_val:.6f}")
    print(f"[ZERO-SHOT TEST] probe_r2_ipc:  {zs_r2_test:.6f}")
    print(f"[FT VAL]  recall@1:      {val_ret['recall@1']:.6f}")
    print(f"[FT VAL]  recall@5:      {val_ret['recall@5']:.6f}")
    print(f"[FT VAL]  recall@10:     {val_ret['recall@10']:.6f}")
    print(f"[FT VAL]  probe_r2_ipc:  {val_probe['probe_r2_ipc']:.6f}")
    print(f"[FT TEST] recall@1:      {test_ret['recall@1']:.6f}")
    print(f"[FT TEST] recall@5:      {test_ret['recall@5']:.6f}")
    print(f"[FT TEST] recall@10:     {test_ret['recall@10']:.6f}")
    print(f"[FT TEST] probe_r2_ipc:  {test_probe['probe_r2_ipc']:.6f}")
    print(f"avg_train_loss:           {total_loss/max(1,step):.6f}")
    print(f"training_seconds:         {training_seconds:.1f}")
    print(f"peak_vram_mb:             {peak_vram_mb:.1f}")
    print(f"num_steps:                {step}")
    print(f"num_params_M:             {total_M:.1f}")


if __name__ == "__main__":
    main()
