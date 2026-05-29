"""
final_train_improved.py — IPC-aware soft contrastive loss + MoCo-style queue.

Two algorithmic improvements over final_train.py:

  1. Soft InfoNCE (IPC-aware):
     Standard InfoNCE uses hard 0/1 diagonal labels — every non-matching pair
     is treated as a pure negative. But two different functions with nearly
     identical IPC should not be strongly repelled. We replace hard labels with
     a soft distribution: P[i,j] ∝ exp(-|log(1+IPC_i) - log(1+IPC_j)| / σ).
     The diagonal always gets maximum weight (|IPC_i - IPC_i| = 0), while
     IPC-similar off-diagonal pairs get partial positive weight. This aligns
     the contrastive objective with the static-performance-prediction use case.

  2. MoCo-style queue:
     In-batch negatives are limited to batch_size-1 = 255. We maintain a
     rolling FIFO buffer of QUEUE_SIZE assembly embeddings so that each source
     query is contrasted against ~4352 keys (256 current + 4096 queue).
     No momentum encoder is needed: the queue turns over every ~16 steps
     (4096/256), fast enough that staleness is minimal.

Usage:
    python final_train_improved.py

Ablation flags (edit at top of file):
    QUEUE_SIZE  — set to 0 to disable queue (in-batch only)
    SMOOTH_IPC  — set to 0.0 to recover hard-label InfoNCE
    SIGMA_IPC   — controls bandwidth of the IPC smoothing component
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

# ── Hyperparameters ───────────────────────────────────────────────────────────

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

# New: soft-loss + queue
QUEUE_SIZE    = 0      # assembly embeddings in memory queue; 0 = in-batch only
SIGMA_IPC     = 0.5    # bandwidth for log-IPC kernel (in log1p space)
SMOOTH_IPC    = 0.05   # IPC smoothing fraction; 0.0 = hard diagonal, 1.0 = full soft
               # set SMOOTH_IPC = 0.0 and SIGMA_IPC = 1e9 to recover hard-label InfoNCE


# ── Encoder (identical to final_train.py) ─────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.pos_emb = nn.Embedding(MAX_SEQ_LEN, EMBED_DIM)
        self.drop    = nn.Dropout(DROPOUT)
        layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=EMBED_DIM * MLP_RATIO,
            dropout=DROPOUT,
            batch_first=True,
            norm_first=True,
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


# ── Soft InfoNCE loss with MoCo queue ─────────────────────────────────────────

class SoftMoCoLoss(nn.Module):
    """
    Source→Assembly: queries s_i vs [current-batch asm keys + queue].
    Assembly→Source: queries a_i vs current-batch src keys only
                     (no source queue to keep symmetry simple).
    Both directions use IPC-based soft targets.
    """

    def __init__(self, queue_size: int, proj_dim: int):
        super().__init__()
        self.queue_size = queue_size
        if queue_size > 0:
            # Initialise with unit-norm random vectors; will be overwritten quickly.
            self.register_buffer(
                "asm_queue",
                F.normalize(torch.randn(queue_size, proj_dim), dim=-1),
            )
            self.register_buffer("ipc_queue", torch.zeros(queue_size))
            self.register_buffer("ptr",      torch.zeros(1, dtype=torch.long))
            self.register_buffer("is_full",  torch.zeros(1, dtype=torch.bool))

    # ── queue management ──────────────────────────────────────────────────────

    @torch.no_grad()
    def _enqueue(self, asm_norm: torch.Tensor, ipc: torch.Tensor) -> None:
        """Insert current-batch assembly embeddings into the circular queue."""
        B   = asm_norm.shape[0]
        ptr = int(self.ptr)
        idx = torch.arange(B, device=asm_norm.device)
        slot = (ptr + idx) % self.queue_size
        self.asm_queue[slot] = asm_norm.detach().float()
        self.ipc_queue[slot] = ipc.detach().float()
        new_ptr = int((ptr + B) % self.queue_size)
        if ptr + B >= self.queue_size:
            self.is_full.fill_(True)
        self.ptr.fill_(new_ptr)

    # ── soft target computation ────────────────────────────────────────────────

    @staticmethod
    def _soft_labels(
        ipc_q: torch.Tensor,   # (B,)  query IPC values
        ipc_k: torch.Tensor,   # (K,)  key   IPC values
        sigma: float,
        smooth: float,
    ) -> torch.Tensor:
        """
        Label-smoothing formulation:
          target = (1 - smooth) * one_hot(diagonal) + smooth * ipc_sim_dist

        The diagonal (ground-truth source-assembly pair) retains (1-smooth) weight.
        The remaining `smooth` mass is distributed proportional to IPC similarity.
        This preserves the retrieval signal while mildly discounting similar-IPC
        off-diagonal negatives, regardless of the number of keys K.

        Bug-fixed vs original: pure softmax(-diff/sigma) collapses to near-uniform
        with K≈4352 keys, giving the diagonal only ~0.4% weight and causing
        recall@1 to collapse to random chance.
        """
        B      = ipc_q.shape[0]
        K      = ipc_k.shape[0]
        device = ipc_q.device

        # IPC-similarity smoothing distribution (row-normalized)
        log_q  = torch.log1p(ipc_q.unsqueeze(1).float())   # (B, 1)
        log_k  = torch.log1p(ipc_k.unsqueeze(0).float())   # (1, K)
        diff   = (log_q - log_k).abs()                      # (B, K)
        sim    = torch.exp(-diff / sigma)                   # (B, K)
        sim    = sim / sim.sum(dim=-1, keepdim=True)        # row-normalize

        # Hard diagonal: ground-truth positive at position i for query i
        hard   = torch.zeros(B, K, device=device)
        hard[torch.arange(B, device=device),
             torch.arange(B, device=device)] = 1.0

        return (1.0 - smooth) * hard + smooth * sim

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        src_emb: torch.Tensor,   # (B, D)  source embeddings (grad)
        asm_emb: torch.Tensor,   # (B, D)  assembly embeddings (grad)
        ipc:     torch.Tensor,   # (B,)    IPC labels
    ) -> torch.Tensor:
        # Always compute in float32 for numerical stability.
        s = F.normalize(src_emb.float(), dim=-1)   # (B, D)
        a = F.normalize(asm_emb.float(), dim=-1)   # (B, D)

        # ── Source → Assembly ──────────────────────────────────────────────
        if self.queue_size > 0:
            if self.is_full:
                q_asm = self.asm_queue            # (Q, D)
                q_ipc = self.ipc_queue            # (Q,)
            else:
                ptr   = int(self.ptr)
                q_asm = self.asm_queue[:ptr]
                q_ipc = self.ipc_queue[:ptr]

            # Current-batch keys first so diagonal is at positions [0..B-1].
            all_asm = torch.cat([a, q_asm], dim=0)          # (B+Q, D)
            all_ipc = torch.cat([ipc.float(), q_ipc], dim=0)  # (B+Q,)
        else:
            all_asm = a
            all_ipc = ipc.float()

        logits_sq = s @ all_asm.T / TEMPERATURE              # (B, B+Q)
        soft_sq   = self._soft_labels(ipc, all_ipc, SIGMA_IPC, SMOOTH_IPC)
        loss_sq   = -(soft_sq * F.log_softmax(logits_sq, dim=-1)).sum(dim=-1).mean()

        # ── Assembly → Source (in-batch) ───────────────────────────────────
        logits_as = a @ s.T / TEMPERATURE                    # (B, B)
        soft_as   = self._soft_labels(ipc, ipc.float(), SIGMA_IPC, SMOOTH_IPC)
        loss_as   = -(soft_as * F.log_softmax(logits_as, dim=-1)).sum(dim=-1).mean()

        # Enqueue after loss computation so current batch is not its own negative.
        if self.queue_size > 0:
            self._enqueue(a, ipc)

        return (loss_sq + loss_as) / 2


# ── Training ──────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def infinite(loader):
    while True:
        yield from loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  seed: {args.seed}")
    print(f"Config: queue_size={QUEUE_SIZE}  sigma_ipc={SIGMA_IPC}  smooth_ipc={SMOOTH_IPC}  tau={TEMPERATURE}")

    print("Loading tokenizers and data ...")
    bpe, ia = load_tokenizers()
    tok = ia if TOKENIZER == "ia" else bpe

    train_loader, val_loader, test_loader = get_dataloaders(
        tok, batch_size=BATCH_SIZE, num_workers=4
    )

    src_vocab = bpe.vocab_size
    asm_vocab = ia.asm_vocab_size

    src_enc   = Encoder(src_vocab).to(device)
    asm_enc   = Encoder(asm_vocab).to(device)
    hw_head   = nn.Linear(PROJ_DIM, 1).to(device)
    criterion = SoftMoCoLoss(QUEUE_SIZE, PROJ_DIM).to(device)

    params       = (list(src_enc.parameters())
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
            loss    = criterion(src_emb, asm_emb, ipc)
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

    ckpt = Path(__file__).parent / f"improved_model_seed{args.seed}.pt"
    torch.save({"src_enc": src_enc.state_dict(),
                "asm_enc": asm_enc.state_dict(),
                "hw_head": hw_head.state_dict()}, ckpt)
    print(f"Saved checkpoint → {ckpt}")

    print("Evaluating on val set ...")
    val_ret   = evaluate_retrieval(src_enc, asm_enc, val_loader,   device)
    val_probe = evaluate_probe(src_enc, train_loader, val_loader,  device)

    print("Evaluating on test set ...")
    test_ret   = evaluate_retrieval(src_enc, asm_enc, test_loader,  device)
    test_probe = evaluate_probe(src_enc, train_loader, test_loader, device)

    print("\n--- Improved Model Results ---")
    print(f"[VAL]  recall@1:      {val_ret['recall@1']:.6f}")
    print(f"[VAL]  recall@5:      {val_ret['recall@5']:.6f}")
    print(f"[VAL]  recall@10:     {val_ret['recall@10']:.6f}")
    print(f"[VAL]  probe_r2_ipc:  {val_probe['probe_r2_ipc']:.6f}")
    print(f"[TEST] recall@1:      {test_ret['recall@1']:.6f}")
    print(f"[TEST] recall@5:      {test_ret['recall@5']:.6f}")
    print(f"[TEST] recall@10:     {test_ret['recall@10']:.6f}")
    print(f"[TEST] probe_r2_ipc:  {test_probe['probe_r2_ipc']:.6f}")
    print(f"avg_train_loss:       {total_loss / max(1, step):.6f}")
    print(f"training_seconds:     {training_seconds:.1f}")
    print(f"peak_vram_mb:         {peak_vram_mb:.1f}")
    print(f"num_steps:            {step}")
    print(f"num_params_M:         {num_params_M:.1f}")
    print(f"queue_size:           {QUEUE_SIZE}")
    print(f"sigma_ipc:            {SIGMA_IPC}")
    print(f"smooth_ipc:           {SMOOTH_IPC}")
    print(f"seed:                 {args.seed}")

    results = {
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
    out = Path(__file__).parent / f"improved_results_seed{args.seed}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Results saved → {out}")


if __name__ == "__main__":
    main()
