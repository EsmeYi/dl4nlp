"""
final_train_nl_v3.py — Two-stage differential-LR 3-way alignment.

Stage 1 — NL warm-up (WARMUP_NL_STEPS steps):
  Load 2-way checkpoint; freeze src_enc + asm_enc.
  Train only NL projection with L_NS + L_NA.
  Maps NL into the existing embedding space without disturbing Src<->Asm.

Stage 2 — Joint fine-tuning (remaining time):
  Unfreeze src_enc + asm_enc with 10x lower LR than NL projection.
  Loss: WEIGHT_SA * L_SA + L_NS + L_NA + lambda_hw * MSE(IPC).
  High L_SA weight protects Src<->Asm alignment while the space adapts to NL.

Optimizer param groups (stage 2):
  nl_proj:              lr = LR_NL = 5e-4
  src_enc + asm_enc:    lr = LR_SA = 5e-5   (10x lower)
  hw_head:              lr = LR_SA
"""

import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from prepare import (
    MAX_SEQ_LEN,
    CACHE_DIR,
    load_tokenizers,
    evaluate_retrieval,
    evaluate_probe,
    get_dataloaders,
)

# ── Hyperparameters ───────────────────────────────────────────────────────────

EMBED_DIM  = 256
PROJ_DIM   = 128
NUM_LAYERS = 4
NUM_HEADS  = 4
MLP_RATIO  = 4
DROPOUT    = 0.1

BATCH_SIZE   = 128
WEIGHT_DECAY = 0.01
TEMPERATURE  = 0.05
LAMBDA_HW    = 0.15
WEIGHT_SA    = 3.0   # amplify L_SA relative to L_NS and L_NA

LR_NL  = 5e-4        # NL projection
LR_SA  = 5e-5        # Src + Asm encoders in stage 2 (10x lower)

WARMUP_NL_STEPS = 500   # stage 1: NL-only warm-up
TOTAL_SECONDS   = 7200  # total wall-clock budget (2 hours)
AMP = True

CODEBERT_MODEL = "microsoft/codebert-base"
NL_MAX_LEN     = 64
HF_CACHE       = "/cephyr/users/lirongy/Alvis/.cache/huggingface"
CKPT_IN        = Path(__file__).parent / "final_model.pt"
CKPT_OUT       = Path(__file__).parent / "final_model_nl_v3.pt"

EVAL_N = 2000

# ── Dataset ───────────────────────────────────────────────────────────────────

class NLSrcDataset(Dataset):
    def __init__(self, split: str):
        nl_map: Dict[str, str] = {}
        nl_path = CACHE_DIR / f"nl_{split}.jsonl"
        if nl_path.exists():
            with open(nl_path) as f:
                for line in f:
                    r = json.loads(line)
                    nl_map[r["id"]] = r["nl"]

        self.records: List[dict] = []
        with open(CACHE_DIR / f"{split}.jsonl") as f:
            for line in f:
                r = json.loads(line)
                if r["id"] in nl_map:
                    self.records.append({
                        "source": r["source"],
                        "asm":    r["asm"],
                        "ipc":    float(r.get("ipc", 0.0)),
                        "nl":     nl_map[r["id"]],
                    })

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def make_collate(ia_tok, nl_hf_tok):
    def collate(batch):
        src_ids = ia_tok.batch_source([b["source"] for b in batch])
        asm_ids = ia_tok.batch_asm(   [b["asm"]    for b in batch])
        nl_out  = nl_hf_tok(
            [b["nl"] for b in batch],
            max_length=NL_MAX_LEN, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        return {
            "src_ids":      src_ids,
            "asm_ids":      asm_ids,
            "nl_input_ids": nl_out["input_ids"],
            "nl_attn_mask": nl_out["attention_mask"],
            "ipc": torch.tensor([b["ipc"] for b in batch], dtype=torch.float),
        }
    return collate


def get_loaders(ia_tok, nl_hf_tok):
    loaders = {}
    for split in ("train", "val", "test"):
        ds = NLSrcDataset(split)
        print(f"  {split}: {len(ds)} records with NL", flush=True)
        loaders[split] = DataLoader(
            ds, batch_size=BATCH_SIZE,
            shuffle=(split == "train"),
            num_workers=4,
            collate_fn=make_collate(ia_tok, nl_hf_tok),
            pin_memory=True,
            persistent_workers=True,
        )
    return loaders["train"], loaders["val"], loaders["test"]


# ── Encoders ──────────────────────────────────────────────────────────────────

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
        pos  = torch.arange(L, device=ids.device).unsqueeze(0)
        x    = self.drop(self.tok_emb(ids) + self.pos_emb(pos))
        mask = (ids == 0)
        x    = self.transformer(x, src_key_padding_mask=mask)
        mf   = (~mask).unsqueeze(-1).float()
        x    = (x * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1)
        x    = self.norm(x)
        return self.proj(x)


class NLEncoder(nn.Module):
    """Frozen CodeBERT + trainable linear projection."""

    def __init__(self, codebert):
        super().__init__()
        self.bert = codebert
        for p in self.bert.parameters():
            p.requires_grad = False
        self.proj = nn.Linear(768, PROJ_DIM, bias=False)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return self.proj(out.last_hidden_state[:, 0])


# ── Loss ──────────────────────────────────────────────────────────────────────

def infonce(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    logits = a @ b.T / TEMPERATURE
    labels = torch.arange(logits.shape[0], device=logits.device)
    return (
        F.cross_entropy(logits,   labels, label_smoothing=0.05)
        + F.cross_entropy(logits.T, labels, label_smoothing=0.05)
    ) / 2


# ── Eval ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def _collect(fn, loader, key, device, n):
    embs, seen = [], 0
    for batch in loader:
        emb = fn(batch, device)
        embs.append(F.normalize(emb, dim=-1).cpu())
        seen += emb.shape[0]
        if seen >= n:
            break
    return torch.cat(embs)[:n]


def recall_at_k(q, k, ks=(1, 5, 10)):
    sim   = q @ k.T
    ranks = sim.argsort(dim=1, descending=True)
    gt    = torch.arange(sim.shape[0]).unsqueeze(1)
    return {f"recall@{k}": (ranks[:, :k] == gt).any(dim=1).float().mean().item() for k in ks}


@torch.no_grad()
def eval_nl(nl_enc, src_enc, loader, device, n=EVAL_N):
    nl_enc.eval(); src_enc.eval()

    nl_embs, src_embs = [], []
    seen = 0
    for batch in loader:
        nl_emb  = F.normalize(nl_enc(
            batch["nl_input_ids"].to(device),
            batch["nl_attn_mask"].to(device),
        ), dim=-1).cpu()
        src_emb = F.normalize(src_enc(batch["src_ids"].to(device)), dim=-1).cpu()
        nl_embs.append(nl_emb)
        src_embs.append(src_emb)
        seen += nl_emb.shape[0]
        if seen >= n:
            break

    nl_embs  = torch.cat(nl_embs)[:n]
    src_embs = torch.cat(src_embs)[:n]
    m = min(len(nl_embs), len(src_embs))
    t1 = recall_at_k(nl_embs[:m], src_embs[:m])
    t2 = recall_at_k(src_embs[:m], nl_embs[:m])
    return t1, t2


# ── Training helpers ──────────────────────────────────────────────────────────

def infinite(loader):
    while True:
        yield from loader


def cosine_lr(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return max(0.05, 0.5 * (1 + math.cos(math.pi * progress)))


def run_stage(
    stage: int,
    model_dict: dict,
    optimizer,
    scheduler,
    loader,
    device,
    scaler,
    amp_ctx,
    max_steps: int = None,
    max_seconds: float = None,
    log_interval: int = 200,
):
    """Generic training loop. Returns (steps, total_loss, elapsed_seconds)."""
    src_enc = model_dict["src_enc"]
    asm_enc = model_dict["asm_enc"]
    nl_enc  = model_dict["nl_enc"]
    hw_head = model_dict["hw_head"]

    src_enc.train(); asm_enc.train(); nl_enc.train(); hw_head.train()

    step, total_loss = 0, 0.0
    t_start = None

    for batch in infinite(loader):
        src_ids = batch["src_ids"].to(device)
        asm_ids = batch["asm_ids"].to(device)
        nl_ids  = batch["nl_input_ids"].to(device)
        nl_mask = batch["nl_attn_mask"].to(device)
        ipc     = batch["ipc"].float().to(device)

        with amp_ctx:
            src_emb = src_enc(src_ids)
            asm_emb = asm_enc(asm_ids)
            nl_emb  = nl_enc(nl_ids, nl_mask)

            if stage == 1:
                loss = infonce(nl_emb, src_emb) + infonce(nl_emb, asm_emb)
            else:
                l_sa = infonce(src_emb, asm_emb)
                l_ns = infonce(nl_emb,  src_emb)
                l_na = infonce(nl_emb,  asm_emb)
                hw_loss = F.mse_loss(hw_head(src_emb).squeeze(-1), ipc)
                loss = (WEIGHT_SA * l_sa + l_ns + l_na) / (WEIGHT_SA + 2) + LAMBDA_HW * hw_loss

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(
            [p for p in [*src_enc.parameters(), *asm_enc.parameters(),
                          *nl_enc.proj.parameters(), *hw_head.parameters()]
             if p.requires_grad],
            0.5,
        )
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()

        if t_start is None:
            t_start = time.time()

        total_loss += loss.item()
        step       += 1

        if step % log_interval == 0:
            elapsed = time.time() - t_start
            print(
                f"  [stage {stage}] step {step:5d}  "
                f"loss {total_loss/step:.4f}  elapsed {elapsed:.0f}s",
                flush=True,
            )

        if max_steps is not None and step >= max_steps:
            break
        if max_seconds is not None and t_start is not None:
            if time.time() - t_start >= max_seconds:
                break

    elapsed = time.time() - t_start if t_start else 0.0
    return step, total_loss, elapsed


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import os
    os.environ["HF_HOME"] = HF_CACHE
    os.environ["TRANSFORMERS_CACHE"] = HF_CACHE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading tokenizers ...", flush=True)
    bpe, ia = load_tokenizers()

    print("Loading CodeBERT ...", flush=True)
    from transformers import AutoModel, AutoTokenizer
    nl_hf_tok = AutoTokenizer.from_pretrained(CODEBERT_MODEL)
    codebert  = AutoModel.from_pretrained(CODEBERT_MODEL)

    print("Loading data ...", flush=True)
    train_loader, val_loader, test_loader = get_loaders(ia, nl_hf_tok)

    # Build models
    src_enc = Encoder(bpe.vocab_size).to(device)
    asm_enc = Encoder(ia.asm_vocab_size).to(device)
    hw_head = nn.Linear(PROJ_DIM, 1).to(device)
    nl_enc  = NLEncoder(codebert).to(device)

    # Load 2-way checkpoint
    print(f"Loading 2-way checkpoint from {CKPT_IN} ...", flush=True)
    ckpt = torch.load(CKPT_IN, map_location=device)
    src_enc.load_state_dict(ckpt["src_enc"])
    asm_enc.load_state_dict(ckpt["asm_enc"])
    hw_head.load_state_dict(ckpt["hw_head"])
    print("  2-way weights loaded.", flush=True)

    model_dict = {"src_enc": src_enc, "asm_enc": asm_enc, "nl_enc": nl_enc, "hw_head": hw_head}

    scaler  = torch.cuda.amp.GradScaler(enabled=AMP and device.type == "cuda")
    amp_ctx = torch.cuda.amp.autocast(enabled=AMP and device.type == "cuda", dtype=torch.bfloat16)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # ── Stage 1: NL warm-up (src/asm frozen) ─────────────────────────────────
    print(f"\n=== Stage 1: NL warm-up ({WARMUP_NL_STEPS} steps, src/asm frozen) ===", flush=True)

    for p in src_enc.parameters(): p.requires_grad = False
    for p in asm_enc.parameters(): p.requires_grad = False
    for p in hw_head.parameters(): p.requires_grad = False

    opt_s1 = AdamW(nl_enc.proj.parameters(), lr=LR_NL, weight_decay=WEIGHT_DECAY)

    def s1_lr_lambda(step):
        return cosine_lr(step, warmup=50, total=WARMUP_NL_STEPS)

    sched_s1 = torch.optim.lr_scheduler.LambdaLR(opt_s1, s1_lr_lambda)

    s1_steps, s1_loss, s1_elapsed = run_stage(
        stage=1,
        model_dict=model_dict,
        optimizer=opt_s1,
        scheduler=sched_s1,
        loader=train_loader,
        device=device,
        scaler=scaler,
        amp_ctx=amp_ctx,
        max_steps=WARMUP_NL_STEPS,
    )
    print(f"Stage 1 done: {s1_steps} steps, avg loss {s1_loss/max(1,s1_steps):.4f}, {s1_elapsed:.0f}s", flush=True)

    # ── Stage 2: joint fine-tuning with differential LR ───────────────────────
    stage2_seconds = TOTAL_SECONDS - s1_elapsed
    print(f"\n=== Stage 2: joint fine-tuning ({stage2_seconds:.0f}s budget) ===", flush=True)
    print(f"  LR: NL proj={LR_NL:.0e}, Src/Asm={LR_SA:.0e}, WEIGHT_SA={WEIGHT_SA}", flush=True)

    for p in src_enc.parameters(): p.requires_grad = True
    for p in asm_enc.parameters(): p.requires_grad = True
    for p in hw_head.parameters(): p.requires_grad = True

    sa_params = (
        list(src_enc.parameters())
        + list(asm_enc.parameters())
        + list(hw_head.parameters())
    )
    opt_s2 = AdamW(
        [
            {"params": list(nl_enc.proj.parameters()), "lr": LR_NL},
            {"params": sa_params,                      "lr": LR_SA},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    # Cosine decay over estimated steps (assume ~100 steps/min)
    est_steps = int(stage2_seconds / 60 * 100)
    def s2_lr_lambda(step):
        return cosine_lr(step, warmup=100, total=max(est_steps, 500))

    sched_s2 = torch.optim.lr_scheduler.LambdaLR(opt_s2, s2_lr_lambda)

    s2_steps, s2_loss, s2_elapsed = run_stage(
        stage=2,
        model_dict=model_dict,
        optimizer=opt_s2,
        scheduler=sched_s2,
        loader=train_loader,
        device=device,
        scaler=scaler,
        amp_ctx=amp_ctx,
        max_seconds=stage2_seconds,
    )
    print(f"Stage 2 done: {s2_steps} steps, avg loss {s2_loss/max(1,s2_steps):.4f}, {s2_elapsed:.0f}s", flush=True)

    total_steps   = s1_steps + s2_steps
    total_elapsed = s1_elapsed + s2_elapsed

    # ── Save checkpoint ───────────────────────────────────────────────────────
    torch.save({
        "src_enc": src_enc.state_dict(),
        "asm_enc": asm_enc.state_dict(),
        "hw_head": hw_head.state_dict(),
        "nl_proj": nl_enc.proj.state_dict(),
    }, CKPT_OUT)
    print(f"\nSaved checkpoint: {CKPT_OUT}", flush=True)

    peak_vram = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0.0

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\nEvaluating ...", flush=True)
    src_enc.eval(); asm_enc.eval(); nl_enc.eval(); hw_head.eval()

    # Src<->Asm (use standard dataloaders without NL)
    _, val_std, test_std = get_dataloaders(ia, batch_size=256, num_workers=4)
    val_ret   = evaluate_retrieval(src_enc, asm_enc, val_std,  device)
    test_ret  = evaluate_retrieval(src_enc, asm_enc, test_std, device)
    val_probe  = evaluate_probe(src_enc, train_loader, val_loader,  device)
    test_probe = evaluate_probe(src_enc, train_loader, test_loader, device)

    # NL retrieval
    val_t1,  val_t2  = eval_nl(nl_enc, src_enc, val_loader,  device)
    test_t1, test_t2 = eval_nl(nl_enc, src_enc, test_loader, device)

    print("---")
    print(f"[VAL  T0] Src<->Asm recall@1:  {val_ret['recall@1']:.6f}")
    print(f"[VAL  T0] Src<->Asm recall@5:  {val_ret['recall@5']:.6f}")
    print(f"[VAL  T0] Src<->Asm recall@10: {val_ret['recall@10']:.6f}")
    print(f"[VAL  T0] probe_r2_ipc:        {val_probe['probe_r2_ipc']:.6f}")
    print(f"[VAL  T1] NL->Code recall@1:   {val_t1['recall@1']:.6f}")
    print(f"[VAL  T1] NL->Code recall@5:   {val_t1['recall@5']:.6f}")
    print(f"[VAL  T1] NL->Code recall@10:  {val_t1['recall@10']:.6f}")
    print(f"[VAL  T2] Code->NL recall@1:   {val_t2['recall@1']:.6f}")
    print(f"[VAL  T2] Code->NL recall@5:   {val_t2['recall@5']:.6f}")
    print(f"[VAL  T2] Code->NL recall@10:  {val_t2['recall@10']:.6f}")
    print(f"[TEST T0] Src<->Asm recall@1:  {test_ret['recall@1']:.6f}")
    print(f"[TEST T0] Src<->Asm recall@5:  {test_ret['recall@5']:.6f}")
    print(f"[TEST T0] Src<->Asm recall@10: {test_ret['recall@10']:.6f}")
    print(f"[TEST T0] probe_r2_ipc:        {test_probe['probe_r2_ipc']:.6f}")
    print(f"[TEST T1] NL->Code recall@1:   {test_t1['recall@1']:.6f}")
    print(f"[TEST T1] NL->Code recall@5:   {test_t1['recall@5']:.6f}")
    print(f"[TEST T1] NL->Code recall@10:  {test_t1['recall@10']:.6f}")
    print(f"[TEST T2] Code->NL recall@1:   {test_t2['recall@1']:.6f}")
    print(f"[TEST T2] Code->NL recall@5:   {test_t2['recall@5']:.6f}")
    print(f"[TEST T2] Code->NL recall@10:  {test_t2['recall@10']:.6f}")
    print(f"total_steps:                   {total_steps}")
    print(f"total_seconds:                 {total_elapsed:.1f}")
    print(f"peak_vram_mb:                  {peak_vram:.1f}")
    print(f"config: LR_NL={LR_NL:.0e}  LR_SA={LR_SA:.0e}  WEIGHT_SA={WEIGHT_SA}  WARMUP_NL_STEPS={WARMUP_NL_STEPS}")


if __name__ == "__main__":
    main()
