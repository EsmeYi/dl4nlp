"""
eval_probe_extended.py — Extended IPC probe comparison.

Reports Ridge probe R² on log1p(IPC) for four encoder configurations:

  1. Ours  / Source    : source encoder from final_model.pt  (known: ~0.710)
  2. Ours  / Assembly  : asm encoder from final_model.pt     (new)
  3. BERT  / Source    : CodeBERT zero-shot on C source       (known: ~0.237)
  4. BERT  / Assembly  : CodeBERT zero-shot on raw assembly   (new)

The Ours/Source → Ours/Assembly gap shows how much hardware intuition
is retained vs. transferred through contrastive alignment.
The BERT/Assembly result shows whether generic code pretraining captures
any micro-architectural signal when fed raw assembly text.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from prepare import CACHE_DIR, MAX_SEQ_LEN, load_tokenizers

# ── Config ────────────────────────────────────────────────────────────────────

EMBED_DIM  = 256
PROJ_DIM   = 128
NUM_LAYERS = 4
NUM_HEADS  = 4
MLP_RATIO  = 4
DROPOUT    = 0.1

PROBE_TR_N  = 5000
PROBE_VAL_N = 2000
BERT_MAX_LEN = 128   # assembly text can be long; truncate to fit CodeBERT

CKPT_PATH      = Path(__file__).parent / "final_model.pt"
CODEBERT_MODEL = "microsoft/codebert-base"
HF_CACHE       = "/cephyr/users/lirongy/Alvis/.cache/huggingface"

# ── Encoder (mirrors final_train.py) ─────────────────────────────────────────

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


# ── Dataset ───────────────────────────────────────────────────────────────────

class RawDataset(Dataset):
    """Returns raw source + asm text alongside tokenised ids and IPC."""

    def __init__(self, split: str):
        self.records = []
        with open(CACHE_DIR / f"{split}.jsonl") as f:
            for line in f:
                r = json.loads(line)
                self.records.append({
                    "source": r["source"],
                    "asm":    r["asm"],
                    "ipc":    float(r.get("ipc", 0.0)),
                })

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]


def make_collate(ia_tok, bert_hf_tok):
    def collate(batch):
        sources = [b["source"] for b in batch]
        asms    = [b["asm"]    for b in batch]

        src_ids = ia_tok.batch_source(sources)
        asm_ids = ia_tok.batch_asm(asms)

        # CodeBERT tokenisation of source (standard BPE)
        bert_src = bert_hf_tok(
            sources, max_length=BERT_MAX_LEN, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        # CodeBERT tokenisation of raw assembly text (standard BPE)
        bert_asm = bert_hf_tok(
            asms, max_length=BERT_MAX_LEN, padding="max_length",
            truncation=True, return_tensors="pt",
        )
        return {
            "src_ids":           src_ids,
            "asm_ids":           asm_ids,
            "bert_src_ids":      bert_src["input_ids"],
            "bert_src_mask":     bert_src["attention_mask"],
            "bert_asm_ids":      bert_asm["input_ids"],
            "bert_asm_mask":     bert_asm["attention_mask"],
            "ipc": torch.tensor([b["ipc"] for b in batch], dtype=torch.float),
        }
    return collate


def get_loaders(ia_tok, bert_hf_tok, batch_size=128):
    loaders = {}
    for split in ("train", "val"):
        ds = RawDataset(split)
        print(f"  {split}: {len(ds)} records", flush=True)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=False,
            num_workers=4, pin_memory=True, persistent_workers=True,
            collate_fn=make_collate(ia_tok, bert_hf_tok),
        )
    return loaders["train"], loaders["val"]


# ── Probe helper ──────────────────────────────────────────────────────────────

def run_probe(enc_fn, train_loader, val_loader, train_n, val_n):
    """enc_fn(batch, device) → (B, D) tensor."""
    device = next(iter([torch.device("cuda") if torch.cuda.is_available()
                        else torch.device("cpu")]))

    def collect(loader, n):
        embs, labels = [], []
        seen = 0
        for batch in loader:
            with torch.no_grad():
                emb = enc_fn(batch, device)
            embs.append(emb.cpu().numpy())
            labels.append(batch["ipc"].numpy())
            seen += emb.shape[0]
            if seen >= n:
                break
        return np.vstack(embs)[:n], np.concatenate(labels)[:n]

    X_tr, y_tr   = collect(train_loader, train_n)
    X_val, y_val = collect(val_loader,   val_n)

    reg = Ridge(alpha=1.0)
    reg.fit(X_tr, np.log1p(y_tr))
    r2 = r2_score(np.log1p(y_val), reg.predict(X_val))
    return round(float(r2), 4)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import os
    os.environ["HF_HOME"] = HF_CACHE
    os.environ["TRANSFORMERS_CACHE"] = HF_CACHE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # Load tokenizers
    print("Loading tokenizers ...", flush=True)
    bpe, ia = load_tokenizers()

    # Load CodeBERT
    print("Loading CodeBERT ...", flush=True)
    from transformers import AutoModel, AutoTokenizer
    bert_hf_tok = AutoTokenizer.from_pretrained(CODEBERT_MODEL)
    codebert    = AutoModel.from_pretrained(CODEBERT_MODEL).to(device).eval()

    # Load data
    print("Loading data ...", flush=True)
    train_loader, val_loader = get_loaders(ia, bert_hf_tok, batch_size=128)

    # Load our checkpoint
    print(f"Loading checkpoint: {CKPT_PATH} ...", flush=True)
    ckpt    = torch.load(CKPT_PATH, map_location=device)
    src_enc = Encoder(bpe.vocab_size).to(device)
    asm_enc = Encoder(ia.asm_vocab_size).to(device)
    src_enc.load_state_dict(ckpt["src_enc"])
    asm_enc.load_state_dict(ckpt["asm_enc"])
    src_enc.eval(); asm_enc.eval()

    results = {}

    # ── 1. Ours / Source ──────────────────────────────────────────────────────
    print("\n[1/4] Probe: Ours / Source ...", flush=True)
    def enc_ours_src(batch, dev):
        return src_enc(batch["src_ids"].to(dev))

    results["ours_source"] = run_probe(
        enc_ours_src, train_loader, val_loader, PROBE_TR_N, PROBE_VAL_N
    )
    print(f"  R² = {results['ours_source']}", flush=True)

    # ── 2. Ours / Assembly ────────────────────────────────────────────────────
    print("[2/4] Probe: Ours / Assembly ...", flush=True)
    def enc_ours_asm(batch, dev):
        return asm_enc(batch["asm_ids"].to(dev))

    results["ours_assembly"] = run_probe(
        enc_ours_asm, train_loader, val_loader, PROBE_TR_N, PROBE_VAL_N
    )
    print(f"  R² = {results['ours_assembly']}", flush=True)

    # ── 3. CodeBERT / Source ──────────────────────────────────────────────────
    print("[3/4] Probe: CodeBERT zero-shot / Source ...", flush=True)
    def enc_bert_src(batch, dev):
        with torch.no_grad():
            out = codebert(
                input_ids=batch["bert_src_ids"].to(dev),
                attention_mask=batch["bert_src_mask"].to(dev),
            )
        return out.last_hidden_state[:, 0]  # [CLS]

    results["bert_source"] = run_probe(
        enc_bert_src, train_loader, val_loader, PROBE_TR_N, PROBE_VAL_N
    )
    print(f"  R² = {results['bert_source']}", flush=True)

    # ── 4. CodeBERT / Assembly ────────────────────────────────────────────────
    print("[4/4] Probe: CodeBERT zero-shot / Assembly ...", flush=True)
    def enc_bert_asm(batch, dev):
        with torch.no_grad():
            out = codebert(
                input_ids=batch["bert_asm_ids"].to(dev),
                attention_mask=batch["bert_asm_mask"].to(dev),
            )
        return out.last_hidden_state[:, 0]  # [CLS]

    results["bert_assembly"] = run_probe(
        enc_bert_asm, train_loader, val_loader, PROBE_TR_N, PROBE_VAL_N
    )
    print(f"  R² = {results['bert_assembly']}", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print(f"{'Encoder':<28} {'Modality':<12} {'Probe R²':>8}")
    print("-"*55)
    print(f"{'Ours (IA tok + contrastive)':<28} {'Assembly':<12} {results['ours_assembly']:>8.4f}")
    print(f"{'Ours (IA tok + contrastive)':<28} {'Source':<12} {results['ours_source']:>8.4f}")
    print(f"{'CodeBERT zero-shot (BPE)':<28} {'Assembly':<12} {results['bert_assembly']:>8.4f}")
    print(f"{'CodeBERT zero-shot (BPE)':<28} {'Source':<12} {results['bert_source']:>8.4f}")
    print("="*55)
    print(f"\nTransfer gap (Asm→Src): "
          f"{results['ours_assembly']:.4f} → {results['ours_source']:.4f} "
          f"({(results['ours_source']-results['ours_assembly'])*100:+.1f}pp)")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal time: {time.time()-t0:.0f}s", flush=True)
