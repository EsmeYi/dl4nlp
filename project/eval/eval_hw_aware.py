"""
eval_hw_aware.py — Hardware-aware embedding analysis.

Experiment A: t-SNE visualisation of source embeddings coloured by IPC category.
  Saved to: hw_aware_tsne.pdf

Experiment C: Hardware-aware retrieval quality.
  Beyond identity Recall@K, measures whether retrieved functions share the same
  IPC category as the query (HW-Precision@K) and mean IPC distance at top-1
  vs a random baseline.
  Saved to: hw_aware_results.txt
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

TSNE_N   = 3000   # points for t-SNE (more = slower)
EVAL_N   = 5000   # points for retrieval quality
TOPK     = (1, 5, 10)
CKPT     = Path(__file__).parent / "final_model.pt"
OUT_DIR  = Path(__file__).parent

# IPC bins — interpretable thresholds
# low: memory-bound (<1.0), mid: balanced (1-3), high: compute-intensive (>=3)
BIN_EDGES  = [0.0, 1.0, 3.0, float("inf")]
BIN_LABELS = ["Low IPC\n(<1.0)", "Mid IPC\n(1.0–3.0)", "High IPC\n(≥3.0)"]
BIN_COLORS = ["#4878CF", "#6ACC65", "#D65F5F"]   # blue / green / red

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
        pos  = torch.arange(L, device=ids.device).unsqueeze(0)
        x    = self.drop(self.tok_emb(ids) + self.pos_emb(pos))
        mask = (ids == 0)
        x    = self.transformer(x, src_key_padding_mask=mask)
        mf   = (~mask).unsqueeze(-1).float()
        x    = (x * mf).sum(dim=1) / mf.sum(dim=1).clamp(min=1)
        x    = self.norm(x)
        return self.proj(x)


# ── Dataset ───────────────────────────────────────────────────────────────────

class SimpleDataset(Dataset):
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

    def __len__(self): return len(self.records)
    def __getitem__(self, i): return self.records[i]


def get_loader(ia_tok, split, batch_size=256):
    def collate(batch):
        src_ids = ia_tok.batch_source([b["source"] for b in batch])
        asm_ids = ia_tok.batch_asm(   [b["asm"]    for b in batch])
        return {
            "src_ids": src_ids,
            "asm_ids": asm_ids,
            "ipc": torch.tensor([b["ipc"] for b in batch], dtype=torch.float),
        }
    ds = SimpleDataset(split)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True,
        collate_fn=collate,
    )


# ── Embedding collection ──────────────────────────────────────────────────────

@torch.no_grad()
def collect_embs_both(src_enc, asm_enc, loader, device, n):
    """Collect src and asm embeddings in one pass to guarantee index alignment."""
    src_enc.eval(); asm_enc.eval()
    src_embs, asm_embs, ipcs = [], [], []
    seen = 0
    for batch in loader:
        se = src_enc(batch["src_ids"].to(device))
        ae = asm_enc(batch["asm_ids"].to(device))
        src_embs.append(F.normalize(se, dim=-1).cpu())
        asm_embs.append(F.normalize(ae, dim=-1).cpu())
        ipcs.append(batch["ipc"])
        seen += se.shape[0]
        if seen >= n:
            break
    src_embs = torch.cat(src_embs)[:n]
    asm_embs = torch.cat(asm_embs)[:n]
    ipcs     = torch.cat(ipcs)[:n]
    return src_embs.numpy(), asm_embs.numpy(), ipcs.numpy()


def ipc_to_bin(ipc: np.ndarray) -> np.ndarray:
    return np.digitize(ipc, BIN_EDGES[1:-1])   # 0=low, 1=mid, 2=high


# ── Experiment A: t-SNE ───────────────────────────────────────────────────────

def run_tsne(src_embs: np.ndarray, ipc: np.ndarray, out_path: Path):
    """Compute t-SNE and save coords + labels for local plotting."""
    from sklearn.manifold import TSNE

    print(f"  Running t-SNE on {len(src_embs)} points ...", flush=True)
    t0 = time.time()
    coords = TSNE(
        n_components=2, perplexity=40, max_iter=1000,
        random_state=42, n_jobs=4,
    ).fit_transform(src_embs)
    print(f"  t-SNE done in {time.time()-t0:.0f}s", flush=True)

    bins = ipc_to_bin(ipc)

    # Save coords, bins, ipc for local plotting
    npy_path = out_path.with_suffix(".npz")
    np.savez(npy_path, coords=coords, bins=bins, ipc=ipc)
    print(f"  Saved t-SNE data: {npy_path}", flush=True)
    print("  To plot locally: python plot_tsne.py", flush=True)

    # Print bin distribution
    for b, label in enumerate(["Low (<1.0)", "Mid (1-3)", "High (≥3)"]):
        n = (bins == b).sum()
        print(f"  {label}: {n} ({100*n/len(bins):.1f}%)", flush=True)


# ── Experiment C: hardware-aware retrieval quality ────────────────────────────

def run_hw_retrieval(src_embs: np.ndarray, asm_embs: np.ndarray,
                     src_ipc: np.ndarray, asm_ipc: np.ndarray,
                     out_path: Path):
    """
    For each source query, retrieve top-K assembly candidates by cosine sim.
    Metrics:
      - HW-Precision@K: fraction of top-K in same IPC bin as query
      - Mean |ΔIPC| @1: mean absolute IPC difference for top-1 retrieval
      - Baseline: expected HW-P@K and mean |ΔIPC| under random retrieval
    """
    n = len(src_embs)
    src_bins = ipc_to_bin(src_ipc)
    asm_bins = ipc_to_bin(asm_ipc)

    print(f"  Computing {n}×{n} similarity matrix ...", flush=True)
    # chunk to avoid OOM on large n
    chunk = 500
    all_topk_idx = []
    for i in range(0, n, chunk):
        q = torch.from_numpy(src_embs[i:i+chunk])           # (chunk, D)
        sim = q @ torch.from_numpy(asm_embs).T               # (chunk, n)
        # exclude diagonal (identity)
        for j in range(len(q)):
            sim[j, i+j] = -1.0
        topk = sim.topk(max(TOPK), dim=1).indices            # (chunk, K)
        all_topk_idx.append(topk.numpy())
    topk_idx = np.concatenate(all_topk_idx, axis=0)          # (n, K)

    results = {}
    lines   = []

    # ── HW-Precision@K ────────────────────────────────────────────────────────
    lines.append("=== Hardware-Aware Retrieval Quality ===\n")
    lines.append(f"Pool size: {n}\n")

    # Random baseline: P(same bin) = fraction of pool in same bin
    bin_fracs = [(asm_bins == b).mean() for b in range(3)]
    random_hwp = sum(
        (src_bins == b).mean() * bin_fracs[b] for b in range(3)
    )
    lines.append(f"\nRandom HW-Precision baseline: {random_hwp:.4f}\n")
    lines.append("\nHW-Precision@K (fraction of top-K in same IPC bin as query):\n")

    for k in TOPK:
        retrieved_bins = asm_bins[topk_idx[:, :k]]           # (n, k)
        query_bins     = src_bins[:, None]                    # (n, 1)
        hw_prec = (retrieved_bins == query_bins).mean()
        results[f"hw_prec@{k}"] = hw_prec
        lines.append(f"  HW-P@{k:2d}: {hw_prec:.4f}  (vs random {random_hwp:.4f}, "
                     f"+{(hw_prec-random_hwp)*100:+.1f}pp)\n")

    # ── Mean |ΔIPC| at top-1 ─────────────────────────────────────────────────
    top1_ipc   = asm_ipc[topk_idx[:, 0]]
    mean_delta = np.abs(top1_ipc - src_ipc).mean()

    # Random baseline: expected |ΔIPC| under random retrieval
    random_delta = np.abs(
        src_ipc[:, None] - asm_ipc[None, :]
    ).mean()

    results["mean_ipc_delta_top1"] = mean_delta
    lines.append(f"\nMean |ΔIPC| @ top-1: {mean_delta:.4f}  "
                 f"(random: {random_delta:.4f}, "
                 f"{(1-mean_delta/random_delta)*100:.1f}% reduction)\n")

    # ── Per-bin breakdown ─────────────────────────────────────────────────────
    lines.append("\nHW-P@1 breakdown by query IPC bin:\n")
    bin_names = ["Low (<1.0)", "Mid (1-3)", "High (≥3)"]
    for b, bname in enumerate(bin_names):
        mask = src_bins == b
        if mask.sum() == 0:
            continue
        retrieved = asm_bins[topk_idx[mask, 0]]
        prec      = (retrieved == b).mean()
        rand_b    = bin_fracs[b]
        lines.append(f"  {bname:12s}: HW-P@1={prec:.4f}  "
                     f"(random={rand_b:.4f}, n={mask.sum()})\n")

    text = "".join(lines)
    print(text, flush=True)
    out_path.write_text(text)
    print(f"  Saved: {out_path}", flush=True)
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    print("Loading tokenizers ...", flush=True)
    bpe, ia = load_tokenizers()

    print("Loading data ...", flush=True)
    loader = get_loader(ia, "test", batch_size=256)

    print(f"Loading checkpoint {CKPT} ...", flush=True)
    ckpt    = torch.load(CKPT, map_location=device)
    src_enc = Encoder(bpe.vocab_size).to(device)
    asm_enc = Encoder(ia.asm_vocab_size).to(device)
    src_enc.load_state_dict(ckpt["src_enc"])
    asm_enc.load_state_dict(ckpt["asm_enc"])

    n_collect = max(TSNE_N, EVAL_N)
    print(f"\nCollecting {n_collect} paired embeddings ...", flush=True)
    src_embs, asm_embs, src_ipc = collect_embs_both(
        src_enc, asm_enc, loader, device, n_collect
    )
    asm_ipc = src_ipc  # same function → same IPC

    # ── Experiment A: t-SNE ──────────────────────────────────────────────────
    print(f"\n{'='*50}", flush=True)
    print("Experiment A: t-SNE visualisation", flush=True)
    run_tsne(src_embs[:TSNE_N], src_ipc[:TSNE_N],
             OUT_DIR / "hw_aware_tsne.pdf")

    # ── Experiment C: hardware-aware retrieval ───────────────────────────────
    print(f"\n{'='*50}", flush=True)
    print("Experiment C: Hardware-aware retrieval quality", flush=True)
    run_hw_retrieval(
        src_embs[:EVAL_N], asm_embs[:EVAL_N],
        src_ipc[:EVAL_N],  asm_ipc[:EVAL_N],
        OUT_DIR / "hw_aware_results.txt",
    )


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nTotal time: {time.time()-t0:.0f}s", flush=True)
