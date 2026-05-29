"""
Fixed infrastructure for cross-modal alignment experiments.
DO NOT modify this file — it is the stable ground truth for data,
tokenization, and evaluation.

One-time setup:
    python prepare.py

Runtime imports (in train.py):
    from prepare import (
        get_dataloaders, load_tokenizers,
        evaluate_retrieval, evaluate_probe,
        MAX_SEQ_LEN, CACHE_DIR,
    )
"""

import json
import hashlib
import os
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

# ── Constants ─────────────────────────────────────────────────────────────────

DATA_PATH   = Path("/cephyr/users/lirongy/Alvis/dl4nlp/Experiment/DataGeneration/triplets.jsonl")
CACHE_DIR   = Path(os.path.expanduser("~/.cache/dl4nlp_crossmodal"))
MAX_SEQ_LEN = 512
VAL_RATIO   = 0.05
TEST_RATIO  = 0.05
BPE_VOCAB_SIZE = 16384

# evaluation budget (keep fast for 5-min experiment loop)
EVAL_N       = 2000   # val samples for retrieval eval
PROBE_TR_N   = 5000   # train samples to fit linear probe
PROBE_VAL_N  = 2000   # val samples to score probe

PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]

# ── IR preprocessing ──────────────────────────────────────────────────────────

_HEADER_RE = re.compile(
    r"^(;\s*ModuleID|source_filename|target datalayout|target triple"
    r"|@llvm\.compiler\.used|;\s*Function Attrs:)"
)
_ATTRS_RE       = re.compile(r"^attributes #\d+ = \{")
_META_RE        = re.compile(r"^(!llvm\.|!\d+ = )")
_INLINE_META_RE = re.compile(r",\s*![a-zA-Z0-9._]+ !\d+")


def strip_ir(ir: str) -> str:
    lines, skip_attrs = [], False
    for line in ir.splitlines():
        s = line.strip()
        if _ATTRS_RE.match(s):
            skip_attrs = True
        if skip_attrs:
            if s.endswith("}"):
                skip_attrs = False
            continue
        if _HEADER_RE.match(s) or _META_RE.match(s):
            continue
        line = _INLINE_META_RE.sub("", line)
        if line.strip():
            lines.append(line)
    return "\n".join(lines)


# ── Data splitting ────────────────────────────────────────────────────────────

def _split_key(rec_id: str) -> str:
    h = int(hashlib.md5(rec_id.encode()).hexdigest(), 16) % 100
    if h < int(TEST_RATIO * 100):
        return "test"
    if h < int((VAL_RATIO + TEST_RATIO) * 100):
        return "val"
    return "train"


def build_splits(force: bool = False):
    train_path = CACHE_DIR / "train.jsonl"
    if train_path.exists() and not force:
        print("Splits already exist — skipping.")
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    handles = {k: open(CACHE_DIR / f"{k}.jsonl", "w") for k in ("train", "val", "test")}
    counts = {"train": 0, "val": 0, "test": 0}
    with open(DATA_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rec["ir"] = strip_ir(rec.get("ir_O2", ""))
            # keep ir_O3 stripped for augmentation experiments
            rec["ir_O3"] = strip_ir(rec.get("ir_O3", ""))
            rec.pop("ir_O2", None)
            split = _split_key(rec["id"])
            handles[split].write(json.dumps(rec) + "\n")
            counts[split] += 1
    for h in handles.values():
        h.close()
    total = sum(counts.values())
    print(f"Splits done — train: {counts['train']}  val: {counts['val']}  test: {counts['test']}  (total {total})")


# ── BPE tokenizer ─────────────────────────────────────────────────────────────

class BPETokenizer:
    """HuggingFace BPE, trained on all three modalities from the training split."""

    def __init__(self):
        self._tok = None

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def build(self, texts: List[str], vocab_size: int = BPE_VOCAB_SIZE):
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.trainers import BpeTrainer

        tok = Tokenizer(BPE(unk_token="[UNK]"))
        tok.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            min_frequency=2,
        )
        tok.train_from_iterator(texts, trainer=trainer)
        self._tok = tok

    def save(self, path: Path):
        self._tok.save(str(path))

    def load(self, path: Path):
        from tokenizers import Tokenizer
        self._tok = Tokenizer.from_file(str(path))

    def encode(self, text: str, max_len: int = MAX_SEQ_LEN) -> List[int]:
        ids = self._tok.encode(text).ids
        return [BOS_ID] + ids[: max_len - 2] + [EOS_ID]

    def batch(self, texts: List[str], max_len: int = MAX_SEQ_LEN) -> torch.Tensor:
        out = []
        for t in texts:
            ids = self.encode(t, max_len)
            ids += [PAD_ID] * (max_len - len(ids))
            out.append(ids[:max_len])
        return torch.tensor(out, dtype=torch.long)


# ── Instruction-aware tokenizer ───────────────────────────────────────────────

# x86-64 opcode vocabulary (covers >95% of AnghaBench assembly)
_X86_OPCODES: List[str] = [
    # data movement
    "mov","movq","movl","movw","movb","movabs",
    "movzx","movsx","movsxd","movzbl","movzbq","movzwl","movzwq",
    "movsbl","movsbq","movswl","movswq",
    "movsd","movss","movaps","movups","movapd","movupd",
    "vmovsd","vmovss","vmovaps","vmovups","vmovapd","vmovupd","vmovdqu","vmovdqa",
    "push","pushq","pop","popq","lea","leaq",
    # arithmetic
    "add","addq","addl","addw","addb","sub","subq","subl","subw","subb",
    "imul","imulq","imull","mul","mulq","mull",
    "div","divq","divl","idiv","idivq","idivl",
    "inc","incq","incl","dec","decq","decl","neg","negq","negl",
    "adc","sbb","cqo","cdq","cdqe","cltq","cwtl",
    # logic / shift
    "and","andq","andl","andw","andb","or","orq","orl","orw","orb",
    "xor","xorq","xorl","xorw","xorb","not","notq","notl",
    "shl","shlq","shll","shr","shrq","shrl","sar","sarq","sarl",
    "rol","ror","rcl","rcr",
    # compare / test
    "cmp","cmpq","cmpl","cmpw","cmpb","test","testq","testl","testw","testb",
    # control flow
    "jmp","je","jne","jz","jnz","jg","jge","jl","jle",
    "ja","jae","jb","jbe","js","jns","jo","jno","jp","jnp",
    "call","callq","ret","retq","leave","nop","hlt","ud2",
    # set / cmov
    "sete","setne","setg","setge","setl","setle","seta","setae","setb","setbe",
    "sets","setns","seto","setno",
    "cmove","cmovne","cmovg","cmovge","cmovl","cmovle",
    "cmova","cmovae","cmovb","cmovbe","cmovs","cmovns",
    # SIMD scalar/packed
    "addps","addpd","addss","addsd","subps","subpd","subss","subsd",
    "mulps","mulpd","mulss","mulsd","divps","divpd","divss","divsd",
    "vaddps","vaddpd","vaddss","vaddsd","vsubps","vsubpd","vsubss","vsubsd",
    "vmulps","vmulpd","vmulss","vmulsd","vdivps","vdivpd","vdivss","vdivsd",
    "pxor","vpxor","pand","vpand","por","vpor","pandn","vpandn",
    "pshufd","vpshufd","shufps","vshufps","pshufb","vpshufb",
    "blendps","vblendps","blendvps","vblendvps",
    "ucomisd","ucomiss","cmpps","cmppd",
    # convert
    "cvtsi2sd","cvtsi2ss","cvttsd2si","cvttss2si","cvtsd2ss","cvtss2sd",
    "vcvtsi2sd","vcvtsi2ss",
    # misc
    "xchg","bsf","bsr","popcnt","lzcnt","tzcnt",
    "endbr64","endbr32",
]

# LLVM IR instruction/keyword vocabulary
_IR_KEYWORDS: List[str] = [
    # terminators
    "ret","br","switch","indirectbr","invoke","resume","unreachable",
    # binary
    "add","fadd","sub","fsub","mul","fmul","udiv","sdiv","fdiv","urem","srem","frem",
    # bitwise
    "shl","lshr","ashr","and","or","xor",
    # memory
    "alloca","load","store","fence","cmpxchg","atomicrmw",
    # GEP / cast
    "getelementptr","trunc","zext","sext","fptrunc","fpext",
    "fptoui","fptosi","uitofp","sitofp","ptrtoint","inttoptr","bitcast","addrspacecast",
    # comparison
    "icmp","fcmp",
    # other
    "phi","select","call","va_arg","landingpad","freeze",
    "extractelement","insertelement","shufflevector","extractvalue","insertvalue",
    # modifiers / keywords
    "define","declare","tail","musttail","notail",
    "nsw","nuw","exact","inbounds","volatile","atomic",
    "nonnull","noundef","nocapture","readonly","writeonly","noalias",
    # types
    "void","ptr","i1","i8","i16","i32","i64","i128",
    "float","double","half","bfloat","x86_fp80","fp128",
    # icmp predicates
    "eq","ne","ugt","uge","ult","ule","sgt","sge","slt","sle",
    # fcmp predicates
    "oeq","ogt","oge","olt","ole","one","ord","ueq","une","uno","true","false",
]

# register class patterns  →  abstract token
_REG_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(rax|rbx|rcx|rdx|rsi|rdi|rsp|rbp|r8|r9|r10|r11|r12|r13|r14|r15)\b"), "[GPR64]"),
    (re.compile(r"\b(eax|ebx|ecx|edx|esi|edi|esp|ebp|r8d|r9d|r10d|r11d|r12d|r13d|r14d|r15d)\b"), "[GPR32]"),
    (re.compile(r"\b(ax|bx|cx|dx|si|di|sp|bp|r\d+w)\b"), "[GPR16]"),
    (re.compile(r"\b(al|bl|cl|dl|sil|dil|spl|bpl|r\d+b)\b"), "[GPR8]"),
    (re.compile(r"\bzmm\d+\b"), "[SIMD512]"),
    (re.compile(r"\bymm\d+\b"), "[SIMD256]"),
    (re.compile(r"\bxmm\d+\b"), "[SIMD128]"),
    (re.compile(r"\bmm\d+\b"), "[MMX]"),
]
_MEM_RE = re.compile(r"-?\d*\s*\(")       # memory operand like 8(%rsp), (%rdi)
_IMM_RE = re.compile(r"\$-?\d+")          # immediate like $8, $-1
_SSA_RE = re.compile(r"%\w+")             # SSA values like %3, %val


class InstructionAwareTokenizer:
    """
    Separate vocabularies for ASM and IR modalities.
    Source code always uses the underlying BPE tokenizer.

    Vocabulary layout (same for both asm_vocab and ir_vocab):
        0: [PAD]   1: [UNK]   2: [BOS]   3: [EOS]
        4+:  domain-specific tokens (opcodes, register classes, IR keywords, ...)
    """

    def __init__(self, bpe: BPETokenizer):
        self.bpe = bpe
        self._asm_vocab: Dict[str, int] = {}
        self._ir_vocab: Dict[str, int] = {}

    def build(self):
        asm_toks = (
            SPECIAL_TOKENS
            + [f"[{op.upper()}]" for op in _X86_OPCODES]
            + ["[GPR64]", "[GPR32]", "[GPR16]", "[GPR8]",
               "[SIMD512]", "[SIMD256]", "[SIMD128]", "[MMX]",
               "[MEM]", "[IMM]", "[LABEL]"]
        )
        self._asm_vocab = {t: i for i, t in enumerate(asm_toks)}

        ir_toks = (
            SPECIAL_TOKENS
            + [f"[{kw.upper()}]" for kw in _IR_KEYWORDS]
            + ["[SSA]", "[GLOBAL]", "[TYPE]"]
        )
        self._ir_vocab = {t: i for i, t in enumerate(ir_toks)}

    @property
    def asm_vocab_size(self) -> int:
        return len(self._asm_vocab)

    @property
    def ir_vocab_size(self) -> int:
        return len(self._ir_vocab)

    def save(self, path: Path):
        with open(path, "wb") as f:
            pickle.dump({"asm": self._asm_vocab, "ir": self._ir_vocab}, f)

    def load(self, path: Path):
        with open(path, "rb") as f:
            d = pickle.load(f)
        self._asm_vocab = d["asm"]
        self._ir_vocab  = d["ir"]

    # ── ASM encoding ────────────────────────────────────────────────────────

    def _asm_tok(self, t: str) -> int:
        return self._asm_vocab.get(t, UNK_ID)

    def _encode_asm_line(self, line: str) -> List[int]:
        line = line.strip()
        if not line or line.startswith(".") or line.startswith("#") or line.endswith(":"):
            return []
        # abstract registers before splitting
        for pat, cls in _REG_PATTERNS:
            line = pat.sub(cls, line)
        # abstract memory and immediates
        line = _MEM_RE.sub("[MEM](", line)
        line = _IMM_RE.sub("[IMM]", line)

        parts = line.split(None, 1)
        opcode_tok = f"[{parts[0].lower().rstrip(',').upper()}]"
        ids = [self._asm_tok(opcode_tok)]
        if len(parts) > 1:
            for w in parts[1].replace(",", " ").split():
                if w in self._asm_vocab:
                    ids.append(self._asm_vocab[w])
                # else: skip unknown operand tokens
        return ids

    def encode_asm(self, asm: str, max_len: int = MAX_SEQ_LEN) -> List[int]:
        ids = [BOS_ID]
        for line in asm.splitlines():
            ids.extend(self._encode_asm_line(line))
            if len(ids) >= max_len - 1:
                break
        return (ids[: max_len - 1] + [EOS_ID])

    # ── IR encoding ─────────────────────────────────────────────────────────

    def _ir_tok(self, t: str) -> int:
        return self._ir_vocab.get(t, UNK_ID)

    def _encode_ir_line(self, line: str) -> List[int]:
        ids = []
        for raw in line.split():
            w = raw.strip(",%()[]{}*!=")
            if not w:
                continue
            kw_tok = f"[{w.upper()}]"
            if kw_tok in self._ir_vocab:
                ids.append(self._ir_vocab[kw_tok])
            elif _SSA_RE.fullmatch(w):
                ids.append(self._ir_tok("[SSA]"))
            elif w.startswith("@"):
                ids.append(self._ir_tok("[GLOBAL]"))
            elif w.startswith("%") and not _SSA_RE.fullmatch(w):
                ids.append(self._ir_tok("[TYPE]"))
            # else: skip (metadata, numeric literals, etc.)
        return ids

    def encode_ir(self, ir: str, max_len: int = MAX_SEQ_LEN) -> List[int]:
        ids = [BOS_ID]
        for line in ir.splitlines():
            ids.extend(self._encode_ir_line(line))
            if len(ids) >= max_len - 1:
                break
        return (ids[: max_len - 1] + [EOS_ID])

    # ── Source encoding (delegates to BPE) ──────────────────────────────────

    def encode_source(self, src: str, max_len: int = MAX_SEQ_LEN) -> List[int]:
        return self.bpe.encode(src, max_len)

    # ── Batch helpers ────────────────────────────────────────────────────────

    def _pad(self, ids: List[int], max_len: int) -> List[int]:
        ids = ids[:max_len]
        return ids + [PAD_ID] * (max_len - len(ids))

    def batch_asm(self, texts: List[str], max_len: int = MAX_SEQ_LEN) -> torch.Tensor:
        return torch.tensor(
            [self._pad(self.encode_asm(t, max_len), max_len) for t in texts],
            dtype=torch.long,
        )

    def batch_ir(self, texts: List[str], max_len: int = MAX_SEQ_LEN) -> torch.Tensor:
        return torch.tensor(
            [self._pad(self.encode_ir(t, max_len), max_len) for t in texts],
            dtype=torch.long,
        )

    def batch_source(self, texts: List[str], max_len: int = MAX_SEQ_LEN) -> torch.Tensor:
        return self.bpe.batch(texts, max_len)


# ── Dataset ───────────────────────────────────────────────────────────────────

class TripletDataset(Dataset):
    """Loads records into memory; tokenization happens in the collate_fn."""

    def __init__(self, path: Path):
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
        return {
            "source": r["source"],
            "asm":    r["asm"],
            "ir":     r.get("ir", ""),
            "ir_O3":  r.get("ir_O3", ""),
            "ipc":    float(r.get("ipc", 0.0)),
            "rth":    float(r.get("block_rthroughput", 0.0)),
        }


def make_collate_fn(tokenizer, max_len: int = MAX_SEQ_LEN):
    """
    Returns a collate_fn compatible with both BPETokenizer and
    InstructionAwareTokenizer.

    Output keys:
        src_ids  (B, L)  — source code token ids
        asm_ids  (B, L)  — assembly token ids
        ir_ids   (B, L)  — IR token ids  (ir_O2 stripped)
        ipc      (B,)    — IPC label (float)
        rth      (B,)    — block_rthroughput label (float)
    """
    is_bpe = isinstance(tokenizer, BPETokenizer)

    def collate(batch: List[dict]) -> dict:
        sources = [b["source"] for b in batch]
        asms    = [b["asm"]    for b in batch]
        irs     = [b["ir"]     for b in batch]

        if is_bpe:
            src_ids = tokenizer.batch(sources, max_len)
            asm_ids = tokenizer.batch(asms,    max_len)
            ir_ids  = tokenizer.batch(irs,     max_len)
        else:
            src_ids = tokenizer.batch_source(sources, max_len)
            asm_ids = tokenizer.batch_asm(asms,       max_len)
            ir_ids  = tokenizer.batch_ir(irs,         max_len)

        return {
            "src_ids": src_ids,
            "asm_ids": asm_ids,
            "ir_ids":  ir_ids,
            "ipc":     torch.tensor([b["ipc"] for b in batch], dtype=torch.float),
            "rth":     torch.tensor([b["rth"] for b in batch], dtype=torch.float),
        }

    return collate


def get_dataloaders(
    tokenizer,
    batch_size: int = 64,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader)."""
    loaders = {}
    for split in ("train", "val", "test"):
        ds = TripletDataset(CACHE_DIR / f"{split}.jsonl")
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            collate_fn=make_collate_fn(tokenizer),
            pin_memory=True,
            persistent_workers=(num_workers > 0),
        )
    return loaders["train"], loaders["val"], loaders["test"]


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def _collect_embeddings(
    encoder,
    loader: DataLoader,
    key: str,
    device: torch.device,
    max_n: int,
) -> torch.Tensor:
    """Encode up to max_n samples from loader[key]; return L2-normalised embeddings."""
    encoder.eval()
    embs, seen = [], 0
    for batch in loader:
        ids = batch[key].to(device)
        emb = encoder(ids)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        embs.append(emb.cpu())
        seen += ids.shape[0]
        if seen >= max_n:
            break
    return torch.cat(embs, dim=0)[:max_n]


def evaluate_retrieval(
    src_enc,
    asm_enc,
    val_loader: DataLoader,
    device: torch.device,
    n: int = EVAL_N,
    ks: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    """
    Source → Assembly cross-modal retrieval.
    Encoders must implement: encoder(input_ids: LongTensor) -> FloatTensor (B, D).
    Returns {"recall@1": ..., "recall@5": ..., "recall@10": ...}.
    """
    src_embs = _collect_embeddings(src_enc, val_loader, "src_ids", device, n)
    asm_embs = _collect_embeddings(asm_enc, val_loader, "asm_ids", device, n)

    sim   = src_embs @ asm_embs.T                       # (n, n) cosine sim
    ranks = sim.argsort(dim=1, descending=True)          # (n, n) sorted indices
    gt    = torch.arange(sim.shape[0]).unsqueeze(1)      # (n, 1)

    results = {}
    for k in ks:
        hit = (ranks[:, :k] == gt).any(dim=1).float().mean().item()
        results[f"recall@{k}"] = hit
    return results


def evaluate_probe(
    src_enc,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    train_n: int = PROBE_TR_N,
    val_n:   int = PROBE_VAL_N,
) -> Dict[str, float]:
    """
    Linear probe: predict log1p(IPC) from frozen source embeddings.
    Fits sklearn Ridge on train_n training samples; reports R² on val_n val samples.
    Returns {"probe_r2_ipc": float}.
    """
    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score

    def collect(loader, enc, n):
        enc.eval()
        embs, labels = [], []
        seen = 0
        with torch.no_grad():
            for batch in loader:
                ids = batch["src_ids"].to(device)
                emb = enc(ids)
                embs.append(emb.cpu().numpy())
                labels.append(batch["ipc"].numpy())
                seen += ids.shape[0]
                if seen >= n:
                    break
        return np.vstack(embs)[:n], np.concatenate(labels)[:n]

    X_tr, y_tr   = collect(train_loader, src_enc, train_n)
    X_val, y_val = collect(val_loader,   src_enc, val_n)

    y_tr_log  = np.log1p(y_tr)
    y_val_log = np.log1p(y_val)

    reg = Ridge(alpha=1.0)
    reg.fit(X_tr, y_tr_log)
    r2 = r2_score(y_val_log, reg.predict(X_val))
    return {"probe_r2_ipc": round(float(r2), 6)}


# ── Tokenizer I/O ─────────────────────────────────────────────────────────────

def load_tokenizers() -> Tuple[BPETokenizer, InstructionAwareTokenizer]:
    bpe = BPETokenizer()
    bpe.load(CACHE_DIR / "bpe_tokenizer.json")
    ia = InstructionAwareTokenizer(bpe)
    ia.load(CACHE_DIR / "ia_tokenizer.pkl")
    return bpe, ia


# ── One-time setup ────────────────────────────────────────────────────────────

def setup(force: bool = False):
    print("=" * 60)
    print("Step 1/3  Building data splits ...")
    build_splits(force=force)

    bpe_path = CACHE_DIR / "bpe_tokenizer.json"
    print("Step 2/3  Training BPE tokenizer ...")
    bpe = BPETokenizer()
    if not bpe_path.exists() or force:
        texts = []
        with open(CACHE_DIR / "train.jsonl") as f:
            for line in f:
                r = json.loads(line)
                texts.extend([r["source"], r["asm"], r.get("ir", "")])
        bpe.build(texts, vocab_size=BPE_VOCAB_SIZE)
        bpe.save(bpe_path)
    else:
        bpe.load(bpe_path)
    print(f"          BPE vocab size: {bpe.vocab_size}")

    ia_path = CACHE_DIR / "ia_tokenizer.pkl"
    print("Step 3/3  Building instruction-aware tokenizer ...")
    ia = InstructionAwareTokenizer(bpe)
    ia.build()
    ia.save(ia_path)
    print(f"          IA ASM vocab size: {ia.asm_vocab_size}")
    print(f"          IA IR  vocab size: {ia.ir_vocab_size}")

    print("=" * 60)
    print(f"Setup complete.  Cache: {CACHE_DIR}")
    return bpe, ia


if __name__ == "__main__":
    setup()
