"""
static_feature_baseline.py — Hand-crafted static features → Ridge regression for IPC prediction.

Extracts interpretable features from C source and x86-64 assembly without any learned
representation. This is the key CGO reviewer baseline: can simple counting features
already predict IPC without compilation or training?

Features extracted:
  Source (C):
    - loc            : non-blank, non-comment lines
    - num_loops      : for/while/do occurrences
    - num_branches   : if/else/switch occurrences
    - num_calls      : function call heuristic (word followed by '(')
    - num_ptrs       : pointer dereference/declaration '*' count
    - num_simd_intrin: SIMD intrinsic calls (_mm*, __m128*, __m256*)
    - num_arith      : +, -, *, / operator occurrences

  Assembly (x86-64):
    - asm_insn_total : total instruction lines
    - asm_simd       : SIMD instructions (v prefix: vmovaps, vaddps, etc.)
    - asm_mem        : memory ops (mov + '[' bracket or 'ptr')
    - asm_branch     : branch instructions (j*)
    - asm_call       : call instructions
    - asm_fp         : scalar FP (divsd, sqrtsd, mulsd, addsd, subsd)
    - asm_mul        : integer multiply (imul, mul)

Target: log1p(IPC)  — same transform used in the neural model's IPC kernel.

Reports R² on val and test, matching evaluate_probe() output format.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

CACHE_DIR = Path.home() / ".cache" / "dl4nlp_crossmodal"


# ── Feature extraction ────────────────────────────────────────────────────────

_SIMD_INTRIN = re.compile(r'\b(_mm[0-9]*_|__m128|__m256|__m512)\w*\s*\(')
_LOOP        = re.compile(r'\b(for|while|do)\b')
_BRANCH      = re.compile(r'\b(if|else|switch)\b')
_CALL        = re.compile(r'\b[a-zA-Z_]\w*\s*\(')
_PTR         = re.compile(r'\*')
_ARITH       = re.compile(r'[+\-*/]')

_ASM_INSN    = re.compile(r'^\s+\w')           # lines starting with whitespace + word char
_ASM_SIMD    = re.compile(r'^\s+v[a-z]')       # AVX/SSE (v-prefix)
_ASM_MEM     = re.compile(r'^\s+\w+.*(\[|ptr\b)', re.IGNORECASE)
_ASM_BRANCH  = re.compile(r'^\s+j[a-z]')       # jmp, je, jne, jl, jg, ...
_ASM_CALL    = re.compile(r'^\s+call\b')
_ASM_FP      = re.compile(r'^\s+(div|sqrt|mul|add|sub)s[sd]\b')
_ASM_MUL     = re.compile(r'^\s+i?mul\b')


def src_features(code: str) -> list:
    lines = code.splitlines()
    non_blank = [l for l in lines if l.strip() and not l.strip().startswith("//")]
    loc = len(non_blank)
    text = code
    return [
        loc,
        len(_LOOP.findall(text)),
        len(_BRANCH.findall(text)),
        len(_CALL.findall(text)),
        len(_PTR.findall(text)),
        len(_SIMD_INTRIN.findall(text)),
        len(_ARITH.findall(text)),
    ]


def asm_features(asm: str) -> list:
    lines = asm.splitlines()
    total  = sum(1 for l in lines if _ASM_INSN.match(l))
    simd   = sum(1 for l in lines if _ASM_SIMD.match(l))
    mem    = sum(1 for l in lines if _ASM_MEM.match(l))
    branch = sum(1 for l in lines if _ASM_BRANCH.match(l))
    call   = sum(1 for l in lines if _ASM_CALL.match(l))
    fp     = sum(1 for l in lines if _ASM_FP.match(l))
    mul    = sum(1 for l in lines if _ASM_MUL.match(l))
    return [total, simd, mem, branch, call, fp, mul]


def load_split(split: str):
    X, y = [], []
    with open(CACHE_DIR / f"{split}.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            feats = src_features(rec["source"]) + asm_features(rec["asm"])
            X.append(feats)
            y.append(np.log1p(float(rec["ipc"])))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading splits ...")
    X_train, y_train = load_split("train")
    X_val,   y_val   = load_split("val")
    X_test,  y_test  = load_split("test")
    print(f"  train={len(y_train)}  val={len(y_val)}  test={len(y_test)}")
    print(f"  feature dim: {X_train.shape[1]}")

    feature_names = [
        "src_loc", "src_loops", "src_branches", "src_calls",
        "src_ptrs", "src_simd_intrin", "src_arith",
        "asm_total", "asm_simd", "asm_mem", "asm_branch",
        "asm_call", "asm_fp", "asm_mul",
    ]

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge",  Ridge(alpha=1.0)),
    ])
    model.fit(X_train, y_train)

    val_r2  = model.score(X_val,  y_val)
    test_r2 = model.score(X_test, y_test)

    print(f"\n--- Static Feature Baseline (Ridge on log1p-IPC) ---")
    print(f"[VAL]  probe_r2_ipc:  {val_r2:.6f}")
    print(f"[TEST] probe_r2_ipc:  {test_r2:.6f}")

    # Feature importance (absolute ridge coefficients after scaling)
    coefs = np.abs(model.named_steps["ridge"].coef_)
    ranked = sorted(zip(feature_names, coefs), key=lambda x: -x[1])
    print("\nFeature importances (|coef|):")
    for name, c in ranked:
        print(f"  {name:<22} {c:.4f}")

    # Baseline: predict mean of training targets
    mean_pred = np.full_like(y_val, y_train.mean())
    null_r2 = 1 - np.sum((y_val - mean_pred)**2) / np.sum((y_val - y_val.mean())**2)
    print(f"\nNull model R² (predict mean): {null_r2:.6f}")

    results = {
        "val_r2": float(val_r2),
        "test_r2": float(test_r2),
        "feature_importances": {n: float(c) for n, c in ranked},
    }
    out = Path(__file__).parent / "static_feature_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    main()
