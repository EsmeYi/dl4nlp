#!/usr/bin/env python3
"""
generate_triplets.py  —  Build (source, IR variants, assembly, hw_metrics) JSONL triplets
                          from the AnghaBench dataset using the LLVM toolchain.

Pipeline per file:
    C source  --(clang -{opt} -S -emit-llvm)-->  LLVM IR (one per --opt-levels)
    IR (primary opt)  --(llc -{primary} -mcpu=<CPU>)-->  Assembly
    Assembly  --(llvm-mca -mcpu=<CPU>)       -->  Total Cycles, Block RThroughput, IPC

The primary opt level (first in --opt-levels, default O2) drives both ASM generation
and llvm-mca so that the scheduling model is consistent.  Additional opt levels (e.g. O3)
produce extra IR variants stored as ir_O3 for contrastive alignment diversity.

Output schema (one JSON object per line):
    id                  relative path from AnghaBench root
    source              raw C source
    ir_O2               LLVM IR at -O2  (always present)
    ir_O3               LLVM IR at -O3  (present if O3 in --opt-levels)
    asm                 x86-64 assembly generated from primary IR
    total_cycles        llvm-mca Total Cycles
    block_rthroughput   llvm-mca Block RThroughput
    ipc                 llvm-mca IPC (Instructions Per Cycle)

Usage examples:
    # Smoke-test on 200 files, 4 workers
    python generate_triplets.py --limit 200 --workers 4

    # Full run with O2 + O3 IR variants (default)
    python generate_triplets.py --workers 64 --output triplets.jsonl

    # Single opt level only
    python generate_triplets.py --workers 32 --opt-levels O2

    # Resume after interruption
    python generate_triplets.py --workers 64 --resume

    # Custom LLVM install (e.g. apt-installed LLVM 18)
    LLVM_BIN=/usr/lib/llvm-18/bin python generate_triplets.py --workers 64
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import re
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import tqdm

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

def _default_llvm_bin() -> str:
    env = os.environ.get("LLVM_BIN")
    if env:
        return env
    if platform.system() == "Darwin":
        return "/opt/homebrew/opt/llvm/bin"
    for candidate in [
        "/usr/lib/llvm-18/bin",
        "/usr/lib/llvm-17/bin",
        "/usr/lib/llvm-16/bin",
        "/usr/local/bin",
    ]:
        if os.path.isfile(os.path.join(candidate, "clang")):
            return candidate
    return "/usr/bin"


_DEFAULT_LLVM_BIN    = _default_llvm_bin()
_DEFAULT_MCPU        = "skylake"
_DEFAULT_TARGET      = ""
_DEFAULT_OPT_LEVELS  = ["O2", "O3"]
_COMPILE_TIMEOUT     = 30   # seconds per clang / llc call
_MCA_TIMEOUT         = 60   # seconds; llvm-mca can be slow on large functions

_RE_CYCLES      = re.compile(r"Total Cycles:\s+(\d+)")
_RE_RTHROUGHPUT = re.compile(r"Block RThroughput:\s+([\d.]+)")
_RE_IPC         = re.compile(r"IPC:\s+([\d.]+)")


# ---------------------------------------------------------------------------
# Per-file worker  (top-level + picklable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _process_file(
    args: tuple[Path, Path, str, str, str, str, str, list[str]],
) -> Optional[dict]:
    """
    Run the full C → IR variants → ASM → hw_metrics pipeline for one .c file.

    Returns a dict on success, None on any failure.  All exceptions are caught
    so a single bad file never terminates the worker pool.

    Tuple layout: (c_path, bench_root, clang, llc, mca, target, mcpu, opt_levels)
    The first element of opt_levels is the *primary* opt used for ASM + llvm-mca.
    """
    c_path, bench_root, clang_bin, llc_bin, mca_bin, target, mcpu, opt_levels = args

    try:
        source = c_path.read_text(errors="replace")
    except OSError:
        return None

    rel_id     = str(c_path.relative_to(bench_root))
    primary    = opt_levels[0]   # drives ASM generation and llvm-mca

    with tempfile.TemporaryDirectory() as tmp:

        # ---- Step 1: C → LLVM IR at each requested opt level ---------------
        ir_variants: dict[str, str] = {}
        primary_ll  = os.path.join(tmp, f"func_{primary}.ll")

        for opt in opt_levels:
            ll = os.path.join(tmp, f"func_{opt}.ll")
            clang_cmd = [clang_bin, f"-{opt}", "-S", "-emit-llvm", "-o", ll, str(c_path)]
            if target:
                clang_cmd[1:1] = [f"--target={target}"]
            try:
                r = subprocess.run(
                    clang_cmd,
                    timeout=_COMPILE_TIMEOUT,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                if r.returncode != 0 or not os.path.exists(ll):
                    return None
                ir_variants[opt] = open(ll).read()
            except (subprocess.TimeoutExpired, OSError):
                return None

        # ---- Step 2: Primary IR → Assembly  (mcpu must match llvm-mca) -----
        s = os.path.join(tmp, "func.s")
        llc_cmd = [llc_bin, f"-{primary}", f"-mcpu={mcpu}", "-o", s, primary_ll]
        try:
            r = subprocess.run(
                llc_cmd,
                timeout=_COMPILE_TIMEOUT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if r.returncode != 0 or not os.path.exists(s):
                return None
            asm = open(s).read()
        except (subprocess.TimeoutExpired, OSError):
            return None

        # ---- Step 3: llvm-mca → hardware metrics  (mcpu must match llc) ----
        mca_cmd = [mca_bin, f"-mcpu={mcpu}", s]
        try:
            r = subprocess.run(
                mca_cmd,
                timeout=_MCA_TIMEOUT,
                capture_output=True,
                text=True,
            )
            mca_out = r.stdout
        except (subprocess.TimeoutExpired, OSError):
            return None

        m_cycles = _RE_CYCLES.search(mca_out)
        m_rt     = _RE_RTHROUGHPUT.search(mca_out)
        m_ipc    = _RE_IPC.search(mca_out)
        if not m_cycles or not m_rt:
            return None

    result: dict = {
        "id":                rel_id,
        "source":            source,
        "asm":               asm,
        "total_cycles":      int(m_cycles.group(1)),
        "block_rthroughput": float(m_rt.group(1)),
        "ipc":               float(m_ipc.group(1)) if m_ipc else None,
    }
    for opt, ir_text in ir_variants.items():
        result[f"ir_{opt}"] = ir_text
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_seen_ids(output_path: Path) -> set[str]:
    seen: set[str] = set()
    if not output_path.exists():
        return seen
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                seen.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return seen


def _check_tools(*paths: str, log: logging.Logger) -> bool:
    ok = True
    for p in paths:
        if not os.path.isfile(p):
            log.error("Tool not found: %s  (set --llvm-bin or LLVM_BIN)", p)
            ok = False
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate (S, IR variants, A, hw_metrics) triplets from AnghaBench.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--bench", type=Path,
        default=Path(__file__).parent / "AnghaBench",
        help="AnghaBench root directory",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).parent / "triplets.jsonl",
        help="Output JSONL file",
    )
    parser.add_argument(
        "--workers", type=int,
        default=os.cpu_count() or 4,
        help="Parallel worker processes (default: all logical CPUs)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap input to N files (smoke-test mode)",
    )
    parser.add_argument(
        "--llvm-bin", type=str, default=_DEFAULT_LLVM_BIN,
        help="Directory containing clang / llc / llvm-mca",
    )
    parser.add_argument(
        "--target", type=str, default=_DEFAULT_TARGET,
        help="clang --target triple (empty = native)",
    )
    parser.add_argument(
        "--mcpu", type=str, default=_DEFAULT_MCPU,
        help="CPU model for both llc -mcpu and llvm-mca -mcpu (must match). "
             "Common values: skylake, haswell, znver3, znver4, cascadelake.",
    )
    parser.add_argument(
        "--opt-levels", nargs="+", default=_DEFAULT_OPT_LEVELS,
        metavar="LEVEL",
        help="Optimization levels for IR generation (e.g. O2 O3). "
             "The first level is used for ASM + llvm-mca.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Append to output and skip files already present in it",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    log = logging.getLogger(__name__)


    clang_bin = os.path.join(args.llvm_bin, "clang")
    llc_bin   = os.path.join(args.llvm_bin, "llc")
    mca_bin   = os.path.join(args.llvm_bin, "llvm-mca")

    if not _check_tools(clang_bin, llc_bin, mca_bin, log=log):
        raise SystemExit(1)

    log.info("CPU model  : %s  (llc + llvm-mca)", args.mcpu)
    log.info("Opt levels : %s  (primary=%s for ASM/metrics)", args.opt_levels, args.opt_levels[0])
    log.info("Target     : %s", args.target or "native")
    log.info("LLVM bin   : %s", args.llvm_bin)
    log.info("Workers    : %d", args.workers)

    c_files = sorted(args.bench.rglob("*.c"))
    if not c_files:
        log.error("No .c files found under %s", args.bench)
        raise SystemExit(1)

    if args.resume:
        seen   = _load_seen_ids(args.output)
        before = len(c_files)
        c_files = [p for p in c_files if str(p.relative_to(args.bench)) not in seen]
        log.info("Resume: skipping %d already-processed files", before - len(c_files))

    if args.limit:
        c_files = c_files[: args.limit]

    total = len(c_files)
    log.info("Files to process : %d  |  output : %s", total, args.output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_mode = "a" if args.resume else "w"

    worker_args = [
        (p, args.bench, clang_bin, llc_bin, mca_bin, args.target, args.mcpu, args.opt_levels)
        for p in c_files
    ]

    n_ok = n_fail = 0
    with open(args.output, write_mode) as out_f, \
         ProcessPoolExecutor(max_workers=args.workers) as pool, \
         tqdm.tqdm(total=total, unit="file", dynamic_ncols=True) as pbar:
        futures = {pool.submit(_process_file, a): a[0] for a in worker_args}
        for fut in as_completed(futures):
            pbar.update(1)
            result = fut.result()
            if result is None:
                n_fail += 1
            else:
                n_ok += 1
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            pbar.set_postfix(ok=n_ok, fail=n_fail, refresh=False)

    yield_rate = 100.0 * n_ok / total if total else 0.0
    log.info("Done.  %d triplets written  (%.1f%% yield, %d failures)  →  %s",
             n_ok, yield_rate, n_fail, args.output)


if __name__ == "__main__":
    main()