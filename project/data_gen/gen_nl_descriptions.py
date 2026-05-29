"""
gen_nl_descriptions.py — Generate NL descriptions for all functions using Gemini Flash.

Reads train/val/test.jsonl from cache, calls Gemini Flash API in batches of 10,
writes results to nl_train.jsonl / nl_val.jsonl / nl_test.jsonl in the same cache dir.

Each output line: {"id": <id>, "nl": "<1-2 sentence description>"}

Run: python gen_nl_descriptions.py
Resume: safe to re-run — skips already-processed IDs.
"""

import json
import os
import sys
import time
from pathlib import Path

import requests

# ── Config ───────────────────────────────────────────────────────────────────

CACHE_DIR   = Path("/cephyr/users/lirongy/Alvis/.cache/dl4nlp_crossmodal")
API_KEY     = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("Set the GEMINI_API_KEY environment variable before running.")
MODEL       = "gemini-2.0-flash"
BATCH_SIZE  = 10
MAX_RETRIES = 5
RETRY_DELAY = 10  # seconds

SPLITS = [
    ("train.jsonl", "nl_train.jsonl"),
    ("val.jsonl",   "nl_val.jsonl"),
    ("test.jsonl",  "nl_test.jsonl"),
]

API_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}"
    f":generateContent?key={API_KEY}"
)

# ── Prompt ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a performance-aware code analysis assistant. For each C function, write a concise \
1-2 sentence natural language description that captures:
1. What the function does (its algorithmic purpose)
2. Its likely performance characteristics using vocabulary like: \
memory-bound, compute-intensive, cache-friendly, vectorizable, branch-heavy, \
loop-carried dependency, high throughput, latency-bound, sequential access, random access.

Use IPC (instructions per cycle) and reciprocal throughput as hints:
- IPC > 2.5: likely compute-intensive or well-pipelined
- IPC 1.0-2.5: balanced
- IPC < 1.0: likely memory-bound or branch-heavy
- Low reciprocal throughput (< 1.0): high throughput potential

Do NOT mention raw numbers in the description. Write naturally for a developer audience.\
"""

def make_batch_prompt(items):
    """Build a prompt for a batch of functions."""
    parts = []
    for i, item in enumerate(items):
        src = item["source"][:800]  # truncate long functions
        ipc = item["ipc"]
        rt  = item["block_rthroughput"]
        parts.append(
            f"[Function {i+1}]\n"
            f"IPC={ipc:.2f}, RecipThroughput={rt:.2f}\n"
            f"```c\n{src}\n```"
        )
    body = "\n\n".join(parts)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Describe each of the following {len(items)} functions. "
        f"Respond with exactly {len(items)} lines, one description per line, "
        f"numbered as '1. ...', '2. ...', etc.\n\n"
        f"{body}"
    )


def call_api(prompt: str) -> str:
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1024},
    }
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(API_URL, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = RETRY_DELAY * (attempt + 1)
                print(f"    API error: {e}, retrying in {wait}s ...", flush=True)
                time.sleep(wait)
            else:
                raise


def parse_numbered_list(text: str, expected: int) -> list[str]:
    """Extract descriptions from a numbered list response."""
    lines = []
    for line in text.strip().split("\n"):
        line = line.strip()
        # Match lines starting with a number and dot/period
        for i in range(1, expected + 1):
            if line.startswith(f"{i}.") or line.startswith(f"{i})"):
                desc = line[len(f"{i}."):].strip().lstrip(")").strip()
                if desc:
                    lines.append(desc)
                break
    # Fallback: if parsing fails, split by non-empty lines
    if len(lines) != expected:
        fallback = [l.strip() for l in text.strip().split("\n") if l.strip()]
        # Remove numbering prefixes from fallback
        cleaned = []
        for l in fallback:
            import re
            l = re.sub(r"^\d+[\.\)]\s*", "", l)
            if l:
                cleaned.append(l)
        if len(cleaned) >= expected:
            return cleaned[:expected]
        # Last resort: pad with generic description
        while len(lines) < expected:
            lines.append("A C function performing computational operations.")
    return lines[:expected]


def load_existing(out_path: Path) -> set:
    """Load already-processed IDs from output file."""
    seen = set()
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    seen.add(json.loads(line)["id"])
                except Exception:
                    pass
    return seen


def process_split(in_name: str, out_name: str):
    in_path  = CACHE_DIR / in_name
    out_path = CACHE_DIR / out_name

    print(f"\n=== Processing {in_name} → {out_name} ===", flush=True)

    # Load all records
    records = []
    with open(in_path) as f:
        for line in f:
            records.append(json.loads(line))
    total = len(records)

    # Skip already done
    seen = load_existing(out_path)
    todo = [r for r in records if r["id"] not in seen]
    print(f"  Total: {total}, already done: {len(seen)}, remaining: {len(todo)}", flush=True)

    if not todo:
        print("  Nothing to do.", flush=True)
        return

    # Process in batches
    out_f = open(out_path, "a")
    n_done = len(seen)
    t0 = time.time()

    for batch_start in range(0, len(todo), BATCH_SIZE):
        batch = todo[batch_start : batch_start + BATCH_SIZE]
        prompt = make_batch_prompt(batch)

        try:
            response = call_api(prompt)
            descriptions = parse_numbered_list(response, len(batch))
        except Exception as e:
            print(f"  FATAL error at batch {batch_start}: {e}", flush=True)
            print("  Writing placeholder descriptions and continuing ...", flush=True)
            descriptions = ["A C function performing computational operations."] * len(batch)

        for item, desc in zip(batch, descriptions):
            out_f.write(json.dumps({"id": item["id"], "nl": desc}) + "\n")
        out_f.flush()

        n_done += len(batch)
        elapsed = time.time() - t0
        rate    = n_done / elapsed if elapsed > 0 else 0
        eta     = (len(todo) - n_done) / rate if rate > 0 else 0
        print(
            f"  {n_done}/{total}  elapsed={elapsed:.0f}s  rate={rate:.1f}/s  ETA={eta:.0f}s",
            flush=True,
        )

        # Respect rate limits: ~1 req/sec is safe for Gemini Flash free tier
        time.sleep(1.0)

    out_f.close()
    print(f"  Done! Output: {out_path}", flush=True)


def main():
    print(f"Using model: {MODEL}", flush=True)
    print(f"Cache dir:   {CACHE_DIR}", flush=True)

    for in_name, out_name in SPLITS:
        process_split(in_name, out_name)

    print("\nAll splits done.", flush=True)


if __name__ == "__main__":
    main()
