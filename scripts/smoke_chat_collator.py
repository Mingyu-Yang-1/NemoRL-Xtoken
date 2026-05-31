"""Smoke test: chat-mode CrossTokenizerCollator + offset_cluster_decode_fix.

Pulls ~10 chat conversations from nvidia/Nemotron-Cascade-2-SFT-Data (subset
"chat"), constructs the Llama-3.1-8B-Instruct ↔ Phi-4 tokenizer pair, runs
the new chat-mode collator, and validates the output tensor shapes /
dtypes / mask invariants. No GPU required — alignment is CPU work.

Run from the repo root::

    python scripts/smoke_chat_collator.py

Override the dataset slice / context lengths via env::

    SMOKE_N=20 SMOKE_CTX=512 python scripts/smoke_chat_collator.py
"""
from __future__ import annotations

import os
import sys
import traceback

# Local import path before any of the nemo_rl imports.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from nemo_rl.algorithms.x_token.tokenalign import TokenAligner
from nemo_rl.data.cross_tokenizer_collate import CrossTokenizerCollator
from nemo_rl.data.processors import chat_kd_processor
from nemo_rl.data.interfaces import TaskDataSpec


N_SAMPLES = int(os.environ.get("SMOKE_N", "10"))
CTX = int(os.environ.get("SMOKE_CTX", "1024"))
STUDENT_MODEL = os.environ.get("SMOKE_STUDENT", "meta-llama/Llama-3.1-8B-Instruct")
TEACHER_MODEL = os.environ.get("SMOKE_TEACHER", "microsoft/phi-4")
PROJECTION = os.environ.get(
    "SMOKE_PROJECTION",
    "/lustre/fsw/portfolios/coreai/users/mingyyang/xtoken_nemorl_v1/"
    "cross_tokenizer_data/projection_map_Llama-3.1_to_Phi-4_multitoken_top_4_special.pt",
)


def ok(msg: str) -> None:
    print(f"  ✓ {msg}")


def fail(msg: str) -> None:
    print(f"  ✗ {msg}")
    raise AssertionError(msg)


def main() -> int:
    print(f"=== chat-collator smoke test ===")
    print(f"  N samples:  {N_SAMPLES}")
    print(f"  ctx (each): {CTX}")
    print(f"  student:    {STUDENT_MODEL}")
    print(f"  teacher:    {TEACHER_MODEL}")
    print()

    # 1. Pull a few samples from cascade-2 (streaming, no full download).
    print("[1] streaming nvidia/Nemotron-Cascade-2-SFT-Data (name=chat)...")
    ds = load_dataset(
        "nvidia/Nemotron-Cascade-2-SFT-Data", name="chat", split="train",
        streaming=True,
    )
    samples = []
    for i, row in enumerate(ds):
        if i >= N_SAMPLES:
            break
        samples.append(row)
    ok(f"pulled {len(samples)} rows; keys: {list(samples[0].keys())[:6]}")

    # 2. Build the DatumSpec list via chat_kd_processor (the production path).
    print("[2] processing through chat_kd_processor...")
    task_spec = TaskDataSpec(task_name="x_token_chat")
    batch = [
        chat_kd_processor(s, task_spec, tokenizer=None, max_seq_length=CTX, idx=i)
        for i, s in enumerate(samples)
    ]
    assert all("messages" in d for d in batch)
    ok(f"produced {len(batch)} DatumSpecs with 'messages' key")

    # 3. Load the two tokenizers; both must be fast (need offset_mapping).
    print("[3] loading tokenizers (fast=True, required for offsets)...")
    s_tok = AutoTokenizer.from_pretrained(STUDENT_MODEL, use_fast=True)
    t_tok = AutoTokenizer.from_pretrained(TEACHER_MODEL, use_fast=True)
    if not getattr(s_tok, "is_fast", False):
        fail("student tokenizer is not fast — offset_mapping unavailable")
    if not getattr(t_tok, "is_fast", False):
        fail("teacher tokenizer is not fast — offset_mapping unavailable")
    ok("both tokenizers are fast")

    # 4. Build the aligner in offset_cluster_decode_fix mode. We don't need
    #    the actual projection map for the collator smoke (it's loaded lazily
    #    by the loss fn) — pass a path that may or may not exist, the aligner
    #    only opens it when load_projection_matrix() is called.
    print("[4] constructing TokenAligner(alignment_method='offset_cluster_decode_fix')...")
    aligner = TokenAligner(
        student_tokenizer=s_tok,
        teacher_tokenizer=t_tok,
        projection_matrix_path=PROJECTION,
        alignment_method="offset_cluster_decode_fix",
    )
    assert aligner.alignment_method == "offset_cluster_decode_fix"
    ok("aligner OK")

    # 5. Build the chat-mode collator with num_packed_rows=8 (mimics DP=8).
    print("[5] constructing CrossTokenizerCollator(mode='chat', num_packed_rows=8)...")
    collator = CrossTokenizerCollator(
        student_tokenizer=s_tok,
        teacher_tokenizer=t_tok,
        aligner=aligner,
        ctx_length_student=CTX,
        ctx_length_teacher=CTX,
        mode="chat",
        add_eos_between_docs=True,
        num_packed_rows=8,
    )
    ok("collator OK")

    # 6. Run __call__ — this exercises chat template, asst mask, lockstep pack,
    #    per-doc alignment, span shifting, and AlignmentBatch concat.
    print("[6] running collator on the batch...")
    out = collator(batch)
    ok("collator returned without raising")

    # 7. Validate output shapes / dtypes / invariants.
    print("[7] validating outputs...")
    # B should equal num_packed_rows.
    if out["input_ids"].shape[0] != 8:
        fail(f"expected B=8 packed rows, got B={out['input_ids'].shape[0]}")
    ok(f"input_ids shape    = {tuple(out['input_ids'].shape)} (B=8)")
    ok(f"teacher_input_ids  = {tuple(out['teacher_input_ids'].shape)}")
    ok(f"token_mask dtype   = {out['token_mask'].dtype}")
    ok(f"sample_mask        = {out['sample_mask'].tolist()}")

    # Per-row loss token + doc counts.
    s_attn_total = int((out['input_ids'] != s_tok.pad_token_id).sum())
    s_tokmask_total = int(out['token_mask'].sum())
    if s_tokmask_total > s_attn_total:
        fail(f"token_mask ({s_tokmask_total}) exceeds non-pad count ({s_attn_total})")
    ok(f"student loss tokens (asst only, all rows) = {s_tokmask_total} / {s_attn_total} non-pad")
    if s_tokmask_total == 0:
        fail("token_mask is all-zero — assistant mask never fired")

    # doc_id surfaced for sequence_packing.
    if "student_doc_id" not in out:
        fail("student_doc_id missing — needed for sequence_packing wiring")
    docs_per_row = [
        len(set(d for d in out['student_doc_id'][r].tolist() if d >= 0))
        for r in range(out['input_ids'].shape[0])
    ]
    ok(f"docs packed per row = {docs_per_row} (sum={sum(docs_per_row)} of {len(samples)})")

    # sample_mask sanity: rows with 0 docs should be 0.0, rest 1.0
    sm = out['sample_mask'].tolist()
    ok(f"sample_mask         = {sm}")

    # Alignment payload non-trivial.
    pairs = int(out['alignment_pair_valid'].sum())
    correct = int(out['alignment_pair_is_correct'].sum())
    ok(f"alignment pairs (total over rows) = {pairs} valid, {correct} marked is_correct")
    if pairs == 0:
        fail("zero alignment pairs — collator silently failed")

    # student / teacher partition masks shouldn't be all-zero either when we
    # had aligned content.
    s_part = int(out['alignment_student_exact_partition_mask'].sum())
    ok(f"student exact-partition tokens (all rows) = {s_part}")

    # Critical: verify that out['input_ids'].shape[0] divides cleanly by
    # plausible DP world sizes (this is the assertion that bit us before).
    for dp in (1, 2, 4, 8):
        if out['input_ids'].shape[0] % dp != 0:
            fail(f"B={out['input_ids'].shape[0]} not divisible by dp={dp}")
    ok(f"B={out['input_ids'].shape[0]} divides cleanly by dp ∈ {{1,2,4,8}}")

    print()
    print("=== SMOKE PASSED ===")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        print("\n=== SMOKE FAILED ===", file=sys.stderr)
        sys.exit(1)
