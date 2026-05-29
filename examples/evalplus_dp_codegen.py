"""Data-parallel HumanEval+/MBPP+ codegen for chat models.

Spawns world_size workers across world_size GPUs. Each worker handles
problems where `i % world_size == rank`. Writes EvalPlus-compatible
samples_rank<R>.jsonl shards, merges to samples.jsonl, then invokes
evalplus.evaluate on the merged file for grading.

Usage:
    python evalplus_dp_codegen.py \
        --model meta-llama/Llama-3.1-8B-Instruct \
        --dataset humaneval \
        --output_dir eval_results/<safe>/humaneval \
        --world_size 8

Why custom: EvalPlus's HF backend uses device_map='auto' (model parallel,
wastes 7 GPUs for an 8B model). We need data parallel to fit 164 problems
inside the cluster's ~40-min preemption window.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import torch.multiprocessing as mp

CHAT_INSTRUCTION = (
    "Please provide a self-contained Python script that solves the following "
    "problem in a markdown code block:"
)


def extract_code(text: str) -> str:
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1)
    return text


def worker(rank, world_size, args, problems_serial):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    tokenizer_path = args.model
    if "qwen3" in args.model.lower():
        tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        orig = tok.chat_template or ""
        if "enable_thinking" not in orig.split("\n", 1)[0]:
            tok.chat_template = "{%- set enable_thinking = false %}\n" + orig
        tmp = tempfile.mkdtemp(prefix=f"qwen3_no_think_rank{rank}_")
        tok.save_pretrained(tmp)
        tokenizer_path = tmp

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device).eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    my_problems = [p for i, p in enumerate(problems_serial) if i % world_size == rank]
    samples = []
    for idx, (task_id, prob) in enumerate(my_problems):
        prompt = CHAT_INSTRUCTION + "\n" + prob["prompt"]
        messages = [{"role": "user", "content": prompt}]
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(chat_text, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen_ids = out[0, inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(gen_ids, skip_special_tokens=True)
        solution = extract_code(response)
        samples.append({"task_id": task_id, "solution": solution})
        if rank == 0 and (idx + 1) % 3 == 0:
            print(f"[rank0] {idx+1}/{len(my_problems)}", flush=True)

    out_file = Path(args.output_dir) / f"samples_rank{rank}.jsonl"
    with open(out_file, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"[rank {rank}] wrote {len(samples)} samples -> {out_file}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", choices=["humaneval", "mbpp"], default="humaneval")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--world_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, only process first N problems (debug).")
    ap.add_argument("--skip_grade", action="store_true",
                    help="Skip evalplus.evaluate at the end (codegen only).")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset == "humaneval":
        from evalplus.data import get_human_eval_plus
        problems = get_human_eval_plus()
    else:
        from evalplus.data import get_mbpp_plus
        problems = get_mbpp_plus()

    problems_serial = sorted(problems.items())
    if args.limit > 0:
        problems_serial = problems_serial[: args.limit]
    print(f"[driver] {args.dataset}: {len(problems_serial)} problems, "
          f"world_size={args.world_size}", flush=True)

    mp.spawn(
        worker,
        args=(args.world_size, args, problems_serial),
        nprocs=args.world_size,
        join=True,
    )

    merged_path = Path(args.output_dir) / "samples.jsonl"
    with open(merged_path, "w") as out:
        for rank in range(args.world_size):
            shard = Path(args.output_dir) / f"samples_rank{rank}.jsonl"
            with open(shard) as f:
                for line in f:
                    out.write(line)
    print(f"[driver] merged -> {merged_path}", flush=True)

    if args.skip_grade:
        print("[driver] --skip_grade set, exiting before grading")
        return

    print("[driver] running evalplus.evaluate on merged samples", flush=True)
    cmd = [
        sys.executable, "-m", "evalplus.evaluate",
        "--dataset", args.dataset,
        "--samples", str(merged_path),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
