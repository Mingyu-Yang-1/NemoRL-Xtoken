#!/usr/bin/env python3
"""Full-suite eval for chat/instruct models — same task list as
`evaluate_full_suite.py` (tokenalign_upstream), but applies the model's
native chat template and formats few-shot examples as multi-turn.

Why a separate driver:
- The base-model eval (`evaluate_full_suite.py`) intentionally evaluates raw
  HF model + tokenizer without any chat formatting. That's correct for
  base/pretrained models (Llama-3.1-8B base, Qwen3-8B-Base, etc.).
- Chat/instruct models (Llama-3.1-8B-Instruct, Phi-4, Qwen3-8B,
  Qwen3-14B, Gemma-4-26B-A4B-it, …) are trained against their native
  chat templates. Evaluating them in raw mode systematically suppresses
  log-likelihoods and gives misleadingly low ARC/MMLU scores — see the
  conversation log around the Phi-4 ARC-E 72.69 anomaly.

This driver mirrors the base driver's task set, shot counts, metrics, and
JSON output schema, with two flags flipped at the lm-eval level:

  apply_chat_template=True       # wrap prompts in <|im_start|>... etc.
  fewshot_as_multiturn=True      # encode few-shot examples as user/assistant turns

Outputs `metrics_full_suite_chat.json` with the same shape as the base eval,
plus a `chat_template_applied: true` marker to disambiguate JSONs on disk.

Usage:
    python evaluate_instruct_full_suite.py --model_name meta-llama/Llama-3.1-8B-Instruct
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# ----------------------------------------------------------------------------
# Task spec — modern instruct-model suite (no ARC/OBQA/PIQA/HellaSwag/WinoG
# since those are saturated at the chat-model level). MMLU is kept as the
# single backward-comparison anchor against the base-model eval JSONs.
#
# Few-shot counts follow the conventions used by recent instruct-model
# papers (Llama-3.1-Instruct, Phi-4, Qwen3): MMLU at 5-shot (chat),
# GSM8K and code tasks at 0-shot (the chat template + instruction is the
# few-shot substitute), MATH/BBH still few-shot CoT.
# ----------------------------------------------------------------------------
TASK_CONFIG = [
    # task                       n-shot  primary metric                          fallback
    ("mmlu",                       5,    "acc,none",                              None),
    ("mmlu_pro",                   5,    "exact_match,custom-extract",            "exact_match,none"),
    # gpqa_diamond_cot_zeroshot dropped — Idavidrein/gpqa is gated on HF
    # and our HF_TOKEN doesn't have access. Add it back if/when access
    # is granted on https://huggingface.co/datasets/Idavidrein/gpqa
    ("ifeval",                     0,    "inst_level_strict_acc,none",            "prompt_level_strict_acc,none"),
    ("bbh",                        3,    "exact_match,flexible-extract",          "exact_match,strict-match"),
    ("gsm8k_cot",                  0,    "exact_match,flexible-extract",          "exact_match,strict-match"),
    ("minerva_math",               4,    "exact_match,none",                      None),
]
CODE_TASKS = [
    # EvalPlus variants (~110x more tests for HumanEval, ~35x for MBPP).
    # Strictly tighter pass@1 than the originals — typically ~10-20 pp
    # lower. Standard for chat-model papers (Llama-3.1-Instruct, Phi-4,
    # Qwen3 all report these). Falls back to vanilla pass@1 metric name
    # if EvalPlus's specific metric key shape differs.
    ("humaneval_plus",  0,    "pass@1,create_test",                    "pass_at_1,none"),
    ("mbpp_plus",       0,    "pass_at_1,none",                        "pass@1,none"),
]


def extract_metric(result_dict, task_name, primary_key, fallback_key):
    if "results" not in result_dict or task_name not in result_dict["results"]:
        return None
    task_res = result_dict["results"][task_name]
    if primary_key in task_res:
        return task_res[primary_key]
    if fallback_key is not None and fallback_key in task_res:
        print(f"  [warn] {task_name}: primary '{primary_key}' missing, used '{fallback_key}'")
        return task_res[fallback_key]
    print(f"  [warn] {task_name}: neither '{primary_key}' nor '{fallback_key}' present in {list(task_res.keys())}")
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True,
                   help="HF repo id, e.g. meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--bootstrap_iters", type=int, default=1000)
    p.add_argument("--parallelize", action="store_true",
                   help="HF device_map='auto' model parallelism — required for >24B")
    p.add_argument("--skip_tasks", type=str, default="",
                   help="Space-separated tasks to skip")
    p.add_argument("--skip_code", action="store_true",
                   help="Skip HumanEval + MBPP")
    p.add_argument("--output_path", type=str, default=None,
                   help="Directory to write metrics_full_suite_chat.json")
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="x_token")
    p.add_argument("--gen_kwargs", type=str, default=None,
                   help="lm-eval gen_kwargs (e.g., max_gen_toks=512)")
    args = p.parse_args()

    # Accept skip-list as whitespace-, comma-, or semicolon-separated. The
    # semicolon path is needed when passing via `sbatch --export=...` because
    # sbatch's CSV parser treats commas as variable separators (so
    # SKIP_TASKS=a,b,c gets truncated to "a"). Use SKIP_TASKS=a;b;c instead.
    if args.skip_tasks:
        normalized = args.skip_tasks.replace(",", " ").replace(";", " ")
        skip_tasks = set(t for t in normalized.split() if t)
    else:
        skip_tasks = set()

    # Build lm-eval HF model_args. lm-eval 0.4.10 forwards every key in
    # model_args to AutoModelForCausalLM.__init__, so chat_template_kwargs
    # cannot be threaded through this dict directly.
    #
    # Qwen3 thinking-mode handling: Qwen3 chat models default to emitting
    # <think>...</think> preambles, which sinks MCQ log-likelihood scoring
    # (we saw MMLU=27% / 35% — basically random). To fix this without
    # changing the metric, we pre-load the tokenizer, prepend
    # `{%- set enable_thinking = false %}` to its chat_template, save the
    # patched tokenizer to a temp dir, and point `tokenizer=` at the temp
    # dir. The patched template still wraps prompts in the model's native
    # <|im_start|>... format but skips the thinking block, so log P(letter)
    # MCQ scoring sees the proper answer-letter slot.
    tokenizer_path = args.model_name
    if "Qwen3" in args.model_name or "qwen3" in args.model_name:
        import tempfile
        from transformers import AutoTokenizer
        print(f"[qwen3-patch] Loading tokenizer for {args.model_name}", flush=True)
        tok = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
        orig = tok.chat_template or ""
        patched = "{%- set enable_thinking = false %}\n" + orig
        tok.chat_template = patched
        tmp = tempfile.mkdtemp(prefix="qwen3_no_think_")
        tok.save_pretrained(tmp)
        print(f"[qwen3-patch] Patched chat_template saved to {tmp}", flush=True)
        tokenizer_path = tmp

    model_args = {
        "pretrained": args.model_name,
        "tokenizer": tokenizer_path,
        "trust_remote_code": True,
    }
    if args.parallelize:
        model_args["parallelize"] = True

    # Output path
    if args.output_path is None:
        safe = args.model_name.replace("/", "__")
        args.output_path = f"eval_results/{safe}__chat"
    Path(args.output_path).mkdir(parents=True, exist_ok=True)

    # Compose final task list
    task_specs = list(TASK_CONFIG)
    if not args.skip_code:
        task_specs += CODE_TASKS

    task_specs = [t for t in task_specs if t[0] not in skip_tasks]

    print(f"\n──────────────────────────────────────────────────────────")
    print(f" Chat-template eval for {args.model_name}")
    print(f"──────────────────────────────────────────────────────────")
    print(f"  apply_chat_template  : True")
    print(f"  fewshot_as_multiturn : True")
    print(f"  batch_size           : {args.batch_size}")
    print(f"  parallelize          : {args.parallelize}")
    print(f"  output_path          : {args.output_path}")
    print(f"  tasks                : {[t[0] for t in task_specs]}")
    print(f"==========================================================\n")

    # lm-eval import is deferred so --help works without lm-eval installed.
    from lm_eval import simple_evaluate

    results = {}
    for task, num_fewshot, primary, fallback in task_specs:
        if task in skip_tasks:
            print(f"[skipped] {task}")
            results[task] = None
            continue

        is_code = task in {"humaneval", "mbpp", "humaneval_plus", "mbpp_plus"}
        print(f"\n=== {task} ({num_fewshot}-shot{', code/generation' if is_code else ''}) ===")

        # Code tasks expect the model to complete a function body, not
        # produce a conversational response with markdown. With
        # apply_chat_template=True, chat models output things like
        # "Here is the function:\n```python\n...\n```" — lm-eval's
        # code_eval metric tries to exec that whole string and fails.
        # The canonical instruct-model methodology disables the chat
        # template for code tasks (the model still understands a
        # function-prefix prompt thanks to instruction tuning).
        chat_on = not is_code
        eval_kwargs = dict(
            model="hf",
            model_args=model_args,
            tasks=[task],
            num_fewshot=num_fewshot,
            batch_size=args.batch_size,
            limit=args.limit,
            bootstrap_iters=args.bootstrap_iters,
            apply_chat_template=chat_on,
            fewshot_as_multiturn=chat_on,
        )
        if is_code:
            eval_kwargs["confirm_run_unsafe_code"] = True
        if args.gen_kwargs:
            eval_kwargs["gen_kwargs"] = args.gen_kwargs

        try:
            result = simple_evaluate(**eval_kwargs)
            score = extract_metric(result, task, primary, fallback)
            if score is not None:
                print(f"  → {score:.4f} ({score*100:.2f}%)")
            results[task] = score
        except Exception as e:
            import traceback
            print(f"  [error] {task} failed: {type(e).__name__}: {e}")
            tb = traceback.format_exc()
            # Print first 30 lines of traceback to disambiguate empty-message exceptions.
            print("\n".join(tb.split("\n")[:30]), flush=True)
            results[task] = None

    # Group averages (matches base driver's reporting).
    def avg(keys):
        vals = [results.get(k) for k in keys if results.get(k) is not None and isinstance(results.get(k), (int, float))]
        return sum(vals) / len(vals) if vals else None

    knowledge = ["mmlu", "mmlu_pro", "gpqa_diamond"]
    reasoning = ["gsm8k_cot", "minerva_math", "bbh"]
    instruction = ["ifeval"]
    code = ["humaneval_plus", "mbpp_plus"]

    print(f"\n============================================================")
    print(f"Results for HuggingFace (chat template): {args.model_name}:")
    print(f"============================================================")
    for task, num_fewshot, *_ in task_specs:
        v = results.get(task)
        v_str = f"{v*100:.2f}%" if isinstance(v, (int, float)) else "N/A"
        print(f"  {task:18s} ({num_fewshot}-shot): {v_str}")
    print()
    for label, keys in (("knowledge_avg  ", knowledge),
                       ("reasoning_avg  ", reasoning),
                       ("instruction_avg", instruction),
                       ("code_avg       ", code)):
        a = avg(keys)
        if a is not None:
            print(f"  {label} ({len(keys)} tasks): {a:.4f} ({a*100:.2f}%)")
    print(f"============================================================\n")

    out = {
        "model_source": f"HuggingFace (chat template): {args.model_name}",
        "checkpoint_iteration": None,
        "chat_template_applied": True,
        "fewshot_as_multiturn": True,
        "task_config": [
            {"task": t, "num_fewshot": n, "metric": p}
            for t, n, p, _ in task_specs
        ],
        "results": results,
    }
    out_file = os.path.join(args.output_path, "metrics_full_suite_chat.json")
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_file}")

    if args.use_wandb:
        try:
            import wandb
            wandb.init(project=args.wandb_project, name=f"eval-chat-{args.model_name}")
            wandb.log({f"eval_chat/{k}": v for k, v in results.items() if v is not None})
            wandb.finish()
        except Exception as e:
            print(f"W&B logging failed: {e}")


if __name__ == "__main__":
    sys.exit(main())
