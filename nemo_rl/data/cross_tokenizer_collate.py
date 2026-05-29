# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Collator that tokenizes raw text twice (student + teacher) and aligns.

The collator runs inside DataLoader worker processes. It does:

1. Tokenizes the same source text once with the student tokenizer and once
   with the teacher tokenizer (no chat template, no special handling).
2. Calls :class:`TokenAligner.align` to produce a dense-padded
   :class:`AlignmentBatch` covering all three loss modes (P-KL, gold_loss,
   xtoken_loss).
3. Returns a :class:`BatchedDataDict` with the keys
   :class:`Policy.train` expects (``input_ids``, ``input_lengths``,
   ``token_mask``, ``sample_mask``) plus teacher tensors and alignment
   tensors.

Loss-side projection-matrix work happens inside the loss fn; nothing related
to KL/CE math runs here.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from nemo_rl.algorithms.x_token.tokenalign import AlignmentBatch, TokenAligner
from nemo_rl.data.interfaces import DatumSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


class CrossTokenizerCollator:
    """Tokenize twice, align once, return a flat tensor batch.

    Two modes:

    * ``mode="text"`` — the original path: each ``DatumSpec`` carries a raw
      text string under ``text_key`` (default ``"raw_text"``). Tokenized
      once per side, aligned end-to-end. Used by the base-model cross-
      tokenizer distillation runs.

    * ``mode="chat"`` — SFT/instruct path: each ``DatumSpec`` carries
      ``messages: list[{"role", "content"}]``. The collator:

      1. Applies each side's chat template (``enable_thinking=False`` when
         supported), tokenizes with ``return_offsets_mapping=True``, and
         builds a per-token assistant mask via char-offset matching.
         ``<think>...</think>`` sub-spans inside assistant content are
         excluded.
      2. Lockstep-packs the batch's conversations into ONE packed row per
         side (one accepted iff it fits BOTH student and teacher budgets,
         to keep doc counts in sync for the cross-tokenizer aligner).
      3. Runs ``aligner.align_one_offset`` PER DOCUMENT, shifts the per-doc
         spans into pack-global indices, and concatenates them into a
         single :class:`AlignmentBatch` matching the text-mode contract.
      4. Multiplies the assistant mask into ``token_mask`` so the
         downstream loss only fires on assistant content tokens.

      In chat mode the output batch dimension is always 1 (one packed row
      per ``__call__``). Use grad-accumulation / sequence packing in the
      policy to scale effective batch size.

    Args:
        student_tokenizer: HF tokenizer matching the student model. Must be
            a fast tokenizer in chat mode (needs ``return_offsets_mapping``).
        teacher_tokenizer: HF tokenizer matching the teacher model.
        aligner: Pre-constructed :class:`TokenAligner`. Must be set to
            ``alignment_method="offset_cluster_decode_fix"`` in chat mode.
        ctx_length_student: Per-side token budget. In ``text`` mode this is
            the hard tokenization cap. In ``chat`` mode this is the
            lockstep packing budget AND the padded output length.
        ctx_length_teacher: Same on the teacher side.
        make_seq_div_by_student: Round student sequence length up to a
            multiple of this value (typically TP * CP * 2 for DTensor V2).
        make_seq_div_by_teacher: Same for the teacher side.
        text_key: Field on :class:`DatumSpec` that holds the raw text in
            ``text`` mode. Unused in ``chat`` mode.
        mode: ``"text"`` (default) or ``"chat"``.
        messages_key: Field that holds the per-sample message list in
            ``chat`` mode. Default ``"messages"``.
        add_eos_between_docs: Append the per-side EOS token between packed
            documents. Default True; matches Pavlo's reference behavior.
    """

    def __init__(
        self,
        *,
        student_tokenizer: PreTrainedTokenizerBase,
        teacher_tokenizer: PreTrainedTokenizerBase,
        aligner: TokenAligner,
        ctx_length_student: int,
        ctx_length_teacher: int,
        make_seq_div_by_student: int = 1,
        make_seq_div_by_teacher: int = 1,
        text_key: str = "raw_text",
        mode: str = "text",
        messages_key: str = "messages",
        add_eos_between_docs: bool = True,
    ):
        if mode not in ("text", "chat"):
            raise ValueError(f"mode must be 'text' or 'chat', got {mode!r}")
        if mode == "chat" and aligner.alignment_method != "offset_cluster_decode_fix":
            raise ValueError(
                "mode='chat' requires the aligner to be set to "
                "alignment_method='offset_cluster_decode_fix' (chat mode "
                "tokenizes with return_offsets_mapping=True and aligns "
                "per-document using char offsets)."
            )

        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer
        self.aligner = aligner
        self.ctx_length_student = ctx_length_student
        self.ctx_length_teacher = ctx_length_teacher
        self.make_seq_div_by_student = make_seq_div_by_student
        self.make_seq_div_by_teacher = make_seq_div_by_teacher
        self.text_key = text_key
        self.mode = mode
        self.messages_key = messages_key
        self.add_eos_between_docs = add_eos_between_docs
        # Defensive: HF tokenizers without a pad token can't pad batches.
        if self.student_tokenizer.pad_token_id is None:
            self.student_tokenizer.pad_token = self.student_tokenizer.eos_token
        if self.teacher_tokenizer.pad_token_id is None:
            self.teacher_tokenizer.pad_token = self.teacher_tokenizer.eos_token

    def __call__(self, batch: List[DatumSpec]) -> BatchedDataDict[Any]:
        if self.mode == "chat":
            return self._call_chat(batch)
        return self._call_text(batch)

    # ------------------------------------------------------------------ #
    # text-mode (existing behavior)
    # ------------------------------------------------------------------ #
    def _call_text(self, batch: List[DatumSpec]) -> BatchedDataDict[Any]:
        texts = [datum[self.text_key] for datum in batch]
        student_input_ids, student_attention_mask = self._tokenize_batch(
            texts,
            self.student_tokenizer,
            self.ctx_length_student,
            self.make_seq_div_by_student,
        )
        teacher_input_ids, teacher_attention_mask = self._tokenize_batch(
            texts,
            self.teacher_tokenizer,
            self.ctx_length_teacher,
            self.make_seq_div_by_teacher,
        )
        alignment = self.aligner.align(student_input_ids, teacher_input_ids)

        sample_mask = torch.tensor(
            [datum["loss_multiplier"] for datum in batch], dtype=torch.float32
        )
        idx = [datum["idx"] for datum in batch]

        return self._build_batched_dict(
            student_input_ids=student_input_ids,
            student_attention_mask=student_attention_mask,
            student_token_mask=student_attention_mask,
            teacher_input_ids=teacher_input_ids,
            teacher_attention_mask=teacher_attention_mask,
            teacher_token_mask=teacher_attention_mask,
            alignment=alignment,
            sample_mask=sample_mask,
            idx=idx,
        )

    # ------------------------------------------------------------------ #
    # chat-mode (SFT/instruct): lockstep packing + per-doc alignment
    # ------------------------------------------------------------------ #
    def _call_chat(self, batch: List[DatumSpec]) -> BatchedDataDict[Any]:
        # Pavlo's design for cross-tokenizer chat: render once on the
        # STUDENT side, then tokenize that same rendered text with the
        # teacher tokenizer too. Both sides' offsets then live in the same
        # character coordinate system, which is what makes
        # offset_cluster_decode_fix produce meaningful cross-side pairs.
        # Trade-off: the teacher sees Llama-format scaffold (special token
        # strings broken into sub-tokens by its BPE) instead of its own
        # native chat scaffolding. For KL extraction this works in
        # practice; the teacher still produces a probability distribution
        # we can target.
        messages_list = [datum[self.messages_key] for datum in batch]

        per_doc_student: List[Dict[str, Any] | None] = []
        per_doc_teacher: List[Dict[str, Any] | None] = []
        for m in messages_list:
            student_text, s_tok = _render_student_and_tokenize(
                self.student_tokenizer, m, self.ctx_length_student,
            )
            if student_text is None:
                per_doc_student.append(None)
                per_doc_teacher.append(None)
                continue
            # Teacher tokenizes the SAME student-rendered text — NOT its
            # own template render. Offsets land in student_text's space.
            t_tok = _tokenize_text_with_assistant_mask(
                self.teacher_tokenizer, student_text, m,
                self.ctx_length_teacher,
            )
            per_doc_student.append(s_tok)
            per_doc_teacher.append(t_tok)

        # Drop rows that failed to tokenize on either side.
        keep = [
            i for i in range(len(messages_list))
            if per_doc_student[i] is not None and per_doc_teacher[i] is not None
        ]

        # Lockstep pack into a single row per side. Tracks doc spans.
        s_pack, t_pack = _pack_lockstep(
            [per_doc_student[i] for i in keep],
            [per_doc_teacher[i] for i in keep],
            student_pad_id=self.student_tokenizer.pad_token_id,
            teacher_pad_id=self.teacher_tokenizer.pad_token_id,
            student_eos_id=(
                self.student_tokenizer.eos_token_id
                if self.add_eos_between_docs
                else None
            ),
            teacher_eos_id=(
                self.teacher_tokenizer.eos_token_id
                if self.add_eos_between_docs
                else None
            ),
            student_max_len=self.ctx_length_student,
            teacher_max_len=self.ctx_length_teacher,
        )

        # Pad packed lengths up to make_seq_div_by_* by appending pads.
        s_input_ids, s_attn, s_asst, s_doc_id = _pad_packed(
            s_pack,
            pad_id=self.student_tokenizer.pad_token_id,
            divisor=self.make_seq_div_by_student,
        )
        t_input_ids, t_attn, t_asst, t_doc_id = _pad_packed(
            t_pack,
            pad_id=self.teacher_tokenizer.pad_token_id,
            divisor=self.make_seq_div_by_teacher,
        )

        # Per-doc alignment in the offset_cluster_decode_fix space, then
        # shift each doc's spans into pack-global indices and concatenate
        # into a single 1-row AlignmentBatch.
        alignment = self._align_packed_per_doc(
            s_input_ids=s_input_ids,
            t_input_ids=t_input_ids,
            s_pack=s_pack,
            t_pack=t_pack,
        )

        # Loss-side token mask: attention AND assistant content. The loss fn
        # already respects token_mask, so zeroing scaffold/user tokens here
        # restricts CE/KL to assistant supervision without touching the loss.
        student_token_mask = (s_attn * s_asst).long()
        teacher_token_mask = (t_attn * t_asst).long()

        # One packed row per call → batch dim is 1. sample_mask: 1.0 if any
        # doc fit, else 0.0 (downstream skips zero-mass samples).
        sample_mask = torch.tensor(
            [1.0 if len(s_pack["doc_starts"]) > 0 else 0.0],
            dtype=torch.float32,
        )
        idx = [batch[keep[0]]["idx"]] if keep else [-1]

        out = self._build_batched_dict(
            student_input_ids=s_input_ids,
            student_attention_mask=s_attn,
            student_token_mask=student_token_mask,
            teacher_input_ids=t_input_ids,
            teacher_attention_mask=t_attn,
            teacher_token_mask=teacher_token_mask,
            alignment=alignment,
            sample_mask=sample_mask,
            idx=idx,
        )
        # Doc ids surface for downstream sequence_packing (cu_seqlens) wiring.
        out["student_doc_id"] = s_doc_id
        out["teacher_doc_id"] = t_doc_id
        return out

    @staticmethod
    def _build_batched_dict(
        *,
        student_input_ids: torch.Tensor,
        student_attention_mask: torch.Tensor,
        student_token_mask: torch.Tensor,
        teacher_input_ids: torch.Tensor,
        teacher_attention_mask: torch.Tensor,
        teacher_token_mask: torch.Tensor,
        alignment: AlignmentBatch,
        sample_mask: torch.Tensor,
        idx: List[int],
    ) -> BatchedDataDict[Any]:
        return BatchedDataDict(
            # Student-side keys map onto Policy.train's expected names.
            input_ids=student_input_ids,
            input_lengths=student_attention_mask.sum(dim=-1).long(),
            token_mask=student_token_mask.long(),
            sample_mask=sample_mask,
            # Teacher-side keys travel with the batch for the teacher
            # forward pass in the trainer.
            teacher_input_ids=teacher_input_ids,
            teacher_input_lengths=teacher_attention_mask.sum(dim=-1).long(),
            teacher_token_mask=teacher_token_mask.long(),
            # Alignment payload, dense-padded so DTensor V2 can shard on dim 0.
            alignment_student_spans=alignment.student_spans,
            alignment_teacher_spans=alignment.teacher_spans,
            alignment_pair_valid=alignment.pair_valid,
            alignment_pair_is_correct=alignment.pair_is_correct,
            alignment_student_exact_partition_mask=(
                alignment.student_exact_partition_mask
            ),
            alignment_teacher_exact_partition_mask=(
                alignment.teacher_exact_partition_mask
            ),
            alignment_student_chunk_id=alignment.student_chunk_id,
            alignment_teacher_chunk_id=alignment.teacher_chunk_id,
            alignment_num_chunks=alignment.num_chunks,
            idx=idx,
        )

    def _align_packed_per_doc(
        self,
        *,
        s_input_ids: torch.Tensor,
        t_input_ids: torch.Tensor,
        s_pack: Dict[str, Any],
        t_pack: Dict[str, Any],
    ) -> AlignmentBatch:
        """Per-document alignment within a packed row, then concat into a
        single-row :class:`AlignmentBatch`.

        Calls :meth:`TokenAligner.align_one_offset` per doc, shifts each
        doc's per-doc span indices into pack-global indices, and runs
        :meth:`TokenAligner._pairs_to_batch` once over the combined list.
        """
        s_ids_list = s_input_ids[0].tolist()
        t_ids_list = t_input_ids[0].tolist()
        s_offsets_full = s_pack["offsets"]
        t_offsets_full = t_pack["offsets"]

        combined_pairs: List[Tuple[Any, ...]] = []
        for s_start, s_len, t_start, t_len in zip(
            s_pack["doc_starts"], s_pack["doc_lens"],
            t_pack["doc_starts"], t_pack["doc_lens"],
        ):
            s_slice_ids = s_ids_list[s_start : s_start + s_len]
            t_slice_ids = t_ids_list[t_start : t_start + t_len]
            s_slice_off = s_offsets_full[s_start : s_start + s_len]
            t_slice_off = t_offsets_full[t_start : t_start + t_len]
            pairs = self.aligner.align_one_offset(
                s_slice_ids, t_slice_ids, s_slice_off, t_slice_off,
            )
            # Shift each pair's spans by the doc's start offset in the
            # packed sequence (leave -1 sentinels for orphan sides alone).
            for s_toks, t_toks, s0, s1, t0, t1, ok in pairs:
                if s0 != -1:
                    s0 += s_start
                    s1 += s_start
                if t0 != -1:
                    t0 += t_start
                    t1 += t_start
                combined_pairs.append(
                    (s_toks, t_toks, s0, s1, t0, t1, ok)
                )

        # Pack the combined list into the dense AlignmentBatch contract
        # (B=1, T_s=packed_T, T_t=packed_T). _pairs_to_batch handles
        # max_pairs padding and per-token partition / chunk_id derivation.
        return TokenAligner._pairs_to_batch(
            [combined_pairs],
            b=1,
            t_s=s_input_ids.shape[1],
            t_t=t_input_ids.shape[1],
        )

    @staticmethod
    def _tokenize_batch(
        texts: List[str],
        tokenizer: PreTrainedTokenizerBase,
        ctx_length: int,
        make_seq_div_by: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a batch and pad to a multiple of ``make_seq_div_by``."""
        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=ctx_length,
            return_tensors="pt",
        )
        input_ids: torch.Tensor = encoded["input_ids"]
        attention_mask: torch.Tensor = encoded["attention_mask"]

        b, t = input_ids.shape
        pad = (make_seq_div_by - (t % make_seq_div_by)) % make_seq_div_by
        if pad > 0:
            pad_ids = torch.full(
                (b, pad),
                tokenizer.pad_token_id,
                dtype=input_ids.dtype,
            )
            pad_mask = torch.zeros((b, pad), dtype=attention_mask.dtype)
            input_ids = torch.cat([input_ids, pad_ids], dim=1)
            attention_mask = torch.cat([attention_mask, pad_mask], dim=1)

        return input_ids, attention_mask


# ---------------------------------------------------------------------------
# Chat-mode helpers (module-level so they're picklable across DataLoader workers)
# ---------------------------------------------------------------------------
def _render_chat_text(
    tokenizer: PreTrainedTokenizerBase,
    messages: List[Dict[str, str]],
) -> str | None:
    """Render the chat template to a single text string (no tokenization).

    Returns ``None`` when the template fails so the caller can skip the row.
    Uses ``enable_thinking=False`` when the template supports it.
    """
    try:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
    except Exception:
        return None


def _tokenize_text_with_assistant_mask(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    messages: List[Dict[str, str]],
    max_len: int,
) -> Dict[str, Any] | None:
    """Tokenize a pre-rendered text and build a per-token assistant mask.

    Used by both sides of the chat-mode collator. The student calls this
    with its own rendered text; the teacher calls it with the **student's**
    rendered text (Pavlo's design — see :meth:`CrossTokenizerCollator._call_chat`)
    so both sides' offsets live in the same character coordinate system.
    That's what makes ``offset_cluster_decode_fix`` produce meaningful
    cross-side pairs on chat data.

    Important:
      * ``add_special_tokens=False`` — the rendered text already contains
        the student's special tokens as literal strings (e.g.
        ``<|begin_of_text|>``); we don't want the tokenizer to prepend its
        OWN BOS/EOS on top. For the teacher side, the teacher's BPE will
        break the student's special-token strings into sub-tokens, which
        is the explicit cost of this design.
      * **Left-truncation** so the assistant response at the tail is
        preserved. Right-truncation (HF default) drops the tail; for long
        conversations that wipes the assistant content, leaving an
        all-zero assistant_mask which then NaNs the loss.
      * ``<think>...</think>`` sub-spans inside assistant content are
        excluded from the mask (non-thinking-mode supervision).

    Returns ``None`` when no assistant content survives (e.g. left-trunc
    cut just past the assistant span); the caller can skip the row.
    """
    prev_side = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = "left"
    try:
        enc = tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=True,
            max_length=max_len,
        )
    finally:
        tokenizer.truncation_side = prev_side

    ids = list(enc["input_ids"])
    offsets = list(enc["offset_mapping"])
    if not ids:
        return None

    # Build assistant-content mask via char-offset matching against the
    # rendered text. Tokens whose (cs, ce) falls inside an assistant
    # `content` span are marked 1; everything else (template scaffolding,
    # user content, padding) is 0. <think>...</think> sub-spans inside
    # assistant content also get 0.
    mask = [0] * len(ids)
    cursor = 0
    for m in messages:
        if m.get("role") != "assistant":
            continue
        content = m.get("content") or ""
        if not content:
            continue
        pos = text.find(content, cursor)
        if pos < 0:
            continue
        end_pos = pos + len(content)
        cursor = end_pos
        think_ranges = [
            (pos + mt.start(), pos + mt.end())
            for mt in _THINK_PATTERN.finditer(content)
        ]
        for i, (s, e) in enumerate(offsets):
            if s == 0 and e == 0:
                continue
            if s >= pos and e <= end_pos:
                in_think = any(ts <= s and e <= te for ts, te in think_ranges)
                if not in_think:
                    mask[i] = 1

    if not any(mask):
        return None

    return {
        "input_ids": ids,
        "offsets": [tuple(o) for o in offsets],
        "asst_mask": mask,
    }


def _render_student_and_tokenize(
    tokenizer: PreTrainedTokenizerBase,
    messages: List[Dict[str, str]],
    max_len: int,
) -> Tuple[str, Dict[str, Any]] | Tuple[None, None]:
    """Convenience wrapper: render student chat template + tokenize.

    Returns ``(rendered_text, tokenization_dict)`` so the caller can feed
    the same ``rendered_text`` into the teacher tokenizer for coordinate-
    system parity. Returns ``(None, None)`` when either step fails.
    """
    text = _render_chat_text(tokenizer, messages)
    if text is None:
        return None, None
    tok = _tokenize_text_with_assistant_mask(tokenizer, text, messages, max_len)
    if tok is None:
        return None, None
    return text, tok


def _pack_lockstep(
    student_docs: List[Dict[str, Any]],
    teacher_docs: List[Dict[str, Any]],
    *,
    student_pad_id: int,
    teacher_pad_id: int,
    student_eos_id: int | None,
    teacher_eos_id: int | None,
    student_max_len: int,
    teacher_max_len: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Pack docs into a single row per side, accepting a doc only if it fits
    BOTH student and teacher budgets.

    Without lockstep, greedy-per-side packing produces mismatched doc counts
    (Pavlo observed student=20, teacher=29 docs on a real mix batch) and the
    cross-tokenizer aligner can no longer match docs across sides.

    Returns ``(student_pack, teacher_pack)`` dicts with::

        input_ids, asst_mask, offsets, doc_id          : list of length=accumulated
        doc_starts, doc_lens                           : per-doc bookkeeping
    """
    add_eos_s = student_eos_id is not None
    add_eos_t = teacher_eos_id is not None

    def _empty() -> Dict[str, Any]:
        return {
            "input_ids": [], "asst_mask": [], "offsets": [], "doc_id": [],
            "doc_starts": [], "doc_lens": [],
        }
    s = _empty()
    t = _empty()

    for doc_idx, (sd, td) in enumerate(zip(student_docs, teacher_docs)):
        s_extra = 1 if add_eos_s else 0
        t_extra = 1 if add_eos_t else 0
        if (len(s["input_ids"]) + len(sd["input_ids"]) + s_extra > student_max_len
                or len(t["input_ids"]) + len(td["input_ids"]) + t_extra > teacher_max_len):
            continue

        # Append doc tokens + optional EOS separator (mask=0 on EOS so it
        # doesn't contribute loss; offsets=(0,0) so the offset aligner
        # treats the EOS as a special, not content).
        s_start = len(s["input_ids"])
        s["input_ids"].extend(sd["input_ids"])
        s["asst_mask"].extend(sd["asst_mask"])
        s["offsets"].extend(sd["offsets"])
        s["doc_id"].extend([doc_idx] * len(sd["input_ids"]))
        if add_eos_s:
            s["input_ids"].append(student_eos_id)
            s["asst_mask"].append(0)
            s["offsets"].append((0, 0))
            s["doc_id"].append(doc_idx)
        s["doc_starts"].append(s_start)
        s["doc_lens"].append(len(sd["input_ids"]) + s_extra)

        t_start = len(t["input_ids"])
        t["input_ids"].extend(td["input_ids"])
        t["asst_mask"].extend(td["asst_mask"])
        t["offsets"].extend(td["offsets"])
        t["doc_id"].extend([doc_idx] * len(td["input_ids"]))
        if add_eos_t:
            t["input_ids"].append(teacher_eos_id)
            t["asst_mask"].append(0)
            t["offsets"].append((0, 0))
            t["doc_id"].append(doc_idx)
        t["doc_starts"].append(t_start)
        t["doc_lens"].append(len(td["input_ids"]) + t_extra)

    # Pad each side to its max_len with pads (mask=0, offsets=(0,0), doc_id=-1).
    for pack, pad_id, max_len in [
        (s, student_pad_id, student_max_len),
        (t, teacher_pad_id, teacher_max_len),
    ]:
        while len(pack["input_ids"]) < max_len:
            pack["input_ids"].append(pad_id)
            pack["asst_mask"].append(0)
            pack["offsets"].append((0, 0))
            pack["doc_id"].append(-1)
        # Hard truncation in case a doc's len barely exceeded the budget.
        for k in ("input_ids", "asst_mask", "offsets", "doc_id"):
            pack[k] = pack[k][:max_len]

    return s, t


def _pad_packed(
    pack: Dict[str, Any],
    *,
    pad_id: int,
    divisor: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a packed dict to ``[1, T]`` tensors and round T up to
    ``divisor`` (TP*CP*2 for DTensor V2).

    The packed dict from :func:`_pack_lockstep` is already padded to its
    side's ``max_len``; this just realigns to the divisor when needed.

    Returns ``(input_ids, attention_mask, assistant_mask, doc_id)``.
    """
    input_ids = torch.tensor(pack["input_ids"], dtype=torch.long)[None, :]
    asst_mask = torch.tensor(pack["asst_mask"], dtype=torch.long)[None, :]
    doc_id = torch.tensor(pack["doc_id"], dtype=torch.long)[None, :]
    # Attention mask: 1 wherever doc_id >= 0 (real token), 0 on padding.
    attention_mask = (doc_id >= 0).long()

    t = input_ids.shape[1]
    pad = (divisor - (t % divisor)) % divisor
    if pad > 0:
        pad_ids = torch.full((1, pad), pad_id, dtype=input_ids.dtype)
        zeros = torch.zeros((1, pad), dtype=torch.long)
        neg_ones = torch.full((1, pad), -1, dtype=torch.long)
        input_ids = torch.cat([input_ids, pad_ids], dim=1)
        attention_mask = torch.cat([attention_mask, zeros], dim=1)
        asst_mask = torch.cat([asst_mask, zeros], dim=1)
        doc_id = torch.cat([doc_id, neg_ones], dim=1)

    return input_ids, attention_mask, asst_mask, doc_id
