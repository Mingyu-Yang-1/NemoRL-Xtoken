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
        num_packed_rows: In ``chat`` mode, the number of packed rows to
            emit per ``__call__`` (output batch dim B = ``num_packed_rows``).
            Must be ≥ the data-parallel world size so the trainer's
            ``shard_by_batch_size(dp_size)`` divides cleanly. Default 1.
            The collator chunks the input batch's candidate conversations
            into ``num_packed_rows`` groups (chunked, not round-robin)
            and lockstep-packs each group into its own row independently.
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
        num_packed_rows: int = 1,
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
        if num_packed_rows < 1:
            raise ValueError(
                f"num_packed_rows must be ≥ 1, got {num_packed_rows}"
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
        self.num_packed_rows = num_packed_rows
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
        # Native-template chat mode: each side renders + tokenizes with its
        # OWN chat template. The teacher gets in-distribution input (vs the
        # earlier student-rendered hack which fed it Llama-format scaffold
        # that its BPE breaks into sub-tokens). Per-assistant-message
        # offset rebasing in the aligner (align_one_offset_per_asst) lets
        # offset_cluster still work: we slice asst-content tokens per asst
        # message and rebase their offsets to start at 0 on each side,
        # where they now share a coordinate system.
        messages_list = [datum[self.messages_key] for datum in batch]

        per_doc_student: List[Dict[str, Any]] = []
        per_doc_teacher: List[Dict[str, Any]] = []
        per_doc_idx: List[int] = []
        for i, m in enumerate(messages_list):
            s_tok = _render_and_tokenize(
                self.student_tokenizer, m, self.ctx_length_student,
            )
            if s_tok is None:
                continue
            t_tok = _render_and_tokenize(
                self.teacher_tokenizer, m, self.ctx_length_teacher,
            )
            if t_tok is None:
                continue
            # Both sides must enumerate the same number of asst turns for
            # per-msg alignment. Chat templates can rarely strip turns
            # differently — skip the row if so.
            if len(s_tok["asst_char_spans"]) != len(t_tok["asst_char_spans"]):
                continue
            per_doc_student.append(s_tok)
            per_doc_teacher.append(t_tok)
            per_doc_idx.append(batch[i]["idx"])

        # Split surviving candidates into num_packed_rows chunks (contiguous;
        # first row gets the first ceil(K/N) docs, etc.). Within each chunk,
        # lockstep packing accepts whichever docs fit BOTH budgets.
        #
        # Pass 1: try chunked assignment.
        # Pass 2 (greedy-refill): for any row that came out empty, duplicate
        # a doc from a non-empty row into it. This avoids the empty-row
        # backward bug: when a rank's token_mask is all zeros, its loss is
        # a graph-detached zero and FSDP backward fails with "element 0 of
        # tensors does not require grad and does not have a grad_fn".
        # Refill makes that case deterministic-non-empty as long as ANY
        # candidate fit; the degenerate "all 64 candidates failed" case is
        # rare in practice (cascade-2 at ctx≥2048 packs >99% of rows) and
        # left as a documented residual edge case.
        N = self.num_packed_rows
        K = len(per_doc_student)
        if K == 0:
            row_chunks: List[List[int]] = [[] for _ in range(N)]
        else:
            chunk = max(1, (K + N - 1) // N)
            row_chunks = []
            for r in range(N):
                start = r * chunk
                end = min(start + chunk, K)
                if start >= K:
                    row_chunks.append([])
                else:
                    row_chunks.append(list(range(start, end)))

        pad_s = self.student_tokenizer.pad_token_id
        pad_t = self.teacher_tokenizer.pad_token_id
        eos_s = (
            self.student_tokenizer.eos_token_id
            if self.add_eos_between_docs else None
        )
        eos_t = (
            self.teacher_tokenizer.eos_token_id
            if self.add_eos_between_docs else None
        )

        def _pack_inds(inds: List[int]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            return _pack_lockstep(
                [per_doc_student[i] for i in inds],
                [per_doc_teacher[i] for i in inds],
                student_pad_id=pad_s, teacher_pad_id=pad_t,
                student_eos_id=eos_s, teacher_eos_id=eos_t,
                student_max_len=self.ctx_length_student,
                teacher_max_len=self.ctx_length_teacher,
            )

        # Pass 1: pack each row from its chunk.
        # row_packs[r] = (s_pack, t_pack, source_inds_used)
        row_packs: List[Tuple[Dict[str, Any], Dict[str, Any], List[int]]] = []
        for inds in row_chunks:
            s_pack, t_pack = _pack_inds(inds)
            row_packs.append((s_pack, t_pack, inds))

        # Pass 2: greedy-refill empty rows. Collect indices of docs that
        # actually fit (i.e., were accepted by lockstep in some row); these
        # are guaranteed to fit when packed alone too. Refill in round-
        # robin to spread duplication.
        fitted_doc_indices: List[int] = []
        for sp, _tp, inds in row_packs:
            for local in sp["accepted_indices"]:
                fitted_doc_indices.append(inds[local])

        if fitted_doc_indices:
            refill_cursor = 0
            for r in range(N):
                sp, _tp, _inds = row_packs[r]
                if sp["doc_starts"]:
                    continue
                src = fitted_doc_indices[refill_cursor % len(fitted_doc_indices)]
                refill_cursor += 1
                s_pack_r, t_pack_r = _pack_inds([src])
                row_packs[r] = (s_pack_r, t_pack_r, [src])
        # else: K == 0 or no candidate fits → all rows stay empty, loss
        # will be a graph-detached 0 and backward will fail. This is a
        # documented residual edge case; in practice it requires either
        # an empty batch or every candidate exceeding both ctx budgets.

        # Collate the (possibly refilled) packs into [N, T] tensors + the
        # per-row alignment pair lists.
        s_input_ids_rows: List[torch.Tensor] = []
        s_attn_rows: List[torch.Tensor] = []
        s_asst_rows: List[torch.Tensor] = []
        s_doc_id_rows: List[torch.Tensor] = []
        t_input_ids_rows: List[torch.Tensor] = []
        t_attn_rows: List[torch.Tensor] = []
        t_asst_rows: List[torch.Tensor] = []
        t_doc_id_rows: List[torch.Tensor] = []
        per_row_pairs: List[List[Tuple[Any, ...]]] = []
        row_sample_mask: List[float] = []
        row_idx: List[int] = []

        for s_pack, t_pack, inds in row_packs:
            s_ids_t, s_attn_t, s_asst_t, s_doc_id_t = _pad_packed(
                s_pack, pad_id=pad_s, divisor=self.make_seq_div_by_student,
            )
            t_ids_t, t_attn_t, t_asst_t, t_doc_id_t = _pad_packed(
                t_pack, pad_id=pad_t, divisor=self.make_seq_div_by_teacher,
            )
            s_input_ids_rows.append(s_ids_t)
            s_attn_rows.append(s_attn_t)
            s_asst_rows.append(s_asst_t)
            s_doc_id_rows.append(s_doc_id_t)
            t_input_ids_rows.append(t_ids_t)
            t_attn_rows.append(t_attn_t)
            t_asst_rows.append(t_asst_t)
            t_doc_id_rows.append(t_doc_id_t)
            per_row_pairs.append(
                self._row_pair_list(s_ids_t, t_ids_t, s_pack, t_pack)
            )
            row_sample_mask.append(
                1.0 if len(s_pack["doc_starts"]) > 0 else 0.0
            )
            row_idx.append(per_doc_idx[inds[0]] if inds else -1)

        # Stack along dim 0 → [N, T] tensors.
        s_input_ids = torch.cat(s_input_ids_rows, dim=0)
        s_attn = torch.cat(s_attn_rows, dim=0)
        s_asst = torch.cat(s_asst_rows, dim=0)
        s_doc_id = torch.cat(s_doc_id_rows, dim=0)
        t_input_ids = torch.cat(t_input_ids_rows, dim=0)
        t_attn = torch.cat(t_attn_rows, dim=0)
        t_asst = torch.cat(t_asst_rows, dim=0)
        t_doc_id = torch.cat(t_doc_id_rows, dim=0)

        # One combined AlignmentBatch covering all N rows.
        alignment = TokenAligner._pairs_to_batch(
            per_row_pairs,
            b=N,
            t_s=s_input_ids.shape[1],
            t_t=t_input_ids.shape[1],
        )

        # Loss-side token mask: attention AND assistant content.
        student_token_mask = (s_attn * s_asst).long()
        teacher_token_mask = (t_attn * t_asst).long()

        sample_mask = torch.tensor(row_sample_mask, dtype=torch.float32)

        out = self._build_batched_dict(
            student_input_ids=s_input_ids,
            student_attention_mask=s_attn,
            student_token_mask=student_token_mask,
            teacher_input_ids=t_input_ids,
            teacher_attention_mask=t_attn,
            teacher_token_mask=teacher_token_mask,
            alignment=alignment,
            sample_mask=sample_mask,
            idx=row_idx,
        )
        # Per-token doc id tensors are emitted but currently CONSUMED BY NO
        # ONE in the training path. They're reserved for a future fix to A2
        # (cross-doc attention contamination within a packed row): tokens
        # from Doc-A and Doc-B in the same row see each other's hidden states.
        # NeMo-RL's policy.sequence_packing.enabled=true does NOT isolate
        # within a row — it only walls between batch entries via cu_seqlens.
        # Within-row isolation needs one of:
        #   (a) build a [T, T] doc-id-equality attention mask from these
        #       tensors in the policy worker (FlashAttention backend
        #       compatibility caveat),
        #   (b) build per-row cu_seqlens from doc_starts/doc_lens and inject
        #       into flash_attn_kwargs (bypasses framework auto-construction),
        #   (c) set num_packed_rows high enough that each row averages 1 doc.
        # In practice the contamination is likely benign: both student and
        # teacher see the same contaminated context, so the KL signal is
        # consistent; loss only fires on assistant content. Deferred to a
        # follow-up if metrics show measurable regression.
        out["student_doc_id"] = s_doc_id
        out["teacher_doc_id"] = t_doc_id
        return out

    def _row_pair_list(
        self,
        s_input_ids: torch.Tensor,
        t_input_ids: torch.Tensor,
        s_pack: Dict[str, Any],
        t_pack: Dict[str, Any],
    ) -> List[Tuple[Any, ...]]:
        """Per-doc alignment for ONE packed row.

        Returns the flat 7-tuple pair list with spans already shifted to
        pack-global indices. Caller stacks the per-row lists into a single
        ``per_sample_pairs`` of length ``B`` and runs ``_pairs_to_batch``.
        """
        s_ids_list = s_input_ids[0].tolist()
        t_ids_list = t_input_ids[0].tolist()
        s_offsets_full = s_pack["offsets"]
        t_offsets_full = t_pack["offsets"]

        combined_pairs: List[Tuple[Any, ...]] = []
        s_doc_spans = s_pack.get("doc_asst_char_spans", [])
        t_doc_spans = t_pack.get("doc_asst_char_spans", [])
        for di, (s_start, s_len, t_start, t_len) in enumerate(zip(
            s_pack["doc_starts"], s_pack["doc_lens"],
            t_pack["doc_starts"], t_pack["doc_lens"],
        )):
            s_slice_ids = s_ids_list[s_start : s_start + s_len]
            t_slice_ids = t_ids_list[t_start : t_start + t_len]
            s_slice_off = s_offsets_full[s_start : s_start + s_len]
            t_slice_off = t_offsets_full[t_start : t_start + t_len]
            # Pass per-doc asst_mask so the aligner excludes
            # <think>...</think> sub-tokens from partition_mask.
            s_slice_asst = s_pack["asst_mask"][s_start : s_start + s_len]
            t_slice_asst = t_pack["asst_mask"][t_start : t_start + t_len]
            # Per-asst alignment uses each side's native chat-template
            # offsets, slicing tokens per asst message and rebasing
            # offsets into a shared per-message coordinate system.
            pairs = self.aligner.align_one_offset_per_asst(
                s_slice_ids, s_slice_off, s_doc_spans[di],
                t_slice_ids, t_slice_off, t_doc_spans[di],
                student_asst_mask=s_slice_asst,
                teacher_asst_mask=t_slice_asst,
            )
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
        return combined_pairs

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

    In native-template chat mode each side renders its OWN chat template and
    calls this with that text. The returned ``asst_char_spans`` list lets
    downstream alignment slice tokens per asst message and rebase offsets to
    a common (0 → len(content)) coordinate system per slice — so
    offset_cluster works on each side's native scaffold despite the
    full-sequence offsets being in different coordinate systems.

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
    # assistant content also get 0. asst_char_spans records the
    # (start_char, end_char) of each assistant message's content within
    # the rendered text; the cross-tokenizer aligner uses this to slice
    # asst-content tokens and rebase their offsets per message.
    mask = [0] * len(ids)
    cursor = 0
    asst_char_spans: List[Tuple[int, int]] = []
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
        asst_char_spans.append((pos, end_pos))
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
        "asst_char_spans": asst_char_spans,
    }


def _render_and_tokenize(
    tokenizer: PreTrainedTokenizerBase,
    messages: List[Dict[str, str]],
    max_len: int,
) -> Dict[str, Any] | None:
    """Apply tokenizer's OWN chat template, tokenize, and build the asst
    mask + asst char spans.

    Each side calls this independently with its own tokenizer in the
    native-template chat mode — the teacher gets in-distribution input
    and per-asst-message offset rebasing keeps the cross-tokenizer
    aligner working. Returns ``None`` when render or asst-content
    extraction fails so the caller can skip the row.
    """
    text = _render_chat_text(tokenizer, messages)
    if text is None:
        return None
    return _tokenize_text_with_assistant_mask(tokenizer, text, messages, max_len)


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
        accepted_indices                               : input indices that fit
    """
    add_eos_s = student_eos_id is not None
    add_eos_t = teacher_eos_id is not None

    def _empty() -> Dict[str, Any]:
        return {
            "input_ids": [], "asst_mask": [], "offsets": [], "doc_id": [],
            "doc_starts": [], "doc_lens": [], "accepted_indices": [],
            # Per-doc list of (start_char, end_char) ranges marking each
            # assistant message's content WITHIN THAT DOC'S RENDERED TEXT.
            # Char positions are in the original (pre-pack) coordinate
            # system of the doc, NOT the packed sequence's text space —
            # they're consumed by the aligner together with the doc's own
            # offsets/ids slice.
            "doc_asst_char_spans": [],
        }
    s = _empty()
    t = _empty()

    for doc_idx, (sd, td) in enumerate(zip(student_docs, teacher_docs)):
        s_extra = 1 if add_eos_s else 0
        t_extra = 1 if add_eos_t else 0
        if (len(s["input_ids"]) + len(sd["input_ids"]) + s_extra > student_max_len
                or len(t["input_ids"]) + len(td["input_ids"]) + t_extra > teacher_max_len):
            continue
        s["accepted_indices"].append(doc_idx)
        t["accepted_indices"].append(doc_idx)

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
        s["doc_asst_char_spans"].append(sd.get("asst_char_spans", []))

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
        t["doc_asst_char_spans"].append(td.get("asst_char_spans", []))

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
