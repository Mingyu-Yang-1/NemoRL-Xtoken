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
"""Offset-based cross-tokenizer alignment (cluster + strict char-end walker).

Ported from ``tokenalign_upstream/src/offset_alignment.py`` — the cluster
method scored 100% correct on every stress category in upstream's
BENCHMARK_RESULTS.md.

Public entry point:
    align_by_offsets_cluster(
        student_ids, student_offsets, student_tokenizer,
        teacher_ids, teacher_offsets, teacher_tokenizer,
    ) -> list[(s_toks, t_toks, s_start, s_end, t_start, t_end, is_correct)]

Output shape matches :class:`TokenAligner`'s 7-tuple. Orphan groups (coverage
divergence) carry ``is_correct=False`` with ``-1`` sentinels on the empty side.

Only the strict-cluster variant is ported. Other upstream methods
(``overlap``, ``canonical``, ``hybrid_offset_dp``) are intentionally omitted
to keep the integration surface minimal — they can be added if/when a
specific tokenizer pair fails on strict cluster.
"""
from __future__ import annotations

from typing import Any, List, Tuple


# ---------------------------------------------------------------------------
# Special-token role pairing
# ---------------------------------------------------------------------------
def _role_of(tok, token_id: int) -> str:
    if token_id == getattr(tok, "bos_token_id", None):
        return "bos"
    if token_id == getattr(tok, "eos_token_id", None):
        return "eos"
    if token_id == getattr(tok, "pad_token_id", None):
        return "pad"
    if token_id == getattr(tok, "unk_token_id", None):
        return "unk"
    if token_id == getattr(tok, "sep_token_id", None):
        return "sep"
    if token_id == getattr(tok, "cls_token_id", None):
        return "cls"
    if token_id == getattr(tok, "mask_token_id", None):
        return "mask"
    special_ids = getattr(tok, "all_special_ids", []) or []
    if token_id in special_ids:
        try:
            return f"special:{tok.convert_ids_to_tokens(int(token_id))}"
        except Exception:
            return f"special:id={int(token_id)}"
    return "content"


def _partition(
    input_ids: List[int], offsets: List[Tuple[int, int]]
) -> Tuple[List[int], List[Tuple[int, int, int]], List[int]]:
    """Split a tokenized sequence into leading-specials / content / trailing-specials
    by ``(0, 0)`` offset. Mid-stream specials are folded into trailing.
    """
    n = len(input_ids)
    is_content = [offsets[i][1] > offsets[i][0] for i in range(n)]
    first = next((i for i in range(n) if is_content[i]), None)
    last = next((i for i in range(n - 1, -1, -1) if is_content[i]), None)
    if first is None:
        return list(range(n)), [], []
    leading = [i for i in range(first) if not is_content[i]]
    trailing = [i for i in range(last + 1, n) if not is_content[i]]
    content = [
        (int(offsets[i][0]), int(offsets[i][1]), i)
        for i in range(first, last + 1)
        if is_content[i]
    ]
    mid_specials = [i for i in range(first, last + 1) if not is_content[i]]
    trailing = sorted(set(trailing + mid_specials))
    return leading, content, trailing


# When one tokenizer has pad_token_id == eos_token_id (e.g. Llama-3.2 with our
# fallback that sets pad_token = eos_token if undefined), trailing pad
# positions get role "eos"; the other tokenizer with a separate pad_token_id
# tags its trailing pads as "pad". DP pairs these positions 1↔1 via post-
# processing, giving the student KL signal at pad positions. Without this
# set, role-based pairing silently drops the mismatched roles and the student
# loses ~2.4% of training signal — observable as a consistent GSM8K/MATH
# regression in offset_cluster runs vs DP runs on Llama+Phi.
_PAD_EQUIVALENT_ROLES = {"pad", "eos"}


def _pair_specials_by_role(
    s_positions: List[int],
    s_ids: List[int],
    s_tok,
    t_positions: List[int],
    t_ids: List[int],
    t_tok,
) -> List[Tuple[List[int], List[int]]]:
    groups: List[Tuple[List[int], List[int]]] = []
    si = ti = 0
    while si < len(s_positions) and ti < len(t_positions):
        s_role = _role_of(s_tok, s_ids[s_positions[si]])
        t_role = _role_of(t_tok, t_ids[t_positions[ti]])
        if s_role == t_role or (
            s_role in _PAD_EQUIVALENT_ROLES and t_role in _PAD_EQUIVALENT_ROLES
        ):
            groups.append(([s_positions[si]], [t_positions[ti]]))
            si += 1
            ti += 1
        elif s_role in _PAD_EQUIVALENT_ROLES:
            si += 1
        elif t_role in _PAD_EQUIVALENT_ROLES:
            ti += 1
        else:
            groups.append(([s_positions[si]], []))
            groups.append(([], [t_positions[ti]]))
            si += 1
            ti += 1
    for i in range(si, len(s_positions)):
        if _role_of(s_tok, s_ids[s_positions[i]]) in _PAD_EQUIVALENT_ROLES:
            continue
        groups.append(([s_positions[i]], []))
    for j in range(ti, len(t_positions)):
        if _role_of(t_tok, t_ids[t_positions[j]]) in _PAD_EQUIVALENT_ROLES:
            continue
        groups.append(([], [t_positions[j]]))
    return groups


# ---------------------------------------------------------------------------
# Content alignment: same-span cluster + strict char-end walker
# ---------------------------------------------------------------------------
def _cluster_same_span(
    content: List[Tuple[int, int, int]],
) -> List[Tuple[int, int, List[int]]]:
    """Collapse consecutive tokens sharing the *exact* same ``(cs, ce)`` into a
    single cluster. Different-span tokens (even overlapping) stay separate.
    """
    if not content:
        return []
    clusters: List[Tuple[int, int, List[int]]] = []
    i = 0
    while i < len(content):
        cs, ce, pos = content[i]
        positions = [pos]
        j = i + 1
        while j < len(content) and content[j][0] == cs and content[j][1] == ce:
            positions.append(content[j][2])
            j += 1
        clusters.append((cs, ce, positions))
        i = j
    return clusters


def _content_align_offset_cluster(
    s_content: List[Tuple[int, int, int]],
    t_content: List[Tuple[int, int, int]],
) -> List[Tuple[List[int], List[int]]]:
    """Pre-merge same-span clusters, then strict char-end walker with orphan
    emission. Every paired group satisfies::

        min(cs over G_s) == min(cs over G_t)
        max(ce over G_s) == max(ce over G_t)
    """
    s_clusters = _cluster_same_span(s_content)
    t_clusters = _cluster_same_span(t_content)
    n_s, n_t = len(s_clusters), len(t_clusters)

    groups: List[Tuple[List[int], List[int]]] = []
    si = ti = 0

    while si < n_s and ti < n_t:
        while (
            si < n_s
            and ti < n_t
            and s_clusters[si][0] != t_clusters[ti][0]
        ):
            if s_clusters[si][0] < t_clusters[ti][0]:
                groups.append((list(s_clusters[si][2]), []))
                si += 1
            else:
                groups.append(([], list(t_clusters[ti][2])))
                ti += 1
        if si >= n_s or ti >= n_t:
            break

        s_group_start = si
        t_group_start = ti
        s_end = s_clusters[si][1]
        t_end = t_clusters[ti][1]
        exhausted = False
        while s_end != t_end:
            if s_end < t_end:
                si += 1
                if si >= n_s:
                    exhausted = True
                    break
                s_end = s_clusters[si][1]
            else:
                ti += 1
                if ti >= n_t:
                    exhausted = True
                    break
                t_end = t_clusters[ti][1]

        if not exhausted:
            s_pos = [
                p for c in s_clusters[s_group_start : si + 1] for p in c[2]
            ]
            t_pos = [
                p for c in t_clusters[t_group_start : ti + 1] for p in c[2]
            ]
            groups.append((s_pos, t_pos))
            si += 1
            ti += 1
        else:
            for c in s_clusters[s_group_start : min(si + 1, n_s)]:
                groups.append((list(c[2]), []))
            for c in t_clusters[t_group_start : min(ti + 1, n_t)]:
                groups.append(([], list(c[2])))
            si = n_s
            ti = n_t
            break

    while si < n_s:
        groups.append((list(s_clusters[si][2]), []))
        si += 1
    while ti < n_t:
        groups.append(([], list(t_clusters[ti][2])))
        ti += 1
    return groups


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def align_by_offsets_cluster(
    student_ids: List[int],
    student_offsets: List[Tuple[int, int]],
    student_tokenizer,
    teacher_ids: List[int],
    teacher_offsets: List[Tuple[int, int]],
    teacher_tokenizer,
) -> List[Tuple[List[str], List[str], int, int, int, int, bool]]:
    """Cluster+strict offset alignment for a single sample.

    Args:
        student_ids: list of student token ids (len = ctx_length, including pad)
        student_offsets: list of ``(cs, ce)`` tuples per token (must come from
            ``return_offsets_mapping=True`` on a fast HF tokenizer)
        student_tokenizer: HF tokenizer (used for special-token role lookup)
        teacher_ids/offsets/tokenizer: same for teacher

    Returns:
        list of 7-tuples ``(s_tok_strs, t_tok_strs, s_start, s_end, t_start,
        t_end, is_correct)`` — paired groups have ``is_correct=True`` and
        contiguous position ranges on both sides; orphan groups have
        ``is_correct=False`` and the empty side carries ``start=end=-1``.
    """
    s_off_tuples = [tuple(o) for o in student_offsets]
    t_off_tuples = [tuple(o) for o in teacher_offsets]

    s_lead, s_cont, s_trail = _partition(student_ids, s_off_tuples)
    t_lead, t_cont, t_trail = _partition(teacher_ids, t_off_tuples)

    groups: List[Tuple[List[int], List[int]]] = []
    groups += _pair_specials_by_role(
        s_lead, student_ids, student_tokenizer,
        t_lead, teacher_ids, teacher_tokenizer,
    )
    groups += _content_align_offset_cluster(s_cont, t_cont)
    groups += _pair_specials_by_role(
        s_trail, student_ids, student_tokenizer,
        t_trail, teacher_ids, teacher_tokenizer,
    )

    student_tokens_str = student_tokenizer.convert_ids_to_tokens(student_ids)
    teacher_tokens_str = teacher_tokenizer.convert_ids_to_tokens(teacher_ids)

    aligned_pairs: List[Tuple[Any, ...]] = []
    for s_pos, t_pos in groups:
        s_toks = [student_tokens_str[i] for i in s_pos]
        t_toks = [teacher_tokens_str[i] for i in t_pos]
        s_start = s_pos[0] if s_pos else -1
        s_end = s_pos[-1] + 1 if s_pos else -1
        t_start = t_pos[0] if t_pos else -1
        t_end = t_pos[-1] + 1 if t_pos else -1
        is_correct = bool(s_pos and t_pos)
        aligned_pairs.append(
            (s_toks, t_toks, s_start, s_end, t_start, t_end, is_correct)
        )

    return aligned_pairs
