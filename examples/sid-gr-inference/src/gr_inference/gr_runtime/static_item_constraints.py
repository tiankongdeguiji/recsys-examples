# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# STATIC index layout adapted from the reference implementation at
# https://github.com/youtube/static-constraint-decoding
# (Copyright 2026 Google LLC, Apache-2.0).

"""STATIC (CSR) tensorized item-constraint mask provider.

Drop-in replacement for :class:`TrieItemMaskProvider` mask computation. The
Python per-beam trie walk with per-token CUDA element writes is replaced by a
STATIC index (Sparse Transition-Accelerated Trie Index): a dense level-0 head
(``start_mask``; the level-0 state of token ``t`` is ``t + 1`` by construction)
plus a flat CSR transition table for deeper levels. ``step_mask`` becomes a
fixed ~15 tensor ops per call - beam-count and fan-out independent - instead
of one CUDA dispatch per allowed token per beam.

Unlike the reference (which masks a small codebook vocabulary), the LLM
vocabulary here is too large for the V x V dense second layer, so only the
level-0 head is dense and all deeper levels use the CSR tail. Catalogs must be
uniform-depth (standard fixed-level SIDs); the dict-trie backend remains the
fallback for mixed-depth catalogs.
"""

from __future__ import annotations

from typing import Any, Sequence

from gr_inference.gr_runtime.generation import GRGenerationState
from gr_inference.gr_runtime.item_constraints import TokenTrie, TrieItemMaskProvider


def build_static_index(
    token_paths: Sequence[Sequence[int]],
    *,
    vocab_size: int,
) -> dict[str, Any]:
    """Build flat STATIC transition arrays from uniform-depth token paths.

    State scheme (mirrors the reference builder): state 0 is the terminal
    sink (no outgoing edges), level-0 token ``t`` maps to state ``t + 1``,
    and deeper trie nodes are assigned ids from ``vocab_size + 1`` upward in
    sorted-path order, which makes the concatenated edge list globally sorted
    by ``(parent_state, token)``.

    Args:
        token_paths: Item token paths, all of the same length ``L >= 2``.
        vocab_size: Model token vocabulary size.

    Returns:
        Dict with numpy arrays ``edge_tokens``/``edge_states`` (int64 ``[E]``),
        ``edge_keys`` (int64 ``[E]``, ``parent * vocab_size + token``),
        ``indptr`` (int64 ``[num_states + 1]``) and ``start_mask``
        (bool ``[vocab_size]``).
    """
    import numpy as np

    if len(token_paths) == 0:
        raise ValueError("static mask backend requires a non-empty catalog")
    if len({len(path) for path in token_paths}) != 1:
        raise ValueError(
            "static mask backend requires uniform-depth token paths; "
            "use the python backend for mixed-depth catalogs"
        )
    paths = np.asarray(token_paths, dtype=np.int64)
    n, depth = paths.shape
    if depth < 2:
        raise ValueError("static mask backend requires token paths of length >= 2")
    if paths.min() < 0 or paths.max() >= vocab_size:
        raise ValueError("catalog token ids must be within the vocabulary")

    paths = np.unique(paths, axis=0)
    n = paths.shape[0]

    start_mask = np.zeros(vocab_size, dtype=bool)
    start_mask[paths[:, 0]] = True

    # a prefix is "new" at a depth when it first appears in the sorted paths
    diff = paths[1:] != paths[:-1]
    first_diff = np.full(n - 1, depth, dtype=np.int64)
    has_diff = diff.any(axis=1)
    first_diff[has_diff] = diff[has_diff].argmax(axis=1)
    is_new = np.zeros((n, depth), dtype=bool)
    is_new[0, :] = True
    for level in range(depth):
        is_new[1:, level] = first_diff <= level

    state_ids = np.zeros((n, depth - 1), dtype=np.int64)
    state_ids[:, 0] = paths[:, 0] + 1
    next_state_id = vocab_size + 1
    for level in range(1, depth - 1):
        new_rows = is_new[:, level]
        num_new = int(new_rows.sum())
        state_ids[new_rows, level] = np.arange(
            next_state_id, next_state_id + num_new, dtype=np.int64
        )
        state_ids[:, level] = np.maximum.accumulate(state_ids[:, level])
        next_state_id += num_new
    num_states = next_state_id

    all_parents, all_tokens, all_children = [], [], []
    for level in range(1, depth):
        new_rows = is_new[:, level]
        all_parents.append(state_ids[new_rows, level - 1])
        all_tokens.append(paths[new_rows, level])
        all_children.append(
            state_ids[new_rows, level]
            if level < depth - 1
            else np.zeros(int(new_rows.sum()), dtype=np.int64)
        )
    parents = np.concatenate(all_parents)
    edge_tokens = np.concatenate(all_tokens)
    edge_states = np.concatenate(all_children)

    edge_keys = parents * vocab_size + edge_tokens
    if not np.all(np.diff(edge_keys) > 0):
        raise AssertionError("STATIC edge list must be sorted by (parent, token)")

    counts = np.bincount(parents, minlength=num_states)
    indptr = np.zeros(num_states + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])

    return {
        "edge_tokens": edge_tokens,
        "edge_states": edge_states,
        "edge_keys": edge_keys,
        "indptr": indptr,
        "start_mask": start_mask,
    }


class StaticTrieItemMaskProvider(TrieItemMaskProvider):
    """TrieItemMaskProvider with STATIC (CSR) tensorized mask computation.

    Inherits completion / item-id resolution (response-time, host-side) from
    the dict-trie base class and overrides the hot-path ``initial_mask`` /
    ``step_mask`` with vectorized transitions over the STATIC index. Mask
    semantics are bit-identical to the base class for uniform-depth catalogs.
    """

    def __init__(
        self,
        trie: TokenTrie,
        *,
        vocab_size: int,
        eos_token_id: int | None = None,
        allow_eos_for_terminal: bool = True,
        token_paths: Sequence[Sequence[int]],
    ) -> None:
        super().__init__(
            trie,
            vocab_size=vocab_size,
            eos_token_id=eos_token_id,
            allow_eos_for_terminal=allow_eos_for_terminal,
        )
        self._index = build_static_index(token_paths, vocab_size=vocab_size)
        self._device_cache: dict[Any, dict[str, Any]] = {}

    def _device_index(self, device: Any) -> dict[str, Any]:
        import torch

        key = str(device)
        cached = self._device_cache.get(key)
        if cached is None:
            cached = {
                name: torch.from_numpy(array).to(device)
                for name, array in self._index.items()
            }
            self._device_cache[key] = cached
        return cached

    def initial_mask(self, logits: Any):
        device = getattr(logits, "device", None)
        return self._device_index(device)["start_mask"].clone()

    def step_mask(self, generation: GRGenerationState, logits: Any):
        import torch

        if generation.beam_path.steps_done == 0:
            raise ValueError("beam_path must be initialized before step masks")
        if not hasattr(logits, "shape") or len(logits.shape) != 3:
            raise ValueError("decode logits must be shaped [B, W, vocab]")
        if logits.shape[0] != 1:
            raise ValueError(
                "StaticTrieItemMaskProvider currently supports batch_size=1"
            )

        width = logits.shape[1]
        device = getattr(logits, "device", None)
        index = self._device_index(device)
        vocab = self.vocab_size
        eos = self.eos_token_id

        traces = [
            generation.beam_path.token_trace(beam) for beam in range(width)
        ]
        eos_rows = [
            beam for beam, trace in enumerate(traces) if eos is not None and eos in trace
        ]
        trace_t = torch.tensor(traces, dtype=torch.int64).to(device)

        # level-0 dense head: state of token t is t + 1
        tok0 = trace_t[:, 0]
        in_vocab = (tok0 >= 0) & (tok0 < vocab)
        alive = index["start_mask"][tok0.clamp(0, vocab - 1)] & in_vocab
        states = torch.where(alive, tok0 + 1, torch.zeros_like(tok0))

        # CSR tail: one searchsorted per remaining trace level
        edge_keys = index["edge_keys"]
        for level in range(1, trace_t.shape[1]):
            query = states * vocab + trace_t[:, level]
            pos = torch.searchsorted(edge_keys, query).clamp(max=edge_keys.shape[0] - 1)
            found = alive & (edge_keys[pos] == query)
            states = torch.where(
                found, index["edge_states"][pos], torch.zeros_like(states)
            )
            alive = found

        starts = index["indptr"][states]
        lens = index["indptr"][states + 1] - starts
        lens = torch.where(alive, lens, torch.zeros_like(lens))
        if eos_rows:
            eos_rows_t = torch.tensor(eos_rows, dtype=torch.int64, device=device)
            lens[eos_rows_t] = 0

        beam_ids = torch.repeat_interleave(
            torch.arange(width, dtype=torch.int64, device=device), lens
        )
        flat_pos = (
            torch.arange(beam_ids.shape[0], dtype=torch.int64, device=device)
            - (torch.cumsum(lens, 0) - lens)[beam_ids]
            + starts[beam_ids]
        )
        mask = torch.zeros((width, vocab), dtype=torch.bool, device=device)
        if beam_ids.shape[0]:
            mask[beam_ids, index["edge_tokens"][flat_pos]] = True

        if eos is not None:
            if self.allow_eos_for_terminal:
                mask[alive & (states == 0), eos] = True
            if eos_rows:
                mask[eos_rows_t, eos] = True
        return mask
