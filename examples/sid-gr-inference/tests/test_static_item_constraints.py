# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Equivalence tests: STATIC CSR mask provider vs the python trie walk."""

import random
from types import SimpleNamespace

import pytest
import torch
from gr_inference.gr_kv.beam_path import BeamPath
from gr_inference.gr_runtime.item_constraints import SemanticItemCatalog, SemanticItem
from gr_inference.gr_runtime.static_item_constraints import (
    StaticTrieItemMaskProvider,
    build_static_index,
)

VOCAB = 512
EOS = 500


def _random_catalog(rng: random.Random, items: int, depth: int) -> SemanticItemCatalog:
    paths = set()
    while len(paths) < items:
        paths.add(tuple(rng.randrange(0, 480) for _ in range(depth)))
    return SemanticItemCatalog(
        items=tuple(
            SemanticItem(item_id=f"i{i}", token_ids=path)
            for i, path in enumerate(sorted(paths))
        )
    )


def _providers(catalog: SemanticItemCatalog, **kwargs):
    python = catalog.provider(vocab_size=VOCAB, mask_backend="python", **kwargs)
    static = catalog.provider(vocab_size=VOCAB, mask_backend="static", **kwargs)
    assert isinstance(static, StaticTrieItemMaskProvider)
    return python, static


def _beam_state(traces: list[tuple[int, ...]]) -> SimpleNamespace:
    width = len(traces)
    steps = len(traces[0])
    path = BeamPath(max_decode_steps=steps + 1, max_beam_width=width)
    prev_width = 1
    for step in range(steps):
        parents = [min(beam, prev_width - 1) for beam in range(width)]
        path.append(parents, [trace[step] for trace in traces], [0.0] * width)
        prev_width = width
    return SimpleNamespace(beam_path=path)


@pytest.mark.parametrize("depth", [2, 3, 4])
@pytest.mark.parametrize("eos_token_id", [None, EOS])
def test_static_step_masks_match_python(depth: int, eos_token_id: int | None) -> None:
    rng = random.Random(depth * 1000 + (eos_token_id or 0))
    catalog = _random_catalog(rng, items=400, depth=depth)
    python, static = _providers(catalog, eos_token_id=eos_token_id)

    all_paths = [item.token_ids for item in catalog.items]
    width = 32
    logits = torch.zeros((1, width, VOCAB))

    for steps in range(1, depth):
        traces = [rng.choice(all_paths)[:steps] for _ in range(width)]
        traces[0] = tuple([499] * steps)  # off-trie beam -> all-False row
        if eos_token_id is not None and steps >= 2:
            traces[1] = traces[1][:-1] + (eos_token_id,)  # eos row -> stop tokens only
        state = _beam_state(traces)
        expected = python.step_mask(state, logits)
        actual = static.step_mask(state, logits)
        assert torch.equal(expected, actual), f"depth={depth} steps={steps}"

    # full-length traces: terminal rows must allow exactly eos (when configured)
    traces = [rng.choice(all_paths) for _ in range(width)]
    state = _beam_state(traces)
    expected = python.step_mask(state, logits)
    actual = static.step_mask(state, logits)
    assert torch.equal(expected, actual)


def test_static_initial_mask_matches_python() -> None:
    rng = random.Random(7)
    catalog = _random_catalog(rng, items=300, depth=3)
    python, static = _providers(catalog, eos_token_id=EOS)
    logits = torch.zeros((1, 8, VOCAB))
    assert torch.equal(python.initial_mask(logits), static.initial_mask(logits))


def test_static_allow_eos_for_terminal_false() -> None:
    rng = random.Random(11)
    catalog = _random_catalog(rng, items=200, depth=3)
    python, static = _providers(
        catalog, eos_token_id=EOS, allow_eos_for_terminal=False
    )
    all_paths = [item.token_ids for item in catalog.items]
    traces = [rng.choice(all_paths) for _ in range(16)]
    state = _beam_state(traces)
    logits = torch.zeros((1, 16, VOCAB))
    assert torch.equal(python.step_mask(state, logits), static.step_mask(state, logits))


def test_static_rejects_mixed_depth() -> None:
    catalog = SemanticItemCatalog(
        items=(
            SemanticItem(item_id="a", token_ids=(1, 2)),
            SemanticItem(item_id="b", token_ids=(1, 2, 3)),
        )
    )
    with pytest.raises(ValueError, match="uniform-depth"):
        catalog.provider(vocab_size=VOCAB, mask_backend="static")


def test_build_static_index_shapes() -> None:
    index = build_static_index([(1, 10, 20), (1, 11, 21), (2, 10, 22)], vocab_size=64)
    assert bool(index["start_mask"][1]) and bool(index["start_mask"][2])
    assert index["start_mask"].sum() == 2
    # edges: 3 level-1 + 3 level-2 transitions
    assert index["edge_tokens"].shape[0] == 6
    assert index["indptr"][0] == 0 and index["indptr"][1] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_static_step_masks_match_python_cuda() -> None:
    rng = random.Random(3)
    catalog = _random_catalog(rng, items=400, depth=3)
    python, static = _providers(catalog, eos_token_id=EOS)
    all_paths = [item.token_ids for item in catalog.items]
    width = 64
    logits = torch.zeros((1, width, VOCAB), device="cuda")
    for steps in (1, 2):
        traces = [rng.choice(all_paths)[:steps] for _ in range(width)]
        state = _beam_state(traces)
        assert torch.equal(
            python.step_mask(state, logits), static.step_mask(state, logits)
        )
