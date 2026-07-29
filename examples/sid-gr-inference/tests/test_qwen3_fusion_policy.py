# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util

import pytest

from gr_inference.gr_models.qwen3.fusion import (
    FUSION_FAMILIES,
    Qwen3FusionPolicy,
    default_qwen3_fusion_policy,
    reset_default_qwen3_fusion_policy,
)


@pytest.fixture(autouse=True)
def reset_policy() -> None:
    reset_default_qwen3_fusion_policy()
    yield
    reset_default_qwen3_fusion_policy()


def test_qwen3_fusion_policy_defaults_to_all_enabled(monkeypatch) -> None:
    monkeypatch.delenv("GR_INFERENCE_QWEN3_DISABLE_FUSIONS", raising=False)

    policy = default_qwen3_fusion_policy()

    assert policy.disabled == frozenset()
    assert all(policy.enabled(family) for family in FUSION_FAMILIES)


def test_qwen3_fusion_policy_parses_combination(monkeypatch) -> None:
    monkeypatch.setenv(
        "GR_INFERENCE_QWEN3_DISABLE_FUSIONS",
        " rope,MLP ",
    )

    policy = default_qwen3_fusion_policy()

    assert policy.disabled == {"mlp", "rope"}
    assert policy.enabled("qk_norm")
    assert not policy.enabled("rope")


def test_qwen3_fusion_policy_expands_all(monkeypatch) -> None:
    monkeypatch.setenv("GR_INFERENCE_QWEN3_DISABLE_FUSIONS", "all")

    policy = default_qwen3_fusion_policy()

    assert policy.disabled == FUSION_FAMILIES
    assert policy.summary()["enabled"] == []


@pytest.mark.parametrize("value", ["unknown", "all,rope"])
def test_qwen3_fusion_policy_rejects_invalid_values(monkeypatch, value: str) -> None:
    monkeypatch.setenv("GR_INFERENCE_QWEN3_DISABLE_FUSIONS", value)

    with pytest.raises(ValueError, match="GR_INFERENCE_QWEN3_DISABLE_FUSIONS"):
        Qwen3FusionPolicy.from_env()


def _tiny_ops():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")
    from gr_inference.gr_models.qwen3 import Qwen3GRConfig
    from gr_inference.gr_models.qwen3.layers import TorchQwen3LayerOps

    return TorchQwen3LayerOps(
        Qwen3GRConfig(
            model_name="tiny-qwen3-gr",
            num_layers=1,
            hidden_size=8,
            num_attention_heads=2,
            num_kv_heads=1,
            head_dim=4,
            max_context_len=8,
            max_seq_len=10,
            max_decode_steps=2,
            max_beam_width=4,
            intermediate_size=16,
        )
    )


def test_qk_norm_fusion_gate_skips_direct_sglang_kernel(monkeypatch) -> None:
    import gr_inference.gr_models.qwen3.layers as layers
    import torch

    ops = _tiny_ops()
    q = torch.randn(1, 2, 2, 4)
    k = torch.randn(1, 2, 1, 4)
    monkeypatch.setattr(
        layers,
        "qwen3_fusion_enabled",
        lambda family: family != "qk_norm",
    )
    monkeypatch.setattr(
        layers,
        "_apply_sglang_fused_qknorm",
        lambda *_args, **_kwargs: pytest.fail("fused qk norm was dispatched"),
    )

    q_out, k_out = ops.qk_norm_rope(q, k)

    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


def test_rope_fusion_gate_skips_sglang_and_flashinfer_kernels(monkeypatch) -> None:
    import gr_inference.gr_models.qwen3.layers as layers
    import torch

    q = torch.randn(1, 2, 2, 4)
    k = torch.randn(1, 2, 1, 4)
    monkeypatch.setattr(
        layers,
        "qwen3_fusion_enabled",
        lambda family: family != "rope",
    )
    monkeypatch.setattr(
        layers,
        "_apply_sglang_rope_inplace",
        lambda *_args, **_kwargs: pytest.fail("SGLang RoPE was dispatched"),
    )

    def fail_flashinfer(*_args, **_kwargs):
        pytest.fail("FlashInfer RoPE was dispatched")

    monkeypatch.setattr(
        layers,
        "_flashinfer_apply_rope_pos_ids",
        lambda: fail_flashinfer,
    )

    q_out, k_out = layers.apply_qwen3_rope(q, k)

    assert q_out.shape == q.shape
    assert k_out.shape == k.shape


def test_add_rmsnorm_fusion_gate_uses_composed_path(monkeypatch) -> None:
    import gr_inference.gr_models.qwen3.layers as layers
    import torch

    ops = _tiny_ops()
    residual = torch.randn(2, 3, 8)
    projected = torch.randn_like(residual)
    monkeypatch.setattr(
        layers,
        "qwen3_fusion_enabled",
        lambda family: family != "add_rmsnorm",
    )

    def fail_fused(*_args, **_kwargs):
        pytest.fail("fused add RMSNorm was dispatched")

    monkeypatch.setattr(
        layers,
        "_flashinfer_fused_add_rmsnorm",
        lambda: fail_fused,
    )
    monkeypatch.setattr(
        layers,
        "_apply_gr_trtllm_exact_fused_add_rmsnorm",
        fail_fused,
    )

    hidden_states, normed = ops.post_attention_residual_norm_projected(
        residual,
        projected,
    )
    decode_fused = layers._apply_decode_next_input_norm(
        projected,
        residual,
        ops.post_attention_layernorm,
    )

    assert torch.allclose(hidden_states, residual + projected)
    assert normed.shape == hidden_states.shape
    assert decode_fused is None


def test_mlp_fusion_gate_uses_explicit_torch_path(monkeypatch) -> None:
    import gr_inference.gr_models.qwen3.layers as layers
    import torch
    from gr_inference.gr_kernels import CAP_FUSED_MLP

    ops = _tiny_ops()
    hidden_states = torch.randn(2, 3, 8)
    monkeypatch.setattr(
        layers,
        "qwen3_fusion_enabled",
        lambda family: family != "mlp",
    )

    def selected_backend(capability: str) -> str:
        if capability == CAP_FUSED_MLP:
            pytest.fail("fused MLP backend was selected")
        return "torch"

    monkeypatch.setattr(layers, "_selected_kernel_backend", selected_backend)

    output = ops.mlp(hidden_states)

    assert output.shape == hidden_states.shape
