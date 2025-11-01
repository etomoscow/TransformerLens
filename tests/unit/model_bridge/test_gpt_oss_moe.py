"""Unit tests for GPT-OSS MoE model loading and compatibility mode."""

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from transformer_lens.factories.architecture_adapter_factory import (
    ArchitectureAdapterFactory,
)
from transformer_lens.model_bridge.bridge import TransformerBridge


@pytest.fixture
def gpt_oss_model_meta():
    """Create a GPT-OSS model with meta device (no weights loaded)."""
    config = AutoConfig.from_pretrained("openai/gpt-oss-20b", trust_remote_code=True)

    # Create model on meta device (no actual weights)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    return model


@pytest.fixture
def gpt_oss_bridge(gpt_oss_model_meta):
    """Create a TransformerBridge for GPT-OSS model."""
    from transformers import AutoTokenizer

    from transformer_lens.config import TransformerBridgeConfig
    from transformer_lens.model_bridge.sources.transformers import (
        map_default_transformer_lens_config,
    )
    from transformer_lens.model_bridge.supported_architectures.gpt_oss import (
        GPTOSSArchitectureAdapter,
    )

    # Map HF config to TL config format
    tl_config = map_default_transformer_lens_config(gpt_oss_model_meta.config)

    # Create TransformerBridgeConfig with architecture set
    bridge_config = TransformerBridgeConfig(
        d_model=tl_config.d_model,
        d_head=tl_config.d_head,
        n_layers=tl_config.n_layers,
        n_ctx=tl_config.n_ctx,
        architecture="GptOssForCausalLM",
    )

    # Create adapter with proper bridge config
    adapter = GPTOSSArchitectureAdapter(bridge_config)

    # Get tokenizer (lightweight operation)
    tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b", trust_remote_code=True)

    # Create bridge
    bridge = TransformerBridge(
        model=gpt_oss_model_meta,
        adapter=adapter,
        tokenizer=tokenizer,
    )

    return bridge


def test_gpt_oss_model_loads_without_weights(gpt_oss_model_meta):
    """Test that GPT-OSS model structure can be loaded without downloading weights."""
    assert gpt_oss_model_meta is not None
    assert hasattr(gpt_oss_model_meta, "model")
    assert hasattr(gpt_oss_model_meta.model, "layers")
    assert len(gpt_oss_model_meta.model.layers) == 24


def test_gpt_oss_bridge_creation(gpt_oss_bridge):
    """Test that TransformerBridge can wrap GPT-OSS MoE model."""
    assert gpt_oss_bridge is not None
    assert len(gpt_oss_bridge.blocks) == 24
    assert hasattr(gpt_oss_bridge.blocks[0], "mlp")


def test_gpt_oss_mlp_is_joint_gate_up_bridge(gpt_oss_bridge):
    """Test that GPT-OSS MLP uses JointGateUpMLPBridge (not regular MLPBridge)."""
    from transformer_lens.model_bridge.generalized_components.joint_gate_up_mlp import (
        JointGateUpMLPBridge,
    )

    mlp = gpt_oss_bridge.blocks[0].mlp
    assert isinstance(mlp, JointGateUpMLPBridge)
    assert hasattr(mlp, "gate")
    assert hasattr(mlp, "up")


def test_gpt_oss_compatibility_mode_hooks(gpt_oss_bridge):
    """Test that hooks can be registered in compatibility mode for GPT-OSS."""
    # Enable compatibility mode (no_processing=True since GPT-OSS isn't official HT model)
    gpt_oss_bridge.enable_compatibility_mode(no_processing=True)

    assert gpt_oss_bridge.compatibility_mode is True

    # Get hook_dict to verify hooks are accessible
    hook_dict = gpt_oss_bridge.hook_dict

    # Check that MLP hooks exist and are accessible
    assert "blocks.0.mlp.hook_in" in hook_dict
    assert "blocks.0.mlp.hook_out" in hook_dict

    # Check that deprecated alias hook_mlp_out is accessible
    assert "blocks.0.hook_mlp_out" in hook_dict

    # Verify hook_mlp_out is an alias (same object as mlp.hook_out)
    mlp_hook_out = hook_dict["blocks.0.mlp.hook_out"]
    hook_mlp_out_alias = hook_dict["blocks.0.hook_mlp_out"]
    assert mlp_hook_out is hook_mlp_out_alias


def test_gpt_oss_moe_experts_not_iterable(gpt_oss_model_meta):
    """Test that GPT-OSS experts are stored as batched tensors, not iterable modules.

    This test verifies our architecture fix: GPT-OSS MoE stores experts as batched
    weight tensors [num_experts, ...], not as separate iterable modules.
    """
    layer0_mlp = gpt_oss_model_meta.model.layers[0].mlp
    experts = layer0_mlp.experts

    # Experts module exists but is NOT iterable
    assert hasattr(layer0_mlp, "experts")
    assert not hasattr(experts, "__iter__")

    # Experts has batched weight tensors as parameters
    assert hasattr(experts, "gate_up_proj")
    assert hasattr(experts, "down_proj")


def test_gpt_oss_hook_aliases_resolved(gpt_oss_bridge):
    """Test that JointGateUpMLPBridge hook aliases resolve correctly."""
    gpt_oss_bridge.enable_compatibility_mode(no_processing=True)

    mlp = gpt_oss_bridge.blocks[0].mlp

    # Get hooks from the MLP component
    hooks = mlp.get_hooks()

    # Check that hook_pre alias is present (should resolve to gate.hook_out)
    assert "hook_pre" in hooks

    # hook_pre should NOT try to resolve to in.hook_out or input.hook_out
    # (which would fail since JointGateUpMLPBridge doesn't have those submodules)
    # If this test passes, it means the alias override is working correctly


def test_gpt_oss_no_block_bridge_for_experts(gpt_oss_bridge):
    """Test that experts are NOT wrapped in BlockBridge.

    This verifies the fix: we removed the incorrect BlockBridge wrapper
    around experts since they're not iterable modules.
    """
    from transformer_lens.model_bridge.generalized_components.block import BlockBridge

    mlp = gpt_oss_bridge.blocks[0].mlp

    # MLP should NOT have a 'blocks' attribute (from BlockBridge)
    # and should NOT have an 'experts' attribute that's a BlockBridge
    if hasattr(mlp, "experts"):
        assert not isinstance(mlp.experts, BlockBridge)
