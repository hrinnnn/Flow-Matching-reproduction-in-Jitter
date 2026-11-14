"""CPU tests for minibatch OT matching and the PyTorch key rewrite."""

import numpy as np

from jfm.paths.minibatch_ot import match_source_to_target, squared_cost
from tools.convert_unet_weights import convert_state_dict, pytorch_key_to_jittor


def test_exact_minibatch_ot_recovers_a_known_permutation():
    rng = np.random.default_rng(0)
    source = rng.normal(size=(6, 4))
    order = np.array([2, 0, 5, 1, 4, 3])
    target = source[order]
    matched, perm = match_source_to_target(source, target, method="exact")
    np.testing.assert_allclose(matched, target)
    np.testing.assert_array_equal(perm, order)
    independent = squared_cost(source, target).trace()
    matched_cost = squared_cost(matched, target).trace()
    assert matched_cost < independent


def test_sinkhorn_matching_beats_the_identity_pairing_on_a_swap():
    source = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    target = source[::-1].copy()
    matched, _ = match_source_to_target(source, target, method="sinkhorn")
    assert squared_cost(matched, target).trace() < squared_cost(source, target).trace()


def test_pytorch_unet_keys_map_onto_jittor_names():
    samples = {
        "time_embed.0.weight": "time_embed.0.weight",
        "time_embed.2.bias": "time_embed.2.bias",
        "label_emb.weight": "label_emb.weight",
        "input_blocks.0.0.weight": "input_blocks.m0.layers.m0.weight",
        "input_blocks.1.0.in_layers.0.weight": "input_blocks.m1.layers.m0.in_norm.weight",
        "input_blocks.1.0.in_layers.2.bias": "input_blocks.m1.layers.m0.in_conv.bias",
        "input_blocks.1.0.emb_layers.1.weight": "input_blocks.m1.layers.m0.emb_linear.weight",
        "input_blocks.1.0.out_layers.0.bias": "input_blocks.m1.layers.m0.out_norm.bias",
        "input_blocks.1.0.out_layers.3.weight": "input_blocks.m1.layers.m0.out_conv.weight",
        "input_blocks.1.1.qkv.weight": "input_blocks.m1.layers.m1.qkv.weight",
        "input_blocks.3.0.op.weight": "input_blocks.m3.layers.m0.op.weight",
        "middle_block.0.in_layers.2.weight": "middle_block.layers.m0.in_conv.weight",
        "middle_block.1.proj_out.weight": "middle_block.layers.m1.proj_out.weight",
        "output_blocks.2.1.conv.weight": "output_blocks.m2.layers.m1.conv.weight",
        "out.2.weight": "out.2.weight",
    }
    for src, dst in samples.items():
        assert pytorch_key_to_jittor(src) == dst
    converted = convert_state_dict({key: np.zeros((2, 2)) for key in samples})
    assert set(converted) == set(samples.values())
