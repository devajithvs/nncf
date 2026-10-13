# Copyright (c) 2026 Intel Corporation
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
The data-free least-error (MSE) scale of the symmetric integer grids (ADR-0017).

With `mse_scale`, a symmetric mode's per-group scale is searched over the candidates `max|w| / k` (k geometric from
`2^(b-1)` down to `1`) on the group's own weights, and refined once by the closed-form minimiser of the winning code
pattern. The grid is the full range `[-2^(b-1), 2^(b-1) - 1]`, and the error every candidate is judged by is the one
of its *stored* (f16) scale, which is the form the compressed weight carries it in. Two properties matter: a group of
two levels, such as a binary checkpoint's `+-d`, is represented exactly (the `k = 1` candidate is `max|w|` itself),
and a group with a spread of magnitudes is free to use every level of the grid.
"""

import numpy as np
import openvino as ov
import pytest
from openvino import opset13 as opset

from nncf import CompressWeightsMode
from nncf.errors import ValidationError
from nncf.quantization import compress_weights
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_integer_quantization
from nncf.tensor import Tensor

# Bonsai's own group magnitude (finding F62), as an f16 checkpoint stores it.
BINARY_D = np.float16(0.0123).astype(np.float32)
# Groups of exactly two levels: the scale that represents them without loss is their magnitude itself.
TWO_LEVEL_GROUPS = [
    np.array([-0.5, 0.5], np.float32),
    np.array([-0.5, 0.0, 0.5], np.float32),
    np.array([BINARY_D, -BINARY_D, BINARY_D, -BINARY_D], np.float32),
]


def _compress(weight: np.ndarray, config: WeightCompressionConfig):
    compressed = do_integer_quantization(Tensor(weight.astype(np.float32)), config, reduction_axes=(1,))
    codes = np.asarray(compressed.tensor.data, np.float32)
    scale = np.asarray(compressed.scale.data, np.float32)
    return codes, scale


def _stored_mse(weight: np.ndarray, codes: np.ndarray, scale: np.ndarray) -> float:
    """The error of what lands in the compressed model: the codes against the scale the IR stores, f16."""
    stored = scale.astype(np.float16).astype(np.float32)
    return float(np.mean((codes * stored - weight) ** 2))


def _build_matmul_model(weight: np.ndarray) -> ov.Model:
    input_node = opset.parameter([1, weight.shape[1]], name="Input")
    matmul = opset.matmul(
        input_node, opset.constant(weight, name="Weights"), transpose_a=False, transpose_b=True, name="MatMul"
    )
    return ov.Model([opset.result(matmul, name="Result")], [input_node])


@pytest.mark.parametrize("mode", [CompressWeightsMode.INT2_SYM, CompressWeightsMode.INT4_SYM])
def test_mse_scale_is_exact_on_two_level_groups(mode):
    # A group of two levels has an exact representation on the full-range grid, and the search lands on it.
    for group in TWO_LEVEL_GROUPS:
        weight = group[None, :]
        codes, scale = _compress(weight, WeightCompressionConfig(mode=mode, mse_scale=True))
        np.testing.assert_array_equal(codes * scale, weight)


def test_mse_scale_on_bonsai_binary_groups_is_the_sign_grid():
    # The owner's requirement (ADR-0017): on a binary checkpoint's `+-d` groups the 2-bit scale is `d` and the codes
    # are `+-1` -- the exact representation. The default full-range rule is the negative control: it clips `+d` to a
    # half-magnitude `+1`.
    group = np.array([-BINARY_D, BINARY_D] * 64, np.float32)
    codes, scale = _compress(group[None, :], WeightCompressionConfig(mode=CompressWeightsMode.INT2_SYM, mse_scale=True))
    np.testing.assert_array_equal(codes * scale, group[None, :])
    np.testing.assert_array_equal(np.unique(codes), np.array([-1.0, 1.0]))
    assert scale[0] == BINARY_D

    default_codes, default_scale = _compress(group[None, :], WeightCompressionConfig(mode=CompressWeightsMode.INT2_SYM))
    np.testing.assert_array_equal(np.unique(default_codes), np.array([-2.0, 1.0]))
    assert default_scale[0] == BINARY_D / 2


@pytest.mark.parametrize("mode", [CompressWeightsMode.INT2_SYM, CompressWeightsMode.INT4_SYM])
def test_mse_scale_uses_every_level_of_the_grid_on_a_spread_group(mode):
    rng = np.random.default_rng(0)
    weight = rng.uniform(-1.0, 1.0, (8, 128)).astype(np.float32)
    codes, _ = _compress(weight, WeightCompressionConfig(mode=mode, mse_scale=True))
    limit = 2 ** (WeightCompressionConfig(mode=mode).num_bits - 1)
    assert codes.min() == -limit
    assert codes.max() == limit - 1


@pytest.mark.parametrize("mode", [CompressWeightsMode.INT2_SYM, CompressWeightsMode.INT4_SYM])
def test_mse_scale_is_never_worse_than_the_full_range_rule(mode):
    # The full-range scale is itself one of the search's candidates, so this holds by construction -- and it also
    # covers the optimized-path gate: the tensor is larger than MIN_INPUT_SIZE_FOR_OPTIMIZED_COMPRESSION, and at
    # INT4_SYM the optimized OpenVINO graph would return the full-range scale, making the strict comparison below
    # fail. A two-level group with an f16 remainder is where the full-range clip really costs.
    rng = np.random.default_rng(1)
    magnitude = rng.uniform(0.01, 1.0, (64, 1)).astype(np.float32)
    weights = np.where(rng.random((64, 256)) < 0.5, -magnitude, magnitude).astype(np.float32)
    searched_codes, searched_scale = _compress(weights, WeightCompressionConfig(mode=mode, mse_scale=True))
    full_codes, full_scale = _compress(weights, WeightCompressionConfig(mode=mode))
    assert _stored_mse(weights, searched_codes, searched_scale) < _stored_mse(weights, full_codes, full_scale)


@pytest.mark.parametrize("mode", [CompressWeightsMode.INT2_SYM, CompressWeightsMode.INT4_SYM])
def test_mse_scale_is_not_worse_than_the_narrow_range_rule(mode):
    rng = np.random.default_rng(3)
    weight = rng.standard_normal((64, 256)).astype(np.float32)
    searched_codes, searched_scale = _compress(weight, WeightCompressionConfig(mode=mode, mse_scale=True))
    narrow_codes, narrow_scale = _compress(weight, WeightCompressionConfig(mode=mode, narrow_range=True))
    assert _stored_mse(weight, searched_codes, searched_scale) <= _stored_mse(weight, narrow_codes, narrow_scale)


@pytest.mark.parametrize(
    "mode", [CompressWeightsMode.INT2_SYM, CompressWeightsMode.INT4_SYM, CompressWeightsMode.INT8_SYM]
)
def test_mse_scale_never_exceeds_the_group_s_largest_magnitude(mode):
    # A scale above max|w| would leave the outer levels unused by construction; the search's candidates are
    # max|w| / k with k >= 1, so it stays at or below it. (A scale *below* max|w| / 2^(b-1) is legitimate: the
    # search may trade an outlier's error for the group's.)
    rng = np.random.default_rng(2)
    weight = rng.standard_normal((16, 128)).astype(np.float32)
    _, scale = _compress(weight, WeightCompressionConfig(mode=mode, mse_scale=True))
    assert np.all(scale[:, 0] <= np.abs(weight).max(axis=1))


@pytest.mark.parametrize(
    "mode",
    [
        CompressWeightsMode.INT4_ASYM,
        CompressWeightsMode.INT8_ASYM,
        CompressWeightsMode.NF4,
        CompressWeightsMode.INT1_SYM,
    ],
)
def test_mse_scale_is_refused_for_other_modes(mode):
    with pytest.raises(ValidationError):
        WeightCompressionConfig(mode=mode, mse_scale=True)


def test_mse_scale_and_narrow_range_are_two_choices_of_one_thing():
    with pytest.raises(ValidationError):
        WeightCompressionConfig(mode=CompressWeightsMode.INT4_SYM, mse_scale=True, narrow_range=True)


def test_mse_scale_default_is_off():
    assert WeightCompressionConfig(mode=CompressWeightsMode.INT4_SYM).mse_scale is False
    assert AdvancedCompressionParameters().mse_scale is False


@pytest.mark.parametrize("mode", [CompressWeightsMode.INT2_SYM, CompressWeightsMode.INT4_SYM])
def test_mse_scale_reaches_a_compressed_model_and_keeps_two_level_weights_exact(mode):
    # End to end through the advanced parameters: the searched scale represents every two-level group of the weight
    # exactly, so the constant's own dequantization reproduces the weight bit for bit. The constant is the mode's
    # sub-byte type with no zero point.
    group_size, num_groups = 128, 4
    hidden_dim = group_size * num_groups
    magnitudes = np.float16(np.linspace(0.01, 0.02, num_groups)).astype(np.float32)
    weight = (
        np.where(np.random.default_rng(4).random((hidden_dim, hidden_dim)) < 0.5, -1.0, 1.0).astype(np.float32)
        * np.repeat(magnitudes, group_size)[None, :]
    )

    model = _build_matmul_model(weight)
    compressed_model = compress_weights(
        model,
        mode=mode,
        group_size=group_size,
        ratio=1.0,
        all_layers=True,
        advanced_parameters=AdvancedCompressionParameters(mse_scale=True),
    )

    constants = {
        op.get_friendly_name(): op for op in compressed_model.get_ordered_ops() if op.get_type_name() == "Constant"
    }
    num_bits = WeightCompressionConfig(mode=mode).num_bits
    expected_type = {2: ov.Type.i2, 4: ov.Type.i4}[num_bits]
    assert constants["Weights"].get_element_type() == expected_type
    assert "Weights/zero_point" not in constants

    # The canonical dequantization the compressed weight carries: Convert(w) -> Multiply(scale), executed on the CPU.
    dequantized = ov.Model(
        [
            opset.result(
                opset.multiply(
                    opset.convert(constants["Weights"], ov.Type.f32),
                    opset.convert(constants["Weights/scale"], ov.Type.f32),
                ),
                name="Result",
            )
        ],
        [],
    )
    reproduced = ov.Core().compile_model(dequantized, "CPU")({})[0].astype(np.float32).reshape(weight.shape)
    np.testing.assert_array_equal(reproduced, weight)


def test_mse_scale_compressed_int2_model_computes_what_the_original_did():
    # For `i2` the CPU plugin's MatMul of a group-scaled weight is bit-exact against the f32 matmul, so the whole
    # model can be checked, not only its constants. (`i4` is not: with the same weight the plugin's MatMul differs
    # from the f32 matmul by up to one code step, so this assertion is the `i2` one.)
    group_size, num_groups = 128, 4
    hidden_dim = group_size * num_groups
    magnitudes = np.float16(np.linspace(0.01, 0.02, num_groups)).astype(np.float32)
    weight = (
        np.where(np.random.default_rng(4).random((hidden_dim, hidden_dim)) < 0.5, -1.0, 1.0).astype(np.float32)
        * np.repeat(magnitudes, group_size)[None, :]
    )

    model = _build_matmul_model(weight)
    input_data = np.random.default_rng(0).standard_normal((1, hidden_dim)).astype(np.float32)
    reference = ov.Core().compile_model(model, "CPU")({0: input_data})[0]
    compressed_model = compress_weights(
        model,
        mode=CompressWeightsMode.INT2_SYM,
        group_size=group_size,
        ratio=1.0,
        all_layers=True,
        advanced_parameters=AdvancedCompressionParameters(mse_scale=True),
    )
    result = ov.Core().compile_model(compressed_model, "CPU")({0: input_data})[0]
    assert np.array_equal(result, reference)
