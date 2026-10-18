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
import numpy as np
import openvino as ov
import pytest
from openvino import opset13 as opset

import nncf
from nncf.parameters import CompressWeightsMode
from nncf.quantization import compress_weights
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_integer_dequantization
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_integer_quantization
from nncf.tensor import Tensor


def _build_matmul_model(weight: np.ndarray) -> ov.Model:
    input_node = opset.parameter([1, weight.shape[1]], name="Input")
    matmul = opset.matmul(
        input_node, opset.constant(weight, name="Weights"), transpose_a=False, transpose_b=True, name="MatMul"
    )
    return ov.Model([opset.result(matmul, name="Result")], [input_node])


@pytest.mark.parametrize("data", ([-2.0, 2.0, 2.0, -2.0], [0.5, -0.5, -0.5, 0.5]))
def test_sign_grid_is_exact_on_two_level_groups(data):
    """
    The sign grid stores the sign and scales it by the group's mean absolute weight; on a +/-d group that is d,
    so the group is reconstructed exactly, with no zero point.
    """
    data = np.array(data, np.float32)
    config = WeightCompressionConfig(mode=CompressWeightsMode.INT1_SYM)
    compressed = do_integer_quantization(Tensor(data), config, -1)
    decoded = do_integer_dequantization(compressed)

    assert np.array_equal(decoded.data, data)
    assert np.array_equal(np.unique(compressed.tensor.data), np.array([-1, 1], np.int8))
    assert compressed.zero_point is None
    assert np.array_equal(compressed.scale.data, np.array([np.abs(data).mean()], np.float32))


def test_sign_grid_is_emitted_as_s1_without_a_zero_point():
    """
    One sign bit per weight: the constant is `s1`, half a byte's worth of payload for eight weights, and the graph
    is Convert -> Multiply by the scale with no zero point. The compressed model reproduces the reference on CPU.
    """
    group_size, num_groups = 8, 2
    hidden_dim = group_size * num_groups
    scales = np.array([0.5, 0.25], np.float32)
    signs = np.resize(np.array([1.0, -1.0, -1.0, 1.0], np.float32), hidden_dim)
    row = signs * np.repeat(scales, group_size)
    weight = np.tile(row, (hidden_dim, 1))

    model = _build_matmul_model(weight)
    input_data = np.random.default_rng(0).standard_normal((1, hidden_dim)).astype(np.float32)
    reference = ov.Core().compile_model(model, "CPU")({0: input_data})[0]

    compressed_model = compress_weights(
        model, mode=CompressWeightsMode.INT1_SYM, group_size=group_size, ratio=1.0, all_layers=True
    )
    result = ov.Core().compile_model(compressed_model, "CPU")({0: input_data})[0]
    assert np.array_equal(result, reference)

    constants = {
        op.get_friendly_name(): op for op in compressed_model.get_ordered_ops() if op.get_type_name() == "Constant"
    }
    assert constants["Weights"].get_element_type() == ov.Type.s1
    assert constants["Weights"].get_byte_size() == weight.size // 8
    assert "Weights/zero_point" not in constants


@pytest.mark.parametrize(
    "params",
    (
        {"dataset": "anything"},
        {"awq": True},
        {"scale_estimation": True},
        {"gptq": True},
        {"lora_correction": True},
    ),
)
def test_sign_grid_refuses_data_aware_options(params):
    with pytest.raises(nncf.ParameterNotSupportedError):
        compress_weights(ov.Model([], []), mode=CompressWeightsMode.INT1_SYM, **params)


def test_narrow_range_is_rejected_for_the_sign_grid():
    with pytest.raises(nncf.ValidationError):
        WeightCompressionConfig(mode=CompressWeightsMode.INT1_SYM, narrow_range=True)
