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
Narrow-range symmetric integer grids (ADR-0014, plan item 39).

With `narrow_range`, a symmetric mode's grid is [-(2^(b-1) - 1), 2^(b-1) - 1] and its scale is max|w| / (2^(b-1) - 1),
so a group of weights symmetric in magnitude (e.g. {-d, 0, +d}) is represented exactly. The default grid keeps the
extra negative level and clips the positive end of such a group.
"""

import numpy as np
import pytest

from nncf import CompressWeightsMode
from nncf.errors import ValidationError
from nncf.openvino.optimized_functions import do_integer_quantization as optimized_do_integer_quantization
from nncf.quantization.advanced_parameters import AdvancedCompressionParameters
from nncf.quantization.algorithms.weight_compression.config import WeightCompressionConfig
from nncf.quantization.algorithms.weight_compression.weight_lowering import do_integer_quantization
from nncf.tensor import Tensor


def _reconstruct(weight: np.ndarray, config: WeightCompressionConfig, optimized: bool = False) -> np.ndarray:
    fn = optimized_do_integer_quantization if optimized else do_integer_quantization
    compressed = fn(Tensor(weight), config, reduction_axes=(1,))
    return np.asarray(compressed.tensor.data, np.float32) * np.asarray(compressed.scale.data, np.float32)


# Each row is one group; its codes are exact on the narrow grid when the group is symmetric in magnitude.
SYMMETRIC_GROUPS = [
    np.array([-0.5, 0.5], np.float32),
    np.array([-0.5, 0.0, 0.5], np.float32),
    np.array([0.25, -0.25, 0.0, 0.25], np.float32),
    np.array([0.75, -0.75, -0.75, 0.75], np.float32),
]


@pytest.mark.parametrize("mode", [CompressWeightsMode.INT2_SYM, CompressWeightsMode.INT4_SYM])
def test_narrow_range_is_exact_on_symmetric_groups(mode):
    for group in SYMMETRIC_GROUPS:
        weight = group[None, :]
        config = WeightCompressionConfig(mode=mode, narrow_range=True)
        np.testing.assert_array_equal(_reconstruct(weight, config), weight)


def test_default_grid_clips_the_positive_end_of_symmetric_groups():
    # The negative control: without the flag, INT2_SYM's grid is {-2, -1, 0, 1} and +d maps to the clipped +1.
    weight = np.array([[-0.5, 0.5]], np.float32)
    config = WeightCompressionConfig(mode=CompressWeightsMode.INT2_SYM)
    assert not np.array_equal(_reconstruct(weight, config), weight)


@pytest.mark.parametrize("mode", [CompressWeightsMode.INT2_SYM, CompressWeightsMode.INT4_SYM])
def test_narrow_range_codes_stay_inside_the_narrow_grid(mode):
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((8, 128)).astype(np.float32)
    config = WeightCompressionConfig(mode=mode, narrow_range=True)
    compressed = do_integer_quantization(Tensor(weight), config, reduction_axes=(1,))
    codes = np.asarray(compressed.tensor.data)
    limit = 2 ** (config.num_bits - 1) - 1
    assert codes.min() >= -limit
    assert codes.max() <= limit


@pytest.mark.parametrize("mode", [CompressWeightsMode.INT4_SYM, CompressWeightsMode.INT8_SYM])
def test_optimized_path_equals_reference_with_narrow_range(mode):
    rng = np.random.default_rng(1)
    weight = rng.standard_normal((16, 128)).astype(np.float32)
    config = WeightCompressionConfig(mode=mode, narrow_range=True)
    ref = _reconstruct(weight, config)
    opt = _reconstruct(weight, config, optimized=True)
    np.testing.assert_allclose(opt, ref, rtol=0, atol=1e-6)


@pytest.mark.parametrize(
    "mode", [CompressWeightsMode.INT4_ASYM, CompressWeightsMode.INT8_ASYM, CompressWeightsMode.NF4]
)
def test_narrow_range_is_refused_for_other_modes(mode):
    with pytest.raises(ValidationError):
        WeightCompressionConfig(mode=mode, narrow_range=True)


def test_narrow_range_default_is_off():
    assert WeightCompressionConfig(mode=CompressWeightsMode.INT4_SYM).narrow_range is False
    assert AdvancedCompressionParameters().narrow_range is False
