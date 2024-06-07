from typing import Callable, Optional, Tuple

import numpy as np
from tqdm import tqdm

from coremltools import _logger as logger
from coremltools.converters.mil._deployment_compatibility import AvailableTarget
from coremltools.converters.mil.backend.mil.load import should_use_weight_file
from coremltools.converters.mil.mil import Builder as mb
from coremltools.converters.mil.mil import Operation, Program, types
from coremltools.converters.mil.mil.block import is_current_opset_version_compatible_with
from coremltools.converters.mil.mil.ops.defs._utils import pack_elements_into_bits
from coremltools.converters.mil.mil.ops.defs.iOS16 import (
    constexpr_affine_dequantize,
    constexpr_lut_to_dense,
    constexpr_sparse_to_dense,
)
from coremltools.converters.mil.mil.passes.defs.quantization import AbstractQuantizationPass
from coremltools.converters.mil.mil.passes.helper import block_context_manager
from coremltools.converters.mil.mil.passes.pass_registry import register_pass
from coremltools.converters.mil.mil.types.type_mapping import nptype_from_builtin
from coremltools.models.neural_network.quantization_utils import _get_kmeans_lookup_table_and_weight
from coremltools.optimize.coreml._config import (
    OpLinearQuantizerConfig,
    OpMagnitudePrunerConfig,
    OpPalettizerConfig,
    OpThresholdPrunerConfig,
    OptimizationConfig,
)
from coremltools.optimize.coreml._quantization_passes import AbstractCompressionPass, SparseParams
from coremltools.models import MLModel as _MLModel
from coremltools.optimize.coreml import OptimizationConfig as _OptimizationConfig
from coremltools.optimize.coreml._post_training_quantization import _apply_graph_pass


from .bitplane_operation import replace_random_from_lower_2_higher

def embedding_weights(mlmodel: _MLModel, config: _OptimizationConfig):

    weight_pruner = _embedding_weights(config, fake_compression=False)
    return _apply_graph_pass(mlmodel, weight_pruner)

@register_pass(namespace="compression")
class _embedding_weights(AbstractCompressionPass):
    """
    This transform works for each ``const`` op if:

    - ``_is_deprecated=True`` and the ``op_selector`` returns ``True``.
    - ``_is_deprecated=False`` and the ``const`` value size ``> weight_threshold``.

    The transform performs the following:

    - The fraction of values with the least absolute value are zeroed out (self.sparsity).
    - If ``fake_compression=False``, the zeroed-out value is encoded using the ``constexpr_sparse_to_dense`` op.
    - If ``fake_compression=True``, the zeroed-out value is encoded using the ``const`` op.
    - Old ``const`` is replaced by a new operation with zeroed-out value.
    """
    _SUPPORTED_CONFIG_TYPE = (OpMagnitudePrunerConfig, OpThresholdPrunerConfig)

    def is_valid_op(self, op: Operation):
        if op.op_type == "const" and should_use_weight_file(op.outputs[0].val):
            return True
        return False

    @staticmethod
    def _pack_val_to_sparse_param(val):
        flattened_val = val.flatten()
        params = SparseParams()
        params.nonzero_data = flattened_val[np.where(flattened_val != 0)]
        params.mask = np.packbits(np.where(flattened_val != 0, 1, 0), bitorder="little")
        params.shape = val.shape
        return params

    @staticmethod
    def compress_by_threshold(val, threshold, minimum_sparsity_percentile):
        val = np.where(np.abs(val) <= threshold, 0, val)
        sparsity_percentile = np.sum(val == 0.0) / val.size
        if sparsity_percentile < minimum_sparsity_percentile:
            msg = (f"weight value has sparsity of {sparsity_percentile} < "
                   f"minimum_sparsity_percentile {minimum_sparsity_percentile}. Skipped."
                  )
            logger.warning(msg)
            return None
        return _embedding_weights._pack_val_to_sparse_param(val)

    @staticmethod
    def embedded_params(val):
        print(val, val.dtype)
        val = replace_random_from_lower_2_higher(val, fraction=19)
        return _embedding_weights._pack_val_to_sparse_param(val)

    @staticmethod
    def decompress(params):
        if not isinstance(params, SparseParams):
            raise ValueError("Invalid type of params")
        return constexpr_sparse_to_dense.decompress(params.nonzero_data, params.mask, params.shape)

    def transform_op(self, op: Operation):
        op_config = self.config._get_const_op_config(op)
        if op_config is None:
            return
        if not self.need_compress_const(op, self.config._is_deprecated, op_config.weight_threshold):
            return

        if not isinstance(op.outputs[0].val, (np.ndarray, np.generic)):
            raise ValueError("Only numpy arrays are supported")
        embedded_params = self.embedded_params(
            val=op.outputs[0].val,
        )
        if not self.fake_compression:
            new_var = mb.constexpr_sparse_to_dense(
                nonzero_data=embedded_params.nonzero_data,
                mask=embedded_params.mask,
                shape=np.uint32(embedded_params.shape),
                before_op=op,
                name=op.name + "_sparsified",
            )
        else:
            decompressed_val = self.decompress(embedded_params)
            new_var = mb.const(
                val=decompressed_val,
                before_op=op,
                name=op.name + "_fake_sparsified",
            )
        
        op.enclosing_block.replace_uses_of_var_after_op(
            anchor_op=op,
            old_var=op.outputs[0],
            new_var=new_var,
            no_check_var_types=True,
        )
        op.enclosing_block.remove_ops([op])