
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\stable_diffusion\engine_builder_ort_cuda.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import gc
import logging
import os

import onnx
import torch
from diffusion_models import PipelineInfo
from engine_builder import EngineBuilder, EngineType
from packaging import version

import onnxruntime as ort
from onnxruntime.transformers.io_binding_helper import CudaSession, GpuBindingManager
from onnxruntime.transformers.onnx_model import OnnxModel

logger = logging.getLogger(__name__)


class OrtCudaEngine:
    def __init__(
        self,
        onnx_path,
        device_id: int = 0,
        enable_cuda_graph: bool = False,
        disable_optimization: bool = False,
        max_cuda_graphs: int = 1,
    ):
        self.onnx_path = onnx_path
        self.provider = "CUDAExecutionProvider"
        self.stream = torch.cuda.current_stream().cuda_stream
        self.provider_options = CudaSession.get_cuda_provider_options(device_id, enable_cuda_graph, self.stream)
        session_options = ort.SessionOptions()

        # When the model has been optimized by onnxruntime, we can disable optimization to save session creation time.
        if disable_optimization:
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

        logger.info("creating CUDA EP session for %s", onnx_path)
        ort_session = ort.InferenceSession(
            onnx_path,
            session_options,
            providers=[
                (self.provider, self.provider_options),
                "CPUExecutionProvider",
            ],
        )
        logger.info("created CUDA EP session for %s", onnx_path)

        device = torch.device("cuda", device_id)
        self.enable_cuda_graph = enable_cuda_graph

        # Support multiple CUDA graphs for different input shapes.
        # For clip2 model that disabled cuda graph, max_cuda_graphs is updated to 0 here.
        self.gpu_binding_manager = GpuBindingManager(
            ort_session=ort_session,
            device=device,
            stream=self.stream,
            max_cuda_graphs=max_cuda_graphs if enable_cuda_graph else 0,
        )

        self.current_gpu_binding = None

    def metadata(self, name: str):
        data = {}
        if self.current_gpu_binding is not None:
            if self.current_gpu_binding.last_run_gpu_graph_id >= 0:
                data[f"{name}.gpu_graph_id"] = self.current_gpu_binding.last_run_gpu_graph_id
        return data

    def infer(self, feed_dict: dict[str, torch.Tensor]):
        return self.current_gpu_binding.infer(feed_dict=feed_dict, disable_cuda_graph_in_run=not self.enable_cuda_graph)

    def allocate_buffers(self, shape_dict, device):
        self.current_gpu_binding = self.gpu_binding_manager.get_binding(
            shape_dict=shape_dict, use_cuda_graph=self.enable_cuda_graph
        )


class _ModelConfig:
    """
    Configuration of one model (like Clip, UNet etc) on ONNX export and optimization for CUDA provider.
    For example, if you want to use fp32 in layer normalization, set the following:
        force_fp32_ops=["SkipLayerNormalization", "LayerNormalization"]
    """

    def __init__(
        self,
        onnx_opset_version: int,
        use_cuda_graph: bool,
        fp16: bool = True,
        force_fp32_ops: list[str] | None = None,
        optimize_by_ort: bool = True,
    ):
        self.onnx_opset_version = onnx_opset_version
        self.use_cuda_graph = use_cuda_graph
        self.fp16 = fp16
        self.force_fp32_ops = force_fp32_ops
        self.optimize_by_ort = optimize_by_ort


class OrtCudaEngineBuilder(EngineBuilder):
    def __init__(
        self,
        pipeline_info: PipelineInfo,
        max_batch_size=16,
        device="cuda",
        use_cuda_graph=False,
    ):
        """
        Initializes the ONNX Runtime TensorRT ExecutionProvider Engine Builder.

        Args:
            pipeline_info (PipelineInfo):
                Version and Type of pipeline.
            max_batch_size (int):
                Maximum batch size for dynamic batch engine.
            device (str):
                device to run.
            use_cuda_graph (bool):
                Use CUDA graph to capture engine execution and then launch inference
        """
        super().__init__(
            EngineType.ORT_CUDA,
            pipeline_info,
            max_batch_size=max_batch_size,
            device=device,
            use_cuda_graph=use_cuda_graph,
        )

        self.model_config = {}

    def _configure(
        self,
        model_name: str,
        onnx_opset_version: int,
        use_cuda_graph: bool,
        fp16: bool = True,
        force_fp32_ops: list[str] | None = None,
        optimize_by_ort: bool = True,
    ):
        self.model_config[model_name] = _ModelConfig(
            onnx_opset_version,
            use_cuda_graph,
            fp16=fp16,
            force_fp32_ops=force_fp32_ops,
            optimize_by_ort=optimize_by_ort,
        )

    def configure_xl(self, onnx_opset_version: int):
        self._configure(
            "clip",
            onnx_opset_version=onnx_opset_version,
            use_cuda_graph=self.use_cuda_graph,
        )
        self._configure(
            "clip2",
            onnx_opset_version=onnx_opset_version,  # TODO: ArgMax-12 is not implemented in CUDA
            use_cuda_graph=False,  # TODO: fix Runtime Error with cuda graph
        )
        self._configure(
            "unetxl",
            onnx_opset_version=onnx_opset_version,
            use_cuda_graph=self.use_cuda_graph,
        )

        self._configure(
            "vae",
            onnx_opset_version=onnx_opset_version,
            use_cuda_graph=self.use_cuda_graph,
        )

    def optimized_onnx_path(self, engine_dir, model_name):
        suffix = "" if self.model_config[model_name].fp16 else ".fp32"
        return self.get_onnx_path(model_name, engine_dir, opt=True, suffix=suffix)

    def import_diffusers_engine(self, diffusers_onnx_dir: str, engine_dir: str):
        """Import optimized onnx models for diffusers from Olive or optimize_pipeline tools.

        Args:
            diffusers_onnx_dir (str): optimized onnx directory of Olive
            engine_dir (str): the directory to store imported onnx
        """
        if version.parse(ort.__version__) < version.parse("1.17.0"):
            print("Skip importing since onnxruntime-gpu version < 1.17.0.")
            return

        for model_name, model_obj in self.models.items():
            onnx_import_path = self.optimized_onnx_path(diffusers_onnx_dir, model_name)
            if not os.path.exists(onnx_import_path):
                print(f"{onnx_import_path} not existed. Skip importing.")
                continue

            onnx_opt_path = self.optimized_onnx_path(engine_dir, model_name)
            if os.path.exists(onnx_opt_path):
                print(f"{onnx_opt_path} existed. Skip importing.")
                continue

            if model_name == "vae" and self.pipeline_info.is_xl():
                print(f"Skip importing VAE since it is not fully compatible with float16: {onnx_import_path}.")
                continue

            model = OnnxModel(onnx.load(onnx_import_path, load_external_data=True))

            if model_name in ["clip", "clip2"]:
                hidden_states_per_layer = []
                for output in model.graph().output:
                    if output.name.startswith("hidden_states."):
                        hidden_states_per_layer.append(output.name)
                if hidden_states_per_layer:
                    kept_hidden_states = hidden_states_per_layer[-2 - model_obj.clip_skip]
                    model.rename_graph_output(kept_hidden_states, "hidden_states")

                model.rename_graph_output(
                    "last_hidden_state" if model_name == "clip" else "text_embeds", "text_embeddings"
                )
                model.prune_graph(
                    ["text_embeddings", "hidden_states"] if hidden_states_per_layer else ["text_embeddings"]
                )

                if model_name == "clip2":
                    model.change_graph_input_type(model.find_graph_input("input_ids"), onnx.TensorProto.INT32)

                model.save_model_to_file(onnx_opt_path, use_external_data_format=(model_name == "clip2"))
            elif model_name in ["unet", "unetxl"]:
                model.rename_graph_output("out_sample", "latent")
                model.save_model_to_file(onnx_opt_path, use_external_data_format=True)

            del model
            continue

    def build_engines(
        self,
        engine_dir: str,
        framework_model_dir: str,
        onnx_dir: str,
        tmp_dir: str | None = None,
        onnx_opset_version: int = 17,
        device_id: int = 0,
        save_fp32_intermediate_model: bool = False,
        import_engine_dir: str | None = None,
        max_cuda_graphs: int = 1,
    ):
        self.torch_device = torch.device("cuda", device_id)
        self.load_models(framework_model_dir)

        if not os.path.isdir(engine_dir):
            os.makedirs(engine_dir)

        if not os.path.isdir(onnx_dir):
            os.makedirs(onnx_dir)

        # Add default configuration if missing
        if self.pipeline_info.is_xl():
            self.configure_xl(onnx_opset_version)
        for model_name in self.models:
            if model_name not in self.model_config:
                self.model_config[model_name] = _ModelConfig(onnx_opset_version, self.use_cuda_graph)

        # Import Engine
        if import_engine_dir:
            if self.pipeline_info.is_xl():
                self.import_diffusers_engine(import_engine_dir, engine_dir)
            else:
                print(f"Only support importing SDXL onnx. Ignore --engine-dir {import_engine_dir}")

        # Load lora only when we need export text encoder or UNet to ONNX.
        load_lora = False
        if self.pipeline_info.lora_weights:
            for model_name in self.models:
                if model_name not in ["clip", "clip2", "unet", "unetxl"]:
                    continue
                onnx_path = self.get_onnx_path(model_name, onnx_dir, opt=False)
                onnx_opt_path = self.optimized_onnx_path(engine_dir, model_name)
                if not os.path.exists(onnx_opt_path):
                    if not os.path.exists(onnx_path):
                        load_lora = True
                        break

        # Export models to ONNX
        self.disable_torch_spda()
        pipe = self.load_pipeline_with_lora() if load_lora else None

        for model_name, model_obj in self.models.items():
            if model_name == "vae" and self.vae_torch_fallback:
                continue

            onnx_path = self.get_onnx_path(model_name, onnx_dir, opt=False)
            onnx_opt_path = self.optimized_onnx_path(engine_dir, model_name)
            if not os.path.exists(onnx_opt_path):
                if not os.path.exists(onnx_path):
                    print("----")
                    logger.info("Exporting model: %s", onnx_path)

                    model = self.get_or_load_model(pipe, model_name, model_obj, framework_model_dir)
                    model = model.to(torch.float32)

                    with torch.inference_mode():
                        # For CUDA EP, export FP32 onnx since some graph fusion only supports fp32 graph pattern.
                        # Export model with sample of batch size 1, image size 512 x 512
                        inputs = model_obj.get_sample_input(1, 512, 512)

                        torch.onnx.export(
                            model,
                            inputs,
                            onnx_path,
                            export_params=True,
                            opset_version=self.model_config[model_name].onnx_opset_version,
                            do_constant_folding=True,
                            input_names=model_obj.get_input_names(),
                            output_names=model_obj.get_output_names(),
                            dynamic_axes=model_obj.get_dynamic_axes(),
                        )
                    del model
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    logger.info("Found cached model: %s", onnx_path)

                # Generate fp32 optimized model.
                # If final target is fp16 model, we save fp32 optimized model so that it is easy to tune
                # fp16 conversion. That could save a lot of time in developing.
                use_fp32_intermediate = save_fp32_intermediate_model and self.model_config[model_name].fp16
                onnx_fp32_path = onnx_path
                if use_fp32_intermediate:
                    onnx_fp32_path = self.get_onnx_path(model_name, engine_dir, opt=True, suffix=".fp32")
                    if not os.path.exists(onnx_fp32_path):
                        print("------")
                        logger.info("Generating optimized model: %s", onnx_fp32_path)
                        model_obj.optimize_ort(
                            onnx_path,
                            onnx_fp32_path,
                            to_fp16=False,
                            fp32_op_list=self.model_config[model_name].force_fp32_ops,
                            optimize_by_ort=self.model_config[model_name].optimize_by_ort,
                            tmp_dir=self.get_model_dir(model_name, tmp_dir, opt=False, suffix=".fp32", create=False),
                        )
                    else:
                        logger.info("Found cached optimized model: %s", onnx_fp32_path)

                # Generate the final optimized model.
                if not os.path.exists(onnx_opt_path):
                    print("------")
                    logger.info("Generating optimized model: %s", onnx_opt_path)

                    # When there is fp32 intermediate optimized model, this will just convert model from fp32 to fp16.
                    optimize_by_ort = False if use_fp32_intermediate else self.model_config[model_name].optimize_by_ort

                    model_obj.optimize_ort(
                        onnx_fp32_path,
                        onnx_opt_path,
                        to_fp16=self.model_config[model_name].fp16,
                        fp32_op_list=self.model_config[model_name].force_fp32_ops,
                        optimize_by_ort=optimize_by_ort,
                        optimize_by_fusion=not use_fp32_intermediate,
                        tmp_dir=self.get_model_dir(model_name, tmp_dir, opt=False, suffix=".ort", create=False),
                    )
                else:
                    logger.info("Found cached optimized model: %s", onnx_opt_path)
        self.enable_torch_spda()

        built_engines = {}
        for model_name in self.models:
            if model_name == "vae" and self.vae_torch_fallback:
                continue

            onnx_opt_path = self.optimized_onnx_path(engine_dir, model_name)
            use_cuda_graph = self.model_config[model_name].use_cuda_graph

            engine = OrtCudaEngine(
                onnx_opt_path,
                device_id=device_id,
                enable_cuda_graph=use_cuda_graph,
                disable_optimization=False,
                max_cuda_graphs=max_cuda_graphs,
            )

            logger.info("%s options for %s: %s", engine.provider, model_name, engine.provider_options)
            built_engines[model_name] = engine

        self.engines = built_engines

    def run_engine(self, model_name, feed_dict):
        return self.engines[model_name].infer(feed_dict)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\reductions.py ===
from types import FunctionType

from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.domains import ZZ, QQ

from .utilities import _get_intermediate_simp, _iszero, _dotprodsimp, _simplify
from .determinant import _find_reasonable_pivot


def _row_reduce_list(mat, rows, cols, one, iszerofunc, simpfunc,
                normalize_last=True, normalize=True, zero_above=True):
    """Row reduce a flat list representation of a matrix and return a tuple
    (rref_matrix, pivot_cols, swaps) where ``rref_matrix`` is a flat list,
    ``pivot_cols`` are the pivot columns and ``swaps`` are any row swaps that
    were used in the process of row reduction.

    Parameters
    ==========

    mat : list
        list of matrix elements, must be ``rows`` * ``cols`` in length

    rows, cols : integer
        number of rows and columns in flat list representation

    one : SymPy object
        represents the value one, from ``Matrix.one``

    iszerofunc : determines if an entry can be used as a pivot

    simpfunc : used to simplify elements and test if they are
        zero if ``iszerofunc`` returns `None`

    normalize_last : indicates where all row reduction should
        happen in a fraction-free manner and then the rows are
        normalized (so that the pivots are 1), or whether
        rows should be normalized along the way (like the naive
        row reduction algorithm)

    normalize : whether pivot rows should be normalized so that
        the pivot value is 1

    zero_above : whether entries above the pivot should be zeroed.
        If ``zero_above=False``, an echelon matrix will be returned.
    """

    def get_col(i):
        return mat[i::cols]

    def row_swap(i, j):
        mat[i*cols:(i + 1)*cols], mat[j*cols:(j + 1)*cols] = \
            mat[j*cols:(j + 1)*cols], mat[i*cols:(i + 1)*cols]

    def cross_cancel(a, i, b, j):
        """Does the row op row[i] = a*row[i] - b*row[j]"""
        q = (j - i)*cols
        for p in range(i*cols, (i + 1)*cols):
            mat[p] = isimp(a*mat[p] - b*mat[p + q])

    isimp = _get_intermediate_simp(_dotprodsimp)
    piv_row, piv_col = 0, 0
    pivot_cols = []
    swaps = []

    # use a fraction free method to zero above and below each pivot
    while piv_col < cols and piv_row < rows:
        pivot_offset, pivot_val, \
        assumed_nonzero, newly_determined = _find_reasonable_pivot(
                get_col(piv_col)[piv_row:], iszerofunc, simpfunc)

        # _find_reasonable_pivot may have simplified some things
        # in the process.  Let's not let them go to waste
        for (offset, val) in newly_determined:
            offset += piv_row
            mat[offset*cols + piv_col] = val

        if pivot_offset is None:
            piv_col += 1
            continue

        pivot_cols.append(piv_col)
        if pivot_offset != 0:
            row_swap(piv_row, pivot_offset + piv_row)
            swaps.append((piv_row, pivot_offset + piv_row))

        # if we aren't normalizing last, we normalize
        # before we zero the other rows
        if normalize_last is False:
            i, j = piv_row, piv_col
            mat[i*cols + j] = one
            for p in range(i*cols + j + 1, (i + 1)*cols):
                mat[p] = isimp(mat[p] / pivot_val)
            # after normalizing, the pivot value is 1
            pivot_val = one

        # zero above and below the pivot
        for row in range(rows):
            # don't zero our current row
            if row == piv_row:
                continue
            # don't zero above the pivot unless we're told.
            if zero_above is False and row < piv_row:
                continue
            # if we're already a zero, don't do anything
            val = mat[row*cols + piv_col]
            if iszerofunc(val):
                continue

            cross_cancel(pivot_val, row, val, piv_row)
        piv_row += 1

    # normalize each row
    if normalize_last is True and normalize is True:
        for piv_i, piv_j in enumerate(pivot_cols):
            pivot_val = mat[piv_i*cols + piv_j]
            mat[piv_i*cols + piv_j] = one
            for p in range(piv_i*cols + piv_j + 1, (piv_i + 1)*cols):
                mat[p] = isimp(mat[p] / pivot_val)

    return mat, tuple(pivot_cols), tuple(swaps)


# This functions is a candidate for caching if it gets implemented for matrices.
def _row_reduce(M, iszerofunc, simpfunc, normalize_last=True,
                normalize=True, zero_above=True):

    mat, pivot_cols, swaps = _row_reduce_list(list(M), M.rows, M.cols, M.one,
            iszerofunc, simpfunc, normalize_last=normalize_last,
            normalize=normalize, zero_above=zero_above)

    return M._new(M.rows, M.cols, mat), pivot_cols, swaps


def _is_echelon(M, iszerofunc=_iszero):
    """Returns `True` if the matrix is in echelon form. That is, all rows of
    zeros are at the bottom, and below each leading non-zero in a row are
    exclusively zeros."""

    if M.rows <= 0 or M.cols <= 0:
        return True

    zeros_below = all(iszerofunc(t) for t in M[1:, 0])

    if iszerofunc(M[0, 0]):
        return zeros_below and _is_echelon(M[:, 1:], iszerofunc)

    return zeros_below and _is_echelon(M[1:, 1:], iszerofunc)


def _echelon_form(M, iszerofunc=_iszero, simplify=False, with_pivots=False):
    """Returns a matrix row-equivalent to ``M`` that is in echelon form. Note
    that echelon form of a matrix is *not* unique, however, properties like the
    row space and the null space are preserved.

    Examples
    ========

    >>> from sympy import Matrix
    >>> M = Matrix([[1, 2], [3, 4]])
    >>> M.echelon_form()
    Matrix([
    [1,  2],
    [0, -2]])
    """

    simpfunc = simplify if isinstance(simplify, FunctionType) else _simplify

    mat, pivots, _ = _row_reduce(M, iszerofunc, simpfunc,
            normalize_last=True, normalize=False, zero_above=False)

    if with_pivots:
        return mat, pivots

    return mat


# This functions is a candidate for caching if it gets implemented for matrices.
def _rank(M, iszerofunc=_iszero, simplify=False):
    """Returns the rank of a matrix.

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.abc import x
    >>> m = Matrix([[1, 2], [x, 1 - 1/x]])
    >>> m.rank()
    2
    >>> n = Matrix(3, 3, range(1, 10))
    >>> n.rank()
    2
    """

    def _permute_complexity_right(M, iszerofunc):
        """Permute columns with complicated elements as
        far right as they can go.  Since the ``sympy`` row reduction
        algorithms start on the left, having complexity right-shifted
        speeds things up.

        Returns a tuple (mat, perm) where perm is a permutation
        of the columns to perform to shift the complex columns right, and mat
        is the permuted matrix."""

        def complexity(i):
            # the complexity of a column will be judged by how many
            # element's zero-ness cannot be determined
            return sum(1 if iszerofunc(e) is None else 0 for e in M[:, i])

        complex = [(complexity(i), i) for i in range(M.cols)]
        perm    = [j for (i, j) in sorted(complex)]

        return (M.permute(perm, orientation='cols'), perm)

    simpfunc = simplify if isinstance(simplify, FunctionType) else _simplify

    # for small matrices, we compute the rank explicitly
    # if is_zero on elements doesn't answer the question
    # for small matrices, we fall back to the full routine.
    if M.rows <= 0 or M.cols <= 0:
        return 0

    if M.rows <= 1 or M.cols <= 1:
        zeros = [iszerofunc(x) for x in M]

        if False in zeros:
            return 1

    if M.rows == 2 and M.cols == 2:
        zeros = [iszerofunc(x) for x in M]

        if False not in zeros and None not in zeros:
            return 0

        d = M.det()

        if iszerofunc(d) and False in zeros:
            return 1
        if iszerofunc(d) is False:
            return 2

    mat, _       = _permute_complexity_right(M, iszerofunc=iszerofunc)
    _, pivots, _ = _row_reduce(mat, iszerofunc, simpfunc, normalize_last=True,
            normalize=False, zero_above=False)

    return len(pivots)


def _to_DM_ZZ_QQ(M):
    # We have to test for _rep here because there are tests that otherwise fail
    # with e.g. "AttributeError: 'SubspaceOnlyMatrix' object has no attribute
    # '_rep'." There is almost certainly no value in such tests. The
    # presumption seems to be that someone could create a new class by
    # inheriting some of the Matrix classes and not the full set that is used
    # by the standard Matrix class but if anyone tried that it would fail in
    # many ways.
    if not hasattr(M, '_rep'):
        return None

    rep = M._rep
    K = rep.domain

    if K.is_ZZ:
        return rep
    elif K.is_QQ:
        try:
            return rep.convert_to(ZZ)
        except CoercionFailed:
            return rep
    else:
        if not all(e.is_Rational for e in M):
            return None
        try:
            return rep.convert_to(ZZ)
        except CoercionFailed:
            return rep.convert_to(QQ)


def _rref_dm(dM):
    """Compute the reduced row echelon form of a DomainMatrix."""
    K = dM.domain

    if K.is_ZZ:
        dM_rref, den, pivots = dM.rref_den(keep_domain=False)
        dM_rref = dM_rref.to_field() / den
    elif K.is_QQ:
        dM_rref, pivots = dM.rref()
    else:
        assert False  # pragma: no cover

    M_rref = dM_rref.to_Matrix()

    return M_rref, pivots


def _rref(M, iszerofunc=_iszero, simplify=False, pivots=True,
        normalize_last=True):
    """Return reduced row-echelon form of matrix and indices
    of pivot vars.

    Parameters
    ==========

    iszerofunc : Function
        A function used for detecting whether an element can
        act as a pivot.  ``lambda x: x.is_zero`` is used by default.

    simplify : Function
        A function used to simplify elements when looking for a pivot.
        By default SymPy's ``simplify`` is used.

    pivots : True or False
        If ``True``, a tuple containing the row-reduced matrix and a tuple
        of pivot columns is returned.  If ``False`` just the row-reduced
        matrix is returned.

    normalize_last : True or False
        If ``True``, no pivots are normalized to `1` until after all
        entries above and below each pivot are zeroed.  This means the row
        reduction algorithm is fraction free until the very last step.
        If ``False``, the naive row reduction procedure is used where
        each pivot is normalized to be `1` before row operations are
        used to zero above and below the pivot.

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.abc import x
    >>> m = Matrix([[1, 2], [x, 1 - 1/x]])
    >>> m.rref()
    (Matrix([
    [1, 0],
    [0, 1]]), (0, 1))
    >>> rref_matrix, rref_pivots = m.rref()
    >>> rref_matrix
    Matrix([
    [1, 0],
    [0, 1]])
    >>> rref_pivots
    (0, 1)

    ``iszerofunc`` can correct rounding errors in matrices with float
    values. In the following example, calling ``rref()`` leads to
    floating point errors, incorrectly row reducing the matrix.
    ``iszerofunc= lambda x: abs(x) < 1e-9`` sets sufficiently small numbers
    to zero, avoiding this error.

    >>> m = Matrix([[0.9, -0.1, -0.2, 0], [-0.8, 0.9, -0.4, 0], [-0.1, -0.8, 0.6, 0]])
    >>> m.rref()
    (Matrix([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0]]), (0, 1, 2))
    >>> m.rref(iszerofunc=lambda x:abs(x)<1e-9)
    (Matrix([
    [1, 0, -0.301369863013699, 0],
    [0, 1, -0.712328767123288, 0],
    [0, 0,         0,          0]]), (0, 1))

    Notes
    =====

    The default value of ``normalize_last=True`` can provide significant
    speedup to row reduction, especially on matrices with symbols.  However,
    if you depend on the form row reduction algorithm leaves entries
    of the matrix, set ``normalize_last=False``
    """
    # Try to use DomainMatrix for ZZ or QQ
    dM = _to_DM_ZZ_QQ(M)

    if dM is not None:
        # Use DomainMatrix for ZZ or QQ
        mat, pivot_cols = _rref_dm(dM)
    else:
        # Use the generic Matrix routine.
        if isinstance(simplify, FunctionType):
            simpfunc = simplify
        else:
            simpfunc = _simplify

        mat, pivot_cols, _ = _row_reduce(M, iszerofunc, simpfunc,
                normalize_last, normalize=True, zero_above=True)

    if pivots:
        return mat, pivot_cols
    else:
        return mat

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\constructor.py ===
"""Tools for constructing domains for expressions. """
from math import prod

from sympy.core import sympify
from sympy.core.evalf import pure_complex
from sympy.core.sorting import ordered
from sympy.polys.domains import ZZ, QQ, ZZ_I, QQ_I, EX
from sympy.polys.domains.complexfield import ComplexField
from sympy.polys.domains.realfield import RealField
from sympy.polys.polyoptions import build_options
from sympy.polys.polyutils import parallel_dict_from_basic
from sympy.utilities import public


def _construct_simple(coeffs, opt):
    """Handle simple domains, e.g.: ZZ, QQ, RR and algebraic domains. """
    rationals = floats = complexes = algebraics = False
    float_numbers = []

    if opt.extension is True:
        is_algebraic = lambda coeff: coeff.is_number and coeff.is_algebraic
    else:
        is_algebraic = lambda coeff: False

    for coeff in coeffs:
        if coeff.is_Rational:
            if not coeff.is_Integer:
                rationals = True
        elif coeff.is_Float:
            if algebraics:
                # there are both reals and algebraics -> EX
                return False
            else:
                floats = True
                float_numbers.append(coeff)
        else:
            is_complex = pure_complex(coeff)
            if is_complex:
                complexes = True
                x, y = is_complex
                if x.is_Rational and y.is_Rational:
                    if not (x.is_Integer and y.is_Integer):
                        rationals = True
                    continue
                else:
                    floats = True
                    if x.is_Float:
                        float_numbers.append(x)
                    if y.is_Float:
                        float_numbers.append(y)
            elif is_algebraic(coeff):
                if floats:
                    # there are both algebraics and reals -> EX
                    return False
                algebraics = True
            else:
                # this is a composite domain, e.g. ZZ[X], EX
                return None

    # Use the maximum precision of all coefficients for the RR or CC
    # precision
    max_prec = max(c._prec for c in float_numbers) if float_numbers else 53

    if algebraics:
        domain, result = _construct_algebraic(coeffs, opt)
    else:
        if floats and complexes:
            domain = ComplexField(prec=max_prec)
        elif floats:
            domain = RealField(prec=max_prec)
        elif rationals or opt.field:
            domain = QQ_I if complexes else QQ
        else:
            domain = ZZ_I if complexes else ZZ

        result = [domain.from_sympy(coeff) for coeff in coeffs]

    return domain, result


def _construct_algebraic(coeffs, opt):
    """We know that coefficients are algebraic so construct the extension. """
    from sympy.polys.numberfields import primitive_element

    exts = set()

    def build_trees(args):
        trees = []
        for a in args:
            if a.is_Rational:
                tree = ('Q', QQ.from_sympy(a))
            elif a.is_Add:
                tree = ('+', build_trees(a.args))
            elif a.is_Mul:
                tree = ('*', build_trees(a.args))
            else:
                tree = ('e', a)
                exts.add(a)
            trees.append(tree)
        return trees

    trees = build_trees(coeffs)
    exts = list(ordered(exts))

    g, span, H = primitive_element(exts, ex=True, polys=True)
    root = sum(s*ext for s, ext in zip(span, exts))

    domain, g = QQ.algebraic_field((g, root)), g.rep.to_list()

    exts_dom = [domain.dtype.from_list(h, g, QQ) for h in H]
    exts_map = dict(zip(exts, exts_dom))

    def convert_tree(tree):
        op, args = tree
        if op == 'Q':
            return domain.dtype.from_list([args], g, QQ)
        elif op == '+':
            return sum((convert_tree(a) for a in args), domain.zero)
        elif op == '*':
            return prod(convert_tree(a) for a in args)
        elif op == 'e':
            return exts_map[args]
        else:
            raise RuntimeError

    result = [convert_tree(tree) for tree in trees]

    return domain, result


def _construct_composite(coeffs, opt):
    """Handle composite domains, e.g.: ZZ[X], QQ[X], ZZ(X), QQ(X). """
    numers, denoms = [], []

    for coeff in coeffs:
        numer, denom = coeff.as_numer_denom()

        numers.append(numer)
        denoms.append(denom)

    polys, gens = parallel_dict_from_basic(numers + denoms)  # XXX: sorting
    if not gens:
        return None

    if opt.composite is None:
        if any(gen.is_number and gen.is_algebraic for gen in gens):
            return None # generators are number-like so lets better use EX

        all_symbols = set()

        for gen in gens:
            symbols = gen.free_symbols

            if all_symbols & symbols:
                return None # there could be algebraic relations between generators
            else:
                all_symbols |= symbols

    n = len(gens)
    k = len(polys)//2

    numers = polys[:k]
    denoms = polys[k:]

    if opt.field:
        fractions = True
    else:
        fractions, zeros = False, (0,)*n

        for denom in denoms:
            if len(denom) > 1 or zeros not in denom:
                fractions = True
                break

    coeffs = set()

    if not fractions:
        for numer, denom in zip(numers, denoms):
            denom = denom[zeros]

            for monom, coeff in numer.items():
                coeff /= denom
                coeffs.add(coeff)
                numer[monom] = coeff
    else:
        for numer, denom in zip(numers, denoms):
            coeffs.update(list(numer.values()))
            coeffs.update(list(denom.values()))

    rationals = floats = complexes = False
    float_numbers = []

    for coeff in coeffs:
        if coeff.is_Rational:
            if not coeff.is_Integer:
                rationals = True
        elif coeff.is_Float:
            floats = True
            float_numbers.append(coeff)
        else:
            is_complex = pure_complex(coeff)
            if is_complex is not None:
                complexes = True
                x, y = is_complex
                if x.is_Rational and y.is_Rational:
                    if not (x.is_Integer and y.is_Integer):
                        rationals = True
                else:
                    floats = True
                    if x.is_Float:
                        float_numbers.append(x)
                    if y.is_Float:
                        float_numbers.append(y)

    max_prec = max(c._prec for c in float_numbers) if float_numbers else 53

    if floats and complexes:
        ground = ComplexField(prec=max_prec)
    elif floats:
        ground = RealField(prec=max_prec)
    elif complexes:
        if rationals:
            ground = QQ_I
        else:
            ground = ZZ_I
    elif rationals:
        ground = QQ
    else:
        ground = ZZ

    result = []

    if not fractions:
        domain = ground.poly_ring(*gens)

        for numer in numers:
            for monom, coeff in numer.items():
                numer[monom] = ground.from_sympy(coeff)

            result.append(domain(numer))
    else:
        domain = ground.frac_field(*gens)

        for numer, denom in zip(numers, denoms):
            for monom, coeff in numer.items():
                numer[monom] = ground.from_sympy(coeff)

            for monom, coeff in denom.items():
                denom[monom] = ground.from_sympy(coeff)

            result.append(domain((numer, denom)))

    return domain, result


def _construct_expression(coeffs, opt):
    """The last resort case, i.e. use the expression domain. """
    domain, result = EX, []

    for coeff in coeffs:
        result.append(domain.from_sympy(coeff))

    return domain, result


@public
def construct_domain(obj, **args):
    """Construct a minimal domain for a list of expressions.

    Explanation
    ===========

    Given a list of normal SymPy expressions (of type :py:class:`~.Expr`)
    ``construct_domain`` will find a minimal :py:class:`~.Domain` that can
    represent those expressions. The expressions will be converted to elements
    of the domain and both the domain and the domain elements are returned.

    Parameters
    ==========

    obj: list or dict
        The expressions to build a domain for.

    **args: keyword arguments
        Options that affect the choice of domain.

    Returns
    =======

    (K, elements): Domain and list of domain elements
        The domain K that can represent the expressions and the list or dict
        of domain elements representing the same expressions as elements of K.

    Examples
    ========

    Given a list of :py:class:`~.Integer` ``construct_domain`` will return the
    domain :ref:`ZZ` and a list of integers as elements of :ref:`ZZ`.

    >>> from sympy import construct_domain, S
    >>> expressions = [S(2), S(3), S(4)]
    >>> K, elements = construct_domain(expressions)
    >>> K
    ZZ
    >>> elements
    [2, 3, 4]
    >>> type(elements[0])  # doctest: +SKIP
    <class 'int'>
    >>> type(expressions[0])
    <class 'sympy.core.numbers.Integer'>

    If there are any :py:class:`~.Rational` then :ref:`QQ` is returned
    instead.

    >>> construct_domain([S(1)/2, S(3)/4])
    (QQ, [1/2, 3/4])

    If there are symbols then a polynomial ring :ref:`K[x]` is returned.

    >>> from sympy import symbols
    >>> x, y = symbols('x, y')
    >>> construct_domain([2*x + 1, S(3)/4])
    (QQ[x], [2*x + 1, 3/4])
    >>> construct_domain([2*x + 1, y])
    (ZZ[x,y], [2*x + 1, y])

    If any symbols appear with negative powers then a rational function field
    :ref:`K(x)` will be returned.

    >>> construct_domain([y/x, x/(1 - y)])
    (ZZ(x,y), [y/x, -x/(y - 1)])

    Irrational algebraic numbers will result in the :ref:`EX` domain by
    default. The keyword argument ``extension=True`` leads to the construction
    of an algebraic number field :ref:`QQ(a)`.

    >>> from sympy import sqrt
    >>> construct_domain([sqrt(2)])
    (EX, [EX(sqrt(2))])
    >>> construct_domain([sqrt(2)], extension=True)  # doctest: +SKIP
    (QQ<sqrt(2)>, [ANP([1, 0], [1, 0, -2], QQ)])

    See also
    ========

    Domain
    Expr
    """
    opt = build_options(args)

    if hasattr(obj, '__iter__'):
        if isinstance(obj, dict):
            if not obj:
                monoms, coeffs = [], []
            else:
                monoms, coeffs = list(zip(*list(obj.items())))
        else:
            coeffs = obj
    else:
        coeffs = [obj]

    coeffs = list(map(sympify, coeffs))
    result = _construct_simple(coeffs, opt)

    if result is not None:
        if result is not False:
            domain, coeffs = result
        else:
            domain, coeffs = _construct_expression(coeffs, opt)
    else:
        if opt.composite is False:
            result = None
        else:
            result = _construct_composite(coeffs, opt)

        if result is not None:
            domain, coeffs = result
        else:
            domain, coeffs = _construct_expression(coeffs, opt)

    if hasattr(obj, '__iter__'):
        if isinstance(obj, dict):
            return domain, dict(list(zip(monoms, coeffs)))
        else:
            return domain, coeffs
    else:
        return domain, coeffs[0]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_core\_io_epoll.py ===
from __future__ import annotations

import contextlib
import select
import sys
from collections import defaultdict
from typing import TYPE_CHECKING, Literal

import attrs

from .. import _core
from ._io_common import wake_all
from ._run import Task, _public
from ._wakeup_socketpair import WakeupSocketpair

if TYPE_CHECKING:
    from typing_extensions import TypeAlias

    from .._core import Abort, RaiseCancelT
    from .._file_io import _HasFileNo


@attrs.define(eq=False)
class EpollWaiters:
    read_task: Task | None = None
    write_task: Task | None = None
    current_flags: int = 0


assert not TYPE_CHECKING or sys.platform == "linux"


EventResult: TypeAlias = "list[tuple[int, int]]"


@attrs.frozen(eq=False)
class _EpollStatistics:
    tasks_waiting_read: int
    tasks_waiting_write: int
    backend: Literal["epoll"] = attrs.field(init=False, default="epoll")


# Some facts about epoll
# ----------------------
#
# Internally, an epoll object is sort of like a WeakKeyDictionary where the
# keys are tuples of (fd number, file object). When you call epoll_ctl, you
# pass in an fd; that gets converted to an (fd number, file object) tuple by
# looking up the fd in the process's fd table at the time of the call. When an
# event happens on the file object, epoll_wait drops the file object part, and
# just returns the fd number in its event. So from the outside it looks like
# it's keeping a table of fds, but really it's a bit more complicated. This
# has some subtle consequences.
#
# In general, file objects inside the kernel are reference counted. Each entry
# in a process's fd table holds a strong reference to the corresponding file
# object, and most operations that use file objects take a temporary strong
# reference while they're working. So when you call close() on an fd, that
# might or might not cause the file object to be deallocated -- it depends on
# whether there are any other references to that file object. Some common ways
# this can happen:
#
# - after calling dup(), you have two fds in the same process referring to the
#   same file object. Even if you close one fd (= remove that entry from the
#   fd table), the file object will be kept alive by the other fd.
# - when calling fork(), the child inherits a copy of the parent's fd table,
#   so all the file objects get another reference. (But if the fork() is
#   followed by exec(), then all of the child's fds that have the CLOEXEC flag
#   set will be closed at that point.)
# - most syscalls that work on fds take a strong reference to the underlying
#   file object while they're using it. So there's one thread blocked in
#   read(fd), and then another thread calls close() on the last fd referring
#   to that object, the underlying file won't actually be closed until
#   after read() returns.
#
# However, epoll does *not* take a reference to any of the file objects in its
# interest set (that's what makes it similar to a WeakKeyDictionary). File
# objects inside an epoll interest set will be deallocated if all *other*
# references to them are closed. And when that happens, the epoll object will
# automatically deregister that file object and stop reporting events on it.
# So that's quite handy.
#
# But, what happens if we do this?
#
#   fd1 = open(...)
#   epoll_ctl(EPOLL_CTL_ADD, fd1, ...)
#   fd2 = dup(fd1)
#   close(fd1)
#
# In this case, the dup() keeps the underlying file object alive, so it
# remains registered in the epoll object's interest set, as the tuple (fd1,
# file object). But, fd1 no longer refers to this file object! You might think
# there was some magic to handle this, but unfortunately no; the consequences
# are totally predictable from what I said above:
#
# If any events occur on the file object, then epoll will report them as
# happening on fd1, even though that doesn't make sense.
#
# Perhaps we would like to deregister fd1 to stop getting nonsensical events.
# But how? When we call epoll_ctl, we have to pass an fd number, which will
# get expanded to an (fd number, file object) tuple. We can't pass fd1,
# because when epoll_ctl tries to look it up, it won't find our file object.
# And we can't pass fd2, because that will get expanded to (fd2, file object),
# which is a different lookup key. In fact, it's *impossible* to de-register
# this fd!
#
# We could even have fd1 get assigned to another file object, and then we can
# have multiple keys registered simultaneously using the same fd number, like:
# (fd1, file object 1), (fd1, file object 2). And if events happen on either
# file object, then epoll will happily report that something happened to
# "fd1".
#
# Now here's what makes this especially nasty: suppose the old file object
# becomes, say, readable. That means that every time we call epoll_wait, it
# will return immediately to tell us that "fd1" is readable. Normally, we
# would handle this by de-registering fd1, waking up the corresponding call to
# wait_readable, then the user will call read() or recv() or something, and
# we're fine. But if this happens on a stale fd where we can't remove the
# registration, then we might get stuck in a state where epoll_wait *always*
# returns immediately, so our event loop becomes unable to sleep, and now our
# program is burning 100% of the CPU doing nothing, with no way out.
#
#
# What does this mean for Trio?
# -----------------------------
#
# Since we don't control the user's code, we have no way to guarantee that we
# don't get stuck with stale fd's in our epoll interest set. For example, a
# user could call wait_readable(fd) in one task, and then while that's
# running, they might close(fd) from another task. In this situation, they're
# *supposed* to call notify_closing(fd) to let us know what's happening, so we
# can interrupt the wait_readable() call and avoid getting into this mess. And
# that's the only thing that can possibly work correctly in all cases. But
# sometimes user code has bugs. So if this does happen, we'd like to degrade
# gracefully, and survive without corrupting Trio's internal state or
# otherwise causing the whole program to explode messily.
#
# Our solution: we always use EPOLLONESHOT. This way, we might get *one*
# spurious event on a stale fd, but then epoll will automatically silence it
# until we explicitly say that we want more events... and if we have a stale
# fd, then we actually can't re-enable it! So we can't get stuck in an
# infinite busy-loop. If there's a stale fd hanging around, then it might
# cause a spurious `BusyResourceError`, or cause one wait_* call to return
# before it should have... but in general, the wait_* functions are allowed to
# have some spurious wakeups; the user code will just attempt the operation,
# get EWOULDBLOCK, and call wait_* again. And the program as a whole will
# survive, any exceptions will propagate, etc.
#
# As a bonus, EPOLLONESHOT also saves us having to explicitly deregister fds
# on the normal wakeup path, so it's a bit more efficient in general.
#
# However, EPOLLONESHOT has a few trade-offs to consider:
#
# First, you can't combine EPOLLONESHOT with EPOLLEXCLUSIVE. This is a bit sad
# in one somewhat rare case: if you have a multi-process server where a group
# of processes all share the same listening socket, then EPOLLEXCLUSIVE can be
# used to avoid "thundering herd" problems when a new connection comes in. But
# this isn't too bad. It's not clear if EPOLLEXCLUSIVE even works for us
# anyway:
#
#   https://stackoverflow.com/questions/41582560/how-does-epolls-epollexclusive-mode-interact-with-level-triggering
#
# And it's not clear that EPOLLEXCLUSIVE is a great approach either:
#
#   https://blog.cloudflare.com/the-sad-state-of-linux-socket-balancing/
#
# And if we do need to support this, we could always add support through some
# more-specialized API in the future. So this isn't a blocker to using
# EPOLLONESHOT.
#
# Second, EPOLLONESHOT does not actually *deregister* the fd after delivering
# an event (EPOLL_CTL_DEL). Instead, it keeps the fd registered, but
# effectively does an EPOLL_CTL_MOD to set the fd's interest flags to
# all-zeros. So we could still end up with an fd hanging around in the
# interest set for a long time, even if we're not using it.
#
# Fortunately, this isn't a problem, because it's only a weak reference – if
# we have a stale fd that's been silenced by EPOLLONESHOT, then it wastes a
# tiny bit of kernel memory remembering this fd that can never be revived, but
# when the underlying file object is eventually closed, that memory will be
# reclaimed. So that's OK.
#
# The other issue is that when someone calls wait_*, using EPOLLONESHOT means
# that if we have ever waited for this fd before, we have to use EPOLL_CTL_MOD
# to re-enable it; but if it's a new fd, we have to use EPOLL_CTL_ADD. How do
# we know which one to use? There's no reasonable way to track which fds are
# currently registered -- remember, we're assuming the user might have gone
# and rearranged their fds without telling us!
#
# Fortunately, this also has a simple solution: if we wait on a socket or
# other fd once, then we'll probably wait on it lots of times. And the epoll
# object itself knows which fds it already has registered. So when an fd comes
# in, we optimistically assume that it's been waited on before, and try doing
# EPOLL_CTL_MOD. And if that fails with an ENOENT error, then we try again
# with EPOLL_CTL_ADD.
#
# So that's why this code is the way it is. And now you know more than you
# wanted to about how epoll works.


@attrs.define(eq=False)
class EpollIOManager:
    # Using lambda here because otherwise crash on import with gevent monkey patching
    # See https://github.com/python-trio/trio/issues/2848
    _epoll: select.epoll = attrs.Factory(lambda: select.epoll())
    # {fd: EpollWaiters}
    _registered: defaultdict[int, EpollWaiters] = attrs.Factory(
        lambda: defaultdict(EpollWaiters),
    )
    _force_wakeup: WakeupSocketpair = attrs.Factory(WakeupSocketpair)
    _force_wakeup_fd: int | None = None

    def __attrs_post_init__(self) -> None:
        self._epoll.register(self._force_wakeup.wakeup_sock, select.EPOLLIN)
        self._force_wakeup_fd = self._force_wakeup.wakeup_sock.fileno()

    def statistics(self) -> _EpollStatistics:
        tasks_waiting_read = 0
        tasks_waiting_write = 0
        for waiter in self._registered.values():
            if waiter.read_task is not None:
                tasks_waiting_read += 1
            if waiter.write_task is not None:
                tasks_waiting_write += 1
        return _EpollStatistics(
            tasks_waiting_read=tasks_waiting_read,
            tasks_waiting_write=tasks_waiting_write,
        )

    def close(self) -> None:
        self._epoll.close()
        self._force_wakeup.close()

    def force_wakeup(self) -> None:
        self._force_wakeup.wakeup_thread_and_signal_safe()

    # Return value must be False-y IFF the timeout expired, NOT if any I/O
    # happened or force_wakeup was called. Otherwise it can be anything; gets
    # passed straight through to process_events.
    def get_events(self, timeout: float) -> EventResult:
        # max_events must be > 0 or epoll gets cranky
        # accessing self._registered from a thread looks dangerous, but it's
        # OK because it doesn't matter if our value is a little bit off.
        max_events = max(1, len(self._registered))
        return self._epoll.poll(timeout, max_events)

    def process_events(self, events: EventResult) -> None:
        for fd, flags in events:
            if fd == self._force_wakeup_fd:
                self._force_wakeup.drain()
                continue
            waiters = self._registered[fd]
            # EPOLLONESHOT always clears the flags when an event is delivered
            waiters.current_flags = 0
            # Clever hack stolen from selectors.EpollSelector: an event
            # with EPOLLHUP or EPOLLERR flags wakes both readers and
            # writers.
            if flags & ~select.EPOLLIN and waiters.write_task is not None:
                _core.reschedule(waiters.write_task)
                waiters.write_task = None
            if flags & ~select.EPOLLOUT and waiters.read_task is not None:
                _core.reschedule(waiters.read_task)
                waiters.read_task = None
            self._update_registrations(fd)

    def _update_registrations(self, fd: int) -> None:
        waiters = self._registered[fd]
        wanted_flags = 0
        if waiters.read_task is not None:
            wanted_flags |= select.EPOLLIN
        if waiters.write_task is not None:
            wanted_flags |= select.EPOLLOUT
        if wanted_flags != waiters.current_flags:
            try:
                try:
                    # First try EPOLL_CTL_MOD
                    self._epoll.modify(fd, wanted_flags | select.EPOLLONESHOT)
                except OSError:
                    # If that fails, it might be a new fd; try EPOLL_CTL_ADD
                    self._epoll.register(fd, wanted_flags | select.EPOLLONESHOT)
                waiters.current_flags = wanted_flags
            except OSError as exc:
                # If everything fails, probably it's a bad fd, e.g. because
                # the fd was closed behind our back. In this case we don't
                # want to try to unregister the fd, because that will probably
                # fail too. Just clear our state and wake everyone up.
                del self._registered[fd]
                # This could raise (in case we're calling this inside one of
                # the to-be-woken tasks), so we have to do it last.
                wake_all(waiters, exc)
                return
        if not wanted_flags:
            del self._registered[fd]

    async def _epoll_wait(self, fd: int | _HasFileNo, attr_name: str) -> None:
        if not isinstance(fd, int):
            fd = fd.fileno()
        waiters = self._registered[fd]
        if getattr(waiters, attr_name) is not None:
            raise _core.BusyResourceError(
                "another task is already reading / writing this fd",
            )
        setattr(waiters, attr_name, _core.current_task())
        self._update_registrations(fd)

        def abort(_: RaiseCancelT) -> Abort:
            setattr(waiters, attr_name, None)
            self._update_registrations(fd)
            return _core.Abort.SUCCEEDED

        await _core.wait_task_rescheduled(abort)

    @_public
    async def wait_readable(self, fd: int | _HasFileNo) -> None:
        """Block until the kernel reports that the given object is readable.

        On Unix systems, ``fd`` must either be an integer file descriptor,
        or else an object with a ``.fileno()`` method which returns an
        integer file descriptor. Any kind of file descriptor can be passed,
        though the exact semantics will depend on your kernel. For example,
        this probably won't do anything useful for on-disk files.

        On Windows systems, ``fd`` must either be an integer ``SOCKET``
        handle, or else an object with a ``.fileno()`` method which returns
        an integer ``SOCKET`` handle. File descriptors aren't supported,
        and neither are handles that refer to anything besides a
        ``SOCKET``.

        :raises trio.BusyResourceError:
            if another task is already waiting for the given socket to
            become readable.
        :raises trio.ClosedResourceError:
            if another task calls :func:`notify_closing` while this
            function is still working.
        """
        await self._epoll_wait(fd, "read_task")

    @_public
    async def wait_writable(self, fd: int | _HasFileNo) -> None:
        """Block until the kernel reports that the given object is writable.

        See `wait_readable` for the definition of ``fd``.

        :raises trio.BusyResourceError:
            if another task is already waiting for the given socket to
            become writable.
        :raises trio.ClosedResourceError:
            if another task calls :func:`notify_closing` while this
            function is still working.
        """
        await self._epoll_wait(fd, "write_task")

    @_public
    def notify_closing(self, fd: int | _HasFileNo) -> None:
        """Notify waiters of the given object that it will be closed.

        Call this before closing a file descriptor (on Unix) or socket (on
        Windows). This will cause any `wait_readable` or `wait_writable`
        calls on the given object to immediately wake up and raise
        `~trio.ClosedResourceError`.

        This doesn't actually close the object – you still have to do that
        yourself afterwards. Also, you want to be careful to make sure no
        new tasks start waiting on the object in between when you call this
        and when it's actually closed. So to close something properly, you
        usually want to do these steps in order:

        1. Explicitly mark the object as closed, so that any new attempts
           to use it will abort before they start.
        2. Call `notify_closing` to wake up any already-existing users.
        3. Actually close the object.

        It's also possible to do them in a different order if that's more
        convenient, *but only if* you make sure not to have any checkpoints in
        between the steps. This way they all happen in a single atomic
        step, so other tasks won't be able to tell what order they happened
        in anyway.
        """
        if not isinstance(fd, int):
            fd = fd.fileno()
        wake_all(
            self._registered[fd],
            _core.ClosedResourceError("another task closed this fd"),
        )
        del self._registered[fd]
        with contextlib.suppress(OSError, ValueError):
            self._epoll.unregister(fd)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\convert_to_packing_mode.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
import logging
import os

import coloredlogs
from constants import (
    AttentionInputIDs,
    AttentionOutputIDs,
    MultiHeadAttentionInputIDs,
    MultiHeadAttentionOutputIDs,
    Operators,
)
from onnx import helper, load_model
from onnx_model import NodeProto, OnnxModel
from shape_infer_helper import SymbolicShapeInferenceHelper

logger = logging.getLogger(__name__)


class PackingAttentionBase:
    def __init__(self, model: OnnxModel, attention_op_type: str):
        self.model: OnnxModel = model
        self.nodes_to_remove: list = []
        self.nodes_to_add: list = []
        self.prune_graph: bool = False
        self.node_name_to_graph_name: dict = {}
        self.this_graph_name: str = self.model.model.graph.name
        self.attention_op_type = attention_op_type
        self.attention_nodes = self.model.get_nodes_by_op_type(attention_op_type)

    def _try_getting_attention_mask(self) -> str | None:
        mask_index = (
            AttentionInputIDs.MASK_INDEX
            if self.attention_op_type == Operators.ATTENTION
            else MultiHeadAttentionInputIDs.KEY_PADDING_MASK
        )
        first_attention_node = self._try_getting_first_attention()
        # check if attention has mask
        if not first_attention_node or len(first_attention_node.input) <= mask_index:
            return None

        attention_mask = first_attention_node.input[mask_index]

        # check if all attention nodes have same mask
        for node in self.attention_nodes:
            if len(node.input) <= mask_index or node.input[mask_index] != attention_mask:
                return None

        return attention_mask

    def _try_getting_first_attention(self) -> NodeProto | None:
        if len(self.attention_nodes) <= 0:
            return None

        return self.attention_nodes[0]

    def _try_getting_last_layernorm(self) -> NodeProto | None:
        last_layernorm_node = None
        for node in self.model.nodes():
            if node.op_type == Operators.LAYERNORM or node.op_type == Operators.SKIPLAYERNORM:
                last_layernorm_node = node
        return last_layernorm_node

    def _are_attentions_supported(self) -> bool:
        raise NotImplementedError()

    def _insert_removepadding_node(self, inputs: list[str], outputs: list[str]) -> None:
        new_node = helper.make_node(
            Operators.REMOVEPADDING,
            inputs=inputs,
            outputs=outputs,
            name=self.model.create_node_name(Operators.REMOVEPADDING),
        )

        new_node.domain = "com.microsoft"
        self.nodes_to_add.append(new_node)
        self.node_name_to_graph_name[new_node.name] = self.this_graph_name

    def _insert_restorepadding_node(self, inputs: list[str], outputs: list[str]) -> None:
        new_node = helper.make_node(
            Operators.RESTOREPADDING,
            inputs=inputs,
            outputs=outputs,
            name=self.model.create_node_name(Operators.RESTOREPADDING),
        )

        new_node.domain = "com.microsoft"
        self.nodes_to_add.append(new_node)
        self.node_name_to_graph_name[new_node.name] = self.this_graph_name

    def _replace_attention_with_packing_attention(self, token_offset: str, cumulative_sequence_length: str) -> None:
        raise NotImplementedError()

    def _get_input_to_remove_padding(self, first_attention_node) -> str | None:
        if self.attention_op_type == Operators.ATTENTION:
            return first_attention_node.input[AttentionInputIDs.INPUT]
        return None

    def convert(self, use_symbolic_shape_infer: bool = True) -> None:
        logger.debug("start converting to packing model...")

        if not self._are_attentions_supported():
            return

        attention_mask = self._try_getting_attention_mask()
        if not attention_mask:
            return

        first_attention_node = self._try_getting_first_attention()
        last_layernorm_node = self._try_getting_last_layernorm()
        if not last_layernorm_node:
            return

        # insert RemovePadding
        input_to_remove_padding = self._get_input_to_remove_padding(first_attention_node)
        if not input_to_remove_padding:
            return

        output_without_padding = input_to_remove_padding + "_no_padding"
        token_offset = input_to_remove_padding + "_token_offset"
        cumulated_seq_len = input_to_remove_padding + "_cumulated_seq_len"
        max_seq_len = input_to_remove_padding + "_max_seq_len"
        self._insert_removepadding_node(
            [input_to_remove_padding, attention_mask],
            [output_without_padding, token_offset, cumulated_seq_len, max_seq_len],
        )
        self.model.replace_input_of_all_nodes(input_to_remove_padding, output_without_padding)
        logger.debug("inserted RemovePadding before Attention")

        # insert RestorePadding
        restorepadding_input = last_layernorm_node.output[0] + "_restore_input"
        self._insert_restorepadding_node([restorepadding_input, token_offset], [last_layernorm_node.output[0]])
        self.model.replace_output_of_all_nodes(last_layernorm_node.output[0], restorepadding_input)
        logger.debug(f"inserted RestorePadding after last {last_layernorm_node.op_type} layer")

        # insert PackedAttention
        self._replace_attention_with_packing_attention(token_offset, cumulated_seq_len)
        logger.debug(f"replaced {self.attention_op_type} with Packed{self.attention_op_type}")

        self.model.remove_nodes(self.nodes_to_remove)
        self.model.add_nodes(self.nodes_to_add, self.node_name_to_graph_name)

        if self.prune_graph:
            self.model.prune_graph()
        elif self.nodes_to_remove or self.nodes_to_add:
            self.model.update_graph()
        self.model.clean_shape_infer()
        if use_symbolic_shape_infer:
            # Use symbolic shape inference since custom operators (like Gelu, SkipLayerNormalization etc)
            # are not recognized by onnx shape inference.
            shape_infer_helper = SymbolicShapeInferenceHelper(self.model.model, verbose=0)
            inferred_model = shape_infer_helper.infer_shapes(self.model.model, auto_merge=True, guess_output_rank=False)
            if inferred_model:
                self.model.model = inferred_model


class PackingAttention(PackingAttentionBase):
    def __init__(self, model: OnnxModel):
        super().__init__(model, Operators.ATTENTION)

    def _are_attentions_supported(self) -> bool:
        for node in self.attention_nodes:
            if OnnxModel.get_node_attribute(node, "past_present_share_buffer") is not None:
                return False
            if OnnxModel.get_node_attribute(node, "do_rotary") is not None:
                return False
            unidirection_attr = OnnxModel.get_node_attribute(node, "unidirectional")
            if unidirection_attr is not None and unidirection_attr != 0:
                return False
            if len(node.input) > AttentionInputIDs.PAST and not node.input[AttentionInputIDs.PAST]:
                return False
            if (
                len(node.input) > AttentionInputIDs.PAST_SEQUENCE_LENGTH
                and not node.input[AttentionInputIDs.PAST_SEQUENCE_LENGTH]
            ):
                return False
        return True

    def _replace_attention_with_packing_attention(self, token_offset: str, cumulative_sequence_length: str) -> None:
        for attention in self.attention_nodes:
            attention_bias = (
                attention.input[AttentionInputIDs.ATTENTION_BIAS]
                if len(attention.input) > AttentionInputIDs.ATTENTION_BIAS
                else ""
            )
            packed_attention = helper.make_node(
                Operators.PACKEDATTENTION,
                inputs=[
                    attention.input[AttentionInputIDs.INPUT],
                    attention.input[AttentionInputIDs.WEIGHTS],
                    attention.input[AttentionInputIDs.BIAS],
                    token_offset,
                    cumulative_sequence_length,
                    attention_bias,
                ],
                outputs=[attention.output[AttentionOutputIDs.OUTPUT]],
                name=self.model.create_node_name(Operators.PACKEDATTENTION),
            )

            attributes = []
            for attr in attention.attribute:
                if attr.name in ["num_heads", "qkv_hidden_sizes", "scale"]:
                    attributes.append(attr)

            packed_attention.attribute.extend(attributes)
            packed_attention.domain = "com.microsoft"
            self.nodes_to_add.append(packed_attention)
            self.nodes_to_remove.append(attention)
            self.node_name_to_graph_name[packed_attention.name] = self.this_graph_name

        logger.info("Converted %d Attention nodes to PackedAttention.", len(self.attention_nodes))


class PackingMultiHeadAttention(PackingAttentionBase):
    def __init__(self, model: OnnxModel):
        super().__init__(model, Operators.MULTI_HEAD_ATTENTION)

    def _check_empty_input(self, node, index: int, name: str):
        """Check a node does not have given input."""
        if len(node.input) > index:
            if len(node.input[index]) > 0:
                logger.error(f"node input {index} ({name}) is not supported in PackedMultiHeadAttention: {node}")
                return False
        return True

    def _check_empty_output(self, node, index: int, name: str):
        """Check a node does not have given input."""
        if len(node.output) > index:
            if len(node.output[index]) > 0:
                logger.error(f"node output {index} ({name}) is not supported in PackedMultiHeadAttention: {node}")
                return False
        return True

    def _are_attentions_supported(self) -> bool:
        for node in self.attention_nodes:
            for attr in node.attribute:
                if attr.name not in ["num_heads", "mask_filter_value", "scale"]:
                    logger.error(f"node attribute {attr.name} is not supported in PackedMultiHeadAttention: {node}")
                    return False

            if node.input[MultiHeadAttentionInputIDs.KEY] and not node.input[MultiHeadAttentionInputIDs.VALUE]:
                logger.error("packed kv format is not supported in PackedMultiHeadAttention")
                return False

            if not (
                self._check_empty_input(node, MultiHeadAttentionInputIDs.PAST_KEY, "past_key")
                and self._check_empty_input(node, MultiHeadAttentionInputIDs.PAST_VALUE, "past_key")
                and self._check_empty_output(node, MultiHeadAttentionOutputIDs.PRESENT_KEY, "present_key")
                and self._check_empty_output(node, MultiHeadAttentionOutputIDs.PRESENT_VALUE, "present_key")
            ):
                return False

        return True

    def _replace_attention_with_packing_attention(self, token_offset: str, cumulative_sequence_length: str) -> None:
        gated_relative_pos_bias_count = 0
        for mha in self.attention_nodes:
            attention_bias = (
                mha.input[MultiHeadAttentionInputIDs.ATTENTION_BIAS]
                if len(mha.input) > MultiHeadAttentionInputIDs.ATTENTION_BIAS
                else ""
            )
            packed_mha = helper.make_node(
                Operators.PACKED_MULTI_HEAD_ATTENTION,
                inputs=[
                    mha.input[MultiHeadAttentionInputIDs.QUERY],
                    mha.input[MultiHeadAttentionInputIDs.KEY],
                    mha.input[MultiHeadAttentionInputIDs.VALUE],
                    mha.input[MultiHeadAttentionInputIDs.BIAS],
                    token_offset,
                    cumulative_sequence_length,
                    attention_bias,
                ],
                outputs=[mha.output[MultiHeadAttentionOutputIDs.OUTPUT]],
                name=self.model.create_node_name(Operators.PACKED_MULTI_HEAD_ATTENTION),
            )

            attributes = []
            for attr in mha.attribute:
                if attr.name in ["num_heads", "mask_filter_value", "scale"]:
                    attributes.append(attr)

            packed_mha.attribute.extend(attributes)
            packed_mha.domain = "com.microsoft"
            self.nodes_to_add.append(packed_mha)
            self.nodes_to_remove.append(mha)
            self.node_name_to_graph_name[packed_mha.name] = self.this_graph_name

            # Append token_offset input to GatedRelativePositionBias
            if attention_bias:
                rel_pos_bias_node = self.model.get_parent(mha, MultiHeadAttentionInputIDs.ATTENTION_BIAS)
                if (
                    rel_pos_bias_node
                    and rel_pos_bias_node.op_type == "GatedRelativePositionBias"
                    and len(rel_pos_bias_node.input) == 6
                ):
                    rel_pos_bias_node.input.append(token_offset)
                    gated_relative_pos_bias_count += 1

        logger.info("Converted %d MultiHeadAttention nodes to PackedMultiHeadAttention.", len(self.attention_nodes))
        logger.info("Converted %d GatedRelativePositionBias nodes to packing mode.", gated_relative_pos_bias_count)

    def _get_input_to_remove_padding(self, first_attention_node) -> str | None:
        # When there are query, key and value inputs, we need to find the first input of the parent MatMul node.
        matmul = self.model.get_parent(first_attention_node, 0)
        if matmul and matmul.op_type == "MatMul":
            return matmul.input[0]
        return None


class PackingMode:
    def __init__(self, model: OnnxModel):
        self.model = model

    def convert(self, use_symbolic_shape_infer: bool = True) -> None:
        if self.model.get_nodes_by_op_type(Operators.ATTENTION):
            if self.model.get_nodes_by_op_type(Operators.MULTI_HEAD_ATTENTION):
                logger.error("Packing mode does not support both Attention and MultiHeadAttention in same graph.")
                return None
            packing = PackingAttention(self.model)
            return packing.convert(use_symbolic_shape_infer)
        elif self.model.get_nodes_by_op_type(Operators.MULTI_HEAD_ATTENTION):
            packing = PackingMultiHeadAttention(self.model)
            return packing.convert(use_symbolic_shape_infer)
        else:
            logger.error("Packing mode requires either Attention or MultiHeadAttention node in onnx graph.")
            return None


def _parse_arguments():
    parser = argparse.ArgumentParser(
        description="Convert to packing mode tool for ONNX Runtime. It converts BERT like model to use packing mode."
    )
    parser.add_argument("--input", required=True, type=str, help="input onnx model path")

    parser.add_argument("--output", required=True, type=str, help="optimized onnx model path")

    parser.add_argument("--verbose", required=False, action="store_true", help="show debug information.")
    parser.set_defaults(verbose=False)

    parser.add_argument(
        "--use_external_data_format",
        required=False,
        action="store_true",
        help="use external data format to store large model (>2GB)",
    )
    parser.set_defaults(use_external_data_format=False)

    args = parser.parse_args()

    return args


def _setup_logger(verbose):
    if verbose:
        coloredlogs.install(
            level="DEBUG",
            fmt="[%(filename)s:%(lineno)s - %(funcName)20s()] %(message)s",
        )
    else:
        coloredlogs.install(fmt="%(funcName)20s: %(message)s")


def main():
    args = _parse_arguments()

    _setup_logger(args.verbose)

    logger.debug(f"arguments:{args}")

    if os.path.realpath(args.input) == os.path.realpath(args.output):
        logger.warning("Specified the same input and output path. Note that this may overwrite the original model")

    model = load_model(args.input)
    packing_mode = PackingMode(OnnxModel(model))
    packing_mode.convert()
    packing_mode.model.save_model_to_file(args.output, use_external_data_format=args.use_external_data_format)


if __name__ == "__main__":
    main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\worksheet\table.py ===
# Copyright (c) 2010-2024 openpyxl

from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import (
    Descriptor,
    Alias,
    Typed,
    Bool,
    Integer,
    NoneSet,
    String,
    Sequence,
)
from openpyxl.descriptors.excel import ExtensionList, CellRange
from openpyxl.descriptors.sequence import NestedSequence
from openpyxl.xml.constants import SHEET_MAIN_NS, REL_NS
from openpyxl.xml.functions import tostring
from openpyxl.utils import range_boundaries
from openpyxl.utils.escape import escape, unescape

from .related import Related

from .filters import (
    AutoFilter,
    SortState,
)

TABLESTYLES = tuple(
    ["TableStyleMedium{0}".format(i) for i in range(1, 29)]
    + ["TableStyleLight{0}".format(i) for i in range(1, 22)]
    + ["TableStyleDark{0}".format(i) for i in range(1, 12)]
)

PIVOTSTYLES = tuple(
    ["PivotStyleMedium{0}".format(i) for i in range(1, 29)]
    + ["PivotStyleLight{0}".format(i) for i in range(1, 29)]
    + ["PivotStyleDark{0}".format(i) for i in range(1, 29)]
)


class TableStyleInfo(Serialisable):

    tagname = "tableStyleInfo"

    name = String(allow_none=True)
    showFirstColumn = Bool(allow_none=True)
    showLastColumn = Bool(allow_none=True)
    showRowStripes = Bool(allow_none=True)
    showColumnStripes = Bool(allow_none=True)

    def __init__(self,
                 name=None,
                 showFirstColumn=None,
                 showLastColumn=None,
                 showRowStripes=None,
                 showColumnStripes=None,
                ):
        self.name = name
        self.showFirstColumn = showFirstColumn
        self.showLastColumn = showLastColumn
        self.showRowStripes = showRowStripes
        self.showColumnStripes = showColumnStripes


class XMLColumnProps(Serialisable):

    tagname = "xmlColumnPr"

    mapId = Integer()
    xpath = String()
    denormalized = Bool(allow_none=True)
    xmlDataType = String()
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = ()

    def __init__(self,
                 mapId=None,
                 xpath=None,
                 denormalized=None,
                 xmlDataType=None,
                 extLst=None,
                ):
        self.mapId = mapId
        self.xpath = xpath
        self.denormalized = denormalized
        self.xmlDataType = xmlDataType


class TableFormula(Serialisable):

    tagname = "tableFormula"

    ## Note formula is stored as the text value

    array = Bool(allow_none=True)
    attr_text = Descriptor()
    text = Alias('attr_text')


    def __init__(self,
                 array=None,
                 attr_text=None,
                ):
        self.array = array
        self.attr_text = attr_text


class TableColumn(Serialisable):

    tagname = "tableColumn"

    id = Integer()
    uniqueName = String(allow_none=True)
    name = String()
    totalsRowFunction = NoneSet(values=(['sum', 'min', 'max', 'average',
                                         'count', 'countNums', 'stdDev', 'var', 'custom']))
    totalsRowLabel = String(allow_none=True)
    queryTableFieldId = Integer(allow_none=True)
    headerRowDxfId = Integer(allow_none=True)
    dataDxfId = Integer(allow_none=True)
    totalsRowDxfId = Integer(allow_none=True)
    headerRowCellStyle = String(allow_none=True)
    dataCellStyle = String(allow_none=True)
    totalsRowCellStyle = String(allow_none=True)
    calculatedColumnFormula = Typed(expected_type=TableFormula, allow_none=True)
    totalsRowFormula = Typed(expected_type=TableFormula, allow_none=True)
    xmlColumnPr = Typed(expected_type=XMLColumnProps, allow_none=True)
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = ('calculatedColumnFormula', 'totalsRowFormula',
                    'xmlColumnPr', 'extLst')

    def __init__(self,
                 id=None,
                 uniqueName=None,
                 name=None,
                 totalsRowFunction=None,
                 totalsRowLabel=None,
                 queryTableFieldId=None,
                 headerRowDxfId=None,
                 dataDxfId=None,
                 totalsRowDxfId=None,
                 headerRowCellStyle=None,
                 dataCellStyle=None,
                 totalsRowCellStyle=None,
                 calculatedColumnFormula=None,
                 totalsRowFormula=None,
                 xmlColumnPr=None,
                 extLst=None,
                ):
        self.id = id
        self.uniqueName = uniqueName
        self.name = name
        self.totalsRowFunction = totalsRowFunction
        self.totalsRowLabel = totalsRowLabel
        self.queryTableFieldId = queryTableFieldId
        self.headerRowDxfId = headerRowDxfId
        self.dataDxfId = dataDxfId
        self.totalsRowDxfId = totalsRowDxfId
        self.headerRowCellStyle = headerRowCellStyle
        self.dataCellStyle = dataCellStyle
        self.totalsRowCellStyle = totalsRowCellStyle
        self.calculatedColumnFormula = calculatedColumnFormula
        self.totalsRowFormula = totalsRowFormula
        self.xmlColumnPr = xmlColumnPr
        self.extLst = extLst


    def __iter__(self):
        for k, v in super().__iter__():
            if k == 'name':
                v = escape(v)
            yield k, v


    @classmethod
    def from_tree(cls, node):
        self = super().from_tree(node)
        self.name = unescape(self.name)
        return self


class TableNameDescriptor(String):

    """
    Table names cannot have spaces in them
    """

    def __set__(self, instance, value):
        if value is not None and " " in value:
            raise ValueError("Table names cannot have spaces")
        super().__set__(instance, value)


class Table(Serialisable):

    _path = "/tables/table{0}.xml"
    mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.table+xml"
    _rel_type = REL_NS + "/table"
    _rel_id = None

    tagname = "table"

    id = Integer()
    name = String(allow_none=True)
    displayName = TableNameDescriptor()
    comment = String(allow_none=True)
    ref = CellRange()
    tableType = NoneSet(values=(['worksheet', 'xml', 'queryTable']))
    headerRowCount = Integer(allow_none=True)
    insertRow = Bool(allow_none=True)
    insertRowShift = Bool(allow_none=True)
    totalsRowCount = Integer(allow_none=True)
    totalsRowShown = Bool(allow_none=True)
    published = Bool(allow_none=True)
    headerRowDxfId = Integer(allow_none=True)
    dataDxfId = Integer(allow_none=True)
    totalsRowDxfId = Integer(allow_none=True)
    headerRowBorderDxfId = Integer(allow_none=True)
    tableBorderDxfId = Integer(allow_none=True)
    totalsRowBorderDxfId = Integer(allow_none=True)
    headerRowCellStyle = String(allow_none=True)
    dataCellStyle = String(allow_none=True)
    totalsRowCellStyle = String(allow_none=True)
    connectionId = Integer(allow_none=True)
    autoFilter = Typed(expected_type=AutoFilter, allow_none=True)
    sortState = Typed(expected_type=SortState, allow_none=True)
    tableColumns = NestedSequence(expected_type=TableColumn, count=True)
    tableStyleInfo = Typed(expected_type=TableStyleInfo, allow_none=True)
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = ('autoFilter', 'sortState', 'tableColumns',
                    'tableStyleInfo')

    def __init__(self,
                 id=1,
                 displayName=None,
                 ref=None,
                 name=None,
                 comment=None,
                 tableType=None,
                 headerRowCount=1,
                 insertRow=None,
                 insertRowShift=None,
                 totalsRowCount=None,
                 totalsRowShown=None,
                 published=None,
                 headerRowDxfId=None,
                 dataDxfId=None,
                 totalsRowDxfId=None,
                 headerRowBorderDxfId=None,
                 tableBorderDxfId=None,
                 totalsRowBorderDxfId=None,
                 headerRowCellStyle=None,
                 dataCellStyle=None,
                 totalsRowCellStyle=None,
                 connectionId=None,
                 autoFilter=None,
                 sortState=None,
                 tableColumns=(),
                 tableStyleInfo=None,
                 extLst=None,
                ):
        self.id = id
        self.displayName = displayName
        if name is None:
            name = displayName
        self.name = name
        self.comment = comment
        self.ref = ref
        self.tableType = tableType
        self.headerRowCount = headerRowCount
        self.insertRow = insertRow
        self.insertRowShift = insertRowShift
        self.totalsRowCount = totalsRowCount
        self.totalsRowShown = totalsRowShown
        self.published = published
        self.headerRowDxfId = headerRowDxfId
        self.dataDxfId = dataDxfId
        self.totalsRowDxfId = totalsRowDxfId
        self.headerRowBorderDxfId = headerRowBorderDxfId
        self.tableBorderDxfId = tableBorderDxfId
        self.totalsRowBorderDxfId = totalsRowBorderDxfId
        self.headerRowCellStyle = headerRowCellStyle
        self.dataCellStyle = dataCellStyle
        self.totalsRowCellStyle = totalsRowCellStyle
        self.connectionId = connectionId
        self.autoFilter = autoFilter
        self.sortState = sortState
        self.tableColumns = tableColumns
        self.tableStyleInfo = tableStyleInfo


    def to_tree(self):
        tree = super().to_tree()
        tree.set("xmlns", SHEET_MAIN_NS)
        return tree


    @property
    def path(self):
        """
        Return path within the archive
        """
        return "/xl" + self._path.format(self.id)


    def _write(self, archive):
        """
        Serialise to XML and write to archive
        """
        xml = self.to_tree()
        archive.writestr(self.path[1:], tostring(xml))


    def _initialise_columns(self):
        """
        Create a list of table columns from a cell range
        Always set a ref if we have headers (the default)
        Column headings must be strings and must match cells in the worksheet.
        """

        min_col, min_row, max_col, max_row = range_boundaries(self.ref)
        for idx in range(min_col, max_col+1):
            col = TableColumn(id=idx, name="Column{0}".format(idx))
            self.tableColumns.append(col)
        if self.headerRowCount and not self.autoFilter:
            self.autoFilter = AutoFilter(ref=self.ref)


    @property
    def column_names(self):
        return [column.name for column in self.tableColumns]


class TablePartList(Serialisable):

    tagname = "tableParts"

    count = Integer(allow_none=True)
    tablePart = Sequence(expected_type=Related)

    __elements__ = ('tablePart',)
    __attrs__ = ('count',)

    def __init__(self,
                 count=None,
                 tablePart=(),
                ):
        self.tablePart = tablePart


    def append(self, part):
        self.tablePart.append(part)


    @property
    def count(self):
        return len(self.tablePart)


    def __bool__(self):
        return bool(self.tablePart)


class TableList(dict):


    def add(self, table):
        if not isinstance(table, Table):
            raise TypeError("You can only add tables")
        self[table.name] = table


    def get(self, name=None, table_range=None):
        if name is not None:
            return super().get(name)
        for table in self.values():
            if table_range == table.ref:
                return table


    def items(self):
        return [(name, table.ref) for name, table in super().items()]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\distlib\manifest.py ===
# -*- coding: utf-8 -*-
#
# Copyright (C) 2012-2023 Python Software Foundation.
# See LICENSE.txt and CONTRIBUTORS.txt.
#
"""
Class representing the list of files in a distribution.

Equivalent to distutils.filelist, but fixes some problems.
"""
import fnmatch
import logging
import os
import re
import sys

from . import DistlibException
from .compat import fsdecode
from .util import convert_path


__all__ = ['Manifest']

logger = logging.getLogger(__name__)

# a \ followed by some spaces + EOL
_COLLAPSE_PATTERN = re.compile('\\\\w*\n', re.M)
_COMMENTED_LINE = re.compile('#.*?(?=\n)|\n(?=$)', re.M | re.S)

#
# Due to the different results returned by fnmatch.translate, we need
# to do slightly different processing for Python 2.7 and 3.2 ... this needed
# to be brought in for Python 3.6 onwards.
#
_PYTHON_VERSION = sys.version_info[:2]


class Manifest(object):
    """
    A list of files built by exploring the filesystem and filtered by applying various
    patterns to what we find there.
    """

    def __init__(self, base=None):
        """
        Initialise an instance.

        :param base: The base directory to explore under.
        """
        self.base = os.path.abspath(os.path.normpath(base or os.getcwd()))
        self.prefix = self.base + os.sep
        self.allfiles = None
        self.files = set()

    #
    # Public API
    #

    def findall(self):
        """Find all files under the base and set ``allfiles`` to the absolute
        pathnames of files found.
        """
        from stat import S_ISREG, S_ISDIR, S_ISLNK

        self.allfiles = allfiles = []
        root = self.base
        stack = [root]
        pop = stack.pop
        push = stack.append

        while stack:
            root = pop()
            names = os.listdir(root)

            for name in names:
                fullname = os.path.join(root, name)

                # Avoid excess stat calls -- just one will do, thank you!
                stat = os.stat(fullname)
                mode = stat.st_mode
                if S_ISREG(mode):
                    allfiles.append(fsdecode(fullname))
                elif S_ISDIR(mode) and not S_ISLNK(mode):
                    push(fullname)

    def add(self, item):
        """
        Add a file to the manifest.

        :param item: The pathname to add. This can be relative to the base.
        """
        if not item.startswith(self.prefix):
            item = os.path.join(self.base, item)
        self.files.add(os.path.normpath(item))

    def add_many(self, items):
        """
        Add a list of files to the manifest.

        :param items: The pathnames to add. These can be relative to the base.
        """
        for item in items:
            self.add(item)

    def sorted(self, wantdirs=False):
        """
        Return sorted files in directory order
        """

        def add_dir(dirs, d):
            dirs.add(d)
            logger.debug('add_dir added %s', d)
            if d != self.base:
                parent, _ = os.path.split(d)
                assert parent not in ('', '/')
                add_dir(dirs, parent)

        result = set(self.files)    # make a copy!
        if wantdirs:
            dirs = set()
            for f in result:
                add_dir(dirs, os.path.dirname(f))
            result |= dirs
        return [os.path.join(*path_tuple) for path_tuple in
                sorted(os.path.split(path) for path in result)]

    def clear(self):
        """Clear all collected files."""
        self.files = set()
        self.allfiles = []

    def process_directive(self, directive):
        """
        Process a directive which either adds some files from ``allfiles`` to
        ``files``, or removes some files from ``files``.

        :param directive: The directive to process. This should be in a format
                     compatible with distutils ``MANIFEST.in`` files:

                     http://docs.python.org/distutils/sourcedist.html#commands
        """
        # Parse the line: split it up, make sure the right number of words
        # is there, and return the relevant words.  'action' is always
        # defined: it's the first word of the line.  Which of the other
        # three are defined depends on the action; it'll be either
        # patterns, (dir and patterns), or (dirpattern).
        action, patterns, thedir, dirpattern = self._parse_directive(directive)

        # OK, now we know that the action is valid and we have the
        # right number of words on the line for that action -- so we
        # can proceed with minimal error-checking.
        if action == 'include':
            for pattern in patterns:
                if not self._include_pattern(pattern, anchor=True):
                    logger.warning('no files found matching %r', pattern)

        elif action == 'exclude':
            for pattern in patterns:
                self._exclude_pattern(pattern, anchor=True)

        elif action == 'global-include':
            for pattern in patterns:
                if not self._include_pattern(pattern, anchor=False):
                    logger.warning('no files found matching %r '
                                   'anywhere in distribution', pattern)

        elif action == 'global-exclude':
            for pattern in patterns:
                self._exclude_pattern(pattern, anchor=False)

        elif action == 'recursive-include':
            for pattern in patterns:
                if not self._include_pattern(pattern, prefix=thedir):
                    logger.warning('no files found matching %r '
                                   'under directory %r', pattern, thedir)

        elif action == 'recursive-exclude':
            for pattern in patterns:
                self._exclude_pattern(pattern, prefix=thedir)

        elif action == 'graft':
            if not self._include_pattern(None, prefix=dirpattern):
                logger.warning('no directories found matching %r',
                               dirpattern)

        elif action == 'prune':
            if not self._exclude_pattern(None, prefix=dirpattern):
                logger.warning('no previously-included directories found '
                               'matching %r', dirpattern)
        else:   # pragma: no cover
            # This should never happen, as it should be caught in
            # _parse_template_line
            raise DistlibException(
                'invalid action %r' % action)

    #
    # Private API
    #

    def _parse_directive(self, directive):
        """
        Validate a directive.
        :param directive: The directive to validate.
        :return: A tuple of action, patterns, thedir, dir_patterns
        """
        words = directive.split()
        if len(words) == 1 and words[0] not in ('include', 'exclude',
                                                'global-include',
                                                'global-exclude',
                                                'recursive-include',
                                                'recursive-exclude',
                                                'graft', 'prune'):
            # no action given, let's use the default 'include'
            words.insert(0, 'include')

        action = words[0]
        patterns = thedir = dir_pattern = None

        if action in ('include', 'exclude',
                      'global-include', 'global-exclude'):
            if len(words) < 2:
                raise DistlibException(
                    '%r expects <pattern1> <pattern2> ...' % action)

            patterns = [convert_path(word) for word in words[1:]]

        elif action in ('recursive-include', 'recursive-exclude'):
            if len(words) < 3:
                raise DistlibException(
                    '%r expects <dir> <pattern1> <pattern2> ...' % action)

            thedir = convert_path(words[1])
            patterns = [convert_path(word) for word in words[2:]]

        elif action in ('graft', 'prune'):
            if len(words) != 2:
                raise DistlibException(
                    '%r expects a single <dir_pattern>' % action)

            dir_pattern = convert_path(words[1])

        else:
            raise DistlibException('unknown action %r' % action)

        return action, patterns, thedir, dir_pattern

    def _include_pattern(self, pattern, anchor=True, prefix=None,
                         is_regex=False):
        """Select strings (presumably filenames) from 'self.files' that
        match 'pattern', a Unix-style wildcard (glob) pattern.

        Patterns are not quite the same as implemented by the 'fnmatch'
        module: '*' and '?'  match non-special characters, where "special"
        is platform-dependent: slash on Unix; colon, slash, and backslash on
        DOS/Windows; and colon on Mac OS.

        If 'anchor' is true (the default), then the pattern match is more
        stringent: "*.py" will match "foo.py" but not "foo/bar.py".  If
        'anchor' is false, both of these will match.

        If 'prefix' is supplied, then only filenames starting with 'prefix'
        (itself a pattern) and ending with 'pattern', with anything in between
        them, will match.  'anchor' is ignored in this case.

        If 'is_regex' is true, 'anchor' and 'prefix' are ignored, and
        'pattern' is assumed to be either a string containing a regex or a
        regex object -- no translation is done, the regex is just compiled
        and used as-is.

        Selected strings will be added to self.files.

        Return True if files are found.
        """
        # XXX docstring lying about what the special chars are?
        found = False
        pattern_re = self._translate_pattern(pattern, anchor, prefix, is_regex)

        # delayed loading of allfiles list
        if self.allfiles is None:
            self.findall()

        for name in self.allfiles:
            if pattern_re.search(name):
                self.files.add(name)
                found = True
        return found

    def _exclude_pattern(self, pattern, anchor=True, prefix=None,
                         is_regex=False):
        """Remove strings (presumably filenames) from 'files' that match
        'pattern'.

        Other parameters are the same as for 'include_pattern()', above.
        The list 'self.files' is modified in place. Return True if files are
        found.

        This API is public to allow e.g. exclusion of SCM subdirs, e.g. when
        packaging source distributions
        """
        found = False
        pattern_re = self._translate_pattern(pattern, anchor, prefix, is_regex)
        for f in list(self.files):
            if pattern_re.search(f):
                self.files.remove(f)
                found = True
        return found

    def _translate_pattern(self, pattern, anchor=True, prefix=None,
                           is_regex=False):
        """Translate a shell-like wildcard pattern to a compiled regular
        expression.

        Return the compiled regex.  If 'is_regex' true,
        then 'pattern' is directly compiled to a regex (if it's a string)
        or just returned as-is (assumes it's a regex object).
        """
        if is_regex:
            if isinstance(pattern, str):
                return re.compile(pattern)
            else:
                return pattern

        if _PYTHON_VERSION > (3, 2):
            # ditch start and end characters
            start, _, end = self._glob_to_re('_').partition('_')

        if pattern:
            pattern_re = self._glob_to_re(pattern)
            if _PYTHON_VERSION > (3, 2):
                assert pattern_re.startswith(start) and pattern_re.endswith(end)
        else:
            pattern_re = ''

        base = re.escape(os.path.join(self.base, ''))
        if prefix is not None:
            # ditch end of pattern character
            if _PYTHON_VERSION <= (3, 2):
                empty_pattern = self._glob_to_re('')
                prefix_re = self._glob_to_re(prefix)[:-len(empty_pattern)]
            else:
                prefix_re = self._glob_to_re(prefix)
                assert prefix_re.startswith(start) and prefix_re.endswith(end)
                prefix_re = prefix_re[len(start): len(prefix_re) - len(end)]
            sep = os.sep
            if os.sep == '\\':
                sep = r'\\'
            if _PYTHON_VERSION <= (3, 2):
                pattern_re = '^' + base + sep.join((prefix_re,
                                                    '.*' + pattern_re))
            else:
                pattern_re = pattern_re[len(start): len(pattern_re) - len(end)]
                pattern_re = r'%s%s%s%s.*%s%s' % (start, base, prefix_re, sep,
                                                  pattern_re, end)
        else:  # no prefix -- respect anchor flag
            if anchor:
                if _PYTHON_VERSION <= (3, 2):
                    pattern_re = '^' + base + pattern_re
                else:
                    pattern_re = r'%s%s%s' % (start, base, pattern_re[len(start):])

        return re.compile(pattern_re)

    def _glob_to_re(self, pattern):
        """Translate a shell-like glob pattern to a regular expression.

        Return a string containing the regex.  Differs from
        'fnmatch.translate()' in that '*' does not match "special characters"
        (which are platform-specific).
        """
        pattern_re = fnmatch.translate(pattern)

        # '?' and '*' in the glob pattern become '.' and '.*' in the RE, which
        # IMHO is wrong -- '?' and '*' aren't supposed to match slash in Unix,
        # and by extension they shouldn't match such "special characters" under
        # any OS.  So change all non-escaped dots in the RE to match any
        # character except the special characters (currently: just os.sep).
        sep = os.sep
        if os.sep == '\\':
            # we're using a regex to manipulate a regex, so we need
            # to escape the backslash twice
            sep = r'\\\\'
        escaped = r'\1[^%s]' % sep
        pattern_re = re.sub(r'((?<!\\)(\\\\)*)\.', escaped, pattern_re)
        return pattern_re

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\configuration.py ===
"""Configuration management setup

Some terminology:
- name
  As written in config files.
- value
  Value associated with a name
- key
  Name combined with it's section (section.name)
- variant
  A single word describing where the configuration key-value pair came from
"""

import configparser
import locale
import os
import sys
from typing import Any, Dict, Iterable, List, NewType, Optional, Tuple

from pip._internal.exceptions import (
    ConfigurationError,
    ConfigurationFileCouldNotBeLoaded,
)
from pip._internal.utils import appdirs
from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.logging import getLogger
from pip._internal.utils.misc import ensure_dir, enum

RawConfigParser = configparser.RawConfigParser  # Shorthand
Kind = NewType("Kind", str)

CONFIG_BASENAME = "pip.ini" if WINDOWS else "pip.conf"
ENV_NAMES_IGNORED = "version", "help"

# The kinds of configurations there are.
kinds = enum(
    USER="user",  # User Specific
    GLOBAL="global",  # System Wide
    SITE="site",  # [Virtual] Environment Specific
    ENV="env",  # from PIP_CONFIG_FILE
    ENV_VAR="env-var",  # from Environment Variables
)
OVERRIDE_ORDER = kinds.GLOBAL, kinds.USER, kinds.SITE, kinds.ENV, kinds.ENV_VAR
VALID_LOAD_ONLY = kinds.USER, kinds.GLOBAL, kinds.SITE

logger = getLogger(__name__)


# NOTE: Maybe use the optionx attribute to normalize keynames.
def _normalize_name(name: str) -> str:
    """Make a name consistent regardless of source (environment or file)"""
    name = name.lower().replace("_", "-")
    if name.startswith("--"):
        name = name[2:]  # only prefer long opts
    return name


def _disassemble_key(name: str) -> List[str]:
    if "." not in name:
        error_message = (
            "Key does not contain dot separated section and key. "
            f"Perhaps you wanted to use 'global.{name}' instead?"
        )
        raise ConfigurationError(error_message)
    return name.split(".", 1)


def get_configuration_files() -> Dict[Kind, List[str]]:
    global_config_files = [
        os.path.join(path, CONFIG_BASENAME) for path in appdirs.site_config_dirs("pip")
    ]

    site_config_file = os.path.join(sys.prefix, CONFIG_BASENAME)
    legacy_config_file = os.path.join(
        os.path.expanduser("~"),
        "pip" if WINDOWS else ".pip",
        CONFIG_BASENAME,
    )
    new_config_file = os.path.join(appdirs.user_config_dir("pip"), CONFIG_BASENAME)
    return {
        kinds.GLOBAL: global_config_files,
        kinds.SITE: [site_config_file],
        kinds.USER: [legacy_config_file, new_config_file],
    }


class Configuration:
    """Handles management of configuration.

    Provides an interface to accessing and managing configuration files.

    This class converts provides an API that takes "section.key-name" style
    keys and stores the value associated with it as "key-name" under the
    section "section".

    This allows for a clean interface wherein the both the section and the
    key-name are preserved in an easy to manage form in the configuration files
    and the data stored is also nice.
    """

    def __init__(self, isolated: bool, load_only: Optional[Kind] = None) -> None:
        super().__init__()

        if load_only is not None and load_only not in VALID_LOAD_ONLY:
            raise ConfigurationError(
                "Got invalid value for load_only - should be one of {}".format(
                    ", ".join(map(repr, VALID_LOAD_ONLY))
                )
            )
        self.isolated = isolated
        self.load_only = load_only

        # Because we keep track of where we got the data from
        self._parsers: Dict[Kind, List[Tuple[str, RawConfigParser]]] = {
            variant: [] for variant in OVERRIDE_ORDER
        }
        self._config: Dict[Kind, Dict[str, Any]] = {
            variant: {} for variant in OVERRIDE_ORDER
        }
        self._modified_parsers: List[Tuple[str, RawConfigParser]] = []

    def load(self) -> None:
        """Loads configuration from configuration files and environment"""
        self._load_config_files()
        if not self.isolated:
            self._load_environment_vars()

    def get_file_to_edit(self) -> Optional[str]:
        """Returns the file with highest priority in configuration"""
        assert self.load_only is not None, "Need to be specified a file to be editing"

        try:
            return self._get_parser_to_modify()[0]
        except IndexError:
            return None

    def items(self) -> Iterable[Tuple[str, Any]]:
        """Returns key-value pairs like dict.items() representing the loaded
        configuration
        """
        return self._dictionary.items()

    def get_value(self, key: str) -> Any:
        """Get a value from the configuration."""
        orig_key = key
        key = _normalize_name(key)
        try:
            return self._dictionary[key]
        except KeyError:
            # disassembling triggers a more useful error message than simply
            # "No such key" in the case that the key isn't in the form command.option
            _disassemble_key(key)
            raise ConfigurationError(f"No such key - {orig_key}")

    def set_value(self, key: str, value: Any) -> None:
        """Modify a value in the configuration."""
        key = _normalize_name(key)
        self._ensure_have_load_only()

        assert self.load_only
        fname, parser = self._get_parser_to_modify()

        if parser is not None:
            section, name = _disassemble_key(key)

            # Modify the parser and the configuration
            if not parser.has_section(section):
                parser.add_section(section)
            parser.set(section, name, value)

        self._config[self.load_only][key] = value
        self._mark_as_modified(fname, parser)

    def unset_value(self, key: str) -> None:
        """Unset a value in the configuration."""
        orig_key = key
        key = _normalize_name(key)
        self._ensure_have_load_only()

        assert self.load_only
        if key not in self._config[self.load_only]:
            raise ConfigurationError(f"No such key - {orig_key}")

        fname, parser = self._get_parser_to_modify()

        if parser is not None:
            section, name = _disassemble_key(key)
            if not (
                parser.has_section(section) and parser.remove_option(section, name)
            ):
                # The option was not removed.
                raise ConfigurationError(
                    "Fatal Internal error [id=1]. Please report as a bug."
                )

            # The section may be empty after the option was removed.
            if not parser.items(section):
                parser.remove_section(section)
            self._mark_as_modified(fname, parser)

        del self._config[self.load_only][key]

    def save(self) -> None:
        """Save the current in-memory state."""
        self._ensure_have_load_only()

        for fname, parser in self._modified_parsers:
            logger.info("Writing to %s", fname)

            # Ensure directory exists.
            ensure_dir(os.path.dirname(fname))

            # Ensure directory's permission(need to be writeable)
            try:
                with open(fname, "w") as f:
                    parser.write(f)
            except OSError as error:
                raise ConfigurationError(
                    f"An error occurred while writing to the configuration file "
                    f"{fname}: {error}"
                )

    #
    # Private routines
    #

    def _ensure_have_load_only(self) -> None:
        if self.load_only is None:
            raise ConfigurationError("Needed a specific file to be modifying.")
        logger.debug("Will be working with %s variant only", self.load_only)

    @property
    def _dictionary(self) -> Dict[str, Any]:
        """A dictionary representing the loaded configuration."""
        # NOTE: Dictionaries are not populated if not loaded. So, conditionals
        #       are not needed here.
        retval = {}

        for variant in OVERRIDE_ORDER:
            retval.update(self._config[variant])

        return retval

    def _load_config_files(self) -> None:
        """Loads configuration from configuration files"""
        config_files = dict(self.iter_config_files())
        if config_files[kinds.ENV][0:1] == [os.devnull]:
            logger.debug(
                "Skipping loading configuration files due to "
                "environment's PIP_CONFIG_FILE being os.devnull"
            )
            return

        for variant, files in config_files.items():
            for fname in files:
                # If there's specific variant set in `load_only`, load only
                # that variant, not the others.
                if self.load_only is not None and variant != self.load_only:
                    logger.debug("Skipping file '%s' (variant: %s)", fname, variant)
                    continue

                parser = self._load_file(variant, fname)

                # Keeping track of the parsers used
                self._parsers[variant].append((fname, parser))

    def _load_file(self, variant: Kind, fname: str) -> RawConfigParser:
        logger.verbose("For variant '%s', will try loading '%s'", variant, fname)
        parser = self._construct_parser(fname)

        for section in parser.sections():
            items = parser.items(section)
            self._config[variant].update(self._normalized_keys(section, items))

        return parser

    def _construct_parser(self, fname: str) -> RawConfigParser:
        parser = configparser.RawConfigParser()
        # If there is no such file, don't bother reading it but create the
        # parser anyway, to hold the data.
        # Doing this is useful when modifying and saving files, where we don't
        # need to construct a parser.
        if os.path.exists(fname):
            locale_encoding = locale.getpreferredencoding(False)
            try:
                parser.read(fname, encoding=locale_encoding)
            except UnicodeDecodeError:
                # See https://github.com/pypa/pip/issues/4963
                raise ConfigurationFileCouldNotBeLoaded(
                    reason=f"contains invalid {locale_encoding} characters",
                    fname=fname,
                )
            except configparser.Error as error:
                # See https://github.com/pypa/pip/issues/4893
                raise ConfigurationFileCouldNotBeLoaded(error=error)
        return parser

    def _load_environment_vars(self) -> None:
        """Loads configuration from environment variables"""
        self._config[kinds.ENV_VAR].update(
            self._normalized_keys(":env:", self.get_environ_vars())
        )

    def _normalized_keys(
        self, section: str, items: Iterable[Tuple[str, Any]]
    ) -> Dict[str, Any]:
        """Normalizes items to construct a dictionary with normalized keys.

        This routine is where the names become keys and are made the same
        regardless of source - configuration files or environment.
        """
        normalized = {}
        for name, val in items:
            key = section + "." + _normalize_name(name)
            normalized[key] = val
        return normalized

    def get_environ_vars(self) -> Iterable[Tuple[str, str]]:
        """Returns a generator with all environmental vars with prefix PIP_"""
        for key, val in os.environ.items():
            if key.startswith("PIP_"):
                name = key[4:].lower()
                if name not in ENV_NAMES_IGNORED:
                    yield name, val

    # XXX: This is patched in the tests.
    def iter_config_files(self) -> Iterable[Tuple[Kind, List[str]]]:
        """Yields variant and configuration files associated with it.

        This should be treated like items of a dictionary. The order
        here doesn't affect what gets overridden. That is controlled
        by OVERRIDE_ORDER. However this does control the order they are
        displayed to the user. It's probably most ergonomic to display
        things in the same order as OVERRIDE_ORDER
        """
        # SMELL: Move the conditions out of this function

        env_config_file = os.environ.get("PIP_CONFIG_FILE", None)
        config_files = get_configuration_files()

        yield kinds.GLOBAL, config_files[kinds.GLOBAL]

        # per-user config is not loaded when env_config_file exists
        should_load_user_config = not self.isolated and not (
            env_config_file and os.path.exists(env_config_file)
        )
        if should_load_user_config:
            # The legacy config file is overridden by the new config file
            yield kinds.USER, config_files[kinds.USER]

        # virtualenv config
        yield kinds.SITE, config_files[kinds.SITE]

        if env_config_file is not None:
            yield kinds.ENV, [env_config_file]
        else:
            yield kinds.ENV, []

    def get_values_in_config(self, variant: Kind) -> Dict[str, Any]:
        """Get values present in a config file"""
        return self._config[variant]

    def _get_parser_to_modify(self) -> Tuple[str, RawConfigParser]:
        # Determine which parser to modify
        assert self.load_only
        parsers = self._parsers[self.load_only]
        if not parsers:
            # This should not happen if everything works correctly.
            raise ConfigurationError(
                "Fatal Internal error [id=2]. Please report as a bug."
            )

        # Use the highest priority parser.
        return parsers[-1]

    # XXX: This is patched in the tests.
    def _mark_as_modified(self, fname: str, parser: RawConfigParser) -> None:
        file_parser_tuple = (fname, parser)
        if file_parser_tuple not in self._modified_parsers:
            self._modified_parsers.append(file_parser_tuple)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._dictionary!r})"

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\drawing\spreadsheet_drawing.py ===
# Copyright (c) 2010-2024 openpyxl

from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import (
    Typed,
    Bool,
    NoneSet,
    Integer,
    Sequence,
    Alias,
)
from openpyxl.descriptors.nested import (
    NestedText,
    NestedNoneSet,
)
from openpyxl.descriptors.excel import Relation

from openpyxl.packaging.relationship import (
    Relationship,
    RelationshipList,
)
from openpyxl.utils import coordinate_to_tuple
from openpyxl.utils.units import (
    cm_to_EMU,
    pixels_to_EMU,
)
from openpyxl.drawing.image import Image

from openpyxl.xml.constants import SHEET_DRAWING_NS

from openpyxl.chart._chart import ChartBase
from .xdr import (
    XDRPoint2D,
    XDRPositiveSize2D,
)
from .fill import Blip
from .connector import Shape
from .graphic import (
    GroupShape,
    GraphicFrame,
    )
from .geometry import PresetGeometry2D
from .picture import PictureFrame
from .relation import ChartRelation


class AnchorClientData(Serialisable):

    fLocksWithSheet = Bool(allow_none=True)
    fPrintsWithSheet = Bool(allow_none=True)

    def __init__(self,
                 fLocksWithSheet=None,
                 fPrintsWithSheet=None,
                 ):
        self.fLocksWithSheet = fLocksWithSheet
        self.fPrintsWithSheet = fPrintsWithSheet


class AnchorMarker(Serialisable):

    tagname = "marker"

    col = NestedText(expected_type=int)
    colOff = NestedText(expected_type=int)
    row = NestedText(expected_type=int)
    rowOff = NestedText(expected_type=int)

    def __init__(self,
                 col=0,
                 colOff=0,
                 row=0,
                 rowOff=0,
                 ):
        self.col = col
        self.colOff = colOff
        self.row = row
        self.rowOff = rowOff


class _AnchorBase(Serialisable):

    #one of
    sp = Typed(expected_type=Shape, allow_none=True)
    shape = Alias("sp")
    grpSp = Typed(expected_type=GroupShape, allow_none=True)
    groupShape = Alias("grpSp")
    graphicFrame = Typed(expected_type=GraphicFrame, allow_none=True)
    cxnSp = Typed(expected_type=Shape, allow_none=True)
    connectionShape = Alias("cxnSp")
    pic = Typed(expected_type=PictureFrame, allow_none=True)
    contentPart = Relation()

    clientData = Typed(expected_type=AnchorClientData)

    __elements__ = ('sp', 'grpSp', 'graphicFrame',
                    'cxnSp', 'pic', 'contentPart', 'clientData')

    def __init__(self,
                 clientData=None,
                 sp=None,
                 grpSp=None,
                 graphicFrame=None,
                 cxnSp=None,
                 pic=None,
                 contentPart=None
                 ):
        if clientData is None:
            clientData = AnchorClientData()
        self.clientData = clientData
        self.sp = sp
        self.grpSp = grpSp
        self.graphicFrame = graphicFrame
        self.cxnSp = cxnSp
        self.pic = pic
        self.contentPart = contentPart


class AbsoluteAnchor(_AnchorBase):

    tagname = "absoluteAnchor"

    pos = Typed(expected_type=XDRPoint2D)
    ext = Typed(expected_type=XDRPositiveSize2D)

    sp = _AnchorBase.sp
    grpSp = _AnchorBase.grpSp
    graphicFrame = _AnchorBase.graphicFrame
    cxnSp = _AnchorBase.cxnSp
    pic = _AnchorBase.pic
    contentPart = _AnchorBase.contentPart
    clientData = _AnchorBase.clientData

    __elements__ = ('pos', 'ext') + _AnchorBase.__elements__

    def __init__(self,
                 pos=None,
                 ext=None,
                 **kw
                ):
        if pos is None:
            pos = XDRPoint2D(0, 0)
        self.pos = pos
        if ext is None:
            ext = XDRPositiveSize2D(0, 0)
        self.ext = ext
        super().__init__(**kw)


class OneCellAnchor(_AnchorBase):

    tagname = "oneCellAnchor"

    _from = Typed(expected_type=AnchorMarker)
    ext = Typed(expected_type=XDRPositiveSize2D)

    sp = _AnchorBase.sp
    grpSp = _AnchorBase.grpSp
    graphicFrame = _AnchorBase.graphicFrame
    cxnSp = _AnchorBase.cxnSp
    pic = _AnchorBase.pic
    contentPart = _AnchorBase.contentPart
    clientData = _AnchorBase.clientData

    __elements__ = ('_from', 'ext') + _AnchorBase.__elements__


    def __init__(self,
                 _from=None,
                 ext=None,
                 **kw
                ):
        if _from is None:
            _from = AnchorMarker()
        self._from = _from
        if ext is None:
            ext = XDRPositiveSize2D(0, 0)
        self.ext = ext
        super().__init__(**kw)


class TwoCellAnchor(_AnchorBase):

    tagname = "twoCellAnchor"

    editAs = NoneSet(values=(['twoCell', 'oneCell', 'absolute']))
    _from = Typed(expected_type=AnchorMarker)
    to = Typed(expected_type=AnchorMarker)

    sp = _AnchorBase.sp
    grpSp = _AnchorBase.grpSp
    graphicFrame = _AnchorBase.graphicFrame
    cxnSp = _AnchorBase.cxnSp
    pic = _AnchorBase.pic
    contentPart = _AnchorBase.contentPart
    clientData = _AnchorBase.clientData

    __elements__ = ('_from', 'to') + _AnchorBase.__elements__

    def __init__(self,
                 editAs=None,
                 _from=None,
                 to=None,
                 **kw
                 ):
        self.editAs = editAs
        if _from is None:
            _from = AnchorMarker()
        self._from = _from
        if to is None:
            to = AnchorMarker()
        self.to = to
        super().__init__(**kw)


def _check_anchor(obj):
    """
    Check whether an object has an existing Anchor object
    If not create a OneCellAnchor using the provided coordinate
    """
    anchor = obj.anchor
    if not isinstance(anchor, _AnchorBase):
        row, col = coordinate_to_tuple(anchor.upper())
        anchor = OneCellAnchor()
        anchor._from.row = row -1
        anchor._from.col = col -1
        if isinstance(obj, ChartBase):
            anchor.ext.width = cm_to_EMU(obj.width)
            anchor.ext.height = cm_to_EMU(obj.height)
        elif isinstance(obj, Image):
            anchor.ext.width = pixels_to_EMU(obj.width)
            anchor.ext.height = pixels_to_EMU(obj.height)
    return anchor


class SpreadsheetDrawing(Serialisable):

    tagname = "wsDr"
    mime_type = "application/vnd.openxmlformats-officedocument.drawing+xml"
    _rel_type = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/drawing"
    _path = PartName="/xl/drawings/drawing{0}.xml"
    _id = None

    twoCellAnchor = Sequence(expected_type=TwoCellAnchor, allow_none=True)
    oneCellAnchor = Sequence(expected_type=OneCellAnchor, allow_none=True)
    absoluteAnchor = Sequence(expected_type=AbsoluteAnchor, allow_none=True)

    __elements__ = ("twoCellAnchor", "oneCellAnchor", "absoluteAnchor")

    def __init__(self,
                 twoCellAnchor=(),
                 oneCellAnchor=(),
                 absoluteAnchor=(),
                 ):
        self.twoCellAnchor = twoCellAnchor
        self.oneCellAnchor = oneCellAnchor
        self.absoluteAnchor = absoluteAnchor
        self.charts = []
        self.images = []
        self._rels = []


    def __hash__(self):
        """
        Just need to check for identity
        """
        return id(self)


    def __bool__(self):
        return bool(self.charts) or bool(self.images)



    def _write(self):
        """
        create required structure and the serialise
        """
        anchors = []
        for idx, obj in enumerate(self.charts + self.images, 1):
            anchor = _check_anchor(obj)
            if isinstance(obj, ChartBase):
                rel = Relationship(type="chart", Target=obj.path)
                anchor.graphicFrame = self._chart_frame(idx)
            elif isinstance(obj, Image):
                rel = Relationship(type="image", Target=obj.path)
                child = anchor.pic or anchor.groupShape and anchor.groupShape.pic
                if not child:
                    anchor.pic = self._picture_frame(idx)
                else:
                    child.blipFill.blip.embed = "rId{0}".format(idx)

            anchors.append(anchor)
            self._rels.append(rel)

        for a in anchors:
            if isinstance(a, OneCellAnchor):
                self.oneCellAnchor.append(a)
            elif isinstance(a, TwoCellAnchor):
                self.twoCellAnchor.append(a)
            else:
                self.absoluteAnchor.append(a)

        tree = self.to_tree()
        tree.set('xmlns', SHEET_DRAWING_NS)
        return tree


    def _chart_frame(self, idx):
        chart_rel = ChartRelation(f"rId{idx}")
        frame = GraphicFrame()
        nv = frame.nvGraphicFramePr.cNvPr
        nv.id = idx
        nv.name = "Chart {0}".format(idx)
        frame.graphic.graphicData.chart = chart_rel
        return frame


    def _picture_frame(self, idx):
        pic = PictureFrame()
        pic.nvPicPr.cNvPr.descr = "Picture"
        pic.nvPicPr.cNvPr.id = idx
        pic.nvPicPr.cNvPr.name = "Image {0}".format(idx)

        pic.blipFill.blip = Blip()
        pic.blipFill.blip.embed = "rId{0}".format(idx)
        pic.blipFill.blip.cstate = "print"

        pic.spPr.prstGeom = PresetGeometry2D(prst="rect")
        pic.spPr.ln = None
        return pic


    def _write_rels(self):
        rels = RelationshipList()
        for r in self._rels:
            rels.append(r)
        return rels.to_tree()


    @property
    def path(self):
        return self._path.format(self._id)


    @property
    def _chart_rels(self):
        """
        Get relationship information for each chart and bind anchor to it
        """
        rels = []
        anchors = self.absoluteAnchor + self.oneCellAnchor + self.twoCellAnchor
        for anchor in anchors:
            if anchor.graphicFrame is not None:
                graphic = anchor.graphicFrame.graphic
                rel = graphic.graphicData.chart
                if rel is not None:
                    rel.anchor = anchor
                    rel.anchor.graphicFrame = None
                    rels.append(rel)
        return rels


    @property
    def _blip_rels(self):
        """
        Get relationship information for each blip and bind anchor to it

        Images that are not part of the XLSX package will be ignored.
        """
        rels = []
        anchors = self.absoluteAnchor + self.oneCellAnchor + self.twoCellAnchor

        for anchor in anchors:
            child = anchor.pic or anchor.groupShape and anchor.groupShape.pic
            if child and child.blipFill:
                rel = child.blipFill.blip
                if rel is not None and rel.embed:
                    rel.anchor = anchor
                    rels.append(rel)

        return rels

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\charset_normalizer\cli\__main__.py ===
from __future__ import annotations

import argparse
import sys
import typing
from json import dumps
from os.path import abspath, basename, dirname, join, realpath
from platform import python_version
from unicodedata import unidata_version

import charset_normalizer.md as md_module
from charset_normalizer import from_fp
from charset_normalizer.models import CliDetectionResult
from charset_normalizer.version import __version__


def query_yes_no(question: str, default: str = "yes") -> bool:
    """Ask a yes/no question via input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".

    Credit goes to (c) https://stackoverflow.com/questions/3041986/apt-command-line-interface-like-yes-no-input
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' (or 'y' or 'n').\n")


class FileType:
    """Factory for creating file object types

    Instances of FileType are typically passed as type= arguments to the
    ArgumentParser add_argument() method.

    Keyword Arguments:
        - mode -- A string indicating how the file is to be opened. Accepts the
            same values as the builtin open() function.
        - bufsize -- The file's desired buffer size. Accepts the same values as
            the builtin open() function.
        - encoding -- The file's encoding. Accepts the same values as the
            builtin open() function.
        - errors -- A string indicating how encoding and decoding errors are to
            be handled. Accepts the same value as the builtin open() function.

    Backported from CPython 3.12
    """

    def __init__(
        self,
        mode: str = "r",
        bufsize: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
    ):
        self._mode = mode
        self._bufsize = bufsize
        self._encoding = encoding
        self._errors = errors

    def __call__(self, string: str) -> typing.IO:  # type: ignore[type-arg]
        # the special argument "-" means sys.std{in,out}
        if string == "-":
            if "r" in self._mode:
                return sys.stdin.buffer if "b" in self._mode else sys.stdin
            elif any(c in self._mode for c in "wax"):
                return sys.stdout.buffer if "b" in self._mode else sys.stdout
            else:
                msg = f'argument "-" with mode {self._mode}'
                raise ValueError(msg)

        # all other arguments are used as file names
        try:
            return open(string, self._mode, self._bufsize, self._encoding, self._errors)
        except OSError as e:
            message = f"can't open '{string}': {e}"
            raise argparse.ArgumentTypeError(message)

    def __repr__(self) -> str:
        args = self._mode, self._bufsize
        kwargs = [("encoding", self._encoding), ("errors", self._errors)]
        args_str = ", ".join(
            [repr(arg) for arg in args if arg != -1]
            + [f"{kw}={arg!r}" for kw, arg in kwargs if arg is not None]
        )
        return f"{type(self).__name__}({args_str})"


def cli_detect(argv: list[str] | None = None) -> int:
    """
    CLI assistant using ARGV and ArgumentParser
    :param argv:
    :return: 0 if everything is fine, anything else equal trouble
    """
    parser = argparse.ArgumentParser(
        description="The Real First Universal Charset Detector. "
        "Discover originating encoding used on text file. "
        "Normalize text to unicode."
    )

    parser.add_argument(
        "files", type=FileType("rb"), nargs="+", help="File(s) to be analysed"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        dest="verbose",
        help="Display complementary information about file if any. "
        "Stdout will contain logs about the detection process.",
    )
    parser.add_argument(
        "-a",
        "--with-alternative",
        action="store_true",
        default=False,
        dest="alternatives",
        help="Output complementary possibilities if any. Top-level JSON WILL be a list.",
    )
    parser.add_argument(
        "-n",
        "--normalize",
        action="store_true",
        default=False,
        dest="normalize",
        help="Permit to normalize input file. If not set, program does not write anything.",
    )
    parser.add_argument(
        "-m",
        "--minimal",
        action="store_true",
        default=False,
        dest="minimal",
        help="Only output the charset detected to STDOUT. Disabling JSON output.",
    )
    parser.add_argument(
        "-r",
        "--replace",
        action="store_true",
        default=False,
        dest="replace",
        help="Replace file when trying to normalize it instead of creating a new one.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        dest="force",
        help="Replace file without asking if you are sure, use this flag with caution.",
    )
    parser.add_argument(
        "-i",
        "--no-preemptive",
        action="store_true",
        default=False,
        dest="no_preemptive",
        help="Disable looking at a charset declaration to hint the detector.",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        action="store",
        default=0.2,
        type=float,
        dest="threshold",
        help="Define a custom maximum amount of noise allowed in decoded content. 0. <= noise <= 1.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="Charset-Normalizer {} - Python {} - Unicode {} - SpeedUp {}".format(
            __version__,
            python_version(),
            unidata_version,
            "OFF" if md_module.__file__.lower().endswith(".py") else "ON",
        ),
        help="Show version information and exit.",
    )

    args = parser.parse_args(argv)

    if args.replace is True and args.normalize is False:
        if args.files:
            for my_file in args.files:
                my_file.close()
        print("Use --replace in addition of --normalize only.", file=sys.stderr)
        return 1

    if args.force is True and args.replace is False:
        if args.files:
            for my_file in args.files:
                my_file.close()
        print("Use --force in addition of --replace only.", file=sys.stderr)
        return 1

    if args.threshold < 0.0 or args.threshold > 1.0:
        if args.files:
            for my_file in args.files:
                my_file.close()
        print("--threshold VALUE should be between 0. AND 1.", file=sys.stderr)
        return 1

    x_ = []

    for my_file in args.files:
        matches = from_fp(
            my_file,
            threshold=args.threshold,
            explain=args.verbose,
            preemptive_behaviour=args.no_preemptive is False,
        )

        best_guess = matches.best()

        if best_guess is None:
            print(
                'Unable to identify originating encoding for "{}". {}'.format(
                    my_file.name,
                    (
                        "Maybe try increasing maximum amount of chaos."
                        if args.threshold < 1.0
                        else ""
                    ),
                ),
                file=sys.stderr,
            )
            x_.append(
                CliDetectionResult(
                    abspath(my_file.name),
                    None,
                    [],
                    [],
                    "Unknown",
                    [],
                    False,
                    1.0,
                    0.0,
                    None,
                    True,
                )
            )
        else:
            x_.append(
                CliDetectionResult(
                    abspath(my_file.name),
                    best_guess.encoding,
                    best_guess.encoding_aliases,
                    [
                        cp
                        for cp in best_guess.could_be_from_charset
                        if cp != best_guess.encoding
                    ],
                    best_guess.language,
                    best_guess.alphabets,
                    best_guess.bom,
                    best_guess.percent_chaos,
                    best_guess.percent_coherence,
                    None,
                    True,
                )
            )

            if len(matches) > 1 and args.alternatives:
                for el in matches:
                    if el != best_guess:
                        x_.append(
                            CliDetectionResult(
                                abspath(my_file.name),
                                el.encoding,
                                el.encoding_aliases,
                                [
                                    cp
                                    for cp in el.could_be_from_charset
                                    if cp != el.encoding
                                ],
                                el.language,
                                el.alphabets,
                                el.bom,
                                el.percent_chaos,
                                el.percent_coherence,
                                None,
                                False,
                            )
                        )

            if args.normalize is True:
                if best_guess.encoding.startswith("utf") is True:
                    print(
                        '"{}" file does not need to be normalized, as it already came from unicode.'.format(
                            my_file.name
                        ),
                        file=sys.stderr,
                    )
                    if my_file.closed is False:
                        my_file.close()
                    continue

                dir_path = dirname(realpath(my_file.name))
                file_name = basename(realpath(my_file.name))

                o_: list[str] = file_name.split(".")

                if args.replace is False:
                    o_.insert(-1, best_guess.encoding)
                    if my_file.closed is False:
                        my_file.close()
                elif (
                    args.force is False
                    and query_yes_no(
                        'Are you sure to normalize "{}" by replacing it ?'.format(
                            my_file.name
                        ),
                        "no",
                    )
                    is False
                ):
                    if my_file.closed is False:
                        my_file.close()
                    continue

                try:
                    x_[0].unicode_path = join(dir_path, ".".join(o_))

                    with open(x_[0].unicode_path, "wb") as fp:
                        fp.write(best_guess.output())
                except OSError as e:
                    print(str(e), file=sys.stderr)
                    if my_file.closed is False:
                        my_file.close()
                    return 2

        if my_file.closed is False:
            my_file.close()

    if args.minimal is False:
        print(
            dumps(
                [el.__dict__ for el in x_] if len(x_) > 1 else x_[0].__dict__,
                ensure_ascii=True,
                indent=4,
            )
        )
    else:
        for my_file in args.files:
            print(
                ", ".join(
                    [
                        el.encoding or "undefined"
                        for el in x_
                        if el.path == abspath(my_file.name)
                    ]
                )
            )

    return 0


if __name__ == "__main__":
    cli_detect()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\matlib.py ===
import warnings

# 2018-05-29, PendingDeprecationWarning added to matrix.__new__
# 2020-01-23, numpy 1.19.0 PendingDeprecatonWarning
warnings.warn("Importing from numpy.matlib is deprecated since 1.19.0. "
              "The matrix subclass is not the recommended way to represent "
              "matrices or deal with linear algebra (see "
              "https://docs.scipy.org/doc/numpy/user/numpy-for-matlab-users.html). "
              "Please adjust your code to use regular ndarray. ",
              PendingDeprecationWarning, stacklevel=2)

import numpy as np

# Matlib.py contains all functions in the numpy namespace with a few
# replacements. See doc/source/reference/routines.matlib.rst for details.
# Need * as we're copying the numpy namespace.
from numpy import *  # noqa: F403
from numpy.matrixlib.defmatrix import asmatrix, matrix

__version__ = np.__version__

__all__ = ['rand', 'randn', 'repmat']
__all__ += np.__all__

def empty(shape, dtype=None, order='C'):
    """Return a new matrix of given shape and type, without initializing entries.

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the empty matrix.
    dtype : data-type, optional
        Desired output data-type.
    order : {'C', 'F'}, optional
        Whether to store multi-dimensional data in row-major
        (C-style) or column-major (Fortran-style) order in
        memory.

    See Also
    --------
    numpy.empty : Equivalent array function.
    matlib.zeros : Return a matrix of zeros.
    matlib.ones : Return a matrix of ones.

    Notes
    -----
    Unlike other matrix creation functions (e.g. `matlib.zeros`,
    `matlib.ones`), `matlib.empty` does not initialize the values of the
    matrix, and may therefore be marginally faster. However, the values
    stored in the newly allocated matrix are arbitrary. For reproducible
    behavior, be sure to set each element of the matrix before reading.

    Examples
    --------
    >>> import numpy.matlib
    >>> np.matlib.empty((2, 2))    # filled with random data
    matrix([[  6.76425276e-320,   9.79033856e-307], # random
            [  7.39337286e-309,   3.22135945e-309]])
    >>> np.matlib.empty((2, 2), dtype=int)
    matrix([[ 6600475,        0], # random
            [ 6586976, 22740995]])

    """
    return ndarray.__new__(matrix, shape, dtype, order=order)

def ones(shape, dtype=None, order='C'):
    """
    Matrix of ones.

    Return a matrix of given shape and type, filled with ones.

    Parameters
    ----------
    shape : {sequence of ints, int}
        Shape of the matrix
    dtype : data-type, optional
        The desired data-type for the matrix, default is np.float64.
    order : {'C', 'F'}, optional
        Whether to store matrix in C- or Fortran-contiguous order,
        default is 'C'.

    Returns
    -------
    out : matrix
        Matrix of ones of given shape, dtype, and order.

    See Also
    --------
    ones : Array of ones.
    matlib.zeros : Zero matrix.

    Notes
    -----
    If `shape` has length one i.e. ``(N,)``, or is a scalar ``N``,
    `out` becomes a single row matrix of shape ``(1,N)``.

    Examples
    --------
    >>> np.matlib.ones((2,3))
    matrix([[1.,  1.,  1.],
            [1.,  1.,  1.]])

    >>> np.matlib.ones(2)
    matrix([[1.,  1.]])

    """
    a = ndarray.__new__(matrix, shape, dtype, order=order)
    a.fill(1)
    return a

def zeros(shape, dtype=None, order='C'):
    """
    Return a matrix of given shape and type, filled with zeros.

    Parameters
    ----------
    shape : int or sequence of ints
        Shape of the matrix
    dtype : data-type, optional
        The desired data-type for the matrix, default is float.
    order : {'C', 'F'}, optional
        Whether to store the result in C- or Fortran-contiguous order,
        default is 'C'.

    Returns
    -------
    out : matrix
        Zero matrix of given shape, dtype, and order.

    See Also
    --------
    numpy.zeros : Equivalent array function.
    matlib.ones : Return a matrix of ones.

    Notes
    -----
    If `shape` has length one i.e. ``(N,)``, or is a scalar ``N``,
    `out` becomes a single row matrix of shape ``(1,N)``.

    Examples
    --------
    >>> import numpy.matlib
    >>> np.matlib.zeros((2, 3))
    matrix([[0.,  0.,  0.],
            [0.,  0.,  0.]])

    >>> np.matlib.zeros(2)
    matrix([[0.,  0.]])

    """
    a = ndarray.__new__(matrix, shape, dtype, order=order)
    a.fill(0)
    return a

def identity(n, dtype=None):
    """
    Returns the square identity matrix of given size.

    Parameters
    ----------
    n : int
        Size of the returned identity matrix.
    dtype : data-type, optional
        Data-type of the output. Defaults to ``float``.

    Returns
    -------
    out : matrix
        `n` x `n` matrix with its main diagonal set to one,
        and all other elements zero.

    See Also
    --------
    numpy.identity : Equivalent array function.
    matlib.eye : More general matrix identity function.

    Examples
    --------
    >>> import numpy.matlib
    >>> np.matlib.identity(3, dtype=int)
    matrix([[1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]])

    """
    a = array([1] + n * [0], dtype=dtype)
    b = empty((n, n), dtype=dtype)
    b.flat = a
    return b

def eye(n, M=None, k=0, dtype=float, order='C'):
    """
    Return a matrix with ones on the diagonal and zeros elsewhere.

    Parameters
    ----------
    n : int
        Number of rows in the output.
    M : int, optional
        Number of columns in the output, defaults to `n`.
    k : int, optional
        Index of the diagonal: 0 refers to the main diagonal,
        a positive value refers to an upper diagonal,
        and a negative value to a lower diagonal.
    dtype : dtype, optional
        Data-type of the returned matrix.
    order : {'C', 'F'}, optional
        Whether the output should be stored in row-major (C-style) or
        column-major (Fortran-style) order in memory.

    Returns
    -------
    I : matrix
        A `n` x `M` matrix where all elements are equal to zero,
        except for the `k`-th diagonal, whose values are equal to one.

    See Also
    --------
    numpy.eye : Equivalent array function.
    identity : Square identity matrix.

    Examples
    --------
    >>> import numpy.matlib
    >>> np.matlib.eye(3, k=1, dtype=float)
    matrix([[0.,  1.,  0.],
            [0.,  0.,  1.],
            [0.,  0.,  0.]])

    """
    return asmatrix(np.eye(n, M=M, k=k, dtype=dtype, order=order))

def rand(*args):
    """
    Return a matrix of random values with given shape.

    Create a matrix of the given shape and propagate it with
    random samples from a uniform distribution over ``[0, 1)``.

    Parameters
    ----------
    \\*args : Arguments
        Shape of the output.
        If given as N integers, each integer specifies the size of one
        dimension.
        If given as a tuple, this tuple gives the complete shape.

    Returns
    -------
    out : ndarray
        The matrix of random values with shape given by `\\*args`.

    See Also
    --------
    randn, numpy.random.RandomState.rand

    Examples
    --------
    >>> np.random.seed(123)
    >>> import numpy.matlib
    >>> np.matlib.rand(2, 3)
    matrix([[0.69646919, 0.28613933, 0.22685145],
            [0.55131477, 0.71946897, 0.42310646]])
    >>> np.matlib.rand((2, 3))
    matrix([[0.9807642 , 0.68482974, 0.4809319 ],
            [0.39211752, 0.34317802, 0.72904971]])

    If the first argument is a tuple, other arguments are ignored:

    >>> np.matlib.rand((2, 3), 4)
    matrix([[0.43857224, 0.0596779 , 0.39804426],
            [0.73799541, 0.18249173, 0.17545176]])

    """
    if isinstance(args[0], tuple):
        args = args[0]
    return asmatrix(np.random.rand(*args))

def randn(*args):
    """
    Return a random matrix with data from the "standard normal" distribution.

    `randn` generates a matrix filled with random floats sampled from a
    univariate "normal" (Gaussian) distribution of mean 0 and variance 1.

    Parameters
    ----------
    \\*args : Arguments
        Shape of the output.
        If given as N integers, each integer specifies the size of one
        dimension. If given as a tuple, this tuple gives the complete shape.

    Returns
    -------
    Z : matrix of floats
        A matrix of floating-point samples drawn from the standard normal
        distribution.

    See Also
    --------
    rand, numpy.random.RandomState.randn

    Notes
    -----
    For random samples from the normal distribution with mean ``mu`` and
    standard deviation ``sigma``, use::

        sigma * np.matlib.randn(...) + mu

    Examples
    --------
    >>> np.random.seed(123)
    >>> import numpy.matlib
    >>> np.matlib.randn(1)
    matrix([[-1.0856306]])
    >>> np.matlib.randn(1, 2, 3)
    matrix([[ 0.99734545,  0.2829785 , -1.50629471],
            [-0.57860025,  1.65143654, -2.42667924]])

    Two-by-four matrix of samples from the normal distribution with
    mean 3 and standard deviation 2.5:

    >>> 2.5 * np.matlib.randn((2, 4)) + 3
    matrix([[1.92771843, 6.16484065, 0.83314899, 1.30278462],
            [2.76322758, 6.72847407, 1.40274501, 1.8900451 ]])

    """
    if isinstance(args[0], tuple):
        args = args[0]
    return asmatrix(np.random.randn(*args))

def repmat(a, m, n):
    """
    Repeat a 0-D to 2-D array or matrix MxN times.

    Parameters
    ----------
    a : array_like
        The array or matrix to be repeated.
    m, n : int
        The number of times `a` is repeated along the first and second axes.

    Returns
    -------
    out : ndarray
        The result of repeating `a`.

    Examples
    --------
    >>> import numpy.matlib
    >>> a0 = np.array(1)
    >>> np.matlib.repmat(a0, 2, 3)
    array([[1, 1, 1],
           [1, 1, 1]])

    >>> a1 = np.arange(4)
    >>> np.matlib.repmat(a1, 2, 2)
    array([[0, 1, 2, 3, 0, 1, 2, 3],
           [0, 1, 2, 3, 0, 1, 2, 3]])

    >>> a2 = np.asmatrix(np.arange(6).reshape(2, 3))
    >>> np.matlib.repmat(a2, 2, 3)
    matrix([[0, 1, 2, 0, 1, 2, 0, 1, 2],
            [3, 4, 5, 3, 4, 5, 3, 4, 5],
            [0, 1, 2, 0, 1, 2, 0, 1, 2],
            [3, 4, 5, 3, 4, 5, 3, 4, 5]])

    """
    a = asanyarray(a)
    ndim = a.ndim
    if ndim == 0:
        origrows, origcols = (1, 1)
    elif ndim == 1:
        origrows, origcols = (1, a.shape[0])
    else:
        origrows, origcols = a.shape
    rows = origrows * m
    cols = origcols * n
    c = a.reshape(1, a.size).repeat(m, 0).reshape(rows, origcols).repeat(n, 0)
    return c.reshape(rows, cols)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\whisper\whisper_inputs.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import logging

import numpy as np
import torch
from transformers import WhisperConfig

from onnxruntime import InferenceSession

logger = logging.getLogger(__name__)


# Create audio_features for encoder
# Shape is (batch_size, feature_size, sequence_length) = (batch_size, num_mel_filters, num_frames)
# where num_mel_filters is a model attribute and num_frames = (chunk_length * sample_rate) // hop_length.
#
# Hard-coded audio hyperparameters:
# SAMPLE_RATE = 16000
# N_FFT = 400
# HOP_LENGTH = 160
# CHUNK_LENGTH = 30  (i.e. 30-second chunk of audio)
# N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE = 30 * 16000 = 480000  (i.e. 480,000 samples in a 30-second chunk of audio)
# N_FRAMES = N_SAMPLES // HOP_LENGTH = 480000 // 160 = 3000  (i.e. 3000 frames in a mel spectrogram input)
#
# N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2 = 160 * 2 = 320
# FRAMES_PER_TOKEN = SAMPLE_RATE // HOP_LENGTH = 16000 // 160 = 100  (i.e. 10 ms per audio frame)
# TOKENS_PER_SECOND = SAMPLE_RATE // N_SAMPLES_PER_TOKEN = 16000 // 320 = 50  (i.e. 20 ms per audio token)
def get_sample_audio_features(
    config: WhisperConfig,
    device: torch.device,
    batch_size: int,
    sequence_length: int = 3000,
    use_fp16: bool = False,
):
    torch_dtype = torch.float16 if use_fp16 else torch.float32
    audio_features = torch.randn(batch_size, config.num_mel_bins, sequence_length, device=device, dtype=torch_dtype)
    return audio_features


# Create input_ids for decoder
# Shape is (batch_size, sequence_length) where sequence_length is the initial decoder sequence length
def get_sample_decoder_input_ids(
    config: WhisperConfig,
    device: torch.device,
    batch_size: int,
    sequence_length: int,
    use_int32: bool = True,
):
    torch_dtype = torch.int32 if use_int32 else torch.int64
    decoder_input_ids = torch.randint(
        low=0, high=config.vocab_size, size=(batch_size, sequence_length), device=device, dtype=torch_dtype
    )
    return decoder_input_ids


# Create encoder_hidden_states for decoder-init
# Shape is (batch_size, num_frames // 2, hidden_size)
def get_sample_encoder_hidden_states(
    config: WhisperConfig,
    device: torch.device,
    batch_size: int,
    use_fp16: bool = False,
):
    torch_dtype = torch.float16 if use_fp16 else torch.float32
    encoder_hidden_states = torch.randn(
        batch_size, config.max_source_positions, config.d_model, device=device, dtype=torch_dtype
    )
    return encoder_hidden_states


# Create past_key_values
# Self-attention KV caches are of shape (batch_size, num_heads, past_sequence_length, head_size)
# Cross-attention KV caches are of shape (batch_size, num_heads, num_frames // 2, head_size)
def get_sample_past_key_values(
    config: WhisperConfig,
    device: torch.device,
    batch_size: int,
    past_seq_len: int,
    use_fp16: bool = False,
):
    num_heads = config.decoder_attention_heads
    head_size = config.d_model // num_heads
    max_source_positions = (
        config.max_source_positions
    )  # equal to num_frames // 2 = encoder's sequence_length // 2 = 3000 // 2 = 1500
    torch_dtype = torch.float16 if use_fp16 else torch.float32
    self_attention_kv_caches = [
        (
            torch.rand(batch_size, num_heads, past_seq_len, head_size, device=device, dtype=torch_dtype),
            torch.rand(batch_size, num_heads, past_seq_len, head_size, device=device, dtype=torch_dtype),
        )
        for _ in range(config.num_hidden_layers)
    ]
    cross_attention_kv_caches = [
        (
            torch.rand(batch_size, num_heads, max_source_positions, head_size, device=device, dtype=torch_dtype),
            torch.rand(batch_size, num_heads, max_source_positions, head_size, device=device, dtype=torch_dtype),
        )
        for _ in range(config.num_hidden_layers)
    ]
    return flatten_past_key_values(self_attention_kv_caches, cross_attention_kv_caches)


# Flatten KV caches into pairs-of-4 where each pair is defined as:
# (self_attn_key_cache, self_attn_value_cache, cross_attn_key_cache, cross_attn_value_cache)
def flatten_past_key_values(
    self_attn_kv_caches: list[tuple[torch.Tensor, torch.Tensor]],
    cross_attn_kv_caches: list[tuple[torch.Tensor, torch.Tensor]],
):
    past_key_values = []
    for (self_k_cache, self_v_cache), (cross_k_cache, cross_v_cache) in zip(
        self_attn_kv_caches, cross_attn_kv_caches, strict=False
    ):
        layer_kv_caches = (self_k_cache, self_v_cache, cross_k_cache, cross_v_cache)
        past_key_values.append(layer_kv_caches)
    return past_key_values


# Group KV caches into two 1D lists where one list contains the self attention KV caches and
# one list contains the cross attention KV caches
def group_past_key_values(
    kv_caches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
):
    self_attn_kv_caches, cross_attn_kv_caches = [], []
    for self_k_cache, self_v_cache, cross_k_cache, cross_v_cache in kv_caches:
        self_attn_kv_caches.append(self_k_cache)
        self_attn_kv_caches.append(self_v_cache)
        cross_attn_kv_caches.append(cross_k_cache)
        cross_attn_kv_caches.append(cross_v_cache)
    return self_attn_kv_caches, cross_attn_kv_caches


# Create alignment heads for timestamps
# Shape is (num_alignment_heads, 2)
def get_sample_alignment_heads(
    config: WhisperConfig,
    device: torch.device,
    num_alignment_heads: int = 6,
    use_int32: bool = True,
):
    torch_dtype = torch.int32 if use_int32 else torch.int64
    alignment_heads = torch.ones((num_alignment_heads, 2), device=device, dtype=torch_dtype)
    return alignment_heads


# Create length of start-of-transcription sequence for timestamps
# Shape is (1)
def get_sample_sot_sequence_length(
    device: torch.device,
    sot_sequence_length: int,
    use_int32: bool = False,
):
    torch_dtype = torch.int32 if use_int32 else torch.int64
    sot_length = torch.tensor([sot_sequence_length], device=device, dtype=torch_dtype)
    return sot_length


# Create segment length for timestamps
# Shape is (1)
def get_sample_segment_length(
    device: torch.device,
    segment_length: int,
    use_int32: bool = False,
):
    torch_dtype = torch.int32 if use_int32 else torch.int64
    segment_size = torch.tensor([segment_length], device=device, dtype=torch_dtype)
    return segment_size


# Create QKs for timestamps
# Shape is (batch_size, num_heads, sequence_length, num_frames // 2)
def get_sample_QKs(  # noqa: N802
    config: WhisperConfig,
    device: torch.device,
    batch_size: int,
    sequence_length: int,
    use_fp16: bool = False,
):
    num_heads = config.decoder_attention_heads
    torch_dtype = torch.float16 if use_fp16 else torch.float32
    QKs = [  # noqa: N806
        torch.rand(
            batch_size, num_heads, sequence_length, config.max_source_positions, device=device, dtype=torch_dtype
        )
        for _ in range(config.num_hidden_layers)
    ]
    return QKs


# Create inputs for encoder component of Whisper
def get_sample_encoder_inputs(
    config: WhisperConfig,
    device: torch.device,
    batch_size: int,
    sequence_length: int = 3000,
    use_fp16: bool = False,
):
    audio_features = get_sample_audio_features(config, device, batch_size, sequence_length, use_fp16)
    return {"audio_features": audio_features}


# Create inputs for encoder component + first pass through decoder component of Whisper
def get_sample_encoder_decoder_init_inputs(
    config: WhisperConfig,
    device: torch.device,
    batch_size: int,
    decoder_sequence_length: int,
    encoder_sequence_length: int = 3000,
    use_fp16: bool = False,
    use_int32: bool = True,
):
    audio_features = get_sample_audio_features(config, device, batch_size, encoder_sequence_length, use_fp16)
    decoder_input_ids = get_sample_decoder_input_ids(config, device, batch_size, decoder_sequence_length, use_int32)
    return {"audio_features": audio_features, "decoder_input_ids": decoder_input_ids}


# Create inputs for decoder component of Whisper
# Inputs for first pass through the decoder (i.e. decoder-init): decoder_input_ids, encoder_hidden_states
# Inputs for subsequent passes through the decoder (i.e. decoder-with-past): decoder_input_ids, past_key_values
def get_sample_decoder_inputs(
    config: WhisperConfig,
    device: torch.device,
    batch_size: int,
    past_sequence_length: int,
    sequence_length: int,
    use_fp16: bool = False,
    use_int32: bool = True,
):
    decoder_input_ids = get_sample_decoder_input_ids(config, device, batch_size, sequence_length, use_int32)
    encoder_hidden_states = get_sample_encoder_hidden_states(config, device, batch_size, use_fp16)
    past_key_values = get_sample_past_key_values(config, device, batch_size, past_sequence_length, use_fp16)
    return {
        "decoder_input_ids": decoder_input_ids,
        "encoder_hidden_states": encoder_hidden_states,
        "past_key_values": past_key_values,
    }


# Create inputs for timestamps component of Whisper
def get_sample_jump_times_inputs(
    config: WhisperConfig,
    device: torch.device,
    batch_size: int,
    sequence_length: int,
    num_alignment_heads: int,
    sot_sequence_length: int,
    segment_length: int,
    use_fp16: bool = False,
    use_int32: bool = True,
):
    alignment_heads = get_sample_alignment_heads(config, device, num_alignment_heads, use_int32)
    # lengths need to be int64 because subsequent 'Slice' ops only take int64 inputs
    sot_sequence_length = get_sample_sot_sequence_length(device, sot_sequence_length)
    segment_length = get_sample_segment_length(device, segment_length)
    QKs = get_sample_QKs(config, device, batch_size, sequence_length, use_fp16)  # noqa: N806
    return {
        "alignment_heads": alignment_heads,
        "sot_sequence_length": sot_sequence_length,
        "segment_length": segment_length,
        "QKs": QKs,
    }


# Convert PyTorch inputs to ONNX Runtime inputs
def convert_inputs_for_ort(
    inputs: dict,
    model: InferenceSession,
):
    self_attn_kv_caches, cross_attn_kv_caches = None, None
    batch_size, num_heads, past_seq_len, head_size = 0, 0, 0, 0
    num_beams, max_seq_len = 1, 448
    if "past_key_values" in inputs:
        (self_attn_kv_caches, cross_attn_kv_caches) = group_past_key_values(inputs["past_key_values"])
        batch_size, num_heads, past_seq_len, head_size = self_attn_kv_caches[0].shape

    ort_inputs = {}
    model_inputs = list(map(lambda i: i.name, model.get_inputs()))  # noqa: C417
    use_buffer_sharing = "cache_indirection" in model_inputs
    for name in model_inputs:
        if name in {"audio_features", "encoder_input_ids"}:
            # Encoder input
            ort_inputs[name] = inputs["audio_features"].detach().cpu().numpy()
        elif name == "encoder_hidden_states":
            # Encoder output
            ort_inputs[name] = inputs["encoder_hidden_states"].detach().cpu().numpy()
        elif name in {"decoder_input_ids", "input_ids"}:
            # Decoder input
            ort_inputs[name] = inputs["decoder_input_ids"].detach().cpu().numpy()
        elif "past_key_self" in name or "past_value_self" in name:
            # Decoder input
            orig_kv_cache = self_attn_kv_caches.pop(0).detach().cpu().numpy()
            if use_buffer_sharing:
                new_kv_cache = np.zeros((batch_size, num_heads, max_seq_len, head_size), dtype=orig_kv_cache.dtype)
                new_kv_cache[:batch_size, :num_heads, :past_seq_len, :head_size] = orig_kv_cache
                ort_inputs[name] = new_kv_cache
            else:
                ort_inputs[name] = orig_kv_cache
        elif "past_key_cross" in name or "past_value_cross" in name:
            # Decoder input
            orig_kv_cache = cross_attn_kv_caches.pop(0).detach().cpu().numpy()
            ort_inputs[name] = orig_kv_cache
        elif name == "past_sequence_length":
            # Decoder input
            ort_inputs[name] = np.array([past_seq_len], dtype=np.int32)
        elif name == "cache_indirection":
            # Decoder input
            ort_inputs[name] = np.zeros((batch_size, num_beams, max_seq_len), dtype=np.int32)
        elif name == "alignment_heads":
            # Jump times input
            ort_inputs[name] = inputs["alignment_heads"].detach().cpu().numpy()
        elif name == "sot_sequence_length":
            # Jump times input
            ort_inputs[name] = inputs["sot_sequence_length"].detach().cpu().numpy()
        elif name == "segment_length":
            # Jump times input
            ort_inputs[name] = inputs["segment_length"].detach().cpu().numpy()
        elif "cross_qk" in name:
            # Jump times input
            ort_inputs[name] = inputs["QKs"].pop(0).detach().cpu().numpy()
        else:
            raise ValueError(f"Unknown name not recognized: {name}")

    return ort_inputs


# Get dynamic axes for all inputs and outputs to the model
def get_model_dynamic_axes(
    config: WhisperConfig,
    input_names: list[str],
    output_names: list[str],
):
    dynamic_axes = {}
    for name in input_names + output_names:
        if name in {"audio_features", "encoder_input_ids"}:
            # shape is (batch_size, num_mels, num_frames)
            dynamic_axes[name] = {0: "batch_size"}
        elif name in {"input_ids", "decoder_input_ids"}:
            # shape is (batch_size, sequence_length)
            dynamic_axes[name] = {0: "batch_size", 1: "sequence_length"}
        elif name == "alignment_heads":
            # shape is (num_alignment_heads, 2)
            dynamic_axes[name] = {0: "num_alignment_heads"}
        elif name in {"sot_sequence_length", "segment_length"}:
            # shape is (1)
            pass
        elif name == "logits":
            # shape is (batch_size, sequence_length, vocab_size)
            dynamic_axes[name] = {0: "batch_size", 1: "sequence_length"}
        elif name == "encoder_hidden_states":
            # shape is (batch_size, num_frames // 2, hidden_size)
            dynamic_axes[name] = {0: "batch_size"}
        elif "past_key_self" in name or "past_value_self" in name:
            # shape is (batch_size, num_heads, past_sequence_length, head_size)
            dynamic_axes[name] = {0: "batch_size", 2: "past_sequence_length"}
        elif "present_key_self" in name or "present_value_self" in name:
            # shape is (batch_size, num_heads, past_sequence_length + sequence_length, head_size),
            # which is equal to (batch_size, num_heads, total_sequence_length, head_size)
            dynamic_axes[name] = {0: "batch_size", 2: "total_sequence_length"}
        elif (
            "past_key_cross" in name
            or "past_value_cross" in name
            or "present_key_cross" in name
            or "present_value_cross" in name
        ):
            # shape is (batch_size, num_heads, num_frames // 2, head_size)
            dynamic_axes[name] = {0: "batch_size"}
        elif "cross_qk" in name:
            # shape is (batch_size, num_heads, source_sequence_length, target_sequence_length)
            dynamic_axes[name] = {0: "batch_size", 2: "sequence_length"}
        elif "jump_times" in name:
            # shape is (batch_size, max_length)
            dynamic_axes[name] = {0: "batch_size", 1: "max_length"}
        else:
            raise Exception(f"Unknown input or output name found: {name}")
    return dynamic_axes

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\__init__.py ===
from __future__ import annotations


# start delvewheel patch
def _delvewheel_patch_1_10_1():
    import os
    if os.path.isdir(libs_dir := os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'pandas.libs'))):
        os.add_dll_directory(libs_dir)


_delvewheel_patch_1_10_1()
del _delvewheel_patch_1_10_1
# end delvewheel patch

import os
import warnings

__docformat__ = "restructuredtext"

# Let users know if they're missing any of our hard dependencies
_hard_dependencies = ("numpy", "pytz", "dateutil")
_missing_dependencies = []

for _dependency in _hard_dependencies:
    try:
        __import__(_dependency)
    except ImportError as _e:  # pragma: no cover
        _missing_dependencies.append(f"{_dependency}: {_e}")

if _missing_dependencies:  # pragma: no cover
    raise ImportError(
        "Unable to import required dependencies:\n" + "\n".join(_missing_dependencies)
    )
del _hard_dependencies, _dependency, _missing_dependencies

try:
    # numpy compat
    from pandas.compat import (
        is_numpy_dev as _is_numpy_dev,  # pyright: ignore[reportUnusedImport] # noqa: F401
    )
except ImportError as _err:  # pragma: no cover
    _module = _err.name
    raise ImportError(
        f"C extension: {_module} not built. If you want to import "
        "pandas from the source directory, you may need to run "
        "'python setup.py build_ext' to build the C extensions first."
    ) from _err

from pandas._config import (
    get_option,
    set_option,
    reset_option,
    describe_option,
    option_context,
    options,
)

# let init-time option registration happen
import pandas.core.config_init  # pyright: ignore[reportUnusedImport] # noqa: F401

from pandas.core.api import (
    # dtype
    ArrowDtype,
    Int8Dtype,
    Int16Dtype,
    Int32Dtype,
    Int64Dtype,
    UInt8Dtype,
    UInt16Dtype,
    UInt32Dtype,
    UInt64Dtype,
    Float32Dtype,
    Float64Dtype,
    CategoricalDtype,
    PeriodDtype,
    IntervalDtype,
    DatetimeTZDtype,
    StringDtype,
    BooleanDtype,
    # missing
    NA,
    isna,
    isnull,
    notna,
    notnull,
    # indexes
    Index,
    CategoricalIndex,
    RangeIndex,
    MultiIndex,
    IntervalIndex,
    TimedeltaIndex,
    DatetimeIndex,
    PeriodIndex,
    IndexSlice,
    # tseries
    NaT,
    Period,
    period_range,
    Timedelta,
    timedelta_range,
    Timestamp,
    date_range,
    bdate_range,
    Interval,
    interval_range,
    DateOffset,
    # conversion
    to_numeric,
    to_datetime,
    to_timedelta,
    # misc
    Flags,
    Grouper,
    factorize,
    unique,
    value_counts,
    NamedAgg,
    array,
    Categorical,
    set_eng_float_format,
    Series,
    DataFrame,
)

from pandas.core.dtypes.dtypes import SparseDtype

from pandas.tseries.api import infer_freq
from pandas.tseries import offsets

from pandas.core.computation.api import eval

from pandas.core.reshape.api import (
    concat,
    lreshape,
    melt,
    wide_to_long,
    merge,
    merge_asof,
    merge_ordered,
    crosstab,
    pivot,
    pivot_table,
    get_dummies,
    from_dummies,
    cut,
    qcut,
)

from pandas import api, arrays, errors, io, plotting, tseries
from pandas import testing
from pandas.util._print_versions import show_versions

from pandas.io.api import (
    # excel
    ExcelFile,
    ExcelWriter,
    read_excel,
    # parsers
    read_csv,
    read_fwf,
    read_table,
    # pickle
    read_pickle,
    to_pickle,
    # pytables
    HDFStore,
    read_hdf,
    # sql
    read_sql,
    read_sql_query,
    read_sql_table,
    # misc
    read_clipboard,
    read_parquet,
    read_orc,
    read_feather,
    read_gbq,
    read_html,
    read_xml,
    read_json,
    read_stata,
    read_sas,
    read_spss,
)

from pandas.io.json._normalize import json_normalize

from pandas.util._tester import test

# use the closest tagged version if possible
_built_with_meson = False
try:
    from pandas._version_meson import (  # pyright: ignore [reportMissingImports]
        __version__,
        __git_version__,
    )

    _built_with_meson = True
except ImportError:
    from pandas._version import get_versions

    v = get_versions()
    __version__ = v.get("closest-tag", v["version"])
    __git_version__ = v.get("full-revisionid")
    del get_versions, v

# GH#55043 - deprecation of the data_manager option
if "PANDAS_DATA_MANAGER" in os.environ:
    warnings.warn(
        "The env variable PANDAS_DATA_MANAGER is set. The data_manager option is "
        "deprecated and will be removed in a future version. Only the BlockManager "
        "will be available. Unset this environment variable to silence this warning.",
        FutureWarning,
        stacklevel=2,
    )

del warnings, os

# module level doc-string
__doc__ = """
pandas - a powerful data analysis and manipulation library for Python
=====================================================================

**pandas** is a Python package providing fast, flexible, and expressive data
structures designed to make working with "relational" or "labeled" data both
easy and intuitive. It aims to be the fundamental high-level building block for
doing practical, **real world** data analysis in Python. Additionally, it has
the broader goal of becoming **the most powerful and flexible open source data
analysis / manipulation tool available in any language**. It is already well on
its way toward this goal.

Main Features
-------------
Here are just a few of the things that pandas does well:

  - Easy handling of missing data in floating point as well as non-floating
    point data.
  - Size mutability: columns can be inserted and deleted from DataFrame and
    higher dimensional objects
  - Automatic and explicit data alignment: objects can be explicitly aligned
    to a set of labels, or the user can simply ignore the labels and let
    `Series`, `DataFrame`, etc. automatically align the data for you in
    computations.
  - Powerful, flexible group by functionality to perform split-apply-combine
    operations on data sets, for both aggregating and transforming data.
  - Make it easy to convert ragged, differently-indexed data in other Python
    and NumPy data structures into DataFrame objects.
  - Intelligent label-based slicing, fancy indexing, and subsetting of large
    data sets.
  - Intuitive merging and joining data sets.
  - Flexible reshaping and pivoting of data sets.
  - Hierarchical labeling of axes (possible to have multiple labels per tick).
  - Robust IO tools for loading data from flat files (CSV and delimited),
    Excel files, databases, and saving/loading data from the ultrafast HDF5
    format.
  - Time series-specific functionality: date range generation and frequency
    conversion, moving window statistics, date shifting and lagging.
"""

# Use __all__ to let type checkers know what is part of the public API.
# Pandas is not (yet) a py.typed library: the public API is determined
# based on the documentation.
__all__ = [
    "ArrowDtype",
    "BooleanDtype",
    "Categorical",
    "CategoricalDtype",
    "CategoricalIndex",
    "DataFrame",
    "DateOffset",
    "DatetimeIndex",
    "DatetimeTZDtype",
    "ExcelFile",
    "ExcelWriter",
    "Flags",
    "Float32Dtype",
    "Float64Dtype",
    "Grouper",
    "HDFStore",
    "Index",
    "IndexSlice",
    "Int16Dtype",
    "Int32Dtype",
    "Int64Dtype",
    "Int8Dtype",
    "Interval",
    "IntervalDtype",
    "IntervalIndex",
    "MultiIndex",
    "NA",
    "NaT",
    "NamedAgg",
    "Period",
    "PeriodDtype",
    "PeriodIndex",
    "RangeIndex",
    "Series",
    "SparseDtype",
    "StringDtype",
    "Timedelta",
    "TimedeltaIndex",
    "Timestamp",
    "UInt16Dtype",
    "UInt32Dtype",
    "UInt64Dtype",
    "UInt8Dtype",
    "api",
    "array",
    "arrays",
    "bdate_range",
    "concat",
    "crosstab",
    "cut",
    "date_range",
    "describe_option",
    "errors",
    "eval",
    "factorize",
    "get_dummies",
    "from_dummies",
    "get_option",
    "infer_freq",
    "interval_range",
    "io",
    "isna",
    "isnull",
    "json_normalize",
    "lreshape",
    "melt",
    "merge",
    "merge_asof",
    "merge_ordered",
    "notna",
    "notnull",
    "offsets",
    "option_context",
    "options",
    "period_range",
    "pivot",
    "pivot_table",
    "plotting",
    "qcut",
    "read_clipboard",
    "read_csv",
    "read_excel",
    "read_feather",
    "read_fwf",
    "read_gbq",
    "read_hdf",
    "read_html",
    "read_json",
    "read_orc",
    "read_parquet",
    "read_pickle",
    "read_sas",
    "read_spss",
    "read_sql",
    "read_sql_query",
    "read_sql_table",
    "read_stata",
    "read_table",
    "read_xml",
    "reset_option",
    "set_eng_float_format",
    "set_option",
    "show_versions",
    "test",
    "testing",
    "timedelta_range",
    "to_datetime",
    "to_numeric",
    "to_pickle",
    "to_timedelta",
    "tseries",
    "unique",
    "value_counts",
    "wide_to_long",
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\action_chains.py ===
# Licensed to the Software Freedom Conservancy (SFC) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The SFC licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""The ActionChains implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from selenium.webdriver.remote.webelement import WebElement

from .actions.action_builder import ActionBuilder
from .actions.key_input import KeyInput
from .actions.pointer_input import PointerInput
from .actions.wheel_input import ScrollOrigin, WheelInput
from .utils import keys_to_typing

if TYPE_CHECKING:
    from selenium.webdriver.remote.webdriver import WebDriver

AnyDevice = Union[PointerInput, KeyInput, WheelInput]


class ActionChains:
    """ActionChains are a way to automate low level interactions such as mouse
    movements, mouse button actions, key press, and context menu interactions.
    This is useful for doing more complex actions like hover over and drag and
    drop.

    Generate user actions.
       When you call methods for actions on the ActionChains object,
       the actions are stored in a queue in the ActionChains object.
       When you call perform(), the events are fired in the order they
       are queued up.

    ActionChains can be used in a chain pattern::

        menu = driver.find_element(By.CSS_SELECTOR, ".nav")
        hidden_submenu = driver.find_element(By.CSS_SELECTOR, ".nav #submenu1")

        ActionChains(driver).move_to_element(menu).click(hidden_submenu).perform()

    Or actions can be queued up one by one, then performed.::

        menu = driver.find_element(By.CSS_SELECTOR, ".nav")
        hidden_submenu = driver.find_element(By.CSS_SELECTOR, ".nav #submenu1")

        actions = ActionChains(driver)
        actions.move_to_element(menu)
        actions.click(hidden_submenu)
        actions.perform()

    Either way, the actions are performed in the order they are called, one after
    another.
    """

    def __init__(self, driver: WebDriver, duration: int = 250, devices: list[AnyDevice] | None = None) -> None:
        """Creates a new ActionChains.

        :Args:
         - driver: The WebDriver instance which performs user actions.
         - duration: override the default 250 msecs of DEFAULT_MOVE_DURATION in PointerInput
        """
        self._driver = driver
        mouse = None
        keyboard = None
        wheel = None
        if devices is not None and isinstance(devices, list):
            for device in devices:
                if isinstance(device, PointerInput):
                    mouse = device
                if isinstance(device, KeyInput):
                    keyboard = device
                if isinstance(device, WheelInput):
                    wheel = device
        self.w3c_actions = ActionBuilder(driver, mouse=mouse, keyboard=keyboard, wheel=wheel, duration=duration)

    def perform(self) -> None:
        """Performs all stored actions."""
        self.w3c_actions.perform()

    def reset_actions(self) -> None:
        """Clears actions that are already stored locally and on the remote
        end."""
        self.w3c_actions.clear_actions()
        for device in self.w3c_actions.devices:
            device.clear_actions()

    def click(self, on_element: WebElement | None = None) -> ActionChains:
        """Clicks an element.

        :Args:
         - on_element: The element to click.
           If None, clicks on current mouse position.
        """
        if on_element:
            self.move_to_element(on_element)

        self.w3c_actions.pointer_action.click()
        self.w3c_actions.key_action.pause()
        self.w3c_actions.key_action.pause()

        return self

    def click_and_hold(self, on_element: WebElement | None = None) -> ActionChains:
        """Holds down the left mouse button on an element.

        :Args:
         - on_element: The element to mouse down.
           If None, clicks on current mouse position.
        """
        if on_element:
            self.move_to_element(on_element)

        self.w3c_actions.pointer_action.click_and_hold()
        self.w3c_actions.key_action.pause()

        return self

    def context_click(self, on_element: WebElement | None = None) -> ActionChains:
        """Performs a context-click (right click) on an element.

        :Args:
         - on_element: The element to context-click.
           If None, clicks on current mouse position.
        """
        if on_element:
            self.move_to_element(on_element)

        self.w3c_actions.pointer_action.context_click()
        self.w3c_actions.key_action.pause()
        self.w3c_actions.key_action.pause()

        return self

    def double_click(self, on_element: WebElement | None = None) -> ActionChains:
        """Double-clicks an element.

        :Args:
         - on_element: The element to double-click.
           If None, clicks on current mouse position.
        """
        if on_element:
            self.move_to_element(on_element)

        self.w3c_actions.pointer_action.double_click()
        for _ in range(4):
            self.w3c_actions.key_action.pause()

        return self

    def drag_and_drop(self, source: WebElement, target: WebElement) -> ActionChains:
        """Holds down the left mouse button on the source element, then moves
        to the target element and releases the mouse button.

        :Args:
         - source: The element to mouse down.
         - target: The element to mouse up.
        """
        self.click_and_hold(source)
        self.release(target)
        return self

    def drag_and_drop_by_offset(self, source: WebElement, xoffset: int, yoffset: int) -> ActionChains:
        """Holds down the left mouse button on the source element, then moves
        to the target offset and releases the mouse button.

        :Args:
         - source: The element to mouse down.
         - xoffset: X offset to move to.
         - yoffset: Y offset to move to.
        """
        self.click_and_hold(source)
        self.move_by_offset(xoffset, yoffset)
        self.release()
        return self

    def key_down(self, value: str, element: WebElement | None = None) -> ActionChains:
        """Sends a key press only, without releasing it. Should only be used
        with modifier keys (Control, Alt and Shift).

        :Args:
         - value: The modifier key to send. Values are defined in `Keys` class.
         - element: The element to send keys.
           If None, sends a key to current focused element.

        Example, pressing ctrl+c::

            ActionChains(driver).key_down(Keys.CONTROL).send_keys("c").key_up(Keys.CONTROL).perform()
        """
        if element:
            self.click(element)

        self.w3c_actions.key_action.key_down(value)
        self.w3c_actions.pointer_action.pause()

        return self

    def key_up(self, value: str, element: WebElement | None = None) -> ActionChains:
        """Releases a modifier key.

        :Args:
         - value: The modifier key to send. Values are defined in Keys class.
         - element: The element to send keys.
           If None, sends a key to current focused element.

        Example, pressing ctrl+c::

            ActionChains(driver).key_down(Keys.CONTROL).send_keys("c").key_up(Keys.CONTROL).perform()
        """
        if element:
            self.click(element)

        self.w3c_actions.key_action.key_up(value)
        self.w3c_actions.pointer_action.pause()

        return self

    def move_by_offset(self, xoffset: int, yoffset: int) -> ActionChains:
        """Moving the mouse to an offset from current mouse position.

        :Args:
         - xoffset: X offset to move to, as a positive or negative integer.
         - yoffset: Y offset to move to, as a positive or negative integer.
        """

        self.w3c_actions.pointer_action.move_by(xoffset, yoffset)
        self.w3c_actions.key_action.pause()

        return self

    def move_to_element(self, to_element: WebElement) -> ActionChains:
        """Moving the mouse to the middle of an element.

        :Args:
         - to_element: The WebElement to move to.
        """

        self.w3c_actions.pointer_action.move_to(to_element)
        self.w3c_actions.key_action.pause()

        return self

    def move_to_element_with_offset(self, to_element: WebElement, xoffset: int, yoffset: int) -> ActionChains:
        """Move the mouse by an offset of the specified element. Offsets are
        relative to the in-view center point of the element.

        :Args:
         - to_element: The WebElement to move to.
         - xoffset: X offset to move to, as a positive or negative integer.
         - yoffset: Y offset to move to, as a positive or negative integer.
        """

        self.w3c_actions.pointer_action.move_to(to_element, int(xoffset), int(yoffset))
        self.w3c_actions.key_action.pause()

        return self

    def pause(self, seconds: float | int) -> ActionChains:
        """Pause all inputs for the specified duration in seconds."""

        self.w3c_actions.pointer_action.pause(seconds)
        self.w3c_actions.key_action.pause(seconds)

        return self

    def release(self, on_element: WebElement | None = None) -> ActionChains:
        """Releasing a held mouse button on an element.

        :Args:
         - on_element: The element to mouse up.
           If None, releases on current mouse position.
        """
        if on_element:
            self.move_to_element(on_element)

        self.w3c_actions.pointer_action.release()
        self.w3c_actions.key_action.pause()

        return self

    def send_keys(self, *keys_to_send: str) -> ActionChains:
        """Sends keys to current focused element.

        :Args:
         - keys_to_send: The keys to send.  Modifier keys constants can be found in the
           'Keys' class.
        """
        typing = keys_to_typing(keys_to_send)

        for key in typing:
            self.key_down(key)
            self.key_up(key)

        return self

    def send_keys_to_element(self, element: WebElement, *keys_to_send: str) -> ActionChains:
        """Sends keys to an element.

        :Args:
         - element: The element to send keys.
         - keys_to_send: The keys to send.  Modifier keys constants can be found in the
           'Keys' class.
        """
        self.click(element)
        self.send_keys(*keys_to_send)
        return self

    def scroll_to_element(self, element: WebElement) -> ActionChains:
        """If the element is outside the viewport, scrolls the bottom of the
        element to the bottom of the viewport.

        :Args:
         - element: Which element to scroll into the viewport.
        """

        self.w3c_actions.wheel_action.scroll(origin=element)
        return self

    def scroll_by_amount(self, delta_x: int, delta_y: int) -> ActionChains:
        """Scrolls by provided amounts with the origin in the top left corner
        of the viewport.

        :Args:
         - delta_x: Distance along X axis to scroll using the wheel. A negative value scrolls left.
         - delta_y: Distance along Y axis to scroll using the wheel. A negative value scrolls up.
        """

        self.w3c_actions.wheel_action.scroll(delta_x=delta_x, delta_y=delta_y)
        return self

    def scroll_from_origin(self, scroll_origin: ScrollOrigin, delta_x: int, delta_y: int) -> ActionChains:
        """Scrolls by provided amount based on a provided origin. The scroll
        origin is either the center of an element or the upper left of the
        viewport plus any offsets. If the origin is an element, and the element
        is not in the viewport, the bottom of the element will first be
        scrolled to the bottom of the viewport.

        :Args:
         - origin: Where scroll originates (viewport or element center) plus provided offsets.
         - delta_x: Distance along X axis to scroll using the wheel. A negative value scrolls left.
         - delta_y: Distance along Y axis to scroll using the wheel. A negative value scrolls up.

         :Raises: If the origin with offset is outside the viewport.
          - MoveTargetOutOfBoundsException - If the origin with offset is outside the viewport.
        """

        if not isinstance(scroll_origin, ScrollOrigin):
            raise TypeError(f"Expected object of type ScrollOrigin, got: {type(scroll_origin)}")

        self.w3c_actions.wheel_action.scroll(
            origin=scroll_origin.origin,
            x=scroll_origin.x_offset,
            y=scroll_origin.y_offset,
            delta_x=delta_x,
            delta_y=delta_y,
        )
        return self

    # Context manager so ActionChains can be used in a 'with .. as' statements.

    def __enter__(self) -> ActionChains:
        return self  # Return created instance of self.

    def __exit__(self, _type, _value, _traceback) -> None:
        pass  # Do nothing, does not require additional cleanup.

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\orm\evaluator.py ===
# orm/evaluator.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors

"""Evaluation functions used **INTERNALLY** by ORM DML use cases.


This module is **private, for internal use by SQLAlchemy**.

.. versionchanged:: 2.0.4 renamed ``EvaluatorCompiler`` to
   ``_EvaluatorCompiler``.

"""


from __future__ import annotations

from typing import Type

from . import exc as orm_exc
from .base import LoaderCallableStatus
from .base import PassiveFlag
from .. import exc
from .. import inspect
from ..sql import and_
from ..sql import operators
from ..sql.sqltypes import Concatenable
from ..sql.sqltypes import Integer
from ..sql.sqltypes import Numeric
from ..util import warn_deprecated


class UnevaluatableError(exc.InvalidRequestError):
    pass


class _NoObject(operators.ColumnOperators):
    def operate(self, *arg, **kw):
        return None

    def reverse_operate(self, *arg, **kw):
        return None


class _ExpiredObject(operators.ColumnOperators):
    def operate(self, *arg, **kw):
        return self

    def reverse_operate(self, *arg, **kw):
        return self


_NO_OBJECT = _NoObject()
_EXPIRED_OBJECT = _ExpiredObject()


class _EvaluatorCompiler:
    def __init__(self, target_cls=None):
        self.target_cls = target_cls

    def process(self, clause, *clauses):
        if clauses:
            clause = and_(clause, *clauses)

        meth = getattr(self, f"visit_{clause.__visit_name__}", None)
        if not meth:
            raise UnevaluatableError(
                f"Cannot evaluate {type(clause).__name__}"
            )
        return meth(clause)

    def visit_grouping(self, clause):
        return self.process(clause.element)

    def visit_null(self, clause):
        return lambda obj: None

    def visit_false(self, clause):
        return lambda obj: False

    def visit_true(self, clause):
        return lambda obj: True

    def visit_column(self, clause):
        try:
            parentmapper = clause._annotations["parentmapper"]
        except KeyError as ke:
            raise UnevaluatableError(
                f"Cannot evaluate column: {clause}"
            ) from ke

        if self.target_cls and not issubclass(
            self.target_cls, parentmapper.class_
        ):
            raise UnevaluatableError(
                "Can't evaluate criteria against "
                f"alternate class {parentmapper.class_}"
            )

        parentmapper._check_configure()

        # we'd like to use "proxy_key" annotation to get the "key", however
        # in relationship primaryjoin cases proxy_key is sometimes deannotated
        # and sometimes apparently not present in the first place (?).
        # While I can stop it from being deannotated (though need to see if
        # this breaks other things), not sure right now  about cases where it's
        # not there in the first place.  can fix at some later point.
        # key = clause._annotations["proxy_key"]

        # for now, use the old way
        try:
            key = parentmapper._columntoproperty[clause].key
        except orm_exc.UnmappedColumnError as err:
            raise UnevaluatableError(
                f"Cannot evaluate expression: {err}"
            ) from err

        # note this used to fall back to a simple `getattr(obj, key)` evaluator
        # if impl was None; as of #8656, we ensure mappers are configured
        # so that impl is available
        impl = parentmapper.class_manager[key].impl

        def get_corresponding_attr(obj):
            if obj is None:
                return _NO_OBJECT
            state = inspect(obj)
            dict_ = state.dict

            value = impl.get(
                state, dict_, passive=PassiveFlag.PASSIVE_NO_FETCH
            )
            if value is LoaderCallableStatus.PASSIVE_NO_RESULT:
                return _EXPIRED_OBJECT
            return value

        return get_corresponding_attr

    def visit_tuple(self, clause):
        return self.visit_clauselist(clause)

    def visit_expression_clauselist(self, clause):
        return self.visit_clauselist(clause)

    def visit_clauselist(self, clause):
        evaluators = [self.process(clause) for clause in clause.clauses]

        dispatch = (
            f"visit_{clause.operator.__name__.rstrip('_')}_clauselist_op"
        )
        meth = getattr(self, dispatch, None)
        if meth:
            return meth(clause.operator, evaluators, clause)
        else:
            raise UnevaluatableError(
                f"Cannot evaluate clauselist with operator {clause.operator}"
            )

    def visit_binary(self, clause):
        eval_left = self.process(clause.left)
        eval_right = self.process(clause.right)

        dispatch = f"visit_{clause.operator.__name__.rstrip('_')}_binary_op"
        meth = getattr(self, dispatch, None)
        if meth:
            return meth(clause.operator, eval_left, eval_right, clause)
        else:
            raise UnevaluatableError(
                f"Cannot evaluate {type(clause).__name__} with "
                f"operator {clause.operator}"
            )

    def visit_or_clauselist_op(self, operator, evaluators, clause):
        def evaluate(obj):
            has_null = False
            for sub_evaluate in evaluators:
                value = sub_evaluate(obj)
                if value is _EXPIRED_OBJECT:
                    return _EXPIRED_OBJECT
                elif value:
                    return True
                has_null = has_null or value is None
            if has_null:
                return None
            return False

        return evaluate

    def visit_and_clauselist_op(self, operator, evaluators, clause):
        def evaluate(obj):
            for sub_evaluate in evaluators:
                value = sub_evaluate(obj)
                if value is _EXPIRED_OBJECT:
                    return _EXPIRED_OBJECT

                if not value:
                    if value is None or value is _NO_OBJECT:
                        return None
                    return False
            return True

        return evaluate

    def visit_comma_op_clauselist_op(self, operator, evaluators, clause):
        def evaluate(obj):
            values = []
            for sub_evaluate in evaluators:
                value = sub_evaluate(obj)
                if value is _EXPIRED_OBJECT:
                    return _EXPIRED_OBJECT
                elif value is None or value is _NO_OBJECT:
                    return None
                values.append(value)
            return tuple(values)

        return evaluate

    def visit_custom_op_binary_op(
        self, operator, eval_left, eval_right, clause
    ):
        if operator.python_impl:
            return self._straight_evaluate(
                operator, eval_left, eval_right, clause
            )
        else:
            raise UnevaluatableError(
                f"Custom operator {operator.opstring!r} can't be evaluated "
                "in Python unless it specifies a callable using "
                "`.python_impl`."
            )

    def visit_is_binary_op(self, operator, eval_left, eval_right, clause):
        def evaluate(obj):
            left_val = eval_left(obj)
            right_val = eval_right(obj)
            if left_val is _EXPIRED_OBJECT or right_val is _EXPIRED_OBJECT:
                return _EXPIRED_OBJECT
            return left_val == right_val

        return evaluate

    def visit_is_not_binary_op(self, operator, eval_left, eval_right, clause):
        def evaluate(obj):
            left_val = eval_left(obj)
            right_val = eval_right(obj)
            if left_val is _EXPIRED_OBJECT or right_val is _EXPIRED_OBJECT:
                return _EXPIRED_OBJECT
            return left_val != right_val

        return evaluate

    def _straight_evaluate(self, operator, eval_left, eval_right, clause):
        def evaluate(obj):
            left_val = eval_left(obj)
            right_val = eval_right(obj)
            if left_val is _EXPIRED_OBJECT or right_val is _EXPIRED_OBJECT:
                return _EXPIRED_OBJECT
            elif left_val is None or right_val is None:
                return None

            return operator(eval_left(obj), eval_right(obj))

        return evaluate

    def _straight_evaluate_numeric_only(
        self, operator, eval_left, eval_right, clause
    ):
        if clause.left.type._type_affinity not in (
            Numeric,
            Integer,
        ) or clause.right.type._type_affinity not in (Numeric, Integer):
            raise UnevaluatableError(
                f'Cannot evaluate math operator "{operator.__name__}" for '
                f"datatypes {clause.left.type}, {clause.right.type}"
            )

        return self._straight_evaluate(operator, eval_left, eval_right, clause)

    visit_add_binary_op = _straight_evaluate_numeric_only
    visit_mul_binary_op = _straight_evaluate_numeric_only
    visit_sub_binary_op = _straight_evaluate_numeric_only
    visit_mod_binary_op = _straight_evaluate_numeric_only
    visit_truediv_binary_op = _straight_evaluate_numeric_only
    visit_lt_binary_op = _straight_evaluate
    visit_le_binary_op = _straight_evaluate
    visit_ne_binary_op = _straight_evaluate
    visit_gt_binary_op = _straight_evaluate
    visit_ge_binary_op = _straight_evaluate
    visit_eq_binary_op = _straight_evaluate

    def visit_in_op_binary_op(self, operator, eval_left, eval_right, clause):
        return self._straight_evaluate(
            lambda a, b: a in b if a is not _NO_OBJECT else None,
            eval_left,
            eval_right,
            clause,
        )

    def visit_not_in_op_binary_op(
        self, operator, eval_left, eval_right, clause
    ):
        return self._straight_evaluate(
            lambda a, b: a not in b if a is not _NO_OBJECT else None,
            eval_left,
            eval_right,
            clause,
        )

    def visit_concat_op_binary_op(
        self, operator, eval_left, eval_right, clause
    ):

        if not issubclass(
            clause.left.type._type_affinity, Concatenable
        ) or not issubclass(clause.right.type._type_affinity, Concatenable):
            raise UnevaluatableError(
                f"Cannot evaluate concatenate operator "
                f'"{operator.__name__}" for '
                f"datatypes {clause.left.type}, {clause.right.type}"
            )

        return self._straight_evaluate(
            lambda a, b: a + b, eval_left, eval_right, clause
        )

    def visit_startswith_op_binary_op(
        self, operator, eval_left, eval_right, clause
    ):
        return self._straight_evaluate(
            lambda a, b: a.startswith(b), eval_left, eval_right, clause
        )

    def visit_endswith_op_binary_op(
        self, operator, eval_left, eval_right, clause
    ):
        return self._straight_evaluate(
            lambda a, b: a.endswith(b), eval_left, eval_right, clause
        )

    def visit_unary(self, clause):
        eval_inner = self.process(clause.element)
        if clause.operator is operators.inv:

            def evaluate(obj):
                value = eval_inner(obj)
                if value is _EXPIRED_OBJECT:
                    return _EXPIRED_OBJECT
                elif value is None:
                    return None
                return not value

            return evaluate
        raise UnevaluatableError(
            f"Cannot evaluate {type(clause).__name__} "
            f"with operator {clause.operator}"
        )

    def visit_bindparam(self, clause):
        if clause.callable:
            val = clause.callable()
        else:
            val = clause.value
        return lambda obj: val


def __getattr__(name: str) -> Type[_EvaluatorCompiler]:
    if name == "EvaluatorCompiler":
        warn_deprecated(
            "Direct use of 'EvaluatorCompiler' is not supported, and this "
            "name will be removed in a future release.  "
            "'_EvaluatorCompiler' is for internal use only",
            "2.0",
        )
        return _EvaluatorCompiler
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\plotting\pygletplot\plot_mode_base.py ===
import pyglet.gl as pgl
from sympy.core import S
from sympy.plotting.pygletplot.color_scheme import ColorScheme
from sympy.plotting.pygletplot.plot_mode import PlotMode
from sympy.utilities.iterables import is_sequence
from time import sleep
from threading import Thread, Event, RLock
import warnings


class PlotModeBase(PlotMode):
    """
    Intended parent class for plotting
    modes. Provides base functionality
    in conjunction with its parent,
    PlotMode.
    """

    ##
    ## Class-Level Attributes
    ##

    """
    The following attributes are meant
    to be set at the class level, and serve
    as parameters to the plot mode registry
    (in PlotMode). See plot_modes.py for
    concrete examples.
    """

    """
    i_vars
        'x' for Cartesian2D
        'xy' for Cartesian3D
        etc.

    d_vars
        'y' for Cartesian2D
        'r' for Polar
        etc.
    """
    i_vars, d_vars = '', ''

    """
    intervals
        Default intervals for each i_var, and in the
        same order. Specified [min, max, steps].
        No variable can be given (it is bound later).
    """
    intervals = []

    """
    aliases
        A list of strings which can be used to
        access this mode.
        'cartesian' for Cartesian2D and Cartesian3D
        'polar' for Polar
        'cylindrical', 'polar' for Cylindrical

        Note that _init_mode chooses the first alias
        in the list as the mode's primary_alias, which
        will be displayed to the end user in certain
        contexts.
    """
    aliases = []

    """
    is_default
        Whether to set this mode as the default
        for arguments passed to PlotMode() containing
        the same number of d_vars as this mode and
        at most the same number of i_vars.
    """
    is_default = False

    """
    All of the above attributes are defined in PlotMode.
    The following ones are specific to PlotModeBase.
    """

    """
    A list of the render styles. Do not modify.
    """
    styles = {'wireframe': 1, 'solid': 2, 'both': 3}

    """
    style_override
        Always use this style if not blank.
    """
    style_override = ''

    """
    default_wireframe_color
    default_solid_color
        Can be used when color is None or being calculated.
        Used by PlotCurve and PlotSurface, but not anywhere
        in PlotModeBase.
    """

    default_wireframe_color = (0.85, 0.85, 0.85)
    default_solid_color = (0.6, 0.6, 0.9)
    default_rot_preset = 'xy'

    ##
    ## Instance-Level Attributes
    ##

    ## 'Abstract' member functions
    def _get_evaluator(self):
        if self.use_lambda_eval:
            try:
                e = self._get_lambda_evaluator()
                return e
            except Exception:
                warnings.warn("\nWarning: creating lambda evaluator failed. "
                       "Falling back on SymPy subs evaluator.")
        return self._get_sympy_evaluator()

    def _get_sympy_evaluator(self):
        raise NotImplementedError()

    def _get_lambda_evaluator(self):
        raise NotImplementedError()

    def _on_calculate_verts(self):
        raise NotImplementedError()

    def _on_calculate_cverts(self):
        raise NotImplementedError()

    ## Base member functions
    def __init__(self, *args, bounds_callback=None, **kwargs):
        self.verts = []
        self.cverts = []
        self.bounds = [[S.Infinity, S.NegativeInfinity, 0],
                       [S.Infinity, S.NegativeInfinity, 0],
                       [S.Infinity, S.NegativeInfinity, 0]]
        self.cbounds = [[S.Infinity, S.NegativeInfinity, 0],
                        [S.Infinity, S.NegativeInfinity, 0],
                        [S.Infinity, S.NegativeInfinity, 0]]

        self._draw_lock = RLock()

        self._calculating_verts = Event()
        self._calculating_cverts = Event()
        self._calculating_verts_pos = 0.0
        self._calculating_verts_len = 0.0
        self._calculating_cverts_pos = 0.0
        self._calculating_cverts_len = 0.0

        self._max_render_stack_size = 3
        self._draw_wireframe = [-1]
        self._draw_solid = [-1]

        self._style = None
        self._color = None

        self.predraw = []
        self.postdraw = []

        self.use_lambda_eval = self.options.pop('use_sympy_eval', None) is None
        self.style = self.options.pop('style', '')
        self.color = self.options.pop('color', 'rainbow')
        self.bounds_callback = bounds_callback

        self._on_calculate()

    def synchronized(f):
        def w(self, *args, **kwargs):
            self._draw_lock.acquire()
            try:
                r = f(self, *args, **kwargs)
                return r
            finally:
                self._draw_lock.release()
        return w

    @synchronized
    def push_wireframe(self, function):
        """
        Push a function which performs gl commands
        used to build a display list. (The list is
        built outside of the function)
        """
        assert callable(function)
        self._draw_wireframe.append(function)
        if len(self._draw_wireframe) > self._max_render_stack_size:
            del self._draw_wireframe[1]  # leave marker element

    @synchronized
    def push_solid(self, function):
        """
        Push a function which performs gl commands
        used to build a display list. (The list is
        built outside of the function)
        """
        assert callable(function)
        self._draw_solid.append(function)
        if len(self._draw_solid) > self._max_render_stack_size:
            del self._draw_solid[1]  # leave marker element

    def _create_display_list(self, function):
        dl = pgl.glGenLists(1)
        pgl.glNewList(dl, pgl.GL_COMPILE)
        function()
        pgl.glEndList()
        return dl

    def _render_stack_top(self, render_stack):
        top = render_stack[-1]
        if top == -1:
            return -1  # nothing to display
        elif callable(top):
            dl = self._create_display_list(top)
            render_stack[-1] = (dl, top)
            return dl  # display newly added list
        elif len(top) == 2:
            if pgl.GL_TRUE == pgl.glIsList(top[0]):
                return top[0]  # display stored list
            dl = self._create_display_list(top[1])
            render_stack[-1] = (dl, top[1])
            return dl  # display regenerated list

    def _draw_solid_display_list(self, dl):
        pgl.glPushAttrib(pgl.GL_ENABLE_BIT | pgl.GL_POLYGON_BIT)
        pgl.glPolygonMode(pgl.GL_FRONT_AND_BACK, pgl.GL_FILL)
        pgl.glCallList(dl)
        pgl.glPopAttrib()

    def _draw_wireframe_display_list(self, dl):
        pgl.glPushAttrib(pgl.GL_ENABLE_BIT | pgl.GL_POLYGON_BIT)
        pgl.glPolygonMode(pgl.GL_FRONT_AND_BACK, pgl.GL_LINE)
        pgl.glEnable(pgl.GL_POLYGON_OFFSET_LINE)
        pgl.glPolygonOffset(-0.005, -50.0)
        pgl.glCallList(dl)
        pgl.glPopAttrib()

    @synchronized
    def draw(self):
        for f in self.predraw:
            if callable(f):
                f()
        if self.style_override:
            style = self.styles[self.style_override]
        else:
            style = self.styles[self._style]
        # Draw solid component if style includes solid
        if style & 2:
            dl = self._render_stack_top(self._draw_solid)
            if dl > 0 and pgl.GL_TRUE == pgl.glIsList(dl):
                self._draw_solid_display_list(dl)
        # Draw wireframe component if style includes wireframe
        if style & 1:
            dl = self._render_stack_top(self._draw_wireframe)
            if dl > 0 and pgl.GL_TRUE == pgl.glIsList(dl):
                self._draw_wireframe_display_list(dl)
        for f in self.postdraw:
            if callable(f):
                f()

    def _on_change_color(self, color):
        Thread(target=self._calculate_cverts).start()

    def _on_calculate(self):
        Thread(target=self._calculate_all).start()

    def _calculate_all(self):
        self._calculate_verts()
        self._calculate_cverts()

    def _calculate_verts(self):
        if self._calculating_verts.is_set():
            return
        self._calculating_verts.set()
        try:
            self._on_calculate_verts()
        finally:
            self._calculating_verts.clear()
        if callable(self.bounds_callback):
            self.bounds_callback()

    def _calculate_cverts(self):
        if self._calculating_verts.is_set():
            return
        while self._calculating_cverts.is_set():
            sleep(0)  # wait for previous calculation
        self._calculating_cverts.set()
        try:
            self._on_calculate_cverts()
        finally:
            self._calculating_cverts.clear()

    def _get_calculating_verts(self):
        return self._calculating_verts.is_set()

    def _get_calculating_verts_pos(self):
        return self._calculating_verts_pos

    def _get_calculating_verts_len(self):
        return self._calculating_verts_len

    def _get_calculating_cverts(self):
        return self._calculating_cverts.is_set()

    def _get_calculating_cverts_pos(self):
        return self._calculating_cverts_pos

    def _get_calculating_cverts_len(self):
        return self._calculating_cverts_len

    ## Property handlers
    def _get_style(self):
        return self._style

    @synchronized
    def _set_style(self, v):
        if v is None:
            return
        if v == '':
            step_max = 0
            for i in self.intervals:
                if i.v_steps is None:
                    continue
                step_max = max([step_max, int(i.v_steps)])
            v = ['both', 'solid'][step_max > 40]
        if v not in self.styles:
            raise ValueError("v should be there in self.styles")
        if v == self._style:
            return
        self._style = v

    def _get_color(self):
        return self._color

    @synchronized
    def _set_color(self, v):
        try:
            if v is not None:
                if is_sequence(v):
                    v = ColorScheme(*v)
                else:
                    v = ColorScheme(v)
            if repr(v) == repr(self._color):
                return
            self._on_change_color(v)
            self._color = v
        except Exception as e:
            raise RuntimeError("Color change failed. "
                                "Reason: %s" % (str(e)))

    style = property(_get_style, _set_style)
    color = property(_get_color, _set_color)

    calculating_verts = property(_get_calculating_verts)
    calculating_verts_pos = property(_get_calculating_verts_pos)
    calculating_verts_len = property(_get_calculating_verts_len)

    calculating_cverts = property(_get_calculating_cverts)
    calculating_cverts_pos = property(_get_calculating_cverts_pos)
    calculating_cverts_len = property(_get_calculating_cverts_len)

    ## String representations

    def __str__(self):
        f = ", ".join(str(d) for d in self.d_vars)
        o = "'mode=%s'" % (self.primary_alias)
        return ", ".join([f, o])

    def __repr__(self):
        f = ", ".join(str(d) for d in self.d_vars)
        i = ", ".join(str(i) for i in self.intervals)
        d = [('mode', self.primary_alias),
             ('color', str(self.color)),
             ('style', str(self.style))]

        o = "'%s'" % ("; ".join("%s=%s" % (k, v)
                                for k, v in d if v != 'None'))
        return ", ".join([f, i, o])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\convert_onnx_models_to_ort.py ===
#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from __future__ import annotations

import argparse
import contextlib
import enum
import os
import pathlib
import tempfile

import onnxruntime as ort

from .file_utils import files_from_file_or_dir, path_match_suffix_ignore_case
from .onnx_model_utils import get_optimization_level
from .ort_format_model import create_config_from_models


class OptimizationStyle(enum.Enum):
    Fixed = 0
    Runtime = 1


def _optimization_suffix(optimization_level_str: str, optimization_style: OptimizationStyle, suffix: str):
    return "{}{}{}".format(
        f".{optimization_level_str}" if optimization_level_str != "all" else "",
        ".with_runtime_opt" if optimization_style == OptimizationStyle.Runtime else "",
        suffix,
    )


def _create_config_file_path(
    model_path_or_dir: pathlib.Path,
    output_dir: pathlib.Path | None,
    optimization_level_str: str,
    optimization_style: OptimizationStyle,
    enable_type_reduction: bool,
):
    config_name = "{}{}".format(
        "required_operators_and_types" if enable_type_reduction else "required_operators",
        _optimization_suffix(optimization_level_str, optimization_style, ".config"),
    )

    if model_path_or_dir.is_dir():
        return (output_dir or model_path_or_dir) / config_name

    model_config_path = model_path_or_dir.with_suffix(f".{config_name}")

    if output_dir is not None:
        return output_dir / model_config_path.name

    return model_config_path


def _create_session_options(
    optimization_level: ort.GraphOptimizationLevel,
    output_model_path: pathlib.Path,
    custom_op_library: pathlib.Path,
    session_options_config_entries: dict[str, str],
):
    so = ort.SessionOptions()
    so.optimized_model_filepath = str(output_model_path)
    so.graph_optimization_level = optimization_level

    if custom_op_library:
        so.register_custom_ops_library(str(custom_op_library))

    for key, value in session_options_config_entries.items():
        so.add_session_config_entry(key, value)

    return so


def _convert(
    model_path_or_dir: pathlib.Path,
    output_dir: pathlib.Path | None,
    optimization_level_str: str,
    optimization_style: OptimizationStyle,
    custom_op_library: pathlib.Path,
    create_optimized_onnx_model: bool,
    allow_conversion_failures: bool,
    target_platform: str,
    session_options_config_entries: dict[str, str],
) -> list[pathlib.Path]:
    model_dir = model_path_or_dir if model_path_or_dir.is_dir() else model_path_or_dir.parent
    output_dir = output_dir or model_dir

    optimization_level = get_optimization_level(optimization_level_str)

    def is_model_file_to_convert(file_path: pathlib.Path):
        if not path_match_suffix_ignore_case(file_path, ".onnx"):
            return False
        # ignore any files with an extension of .optimized.onnx which are presumably from previous executions
        # of this script
        if path_match_suffix_ignore_case(file_path, ".optimized.onnx"):
            print(f"Ignoring '{file_path}'")
            return False
        return True

    models = files_from_file_or_dir(model_path_or_dir, is_model_file_to_convert)

    if len(models) == 0:
        raise ValueError(f"No model files were found in '{model_path_or_dir}'")

    providers = ["CPUExecutionProvider"]

    # if the optimization level is 'all' we manually exclude the NCHWc transformer. It's not applicable to ARM
    # devices, and creates a device specific model which won't run on all hardware.
    # If someone really really really wants to run it they could manually create an optimized onnx model first,
    # or they could comment out this code.
    optimizer_filter = None
    if optimization_level == ort.GraphOptimizationLevel.ORT_ENABLE_ALL and target_platform != "amd64":
        optimizer_filter = ["NchwcTransformer"]

    converted_models = []

    for model in models:
        try:
            relative_model_path = model.relative_to(model_dir)

            (output_dir / relative_model_path).parent.mkdir(parents=True, exist_ok=True)

            ort_target_path = (output_dir / relative_model_path).with_suffix(
                _optimization_suffix(optimization_level_str, optimization_style, ".ort")
            )

            if create_optimized_onnx_model:
                # Create an ONNX file with the same optimization level that will be used for the ORT format file.
                # This allows the ONNX equivalent of the ORT format model to be easily viewed in Netron.
                # If runtime optimizations are saved in the ORT format model, there may be some difference in the
                # graphs at runtime between the ORT format model and this saved ONNX model.
                optimized_target_path = (output_dir / relative_model_path).with_suffix(
                    _optimization_suffix(optimization_level_str, optimization_style, ".optimized.onnx")
                )
                so = _create_session_options(
                    optimization_level, optimized_target_path, custom_op_library, session_options_config_entries
                )
                if optimization_style == OptimizationStyle.Runtime:
                    # Limit the optimizations to those that can run in a model with runtime optimizations.
                    so.add_session_config_entry("optimization.minimal_build_optimizations", "apply")

                print(f"Saving optimized ONNX model {model} to {optimized_target_path}")
                _ = ort.InferenceSession(
                    str(model), sess_options=so, providers=providers, disabled_optimizers=optimizer_filter
                )

            # Load ONNX model, optimize, and save to ORT format
            so = _create_session_options(
                optimization_level, ort_target_path, custom_op_library, session_options_config_entries
            )
            so.add_session_config_entry("session.save_model_format", "ORT")
            if optimization_style == OptimizationStyle.Runtime:
                so.add_session_config_entry("optimization.minimal_build_optimizations", "save")

            print(f"Converting optimized ONNX model {model} to ORT format model {ort_target_path}")
            _ = ort.InferenceSession(
                str(model), sess_options=so, providers=providers, disabled_optimizers=optimizer_filter
            )

            converted_models.append(ort_target_path)

            # orig_size = os.path.getsize(onnx_target_path)
            # new_size = os.path.getsize(ort_target_path)
            # print("Serialized {} to {}. Sizes: orig={} new={} diff={} new:old={:.4f}:1.0".format(
            #     onnx_target_path, ort_target_path, orig_size, new_size, new_size - orig_size, new_size / orig_size))
        except Exception as e:
            print(f"Error converting {model}: {e}")
            if not allow_conversion_failures:
                raise

    print(f"Converted {len(converted_models)}/{len(models)} models successfully.")

    return converted_models


def parse_args():
    parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description="""Convert the ONNX format model/s in the provided directory to ORT format models.
        All files with a `.onnx` extension will be processed. For each one, an ORT format model will be created in the
        given output directory, if specified, or the same directory.
        A configuration file will also be created containing the list of required operators for all
        converted models. This configuration file should be used as input to the minimal build via the
        `--include_ops_by_config` parameter.
        """,
    )

    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        help="Provide an output directory for the converted model/s and configuration file. "
        "If unspecified, the converted ORT format model/s will be in the same directory as the ONNX model/s.",
    )

    parser.add_argument(
        "--optimization_style",
        nargs="+",
        default=[OptimizationStyle.Fixed.name, OptimizationStyle.Runtime.name],
        choices=[e.name for e in OptimizationStyle],
        help="Style of optimization to perform on the ORT format model. "
        "Multiple values may be provided. The conversion will run once for each value. "
        "The general guidance is to use models optimized with "
        f"'{OptimizationStyle.Runtime.name}' style when using NNAPI or CoreML and "
        f"'{OptimizationStyle.Fixed.name}' style otherwise. "
        f"'{OptimizationStyle.Fixed.name}': Run optimizations directly before saving the ORT "
        "format model. This bakes in any platform-specific optimizations. "
        f"'{OptimizationStyle.Runtime.name}': Run basic optimizations directly and save certain "
        "other optimizations to be applied at runtime if possible. This is useful when using a "
        "compiling EP like NNAPI or CoreML that may run an unknown (at model conversion time) "
        "number of nodes. The saved optimizations can further optimize nodes not assigned to the "
        "compiling EP at runtime.",
    )

    parser.add_argument(
        "--enable_type_reduction",
        action="store_true",
        help="Add operator specific type information to the configuration file to potentially reduce "
        "the types supported by individual operator implementations.",
    )

    parser.add_argument(
        "--custom_op_library",
        type=pathlib.Path,
        default=None,
        help="Provide path to shared library containing custom operator kernels to register.",
    )

    parser.add_argument(
        "--save_optimized_onnx_model",
        action="store_true",
        help="Save the optimized version of each ONNX model. "
        "This will have the same level of optimizations applied as the ORT format model.",
    )

    parser.add_argument(
        "--allow_conversion_failures",
        action="store_true",
        help="Whether to proceed after encountering model conversion failures.",
    )

    parser.add_argument(
        "--target_platform",
        type=str,
        default=None,
        choices=["arm", "amd64"],
        help="Specify the target platform where the exported model will be used. "
        "This parameter can be used to choose between platform-specific options, "
        "such as QDQIsInt8Allowed(arm), NCHWc (amd64) and NHWC (arm/amd64) format, different "
        "optimizer level options, etc.",
    )

    parser.add_argument(
        "model_path_or_dir",
        type=pathlib.Path,
        help="Provide path to ONNX model or directory containing ONNX model/s to convert. "
        "All files with a .onnx extension, including those in subdirectories, will be "
        "processed.",
    )

    parsed_args = parser.parse_args()
    parsed_args.optimization_style = [OptimizationStyle[style_str] for style_str in parsed_args.optimization_style]
    return parsed_args


def convert_onnx_models_to_ort(
    model_path_or_dir: pathlib.Path,
    output_dir: pathlib.Path | None = None,
    optimization_styles: list[OptimizationStyle] | None = None,
    custom_op_library_path: pathlib.Path | None = None,
    target_platform: str | None = None,
    save_optimized_onnx_model: bool = False,
    allow_conversion_failures: bool = False,
    enable_type_reduction: bool = False,
):
    if output_dir is not None:
        if not output_dir.is_dir():
            output_dir.mkdir(parents=True)
        output_dir = output_dir.resolve(strict=True)

    optimization_styles = optimization_styles or []

    # setting optimization level is not expected to be needed by typical users, but it can be set with this
    # environment variable
    optimization_level_str = os.getenv("ORT_CONVERT_ONNX_MODELS_TO_ORT_OPTIMIZATION_LEVEL", "all")
    model_path_or_dir = model_path_or_dir.resolve()
    custom_op_library = custom_op_library_path.resolve() if custom_op_library_path else None

    if not model_path_or_dir.is_dir() and not model_path_or_dir.is_file():
        raise FileNotFoundError(f"Model path '{model_path_or_dir}' is not a file or directory.")

    if custom_op_library and not custom_op_library.is_file():
        raise FileNotFoundError(f"Unable to find custom operator library '{custom_op_library}'")

    session_options_config_entries = {}

    if target_platform is not None and target_platform == "arm":
        session_options_config_entries["session.qdqisint8allowed"] = "1"
    else:
        session_options_config_entries["session.qdqisint8allowed"] = "0"

    for optimization_style in optimization_styles:
        print(
            f"Converting models with optimization style '{optimization_style.name}' and level '{optimization_level_str}'"
        )

        converted_models = _convert(
            model_path_or_dir=model_path_or_dir,
            output_dir=output_dir,
            optimization_level_str=optimization_level_str,
            optimization_style=optimization_style,
            custom_op_library=custom_op_library,
            create_optimized_onnx_model=save_optimized_onnx_model,
            allow_conversion_failures=allow_conversion_failures,
            target_platform=target_platform,
            session_options_config_entries=session_options_config_entries,
        )

        with contextlib.ExitStack() as context_stack:
            if optimization_style == OptimizationStyle.Runtime:
                # Convert models again without runtime optimizations.
                # Runtime optimizations may not end up being applied, so we need to use both converted models with and
                # without runtime optimizations to get a complete set of ops that may be needed for the config file.
                model_dir = model_path_or_dir if model_path_or_dir.is_dir() else model_path_or_dir.parent
                temp_output_dir = context_stack.enter_context(
                    tempfile.TemporaryDirectory(dir=model_dir, suffix=".without_runtime_opt")
                )
                session_options_config_entries_for_second_conversion = session_options_config_entries.copy()
                # Limit the optimizations to those that can run in a model with runtime optimizations.
                session_options_config_entries_for_second_conversion["optimization.minimal_build_optimizations"] = (
                    "apply"
                )

                print(
                    "Converting models again without runtime optimizations to generate a complete config file. "
                    "These converted models are temporary and will be deleted."
                )
                converted_models += _convert(
                    model_path_or_dir=model_path_or_dir,
                    output_dir=temp_output_dir,
                    optimization_level_str=optimization_level_str,
                    optimization_style=OptimizationStyle.Fixed,
                    custom_op_library=custom_op_library,
                    create_optimized_onnx_model=False,  # not useful as they would be created in a temp directory
                    allow_conversion_failures=allow_conversion_failures,
                    target_platform=target_platform,
                    session_options_config_entries=session_options_config_entries_for_second_conversion,
                )

            print(
                f"Generating config file from ORT format models with optimization style '{optimization_style.name}' and level '{optimization_level_str}'"
            )

            config_file = _create_config_file_path(
                model_path_or_dir,
                output_dir,
                optimization_level_str,
                optimization_style,
                enable_type_reduction,
            )

            create_config_from_models(converted_models, config_file, enable_type_reduction)


if __name__ == "__main__":
    args = parse_args()
    convert_onnx_models_to_ort(
        args.model_path_or_dir,
        output_dir=args.output_dir,
        optimization_styles=args.optimization_style,
        custom_op_library_path=args.custom_op_library,
        target_platform=args.target_platform,
        save_optimized_onnx_model=args.save_optimized_onnx_model,
        allow_conversion_failures=args.allow_conversion_failures,
        enable_type_reduction=args.enable_type_reduction,
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\units\systems\si.py ===
"""
SI unit system.
Based on MKSA, which stands for "meter, kilogram, second, ampere".
Added kelvin, candela and mole.

"""

from __future__ import annotations

from sympy.physics.units import DimensionSystem, Dimension, dHg0

from sympy.physics.units.quantities import Quantity

from sympy.core.numbers import (Rational, pi)
from sympy.core.singleton import S
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.physics.units.definitions.dimension_definitions import (
    acceleration, action, current, impedance, length, mass, time, velocity,
    amount_of_substance, temperature, information, frequency, force, pressure,
    energy, power, charge, voltage, capacitance, conductance, magnetic_flux,
    magnetic_density, inductance, luminous_intensity
)
from sympy.physics.units.definitions import (
    kilogram, newton, second, meter, gram, cd, K, joule, watt, pascal, hertz,
    coulomb, volt, ohm, siemens, farad, henry, tesla, weber, dioptre, lux,
    katal, gray, becquerel, inch, liter, julian_year, gravitational_constant,
    speed_of_light, elementary_charge, planck, hbar, electronvolt,
    avogadro_number, avogadro_constant, boltzmann_constant, electron_rest_mass,
    stefan_boltzmann_constant, Da, atomic_mass_constant, molar_gas_constant,
    faraday_constant, josephson_constant, von_klitzing_constant,
    acceleration_due_to_gravity, magnetic_constant, vacuum_permittivity,
    vacuum_impedance, coulomb_constant, atmosphere, bar, pound, psi, mmHg,
    milli_mass_unit, quart, lightyear, astronomical_unit, planck_mass,
    planck_time, planck_temperature, planck_length, planck_charge, planck_area,
    planck_volume, planck_momentum, planck_energy, planck_force, planck_power,
    planck_density, planck_energy_density, planck_intensity,
    planck_angular_frequency, planck_pressure, planck_current, planck_voltage,
    planck_impedance, planck_acceleration, bit, byte, kibibyte, mebibyte,
    gibibyte, tebibyte, pebibyte, exbibyte, curie, rutherford, radian, degree,
    steradian, angular_mil, atomic_mass_unit, gee, kPa, ampere, u0, c, kelvin,
    mol, mole, candela, m, kg, s, electric_constant, G, boltzmann
)
from sympy.physics.units.prefixes import PREFIXES, prefix_unit
from sympy.physics.units.systems.mksa import MKSA, dimsys_MKSA

derived_dims = (frequency, force, pressure, energy, power, charge, voltage,
                capacitance, conductance, magnetic_flux,
                magnetic_density, inductance, luminous_intensity)
base_dims = (amount_of_substance, luminous_intensity, temperature)

units = [mol, cd, K, lux, hertz, newton, pascal, joule, watt, coulomb, volt,
        farad, ohm, siemens, weber, tesla, henry, candela, lux, becquerel,
        gray, katal]

all_units: list[Quantity] = []
for u in units:
    all_units.extend(prefix_unit(u, PREFIXES))

all_units.extend(units)
all_units.extend([mol, cd, K, lux])


dimsys_SI = dimsys_MKSA.extend(
    [
        # Dimensional dependencies for other base dimensions:
        temperature,
        amount_of_substance,
        luminous_intensity,
    ])

dimsys_default = dimsys_SI.extend(
    [information],
)

SI = MKSA.extend(base=(mol, cd, K), units=all_units, name='SI', dimension_system=dimsys_SI, derived_units={
    power: watt,
    magnetic_flux: weber,
    time: second,
    impedance: ohm,
    pressure: pascal,
    current: ampere,
    voltage: volt,
    length: meter,
    frequency: hertz,
    inductance: henry,
    temperature: kelvin,
    amount_of_substance: mole,
    luminous_intensity: candela,
    conductance: siemens,
    mass: kilogram,
    magnetic_density: tesla,
    charge: coulomb,
    force: newton,
    capacitance: farad,
    energy: joule,
    velocity: meter/second,
})

One = S.One

SI.set_quantity_dimension(radian, One)

SI.set_quantity_scale_factor(ampere, One)

SI.set_quantity_scale_factor(kelvin, One)

SI.set_quantity_scale_factor(mole, One)

SI.set_quantity_scale_factor(candela, One)

# MKSA extension to MKS: derived units

SI.set_quantity_scale_factor(coulomb, One)

SI.set_quantity_scale_factor(volt, joule/coulomb)

SI.set_quantity_scale_factor(ohm, volt/ampere)

SI.set_quantity_scale_factor(siemens, ampere/volt)

SI.set_quantity_scale_factor(farad, coulomb/volt)

SI.set_quantity_scale_factor(henry, volt*second/ampere)

SI.set_quantity_scale_factor(tesla, volt*second/meter**2)

SI.set_quantity_scale_factor(weber, joule/ampere)


SI.set_quantity_dimension(lux, luminous_intensity / length ** 2)
SI.set_quantity_scale_factor(lux, steradian*candela/meter**2)

# katal is the SI unit of catalytic activity

SI.set_quantity_dimension(katal, amount_of_substance / time)
SI.set_quantity_scale_factor(katal, mol/second)

# gray is the SI unit of absorbed dose

SI.set_quantity_dimension(gray, energy / mass)
SI.set_quantity_scale_factor(gray, meter**2/second**2)

# becquerel is the SI unit of radioactivity

SI.set_quantity_dimension(becquerel, 1 / time)
SI.set_quantity_scale_factor(becquerel, 1/second)

#### CONSTANTS ####

# elementary charge
# REF: NIST SP 959 (June 2019)

SI.set_quantity_dimension(elementary_charge, charge)
SI.set_quantity_scale_factor(elementary_charge, 1.602176634e-19*coulomb)

# Electronvolt
# REF: NIST SP 959 (June 2019)

SI.set_quantity_dimension(electronvolt, energy)
SI.set_quantity_scale_factor(electronvolt, 1.602176634e-19*joule)

# Avogadro number
# REF: NIST SP 959 (June 2019)

SI.set_quantity_dimension(avogadro_number, One)
SI.set_quantity_scale_factor(avogadro_number, 6.02214076e23)

# Avogadro constant

SI.set_quantity_dimension(avogadro_constant, amount_of_substance ** -1)
SI.set_quantity_scale_factor(avogadro_constant, avogadro_number / mol)

# Boltzmann constant
# REF: NIST SP 959 (June 2019)

SI.set_quantity_dimension(boltzmann_constant, energy / temperature)
SI.set_quantity_scale_factor(boltzmann_constant, 1.380649e-23*joule/kelvin)

# Stefan-Boltzmann constant
# REF: NIST SP 959 (June 2019)

SI.set_quantity_dimension(stefan_boltzmann_constant, energy * time ** -1 * length ** -2 * temperature ** -4)
SI.set_quantity_scale_factor(stefan_boltzmann_constant, pi**2 * boltzmann_constant**4 / (60 * hbar**3 * speed_of_light ** 2))

# Atomic mass
# REF: NIST SP 959 (June 2019)

SI.set_quantity_dimension(atomic_mass_constant, mass)
SI.set_quantity_scale_factor(atomic_mass_constant, 1.66053906660e-24*gram)

# Molar gas constant
# REF: NIST SP 959 (June 2019)

SI.set_quantity_dimension(molar_gas_constant, energy / (temperature * amount_of_substance))
SI.set_quantity_scale_factor(molar_gas_constant, boltzmann_constant * avogadro_constant)

# Faraday constant

SI.set_quantity_dimension(faraday_constant, charge / amount_of_substance)
SI.set_quantity_scale_factor(faraday_constant, elementary_charge * avogadro_constant)

# Josephson constant

SI.set_quantity_dimension(josephson_constant, frequency / voltage)
SI.set_quantity_scale_factor(josephson_constant, 0.5 * planck / elementary_charge)

# Von Klitzing constant

SI.set_quantity_dimension(von_klitzing_constant, voltage / current)
SI.set_quantity_scale_factor(von_klitzing_constant, hbar / elementary_charge ** 2)

# Acceleration due to gravity (on the Earth surface)

SI.set_quantity_dimension(acceleration_due_to_gravity, acceleration)
SI.set_quantity_scale_factor(acceleration_due_to_gravity, 9.80665*meter/second**2)

# magnetic constant:

SI.set_quantity_dimension(magnetic_constant, force / current ** 2)
SI.set_quantity_scale_factor(magnetic_constant, 4*pi/10**7 * newton/ampere**2)

# electric constant:

SI.set_quantity_dimension(vacuum_permittivity, capacitance / length)
SI.set_quantity_scale_factor(vacuum_permittivity, 1/(u0 * c**2))

# vacuum impedance:

SI.set_quantity_dimension(vacuum_impedance, impedance)
SI.set_quantity_scale_factor(vacuum_impedance, u0 * c)

# Electron rest mass
SI.set_quantity_dimension(electron_rest_mass, mass)
SI.set_quantity_scale_factor(electron_rest_mass, 9.1093837015e-31*kilogram)

# Coulomb's constant:
SI.set_quantity_dimension(coulomb_constant, force * length ** 2 / charge ** 2)
SI.set_quantity_scale_factor(coulomb_constant, 1/(4*pi*vacuum_permittivity))

SI.set_quantity_dimension(psi, pressure)
SI.set_quantity_scale_factor(psi, pound * gee / inch ** 2)

SI.set_quantity_dimension(mmHg, pressure)
SI.set_quantity_scale_factor(mmHg, dHg0 * acceleration_due_to_gravity * kilogram / meter**2)

SI.set_quantity_dimension(milli_mass_unit, mass)
SI.set_quantity_scale_factor(milli_mass_unit, atomic_mass_unit/1000)

SI.set_quantity_dimension(quart, length ** 3)
SI.set_quantity_scale_factor(quart, Rational(231, 4) * inch**3)

# Other convenient units and magnitudes

SI.set_quantity_dimension(lightyear, length)
SI.set_quantity_scale_factor(lightyear, speed_of_light*julian_year)

SI.set_quantity_dimension(astronomical_unit, length)
SI.set_quantity_scale_factor(astronomical_unit, 149597870691*meter)

# Fundamental Planck units:

SI.set_quantity_dimension(planck_mass, mass)
SI.set_quantity_scale_factor(planck_mass, sqrt(hbar*speed_of_light/G))

SI.set_quantity_dimension(planck_time, time)
SI.set_quantity_scale_factor(planck_time, sqrt(hbar*G/speed_of_light**5))

SI.set_quantity_dimension(planck_temperature, temperature)
SI.set_quantity_scale_factor(planck_temperature, sqrt(hbar*speed_of_light**5/G/boltzmann**2))

SI.set_quantity_dimension(planck_length, length)
SI.set_quantity_scale_factor(planck_length, sqrt(hbar*G/speed_of_light**3))

SI.set_quantity_dimension(planck_charge, charge)
SI.set_quantity_scale_factor(planck_charge, sqrt(4*pi*electric_constant*hbar*speed_of_light))

# Derived Planck units:

SI.set_quantity_dimension(planck_area, length ** 2)
SI.set_quantity_scale_factor(planck_area, planck_length**2)

SI.set_quantity_dimension(planck_volume, length ** 3)
SI.set_quantity_scale_factor(planck_volume, planck_length**3)

SI.set_quantity_dimension(planck_momentum, mass * velocity)
SI.set_quantity_scale_factor(planck_momentum, planck_mass * speed_of_light)

SI.set_quantity_dimension(planck_energy, energy)
SI.set_quantity_scale_factor(planck_energy, planck_mass * speed_of_light**2)

SI.set_quantity_dimension(planck_force, force)
SI.set_quantity_scale_factor(planck_force, planck_energy / planck_length)

SI.set_quantity_dimension(planck_power, power)
SI.set_quantity_scale_factor(planck_power, planck_energy / planck_time)

SI.set_quantity_dimension(planck_density, mass / length ** 3)
SI.set_quantity_scale_factor(planck_density, planck_mass / planck_length**3)

SI.set_quantity_dimension(planck_energy_density, energy / length ** 3)
SI.set_quantity_scale_factor(planck_energy_density, planck_energy / planck_length**3)

SI.set_quantity_dimension(planck_intensity, mass * time ** (-3))
SI.set_quantity_scale_factor(planck_intensity, planck_energy_density * speed_of_light)

SI.set_quantity_dimension(planck_angular_frequency, 1 / time)
SI.set_quantity_scale_factor(planck_angular_frequency, 1 / planck_time)

SI.set_quantity_dimension(planck_pressure, pressure)
SI.set_quantity_scale_factor(planck_pressure, planck_force / planck_length**2)

SI.set_quantity_dimension(planck_current, current)
SI.set_quantity_scale_factor(planck_current, planck_charge / planck_time)

SI.set_quantity_dimension(planck_voltage, voltage)
SI.set_quantity_scale_factor(planck_voltage, planck_energy / planck_charge)

SI.set_quantity_dimension(planck_impedance, impedance)
SI.set_quantity_scale_factor(planck_impedance, planck_voltage / planck_current)

SI.set_quantity_dimension(planck_acceleration, acceleration)
SI.set_quantity_scale_factor(planck_acceleration, speed_of_light / planck_time)

# Older units for radioactivity

SI.set_quantity_dimension(curie, 1 / time)
SI.set_quantity_scale_factor(curie, 37000000000*becquerel)

SI.set_quantity_dimension(rutherford, 1 / time)
SI.set_quantity_scale_factor(rutherford, 1000000*becquerel)


# check that scale factors are the right SI dimensions:
for _scale_factor, _dimension in zip(
    SI._quantity_scale_factors.values(),
    SI._quantity_dimension_map.values()
):
    dimex = SI.get_dimensional_expr(_scale_factor)
    if dimex != 1:
        # XXX: equivalent_dims is an instance method taking two arguments in
        # addition to self so this can not work:
        if not DimensionSystem.equivalent_dims(_dimension, Dimension(dimex)):  # type: ignore
            raise ValueError("quantity value and dimension mismatch")
del _scale_factor, _dimension

__all__ = [
    'mmHg', 'atmosphere', 'inductance', 'newton', 'meter',
    'vacuum_permittivity', 'pascal', 'magnetic_constant', 'voltage',
    'angular_mil', 'luminous_intensity', 'all_units',
    'julian_year', 'weber', 'exbibyte', 'liter',
    'molar_gas_constant', 'faraday_constant', 'avogadro_constant',
    'lightyear', 'planck_density', 'gee', 'mol', 'bit', 'gray',
    'planck_momentum', 'bar', 'magnetic_density', 'prefix_unit', 'PREFIXES',
    'planck_time', 'dimex', 'gram', 'candela', 'force', 'planck_intensity',
    'energy', 'becquerel', 'planck_acceleration', 'speed_of_light',
    'conductance', 'frequency', 'coulomb_constant', 'degree', 'lux', 'planck',
    'current', 'planck_current', 'tebibyte', 'planck_power', 'MKSA', 'power',
    'K', 'planck_volume', 'quart', 'pressure', 'amount_of_substance',
    'joule', 'boltzmann_constant', 'Dimension', 'c', 'planck_force', 'length',
    'watt', 'action', 'hbar', 'gibibyte', 'DimensionSystem', 'cd', 'volt',
    'planck_charge', 'dioptre', 'vacuum_impedance', 'dimsys_default', 'farad',
    'charge', 'gravitational_constant', 'temperature', 'u0', 'hertz',
    'capacitance', 'tesla', 'steradian', 'planck_mass', 'josephson_constant',
    'planck_area', 'stefan_boltzmann_constant', 'base_dims',
    'astronomical_unit', 'radian', 'planck_voltage', 'impedance',
    'planck_energy', 'Da', 'atomic_mass_constant', 'rutherford', 'second', 'inch',
    'elementary_charge', 'SI', 'electronvolt', 'dimsys_SI', 'henry',
    'planck_angular_frequency', 'ohm', 'pound', 'planck_pressure', 'G', 'psi',
    'dHg0', 'von_klitzing_constant', 'planck_length', 'avogadro_number',
    'mole', 'acceleration', 'information', 'planck_energy_density',
    'mebibyte', 's', 'acceleration_due_to_gravity', 'electron_rest_mass',
    'planck_temperature', 'units', 'mass', 'dimsys_MKSA', 'kelvin', 'kPa',
    'boltzmann', 'milli_mass_unit', 'planck_impedance', 'electric_constant',
    'derived_dims', 'kg', 'coulomb', 'siemens', 'byte', 'magnetic_flux',
    'atomic_mass_unit', 'm', 'kibibyte', 'kilogram', 'One', 'curie', 'u',
    'time', 'pebibyte', 'velocity', 'ampere', 'katal',
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\humanfriendly\prompts.py ===
# vim: fileencoding=utf-8

# Human friendly input/output in Python.
#
# Author: Peter Odding <peter@peterodding.com>
# Last Change: February 9, 2020
# URL: https://humanfriendly.readthedocs.io

"""
Interactive terminal prompts.

The :mod:`~humanfriendly.prompts` module enables interaction with the user
(operator) by asking for confirmation (:func:`prompt_for_confirmation()`) and
asking to choose from a list of options (:func:`prompt_for_choice()`). It works
by rendering interactive prompts on the terminal.
"""

# Standard library modules.
import logging
import sys

# Modules included in our package.
from humanfriendly.compat import interactive_prompt
from humanfriendly.terminal import (
    HIGHLIGHT_COLOR,
    ansi_strip,
    ansi_wrap,
    connected_to_terminal,
    terminal_supports_colors,
    warning,
)
from humanfriendly.text import format, concatenate

# Public identifiers that require documentation.
__all__ = (
    'MAX_ATTEMPTS',
    'TooManyInvalidReplies',
    'logger',
    'prepare_friendly_prompts',
    'prepare_prompt_text',
    'prompt_for_choice',
    'prompt_for_confirmation',
    'prompt_for_input',
    'retry_limit',
)

MAX_ATTEMPTS = 10
"""The number of times an interactive prompt is shown on invalid input (an integer)."""

# Initialize a logger for this module.
logger = logging.getLogger(__name__)


def prompt_for_confirmation(question, default=None, padding=True):
    """
    Prompt the user for confirmation.

    :param question: The text that explains what the user is confirming (a string).
    :param default: The default value (a boolean) or :data:`None`.
    :param padding: Refer to the documentation of :func:`prompt_for_input()`.
    :returns: - If the user enters 'yes' or 'y' then :data:`True` is returned.
              - If the user enters 'no' or 'n' then :data:`False`  is returned.
              - If the user doesn't enter any text or standard input is not
                connected to a terminal (which makes it impossible to prompt
                the user) the value of the keyword argument ``default`` is
                returned (if that value is not :data:`None`).
    :raises: - Any exceptions raised by :func:`retry_limit()`.
             - Any exceptions raised by :func:`prompt_for_input()`.

    When `default` is :data:`False` and the user doesn't enter any text an
    error message is printed and the prompt is repeated:

    >>> prompt_for_confirmation("Are you sure?")
     <BLANKLINE>
     Are you sure? [y/n]
     <BLANKLINE>
     Error: Please enter 'yes' or 'no' (there's no default choice).
     <BLANKLINE>
     Are you sure? [y/n]

    The same thing happens when the user enters text that isn't recognized:

    >>> prompt_for_confirmation("Are you sure?")
     <BLANKLINE>
     Are you sure? [y/n] about what?
     <BLANKLINE>
     Error: Please enter 'yes' or 'no' (the text 'about what?' is not recognized).
     <BLANKLINE>
     Are you sure? [y/n]
    """
    # Generate the text for the prompt.
    prompt_text = prepare_prompt_text(question, bold=True)
    # Append the valid replies (and default reply) to the prompt text.
    hint = "[Y/n]" if default else "[y/N]" if default is not None else "[y/n]"
    prompt_text += " %s " % prepare_prompt_text(hint, color=HIGHLIGHT_COLOR)
    # Loop until a valid response is given.
    logger.debug("Requesting interactive confirmation from terminal: %r", ansi_strip(prompt_text).rstrip())
    for attempt in retry_limit():
        reply = prompt_for_input(prompt_text, '', padding=padding, strip=True)
        if reply.lower() in ('y', 'yes'):
            logger.debug("Confirmation granted by reply (%r).", reply)
            return True
        elif reply.lower() in ('n', 'no'):
            logger.debug("Confirmation denied by reply (%r).", reply)
            return False
        elif (not reply) and default is not None:
            logger.debug("Default choice selected by empty reply (%r).",
                         "granted" if default else "denied")
            return default
        else:
            details = ("the text '%s' is not recognized" % reply
                       if reply else "there's no default choice")
            logger.debug("Got %s reply (%s), retrying (%i/%i) ..",
                         "invalid" if reply else "empty", details,
                         attempt, MAX_ATTEMPTS)
            warning("{indent}Error: Please enter 'yes' or 'no' ({details}).",
                    indent=' ' if padding else '', details=details)


def prompt_for_choice(choices, default=None, padding=True):
    """
    Prompt the user to select a choice from a group of options.

    :param choices: A sequence of strings with available options.
    :param default: The default choice if the user simply presses Enter
                    (expected to be a string, defaults to :data:`None`).
    :param padding: Refer to the documentation of
                    :func:`~humanfriendly.prompts.prompt_for_input()`.
    :returns: The string corresponding to the user's choice.
    :raises: - :exc:`~exceptions.ValueError` if `choices` is an empty sequence.
             - Any exceptions raised by
               :func:`~humanfriendly.prompts.retry_limit()`.
             - Any exceptions raised by
               :func:`~humanfriendly.prompts.prompt_for_input()`.

    When no options are given an exception is raised:

    >>> prompt_for_choice([])
    Traceback (most recent call last):
      File "humanfriendly/prompts.py", line 148, in prompt_for_choice
        raise ValueError("Can't prompt for choice without any options!")
    ValueError: Can't prompt for choice without any options!

    If a single option is given the user isn't prompted:

    >>> prompt_for_choice(['only one choice'])
    'only one choice'

    Here's what the actual prompt looks like by default:

    >>> prompt_for_choice(['first option', 'second option'])
    <BLANKLINE>
      1. first option
      2. second option
    <BLANKLINE>
     Enter your choice as a number or unique substring (Control-C aborts): second
    <BLANKLINE>
    'second option'

    If you don't like the whitespace (empty lines and indentation):

    >>> prompt_for_choice(['first option', 'second option'], padding=False)
     1. first option
     2. second option
    Enter your choice as a number or unique substring (Control-C aborts): first
    'first option'
    """
    indent = ' ' if padding else ''
    # Make sure we can use 'choices' more than once (i.e. not a generator).
    choices = list(choices)
    if len(choices) == 1:
        # If there's only one option there's no point in prompting the user.
        logger.debug("Skipping interactive prompt because there's only option (%r).", choices[0])
        return choices[0]
    elif not choices:
        # We can't render a choice prompt without any options.
        raise ValueError("Can't prompt for choice without any options!")
    # Generate the prompt text.
    prompt_text = ('\n\n' if padding else '\n').join([
        # Present the available choices in a user friendly way.
        "\n".join([
            (u" %i. %s" % (i, choice)) + (" (default choice)" if choice == default else "")
            for i, choice in enumerate(choices, start=1)
        ]),
        # Instructions for the user.
        "Enter your choice as a number or unique substring (Control-C aborts): ",
    ])
    prompt_text = prepare_prompt_text(prompt_text, bold=True)
    # Loop until a valid choice is made.
    logger.debug("Requesting interactive choice on terminal (options are %s) ..",
                 concatenate(map(repr, choices)))
    for attempt in retry_limit():
        reply = prompt_for_input(prompt_text, '', padding=padding, strip=True)
        if not reply and default is not None:
            logger.debug("Default choice selected by empty reply (%r).", default)
            return default
        elif reply.isdigit():
            index = int(reply) - 1
            if 0 <= index < len(choices):
                logger.debug("Option (%r) selected by numeric reply (%s).", choices[index], reply)
                return choices[index]
        # Check for substring matches.
        matches = []
        for choice in choices:
            lower_reply = reply.lower()
            lower_choice = choice.lower()
            if lower_reply == lower_choice:
                # If we have an 'exact' match we return it immediately.
                logger.debug("Option (%r) selected by reply (exact match).", choice)
                return choice
            elif lower_reply in lower_choice and len(lower_reply) > 0:
                # Otherwise we gather substring matches.
                matches.append(choice)
        if len(matches) == 1:
            # If a single choice was matched we return it.
            logger.debug("Option (%r) selected by reply (substring match on %r).", matches[0], reply)
            return matches[0]
        else:
            # Give the user a hint about what went wrong.
            if matches:
                details = format("text '%s' matches more than one choice: %s", reply, concatenate(matches))
            elif reply.isdigit():
                details = format("number %i is not a valid choice", int(reply))
            elif reply and not reply.isspace():
                details = format("text '%s' doesn't match any choices", reply)
            else:
                details = "there's no default choice"
            logger.debug("Got %s reply (%s), retrying (%i/%i) ..",
                         "invalid" if reply else "empty", details,
                         attempt, MAX_ATTEMPTS)
            warning("%sError: Invalid input (%s).", indent, details)


def prompt_for_input(question, default=None, padding=True, strip=True):
    """
    Prompt the user for input (free form text).

    :param question: An explanation of what is expected from the user (a string).
    :param default: The return value if the user doesn't enter any text or
                    standard input is not connected to a terminal (which
                    makes it impossible to prompt the user).
    :param padding: Render empty lines before and after the prompt to make it
                    stand out from the surrounding text? (a boolean, defaults
                    to :data:`True`)
    :param strip: Strip leading/trailing whitespace from the user's reply?
    :returns: The text entered by the user (a string) or the value of the
              `default` argument.
    :raises: - :exc:`~exceptions.KeyboardInterrupt` when the program is
               interrupted_ while the prompt is active, for example
               because the user presses Control-C_.
             - :exc:`~exceptions.EOFError` when reading from `standard input`_
               fails, for example because the user presses Control-D_ or
               because the standard input stream is redirected (only if
               `default` is :data:`None`).

    .. _Control-C: https://en.wikipedia.org/wiki/Control-C#In_command-line_environments
    .. _Control-D: https://en.wikipedia.org/wiki/End-of-transmission_character#Meaning_in_Unix
    .. _interrupted: https://en.wikipedia.org/wiki/Unix_signal#SIGINT
    .. _standard input: https://en.wikipedia.org/wiki/Standard_streams#Standard_input_.28stdin.29
    """
    prepare_friendly_prompts()
    reply = None
    try:
        # Prefix an empty line to the text and indent by one space?
        if padding:
            question = '\n' + question
            question = question.replace('\n', '\n ')
        # Render the prompt and wait for the user's reply.
        try:
            reply = interactive_prompt(question)
        finally:
            if reply is None:
                # If the user terminated the prompt using Control-C or
                # Control-D instead of pressing Enter no newline will be
                # rendered after the prompt's text. The result looks kind of
                # weird:
                #
                #   $ python -c 'print(raw_input("Are you sure? "))'
                #   Are you sure? ^CTraceback (most recent call last):
                #     File "<string>", line 1, in <module>
                #   KeyboardInterrupt
                #
                # We can avoid this by emitting a newline ourselves if an
                # exception was raised (signaled by `reply' being None).
                sys.stderr.write('\n')
            if padding:
                # If the caller requested (didn't opt out of) `padding' then we'll
                # emit a newline regardless of whether an exception is being
                # handled. This helps to make interactive prompts `stand out' from
                # a surrounding `wall of text' on the terminal.
                sys.stderr.write('\n')
    except BaseException as e:
        if isinstance(e, EOFError) and default is not None:
            # If standard input isn't connected to an interactive terminal
            # but the caller provided a default we'll return that.
            logger.debug("Got EOF from terminal, returning default value (%r) ..", default)
            return default
        else:
            # Otherwise we log that the prompt was interrupted but propagate
            # the exception to the caller.
            logger.warning("Interactive prompt was interrupted by exception!", exc_info=True)
            raise
    if default is not None and not reply:
        # If the reply is empty and `default' is None we don't want to return
        # None because it's nicer for callers to be able to assume that the
        # return value is always a string.
        return default
    else:
        return reply.strip()


def prepare_prompt_text(prompt_text, **options):
    """
    Wrap a text to be rendered as an interactive prompt in ANSI escape sequences.

    :param prompt_text: The text to render on the prompt (a string).
    :param options: Any keyword arguments are passed on to :func:`.ansi_wrap()`.
    :returns: The resulting prompt text (a string).

    ANSI escape sequences are only used when the standard output stream is
    connected to a terminal. When the standard input stream is connected to a
    terminal any escape sequences are wrapped in "readline hints".
    """
    return (ansi_wrap(prompt_text, readline_hints=connected_to_terminal(sys.stdin), **options)
            if terminal_supports_colors(sys.stdout)
            else prompt_text)


def prepare_friendly_prompts():
    u"""
    Make interactive prompts more user friendly.

    The prompts presented by :func:`python2:raw_input()` (in Python 2) and
    :func:`python3:input()` (in Python 3) are not very user friendly by
    default, for example the cursor keys (:kbd:`←`, :kbd:`↑`, :kbd:`→` and
    :kbd:`↓`) and the :kbd:`Home` and :kbd:`End` keys enter characters instead
    of performing the action you would expect them to. By simply importing the
    :mod:`readline` module these prompts become much friendlier (as mentioned
    in the Python standard library documentation).

    This function is called by the other functions in this module to enable
    user friendly prompts.
    """
    try:
        import readline  # NOQA
    except ImportError:
        # might not be available on Windows if pyreadline isn't installed
        pass


def retry_limit(limit=MAX_ATTEMPTS):
    """
    Allow the user to provide valid input up to `limit` times.

    :param limit: The maximum number of attempts (a number,
                  defaults to :data:`MAX_ATTEMPTS`).
    :returns: A generator of numbers starting from one.
    :raises: :exc:`TooManyInvalidReplies` when an interactive prompt
             receives repeated invalid input (:data:`MAX_ATTEMPTS`).

    This function returns a generator for interactive prompts that want to
    repeat on invalid input without getting stuck in infinite loops.
    """
    for i in range(limit):
        yield i + 1
    msg = "Received too many invalid replies on interactive prompt, giving up! (tried %i times)"
    formatted_msg = msg % limit
    # Make sure the event is logged.
    logger.warning(formatted_msg)
    # Force the caller to decide what to do now.
    raise TooManyInvalidReplies(formatted_msg)


class TooManyInvalidReplies(Exception):

    """Raised by interactive prompts when they've received too many invalid inputs."""

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\live.py ===
import sys
from threading import Event, RLock, Thread
from types import TracebackType
from typing import IO, Any, Callable, List, Optional, TextIO, Type, cast

from . import get_console
from .console import Console, ConsoleRenderable, RenderableType, RenderHook
from .control import Control
from .file_proxy import FileProxy
from .jupyter import JupyterMixin
from .live_render import LiveRender, VerticalOverflowMethod
from .screen import Screen
from .text import Text


class _RefreshThread(Thread):
    """A thread that calls refresh() at regular intervals."""

    def __init__(self, live: "Live", refresh_per_second: float) -> None:
        self.live = live
        self.refresh_per_second = refresh_per_second
        self.done = Event()
        super().__init__(daemon=True)

    def stop(self) -> None:
        self.done.set()

    def run(self) -> None:
        while not self.done.wait(1 / self.refresh_per_second):
            with self.live._lock:
                if not self.done.is_set():
                    self.live.refresh()


class Live(JupyterMixin, RenderHook):
    """Renders an auto-updating live display of any given renderable.

    Args:
        renderable (RenderableType, optional): The renderable to live display. Defaults to displaying nothing.
        console (Console, optional): Optional Console instance. Defaults to an internal Console instance writing to stdout.
        screen (bool, optional): Enable alternate screen mode. Defaults to False.
        auto_refresh (bool, optional): Enable auto refresh. If disabled, you will need to call `refresh()` or `update()` with refresh flag. Defaults to True
        refresh_per_second (float, optional): Number of times per second to refresh the live display. Defaults to 4.
        transient (bool, optional): Clear the renderable on exit (has no effect when screen=True). Defaults to False.
        redirect_stdout (bool, optional): Enable redirection of stdout, so ``print`` may be used. Defaults to True.
        redirect_stderr (bool, optional): Enable redirection of stderr. Defaults to True.
        vertical_overflow (VerticalOverflowMethod, optional): How to handle renderable when it is too tall for the console. Defaults to "ellipsis".
        get_renderable (Callable[[], RenderableType], optional): Optional callable to get renderable. Defaults to None.
    """

    def __init__(
        self,
        renderable: Optional[RenderableType] = None,
        *,
        console: Optional[Console] = None,
        screen: bool = False,
        auto_refresh: bool = True,
        refresh_per_second: float = 4,
        transient: bool = False,
        redirect_stdout: bool = True,
        redirect_stderr: bool = True,
        vertical_overflow: VerticalOverflowMethod = "ellipsis",
        get_renderable: Optional[Callable[[], RenderableType]] = None,
    ) -> None:
        assert refresh_per_second > 0, "refresh_per_second must be > 0"
        self._renderable = renderable
        self.console = console if console is not None else get_console()
        self._screen = screen
        self._alt_screen = False

        self._redirect_stdout = redirect_stdout
        self._redirect_stderr = redirect_stderr
        self._restore_stdout: Optional[IO[str]] = None
        self._restore_stderr: Optional[IO[str]] = None

        self._lock = RLock()
        self.ipy_widget: Optional[Any] = None
        self.auto_refresh = auto_refresh
        self._started: bool = False
        self.transient = True if screen else transient

        self._refresh_thread: Optional[_RefreshThread] = None
        self.refresh_per_second = refresh_per_second

        self.vertical_overflow = vertical_overflow
        self._get_renderable = get_renderable
        self._live_render = LiveRender(
            self.get_renderable(), vertical_overflow=vertical_overflow
        )

    @property
    def is_started(self) -> bool:
        """Check if live display has been started."""
        return self._started

    def get_renderable(self) -> RenderableType:
        renderable = (
            self._get_renderable()
            if self._get_renderable is not None
            else self._renderable
        )
        return renderable or ""

    def start(self, refresh: bool = False) -> None:
        """Start live rendering display.

        Args:
            refresh (bool, optional): Also refresh. Defaults to False.
        """
        with self._lock:
            if self._started:
                return
            self.console.set_live(self)
            self._started = True
            if self._screen:
                self._alt_screen = self.console.set_alt_screen(True)
            self.console.show_cursor(False)
            self._enable_redirect_io()
            self.console.push_render_hook(self)
            if refresh:
                try:
                    self.refresh()
                except Exception:
                    # If refresh fails, we want to stop the redirection of sys.stderr,
                    # so the error stacktrace is properly displayed in the terminal.
                    # (or, if the code that calls Rich captures the exception and wants to display something,
                    # let this be displayed in the terminal).
                    self.stop()
                    raise
            if self.auto_refresh:
                self._refresh_thread = _RefreshThread(self, self.refresh_per_second)
                self._refresh_thread.start()

    def stop(self) -> None:
        """Stop live rendering display."""
        with self._lock:
            if not self._started:
                return
            self.console.clear_live()
            self._started = False

            if self.auto_refresh and self._refresh_thread is not None:
                self._refresh_thread.stop()
                self._refresh_thread = None
            # allow it to fully render on the last even if overflow
            self.vertical_overflow = "visible"
            with self.console:
                try:
                    if not self._alt_screen and not self.console.is_jupyter:
                        self.refresh()
                finally:
                    self._disable_redirect_io()
                    self.console.pop_render_hook()
                    if not self._alt_screen and self.console.is_terminal:
                        self.console.line()
                    self.console.show_cursor(True)
                    if self._alt_screen:
                        self.console.set_alt_screen(False)

                    if self.transient and not self._alt_screen:
                        self.console.control(self._live_render.restore_cursor())
                    if self.ipy_widget is not None and self.transient:
                        self.ipy_widget.close()  # pragma: no cover

    def __enter__(self) -> "Live":
        self.start(refresh=self._renderable is not None)
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.stop()

    def _enable_redirect_io(self) -> None:
        """Enable redirecting of stdout / stderr."""
        if self.console.is_terminal or self.console.is_jupyter:
            if self._redirect_stdout and not isinstance(sys.stdout, FileProxy):
                self._restore_stdout = sys.stdout
                sys.stdout = cast("TextIO", FileProxy(self.console, sys.stdout))
            if self._redirect_stderr and not isinstance(sys.stderr, FileProxy):
                self._restore_stderr = sys.stderr
                sys.stderr = cast("TextIO", FileProxy(self.console, sys.stderr))

    def _disable_redirect_io(self) -> None:
        """Disable redirecting of stdout / stderr."""
        if self._restore_stdout:
            sys.stdout = cast("TextIO", self._restore_stdout)
            self._restore_stdout = None
        if self._restore_stderr:
            sys.stderr = cast("TextIO", self._restore_stderr)
            self._restore_stderr = None

    @property
    def renderable(self) -> RenderableType:
        """Get the renderable that is being displayed

        Returns:
            RenderableType: Displayed renderable.
        """
        renderable = self.get_renderable()
        return Screen(renderable) if self._alt_screen else renderable

    def update(self, renderable: RenderableType, *, refresh: bool = False) -> None:
        """Update the renderable that is being displayed

        Args:
            renderable (RenderableType): New renderable to use.
            refresh (bool, optional): Refresh the display. Defaults to False.
        """
        if isinstance(renderable, str):
            renderable = self.console.render_str(renderable)
        with self._lock:
            self._renderable = renderable
            if refresh:
                self.refresh()

    def refresh(self) -> None:
        """Update the display of the Live Render."""
        with self._lock:
            self._live_render.set_renderable(self.renderable)
            if self.console.is_jupyter:  # pragma: no cover
                try:
                    from IPython.display import display
                    from ipywidgets import Output
                except ImportError:
                    import warnings

                    warnings.warn('install "ipywidgets" for Jupyter support')
                else:
                    if self.ipy_widget is None:
                        self.ipy_widget = Output()
                        display(self.ipy_widget)

                    with self.ipy_widget:
                        self.ipy_widget.clear_output(wait=True)
                        self.console.print(self._live_render.renderable)
            elif self.console.is_terminal and not self.console.is_dumb_terminal:
                with self.console:
                    self.console.print(Control())
            elif (
                not self._started and not self.transient
            ):  # if it is finished allow files or dumb-terminals to see final result
                with self.console:
                    self.console.print(Control())

    def process_renderables(
        self, renderables: List[ConsoleRenderable]
    ) -> List[ConsoleRenderable]:
        """Process renderables to restore cursor and display progress."""
        self._live_render.vertical_overflow = self.vertical_overflow
        if self.console.is_interactive:
            # lock needs acquiring as user can modify live_render renderable at any time unlike in Progress.
            with self._lock:
                reset = (
                    Control.home()
                    if self._alt_screen
                    else self._live_render.position_cursor()
                )
                renderables = [reset, *renderables, self._live_render]
        elif (
            not self._started and not self.transient
        ):  # if it is finished render the final output for files or dumb_terminals
            renderables = [*renderables, self._live_render]

        return renderables


if __name__ == "__main__":  # pragma: no cover
    import random
    import time
    from itertools import cycle
    from typing import Dict, List, Tuple

    from .align import Align
    from .console import Console
    from .live import Live as Live
    from .panel import Panel
    from .rule import Rule
    from .syntax import Syntax
    from .table import Table

    console = Console()

    syntax = Syntax(
        '''def loop_last(values: Iterable[T]) -> Iterable[Tuple[bool, T]]:
    """Iterate and generate a tuple with a flag for last value."""
    iter_values = iter(values)
    try:
        previous_value = next(iter_values)
    except StopIteration:
        return
    for value in iter_values:
        yield False, previous_value
        previous_value = value
    yield True, previous_value''',
        "python",
        line_numbers=True,
    )

    table = Table("foo", "bar", "baz")
    table.add_row("1", "2", "3")

    progress_renderables = [
        "You can make the terminal shorter and taller to see the live table hide"
        "Text may be printed while the progress bars are rendering.",
        Panel("In fact, [i]any[/i] renderable will work"),
        "Such as [magenta]tables[/]...",
        table,
        "Pretty printed structures...",
        {"type": "example", "text": "Pretty printed"},
        "Syntax...",
        syntax,
        Rule("Give it a try!"),
    ]

    examples = cycle(progress_renderables)

    exchanges = [
        "SGD",
        "MYR",
        "EUR",
        "USD",
        "AUD",
        "JPY",
        "CNH",
        "HKD",
        "CAD",
        "INR",
        "DKK",
        "GBP",
        "RUB",
        "NZD",
        "MXN",
        "IDR",
        "TWD",
        "THB",
        "VND",
    ]
    with Live(console=console) as live_table:
        exchange_rate_dict: Dict[Tuple[str, str], float] = {}

        for index in range(100):
            select_exchange = exchanges[index % len(exchanges)]

            for exchange in exchanges:
                if exchange == select_exchange:
                    continue
                time.sleep(0.4)
                if random.randint(0, 10) < 1:
                    console.log(next(examples))
                exchange_rate_dict[(select_exchange, exchange)] = 200 / (
                    (random.random() * 320) + 1
                )
                if len(exchange_rate_dict) > len(exchanges) - 1:
                    exchange_rate_dict.pop(list(exchange_rate_dict.keys())[0])
                table = Table(title="Exchange Rates")

                table.add_column("Source Currency")
                table.add_column("Destination Currency")
                table.add_column("Exchange Rate")

                for (source, dest), exchange_rate in exchange_rate_dict.items():
                    table.add_row(
                        source,
                        dest,
                        Text(
                            f"{exchange_rate:.4f}",
                            style="red" if exchange_rate < 1.0 else "green",
                        ),
                    )

                live_table.update(Align.center(table))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\vector\basisdependent.py ===
from __future__ import annotations
from typing import TYPE_CHECKING

from sympy.simplify import simplify as simp, trigsimp as tsimp  # type: ignore
from sympy.core.decorators import call_highest_priority, _sympifyit
from sympy.core.assumptions import StdFactKB
from sympy.core.function import diff as df
from sympy.integrals.integrals import Integral
from sympy.polys.polytools import factor as fctr
from sympy.core import S, Add, Mul
from sympy.core.expr import Expr

if TYPE_CHECKING:
    from sympy.vector.vector import BaseVector


class BasisDependent(Expr):
    """
    Super class containing functionality common to vectors and
    dyadics.
    Named so because the representation of these quantities in
    sympy.vector is dependent on the basis they are expressed in.
    """

    zero: BasisDependentZero

    @call_highest_priority('__radd__')
    def __add__(self, other):
        return self._add_func(self, other)

    @call_highest_priority('__add__')
    def __radd__(self, other):
        return self._add_func(other, self)

    @call_highest_priority('__rsub__')
    def __sub__(self, other):
        return self._add_func(self, -other)

    @call_highest_priority('__sub__')
    def __rsub__(self, other):
        return self._add_func(other, -self)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rmul__')
    def __mul__(self, other):
        return self._mul_func(self, other)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__mul__')
    def __rmul__(self, other):
        return self._mul_func(other, self)

    def __neg__(self):
        return self._mul_func(S.NegativeOne, self)

    @_sympifyit('other', NotImplemented)
    @call_highest_priority('__rtruediv__')
    def __truediv__(self, other):
        return self._div_helper(other)

    @call_highest_priority('__truediv__')
    def __rtruediv__(self, other):
        return TypeError("Invalid divisor for division")

    def evalf(self, n=15, subs=None, maxn=100, chop=False, strict=False, quad=None, verbose=False):
        """
        Implements the SymPy evalf routine for this quantity.

        evalf's documentation
        =====================

        """
        options = {'subs':subs, 'maxn':maxn, 'chop':chop, 'strict':strict,
                'quad':quad, 'verbose':verbose}
        vec = self.zero
        for k, v in self.components.items():
            vec += v.evalf(n, **options) * k
        return vec

    evalf.__doc__ += Expr.evalf.__doc__  # type: ignore

    n = evalf # type: ignore

    def simplify(self, **kwargs):
        """
        Implements the SymPy simplify routine for this quantity.

        simplify's documentation
        ========================

        """
        simp_components = [simp(v, **kwargs) * k for
                           k, v in self.components.items()]
        return self._add_func(*simp_components)

    simplify.__doc__ += simp.__doc__  # type: ignore

    def trigsimp(self, **opts):
        """
        Implements the SymPy trigsimp routine, for this quantity.

        trigsimp's documentation
        ========================

        """
        trig_components = [tsimp(v, **opts) * k for
                           k, v in self.components.items()]
        return self._add_func(*trig_components)

    trigsimp.__doc__ += tsimp.__doc__  # type: ignore

    def _eval_simplify(self, **kwargs):
        return self.simplify(**kwargs)

    def _eval_trigsimp(self, **opts):
        return self.trigsimp(**opts)

    def _eval_derivative(self, wrt):
        return self.diff(wrt)

    def _eval_Integral(self, *symbols, **assumptions):
        integral_components = [Integral(v, *symbols, **assumptions) * k
                               for k, v in self.components.items()]
        return self._add_func(*integral_components)

    def as_numer_denom(self):
        """
        Returns the expression as a tuple wrt the following
        transformation -

        expression -> a/b -> a, b

        """
        return self, S.One

    def factor(self, *args, **kwargs):
        """
        Implements the SymPy factor routine, on the scalar parts
        of a basis-dependent expression.

        factor's documentation
        ========================

        """
        fctr_components = [fctr(v, *args, **kwargs) * k for
                           k, v in self.components.items()]
        return self._add_func(*fctr_components)

    factor.__doc__ += fctr.__doc__  # type: ignore

    def as_coeff_Mul(self, rational=False):
        """Efficiently extract the coefficient of a product."""
        return (S.One, self)

    def as_coeff_add(self, *deps):
        """Efficiently extract the coefficient of a summation."""
        return 0, tuple(x * self.components[x] for x in self.components)

    def diff(self, *args, **kwargs):
        """
        Implements the SymPy diff routine, for vectors.

        diff's documentation
        ========================

        """
        for x in args:
            if isinstance(x, BasisDependent):
                raise TypeError("Invalid arg for differentiation")
        diff_components = [df(v, *args, **kwargs) * k for
                           k, v in self.components.items()]
        return self._add_func(*diff_components)

    diff.__doc__ += df.__doc__  # type: ignore

    def doit(self, **hints):
        """Calls .doit() on each term in the Dyadic"""
        doit_components = [self.components[x].doit(**hints) * x
                           for x in self.components]
        return self._add_func(*doit_components)


class BasisDependentAdd(BasisDependent, Add):
    """
    Denotes sum of basis dependent quantities such that they cannot
    be expressed as base or Mul instances.
    """

    def __new__(cls, *args, **options):
        components = {}

        # Check each arg and simultaneously learn the components
        for arg in args:
            if not isinstance(arg, cls._expr_type):
                if isinstance(arg, Mul):
                    arg = cls._mul_func(*(arg.args))
                elif isinstance(arg, Add):
                    arg = cls._add_func(*(arg.args))
                else:
                    raise TypeError(str(arg) +
                                    " cannot be interpreted correctly")
            # If argument is zero, ignore
            if arg == cls.zero:
                continue
            # Else, update components accordingly
            for x in arg.components:
                components[x] = components.get(x, 0) + arg.components[x]

        temp = list(components.keys())
        for x in temp:
            if components[x] == 0:
                del components[x]

        # Handle case of zero vector
        if len(components) == 0:
            return cls.zero

        # Build object
        newargs = [x * components[x] for x in components]
        obj = super().__new__(cls, *newargs, **options)
        if isinstance(obj, Mul):
            return cls._mul_func(*obj.args)
        assumptions = {'commutative': True}
        obj._assumptions = StdFactKB(assumptions)
        obj._components = components
        obj._sys = (list(components.keys()))[0]._sys

        return obj


class BasisDependentMul(BasisDependent, Mul):
    """
    Denotes product of base- basis dependent quantity with a scalar.
    """

    def __new__(cls, *args, **options):
        obj = cls._new(*args, **options)
        return obj

    def _new_rawargs(self, *args):
        # XXX: This is needed because Add.flatten() uses it but the default
        # implementation does not work for Vectors because they assign
        # attributes outside of .args.
        return type(self)(*args)

    @classmethod
    def _new(cls, *args, **options):
        from sympy.vector import Cross, Dot, Curl, Gradient
        count = 0
        measure_number = S.One
        zeroflag = False
        extra_args = []

        # Determine the component and check arguments
        # Also keep a count to ensure two vectors aren't
        # being multiplied
        for arg in args:
            if isinstance(arg, cls._zero_func):
                count += 1
                zeroflag = True
            elif arg == S.Zero:
                zeroflag = True
            elif isinstance(arg, (cls._base_func, cls._mul_func)):
                count += 1
                expr = arg._base_instance
                measure_number *= arg._measure_number
            elif isinstance(arg, cls._add_func):
                count += 1
                expr = arg
            elif isinstance(arg, (Cross, Dot, Curl, Gradient)):
                extra_args.append(arg)
            else:
                measure_number *= arg
        # Make sure incompatible types weren't multiplied
        if count > 1:
            raise ValueError("Invalid multiplication")
        elif count == 0:
            return Mul(*args, **options)
        # Handle zero vector case
        if zeroflag:
            return cls.zero

        # If one of the args was a VectorAdd, return an
        # appropriate VectorAdd instance
        if isinstance(expr, cls._add_func):
            newargs = [cls._mul_func(measure_number, x) for
                       x in expr.args]
            return cls._add_func(*newargs)

        obj = super().__new__(cls, measure_number,
                              expr._base_instance,
                              *extra_args,
                              **options)
        if isinstance(obj, Add):
            return cls._add_func(*obj.args)
        obj._base_instance = expr._base_instance
        obj._measure_number = measure_number
        assumptions = {'commutative': True}
        obj._assumptions = StdFactKB(assumptions)
        obj._components = {expr._base_instance: measure_number}
        obj._sys = expr._base_instance._sys

        return obj

    def _sympystr(self, printer):
        measure_str = printer._print(self._measure_number)
        if ('(' in measure_str or '-' in measure_str or
                '+' in measure_str):
            measure_str = '(' + measure_str + ')'
        return measure_str + '*' + printer._print(self._base_instance)


class BasisDependentZero(BasisDependent):
    """
    Class to denote a zero basis dependent instance.
    """
    components: dict['BaseVector', Expr] = {}
    _latex_form: str

    def __new__(cls):
        obj = super().__new__(cls)
        # Pre-compute a specific hash value for the zero vector
        # Use the same one always
        obj._hash = (S.Zero, cls).__hash__()
        return obj

    def __hash__(self):
        return self._hash

    @call_highest_priority('__req__')
    def __eq__(self, other):
        return isinstance(other, self._zero_func)

    __req__ = __eq__

    @call_highest_priority('__radd__')
    def __add__(self, other):
        if isinstance(other, self._expr_type):
            return other
        else:
            raise TypeError("Invalid argument types for addition")

    @call_highest_priority('__add__')
    def __radd__(self, other):
        if isinstance(other, self._expr_type):
            return other
        else:
            raise TypeError("Invalid argument types for addition")

    @call_highest_priority('__rsub__')
    def __sub__(self, other):
        if isinstance(other, self._expr_type):
            return -other
        else:
            raise TypeError("Invalid argument types for subtraction")

    @call_highest_priority('__sub__')
    def __rsub__(self, other):
        if isinstance(other, self._expr_type):
            return other
        else:
            raise TypeError("Invalid argument types for subtraction")

    def __neg__(self):
        return self

    def normalize(self):
        """
        Returns the normalized version of this vector.
        """
        return self

    def _sympystr(self, printer):
        return '0'

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\websocket\_http.py ===
"""
_http.py
websocket - WebSocket client library for Python

Copyright 2024 engn33r

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import errno
import os
import socket
from base64 import encodebytes as base64encode

from ._exceptions import (
    WebSocketAddressException,
    WebSocketException,
    WebSocketProxyException,
)
from ._logging import debug, dump, trace
from ._socket import DEFAULT_SOCKET_OPTION, recv_line, send
from ._ssl_compat import HAVE_SSL, ssl
from ._url import get_proxy_info, parse_url

__all__ = ["proxy_info", "connect", "read_headers"]

try:
    from python_socks._errors import *
    from python_socks._types import ProxyType
    from python_socks.sync import Proxy

    HAVE_PYTHON_SOCKS = True
except:
    HAVE_PYTHON_SOCKS = False

    class ProxyError(Exception):
        pass

    class ProxyTimeoutError(Exception):
        pass

    class ProxyConnectionError(Exception):
        pass


class proxy_info:
    def __init__(self, **options):
        self.proxy_host = options.get("http_proxy_host", None)
        if self.proxy_host:
            self.proxy_port = options.get("http_proxy_port", 0)
            self.auth = options.get("http_proxy_auth", None)
            self.no_proxy = options.get("http_no_proxy", None)
            self.proxy_protocol = options.get("proxy_type", "http")
            # Note: If timeout not specified, default python-socks timeout is 60 seconds
            self.proxy_timeout = options.get("http_proxy_timeout", None)
            if self.proxy_protocol not in [
                "http",
                "socks4",
                "socks4a",
                "socks5",
                "socks5h",
            ]:
                raise ProxyError(
                    "Only http, socks4, socks5 proxy protocols are supported"
                )
        else:
            self.proxy_port = 0
            self.auth = None
            self.no_proxy = None
            self.proxy_protocol = "http"


def _start_proxied_socket(url: str, options, proxy) -> tuple:
    if not HAVE_PYTHON_SOCKS:
        raise WebSocketException(
            "Python Socks is needed for SOCKS proxying but is not available"
        )

    hostname, port, resource, is_secure = parse_url(url)

    if proxy.proxy_protocol == "socks4":
        rdns = False
        proxy_type = ProxyType.SOCKS4
    # socks4a sends DNS through proxy
    elif proxy.proxy_protocol == "socks4a":
        rdns = True
        proxy_type = ProxyType.SOCKS4
    elif proxy.proxy_protocol == "socks5":
        rdns = False
        proxy_type = ProxyType.SOCKS5
    # socks5h sends DNS through proxy
    elif proxy.proxy_protocol == "socks5h":
        rdns = True
        proxy_type = ProxyType.SOCKS5

    ws_proxy = Proxy.create(
        proxy_type=proxy_type,
        host=proxy.proxy_host,
        port=int(proxy.proxy_port),
        username=proxy.auth[0] if proxy.auth else None,
        password=proxy.auth[1] if proxy.auth else None,
        rdns=rdns,
    )

    sock = ws_proxy.connect(hostname, port, timeout=proxy.proxy_timeout)

    if is_secure:
        if HAVE_SSL:
            sock = _ssl_socket(sock, options.sslopt, hostname)
        else:
            raise WebSocketException("SSL not available.")

    return sock, (hostname, port, resource)


def connect(url: str, options, proxy, socket):
    # Use _start_proxied_socket() only for socks4 or socks5 proxy
    # Use _tunnel() for http proxy
    # TODO: Use python-socks for http protocol also, to standardize flow
    if proxy.proxy_host and not socket and proxy.proxy_protocol != "http":
        return _start_proxied_socket(url, options, proxy)

    hostname, port_from_url, resource, is_secure = parse_url(url)

    if socket:
        return socket, (hostname, port_from_url, resource)

    addrinfo_list, need_tunnel, auth = _get_addrinfo_list(
        hostname, port_from_url, is_secure, proxy
    )
    if not addrinfo_list:
        raise WebSocketException(f"Host not found.: {hostname}:{port_from_url}")

    sock = None
    try:
        sock = _open_socket(addrinfo_list, options.sockopt, options.timeout)
        if need_tunnel:
            sock = _tunnel(sock, hostname, port_from_url, auth)

        if is_secure:
            if HAVE_SSL:
                sock = _ssl_socket(sock, options.sslopt, hostname)
            else:
                raise WebSocketException("SSL not available.")

        return sock, (hostname, port_from_url, resource)
    except:
        if sock:
            sock.close()
        raise


def _get_addrinfo_list(hostname, port: int, is_secure: bool, proxy) -> tuple:
    phost, pport, pauth = get_proxy_info(
        hostname,
        is_secure,
        proxy.proxy_host,
        proxy.proxy_port,
        proxy.auth,
        proxy.no_proxy,
    )
    try:
        # when running on windows 10, getaddrinfo without socktype returns a socktype 0.
        # This generates an error exception: `_on_error: exception Socket type must be stream or datagram, not 0`
        # or `OSError: [Errno 22] Invalid argument` when creating socket. Force the socket type to SOCK_STREAM.
        if not phost:
            addrinfo_list = socket.getaddrinfo(
                hostname, port, 0, socket.SOCK_STREAM, socket.SOL_TCP
            )
            return addrinfo_list, False, None
        else:
            pport = pport and pport or 80
            # when running on windows 10, the getaddrinfo used above
            # returns a socktype 0. This generates an error exception:
            # _on_error: exception Socket type must be stream or datagram, not 0
            # Force the socket type to SOCK_STREAM
            addrinfo_list = socket.getaddrinfo(
                phost, pport, 0, socket.SOCK_STREAM, socket.SOL_TCP
            )
            return addrinfo_list, True, pauth
    except socket.gaierror as e:
        raise WebSocketAddressException(e)


def _open_socket(addrinfo_list, sockopt, timeout):
    err = None
    for addrinfo in addrinfo_list:
        family, socktype, proto = addrinfo[:3]
        sock = socket.socket(family, socktype, proto)
        sock.settimeout(timeout)
        for opts in DEFAULT_SOCKET_OPTION:
            sock.setsockopt(*opts)
        for opts in sockopt:
            sock.setsockopt(*opts)

        address = addrinfo[4]
        err = None
        while not err:
            try:
                sock.connect(address)
            except socket.error as error:
                sock.close()
                error.remote_ip = str(address[0])
                try:
                    eConnRefused = (
                        errno.ECONNREFUSED,
                        errno.WSAECONNREFUSED,
                        errno.ENETUNREACH,
                    )
                except AttributeError:
                    eConnRefused = (errno.ECONNREFUSED, errno.ENETUNREACH)
                if error.errno not in eConnRefused:
                    raise error
                err = error
                continue
            else:
                break
        else:
            continue
        break
    else:
        if err:
            raise err

    return sock


def _wrap_sni_socket(sock: socket.socket, sslopt: dict, hostname, check_hostname):
    context = sslopt.get("context", None)
    if not context:
        context = ssl.SSLContext(sslopt.get("ssl_version", ssl.PROTOCOL_TLS_CLIENT))
        # Non default context need to manually enable SSLKEYLOGFILE support by setting the keylog_filename attribute.
        # For more details see also:
        # * https://docs.python.org/3.8/library/ssl.html?highlight=sslkeylogfile#context-creation
        # * https://docs.python.org/3.8/library/ssl.html?highlight=sslkeylogfile#ssl.SSLContext.keylog_filename
        context.keylog_filename = os.environ.get("SSLKEYLOGFILE", None)

        if sslopt.get("cert_reqs", ssl.CERT_NONE) != ssl.CERT_NONE:
            cafile = sslopt.get("ca_certs", None)
            capath = sslopt.get("ca_cert_path", None)
            if cafile or capath:
                context.load_verify_locations(cafile=cafile, capath=capath)
            elif hasattr(context, "load_default_certs"):
                context.load_default_certs(ssl.Purpose.SERVER_AUTH)
        if sslopt.get("certfile", None):
            context.load_cert_chain(
                sslopt["certfile"],
                sslopt.get("keyfile", None),
                sslopt.get("password", None),
            )

        # Python 3.10 switch to PROTOCOL_TLS_CLIENT defaults to "cert_reqs = ssl.CERT_REQUIRED" and "check_hostname = True"
        # If both disabled, set check_hostname before verify_mode
        # see https://github.com/liris/websocket-client/commit/b96a2e8fa765753e82eea531adb19716b52ca3ca#commitcomment-10803153
        if sslopt.get("cert_reqs", ssl.CERT_NONE) == ssl.CERT_NONE and not sslopt.get(
            "check_hostname", False
        ):
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
        else:
            context.check_hostname = sslopt.get("check_hostname", True)
            context.verify_mode = sslopt.get("cert_reqs", ssl.CERT_REQUIRED)

        if "ciphers" in sslopt:
            context.set_ciphers(sslopt["ciphers"])
        if "cert_chain" in sslopt:
            certfile, keyfile, password = sslopt["cert_chain"]
            context.load_cert_chain(certfile, keyfile, password)
        if "ecdh_curve" in sslopt:
            context.set_ecdh_curve(sslopt["ecdh_curve"])

    return context.wrap_socket(
        sock,
        do_handshake_on_connect=sslopt.get("do_handshake_on_connect", True),
        suppress_ragged_eofs=sslopt.get("suppress_ragged_eofs", True),
        server_hostname=hostname,
    )


def _ssl_socket(sock: socket.socket, user_sslopt: dict, hostname):
    sslopt: dict = {"cert_reqs": ssl.CERT_REQUIRED}
    sslopt.update(user_sslopt)

    cert_path = os.environ.get("WEBSOCKET_CLIENT_CA_BUNDLE")
    if (
        cert_path
        and os.path.isfile(cert_path)
        and user_sslopt.get("ca_certs", None) is None
    ):
        sslopt["ca_certs"] = cert_path
    elif (
        cert_path
        and os.path.isdir(cert_path)
        and user_sslopt.get("ca_cert_path", None) is None
    ):
        sslopt["ca_cert_path"] = cert_path

    if sslopt.get("server_hostname", None):
        hostname = sslopt["server_hostname"]

    check_hostname = sslopt.get("check_hostname", True)
    sock = _wrap_sni_socket(sock, sslopt, hostname, check_hostname)

    return sock


def _tunnel(sock: socket.socket, host, port: int, auth) -> socket.socket:
    debug("Connecting proxy...")
    connect_header = f"CONNECT {host}:{port} HTTP/1.1\r\n"
    connect_header += f"Host: {host}:{port}\r\n"

    # TODO: support digest auth.
    if auth and auth[0]:
        auth_str = auth[0]
        if auth[1]:
            auth_str += f":{auth[1]}"
        encoded_str = base64encode(auth_str.encode()).strip().decode().replace("\n", "")
        connect_header += f"Proxy-Authorization: Basic {encoded_str}\r\n"
    connect_header += "\r\n"
    dump("request header", connect_header)

    send(sock, connect_header)

    try:
        status, _, _ = read_headers(sock)
    except Exception as e:
        raise WebSocketProxyException(str(e))

    if status != 200:
        raise WebSocketProxyException(f"failed CONNECT via proxy status: {status}")

    return sock


def read_headers(sock: socket.socket) -> tuple:
    status = None
    status_message = None
    headers: dict = {}
    trace("--- response header ---")

    while True:
        line = recv_line(sock)
        line = line.decode("utf-8").strip()
        if not line:
            break
        trace(line)
        if not status:
            status_info = line.split(" ", 2)
            status = int(status_info[1])
            if len(status_info) > 2:
                status_message = status_info[2]
        else:
            kv = line.split(":", 1)
            if len(kv) != 2:
                raise WebSocketException("Invalid header")
            key, value = kv
            if key.lower() == "set-cookie" and headers.get("set-cookie"):
                headers["set-cookie"] = headers.get("set-cookie") + "; " + value.strip()
            else:
                headers[key.lower()] = value.strip()

    trace("-----------------------")

    return status, headers, status_message

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\pool\events.py ===
# pool/events.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
from __future__ import annotations

import typing
from typing import Any
from typing import Optional
from typing import Type
from typing import Union

from .base import ConnectionPoolEntry
from .base import Pool
from .base import PoolProxiedConnection
from .base import PoolResetState
from .. import event
from .. import util

if typing.TYPE_CHECKING:
    from ..engine import Engine
    from ..engine.interfaces import DBAPIConnection


class PoolEvents(event.Events[Pool]):
    """Available events for :class:`_pool.Pool`.

    The methods here define the name of an event as well
    as the names of members that are passed to listener
    functions.

    e.g.::

        from sqlalchemy import event


        def my_on_checkout(dbapi_conn, connection_rec, connection_proxy):
            "handle an on checkout event"


        event.listen(Pool, "checkout", my_on_checkout)

    In addition to accepting the :class:`_pool.Pool` class and
    :class:`_pool.Pool` instances, :class:`_events.PoolEvents` also accepts
    :class:`_engine.Engine` objects and the :class:`_engine.Engine` class as
    targets, which will be resolved to the ``.pool`` attribute of the
    given engine or the :class:`_pool.Pool` class::

        engine = create_engine("postgresql+psycopg2://scott:tiger@localhost/test")

        # will associate with engine.pool
        event.listen(engine, "checkout", my_on_checkout)

    """  # noqa: E501

    _target_class_doc = "SomeEngineOrPool"
    _dispatch_target = Pool

    @util.preload_module("sqlalchemy.engine")
    @classmethod
    def _accept_with(
        cls,
        target: Union[Pool, Type[Pool], Engine, Type[Engine]],
        identifier: str,
    ) -> Optional[Union[Pool, Type[Pool]]]:
        if not typing.TYPE_CHECKING:
            Engine = util.preloaded.engine.Engine

        if isinstance(target, type):
            if issubclass(target, Engine):
                return Pool
            else:
                assert issubclass(target, Pool)
                return target
        elif isinstance(target, Engine):
            return target.pool
        elif isinstance(target, Pool):
            return target
        elif hasattr(target, "_no_async_engine_events"):
            target._no_async_engine_events()
        else:
            return None

    @classmethod
    def _listen(
        cls,
        event_key: event._EventKey[Pool],
        **kw: Any,
    ) -> None:
        target = event_key.dispatch_target

        kw.setdefault("asyncio", target._is_asyncio)

        event_key.base_listen(**kw)

    def connect(
        self,
        dbapi_connection: DBAPIConnection,
        connection_record: ConnectionPoolEntry,
    ) -> None:
        """Called at the moment a particular DBAPI connection is first
        created for a given :class:`_pool.Pool`.

        This event allows one to capture the point directly after which
        the DBAPI module-level ``.connect()`` method has been used in order
        to produce a new DBAPI connection.

        :param dbapi_connection: a DBAPI connection.
         The :attr:`.ConnectionPoolEntry.dbapi_connection` attribute.

        :param connection_record: the :class:`.ConnectionPoolEntry` managing
         the DBAPI connection.

        """

    def first_connect(
        self,
        dbapi_connection: DBAPIConnection,
        connection_record: ConnectionPoolEntry,
    ) -> None:
        """Called exactly once for the first time a DBAPI connection is
        checked out from a particular :class:`_pool.Pool`.

        The rationale for :meth:`_events.PoolEvents.first_connect`
        is to determine
        information about a particular series of database connections based
        on the settings used for all connections.  Since a particular
        :class:`_pool.Pool`
        refers to a single "creator" function (which in terms
        of a :class:`_engine.Engine`
        refers to the URL and connection options used),
        it is typically valid to make observations about a single connection
        that can be safely assumed to be valid about all subsequent
        connections, such as the database version, the server and client
        encoding settings, collation settings, and many others.

        :param dbapi_connection: a DBAPI connection.
         The :attr:`.ConnectionPoolEntry.dbapi_connection` attribute.

        :param connection_record: the :class:`.ConnectionPoolEntry` managing
         the DBAPI connection.

        """

    def checkout(
        self,
        dbapi_connection: DBAPIConnection,
        connection_record: ConnectionPoolEntry,
        connection_proxy: PoolProxiedConnection,
    ) -> None:
        """Called when a connection is retrieved from the Pool.

        :param dbapi_connection: a DBAPI connection.
         The :attr:`.ConnectionPoolEntry.dbapi_connection` attribute.

        :param connection_record: the :class:`.ConnectionPoolEntry` managing
         the DBAPI connection.

        :param connection_proxy: the :class:`.PoolProxiedConnection` object
          which will proxy the public interface of the DBAPI connection for the
          lifespan of the checkout.

        If you raise a :class:`~sqlalchemy.exc.DisconnectionError`, the current
        connection will be disposed and a fresh connection retrieved.
        Processing of all checkout listeners will abort and restart
        using the new connection.

        .. seealso:: :meth:`_events.ConnectionEvents.engine_connect`
           - a similar event
           which occurs upon creation of a new :class:`_engine.Connection`.

        """

    def checkin(
        self,
        dbapi_connection: Optional[DBAPIConnection],
        connection_record: ConnectionPoolEntry,
    ) -> None:
        """Called when a connection returns to the pool.

        Note that the connection may be closed, and may be None if the
        connection has been invalidated.  ``checkin`` will not be called
        for detached connections.  (They do not return to the pool.)

        :param dbapi_connection: a DBAPI connection.
         The :attr:`.ConnectionPoolEntry.dbapi_connection` attribute.

        :param connection_record: the :class:`.ConnectionPoolEntry` managing
         the DBAPI connection.

        """

    @event._legacy_signature(
        "2.0",
        ["dbapi_connection", "connection_record"],
        lambda dbapi_connection, connection_record, reset_state: (
            dbapi_connection,
            connection_record,
        ),
    )
    def reset(
        self,
        dbapi_connection: DBAPIConnection,
        connection_record: ConnectionPoolEntry,
        reset_state: PoolResetState,
    ) -> None:
        """Called before the "reset" action occurs for a pooled connection.

        This event represents
        when the ``rollback()`` method is called on the DBAPI connection
        before it is returned to the pool or discarded.
        A custom "reset" strategy may be implemented using this event hook,
        which may also be combined with disabling the default "reset"
        behavior using the :paramref:`_pool.Pool.reset_on_return` parameter.

        The primary difference between the :meth:`_events.PoolEvents.reset` and
        :meth:`_events.PoolEvents.checkin` events are that
        :meth:`_events.PoolEvents.reset` is called not just for pooled
        connections that are being returned to the pool, but also for
        connections that were detached using the
        :meth:`_engine.Connection.detach` method as well as asyncio connections
        that are being discarded due to garbage collection taking place on
        connections before the connection was checked in.

        Note that the event **is not** invoked for connections that were
        invalidated using :meth:`_engine.Connection.invalidate`.    These
        events may be intercepted using the :meth:`.PoolEvents.soft_invalidate`
        and :meth:`.PoolEvents.invalidate` event hooks, and all "connection
        close" events may be intercepted using :meth:`.PoolEvents.close`.

        The :meth:`_events.PoolEvents.reset` event is usually followed by the
        :meth:`_events.PoolEvents.checkin` event, except in those
        cases where the connection is discarded immediately after reset.

        :param dbapi_connection: a DBAPI connection.
         The :attr:`.ConnectionPoolEntry.dbapi_connection` attribute.

        :param connection_record: the :class:`.ConnectionPoolEntry` managing
         the DBAPI connection.

        :param reset_state: :class:`.PoolResetState` instance which provides
         information about the circumstances under which the connection
         is being reset.

         .. versionadded:: 2.0

        .. seealso::

            :ref:`pool_reset_on_return`

            :meth:`_events.ConnectionEvents.rollback`

            :meth:`_events.ConnectionEvents.commit`

        """

    def invalidate(
        self,
        dbapi_connection: DBAPIConnection,
        connection_record: ConnectionPoolEntry,
        exception: Optional[BaseException],
    ) -> None:
        """Called when a DBAPI connection is to be "invalidated".

        This event is called any time the
        :meth:`.ConnectionPoolEntry.invalidate` method is invoked, either from
        API usage or via "auto-invalidation", without the ``soft`` flag.

        The event occurs before a final attempt to call ``.close()`` on the
        connection occurs.

        :param dbapi_connection: a DBAPI connection.
         The :attr:`.ConnectionPoolEntry.dbapi_connection` attribute.

        :param connection_record: the :class:`.ConnectionPoolEntry` managing
         the DBAPI connection.

        :param exception: the exception object corresponding to the reason
         for this invalidation, if any.  May be ``None``.

        .. seealso::

            :ref:`pool_connection_invalidation`

        """

    def soft_invalidate(
        self,
        dbapi_connection: DBAPIConnection,
        connection_record: ConnectionPoolEntry,
        exception: Optional[BaseException],
    ) -> None:
        """Called when a DBAPI connection is to be "soft invalidated".

        This event is called any time the
        :meth:`.ConnectionPoolEntry.invalidate`
        method is invoked with the ``soft`` flag.

        Soft invalidation refers to when the connection record that tracks
        this connection will force a reconnect after the current connection
        is checked in.   It does not actively close the dbapi_connection
        at the point at which it is called.

        :param dbapi_connection: a DBAPI connection.
         The :attr:`.ConnectionPoolEntry.dbapi_connection` attribute.

        :param connection_record: the :class:`.ConnectionPoolEntry` managing
         the DBAPI connection.

        :param exception: the exception object corresponding to the reason
         for this invalidation, if any.  May be ``None``.

        """

    def close(
        self,
        dbapi_connection: DBAPIConnection,
        connection_record: ConnectionPoolEntry,
    ) -> None:
        """Called when a DBAPI connection is closed.

        The event is emitted before the close occurs.

        The close of a connection can fail; typically this is because
        the connection is already closed.  If the close operation fails,
        the connection is discarded.

        The :meth:`.close` event corresponds to a connection that's still
        associated with the pool. To intercept close events for detached
        connections use :meth:`.close_detached`.

        :param dbapi_connection: a DBAPI connection.
         The :attr:`.ConnectionPoolEntry.dbapi_connection` attribute.

        :param connection_record: the :class:`.ConnectionPoolEntry` managing
         the DBAPI connection.

        """

    def detach(
        self,
        dbapi_connection: DBAPIConnection,
        connection_record: ConnectionPoolEntry,
    ) -> None:
        """Called when a DBAPI connection is "detached" from a pool.

        This event is emitted after the detach occurs.  The connection
        is no longer associated with the given connection record.

        :param dbapi_connection: a DBAPI connection.
         The :attr:`.ConnectionPoolEntry.dbapi_connection` attribute.

        :param connection_record: the :class:`.ConnectionPoolEntry` managing
         the DBAPI connection.

        """

    def close_detached(self, dbapi_connection: DBAPIConnection) -> None:
        """Called when a detached DBAPI connection is closed.

        The event is emitted before the close occurs.

        The close of a connection can fail; typically this is because
        the connection is already closed.  If the close operation fails,
        the connection is discarded.

        :param dbapi_connection: a DBAPI connection.
         The :attr:`.ConnectionPoolEntry.dbapi_connection` attribute.

        """

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\whisper\whisper_encoder_decoder_init.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import logging
import os
import tempfile
from itertools import chain
from pathlib import Path

import numpy as np
import onnx
import torch
from float16 import convert_float_to_float16
from onnx import ModelProto, ValueInfoProto
from onnx_model import OnnxModel
from transformers import WhisperConfig
from whisper_decoder import WhisperDecoder
from whisper_encoder import WhisperEncoder
from whisper_inputs import (
    convert_inputs_for_ort,
    get_model_dynamic_axes,
    get_sample_encoder_decoder_init_inputs,
    group_past_key_values,
)

from onnxruntime import InferenceSession

logger = logging.getLogger(__name__)


class WhisperEncoderDecoderInit(torch.nn.Module):
    """Whisper encoder component + first pass through Whisper decoder component to initialize KV caches"""

    def __init__(self, config: WhisperConfig, model: torch.nn.Module, model_impl: str, no_beam_search_op: bool = False):
        super().__init__()
        self.config = config
        self.device = model.device
        self.model_impl = model_impl
        self.no_beam_search_op = no_beam_search_op

        self.encoder = WhisperEncoder(config, model, model_impl)
        self.decoder = WhisperDecoder(config, model, model_impl, no_beam_search_op)

        self.max_source_positions = self.config.max_source_positions
        self.num_heads = self.config.decoder_attention_heads
        self.head_size = self.config.d_model // self.num_heads

    def hf_forward_for_beam_search_op(self, audio_features: torch.Tensor, decoder_input_ids: torch.Tensor):
        encoder_hidden_states = self.encoder(audio_features)
        logits, present_key_values = self.decoder(decoder_input_ids, encoder_hidden_states)
        return logits, encoder_hidden_states, present_key_values

    def hf_forward_for_no_beam_search_op(self, audio_features: torch.Tensor):
        encoder_hidden_states = self.encoder(audio_features)

        # Get cross attention KV caches and return them for this model
        # We do this because these MatMuls are only run once before their outputs are being re-used in the decoder
        present_cross_attention_key_value_caches = []
        for layer in self.decoder.decoder.layers:
            cross_attn_key_cache = (
                layer.encoder_attn.k_proj(encoder_hidden_states)
                .view(-1, self.max_source_positions, self.num_heads, self.head_size)
                .transpose(1, 2)
            )
            cross_attn_value_cache = (
                layer.encoder_attn.v_proj(encoder_hidden_states)
                .view(-1, self.max_source_positions, self.num_heads, self.head_size)
                .transpose(1, 2)
            )
            present_cross_attention_key_value_caches.append(cross_attn_key_cache)
            present_cross_attention_key_value_caches.append(cross_attn_value_cache)

        return encoder_hidden_states, present_cross_attention_key_value_caches

    def oai_forward_for_beam_search_op(self, audio_features: torch.Tensor, decoder_input_ids: torch.Tensor):
        encoder_hidden_states = self.encoder(audio_features)
        logits, present_key_values = self.decoder(decoder_input_ids, encoder_hidden_states)
        return logits, encoder_hidden_states, present_key_values

    def oai_forward_for_no_beam_search_op(self, audio_features: torch.Tensor):
        encoder_hidden_states = self.encoder(audio_features)

        # Get cross attention KV caches and return them for this model
        # We do this because these MatMuls are only run once before their outputs are being re-used in the decoder
        present_cross_attention_key_value_caches = []
        for block in self.decoder.model.decoder.blocks:
            cross_attn_key_cache = (
                block.cross_attn.key(encoder_hidden_states)
                .view(-1, self.max_source_positions, self.num_heads, self.head_size)
                .transpose(1, 2)
            )
            cross_attn_value_cache = (
                block.cross_attn.value(encoder_hidden_states)
                .view(-1, self.max_source_positions, self.num_heads, self.head_size)
                .transpose(1, 2)
            )
            present_cross_attention_key_value_caches.append(cross_attn_key_cache)
            present_cross_attention_key_value_caches.append(cross_attn_value_cache)

        return encoder_hidden_states, present_cross_attention_key_value_caches

    def forward(self, audio_features: torch.Tensor, decoder_input_ids: torch.Tensor | None = None):
        if self.model_impl == "openai":
            if self.no_beam_search_op:
                return self.oai_forward_for_no_beam_search_op(audio_features)
            return self.oai_forward_for_beam_search_op(audio_features, decoder_input_ids)

        # Hugging Face implementation
        if self.no_beam_search_op:
            return self.hf_forward_for_no_beam_search_op(audio_features)
        return self.hf_forward_for_beam_search_op(audio_features, decoder_input_ids)

    def input_names(self):
        if self.no_beam_search_op:
            input_names = ["audio_features"]
        else:
            input_names = ["encoder_input_ids", "decoder_input_ids"]
        return input_names

    def output_names(self):
        if self.no_beam_search_op:
            output_names = [
                "encoder_hidden_states",
                *list(
                    chain.from_iterable(
                        (f"present_key_cross_{i}", f"present_value_cross_{i}")
                        for i in range(self.config.num_hidden_layers)
                    )
                ),
            ]
        else:
            output_names = [
                "logits",
                "encoder_hidden_states",
                *list(
                    chain.from_iterable(
                        (
                            f"present_key_self_{i}",
                            f"present_value_self_{i}",
                            f"present_key_cross_{i}",
                            f"present_value_cross_{i}",
                        )
                        for i in range(self.config.num_hidden_layers)
                    )
                ),
            ]
        return output_names

    def dynamic_axes(self, input_names, output_names):
        dynamic_axes = get_model_dynamic_axes(self.config, input_names, output_names)
        return dynamic_axes

    def inputs(self, use_fp16_inputs: bool, use_int32_inputs: bool, return_dict: bool = False):
        inputs = get_sample_encoder_decoder_init_inputs(
            self.config,
            self.device,
            batch_size=2,
            decoder_sequence_length=6,
            use_fp16=use_fp16_inputs,
            use_int32=use_int32_inputs,
        )
        if return_dict:
            if self.no_beam_search_op:
                del inputs["decoder_input_ids"]
            return inputs

        if self.no_beam_search_op:
            return (inputs["audio_features"],)
        return (
            inputs["audio_features"],
            inputs["decoder_input_ids"],
        )

    def fix_key_value_cache_dims(self, output: ValueInfoProto, is_cross: bool = False):
        # Shape should be (batch_size, num_heads, sequence_length, head_size) for self attention KV caches
        # and (batch_size, num_heads, num_frames // 2, head_size) for cross attention KV caches
        num_heads = output.type.tensor_type.shape.dim[1]
        if "_dim_" in num_heads.dim_param:
            num_heads.Clear()
            num_heads.dim_value = self.num_heads
        sequence_length = output.type.tensor_type.shape.dim[2]
        if "_dim_" in sequence_length.dim_param:
            sequence_length.Clear()
            if is_cross:
                sequence_length.dim_value = self.max_source_positions
            else:
                sequence_length.dim_param = "total_sequence_length"
        head_size = output.type.tensor_type.shape.dim[3]
        if "_dim_" in head_size.dim_param:
            head_size.Clear()
            head_size.dim_value = self.head_size
        return output

    def fix_outputs(self, model: ModelProto):
        # ONNX exporter might mark dimensions like 'Transposepresent_value_self_1_dim_2' in shape inference.
        # We now change the dim_values to the correct one.
        reordered_outputs = []
        self_attn_kv_caches = []
        cross_attn_kv_caches = []

        for output in model.graph.output:
            if "present" not in output.name:
                reordered_outputs.append(output)

            elif "self" in output.name:
                # Self attention KV caches
                new_output = self.fix_key_value_cache_dims(output, is_cross=False)
                if self.no_beam_search_op:
                    reordered_outputs.append(new_output)
                else:
                    self_attn_kv_caches.append(new_output)
            else:
                # Cross attention KV caches
                new_output = self.fix_key_value_cache_dims(output, is_cross=True)
                if self.no_beam_search_op:
                    reordered_outputs.append(new_output)
                else:
                    cross_attn_kv_caches.append(new_output)

        if not self.no_beam_search_op:
            reordered_outputs += self_attn_kv_caches + cross_attn_kv_caches

        while len(model.graph.output) > 0:
            model.graph.output.pop()
        model.graph.output.extend(reordered_outputs)
        return model

    def fix_layernorm_weights(self, model: ModelProto, use_fp16_inputs: bool):
        if self.model_impl == "openai" and use_fp16_inputs:
            # Cast ONNX model to float16 to ensure LayerNorm weights are converted from
            # float32 to float16 since exported model already has float16 weights everywhere
            # except for LayerNorm ops. This happens because OpenAI always upcasts to float32
            # when computing LayerNorm.
            #
            # Reference:
            # https://github.com/openai/whisper/blob/90db0de1896c23cbfaf0c58bc2d30665f709f170/whisper/model.py#L41
            model = convert_float_to_float16(model)
        return model

    def export_onnx(
        self,
        onnx_model_path: str,
        provider: str,
        verbose: bool = True,
        use_external_data_format: bool = False,
        use_fp16_inputs: bool = False,
        use_int32_inputs: bool = True,
    ):
        """Export encoder-decoder-init to ONNX

        Args:
            onnx_model_path (str): path to save ONNX model
            provider (str): provider to use for verifying parity on ONNX model
            verbose (bool, optional): print verbose information. Defaults to True.
            use_external_data_format (bool, optional): use external data format or not. Defaults to False.
            use_fp16_inputs (bool, optional): use float16 inputs for the audio_features. Defaults to False.
            use_int32_inputs (bool, optional): use int32 inputs for the decoder_input_ids. Defaults to True.
        """
        # Shape of encoder's tensors:
        # Inputs:
        #    audio_features: (batch_size, num_mels, num_frames)
        # Outputs:
        #    encoder_hidden_states: (batch_size, num_frames // 2, hidden_size)

        # Shape of decoder's tensors:
        # Inputs:
        #    decoder_input_ids: (batch_size, sequence_length)
        #    encoder_hidden_states (comes from encoder's outputs): (batch_size, num_frames // 2, hidden_size)
        # Outputs:
        #    logits: (batch_size, sequence_length, vocab_size)
        #    present_{key/value}_self_* (present self attention KV caches): (batch_size, num_heads, past_sequence_length + sequence_length, head_size)
        #    present_{key/value}_cross_* (present cross attention KV caches): (batch_size, num_heads, num_frames // 2, head_size)

        inputs = self.inputs(use_fp16_inputs=use_fp16_inputs, use_int32_inputs=use_int32_inputs)
        input_names = self.input_names()
        output_names = self.output_names()
        dynamic_axes = self.dynamic_axes(input_names, output_names)

        Path(onnx_model_path).parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            temp_onnx_model_path = os.path.join(tmp_dir_name, "encoder_decoder_init.onnx")
            Path(temp_onnx_model_path).parent.mkdir(parents=True, exist_ok=True)
            out_path = temp_onnx_model_path if use_external_data_format else onnx_model_path

            torch.onnx.export(
                self,
                args=inputs,
                f=out_path,
                export_params=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                opset_version=17,
                do_constant_folding=True,
                verbose=verbose,
            )

            model = onnx.load_model(out_path, load_external_data=use_external_data_format)
            model = self.fix_outputs(model)
            model = self.fix_layernorm_weights(model, use_fp16_inputs)
            OnnxModel.save(
                model,
                onnx_model_path,
                save_as_external_data=use_external_data_format,
                all_tensors_to_one_file=True,
            )

        self.verify_onnx(onnx_model_path, provider, use_fp16_inputs, use_int32_inputs)

    def verify_onnx(
        self,
        onnx_model_path: str,
        provider: str,
        use_fp16_inputs: bool,
        use_int32_inputs: bool,
    ):
        """Verify ONNX model outputs and PyTorch model outputs match

        Args:
            onnx_model_path (str): path to save ONNX model
            provider (str): execution provider for ONNX model
            use_fp16_inputs (bool, optional): use float16 inputs for the audio_features
            use_int32_inputs (bool, optional): use int32 inputs for the decoder_input_ids
        """
        # Shape of encoder's tensors:
        # Inputs:
        #    audio_features: (batch_size, num_mels, num_frames)
        # Outputs:
        #    encoder_hidden_states: (batch_size, num_frames // 2, hidden_size)

        # Shape of decoder's tensors:
        # Inputs:
        #    decoder_input_ids: (batch_size, sequence_length)
        #    encoder_hidden_states (comes from encoder's outputs): (batch_size, num_frames // 2, hidden_size)
        # Outputs:
        #    logits: (batch_size, sequence_length, vocab_size)
        #    present_{key/value}_self_* (present self attention KV caches): (batch_size, num_heads, past_sequence_length + sequence_length, head_size)
        #    present_{key/value}_cross_* (present cross attention KV caches): (batch_size, num_heads, num_frames // 2, head_size)

        inputs = self.inputs(use_fp16_inputs=use_fp16_inputs, use_int32_inputs=use_int32_inputs, return_dict=True)

        # Run PyTorch model
        pt_outputs = []
        if self.no_beam_search_op:
            out = self.forward(**inputs)
            pt_outputs.append(out[0].detach().cpu().numpy())
            for present_cross_attn_cache in out[1]:
                pt_outputs.append(present_cross_attn_cache.detach().cpu().numpy())
        else:
            out = self.forward(**inputs)
            pt_outputs.append(out[0].detach().cpu().numpy())
            pt_outputs.append(out[1].detach().cpu().numpy())

            (self_attn_kv_caches, cross_attn_kv_caches) = group_past_key_values(out[2])
            pt_outputs.extend([self_attn_kv_cache.detach().cpu().numpy() for self_attn_kv_cache in self_attn_kv_caches])
            pt_outputs.extend(
                [cross_attn_kv_cache.detach().cpu().numpy() for cross_attn_kv_cache in cross_attn_kv_caches]
            )

        # Run ONNX model
        sess = InferenceSession(onnx_model_path, providers=[provider])
        ort_outputs = sess.run(None, convert_inputs_for_ort(inputs, sess))

        # Calculate output difference
        for i, output_name in enumerate(self.output_names()):
            diff = np.abs(pt_outputs[i] - ort_outputs[i])
            logger.warning(f"Comparing {output_name}...")
            logger.warning(f"Max diff: {np.max(diff)}")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\vector\printing.py ===
from sympy.core.function import Derivative
from sympy.core.function import UndefinedFunction, AppliedUndef
from sympy.core.symbol import Symbol
from sympy.interactive.printing import init_printing
from sympy.printing.latex import LatexPrinter
from sympy.printing.pretty.pretty import PrettyPrinter
from sympy.printing.pretty.pretty_symbology import center_accent
from sympy.printing.str import StrPrinter
from sympy.printing.precedence import PRECEDENCE

__all__ = ['vprint', 'vsstrrepr', 'vsprint', 'vpprint', 'vlatex',
           'init_vprinting']


class VectorStrPrinter(StrPrinter):
    """String Printer for vector expressions. """

    def _print_Derivative(self, e):
        from sympy.physics.vector.functions import dynamicsymbols
        t = dynamicsymbols._t
        if (bool(sum(i == t for i in e.variables)) &
                isinstance(type(e.args[0]), UndefinedFunction)):
            ol = str(e.args[0].func)
            for i, v in enumerate(e.variables):
                ol += dynamicsymbols._str
            return ol
        else:
            return StrPrinter().doprint(e)

    def _print_Function(self, e):
        from sympy.physics.vector.functions import dynamicsymbols
        t = dynamicsymbols._t
        if isinstance(type(e), UndefinedFunction):
            return StrPrinter().doprint(e).replace("(%s)" % t, '')
        return e.func.__name__ + "(%s)" % self.stringify(e.args, ", ")


class VectorStrReprPrinter(VectorStrPrinter):
    """String repr printer for vector expressions."""
    def _print_str(self, s):
        return repr(s)


class VectorLatexPrinter(LatexPrinter):
    """Latex Printer for vector expressions. """

    def _print_Function(self, expr, exp=None):
        from sympy.physics.vector.functions import dynamicsymbols
        func = expr.func.__name__
        t = dynamicsymbols._t

        if (hasattr(self, '_print_' + func) and not
            isinstance(type(expr), UndefinedFunction)):
            return getattr(self, '_print_' + func)(expr, exp)
        elif isinstance(type(expr), UndefinedFunction) and (expr.args == (t,)):
            # treat this function like a symbol
            expr = Symbol(func)
            if exp is not None:
                # copied from LatexPrinter._helper_print_standard_power, which
                # we can't call because we only have exp as a string.
                base = self.parenthesize(expr, PRECEDENCE['Pow'])
                base = self.parenthesize_super(base)
                return r"%s^{%s}" % (base, exp)
            else:
                return super()._print(expr)
        else:
            return super()._print_Function(expr, exp)

    def _print_Derivative(self, der_expr):
        from sympy.physics.vector.functions import dynamicsymbols
        # make sure it is in the right form
        der_expr = der_expr.doit()
        if not isinstance(der_expr, Derivative):
            return r"\left(%s\right)" % self.doprint(der_expr)

        # check if expr is a dynamicsymbol
        t = dynamicsymbols._t
        expr = der_expr.expr
        red = expr.atoms(AppliedUndef)
        syms = der_expr.variables
        test1 = not all(True for i in red if i.free_symbols == {t})
        test2 = not all(t == i for i in syms)
        if test1 or test2:
            return super()._print_Derivative(der_expr)

        # done checking
        dots = len(syms)
        base = self._print_Function(expr)
        base_split = base.split('_', 1)
        base = base_split[0]
        if dots == 1:
            base = r"\dot{%s}" % base
        elif dots == 2:
            base = r"\ddot{%s}" % base
        elif dots == 3:
            base = r"\dddot{%s}" % base
        elif dots == 4:
            base = r"\ddddot{%s}" % base
        else:  # Fallback to standard printing
            return super()._print_Derivative(der_expr)
        if len(base_split) != 1:
            base += '_' + base_split[1]
        return base


class VectorPrettyPrinter(PrettyPrinter):
    """Pretty Printer for vectorialexpressions. """

    def _print_Derivative(self, deriv):
        from sympy.physics.vector.functions import dynamicsymbols
        # XXX use U('PARTIAL DIFFERENTIAL') here ?
        t = dynamicsymbols._t
        dot_i = 0
        syms = list(reversed(deriv.variables))

        while len(syms) > 0:
            if syms[-1] == t:
                syms.pop()
                dot_i += 1
            else:
                return super()._print_Derivative(deriv)

        if not (isinstance(type(deriv.expr), UndefinedFunction) and
                (deriv.expr.args == (t,))):
            return super()._print_Derivative(deriv)
        else:
            pform = self._print_Function(deriv.expr)

        # the following condition would happen with some sort of non-standard
        # dynamic symbol I guess, so we'll just print the SymPy way
        if len(pform.picture) > 1:
            return super()._print_Derivative(deriv)

        # There are only special symbols up to fourth-order derivatives
        if dot_i >= 5:
            return super()._print_Derivative(deriv)

        # Deal with special symbols
        dots = {0: "",
                1: "\N{COMBINING DOT ABOVE}",
                2: "\N{COMBINING DIAERESIS}",
                3: "\N{COMBINING THREE DOTS ABOVE}",
                4: "\N{COMBINING FOUR DOTS ABOVE}"}

        d = pform.__dict__
        # if unicode is false then calculate number of apostrophes needed and
        # add to output
        if not self._use_unicode:
            apostrophes = ""
            for i in range(0, dot_i):
                apostrophes += "'"
            d['picture'][0] += apostrophes + "(t)"
        else:
            d['picture'] = [center_accent(d['picture'][0], dots[dot_i])]
        return pform

    def _print_Function(self, e):
        from sympy.physics.vector.functions import dynamicsymbols
        t = dynamicsymbols._t
        # XXX works only for applied functions
        func = e.func
        args = e.args
        func_name = func.__name__
        pform = self._print_Symbol(Symbol(func_name))
        # If this function is an Undefined function of t, it is probably a
        # dynamic symbol, so we'll skip the (t). The rest of the code is
        # identical to the normal PrettyPrinter code
        if not (isinstance(func, UndefinedFunction) and (args == (t,))):
            return super()._print_Function(e)
        return pform


def vprint(expr, **settings):
    r"""Function for printing of expressions generated in the
    sympy.physics vector package.

    Extends SymPy's StrPrinter, takes the same setting accepted by SymPy's
    :func:`~.sstr`, and is equivalent to ``print(sstr(foo))``.

    Parameters
    ==========

    expr : valid SymPy object
        SymPy expression to print.
    settings : args
        Same as the settings accepted by SymPy's sstr().

    Examples
    ========

    >>> from sympy.physics.vector import vprint, dynamicsymbols
    >>> u1 = dynamicsymbols('u1')
    >>> print(u1)
    u1(t)
    >>> vprint(u1)
    u1

    """

    outstr = vsprint(expr, **settings)

    import builtins
    if (outstr != 'None'):
        builtins._ = outstr
        print(outstr)


def vsstrrepr(expr, **settings):
    """Function for displaying expression representation's with vector
    printing enabled.

    Parameters
    ==========

    expr : valid SymPy object
        SymPy expression to print.
    settings : args
        Same as the settings accepted by SymPy's sstrrepr().

    """
    p = VectorStrReprPrinter(settings)
    return p.doprint(expr)


def vsprint(expr, **settings):
    r"""Function for displaying expressions generated in the
    sympy.physics vector package.

    Returns the output of vprint() as a string.

    Parameters
    ==========

    expr : valid SymPy object
        SymPy expression to print
    settings : args
        Same as the settings accepted by SymPy's sstr().

    Examples
    ========

    >>> from sympy.physics.vector import vsprint, dynamicsymbols
    >>> u1, u2 = dynamicsymbols('u1 u2')
    >>> u2d = dynamicsymbols('u2', level=1)
    >>> print("%s = %s" % (u1, u2 + u2d))
    u1(t) = u2(t) + Derivative(u2(t), t)
    >>> print("%s = %s" % (vsprint(u1), vsprint(u2 + u2d)))
    u1 = u2 + u2'

    """

    string_printer = VectorStrPrinter(settings)
    return string_printer.doprint(expr)


def vpprint(expr, **settings):
    r"""Function for pretty printing of expressions generated in the
    sympy.physics vector package.

    Mainly used for expressions not inside a vector; the output of running
    scripts and generating equations of motion. Takes the same options as
    SymPy's :func:`~.pretty_print`; see that function for more information.

    Parameters
    ==========

    expr : valid SymPy object
        SymPy expression to pretty print
    settings : args
        Same as those accepted by SymPy's pretty_print.


    """

    pp = VectorPrettyPrinter(settings)

    # Note that this is copied from sympy.printing.pretty.pretty_print:

    # XXX: this is an ugly hack, but at least it works
    use_unicode = pp._settings['use_unicode']
    from sympy.printing.pretty.pretty_symbology import pretty_use_unicode
    uflag = pretty_use_unicode(use_unicode)

    try:
        return pp.doprint(expr)
    finally:
        pretty_use_unicode(uflag)


def vlatex(expr, **settings):
    r"""Function for printing latex representation of sympy.physics.vector
    objects.

    For latex representation of Vectors, Dyadics, and dynamicsymbols. Takes the
    same options as SymPy's :func:`~.latex`; see that function for more
    information;

    Parameters
    ==========

    expr : valid SymPy object
        SymPy expression to represent in LaTeX form
    settings : args
        Same as latex()

    Examples
    ========

    >>> from sympy.physics.vector import vlatex, ReferenceFrame, dynamicsymbols
    >>> N = ReferenceFrame('N')
    >>> q1, q2 = dynamicsymbols('q1 q2')
    >>> q1d, q2d = dynamicsymbols('q1 q2', 1)
    >>> q1dd, q2dd = dynamicsymbols('q1 q2', 2)
    >>> vlatex(N.x + N.y)
    '\\mathbf{\\hat{n}_x} + \\mathbf{\\hat{n}_y}'
    >>> vlatex(q1 + q2)
    'q_{1} + q_{2}'
    >>> vlatex(q1d)
    '\\dot{q}_{1}'
    >>> vlatex(q1 * q2d)
    'q_{1} \\dot{q}_{2}'
    >>> vlatex(q1dd * q1 / q1d)
    '\\frac{q_{1} \\ddot{q}_{1}}{\\dot{q}_{1}}'

    """
    latex_printer = VectorLatexPrinter(settings)

    return latex_printer.doprint(expr)


def init_vprinting(**kwargs):
    """Initializes time derivative printing for all SymPy objects, i.e. any
    functions of time will be displayed in a more compact notation. The main
    benefit of this is for printing of time derivatives; instead of
    displaying as ``Derivative(f(t),t)``, it will display ``f'``. This is
    only actually needed for when derivatives are present and are not in a
    physics.vector.Vector or physics.vector.Dyadic object. This function is a
    light wrapper to :func:`~.init_printing`. Any keyword
    arguments for it are valid here.

    {0}

    Examples
    ========

    >>> from sympy import Function, symbols
    >>> t, x = symbols('t, x')
    >>> omega = Function('omega')
    >>> omega(x).diff()
    Derivative(omega(x), x)
    >>> omega(t).diff()
    Derivative(omega(t), t)

    Now use the string printer:

    >>> from sympy.physics.vector import init_vprinting
    >>> init_vprinting(pretty_print=False)
    >>> omega(x).diff()
    Derivative(omega(x), x)
    >>> omega(t).diff()
    omega'

    """
    kwargs['str_printer'] = vsstrrepr
    kwargs['pretty_printer'] = vpprint
    kwargs['latex_printer'] = vlatex
    init_printing(**kwargs)


params = init_printing.__doc__.split('Examples\n    ========')[0]  # type: ignore
init_vprinting.__doc__ = init_vprinting.__doc__.format(params)  # type: ignore

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\dateutil\tz\win.py ===
# -*- coding: utf-8 -*-
"""
This module provides an interface to the native time zone data on Windows,
including :py:class:`datetime.tzinfo` implementations.

Attempting to import this module on a non-Windows platform will raise an
:py:obj:`ImportError`.
"""
# This code was originally contributed by Jeffrey Harris.
import datetime
import struct

from six.moves import winreg
from six import text_type

try:
    import ctypes
    from ctypes import wintypes
except ValueError:
    # ValueError is raised on non-Windows systems for some horrible reason.
    raise ImportError("Running tzwin on non-Windows system")

from ._common import tzrangebase

__all__ = ["tzwin", "tzwinlocal", "tzres"]

ONEWEEK = datetime.timedelta(7)

TZKEYNAMENT = r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Time Zones"
TZKEYNAME9X = r"SOFTWARE\Microsoft\Windows\CurrentVersion\Time Zones"
TZLOCALKEYNAME = r"SYSTEM\CurrentControlSet\Control\TimeZoneInformation"


def _settzkeyname():
    handle = winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE)
    try:
        winreg.OpenKey(handle, TZKEYNAMENT).Close()
        TZKEYNAME = TZKEYNAMENT
    except WindowsError:
        TZKEYNAME = TZKEYNAME9X
    handle.Close()
    return TZKEYNAME


TZKEYNAME = _settzkeyname()


class tzres(object):
    """
    Class for accessing ``tzres.dll``, which contains timezone name related
    resources.

    .. versionadded:: 2.5.0
    """
    p_wchar = ctypes.POINTER(wintypes.WCHAR)        # Pointer to a wide char

    def __init__(self, tzres_loc='tzres.dll'):
        # Load the user32 DLL so we can load strings from tzres
        user32 = ctypes.WinDLL('user32')

        # Specify the LoadStringW function
        user32.LoadStringW.argtypes = (wintypes.HINSTANCE,
                                       wintypes.UINT,
                                       wintypes.LPWSTR,
                                       ctypes.c_int)

        self.LoadStringW = user32.LoadStringW
        self._tzres = ctypes.WinDLL(tzres_loc)
        self.tzres_loc = tzres_loc

    def load_name(self, offset):
        """
        Load a timezone name from a DLL offset (integer).

        >>> from dateutil.tzwin import tzres
        >>> tzr = tzres()
        >>> print(tzr.load_name(112))
        'Eastern Standard Time'

        :param offset:
            A positive integer value referring to a string from the tzres dll.

        .. note::

            Offsets found in the registry are generally of the form
            ``@tzres.dll,-114``. The offset in this case is 114, not -114.

        """
        resource = self.p_wchar()
        lpBuffer = ctypes.cast(ctypes.byref(resource), wintypes.LPWSTR)
        nchar = self.LoadStringW(self._tzres._handle, offset, lpBuffer, 0)
        return resource[:nchar]

    def name_from_string(self, tzname_str):
        """
        Parse strings as returned from the Windows registry into the time zone
        name as defined in the registry.

        >>> from dateutil.tzwin import tzres
        >>> tzr = tzres()
        >>> print(tzr.name_from_string('@tzres.dll,-251'))
        'Dateline Daylight Time'
        >>> print(tzr.name_from_string('Eastern Standard Time'))
        'Eastern Standard Time'

        :param tzname_str:
            A timezone name string as returned from a Windows registry key.

        :return:
            Returns the localized timezone string from tzres.dll if the string
            is of the form `@tzres.dll,-offset`, else returns the input string.
        """
        if not tzname_str.startswith('@'):
            return tzname_str

        name_splt = tzname_str.split(',-')
        try:
            offset = int(name_splt[1])
        except:
            raise ValueError("Malformed timezone string.")

        return self.load_name(offset)


class tzwinbase(tzrangebase):
    """tzinfo class based on win32's timezones available in the registry."""
    def __init__(self):
        raise NotImplementedError('tzwinbase is an abstract base class')

    def __eq__(self, other):
        # Compare on all relevant dimensions, including name.
        if not isinstance(other, tzwinbase):
            return NotImplemented

        return  (self._std_offset == other._std_offset and
                 self._dst_offset == other._dst_offset and
                 self._stddayofweek == other._stddayofweek and
                 self._dstdayofweek == other._dstdayofweek and
                 self._stdweeknumber == other._stdweeknumber and
                 self._dstweeknumber == other._dstweeknumber and
                 self._stdhour == other._stdhour and
                 self._dsthour == other._dsthour and
                 self._stdminute == other._stdminute and
                 self._dstminute == other._dstminute and
                 self._std_abbr == other._std_abbr and
                 self._dst_abbr == other._dst_abbr)

    @staticmethod
    def list():
        """Return a list of all time zones known to the system."""
        with winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE) as handle:
            with winreg.OpenKey(handle, TZKEYNAME) as tzkey:
                result = [winreg.EnumKey(tzkey, i)
                          for i in range(winreg.QueryInfoKey(tzkey)[0])]
        return result

    def display(self):
        """
        Return the display name of the time zone.
        """
        return self._display

    def transitions(self, year):
        """
        For a given year, get the DST on and off transition times, expressed
        always on the standard time side. For zones with no transitions, this
        function returns ``None``.

        :param year:
            The year whose transitions you would like to query.

        :return:
            Returns a :class:`tuple` of :class:`datetime.datetime` objects,
            ``(dston, dstoff)`` for zones with an annual DST transition, or
            ``None`` for fixed offset zones.
        """

        if not self.hasdst:
            return None

        dston = picknthweekday(year, self._dstmonth, self._dstdayofweek,
                               self._dsthour, self._dstminute,
                               self._dstweeknumber)

        dstoff = picknthweekday(year, self._stdmonth, self._stddayofweek,
                                self._stdhour, self._stdminute,
                                self._stdweeknumber)

        # Ambiguous dates default to the STD side
        dstoff -= self._dst_base_offset

        return dston, dstoff

    def _get_hasdst(self):
        return self._dstmonth != 0

    @property
    def _dst_base_offset(self):
        return self._dst_base_offset_


class tzwin(tzwinbase):
    """
    Time zone object created from the zone info in the Windows registry

    These are similar to :py:class:`dateutil.tz.tzrange` objects in that
    the time zone data is provided in the format of a single offset rule
    for either 0 or 2 time zone transitions per year.

    :param: name
        The name of a Windows time zone key, e.g. "Eastern Standard Time".
        The full list of keys can be retrieved with :func:`tzwin.list`.
    """

    def __init__(self, name):
        self._name = name

        with winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE) as handle:
            tzkeyname = text_type("{kn}\\{name}").format(kn=TZKEYNAME, name=name)
            with winreg.OpenKey(handle, tzkeyname) as tzkey:
                keydict = valuestodict(tzkey)

        self._std_abbr = keydict["Std"]
        self._dst_abbr = keydict["Dlt"]

        self._display = keydict["Display"]

        # See http://ww_winreg.jsiinc.com/SUBA/tip0300/rh0398.htm
        tup = struct.unpack("=3l16h", keydict["TZI"])
        stdoffset = -tup[0]-tup[1]          # Bias + StandardBias * -1
        dstoffset = stdoffset-tup[2]        # + DaylightBias * -1
        self._std_offset = datetime.timedelta(minutes=stdoffset)
        self._dst_offset = datetime.timedelta(minutes=dstoffset)

        # for the meaning see the win32 TIME_ZONE_INFORMATION structure docs
        # http://msdn.microsoft.com/en-us/library/windows/desktop/ms725481(v=vs.85).aspx
        (self._stdmonth,
         self._stddayofweek,   # Sunday = 0
         self._stdweeknumber,  # Last = 5
         self._stdhour,
         self._stdminute) = tup[4:9]

        (self._dstmonth,
         self._dstdayofweek,   # Sunday = 0
         self._dstweeknumber,  # Last = 5
         self._dsthour,
         self._dstminute) = tup[12:17]

        self._dst_base_offset_ = self._dst_offset - self._std_offset
        self.hasdst = self._get_hasdst()

    def __repr__(self):
        return "tzwin(%s)" % repr(self._name)

    def __reduce__(self):
        return (self.__class__, (self._name,))


class tzwinlocal(tzwinbase):
    """
    Class representing the local time zone information in the Windows registry

    While :class:`dateutil.tz.tzlocal` makes system calls (via the :mod:`time`
    module) to retrieve time zone information, ``tzwinlocal`` retrieves the
    rules directly from the Windows registry and creates an object like
    :class:`dateutil.tz.tzwin`.

    Because Windows does not have an equivalent of :func:`time.tzset`, on
    Windows, :class:`dateutil.tz.tzlocal` instances will always reflect the
    time zone settings *at the time that the process was started*, meaning
    changes to the machine's time zone settings during the run of a program
    on Windows will **not** be reflected by :class:`dateutil.tz.tzlocal`.
    Because ``tzwinlocal`` reads the registry directly, it is unaffected by
    this issue.
    """
    def __init__(self):
        with winreg.ConnectRegistry(None, winreg.HKEY_LOCAL_MACHINE) as handle:
            with winreg.OpenKey(handle, TZLOCALKEYNAME) as tzlocalkey:
                keydict = valuestodict(tzlocalkey)

            self._std_abbr = keydict["StandardName"]
            self._dst_abbr = keydict["DaylightName"]

            try:
                tzkeyname = text_type('{kn}\\{sn}').format(kn=TZKEYNAME,
                                                          sn=self._std_abbr)
                with winreg.OpenKey(handle, tzkeyname) as tzkey:
                    _keydict = valuestodict(tzkey)
                    self._display = _keydict["Display"]
            except OSError:
                self._display = None

        stdoffset = -keydict["Bias"]-keydict["StandardBias"]
        dstoffset = stdoffset-keydict["DaylightBias"]

        self._std_offset = datetime.timedelta(minutes=stdoffset)
        self._dst_offset = datetime.timedelta(minutes=dstoffset)

        # For reasons unclear, in this particular key, the day of week has been
        # moved to the END of the SYSTEMTIME structure.
        tup = struct.unpack("=8h", keydict["StandardStart"])

        (self._stdmonth,
         self._stdweeknumber,  # Last = 5
         self._stdhour,
         self._stdminute) = tup[1:5]

        self._stddayofweek = tup[7]

        tup = struct.unpack("=8h", keydict["DaylightStart"])

        (self._dstmonth,
         self._dstweeknumber,  # Last = 5
         self._dsthour,
         self._dstminute) = tup[1:5]

        self._dstdayofweek = tup[7]

        self._dst_base_offset_ = self._dst_offset - self._std_offset
        self.hasdst = self._get_hasdst()

    def __repr__(self):
        return "tzwinlocal()"

    def __str__(self):
        # str will return the standard name, not the daylight name.
        return "tzwinlocal(%s)" % repr(self._std_abbr)

    def __reduce__(self):
        return (self.__class__, ())


def picknthweekday(year, month, dayofweek, hour, minute, whichweek):
    """ dayofweek == 0 means Sunday, whichweek 5 means last instance """
    first = datetime.datetime(year, month, 1, hour, minute)

    # This will work if dayofweek is ISO weekday (1-7) or Microsoft-style (0-6),
    # Because 7 % 7 = 0
    weekdayone = first.replace(day=((dayofweek - first.isoweekday()) % 7) + 1)
    wd = weekdayone + ((whichweek - 1) * ONEWEEK)
    if (wd.month != month):
        wd -= ONEWEEK

    return wd


def valuestodict(key):
    """Convert a registry key's values to a dictionary."""
    dout = {}
    size = winreg.QueryInfoKey(key)[1]
    tz_res = None

    for i in range(size):
        key_name, value, dtype = winreg.EnumValue(key, i)
        if dtype == winreg.REG_DWORD or dtype == winreg.REG_DWORD_LITTLE_ENDIAN:
            # If it's a DWORD (32-bit integer), it's stored as unsigned - convert
            # that to a proper signed integer
            if value & (1 << 31):
                value = value - (1 << 32)
        elif dtype == winreg.REG_SZ:
            # If it's a reference to the tzres DLL, load the actual string
            if value.startswith('@tzres'):
                tz_res = tz_res or tzres()
                value = tz_res.name_from_string(value)

            value = value.rstrip('\x00')    # Remove trailing nulls

        dout[key_name] = value

    return dout

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\plotting\_matplotlib\timeseries.py ===
# TODO: Use the fact that axis can have units to simplify the process

from __future__ import annotations

import functools
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)
import warnings

import numpy as np

from pandas._libs.tslibs import (
    BaseOffset,
    Period,
    to_offset,
)
from pandas._libs.tslibs.dtypes import (
    OFFSET_TO_PERIOD_FREQSTR,
    FreqGroup,
)

from pandas.core.dtypes.generic import (
    ABCDatetimeIndex,
    ABCPeriodIndex,
    ABCTimedeltaIndex,
)

from pandas.io.formats.printing import pprint_thing
from pandas.plotting._matplotlib.converter import (
    TimeSeries_DateFormatter,
    TimeSeries_DateLocator,
    TimeSeries_TimedeltaFormatter,
)
from pandas.tseries.frequencies import (
    get_period_alias,
    is_subperiod,
    is_superperiod,
)

if TYPE_CHECKING:
    from datetime import timedelta

    from matplotlib.axes import Axes

    from pandas._typing import NDFrameT

    from pandas import (
        DataFrame,
        DatetimeIndex,
        Index,
        PeriodIndex,
        Series,
    )

# ---------------------------------------------------------------------
# Plotting functions and monkey patches


def maybe_resample(series: Series, ax: Axes, kwargs: dict[str, Any]):
    # resample against axes freq if necessary

    if "how" in kwargs:
        raise ValueError(
            "'how' is not a valid keyword for plotting functions. If plotting "
            "multiple objects on shared axes, resample manually first."
        )

    freq, ax_freq = _get_freq(ax, series)

    if freq is None:  # pragma: no cover
        raise ValueError("Cannot use dynamic axis without frequency info")

    # Convert DatetimeIndex to PeriodIndex
    if isinstance(series.index, ABCDatetimeIndex):
        series = series.to_period(freq=freq)

    if ax_freq is not None and freq != ax_freq:
        if is_superperiod(freq, ax_freq):  # upsample input
            series = series.copy()
            # error: "Index" has no attribute "asfreq"
            series.index = series.index.asfreq(  # type: ignore[attr-defined]
                ax_freq, how="s"
            )
            freq = ax_freq
        elif _is_sup(freq, ax_freq):  # one is weekly
            # Resampling with PeriodDtype is deprecated, so we convert to
            #  DatetimeIndex, resample, then convert back.
            ser_ts = series.to_timestamp()
            ser_d = ser_ts.resample("D").last().dropna()
            ser_freq = ser_d.resample(ax_freq).last().dropna()
            series = ser_freq.to_period(ax_freq)
            freq = ax_freq
        elif is_subperiod(freq, ax_freq) or _is_sub(freq, ax_freq):
            _upsample_others(ax, freq, kwargs)
        else:  # pragma: no cover
            raise ValueError("Incompatible frequency conversion")
    return freq, series


def _is_sub(f1: str, f2: str) -> bool:
    return (f1.startswith("W") and is_subperiod("D", f2)) or (
        f2.startswith("W") and is_subperiod(f1, "D")
    )


def _is_sup(f1: str, f2: str) -> bool:
    return (f1.startswith("W") and is_superperiod("D", f2)) or (
        f2.startswith("W") and is_superperiod(f1, "D")
    )


def _upsample_others(ax: Axes, freq: BaseOffset, kwargs: dict[str, Any]) -> None:
    legend = ax.get_legend()
    lines, labels = _replot_ax(ax, freq)
    _replot_ax(ax, freq)

    other_ax = None
    if hasattr(ax, "left_ax"):
        other_ax = ax.left_ax
    if hasattr(ax, "right_ax"):
        other_ax = ax.right_ax

    if other_ax is not None:
        rlines, rlabels = _replot_ax(other_ax, freq)
        lines.extend(rlines)
        labels.extend(rlabels)

    if legend is not None and kwargs.get("legend", True) and len(lines) > 0:
        title: str | None = legend.get_title().get_text()
        if title == "None":
            title = None
        ax.legend(lines, labels, loc="best", title=title)


def _replot_ax(ax: Axes, freq: BaseOffset):
    data = getattr(ax, "_plot_data", None)

    # clear current axes and data
    # TODO #54485
    ax._plot_data = []  # type: ignore[attr-defined]
    ax.clear()

    decorate_axes(ax, freq)

    lines = []
    labels = []
    if data is not None:
        for series, plotf, kwds in data:
            series = series.copy()
            idx = series.index.asfreq(freq, how="S")
            series.index = idx
            # TODO #54485
            ax._plot_data.append((series, plotf, kwds))  # type: ignore[attr-defined]

            # for tsplot
            if isinstance(plotf, str):
                from pandas.plotting._matplotlib import PLOT_CLASSES

                plotf = PLOT_CLASSES[plotf]._plot

            lines.append(plotf(ax, series.index._mpl_repr(), series.values, **kwds)[0])
            labels.append(pprint_thing(series.name))

    return lines, labels


def decorate_axes(ax: Axes, freq: BaseOffset) -> None:
    """Initialize axes for time-series plotting"""
    if not hasattr(ax, "_plot_data"):
        # TODO #54485
        ax._plot_data = []  # type: ignore[attr-defined]

    # TODO #54485
    ax.freq = freq  # type: ignore[attr-defined]
    xaxis = ax.get_xaxis()
    # TODO #54485
    xaxis.freq = freq  # type: ignore[attr-defined]


def _get_ax_freq(ax: Axes):
    """
    Get the freq attribute of the ax object if set.
    Also checks shared axes (eg when using secondary yaxis, sharex=True
    or twinx)
    """
    ax_freq = getattr(ax, "freq", None)
    if ax_freq is None:
        # check for left/right ax in case of secondary yaxis
        if hasattr(ax, "left_ax"):
            ax_freq = getattr(ax.left_ax, "freq", None)
        elif hasattr(ax, "right_ax"):
            ax_freq = getattr(ax.right_ax, "freq", None)
    if ax_freq is None:
        # check if a shared ax (sharex/twinx) has already freq set
        shared_axes = ax.get_shared_x_axes().get_siblings(ax)
        if len(shared_axes) > 1:
            for shared_ax in shared_axes:
                ax_freq = getattr(shared_ax, "freq", None)
                if ax_freq is not None:
                    break
    return ax_freq


def _get_period_alias(freq: timedelta | BaseOffset | str) -> str | None:
    if isinstance(freq, BaseOffset):
        freqstr = freq.name
    else:
        freqstr = to_offset(freq, is_period=True).rule_code

    return get_period_alias(freqstr)


def _get_freq(ax: Axes, series: Series):
    # get frequency from data
    freq = getattr(series.index, "freq", None)
    if freq is None:
        freq = getattr(series.index, "inferred_freq", None)
        freq = to_offset(freq, is_period=True)

    ax_freq = _get_ax_freq(ax)

    # use axes freq if no data freq
    if freq is None:
        freq = ax_freq

    # get the period frequency
    freq = _get_period_alias(freq)
    return freq, ax_freq


def use_dynamic_x(ax: Axes, data: DataFrame | Series) -> bool:
    freq = _get_index_freq(data.index)
    ax_freq = _get_ax_freq(ax)

    if freq is None:  # convert irregular if axes has freq info
        freq = ax_freq
    # do not use tsplot if irregular was plotted first
    elif (ax_freq is None) and (len(ax.get_lines()) > 0):
        return False

    if freq is None:
        return False

    freq_str = _get_period_alias(freq)

    if freq_str is None:
        return False

    # FIXME: hack this for 0.10.1, creating more technical debt...sigh
    if isinstance(data.index, ABCDatetimeIndex):
        # error: "BaseOffset" has no attribute "_period_dtype_code"
        freq_str = OFFSET_TO_PERIOD_FREQSTR.get(freq_str, freq_str)
        base = to_offset(
            freq_str, is_period=True
        )._period_dtype_code  # type: ignore[attr-defined]
        x = data.index
        if base <= FreqGroup.FR_DAY.value:
            return x[:1].is_normalized
        period = Period(x[0], freq_str)
        assert isinstance(period, Period)
        return period.to_timestamp().tz_localize(x.tz) == x[0]
    return True


def _get_index_freq(index: Index) -> BaseOffset | None:
    freq = getattr(index, "freq", None)
    if freq is None:
        freq = getattr(index, "inferred_freq", None)
        if freq == "B":
            # error: "Index" has no attribute "dayofweek"
            weekdays = np.unique(index.dayofweek)  # type: ignore[attr-defined]
            if (5 in weekdays) or (6 in weekdays):
                freq = None

    freq = to_offset(freq)
    return freq


def maybe_convert_index(ax: Axes, data: NDFrameT) -> NDFrameT:
    # tsplot converts automatically, but don't want to convert index
    # over and over for DataFrames
    if isinstance(data.index, (ABCDatetimeIndex, ABCPeriodIndex)):
        freq: str | BaseOffset | None = data.index.freq

        if freq is None:
            # We only get here for DatetimeIndex
            data.index = cast("DatetimeIndex", data.index)
            freq = data.index.inferred_freq
            freq = to_offset(freq)

        if freq is None:
            freq = _get_ax_freq(ax)

        if freq is None:
            raise ValueError("Could not get frequency alias for plotting")

        freq_str = _get_period_alias(freq)

        with warnings.catch_warnings():
            # suppress Period[B] deprecation warning
            # TODO: need to find an alternative to this before the deprecation
            #  is enforced!
            warnings.filterwarnings(
                "ignore",
                r"PeriodDtype\[B\] is deprecated",
                category=FutureWarning,
            )

            if isinstance(data.index, ABCDatetimeIndex):
                data = data.tz_localize(None).to_period(freq=freq_str)
            elif isinstance(data.index, ABCPeriodIndex):
                data.index = data.index.asfreq(freq=freq_str)
    return data


# Patch methods for subplot.


def _format_coord(freq, t, y) -> str:
    time_period = Period(ordinal=int(t), freq=freq)
    return f"t = {time_period}  y = {y:8f}"


def format_dateaxis(
    subplot, freq: BaseOffset, index: DatetimeIndex | PeriodIndex
) -> None:
    """
    Pretty-formats the date axis (x-axis).

    Major and minor ticks are automatically set for the frequency of the
    current underlying series.  As the dynamic mode is activated by
    default, changing the limits of the x axis will intelligently change
    the positions of the ticks.
    """
    from matplotlib import pylab

    # handle index specific formatting
    # Note: DatetimeIndex does not use this
    # interface. DatetimeIndex uses matplotlib.date directly
    if isinstance(index, ABCPeriodIndex):
        majlocator = TimeSeries_DateLocator(
            freq, dynamic_mode=True, minor_locator=False, plot_obj=subplot
        )
        minlocator = TimeSeries_DateLocator(
            freq, dynamic_mode=True, minor_locator=True, plot_obj=subplot
        )
        subplot.xaxis.set_major_locator(majlocator)
        subplot.xaxis.set_minor_locator(minlocator)

        majformatter = TimeSeries_DateFormatter(
            freq, dynamic_mode=True, minor_locator=False, plot_obj=subplot
        )
        minformatter = TimeSeries_DateFormatter(
            freq, dynamic_mode=True, minor_locator=True, plot_obj=subplot
        )
        subplot.xaxis.set_major_formatter(majformatter)
        subplot.xaxis.set_minor_formatter(minformatter)

        # x and y coord info
        subplot.format_coord = functools.partial(_format_coord, freq)

    elif isinstance(index, ABCTimedeltaIndex):
        subplot.xaxis.set_major_formatter(TimeSeries_TimedeltaFormatter())
    else:
        raise TypeError("index type not supported")

    pylab.draw_if_interactive()