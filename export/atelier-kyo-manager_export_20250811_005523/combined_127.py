
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\gpt2\gpt2_tester.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
# This script helps evaluation of GPT-2 model.
import logging
import math
import os
import statistics
import timeit

import numpy
import torch
from benchmark_helper import Precision
from gpt2_helper import Gpt2Helper, Gpt2Inputs

logger = logging.getLogger(__name__)


class Gpt2Metric:
    def __init__(self, treatment_name, baseline_name="Torch", top_k=20):
        assert top_k > 1 and top_k <= 100
        self.baseline = baseline_name
        self.treatment = treatment_name
        self.name: str = f"{treatment_name} vs {baseline_name}"
        self.top_k = top_k
        self.top_1_error: int = 0
        self.top_k_error: int = 0
        self.total_samples: int = 0
        self.max_logits_diff: float = 0  # for non-empty past state
        self.max_logits_diff_no_past: float = 0  # for empty past state
        self.batch_top1_error: torch.FloatTensor = None  # top 1 error for current batch
        self.batch_topk_error: torch.FloatTensor = None  # top k error for current batch
        self.seq_len_latency = {}

    def print(self):
        if self.baseline != self.treatment:
            print("---")
            print(f"Metrics for {self.treatment} (baseline={self.baseline}):")
            if self.total_samples > 0:
                top_1_error_rate = 100.0 * self.top_1_error / self.total_samples
                top_k_error_rate = 100.0 * self.top_k_error / self.total_samples
                print(
                    f"Total={self.total_samples} Top1Error={self.top_1_error} ({top_1_error_rate:.2f}%) Top{self.top_k}Error={self.top_k_error} ({top_k_error_rate:.2f}%)"
                )
            print("Max logits diffs:")
            print(f"\twith past  = {self.max_logits_diff:.6f}")
            print(f"\tempty past = {self.max_logits_diff_no_past:.6f}")
        else:
            print(f"Metrics for {self.treatment} (baseline):")

        if self.seq_len_latency:
            print("Past sequence length range and average latency:")
            total = 0
            count = 0
            for key in sorted(self.seq_len_latency.keys()):
                average = statistics.mean(self.seq_len_latency[key]) * 1000.0
                if key == 0:
                    print(f"\t{key}:         \t{average:.2f} ms")
                else:
                    print(f"\t[{2**key}, {2 ** (key + 1) - 1}]:\t{average:.2f} ms")
                total += average * len(self.seq_len_latency[key])
                count += len(self.seq_len_latency[key])
            print(f"Average Latency: {total / count:.2f} ms")

    def diff_logits(self, baseline_logits, treatment_logits, is_empty_past: bool):
        diff = (baseline_logits - treatment_logits).abs().max()
        if is_empty_past:
            self.max_logits_diff_no_past = max(self.max_logits_diff_no_past, diff)
        else:
            self.max_logits_diff = max(self.max_logits_diff, diff)

        return diff

    def start_batch(self, batch_size: int):
        self.total_samples += batch_size
        self.batch_top1_error = torch.zeros((batch_size, 1), dtype=torch.bool)
        self.batch_topk_error = torch.zeros((batch_size, 1), dtype=torch.bool)

    def eval_batch(self, baseline, treatment, past_seq_len, verbose=True):
        self._eval_topk(baseline.top_1_tokens, treatment.top_1_tokens, 1, verbose)
        self._eval_topk(baseline.top_k_tokens, treatment.top_k_tokens, self.top_k, verbose)

        max_diff = self.diff_logits(baseline.logits, treatment.logits, past_seq_len == 0)
        if verbose:
            print(f"Max logits diffs of {self.name}: {max_diff}")

    def _eval_topk(self, baseline_topk, treatment_topk, top_k, verbose=True):
        if not torch.all(torch.eq(baseline_topk, treatment_topk)):
            if top_k == 1:
                if verbose:
                    print(f"Generated tokens not matched for {self.name}")
                self.batch_top1_error |= torch.eq(baseline_topk, treatment_topk).logical_not()
            else:
                if verbose:
                    print(
                        f"Top {top_k} tokens not matched for {self.name}. This will lead to wrong beam search results"
                    )
                self.batch_topk_error |= (
                    torch.eq(baseline_topk, treatment_topk).logical_not().sum(1).unsqueeze(dim=1) > 0
                )

    def end_batch(self):
        self.top_1_error += self.batch_top1_error.sum()
        self.top_k_error += self.batch_topk_error.sum()

    def add_latency(self, past_seq_len, latency):
        key = int(math.log2(past_seq_len)) + 1 if past_seq_len > 0 else 0
        if key not in self.seq_len_latency:
            self.seq_len_latency[key] = []
        self.seq_len_latency[key].append(latency)


class Gpt2Tester:
    def __init__(
        self,
        input_ids,
        position_ids,
        attention_mask,
        num_attention_heads,
        hidden_size,
        num_layer,
        device,
        is_fp16=False,
        top_k=20,
        top_k_required_order=False,
    ):
        self.batch_size = input_ids.shape[0]
        self.input_length = input_ids.shape[1]
        self.n_layer = num_layer

        self.input_ids = input_ids
        self.position_ids = position_ids
        self.attention_mask = attention_mask

        self.has_position_ids = position_ids is not None
        self.has_attention_mask = attention_mask is not None

        # Empty past state for first inference
        self.past = []
        past_shape = [
            2,
            self.batch_size,
            num_attention_heads,
            0,
            hidden_size // num_attention_heads,
        ]
        for _i in range(num_layer):
            empty_past = torch.empty(past_shape).type(torch.float16 if is_fp16 else torch.float32)
            self.past.append(empty_past.to(device))

        self.logits = None
        self.top_1_tokens = None
        self.top_k_tokens = None
        self.top_k = top_k
        self.top_k_required_order = top_k_required_order

    def get_inputs(self) -> Gpt2Inputs:
        return Gpt2Inputs(self.input_ids, self.position_ids, self.attention_mask, self.past)

    def save_test_data(self, session, output, save_test_data_dir, test_case_id):
        from onnx import numpy_helper

        path = os.path.join(save_test_data_dir, "test_data_set_" + str(test_case_id))
        if os.path.exists(path):
            print(f"Directory {path} existed. Skip saving test data")
            return

        os.makedirs(path, exist_ok=True)

        def add_tensor(input_tensors, torch_tensor, name):
            input_tensors.append(numpy_helper.from_array(torch_tensor.clone().cpu().numpy(), name))

        input_tensors = []
        add_tensor(input_tensors, self.input_ids, "input_ids")

        if self.has_position_ids:
            add_tensor(input_tensors, self.position_ids, "position_ids")

        if self.has_attention_mask:
            add_tensor(input_tensors, self.attention_mask, "attention_mask")

        for i in range(self.n_layer):
            add_tensor(input_tensors, self.past[i], "past_" + str(i))

        for i, tensor in enumerate(input_tensors):
            with open(os.path.join(path, f"input_{i}.pb"), "wb") as f:
                f.write(tensor.SerializeToString())

        output_names = [output.name for output in session.get_outputs()]
        for i, _name in enumerate(output_names):
            tensor = numpy_helper.from_array(
                output[i] if isinstance(output[i], numpy.ndarray) else output[i].clone().cpu().numpy()
            )
            with open(os.path.join(path, f"output_{i}.pb"), "wb") as f:
                f.write(tensor.SerializeToString())

        print(f"Test data saved to directory {path}")

    def update(self, output, step, device):
        """
        Update the inputs for next inference.
        """
        self.logits = (
            torch.from_numpy(output[0]) if isinstance(output[0], numpy.ndarray) else output[0].clone().detach().cpu()
        )

        self.top_1_tokens = Gpt2Tester.predict_next_token(self.logits)
        self.top_k_tokens = Gpt2Tester.predict_next_token(self.logits, self.top_k, self.top_k_required_order)

        self.input_ids = self.top_1_tokens.clone().detach().reshape([self.batch_size, 1]).to(device)

        if self.has_position_ids:
            self.position_ids = (
                torch.tensor([self.input_length + step - 1]).unsqueeze(0).repeat(self.batch_size, 1).to(device)
            )

        if self.has_attention_mask:
            self.attention_mask = torch.cat(
                [
                    self.attention_mask,
                    torch.ones([self.batch_size, 1]).type_as(self.attention_mask),
                ],
                1,
            ).to(device)

        self.past = []

        if isinstance(output[1], tuple):  # past in torch output is tuple
            self.past = list(output[1])
        else:
            for i in range(self.n_layer):
                past_i = (
                    torch.from_numpy(output[i + 1])
                    if isinstance(output[i + 1], numpy.ndarray)
                    else output[i + 1].clone().detach()
                )
                self.past.append(past_i.to(device))

    def diff(self, baseline):
        """
        Compare inputs and logits output.
        """

        print("start diff...")
        if self.logits is not None:
            max_io_diff = (self.logits - baseline.logits).abs().max()
            if max_io_diff > 1e-4:
                print(f"Max logits difference is too large: {max_io_diff}")

        if not torch.all(self.input_ids == baseline.input_ids):
            print("Input_ids is different", self.input_ids, baseline.input_ids)

        if self.has_position_ids:
            if not torch.all(self.position_ids == baseline.position_ids):
                print(
                    "position_ids is different",
                    self.position_ids,
                    baseline.position_ids,
                )

        if self.has_attention_mask:
            if not torch.all(self.attention_mask == baseline.attention_mask):
                print(
                    "attention_mask is different",
                    self.attention_mask,
                    baseline.attention_mask,
                )

        assert len(self.past) == len(baseline.past)

        for i, past_i in enumerate(self.past):
            assert past_i.shape == baseline.past[i].shape
            if past_i.nelement() > 0:
                max_past_diff = (past_i - baseline.past[i]).abs().max()
                if max_past_diff > 1e-4:
                    print(f"max_past_diff[{i}]={max_past_diff}")

    @staticmethod
    def predict_next_token(logits, top_k=1, required_order=False):
        """
        Get top k topkens based on logits.
        """

        # logits has shape (batch_size, seq_len, vocab_size)
        # last token logits has shape (batch_size, vocab_size)
        lastTokenLogits = logits[:, -1]  # noqa: N806
        if top_k == 1:
            generatedTokens = torch.argmax(lastTokenLogits, 1, True)  # noqa: N806
            return generatedTokens
        else:
            topk = torch.argsort(lastTokenLogits, -1, descending=True)[:, :top_k]
            if not required_order:
                sorted_topk, _ = topk.sort()
                return sorted_topk
            return topk

    @staticmethod
    def diff_present(onnx_output, onnx_io_output, n_layer):
        """
        Compare the present outputs of two outputs from ONNX Runtime.
        """
        present_diff_max = []
        for i in range(n_layer):
            onnx_present_i = (
                torch.from_numpy(onnx_output[i + 1])
                if isinstance(onnx_output[i + 1], numpy.ndarray)
                else onnx_output[i + 1]
            )
            onnx_io_present_i = (
                torch.from_numpy(onnx_io_output[i + 1])
                if isinstance(onnx_io_output[i + 1], numpy.ndarray)
                else onnx_io_output[i + 1]
            )
            max_diff = (onnx_present_i - onnx_io_present_i).abs().max()
            present_diff_max.append(max_diff)
        print(f"present_diff_max={present_diff_max}")

    @staticmethod
    def is_quantized_onnx_model(onnx_model_path):
        """
        Returns True if the ONNX model is quantized.
        """
        from onnx import load

        model = load(onnx_model_path)
        from onnxruntime.quantization.quantize import __producer__ as quantize_producer

        return model.producer_name == quantize_producer

    @staticmethod
    def test_generation(
        session,
        model,
        device,
        test_inputs,
        precision=Precision.FLOAT32,
        model_class="Gpt2LMHeadModel",
        top_k=20,
        top_k_no_order=True,
        max_steps=24,
        max_inputs=0,
        verbose=False,
        save_test_data=0,
        save_test_data_dir=".",
    ):
        """
        Test Generation using greedy beam search (without sampling) to compare PyTorch and ONNX model.
        It will print top 1 and top k errors on the given test inputs.
        """
        print(
            f"start test generation: (top_k={top_k} top_k_no_order={top_k_no_order} max_steps={max_steps} test_inputs={len(test_inputs)} max_inputs={max_inputs})"
        )
        n_layer = model.config.n_layer
        n_head = model.config.n_head
        n_embd = model.config.n_embd
        eos_token_id = model.config.eos_token_id
        test_data_saved = 0

        is_float16 = precision == Precision.FLOAT16
        if is_float16:
            assert "float16" in session.get_outputs()[0].type

        # We will still use fp32 torch model as baseline when onnx model if fp16
        model.eval().to(device)

        # Allocate initial buffers for IO Binding of ONNX Runtimne. The buffer size will automatically increase later.
        init_output_shapes = Gpt2Helper.get_output_shapes(
            batch_size=4,
            past_sequence_length=128,
            sequence_length=32,
            config=model.config,
            model_class=model_class,
        )
        output_buffers = Gpt2Helper.get_output_buffers(init_output_shapes, device, is_float16=is_float16)

        baseline_name = "Torch"
        treatment_name = "Quantized Onnx" if precision == Precision.INT8 else "Onnx"
        torch_metric = Gpt2Metric(baseline_name, baseline_name, top_k)
        onnx_metric = Gpt2Metric(treatment_name, baseline_name, top_k)
        onnx_io_metric = Gpt2Metric(treatment_name + " with IO Binding", baseline_name, top_k)

        for i, inputs in enumerate(test_inputs):
            if max_inputs > 0 and i == max_inputs:
                break
            if i % 10 == 0:
                print(f"{i}")
            input_ids = inputs["input_ids"]
            position_ids = inputs.get("position_ids", None)
            attention_mask = inputs.get("attention_mask", None)

            onnx_runner = Gpt2Tester(
                input_ids,
                position_ids,
                attention_mask,
                n_head,
                n_embd,
                n_layer,
                device,
                is_float16,
                top_k,
                not top_k_no_order,
            )
            onnx_io_runner = Gpt2Tester(
                input_ids,
                position_ids,
                attention_mask,
                n_head,
                n_embd,
                n_layer,
                device,
                is_float16,
                top_k,
                not top_k_no_order,
            )
            torch_runner = Gpt2Tester(
                input_ids,
                position_ids,
                attention_mask,
                n_head,
                n_embd,
                n_layer,
                device,
                False,
                top_k,
                not top_k_no_order,
            )  # Torch model baseline is fp32

            batch_size = torch_runner.batch_size
            onnx_metric.start_batch(batch_size)
            onnx_io_metric.start_batch(batch_size)

            with torch.no_grad():
                done = torch.zeros(batch_size, dtype=torch.bool)
                for step in range(max_steps):
                    seq_len = list(onnx_runner.input_ids.size())[1]
                    past_seq_len = list(onnx_runner.past[0].size())[3]

                    start_time = timeit.default_timer()
                    pytorch_output = Gpt2Helper.pytorch_inference(model, torch_runner.get_inputs())
                    torch_metric.add_latency(past_seq_len, timeit.default_timer() - start_time)
                    torch_runner.update(pytorch_output, step, device)

                    onnx_output, avg_latency_ms = Gpt2Helper.onnxruntime_inference(
                        session, onnx_runner.get_inputs(), total_runs=1
                    )
                    onnx_metric.add_latency(past_seq_len, avg_latency_ms / 1000.0)
                    onnx_runner.update(onnx_output, step, device)

                    output_shapes = Gpt2Helper.get_output_shapes(
                        batch_size,
                        past_seq_len,
                        seq_len,
                        model.config,
                        model_class=model_class,
                    )
                    Gpt2Helper.auto_increase_buffer_size(output_buffers, output_shapes)

                    (
                        onnx_io_output,
                        avg_latency_ms,
                    ) = Gpt2Helper.onnxruntime_inference_with_binded_io(
                        session,
                        onnx_io_runner.get_inputs(),
                        output_buffers,
                        output_shapes,
                        total_runs=1,
                        return_numpy=False,
                        include_copy_output_latency=True,
                    )
                    onnx_io_metric.add_latency(past_seq_len, avg_latency_ms / 1000.0)

                    if test_data_saved < save_test_data:
                        onnx_io_runner.save_test_data(session, onnx_io_output, save_test_data_dir, test_data_saved)
                        test_data_saved += 1

                    onnx_io_runner.update(onnx_io_output, step, device)

                    if verbose:
                        onnx_runner.diff(onnx_io_runner)
                        Gpt2Tester.diff_present(onnx_output, onnx_io_output, n_layer)

                        print("Top 1 tokens:")
                        print("\tTorch", torch_runner.top_1_tokens)
                        print("\tONNX", onnx_runner.top_1_tokens)
                        print("\tONNX with IO binding", onnx_io_runner.top_1_tokens)

                    onnx_metric.eval_batch(torch_runner, onnx_runner, past_seq_len, verbose=verbose)
                    onnx_io_metric.eval_batch(torch_runner, onnx_io_runner, past_seq_len, verbose=verbose)

                    done = done | (torch_runner.top_1_tokens == eos_token_id).any()
                    if torch.all(done):
                        break

            onnx_metric.end_batch()
            onnx_io_metric.end_batch()

        torch_metric.print()
        onnx_metric.print()
        onnx_io_metric.print()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\expressions\matmul.py ===
from sympy.assumptions.ask import ask, Q
from sympy.assumptions.refine import handlers_dict
from sympy.core import Basic, sympify, S
from sympy.core.mul import mul, Mul
from sympy.core.numbers import Number, Integer
from sympy.core.symbol import Dummy
from sympy.functions import adjoint
from sympy.strategies import (rm_id, unpack, typed, flatten, exhaust,
        do_one, new)
from sympy.matrices.exceptions import NonInvertibleMatrixError
from sympy.matrices.matrixbase import MatrixBase
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.matrices.expressions._shape import validate_matmul_integer as validate

from .inverse import Inverse
from .matexpr import MatrixExpr
from .matpow import MatPow
from .transpose import transpose
from .permutation import PermutationMatrix
from .special import ZeroMatrix, Identity, GenericIdentity, OneMatrix


# XXX: MatMul should perhaps not subclass directly from Mul
class MatMul(MatrixExpr, Mul):
    """
    A product of matrix expressions

    Examples
    ========

    >>> from sympy import MatMul, MatrixSymbol
    >>> A = MatrixSymbol('A', 5, 4)
    >>> B = MatrixSymbol('B', 4, 3)
    >>> C = MatrixSymbol('C', 3, 6)
    >>> MatMul(A, B, C)
    A*B*C
    """
    is_MatMul = True

    identity = GenericIdentity()

    def __new__(cls, *args, evaluate=False, check=None, _sympify=True):
        if not args:
            return cls.identity

        # This must be removed aggressively in the constructor to avoid
        # TypeErrors from GenericIdentity().shape
        args = list(filter(lambda i: cls.identity != i, args))
        if _sympify:
            args = list(map(sympify, args))
        obj = Basic.__new__(cls, *args)
        factor, matrices = obj.as_coeff_matrices()

        if check is not None:
            sympy_deprecation_warning(
                "Passing check to MatMul is deprecated and the check argument will be removed in a future version.",
                deprecated_since_version="1.11",
                active_deprecations_target='remove-check-argument-from-matrix-operations')

        if check is not False:
            validate(*matrices)

        if not matrices:
            # Should it be
            #
            # return Basic.__neq__(cls, factor, GenericIdentity()) ?
            return factor

        if evaluate:
            return cls._evaluate(obj)

        return obj

    @classmethod
    def _evaluate(cls, expr):
        return canonicalize(expr)

    @property
    def shape(self):
        matrices = [arg for arg in self.args if arg.is_Matrix]
        return (matrices[0].rows, matrices[-1].cols)

    def _entry(self, i, j, expand=True, **kwargs):
        # Avoid cyclic imports
        from sympy.concrete.summations import Sum
        from sympy.matrices.immutable import ImmutableMatrix

        coeff, matrices = self.as_coeff_matrices()

        if len(matrices) == 1:  # situation like 2*X, matmul is just X
            return coeff * matrices[0][i, j]

        indices = [None]*(len(matrices) + 1)
        ind_ranges = [None]*(len(matrices) - 1)
        indices[0] = i
        indices[-1] = j

        def f():
            counter = 1
            while True:
                yield Dummy("i_%i" % counter)
                counter += 1

        dummy_generator = kwargs.get("dummy_generator", f())

        for i in range(1, len(matrices)):
            indices[i] = next(dummy_generator)

        for i, arg in enumerate(matrices[:-1]):
            ind_ranges[i] = arg.shape[1] - 1
        matrices = [arg._entry(indices[i], indices[i+1], dummy_generator=dummy_generator) for i, arg in enumerate(matrices)]
        expr_in_sum = Mul.fromiter(matrices)
        if any(v.has(ImmutableMatrix) for v in matrices):
            expand = True
        result = coeff*Sum(
                expr_in_sum,
                *zip(indices[1:-1], [0]*len(ind_ranges), ind_ranges)
            )

        # Don't waste time in result.doit() if the sum bounds are symbolic
        if not any(isinstance(v, (Integer, int)) for v in ind_ranges):
            expand = False
        return result.doit() if expand else result

    def as_coeff_matrices(self):
        scalars = [x for x in self.args if not x.is_Matrix]
        matrices = [x for x in self.args if x.is_Matrix]
        coeff = Mul(*scalars)
        if coeff.is_commutative is False:
            raise NotImplementedError("noncommutative scalars in MatMul are not supported.")

        return coeff, matrices

    def as_coeff_mmul(self):
        coeff, matrices = self.as_coeff_matrices()
        return coeff, MatMul(*matrices)

    def expand(self, **kwargs):
        expanded = super(MatMul, self).expand(**kwargs)
        return self._evaluate(expanded)

    def _eval_transpose(self):
        """Transposition of matrix multiplication.

        Notes
        =====

        The following rules are applied.

        Transposition for matrix multiplied with another matrix:
        `\\left(A B\\right)^{T} = B^{T} A^{T}`

        Transposition for matrix multiplied with scalar:
        `\\left(c A\\right)^{T} = c A^{T}`

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Transpose
        """
        coeff, matrices = self.as_coeff_matrices()
        return MatMul(
            coeff, *[transpose(arg) for arg in matrices[::-1]]).doit()

    def _eval_adjoint(self):
        return MatMul(*[adjoint(arg) for arg in self.args[::-1]]).doit()

    def _eval_trace(self):
        factor, mmul = self.as_coeff_mmul()
        if factor != 1:
            from .trace import trace
            return factor * trace(mmul.doit())

    def _eval_determinant(self):
        from sympy.matrices.expressions.determinant import Determinant
        factor, matrices = self.as_coeff_matrices()
        square_matrices = only_squares(*matrices)
        return factor**self.rows * Mul(*list(map(Determinant, square_matrices)))

    def _eval_inverse(self):
        if all(arg.is_square for arg in self.args if isinstance(arg, MatrixExpr)):
            return MatMul(*(
                arg.inverse() if isinstance(arg, MatrixExpr) else arg**-1
                    for arg in self.args[::-1]
                )
            ).doit()
        return Inverse(self)

    def doit(self, **hints):
        deep = hints.get('deep', True)
        if deep:
            args = tuple(arg.doit(**hints) for arg in self.args)
        else:
            args = self.args

        # treat scalar*MatrixSymbol or scalar*MatPow separately
        expr = canonicalize(MatMul(*args))
        return expr

    # Needed for partial compatibility with Mul
    def args_cnc(self, cset=False, warn=True, **kwargs):
        coeff_c = [x for x in self.args if x.is_commutative]
        coeff_nc = [x for x in self.args if not x.is_commutative]
        if cset:
            clen = len(coeff_c)
            coeff_c = set(coeff_c)
            if clen and warn and len(coeff_c) != clen:
                raise ValueError('repeated commutative arguments: %s' %
                                 [ci for ci in coeff_c if list(self.args).count(ci) > 1])
        return [coeff_c, coeff_nc]

    def _eval_derivative_matrix_lines(self, x):
        from .transpose import Transpose
        with_x_ind = [i for i, arg in enumerate(self.args) if arg.has(x)]
        lines = []
        for ind in with_x_ind:
            left_args = self.args[:ind]
            right_args = self.args[ind+1:]

            if right_args:
                right_mat = MatMul.fromiter(right_args)
            else:
                right_mat = Identity(self.shape[1])
            if left_args:
                left_rev = MatMul.fromiter([Transpose(i).doit() if i.is_Matrix else i for i in reversed(left_args)])
            else:
                left_rev = Identity(self.shape[0])

            d = self.args[ind]._eval_derivative_matrix_lines(x)
            for i in d:
                i.append_first(left_rev)
                i.append_second(right_mat)
                lines.append(i)

        return lines

mul.register_handlerclass((Mul, MatMul), MatMul)


# Rules
def newmul(*args):
    if args[0] == 1:
        args = args[1:]
    return new(MatMul, *args)

def any_zeros(mul):
    if any(arg.is_zero or (arg.is_Matrix and arg.is_ZeroMatrix)
                       for arg in mul.args):
        matrices = [arg for arg in mul.args if arg.is_Matrix]
        return ZeroMatrix(matrices[0].rows, matrices[-1].cols)
    return mul

def merge_explicit(matmul):
    """ Merge explicit MatrixBase arguments

    >>> from sympy import MatrixSymbol, Matrix, MatMul, pprint
    >>> from sympy.matrices.expressions.matmul import merge_explicit
    >>> A = MatrixSymbol('A', 2, 2)
    >>> B = Matrix([[1, 1], [1, 1]])
    >>> C = Matrix([[1, 2], [3, 4]])
    >>> X = MatMul(A, B, C)
    >>> pprint(X)
      [1  1] [1  2]
    A*[    ]*[    ]
      [1  1] [3  4]
    >>> pprint(merge_explicit(X))
      [4  6]
    A*[    ]
      [4  6]

    >>> X = MatMul(B, A, C)
    >>> pprint(X)
    [1  1]   [1  2]
    [    ]*A*[    ]
    [1  1]   [3  4]
    >>> pprint(merge_explicit(X))
    [1  1]   [1  2]
    [    ]*A*[    ]
    [1  1]   [3  4]
    """
    if not any(isinstance(arg, MatrixBase) for arg in matmul.args):
        return matmul
    newargs = []
    last = matmul.args[0]
    for arg in matmul.args[1:]:
        if isinstance(arg, (MatrixBase, Number)) and isinstance(last, (MatrixBase, Number)):
            last = last * arg
        else:
            newargs.append(last)
            last = arg
    newargs.append(last)

    return MatMul(*newargs)

def remove_ids(mul):
    """ Remove Identities from a MatMul

    This is a modified version of sympy.strategies.rm_id.
    This is necessary because MatMul may contain both MatrixExprs and Exprs
    as args.

    See Also
    ========

    sympy.strategies.rm_id
    """
    # Separate Exprs from MatrixExprs in args
    factor, mmul = mul.as_coeff_mmul()
    # Apply standard rm_id for MatMuls
    result = rm_id(lambda x: x.is_Identity is True)(mmul)
    if result != mmul:
        return newmul(factor, *result.args)  # Recombine and return
    else:
        return mul

def factor_in_front(mul):
    factor, matrices = mul.as_coeff_matrices()
    if factor != 1:
        return newmul(factor, *matrices)
    return mul

def combine_powers(mul):
    r"""Combine consecutive powers with the same base into one, e.g.
    $$A \times A^2 \Rightarrow A^3$$

    This also cancels out the possible matrix inverses using the
    knowledgebase of :class:`~.Inverse`, e.g.,
    $$ Y \times X \times X^{-1} \Rightarrow Y $$
    """
    factor, args = mul.as_coeff_matrices()
    new_args = [args[0]]

    for i in range(1, len(args)):
        A = new_args[-1]
        B = args[i]

        if isinstance(B, Inverse) and isinstance(B.arg, MatMul):
            Bargs = B.arg.args
            l = len(Bargs)
            if list(Bargs) == new_args[-l:]:
                new_args = new_args[:-l] + [Identity(B.shape[0])]
                continue

        if isinstance(A, Inverse) and isinstance(A.arg, MatMul):
            Aargs = A.arg.args
            l = len(Aargs)
            if list(Aargs) == args[i:i+l]:
                identity = Identity(A.shape[0])
                new_args[-1] = identity
                for j in range(i, i+l):
                    args[j] = identity
                continue

        if A.is_square == False or B.is_square == False:
            new_args.append(B)
            continue

        if isinstance(A, MatPow):
            A_base, A_exp = A.args
        else:
            A_base, A_exp = A, S.One

        if isinstance(B, MatPow):
            B_base, B_exp = B.args
        else:
            B_base, B_exp = B, S.One

        if A_base == B_base:
            new_exp = A_exp + B_exp
            new_args[-1] = MatPow(A_base, new_exp).doit(deep=False)
            continue
        elif not isinstance(B_base, MatrixBase):
            try:
                B_base_inv = B_base.inverse()
            except NonInvertibleMatrixError:
                B_base_inv = None
            if B_base_inv is not None and A_base == B_base_inv:
                new_exp = A_exp - B_exp
                new_args[-1] = MatPow(A_base, new_exp).doit(deep=False)
                continue
        new_args.append(B)

    return newmul(factor, *new_args)

def combine_permutations(mul):
    """Refine products of permutation matrices as the products of cycles.
    """
    args = mul.args
    l = len(args)
    if l < 2:
        return mul

    result = [args[0]]
    for i in range(1, l):
        A = result[-1]
        B = args[i]
        if isinstance(A, PermutationMatrix) and \
            isinstance(B, PermutationMatrix):
            cycle_1 = A.args[0]
            cycle_2 = B.args[0]
            result[-1] = PermutationMatrix(cycle_1 * cycle_2)
        else:
            result.append(B)

    return MatMul(*result)

def combine_one_matrices(mul):
    """
    Combine products of OneMatrix

    e.g. OneMatrix(2, 3) * OneMatrix(3, 4) -> 3 * OneMatrix(2, 4)
    """
    factor, args = mul.as_coeff_matrices()
    new_args = [args[0]]

    for B in args[1:]:
        A = new_args[-1]
        if not isinstance(A, OneMatrix) or not isinstance(B, OneMatrix):
            new_args.append(B)
            continue
        new_args.pop()
        new_args.append(OneMatrix(A.shape[0], B.shape[1]))
        factor *= A.shape[1]

    return newmul(factor, *new_args)

def distribute_monom(mul):
    """
    Simplify MatMul expressions but distributing
    rational term to MatMul.

    e.g. 2*(A+B) -> 2*A + 2*B
    """
    args = mul.args
    if len(args) == 2:
        from .matadd import MatAdd
        if args[0].is_MatAdd and args[1].is_Rational:
            return MatAdd(*[MatMul(mat, args[1]).doit() for mat in args[0].args])
        if args[1].is_MatAdd and args[0].is_Rational:
            return MatAdd(*[MatMul(args[0], mat).doit() for mat in args[1].args])
    return mul

rules = (
    distribute_monom, any_zeros, remove_ids, combine_one_matrices, combine_powers, unpack, rm_id(lambda x: x == 1),
    merge_explicit, factor_in_front, flatten, combine_permutations)

canonicalize = exhaust(typed({MatMul: do_one(*rules)}))

def only_squares(*matrices):
    """factor matrices only if they are square"""
    if matrices[0].rows != matrices[-1].cols:
        raise RuntimeError("Invalid matrices being multiplied")
    out = []
    start = 0
    for i, M in enumerate(matrices):
        if M.cols == matrices[start].rows:
            out.append(MatMul(*matrices[start:i+1]).doit())
            start = i+1
    return out


def refine_MatMul(expr, assumptions):
    """
    >>> from sympy import MatrixSymbol, Q, assuming, refine
    >>> X = MatrixSymbol('X', 2, 2)
    >>> expr = X * X.T
    >>> print(expr)
    X*X.T
    >>> with assuming(Q.orthogonal(X)):
    ...     print(refine(expr))
    I
    """
    newargs = []
    exprargs = []

    for args in expr.args:
        if args.is_Matrix:
            exprargs.append(args)
        else:
            newargs.append(args)

    last = exprargs[0]
    for arg in exprargs[1:]:
        if arg == last.T and ask(Q.orthogonal(arg), assumptions):
            last = Identity(arg.shape[0])
        elif arg == last.conjugate() and ask(Q.unitary(arg), assumptions):
            last = Identity(arg.shape[0])
        else:
            newargs.append(last)
            last = arg
    newargs.append(last)

    return MatMul(*newargs)


handlers_dict['MatMul'] = refine_MatMul

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\partfrac.py ===
"""Algorithms for partial fraction decomposition of rational functions. """


from sympy.core import S, Add, sympify, Function, Lambda, Dummy
from sympy.core.traversal import preorder_traversal
from sympy.polys import Poly, RootSum, cancel, factor
from sympy.polys.polyerrors import PolynomialError
from sympy.polys.polyoptions import allowed_flags, set_defaults
from sympy.polys.polytools import parallel_poly_from_expr
from sympy.utilities import numbered_symbols, take, xthreaded, public


@xthreaded
@public
def apart(f, x=None, full=False, **options):
    """
    Compute partial fraction decomposition of a rational function.

    Given a rational function ``f``, computes the partial fraction
    decomposition of ``f``. Two algorithms are available: One is based on the
    undetermined coefficients method, the other is Bronstein's full partial
    fraction decomposition algorithm.

    The undetermined coefficients method (selected by ``full=False``) uses
    polynomial factorization (and therefore accepts the same options as
    factor) for the denominator. Per default it works over the rational
    numbers, therefore decomposition of denominators with non-rational roots
    (e.g. irrational, complex roots) is not supported by default (see options
    of factor).

    Bronstein's algorithm can be selected by using ``full=True`` and allows a
    decomposition of denominators with non-rational roots. A human-readable
    result can be obtained via ``doit()`` (see examples below).

    Examples
    ========

    >>> from sympy.polys.partfrac import apart
    >>> from sympy.abc import x, y

    By default, using the undetermined coefficients method:

    >>> apart(y/(x + 2)/(x + 1), x)
    -y/(x + 2) + y/(x + 1)

    The undetermined coefficients method does not provide a result when the
    denominators roots are not rational:

    >>> apart(y/(x**2 + x + 1), x)
    y/(x**2 + x + 1)

    You can choose Bronstein's algorithm by setting ``full=True``:

    >>> apart(y/(x**2 + x + 1), x, full=True)
    RootSum(_w**2 + _w + 1, Lambda(_a, (-2*_a*y/3 - y/3)/(-_a + x)))

    Calling ``doit()`` yields a human-readable result:

    >>> apart(y/(x**2 + x + 1), x, full=True).doit()
    (-y/3 - 2*y*(-1/2 - sqrt(3)*I/2)/3)/(x + 1/2 + sqrt(3)*I/2) + (-y/3 -
        2*y*(-1/2 + sqrt(3)*I/2)/3)/(x + 1/2 - sqrt(3)*I/2)


    See Also
    ========

    apart_list, assemble_partfrac_list
    """
    allowed_flags(options, [])

    f = sympify(f)

    if f.is_Atom:
        return f
    else:
        P, Q = f.as_numer_denom()

    _options = options.copy()
    options = set_defaults(options, extension=True)
    try:
        (P, Q), opt = parallel_poly_from_expr((P, Q), x, **options)
    except PolynomialError as msg:
        if f.is_commutative:
            raise PolynomialError(msg)
        # non-commutative
        if f.is_Mul:
            c, nc = f.args_cnc(split_1=False)
            nc = f.func(*nc)
            if c:
                c = apart(f.func._from_args(c), x=x, full=full, **_options)
                return c*nc
            else:
                return nc
        elif f.is_Add:
            c = []
            nc = []
            for i in f.args:
                if i.is_commutative:
                    c.append(i)
                else:
                    try:
                        nc.append(apart(i, x=x, full=full, **_options))
                    except NotImplementedError:
                        nc.append(i)
            return apart(f.func(*c), x=x, full=full, **_options) + f.func(*nc)
        else:
            reps = []
            pot = preorder_traversal(f)
            next(pot)
            for e in pot:
                try:
                    reps.append((e, apart(e, x=x, full=full, **_options)))
                    pot.skip()  # this was handled successfully
                except NotImplementedError:
                    pass
            return f.xreplace(dict(reps))

    if P.is_multivariate:
        fc = f.cancel()
        if fc != f:
            return apart(fc, x=x, full=full, **_options)

        raise NotImplementedError(
            "multivariate partial fraction decomposition")

    common, P, Q = P.cancel(Q)

    poly, P = P.div(Q, auto=True)
    P, Q = P.rat_clear_denoms(Q)

    if Q.degree() <= 1:
        partial = P/Q
    else:
        if not full:
            partial = apart_undetermined_coeffs(P, Q)
        else:
            partial = apart_full_decomposition(P, Q)

    terms = S.Zero

    for term in Add.make_args(partial):
        if term.has(RootSum):
            terms += term
        else:
            terms += factor(term)

    return common*(poly.as_expr() + terms)


def apart_undetermined_coeffs(P, Q):
    """Partial fractions via method of undetermined coefficients. """
    X = numbered_symbols(cls=Dummy)
    partial, symbols = [], []

    _, factors = Q.factor_list()

    for f, k in factors:
        n, q = f.degree(), Q

        for i in range(1, k + 1):
            coeffs, q = take(X, n), q.quo(f)
            partial.append((coeffs, q, f, i))
            symbols.extend(coeffs)

    dom = Q.get_domain().inject(*symbols)
    F = Poly(0, Q.gen, domain=dom)

    for i, (coeffs, q, f, k) in enumerate(partial):
        h = Poly(coeffs, Q.gen, domain=dom)
        partial[i] = (h, f, k)
        q = q.set_domain(dom)
        F += h*q

    system, result = [], S.Zero

    for (k,), coeff in F.terms():
        system.append(coeff - P.nth(k))

    from sympy.solvers import solve
    solution = solve(system, symbols)

    for h, f, k in partial:
        h = h.as_expr().subs(solution)
        result += h/f.as_expr()**k

    return result


def apart_full_decomposition(P, Q):
    """
    Bronstein's full partial fraction decomposition algorithm.

    Given a univariate rational function ``f``, performing only GCD
    operations over the algebraic closure of the initial ground domain
    of definition, compute full partial fraction decomposition with
    fractions having linear denominators.

    Note that no factorization of the initial denominator of ``f`` is
    performed. The final decomposition is formed in terms of a sum of
    :class:`RootSum` instances.

    References
    ==========

    .. [1] [Bronstein93]_

    """
    return assemble_partfrac_list(apart_list(P/Q, P.gens[0]))


@public
def apart_list(f, x=None, dummies=None, **options):
    """
    Compute partial fraction decomposition of a rational function
    and return the result in structured form.

    Given a rational function ``f`` compute the partial fraction decomposition
    of ``f``. Only Bronstein's full partial fraction decomposition algorithm
    is supported by this method. The return value is highly structured and
    perfectly suited for further algorithmic treatment rather than being
    human-readable. The function returns a tuple holding three elements:

    * The first item is the common coefficient, free of the variable `x` used
      for decomposition. (It is an element of the base field `K`.)

    * The second item is the polynomial part of the decomposition. This can be
      the zero polynomial. (It is an element of `K[x]`.)

    * The third part itself is a list of quadruples. Each quadruple
      has the following elements in this order:

      - The (not necessarily irreducible) polynomial `D` whose roots `w_i` appear
        in the linear denominator of a bunch of related fraction terms. (This item
        can also be a list of explicit roots. However, at the moment ``apart_list``
        never returns a result this way, but the related ``assemble_partfrac_list``
        function accepts this format as input.)

      - The numerator of the fraction, written as a function of the root `w`

      - The linear denominator of the fraction *excluding its power exponent*,
        written as a function of the root `w`.

      - The power to which the denominator has to be raised.

    On can always rebuild a plain expression by using the function ``assemble_partfrac_list``.

    Examples
    ========

    A first example:

    >>> from sympy.polys.partfrac import apart_list, assemble_partfrac_list
    >>> from sympy.abc import x, t

    >>> f = (2*x**3 - 2*x) / (x**2 - 2*x + 1)
    >>> pfd = apart_list(f)
    >>> pfd
    (1,
    Poly(2*x + 4, x, domain='ZZ'),
    [(Poly(_w - 1, _w, domain='ZZ'), Lambda(_a, 4), Lambda(_a, -_a + x), 1)])

    >>> assemble_partfrac_list(pfd)
    2*x + 4 + 4/(x - 1)

    Second example:

    >>> f = (-2*x - 2*x**2) / (3*x**2 - 6*x)
    >>> pfd = apart_list(f)
    >>> pfd
    (-1,
    Poly(2/3, x, domain='QQ'),
    [(Poly(_w - 2, _w, domain='ZZ'), Lambda(_a, 2), Lambda(_a, -_a + x), 1)])

    >>> assemble_partfrac_list(pfd)
    -2/3 - 2/(x - 2)

    Another example, showing symbolic parameters:

    >>> pfd = apart_list(t/(x**2 + x + t), x)
    >>> pfd
    (1,
    Poly(0, x, domain='ZZ[t]'),
    [(Poly(_w**2 + _w + t, _w, domain='ZZ[t]'),
    Lambda(_a, -2*_a*t/(4*t - 1) - t/(4*t - 1)),
    Lambda(_a, -_a + x),
    1)])

    >>> assemble_partfrac_list(pfd)
    RootSum(_w**2 + _w + t, Lambda(_a, (-2*_a*t/(4*t - 1) - t/(4*t - 1))/(-_a + x)))

    This example is taken from Bronstein's original paper:

    >>> f = 36 / (x**5 - 2*x**4 - 2*x**3 + 4*x**2 + x - 2)
    >>> pfd = apart_list(f)
    >>> pfd
    (1,
    Poly(0, x, domain='ZZ'),
    [(Poly(_w - 2, _w, domain='ZZ'), Lambda(_a, 4), Lambda(_a, -_a + x), 1),
    (Poly(_w**2 - 1, _w, domain='ZZ'), Lambda(_a, -3*_a - 6), Lambda(_a, -_a + x), 2),
    (Poly(_w + 1, _w, domain='ZZ'), Lambda(_a, -4), Lambda(_a, -_a + x), 1)])

    >>> assemble_partfrac_list(pfd)
    -4/(x + 1) - 3/(x + 1)**2 - 9/(x - 1)**2 + 4/(x - 2)

    See also
    ========

    apart, assemble_partfrac_list

    References
    ==========

    .. [1] [Bronstein93]_

    """
    allowed_flags(options, [])

    f = sympify(f)

    if f.is_Atom:
        return f
    else:
        P, Q = f.as_numer_denom()

    options = set_defaults(options, extension=True)
    (P, Q), opt = parallel_poly_from_expr((P, Q), x, **options)

    if P.is_multivariate:
        raise NotImplementedError(
            "multivariate partial fraction decomposition")

    common, P, Q = P.cancel(Q)

    poly, P = P.div(Q, auto=True)
    P, Q = P.rat_clear_denoms(Q)

    polypart = poly

    if dummies is None:
        def dummies(name):
            d = Dummy(name)
            while True:
                yield d

        dummies = dummies("w")

    rationalpart = apart_list_full_decomposition(P, Q, dummies)

    return (common, polypart, rationalpart)


def apart_list_full_decomposition(P, Q, dummygen):
    """
    Bronstein's full partial fraction decomposition algorithm.

    Given a univariate rational function ``f``, performing only GCD
    operations over the algebraic closure of the initial ground domain
    of definition, compute full partial fraction decomposition with
    fractions having linear denominators.

    Note that no factorization of the initial denominator of ``f`` is
    performed. The final decomposition is formed in terms of a sum of
    :class:`RootSum` instances.

    References
    ==========

    .. [1] [Bronstein93]_

    """
    P_orig, Q_orig, x, U = P, Q, P.gen, []

    u = Function('u')(x)
    a = Dummy('a')

    partial = []

    for d, n in Q.sqf_list_include(all=True):
        b = d.as_expr()
        U += [ u.diff(x, n - 1) ]

        h = cancel(P_orig/Q_orig.quo(d**n)) / u**n

        H, subs = [h], []

        for j in range(1, n):
            H += [ H[-1].diff(x) / j ]

        for j in range(1, n + 1):
            subs += [ (U[j - 1], b.diff(x, j) / j) ]

        for j in range(0, n):
            P, Q = cancel(H[j]).as_numer_denom()

            for i in range(0, j + 1):
                P = P.subs(*subs[j - i])

            Q = Q.subs(*subs[0])

            P = Poly(P, x)
            Q = Poly(Q, x)

            G = P.gcd(d)
            D = d.quo(G)

            B, g = Q.half_gcdex(D)
            b = (P * B.quo(g)).rem(D)

            Dw = D.subs(x, next(dummygen))
            numer = Lambda(a, b.as_expr().subs(x, a))
            denom = Lambda(a, (x - a))
            exponent = n-j

            partial.append((Dw, numer, denom, exponent))

    return partial


@public
def assemble_partfrac_list(partial_list):
    r"""Reassemble a full partial fraction decomposition
    from a structured result obtained by the function ``apart_list``.

    Examples
    ========

    This example is taken from Bronstein's original paper:

    >>> from sympy.polys.partfrac import apart_list, assemble_partfrac_list
    >>> from sympy.abc import x

    >>> f = 36 / (x**5 - 2*x**4 - 2*x**3 + 4*x**2 + x - 2)
    >>> pfd = apart_list(f)
    >>> pfd
    (1,
    Poly(0, x, domain='ZZ'),
    [(Poly(_w - 2, _w, domain='ZZ'), Lambda(_a, 4), Lambda(_a, -_a + x), 1),
    (Poly(_w**2 - 1, _w, domain='ZZ'), Lambda(_a, -3*_a - 6), Lambda(_a, -_a + x), 2),
    (Poly(_w + 1, _w, domain='ZZ'), Lambda(_a, -4), Lambda(_a, -_a + x), 1)])

    >>> assemble_partfrac_list(pfd)
    -4/(x + 1) - 3/(x + 1)**2 - 9/(x - 1)**2 + 4/(x - 2)

    If we happen to know some roots we can provide them easily inside the structure:

    >>> pfd = apart_list(2/(x**2-2))
    >>> pfd
    (1,
    Poly(0, x, domain='ZZ'),
    [(Poly(_w**2 - 2, _w, domain='ZZ'),
    Lambda(_a, _a/2),
    Lambda(_a, -_a + x),
    1)])

    >>> pfda = assemble_partfrac_list(pfd)
    >>> pfda
    RootSum(_w**2 - 2, Lambda(_a, _a/(-_a + x)))/2

    >>> pfda.doit()
    -sqrt(2)/(2*(x + sqrt(2))) + sqrt(2)/(2*(x - sqrt(2)))

    >>> from sympy import Dummy, Poly, Lambda, sqrt
    >>> a = Dummy("a")
    >>> pfd = (1, Poly(0, x, domain='ZZ'), [([sqrt(2),-sqrt(2)], Lambda(a, a/2), Lambda(a, -a + x), 1)])

    >>> assemble_partfrac_list(pfd)
    -sqrt(2)/(2*(x + sqrt(2))) + sqrt(2)/(2*(x - sqrt(2)))

    See Also
    ========

    apart, apart_list
    """
    # Common factor
    common = partial_list[0]

    # Polynomial part
    polypart = partial_list[1]
    pfd = polypart.as_expr()

    # Rational parts
    for r, nf, df, ex in partial_list[2]:
        if isinstance(r, Poly):
            # Assemble in case the roots are given implicitly by a polynomials
            an, nu = nf.variables, nf.expr
            ad, de = df.variables, df.expr
            # Hack to make dummies equal because Lambda created new Dummies
            de = de.subs(ad[0], an[0])
            func = Lambda(tuple(an), nu/de**ex)
            pfd += RootSum(r, func, auto=False, quadratic=False)
        else:
            # Assemble in case the roots are given explicitly by a list of algebraic numbers
            for root in r:
                pfd += nf(root)/df(root)**ex

    return common*pfd

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\ctx_base.py ===
from operator import gt, lt

from .libmp.backend import xrange

from .functions.functions import SpecialFunctions
from .functions.rszeta import RSCache
from .calculus.quadrature import QuadratureMethods
from .calculus.inverselaplace import LaplaceTransformInversionMethods
from .calculus.calculus import CalculusMethods
from .calculus.optimization import OptimizationMethods
from .calculus.odes import ODEMethods
from .matrices.matrices import MatrixMethods
from .matrices.calculus import MatrixCalculusMethods
from .matrices.linalg import LinearAlgebraMethods
from .matrices.eigen import Eigen
from .identification import IdentificationMethods
from .visualization import VisualizationMethods

from . import libmp

class Context(object):
    pass

class StandardBaseContext(Context,
    SpecialFunctions,
    RSCache,
    QuadratureMethods,
    LaplaceTransformInversionMethods,
    CalculusMethods,
    MatrixMethods,
    MatrixCalculusMethods,
    LinearAlgebraMethods,
    Eigen,
    IdentificationMethods,
    OptimizationMethods,
    ODEMethods,
    VisualizationMethods):

    NoConvergence = libmp.NoConvergence
    ComplexResult = libmp.ComplexResult

    def __init__(ctx):
        ctx._aliases = {}
        # Call those that need preinitialization (e.g. for wrappers)
        SpecialFunctions.__init__(ctx)
        RSCache.__init__(ctx)
        QuadratureMethods.__init__(ctx)
        LaplaceTransformInversionMethods.__init__(ctx)
        CalculusMethods.__init__(ctx)
        MatrixMethods.__init__(ctx)

    def _init_aliases(ctx):
        for alias, value in ctx._aliases.items():
            try:
                setattr(ctx, alias, getattr(ctx, value))
            except AttributeError:
                pass

    _fixed_precision = False

    # XXX
    verbose = False

    def warn(ctx, msg):
        print("Warning:", msg)

    def bad_domain(ctx, msg):
        raise ValueError(msg)

    def _re(ctx, x):
        if hasattr(x, "real"):
            return x.real
        return x

    def _im(ctx, x):
        if hasattr(x, "imag"):
            return x.imag
        return ctx.zero

    def _as_points(ctx, x):
        return x

    def fneg(ctx, x, **kwargs):
        return -ctx.convert(x)

    def fadd(ctx, x, y, **kwargs):
        return ctx.convert(x)+ctx.convert(y)

    def fsub(ctx, x, y, **kwargs):
        return ctx.convert(x)-ctx.convert(y)

    def fmul(ctx, x, y, **kwargs):
        return ctx.convert(x)*ctx.convert(y)

    def fdiv(ctx, x, y, **kwargs):
        return ctx.convert(x)/ctx.convert(y)

    def fsum(ctx, args, absolute=False, squared=False):
        if absolute:
            if squared:
                return sum((abs(x)**2 for x in args), ctx.zero)
            return sum((abs(x) for x in args), ctx.zero)
        if squared:
            return sum((x**2 for x in args), ctx.zero)
        return sum(args, ctx.zero)

    def fdot(ctx, xs, ys=None, conjugate=False):
        if ys is not None:
            xs = zip(xs, ys)
        if conjugate:
            cf = ctx.conj
            return sum((x*cf(y) for (x,y) in xs), ctx.zero)
        else:
            return sum((x*y for (x,y) in xs), ctx.zero)

    def fprod(ctx, args):
        prod = ctx.one
        for arg in args:
            prod *= arg
        return prod

    def nprint(ctx, x, n=6, **kwargs):
        """
        Equivalent to ``print(nstr(x, n))``.
        """
        print(ctx.nstr(x, n, **kwargs))

    def chop(ctx, x, tol=None):
        """
        Chops off small real or imaginary parts, or converts
        numbers close to zero to exact zeros. The input can be a
        single number or an iterable::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> chop(5+1e-10j, tol=1e-9)
            mpf('5.0')
            >>> nprint(chop([1.0, 1e-20, 3+1e-18j, -4, 2]))
            [1.0, 0.0, 3.0, -4.0, 2.0]

        The tolerance defaults to ``100*eps``.
        """
        if tol is None:
            tol = 100*ctx.eps
        try:
            x = ctx.convert(x)
            absx = abs(x)
            if abs(x) < tol:
                return ctx.zero
            if ctx._is_complex_type(x):
                #part_tol = min(tol, absx*tol)
                part_tol = max(tol, absx*tol)
                if abs(x.imag) < part_tol:
                    return x.real
                if abs(x.real) < part_tol:
                    return ctx.mpc(0, x.imag)
        except TypeError:
            if isinstance(x, ctx.matrix):
                return x.apply(lambda a: ctx.chop(a, tol))
            if hasattr(x, "__iter__"):
                return [ctx.chop(a, tol) for a in x]
        return x

    def almosteq(ctx, s, t, rel_eps=None, abs_eps=None):
        r"""
        Determine whether the difference between `s` and `t` is smaller
        than a given epsilon, either relatively or absolutely.

        Both a maximum relative difference and a maximum difference
        ('epsilons') may be specified. The absolute difference is
        defined as `|s-t|` and the relative difference is defined
        as `|s-t|/\max(|s|, |t|)`.

        If only one epsilon is given, both are set to the same value.
        If none is given, both epsilons are set to `2^{-p+m}` where
        `p` is the current working precision and `m` is a small
        integer. The default setting typically allows :func:`~mpmath.almosteq`
        to be used to check for mathematical equality
        in the presence of small rounding errors.

        **Examples**

            >>> from mpmath import *
            >>> mp.dps = 15
            >>> almosteq(3.141592653589793, 3.141592653589790)
            True
            >>> almosteq(3.141592653589793, 3.141592653589700)
            False
            >>> almosteq(3.141592653589793, 3.141592653589700, 1e-10)
            True
            >>> almosteq(1e-20, 2e-20)
            True
            >>> almosteq(1e-20, 2e-20, rel_eps=0, abs_eps=0)
            False

        """
        t = ctx.convert(t)
        if abs_eps is None and rel_eps is None:
            rel_eps = abs_eps = ctx.ldexp(1, -ctx.prec+4)
        if abs_eps is None:
            abs_eps = rel_eps
        elif rel_eps is None:
            rel_eps = abs_eps
        diff = abs(s-t)
        if diff <= abs_eps:
            return True
        abss = abs(s)
        abst = abs(t)
        if abss < abst:
            err = diff/abst
        else:
            err = diff/abss
        return err <= rel_eps

    def arange(ctx, *args):
        r"""
        This is a generalized version of Python's :func:`~mpmath.range` function
        that accepts fractional endpoints and step sizes and
        returns a list of ``mpf`` instances. Like :func:`~mpmath.range`,
        :func:`~mpmath.arange` can be called with 1, 2 or 3 arguments:

        ``arange(b)``
            `[0, 1, 2, \ldots, x]`
        ``arange(a, b)``
            `[a, a+1, a+2, \ldots, x]`
        ``arange(a, b, h)``
            `[a, a+h, a+h, \ldots, x]`

        where `b-1 \le x < b` (in the third case, `b-h \le x < b`).

        Like Python's :func:`~mpmath.range`, the endpoint is not included. To
        produce ranges where the endpoint is included, :func:`~mpmath.linspace`
        is more convenient.

        **Examples**

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> arange(4)
            [mpf('0.0'), mpf('1.0'), mpf('2.0'), mpf('3.0')]
            >>> arange(1, 2, 0.25)
            [mpf('1.0'), mpf('1.25'), mpf('1.5'), mpf('1.75')]
            >>> arange(1, -1, -0.75)
            [mpf('1.0'), mpf('0.25'), mpf('-0.5')]

        """
        if not len(args) <= 3:
            raise TypeError('arange expected at most 3 arguments, got %i'
                            % len(args))
        if not len(args) >= 1:
            raise TypeError('arange expected at least 1 argument, got %i'
                            % len(args))
        # set default
        a = 0
        dt = 1
        # interpret arguments
        if len(args) == 1:
            b = args[0]
        elif len(args) >= 2:
            a = args[0]
            b = args[1]
        if len(args) == 3:
            dt = args[2]
        a, b, dt = ctx.mpf(a), ctx.mpf(b), ctx.mpf(dt)
        assert a + dt != a, 'dt is too small and would cause an infinite loop'
        # adapt code for sign of dt
        if a > b:
            if dt > 0:
                return []
            op = gt
        else:
            if dt < 0:
                return []
            op = lt
        # create list
        result = []
        i = 0
        t = a
        while 1:
            t = a + dt*i
            i += 1
            if op(t, b):
                result.append(t)
            else:
                break
        return result

    def linspace(ctx, *args, **kwargs):
        """
        ``linspace(a, b, n)`` returns a list of `n` evenly spaced
        samples from `a` to `b`. The syntax ``linspace(mpi(a,b), n)``
        is also valid.

        This function is often more convenient than :func:`~mpmath.arange`
        for partitioning an interval into subintervals, since
        the endpoint is included::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = False
            >>> linspace(1, 4, 4)
            [mpf('1.0'), mpf('2.0'), mpf('3.0'), mpf('4.0')]

        You may also provide the keyword argument ``endpoint=False``::

            >>> linspace(1, 4, 4, endpoint=False)
            [mpf('1.0'), mpf('1.75'), mpf('2.5'), mpf('3.25')]

        """
        if len(args) == 3:
            a = ctx.mpf(args[0])
            b = ctx.mpf(args[1])
            n = int(args[2])
        elif len(args) == 2:
            assert hasattr(args[0], '_mpi_')
            a = args[0].a
            b = args[0].b
            n = int(args[1])
        else:
            raise TypeError('linspace expected 2 or 3 arguments, got %i' \
                            % len(args))
        if n < 1:
            raise ValueError('n must be greater than 0')
        if not 'endpoint' in kwargs or kwargs['endpoint']:
            if n == 1:
                return [ctx.mpf(a)]
            step = (b - a) / ctx.mpf(n - 1)
            y = [i*step + a for i in xrange(n)]
            y[-1] = b
        else:
            step = (b - a) / ctx.mpf(n)
            y = [i*step + a for i in xrange(n)]
        return y

    def cos_sin(ctx, z, **kwargs):
        return ctx.cos(z, **kwargs), ctx.sin(z, **kwargs)

    def cospi_sinpi(ctx, z, **kwargs):
        return ctx.cospi(z, **kwargs), ctx.sinpi(z, **kwargs)

    def _default_hyper_maxprec(ctx, p):
        return int(1000 * p**0.25 + 4*p)

    _gcd = staticmethod(libmp.gcd)
    list_primes = staticmethod(libmp.list_primes)
    isprime = staticmethod(libmp.isprime)
    bernfrac = staticmethod(libmp.bernfrac)
    moebius = staticmethod(libmp.moebius)
    _ifac = staticmethod(libmp.ifac)
    _eulernum = staticmethod(libmp.eulernum)
    _stirling1 = staticmethod(libmp.stirling1)
    _stirling2 = staticmethod(libmp.stirling2)

    def sum_accurately(ctx, terms, check_step=1):
        prec = ctx.prec
        try:
            extraprec = 10
            while 1:
                ctx.prec = prec + extraprec + 5
                max_mag = ctx.ninf
                s = ctx.zero
                k = 0
                for term in terms():
                    s += term
                    if (not k % check_step) and term:
                        term_mag = ctx.mag(term)
                        max_mag = max(max_mag, term_mag)
                        sum_mag = ctx.mag(s)
                        if sum_mag - term_mag > ctx.prec:
                            break
                    k += 1
                cancellation = max_mag - sum_mag
                if cancellation != cancellation:
                    break
                if cancellation < extraprec or ctx._fixed_precision:
                    break
                extraprec += min(ctx.prec, cancellation)
            return s
        finally:
            ctx.prec = prec

    def mul_accurately(ctx, factors, check_step=1):
        prec = ctx.prec
        try:
            extraprec = 10
            while 1:
                ctx.prec = prec + extraprec + 5
                max_mag = ctx.ninf
                one = ctx.one
                s = one
                k = 0
                for factor in factors():
                    s *= factor
                    term = factor - one
                    if (not k % check_step):
                        term_mag = ctx.mag(term)
                        max_mag = max(max_mag, term_mag)
                        sum_mag = ctx.mag(s-one)
                        #if sum_mag - term_mag > ctx.prec:
                        #    break
                        if -term_mag > ctx.prec:
                            break
                    k += 1
                cancellation = max_mag - sum_mag
                if cancellation != cancellation:
                    break
                if cancellation < extraprec or ctx._fixed_precision:
                    break
                extraprec += min(ctx.prec, cancellation)
            return s
        finally:
            ctx.prec = prec

    def power(ctx, x, y):
        r"""Converts `x` and `y` to mpmath numbers and evaluates
        `x^y = \exp(y \log(x))`::

            >>> from mpmath import *
            >>> mp.dps = 30; mp.pretty = True
            >>> power(2, 0.5)
            1.41421356237309504880168872421

        This shows the leading few digits of a large Mersenne prime
        (performing the exact calculation ``2**43112609-1`` and
        displaying the result in Python would be very slow)::

            >>> power(2, 43112609)-1
            3.16470269330255923143453723949e+12978188
        """
        return ctx.convert(x) ** ctx.convert(y)

    def _zeta_int(ctx, n):
        return ctx.zeta(n)

    def maxcalls(ctx, f, N):
        """
        Return a wrapped copy of *f* that raises ``NoConvergence`` when *f*
        has been called more than *N* times::

            >>> from mpmath import *
            >>> mp.dps = 15
            >>> f = maxcalls(sin, 10)
            >>> print(sum(f(n) for n in range(10)))
            1.95520948210738
            >>> f(10) # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
              ...
            NoConvergence: maxcalls: function evaluated 10 times

        """
        counter = [0]
        def f_maxcalls_wrapped(*args, **kwargs):
            counter[0] += 1
            if counter[0] > N:
                raise ctx.NoConvergence("maxcalls: function evaluated %i times" % N)
            return f(*args, **kwargs)
        return f_maxcalls_wrapped

    def memoize(ctx, f):
        """
        Return a wrapped copy of *f* that caches computed values, i.e.
        a memoized copy of *f*. Values are only reused if the cached precision
        is equal to or higher than the working precision::

            >>> from mpmath import *
            >>> mp.dps = 15; mp.pretty = True
            >>> f = memoize(maxcalls(sin, 1))
            >>> f(2)
            0.909297426825682
            >>> f(2)
            0.909297426825682
            >>> mp.dps = 25
            >>> f(2) # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
              ...
            NoConvergence: maxcalls: function evaluated 1 times

        """
        f_cache = {}
        def f_cached(*args, **kwargs):
            if kwargs:
                key = args, tuple(kwargs.items())
            else:
                key = args
            prec = ctx.prec
            if key in f_cache:
                cprec, cvalue = f_cache[key]
                if cprec >= prec:
                    return +cvalue
            value = f(*args, **kwargs)
            f_cache[key] = (prec, value)
            return value
        f_cached.__name__ = f.__name__
        f_cached.__doc__ = f.__doc__
        return f_cached

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\index\collector.py ===
"""
The main purpose of this module is to expose LinkCollector.collect_sources().
"""

import collections
import email.message
import functools
import itertools
import json
import logging
import os
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from optparse import Values
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    MutableMapping,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
)

from pip._vendor import requests
from pip._vendor.requests import Response
from pip._vendor.requests.exceptions import RetryError, SSLError

from pip._internal.exceptions import NetworkConnectionError
from pip._internal.models.link import Link
from pip._internal.models.search_scope import SearchScope
from pip._internal.network.session import PipSession
from pip._internal.network.utils import raise_for_status
from pip._internal.utils.filetypes import is_archive_file
from pip._internal.utils.misc import redact_auth_from_url
from pip._internal.vcs import vcs

from .sources import CandidatesFromPage, LinkSource, build_source

logger = logging.getLogger(__name__)

ResponseHeaders = MutableMapping[str, str]


def _match_vcs_scheme(url: str) -> Optional[str]:
    """Look for VCS schemes in the URL.

    Returns the matched VCS scheme, or None if there's no match.
    """
    for scheme in vcs.schemes:
        if url.lower().startswith(scheme) and url[len(scheme)] in "+:":
            return scheme
    return None


class _NotAPIContent(Exception):
    def __init__(self, content_type: str, request_desc: str) -> None:
        super().__init__(content_type, request_desc)
        self.content_type = content_type
        self.request_desc = request_desc


def _ensure_api_header(response: Response) -> None:
    """
    Check the Content-Type header to ensure the response contains a Simple
    API Response.

    Raises `_NotAPIContent` if the content type is not a valid content-type.
    """
    content_type = response.headers.get("Content-Type", "Unknown")

    content_type_l = content_type.lower()
    if content_type_l.startswith(
        (
            "text/html",
            "application/vnd.pypi.simple.v1+html",
            "application/vnd.pypi.simple.v1+json",
        )
    ):
        return

    raise _NotAPIContent(content_type, response.request.method)


class _NotHTTP(Exception):
    pass


def _ensure_api_response(url: str, session: PipSession) -> None:
    """
    Send a HEAD request to the URL, and ensure the response contains a simple
    API Response.

    Raises `_NotHTTP` if the URL is not available for a HEAD request, or
    `_NotAPIContent` if the content type is not a valid content type.
    """
    scheme, netloc, path, query, fragment = urllib.parse.urlsplit(url)
    if scheme not in {"http", "https"}:
        raise _NotHTTP()

    resp = session.head(url, allow_redirects=True)
    raise_for_status(resp)

    _ensure_api_header(resp)


def _get_simple_response(url: str, session: PipSession) -> Response:
    """Access an Simple API response with GET, and return the response.

    This consists of three parts:

    1. If the URL looks suspiciously like an archive, send a HEAD first to
       check the Content-Type is HTML or Simple API, to avoid downloading a
       large file. Raise `_NotHTTP` if the content type cannot be determined, or
       `_NotAPIContent` if it is not HTML or a Simple API.
    2. Actually perform the request. Raise HTTP exceptions on network failures.
    3. Check the Content-Type header to make sure we got a Simple API response,
       and raise `_NotAPIContent` otherwise.
    """
    if is_archive_file(Link(url).filename):
        _ensure_api_response(url, session=session)

    logger.debug("Getting page %s", redact_auth_from_url(url))

    resp = session.get(
        url,
        headers={
            "Accept": ", ".join(
                [
                    "application/vnd.pypi.simple.v1+json",
                    "application/vnd.pypi.simple.v1+html; q=0.1",
                    "text/html; q=0.01",
                ]
            ),
            # We don't want to blindly returned cached data for
            # /simple/, because authors generally expecting that
            # twine upload && pip install will function, but if
            # they've done a pip install in the last ~10 minutes
            # it won't. Thus by setting this to zero we will not
            # blindly use any cached data, however the benefit of
            # using max-age=0 instead of no-cache, is that we will
            # still support conditional requests, so we will still
            # minimize traffic sent in cases where the page hasn't
            # changed at all, we will just always incur the round
            # trip for the conditional GET now instead of only
            # once per 10 minutes.
            # For more information, please see pypa/pip#5670.
            "Cache-Control": "max-age=0",
        },
    )
    raise_for_status(resp)

    # The check for archives above only works if the url ends with
    # something that looks like an archive. However that is not a
    # requirement of an url. Unless we issue a HEAD request on every
    # url we cannot know ahead of time for sure if something is a
    # Simple API response or not. However we can check after we've
    # downloaded it.
    _ensure_api_header(resp)

    logger.debug(
        "Fetched page %s as %s",
        redact_auth_from_url(url),
        resp.headers.get("Content-Type", "Unknown"),
    )

    return resp


def _get_encoding_from_headers(headers: ResponseHeaders) -> Optional[str]:
    """Determine if we have any encoding information in our headers."""
    if headers and "Content-Type" in headers:
        m = email.message.Message()
        m["content-type"] = headers["Content-Type"]
        charset = m.get_param("charset")
        if charset:
            return str(charset)
    return None


class CacheablePageContent:
    def __init__(self, page: "IndexContent") -> None:
        assert page.cache_link_parsing
        self.page = page

    def __eq__(self, other: object) -> bool:
        return isinstance(other, type(self)) and self.page.url == other.page.url

    def __hash__(self) -> int:
        return hash(self.page.url)


class ParseLinks(Protocol):
    def __call__(self, page: "IndexContent") -> Iterable[Link]: ...


def with_cached_index_content(fn: ParseLinks) -> ParseLinks:
    """
    Given a function that parses an Iterable[Link] from an IndexContent, cache the
    function's result (keyed by CacheablePageContent), unless the IndexContent
    `page` has `page.cache_link_parsing == False`.
    """

    @functools.lru_cache(maxsize=None)
    def wrapper(cacheable_page: CacheablePageContent) -> List[Link]:
        return list(fn(cacheable_page.page))

    @functools.wraps(fn)
    def wrapper_wrapper(page: "IndexContent") -> List[Link]:
        if page.cache_link_parsing:
            return wrapper(CacheablePageContent(page))
        return list(fn(page))

    return wrapper_wrapper


@with_cached_index_content
def parse_links(page: "IndexContent") -> Iterable[Link]:
    """
    Parse a Simple API's Index Content, and yield its anchor elements as Link objects.
    """

    content_type_l = page.content_type.lower()
    if content_type_l.startswith("application/vnd.pypi.simple.v1+json"):
        data = json.loads(page.content)
        for file in data.get("files", []):
            link = Link.from_json(file, page.url)
            if link is None:
                continue
            yield link
        return

    parser = HTMLLinkParser(page.url)
    encoding = page.encoding or "utf-8"
    parser.feed(page.content.decode(encoding))

    url = page.url
    base_url = parser.base_url or url
    for anchor in parser.anchors:
        link = Link.from_element(anchor, page_url=url, base_url=base_url)
        if link is None:
            continue
        yield link


@dataclass(frozen=True)
class IndexContent:
    """Represents one response (or page), along with its URL.

    :param encoding: the encoding to decode the given content.
    :param url: the URL from which the HTML was downloaded.
    :param cache_link_parsing: whether links parsed from this page's url
                               should be cached. PyPI index urls should
                               have this set to False, for example.
    """

    content: bytes
    content_type: str
    encoding: Optional[str]
    url: str
    cache_link_parsing: bool = True

    def __str__(self) -> str:
        return redact_auth_from_url(self.url)


class HTMLLinkParser(HTMLParser):
    """
    HTMLParser that keeps the first base HREF and a list of all anchor
    elements' attributes.
    """

    def __init__(self, url: str) -> None:
        super().__init__(convert_charrefs=True)

        self.url: str = url
        self.base_url: Optional[str] = None
        self.anchors: List[Dict[str, Optional[str]]] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag == "base" and self.base_url is None:
            href = self.get_href(attrs)
            if href is not None:
                self.base_url = href
        elif tag == "a":
            self.anchors.append(dict(attrs))

    def get_href(self, attrs: List[Tuple[str, Optional[str]]]) -> Optional[str]:
        for name, value in attrs:
            if name == "href":
                return value
        return None


def _handle_get_simple_fail(
    link: Link,
    reason: Union[str, Exception],
    meth: Optional[Callable[..., None]] = None,
) -> None:
    if meth is None:
        meth = logger.debug
    meth("Could not fetch URL %s: %s - skipping", link, reason)


def _make_index_content(
    response: Response, cache_link_parsing: bool = True
) -> IndexContent:
    encoding = _get_encoding_from_headers(response.headers)
    return IndexContent(
        response.content,
        response.headers["Content-Type"],
        encoding=encoding,
        url=response.url,
        cache_link_parsing=cache_link_parsing,
    )


def _get_index_content(link: Link, *, session: PipSession) -> Optional["IndexContent"]:
    url = link.url.split("#", 1)[0]

    # Check for VCS schemes that do not support lookup as web pages.
    vcs_scheme = _match_vcs_scheme(url)
    if vcs_scheme:
        logger.warning(
            "Cannot look at %s URL %s because it does not support lookup as web pages.",
            vcs_scheme,
            link,
        )
        return None

    # Tack index.html onto file:// URLs that point to directories
    scheme, _, path, _, _, _ = urllib.parse.urlparse(url)
    if scheme == "file" and os.path.isdir(urllib.request.url2pathname(path)):
        # add trailing slash if not present so urljoin doesn't trim
        # final segment
        if not url.endswith("/"):
            url += "/"
        # TODO: In the future, it would be nice if pip supported PEP 691
        #       style responses in the file:// URLs, however there's no
        #       standard file extension for application/vnd.pypi.simple.v1+json
        #       so we'll need to come up with something on our own.
        url = urllib.parse.urljoin(url, "index.html")
        logger.debug(" file: URL is directory, getting %s", url)

    try:
        resp = _get_simple_response(url, session=session)
    except _NotHTTP:
        logger.warning(
            "Skipping page %s because it looks like an archive, and cannot "
            "be checked by a HTTP HEAD request.",
            link,
        )
    except _NotAPIContent as exc:
        logger.warning(
            "Skipping page %s because the %s request got Content-Type: %s. "
            "The only supported Content-Types are application/vnd.pypi.simple.v1+json, "
            "application/vnd.pypi.simple.v1+html, and text/html",
            link,
            exc.request_desc,
            exc.content_type,
        )
    except NetworkConnectionError as exc:
        _handle_get_simple_fail(link, exc)
    except RetryError as exc:
        _handle_get_simple_fail(link, exc)
    except SSLError as exc:
        reason = "There was a problem confirming the ssl certificate: "
        reason += str(exc)
        _handle_get_simple_fail(link, reason, meth=logger.info)
    except requests.ConnectionError as exc:
        _handle_get_simple_fail(link, f"connection error: {exc}")
    except requests.Timeout:
        _handle_get_simple_fail(link, "timed out")
    else:
        return _make_index_content(resp, cache_link_parsing=link.cache_link_parsing)
    return None


class CollectedSources(NamedTuple):
    find_links: Sequence[Optional[LinkSource]]
    index_urls: Sequence[Optional[LinkSource]]


class LinkCollector:
    """
    Responsible for collecting Link objects from all configured locations,
    making network requests as needed.

    The class's main method is its collect_sources() method.
    """

    def __init__(
        self,
        session: PipSession,
        search_scope: SearchScope,
    ) -> None:
        self.search_scope = search_scope
        self.session = session

    @classmethod
    def create(
        cls,
        session: PipSession,
        options: Values,
        suppress_no_index: bool = False,
    ) -> "LinkCollector":
        """
        :param session: The Session to use to make requests.
        :param suppress_no_index: Whether to ignore the --no-index option
            when constructing the SearchScope object.
        """
        index_urls = [options.index_url] + options.extra_index_urls
        if options.no_index and not suppress_no_index:
            logger.debug(
                "Ignoring indexes: %s",
                ",".join(redact_auth_from_url(url) for url in index_urls),
            )
            index_urls = []

        # Make sure find_links is a list before passing to create().
        find_links = options.find_links or []

        search_scope = SearchScope.create(
            find_links=find_links,
            index_urls=index_urls,
            no_index=options.no_index,
        )
        link_collector = LinkCollector(
            session=session,
            search_scope=search_scope,
        )
        return link_collector

    @property
    def find_links(self) -> List[str]:
        return self.search_scope.find_links

    def fetch_response(self, location: Link) -> Optional[IndexContent]:
        """
        Fetch an HTML page containing package links.
        """
        return _get_index_content(location, session=self.session)

    def collect_sources(
        self,
        project_name: str,
        candidates_from_page: CandidatesFromPage,
    ) -> CollectedSources:
        # The OrderedDict calls deduplicate sources by URL.
        index_url_sources = collections.OrderedDict(
            build_source(
                loc,
                candidates_from_page=candidates_from_page,
                page_validator=self.session.is_secure_origin,
                expand_dir=False,
                cache_link_parsing=False,
                project_name=project_name,
            )
            for loc in self.search_scope.get_index_urls_locations(project_name)
        ).values()
        find_links_sources = collections.OrderedDict(
            build_source(
                loc,
                candidates_from_page=candidates_from_page,
                page_validator=self.session.is_secure_origin,
                expand_dir=True,
                cache_link_parsing=True,
                project_name=project_name,
            )
            for loc in self.find_links
        ).values()

        if logger.isEnabledFor(logging.DEBUG):
            lines = [
                f"* {s.link}"
                for s in itertools.chain(find_links_sources, index_url_sources)
                if s is not None and s.link is not None
            ]
            lines = [
                f"{len(lines)} location(s) to search "
                f"for versions of {project_name}:"
            ] + lines
            logger.debug("\n".join(lines))

        return CollectedSources(
            find_links=list(find_links_sources),
            index_urls=list(index_url_sources),
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\functions\orthogonal.py ===
from .functions import defun, defun_wrapped

def _hermite_param(ctx, n, z, parabolic_cylinder):
    """
    Combined calculation of the Hermite polynomial H_n(z) (and its
    generalization to complex n) and the parabolic cylinder
    function D.
    """
    n, ntyp = ctx._convert_param(n)
    z = ctx.convert(z)
    q = -ctx.mpq_1_2
    # For re(z) > 0, 2F0 -- http://functions.wolfram.com/
    #     HypergeometricFunctions/HermiteHGeneral/06/02/0009/
    # Otherwise, there is a reflection formula
    # 2F0 + http://functions.wolfram.com/HypergeometricFunctions/
    #           HermiteHGeneral/16/01/01/0006/
    #
    # TODO:
    # An alternative would be to use
    # http://functions.wolfram.com/HypergeometricFunctions/
    #     HermiteHGeneral/06/02/0006/
    #
    # Also, the 1F1 expansion
    # http://functions.wolfram.com/HypergeometricFunctions/
    #     HermiteHGeneral/26/01/02/0001/
    # should probably be used for tiny z
    if not z:
        T1 = [2, ctx.pi], [n, 0.5], [], [q*(n-1)], [], [], 0
        if parabolic_cylinder:
            T1[1][0] += q*n
        return T1,
    can_use_2f0 = ctx.isnpint(-n) or ctx.re(z) > 0 or \
        (ctx.re(z) == 0 and ctx.im(z) > 0)
    expprec = ctx.prec*4 + 20
    if parabolic_cylinder:
        u = ctx.fmul(ctx.fmul(z,z,prec=expprec), -0.25, exact=True)
        w = ctx.fmul(z, ctx.sqrt(0.5,prec=expprec), prec=expprec)
    else:
        w = z
    w2 = ctx.fmul(w, w, prec=expprec)
    rw2 = ctx.fdiv(1, w2, prec=expprec)
    nrw2 = ctx.fneg(rw2, exact=True)
    nw = ctx.fneg(w, exact=True)
    if can_use_2f0:
        T1 = [2, w], [n, n], [], [], [q*n, q*(n-1)], [], nrw2
        terms = [T1]
    else:
        T1 = [2, nw], [n, n], [], [], [q*n, q*(n-1)], [], nrw2
        T2 = [2, ctx.pi, nw], [n+2, 0.5, 1], [], [q*n], [q*(n-1)], [1-q], w2
        terms = [T1,T2]
    # Multiply by prefactor for D_n
    if parabolic_cylinder:
        expu = ctx.exp(u)
        for i in range(len(terms)):
            terms[i][1][0] += q*n
            terms[i][0].append(expu)
            terms[i][1].append(1)
    return tuple(terms)

@defun
def hermite(ctx, n, z, **kwargs):
    return ctx.hypercomb(lambda: _hermite_param(ctx, n, z, 0), [], **kwargs)

@defun
def pcfd(ctx, n, z, **kwargs):
    r"""
    Gives the parabolic cylinder function in Whittaker's notation
    `D_n(z) = U(-n-1/2, z)` (see :func:`~mpmath.pcfu`).
    It solves the differential equation

    .. math ::

        y'' + \left(n + \frac{1}{2} - \frac{1}{4} z^2\right) y = 0.

    and can be represented in terms of Hermite polynomials
    (see :func:`~mpmath.hermite`) as

    .. math ::

        D_n(z) = 2^{-n/2} e^{-z^2/4} H_n\left(\frac{z}{\sqrt{2}}\right).

    **Plots**

    .. literalinclude :: /plots/pcfd.py
    .. image :: /plots/pcfd.png

    **Examples**

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> pcfd(0,0); pcfd(1,0); pcfd(2,0); pcfd(3,0)
        1.0
        0.0
        -1.0
        0.0
        >>> pcfd(4,0); pcfd(-3,0)
        3.0
        0.6266570686577501256039413
        >>> pcfd('1/2', 2+3j)
        (-5.363331161232920734849056 - 3.858877821790010714163487j)
        >>> pcfd(2, -10)
        1.374906442631438038871515e-9

    Verifying the differential equation::

        >>> n = mpf(2.5)
        >>> y = lambda z: pcfd(n,z)
        >>> z = 1.75
        >>> chop(diff(y,z,2) + (n+0.5-0.25*z**2)*y(z))
        0.0

    Rational Taylor series expansion when `n` is an integer::

        >>> taylor(lambda z: pcfd(5,z), 0, 7)
        [0.0, 15.0, 0.0, -13.75, 0.0, 3.96875, 0.0, -0.6015625]

    """
    return ctx.hypercomb(lambda: _hermite_param(ctx, n, z, 1), [], **kwargs)

@defun
def pcfu(ctx, a, z, **kwargs):
    r"""
    Gives the parabolic cylinder function `U(a,z)`, which may be
    defined for `\Re(z) > 0` in terms of the confluent
    U-function (see :func:`~mpmath.hyperu`) by

    .. math ::

        U(a,z) = 2^{-\frac{1}{4}-\frac{a}{2}} e^{-\frac{1}{4} z^2}
            U\left(\frac{a}{2}+\frac{1}{4},
            \frac{1}{2}, \frac{1}{2}z^2\right)

    or, for arbitrary `z`,

    .. math ::

        e^{-\frac{1}{4}z^2} U(a,z) =
            U(a,0) \,_1F_1\left(-\tfrac{a}{2}+\tfrac{1}{4};
            \tfrac{1}{2}; -\tfrac{1}{2}z^2\right) +
            U'(a,0) z \,_1F_1\left(-\tfrac{a}{2}+\tfrac{3}{4};
            \tfrac{3}{2}; -\tfrac{1}{2}z^2\right).

    **Examples**

    Connection to other functions::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> z = mpf(3)
        >>> pcfu(0.5,z)
        0.03210358129311151450551963
        >>> sqrt(pi/2)*exp(z**2/4)*erfc(z/sqrt(2))
        0.03210358129311151450551963
        >>> pcfu(0.5,-z)
        23.75012332835297233711255
        >>> sqrt(pi/2)*exp(z**2/4)*erfc(-z/sqrt(2))
        23.75012332835297233711255
        >>> pcfu(0.5,-z)
        23.75012332835297233711255
        >>> sqrt(pi/2)*exp(z**2/4)*erfc(-z/sqrt(2))
        23.75012332835297233711255

    """
    n, _ = ctx._convert_param(a)
    return ctx.pcfd(-n-ctx.mpq_1_2, z)

@defun
def pcfv(ctx, a, z, **kwargs):
    r"""
    Gives the parabolic cylinder function `V(a,z)`, which can be
    represented in terms of :func:`~mpmath.pcfu` as

    .. math ::

        V(a,z) = \frac{\Gamma(a+\tfrac{1}{2}) (U(a,-z)-\sin(\pi a) U(a,z)}{\pi}.

    **Examples**

    Wronskian relation between `U` and `V`::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> a, z = 2, 3
        >>> pcfu(a,z)*diff(pcfv,(a,z),(0,1))-diff(pcfu,(a,z),(0,1))*pcfv(a,z)
        0.7978845608028653558798921
        >>> sqrt(2/pi)
        0.7978845608028653558798921
        >>> a, z = 2.5, 3
        >>> pcfu(a,z)*diff(pcfv,(a,z),(0,1))-diff(pcfu,(a,z),(0,1))*pcfv(a,z)
        0.7978845608028653558798921
        >>> a, z = 0.25, -1
        >>> pcfu(a,z)*diff(pcfv,(a,z),(0,1))-diff(pcfu,(a,z),(0,1))*pcfv(a,z)
        0.7978845608028653558798921
        >>> a, z = 2+1j, 2+3j
        >>> chop(pcfu(a,z)*diff(pcfv,(a,z),(0,1))-diff(pcfu,(a,z),(0,1))*pcfv(a,z))
        0.7978845608028653558798921

    """
    n, ntype = ctx._convert_param(a)
    z = ctx.convert(z)
    q = ctx.mpq_1_2
    r = ctx.mpq_1_4
    if ntype == 'Q' and ctx.isint(n*2):
        # Faster for half-integers
        def h():
            jz = ctx.fmul(z, -1j, exact=True)
            T1terms = _hermite_param(ctx, -n-q, z, 1)
            T2terms = _hermite_param(ctx, n-q, jz, 1)
            for T in T1terms:
                T[0].append(1j)
                T[1].append(1)
                T[3].append(q-n)
            u = ctx.expjpi((q*n-r)) * ctx.sqrt(2/ctx.pi)
            for T in T2terms:
                T[0].append(u)
                T[1].append(1)
            return T1terms + T2terms
        v = ctx.hypercomb(h, [], **kwargs)
        if ctx._is_real_type(n) and ctx._is_real_type(z):
            v = ctx._re(v)
        return v
    else:
        def h(n):
            w = ctx.square_exp_arg(z, -0.25)
            u = ctx.square_exp_arg(z, 0.5)
            e = ctx.exp(w)
            l = [ctx.pi, q, ctx.exp(w)]
            Y1 = l, [-q, n*q+r, 1], [r-q*n], [], [q*n+r], [q], u
            Y2 = l + [z], [-q, n*q-r, 1, 1], [1-r-q*n], [], [q*n+1-r], [1+q], u
            c, s = ctx.cospi_sinpi(r+q*n)
            Y1[0].append(s)
            Y2[0].append(c)
            for Y in (Y1, Y2):
                Y[1].append(1)
                Y[3].append(q-n)
            return Y1, Y2
        return ctx.hypercomb(h, [n], **kwargs)


@defun
def pcfw(ctx, a, z, **kwargs):
    r"""
    Gives the parabolic cylinder function `W(a,z)` defined in (DLMF 12.14).

    **Examples**

    Value at the origin::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> a = mpf(0.25)
        >>> pcfw(a,0)
        0.9722833245718180765617104
        >>> power(2,-0.75)*sqrt(abs(gamma(0.25+0.5j*a)/gamma(0.75+0.5j*a)))
        0.9722833245718180765617104
        >>> diff(pcfw,(a,0),(0,1))
        -0.5142533944210078966003624
        >>> -power(2,-0.25)*sqrt(abs(gamma(0.75+0.5j*a)/gamma(0.25+0.5j*a)))
        -0.5142533944210078966003624

    """
    n, _ = ctx._convert_param(a)
    z = ctx.convert(z)
    def terms():
        phi2 = ctx.arg(ctx.gamma(0.5 + ctx.j*n))
        phi2 = (ctx.loggamma(0.5+ctx.j*n) - ctx.loggamma(0.5-ctx.j*n))/2j
        rho = ctx.pi/8 + 0.5*phi2
        # XXX: cancellation computing k
        k = ctx.sqrt(1 + ctx.exp(2*ctx.pi*n)) - ctx.exp(ctx.pi*n)
        C = ctx.sqrt(k/2) * ctx.exp(0.25*ctx.pi*n)
        yield C * ctx.expj(rho) * ctx.pcfu(ctx.j*n, z*ctx.expjpi(-0.25))
        yield C * ctx.expj(-rho) * ctx.pcfu(-ctx.j*n, z*ctx.expjpi(0.25))
    v = ctx.sum_accurately(terms)
    if ctx._is_real_type(n) and ctx._is_real_type(z):
        v = ctx._re(v)
    return v

"""
Even/odd PCFs. Useful?

@defun
def pcfy1(ctx, a, z, **kwargs):
    a, _ = ctx._convert_param(n)
    z = ctx.convert(z)
    def h():
        w = ctx.square_exp_arg(z)
        w1 = ctx.fmul(w, -0.25, exact=True)
        w2 = ctx.fmul(w, 0.5, exact=True)
        e = ctx.exp(w1)
        return [e], [1], [], [], [ctx.mpq_1_2*a+ctx.mpq_1_4], [ctx.mpq_1_2], w2
    return ctx.hypercomb(h, [], **kwargs)

@defun
def pcfy2(ctx, a, z, **kwargs):
    a, _ = ctx._convert_param(n)
    z = ctx.convert(z)
    def h():
        w = ctx.square_exp_arg(z)
        w1 = ctx.fmul(w, -0.25, exact=True)
        w2 = ctx.fmul(w, 0.5, exact=True)
        e = ctx.exp(w1)
        return [e, z], [1, 1], [], [], [ctx.mpq_1_2*a+ctx.mpq_3_4], \
            [ctx.mpq_3_2], w2
    return ctx.hypercomb(h, [], **kwargs)
"""

@defun_wrapped
def gegenbauer(ctx, n, a, z, **kwargs):
    # Special cases: a+0.5, a*2 poles
    if ctx.isnpint(a):
        return 0*(z+n)
    if ctx.isnpint(a+0.5):
        # TODO: something else is required here
        # E.g.: gegenbauer(-2, -0.5, 3) == -12
        if ctx.isnpint(n+1):
            raise NotImplementedError("Gegenbauer function with two limits")
        def h(a):
            a2 = 2*a
            T = [], [], [n+a2], [n+1, a2], [-n, n+a2], [a+0.5], 0.5*(1-z)
            return [T]
        return ctx.hypercomb(h, [a], **kwargs)
    def h(n):
        a2 = 2*a
        T = [], [], [n+a2], [n+1, a2], [-n, n+a2], [a+0.5], 0.5*(1-z)
        return [T]
    return ctx.hypercomb(h, [n], **kwargs)

@defun_wrapped
def jacobi(ctx, n, a, b, x, **kwargs):
    if not ctx.isnpint(a):
        def h(n):
            return (([], [], [a+n+1], [n+1, a+1], [-n, a+b+n+1], [a+1], (1-x)*0.5),)
        return ctx.hypercomb(h, [n], **kwargs)
    if not ctx.isint(b):
        def h(n, a):
            return (([], [], [-b], [n+1, -b-n], [-n, a+b+n+1], [b+1], (x+1)*0.5),)
        return ctx.hypercomb(h, [n, a], **kwargs)
    # XXX: determine appropriate limit
    return ctx.binomial(n+a,n) * ctx.hyp2f1(-n,1+n+a+b,a+1,(1-x)/2, **kwargs)

@defun_wrapped
def laguerre(ctx, n, a, z, **kwargs):
    # XXX: limits, poles
    #if ctx.isnpint(n):
    #    return 0*(a+z)
    def h(a):
        return (([], [], [a+n+1], [a+1, n+1], [-n], [a+1], z),)
    return ctx.hypercomb(h, [a], **kwargs)

@defun_wrapped
def legendre(ctx, n, x, **kwargs):
    if ctx.isint(n):
        n = int(n)
        # Accuracy near zeros
        if (n + (n < 0)) & 1:
            if not x:
                return x
            mag = ctx.mag(x)
            if mag < -2*ctx.prec-10:
                return x
            if mag < -5:
                ctx.prec += -mag
    return ctx.hyp2f1(-n,n+1,1,(1-x)/2, **kwargs)

@defun
def legenp(ctx, n, m, z, type=2, **kwargs):
    # Legendre function, 1st kind
    n = ctx.convert(n)
    m = ctx.convert(m)
    # Faster
    if not m:
        return ctx.legendre(n, z, **kwargs)
    # TODO: correct evaluation at singularities
    if type == 2:
        def h(n,m):
            g = m*0.5
            T = [1+z, 1-z], [g, -g], [], [1-m], [-n, n+1], [1-m], 0.5*(1-z)
            return (T,)
        return ctx.hypercomb(h, [n,m], **kwargs)
    if type == 3:
        def h(n,m):
            g = m*0.5
            T = [z+1, z-1], [g, -g], [], [1-m], [-n, n+1], [1-m], 0.5*(1-z)
            return (T,)
        return ctx.hypercomb(h, [n,m], **kwargs)
    raise ValueError("requires type=2 or type=3")

@defun
def legenq(ctx, n, m, z, type=2, **kwargs):
    # Legendre function, 2nd kind
    n = ctx.convert(n)
    m = ctx.convert(m)
    z = ctx.convert(z)
    if z in (1, -1):
        #if ctx.isint(m):
        #    return ctx.nan
        #return ctx.inf  # unsigned
        return ctx.nan
    if type == 2:
        def h(n, m):
            cos, sin = ctx.cospi_sinpi(m)
            s = 2 * sin / ctx.pi
            c = cos
            a = 1+z
            b = 1-z
            u = m/2
            w = (1-z)/2
            T1 = [s, c, a, b], [-1, 1, u, -u], [], [1-m], \
                [-n, n+1], [1-m], w
            T2 = [-s, a, b], [-1, -u, u], [n+m+1], [n-m+1, m+1], \
                [-n, n+1], [m+1], w
            return T1, T2
        return ctx.hypercomb(h, [n, m], **kwargs)
    if type == 3:
        # The following is faster when there only is a single series
        # Note: not valid for -1 < z < 0 (?)
        if abs(z) > 1:
            def h(n, m):
                T1 = [ctx.expjpi(m), 2, ctx.pi, z, z-1, z+1], \
                     [1, -n-1, 0.5, -n-m-1, 0.5*m, 0.5*m], \
                     [n+m+1], [n+1.5], \
                     [0.5*(2+n+m), 0.5*(1+n+m)], [n+1.5], z**(-2)
                return [T1]
            return ctx.hypercomb(h, [n, m], **kwargs)
        else:
            # not valid for 1 < z < inf ?
            def h(n, m):
                s = 2 * ctx.sinpi(m) / ctx.pi
                c = ctx.expjpi(m)
                a = 1+z
                b = z-1
                u = m/2
                w = (1-z)/2
                T1 = [s, c, a, b], [-1, 1, u, -u], [], [1-m], \
                    [-n, n+1], [1-m], w
                T2 = [-s, c, a, b], [-1, 1, -u, u], [n+m+1], [n-m+1, m+1], \
                    [-n, n+1], [m+1], w
                return T1, T2
            return ctx.hypercomb(h, [n, m], **kwargs)
    raise ValueError("requires type=2 or type=3")

@defun_wrapped
def chebyt(ctx, n, x, **kwargs):
    if (not x) and ctx.isint(n) and int(ctx._re(n)) % 2 == 1:
        return x * 0
    return ctx.hyp2f1(-n,n,(1,2),(1-x)/2, **kwargs)

@defun_wrapped
def chebyu(ctx, n, x, **kwargs):
    if (not x) and ctx.isint(n) and int(ctx._re(n)) % 2 == 1:
        return x * 0
    return (n+1) * ctx.hyp2f1(-n, n+2, (3,2), (1-x)/2, **kwargs)

@defun
def spherharm(ctx, l, m, theta, phi, **kwargs):
    l = ctx.convert(l)
    m = ctx.convert(m)
    theta = ctx.convert(theta)
    phi = ctx.convert(phi)
    l_isint = ctx.isint(l)
    l_natural = l_isint and l >= 0
    m_isint = ctx.isint(m)
    if l_isint and l < 0 and m_isint:
        return ctx.spherharm(-(l+1), m, theta, phi, **kwargs)
    if theta == 0 and m_isint and m < 0:
        return ctx.zero * 1j
    if l_natural and m_isint:
        if abs(m) > l:
            return ctx.zero * 1j
        # http://functions.wolfram.com/Polynomials/
        #     SphericalHarmonicY/26/01/02/0004/
        def h(l,m):
            absm = abs(m)
            C = [-1, ctx.expj(m*phi),
                 (2*l+1)*ctx.fac(l+absm)/ctx.pi/ctx.fac(l-absm),
                 ctx.sin(theta)**2,
                 ctx.fac(absm), 2]
            P = [0.5*m*(ctx.sign(m)+1), 1, 0.5, 0.5*absm, -1, -absm-1]
            return ((C, P, [], [], [absm-l, l+absm+1], [absm+1],
                ctx.sin(0.5*theta)**2),)
    else:
        # http://functions.wolfram.com/HypergeometricFunctions/
        #     SphericalHarmonicYGeneral/26/01/02/0001/
        def h(l,m):
            if ctx.isnpint(l-m+1) or ctx.isnpint(l+m+1) or ctx.isnpint(1-m):
                return (([0], [-1], [], [], [], [], 0),)
            cos, sin = ctx.cos_sin(0.5*theta)
            C = [0.5*ctx.expj(m*phi), (2*l+1)/ctx.pi,
                 ctx.gamma(l-m+1), ctx.gamma(l+m+1),
                 cos**2, sin**2]
            P = [1, 0.5, 0.5, -0.5, 0.5*m, -0.5*m]
            return ((C, P, [], [1-m], [-l,l+1], [1-m], sin**2),)
    return ctx.hypercomb(h, [l,m], **kwargs)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\simplify\gammasimp.py ===
from sympy.core import Function, S, Mul, Pow, Add
from sympy.core.sorting import ordered, default_sort_key
from sympy.core.function import expand_func
from sympy.core.symbol import Dummy
from sympy.functions import gamma, sqrt, sin
from sympy.polys import factor, cancel
from sympy.utilities.iterables import sift, uniq


def gammasimp(expr):
    r"""
    Simplify expressions with gamma functions.

    Explanation
    ===========

    This function takes as input an expression containing gamma
    functions or functions that can be rewritten in terms of gamma
    functions and tries to minimize the number of those functions and
    reduce the size of their arguments.

    The algorithm works by rewriting all gamma functions as expressions
    involving rising factorials (Pochhammer symbols) and applies
    recurrence relations and other transformations applicable to rising
    factorials, to reduce their arguments, possibly letting the resulting
    rising factorial to cancel. Rising factorials with the second argument
    being an integer are expanded into polynomial forms and finally all
    other rising factorial are rewritten in terms of gamma functions.

    Then the following two steps are performed.

    1. Reduce the number of gammas by applying the reflection theorem
       gamma(x)*gamma(1-x) == pi/sin(pi*x).
    2. Reduce the number of gammas by applying the multiplication theorem
       gamma(x)*gamma(x+1/n)*...*gamma(x+(n-1)/n) == C*gamma(n*x).

    It then reduces the number of prefactors by absorbing them into gammas
    where possible and expands gammas with rational argument.

    All transformation rules can be found (or were derived from) here:

    .. [1] https://functions.wolfram.com/GammaBetaErf/Pochhammer/17/01/02/
    .. [2] https://functions.wolfram.com/GammaBetaErf/Pochhammer/27/01/0005/

    Examples
    ========

    >>> from sympy.simplify import gammasimp
    >>> from sympy import gamma, Symbol
    >>> from sympy.abc import x
    >>> n = Symbol('n', integer = True)

    >>> gammasimp(gamma(x)/gamma(x - 3))
    (x - 3)*(x - 2)*(x - 1)
    >>> gammasimp(gamma(n + 3))
    gamma(n + 3)

    """

    expr = expr.rewrite(gamma)

    # compute_ST will be looking for Functions and we don't want
    # it looking for non-gamma functions: issue 22606
    # so we mask free, non-gamma functions
    f = expr.atoms(Function)
    # take out gammas
    gammas = {i for i in f if isinstance(i, gamma)}
    if not gammas:
        return expr  # avoid side effects like factoring
    f -= gammas
    # keep only those without bound symbols
    f = f & expr.as_dummy().atoms(Function)
    if f:
        dum, fun, simp = zip(*[
            (Dummy(), fi, fi.func(*[
                _gammasimp(a, as_comb=False) for a in fi.args]))
            for fi in ordered(f)])
        d = expr.xreplace(dict(zip(fun, dum)))
        return _gammasimp(d, as_comb=False).xreplace(dict(zip(dum, simp)))

    return _gammasimp(expr, as_comb=False)


def _gammasimp(expr, as_comb):
    """
    Helper function for gammasimp and combsimp.

    Explanation
    ===========

    Simplifies expressions written in terms of gamma function. If
    as_comb is True, it tries to preserve integer arguments. See
    docstring of gammasimp for more information. This was part of
    combsimp() in combsimp.py.
    """
    expr = expr.replace(gamma,
        lambda n: _rf(1, (n - 1).expand()))

    if as_comb:
        expr = expr.replace(_rf,
            lambda a, b: gamma(b + 1))
    else:
        expr = expr.replace(_rf,
            lambda a, b: gamma(a + b)/gamma(a))

    def rule_gamma(expr, level=0):
        """ Simplify products of gamma functions further. """

        if expr.is_Atom:
            return expr

        def gamma_rat(x):
            # helper to simplify ratios of gammas
            was = x.count(gamma)
            xx = x.replace(gamma, lambda n: _rf(1, (n - 1).expand()
                ).replace(_rf, lambda a, b: gamma(a + b)/gamma(a)))
            if xx.count(gamma) < was:
                x = xx
            return x

        def gamma_factor(x):
            # return True if there is a gamma factor in shallow args
            if isinstance(x, gamma):
                return True
            if x.is_Add or x.is_Mul:
                return any(gamma_factor(xi) for xi in x.args)
            if x.is_Pow and (x.exp.is_integer or x.base.is_positive):
                return gamma_factor(x.base)
            return False

        # recursion step
        if level == 0:
            expr = expr.func(*[rule_gamma(x, level + 1) for x in expr.args])
            level += 1

        if not expr.is_Mul:
            return expr

        # non-commutative step
        if level == 1:
            args, nc = expr.args_cnc()
            if not args:
                return expr
            if nc:
                return rule_gamma(Mul._from_args(args), level + 1)*Mul._from_args(nc)
            level += 1

        # pure gamma handling, not factor absorption
        if level == 2:
            T, F = sift(expr.args, gamma_factor, binary=True)
            gamma_ind = Mul(*F)
            d = Mul(*T)

            nd, dd = d.as_numer_denom()
            for ipass in range(2):
                args = list(ordered(Mul.make_args(nd)))
                for i, ni in enumerate(args):
                    if ni.is_Add:
                        ni, dd = Add(*[
                            rule_gamma(gamma_rat(a/dd), level + 1) for a in ni.args]
                            ).as_numer_denom()
                        args[i] = ni
                        if not dd.has(gamma):
                            break
                nd = Mul(*args)
                if ipass ==  0 and not gamma_factor(nd):
                    break
                nd, dd = dd, nd  # now process in reversed order
            expr = gamma_ind*nd/dd
            if not (expr.is_Mul and (gamma_factor(dd) or gamma_factor(nd))):
                return expr
            level += 1

        # iteration until constant
        if level == 3:
            while True:
                was = expr
                expr = rule_gamma(expr, 4)
                if expr == was:
                    return expr

        numer_gammas = []
        denom_gammas = []
        numer_others = []
        denom_others = []
        def explicate(p):
            if p is S.One:
                return None, []
            b, e = p.as_base_exp()
            if e.is_Integer:
                if isinstance(b, gamma):
                    return True, [b.args[0]]*e
                else:
                    return False, [b]*e
            else:
                return False, [p]

        newargs = list(ordered(expr.args))
        while newargs:
            n, d = newargs.pop().as_numer_denom()
            isg, l = explicate(n)
            if isg:
                numer_gammas.extend(l)
            elif isg is False:
                numer_others.extend(l)
            isg, l = explicate(d)
            if isg:
                denom_gammas.extend(l)
            elif isg is False:
                denom_others.extend(l)

        # =========== level 2 work: pure gamma manipulation =========

        if not as_comb:
            # Try to reduce the number of gamma factors by applying the
            # reflection formula gamma(x)*gamma(1-x) = pi/sin(pi*x)
            for gammas, numer, denom in [(
                numer_gammas, numer_others, denom_others),
                    (denom_gammas, denom_others, numer_others)]:
                new = []
                while gammas:
                    g1 = gammas.pop()
                    if g1.is_integer:
                        new.append(g1)
                        continue
                    for i, g2 in enumerate(gammas):
                        n = g1 + g2 - 1
                        if not n.is_Integer:
                            continue
                        numer.append(S.Pi)
                        denom.append(sin(S.Pi*g1))
                        gammas.pop(i)
                        if n > 0:
                            numer.extend(1 - g1 + k for k in range(n))
                        elif n < 0:
                            denom.extend(-g1 - k for k in range(-n))
                        break
                    else:
                        new.append(g1)
                # /!\ updating IN PLACE
                gammas[:] = new

            # Try to reduce the number of gammas by using the duplication
            # theorem to cancel an upper and lower: gamma(2*s)/gamma(s) =
            # 2**(2*s + 1)/(4*sqrt(pi))*gamma(s + 1/2). Although this could
            # be done with higher argument ratios like gamma(3*x)/gamma(x),
            # this would not reduce the number of gammas as in this case.
            for ng, dg, no, do in [(numer_gammas, denom_gammas, numer_others,
                                    denom_others),
                                   (denom_gammas, numer_gammas, denom_others,
                                    numer_others)]:

                while True:
                    for x in ng:
                        for y in dg:
                            n = x - 2*y
                            if n.is_Integer:
                                break
                        else:
                            continue
                        break
                    else:
                        break
                    ng.remove(x)
                    dg.remove(y)
                    if n > 0:
                        no.extend(2*y + k for k in range(n))
                    elif n < 0:
                        do.extend(2*y - 1 - k for k in range(-n))
                    ng.append(y + S.Half)
                    no.append(2**(2*y - 1))
                    do.append(sqrt(S.Pi))

            # Try to reduce the number of gamma factors by applying the
            # multiplication theorem (used when n gammas with args differing
            # by 1/n mod 1 are encountered).
            #
            # run of 2 with args differing by 1/2
            #
            # >>> gammasimp(gamma(x)*gamma(x+S.Half))
            # 2*sqrt(2)*2**(-2*x - 1/2)*sqrt(pi)*gamma(2*x)
            #
            # run of 3 args differing by 1/3 (mod 1)
            #
            # >>> gammasimp(gamma(x)*gamma(x+S(1)/3)*gamma(x+S(2)/3))
            # 6*3**(-3*x - 1/2)*pi*gamma(3*x)
            # >>> gammasimp(gamma(x)*gamma(x+S(1)/3)*gamma(x+S(5)/3))
            # 2*3**(-3*x - 1/2)*pi*(3*x + 2)*gamma(3*x)
            #
            def _run(coeffs):
                # find runs in coeffs such that the difference in terms (mod 1)
                # of t1, t2, ..., tn is 1/n
                u = list(uniq(coeffs))
                for i in range(len(u)):
                    dj = ([((u[j] - u[i]) % 1, j) for j in range(i + 1, len(u))])
                    for one, j in dj:
                        if one.p == 1 and one.q != 1:
                            n = one.q
                            got = [i]
                            get = list(range(1, n))
                            for d, j in dj:
                                m = n*d
                                if m.is_Integer and m in get:
                                    get.remove(m)
                                    got.append(j)
                                    if not get:
                                        break
                            else:
                                continue
                            for i, j in enumerate(got):
                                c = u[j]
                                coeffs.remove(c)
                                got[i] = c
                            return one.q, got[0], got[1:]

            def _mult_thm(gammas, numer, denom):
                # pull off and analyze the leading coefficient from each gamma arg
                # looking for runs in those Rationals

                # expr -> coeff + resid -> rats[resid] = coeff
                rats = {}
                for g in gammas:
                    c, resid = g.as_coeff_Add()
                    rats.setdefault(resid, []).append(c)

                # look for runs in Rationals for each resid
                keys = sorted(rats, key=default_sort_key)
                for resid in keys:
                    coeffs = sorted(rats[resid])
                    new = []
                    while True:
                        run = _run(coeffs)
                        if run is None:
                            break

                        # process the sequence that was found:
                        # 1) convert all the gamma functions to have the right
                        #    argument (could be off by an integer)
                        # 2) append the factors corresponding to the theorem
                        # 3) append the new gamma function

                        n, ui, other = run

                        # (1)
                        for u in other:
                            con = resid + u - 1
                            for k in range(int(u - ui)):
                                numer.append(con - k)

                        con = n*(resid + ui)  # for (2) and (3)

                        # (2)
                        numer.append((2*S.Pi)**(S(n - 1)/2)*
                                     n**(S.Half - con))
                        # (3)
                        new.append(con)

                    # restore resid to coeffs
                    rats[resid] = [resid + c for c in coeffs] + new

                # rebuild the gamma arguments
                g = []
                for resid in keys:
                    g += rats[resid]
                # /!\ updating IN PLACE
                gammas[:] = g

            for l, numer, denom in [(numer_gammas, numer_others, denom_others),
                                    (denom_gammas, denom_others, numer_others)]:
                _mult_thm(l, numer, denom)

        # =========== level >= 2 work: factor absorption =========

        if level >= 2:
            # Try to absorb factors into the gammas: x*gamma(x) -> gamma(x + 1)
            # and gamma(x)/(x - 1) -> gamma(x - 1)
            # This code (in particular repeated calls to find_fuzzy) can be very
            # slow.
            def find_fuzzy(l, x):
                if not l:
                    return
                S1, T1 = compute_ST(x)
                for y in l:
                    S2, T2 = inv[y]
                    if T1 != T2 or (not S1.intersection(S2) and
                                    (S1 != set() or S2 != set())):
                        continue
                    # XXX we want some simplification (e.g. cancel or
                    # simplify) but no matter what it's slow.
                    a = len(cancel(x/y).free_symbols)
                    b = len(x.free_symbols)
                    c = len(y.free_symbols)
                    # TODO is there a better heuristic?
                    if a == 0 and (b > 0 or c > 0):
                        return y

            # We thus try to avoid expensive calls by building the following
            # "invariants": For every factor or gamma function argument
            #   - the set of free symbols S
            #   - the set of functional components T
            # We will only try to absorb if T1==T2 and (S1 intersect S2 != emptyset
            # or S1 == S2 == emptyset)
            inv = {}

            def compute_ST(expr):
                if expr in inv:
                    return inv[expr]
                return (expr.free_symbols, expr.atoms(Function).union(
                        {e.exp for e in expr.atoms(Pow)}))

            def update_ST(expr):
                inv[expr] = compute_ST(expr)
            for expr in numer_gammas + denom_gammas + numer_others + denom_others:
                update_ST(expr)

            for gammas, numer, denom in [(
                numer_gammas, numer_others, denom_others),
                    (denom_gammas, denom_others, numer_others)]:
                new = []
                while gammas:
                    g = gammas.pop()
                    cont = True
                    while cont:
                        cont = False
                        y = find_fuzzy(numer, g)
                        if y is not None:
                            numer.remove(y)
                            if y != g:
                                numer.append(y/g)
                                update_ST(y/g)
                            g += 1
                            cont = True
                        y = find_fuzzy(denom, g - 1)
                        if y is not None:
                            denom.remove(y)
                            if y != g - 1:
                                numer.append((g - 1)/y)
                                update_ST((g - 1)/y)
                            g -= 1
                            cont = True
                    new.append(g)
                # /!\ updating IN PLACE
                gammas[:] = new

        # =========== rebuild expr ==================================

        return Mul(*[gamma(g) for g in numer_gammas]) \
            / Mul(*[gamma(g) for g in denom_gammas]) \
            * Mul(*numer_others) / Mul(*denom_others)

    was = factor(expr)
    # (for some reason we cannot use Basic.replace in this case)
    expr = rule_gamma(was)
    if expr != was:
        expr = factor(expr)

    expr = expr.replace(gamma,
        lambda n: expand_func(gamma(n)) if n.is_Rational else gamma(n))

    return expr


class _rf(Function):
    @classmethod
    def eval(cls, a, b):
        if b.is_Integer:
            if not b:
                return S.One

            n = int(b)

            if n > 0:
                return Mul(*[a + i for i in range(n)])
            elif n < 0:
                return 1/Mul(*[a - i for i in range(1, -n + 1)])
        else:
            if b.is_Add:
                c, _b = b.as_coeff_Add()

                if c.is_Integer:
                    if c > 0:
                        return _rf(a, _b)*_rf(a + _b, c)
                    elif c < 0:
                        return _rf(a, _b)/_rf(a + _b + c, -c)

            if a.is_Add:
                c, _a = a.as_coeff_Add()

                if c.is_Integer:
                    if c > 0:
                        return _rf(_a, b)*_rf(_a + b, c)/_rf(_a, c)
                    elif c < 0:
                        return _rf(_a, b)*_rf(_a + c, -c)/_rf(_a + b + c, -c)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\fusion_fastgelu.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from logging import getLogger

from fusion_base import Fusion
from onnx import helper
from onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionFastGelu(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "FastGelu", "Tanh")

    def fuse(self, tanh_node, input_name_to_nodes: dict, output_name_to_node: dict):
        if self.fuse_1(tanh_node, input_name_to_nodes, output_name_to_node):
            return

        if self.fuse_2(tanh_node, input_name_to_nodes, output_name_to_node):
            return

        if self.fuse_3(tanh_node, input_name_to_nodes, output_name_to_node):
            return

        if self.fuse_4(tanh_node, input_name_to_nodes, output_name_to_node):
            return

    def fuse_1(self, tanh_node, input_name_to_nodes, output_name_to_node) -> bool | None:
        """
        Fuse Gelu with tanh into one node:
              +---------------------------+
              |                           |
              |                           v
            [root] --> Pow --> Mul -----> Add  --> Mul --> Tanh --> Add --> Mul
              |       (Y=3)   (B=0.0447...)       (B=0.7978...)    (B=1)     ^
              |                                                              |
              +------> Mul(B=0.5)--------------------------------------------+
        Note that constant input for Add and Mul could be first or second input: like either A=0.5 or B=0.5 is fine.
        """
        if tanh_node.output[0] not in input_name_to_nodes:
            return
        children = input_name_to_nodes[tanh_node.output[0]]
        if len(children) != 1 or children[0].op_type != "Add":
            return
        add_after_tanh = children[0]

        if not self.model.has_constant_input(add_after_tanh, 1.0):
            return

        if add_after_tanh.output[0] not in input_name_to_nodes:
            return
        children = input_name_to_nodes[add_after_tanh.output[0]]
        if len(children) != 1 or children[0].op_type != "Mul":
            return
        mul_after_tanh = children[0]

        mul_half = self.model.match_parent(mul_after_tanh, "Mul", None, output_name_to_node)
        if mul_half is None:
            return

        i = self.model.find_constant_input(mul_half, 0.5)
        if i < 0:
            return

        root_input = mul_half.input[0 if i == 1 else 1]

        # root_node could be None when root_input is graph input
        root_node = self.model.get_parent(mul_half, 0 if i == 1 else 1, output_name_to_node)

        mul_before_tanh = self.model.match_parent(tanh_node, "Mul", 0, output_name_to_node)
        if mul_before_tanh is None:
            return

        i = self.model.find_constant_input(mul_before_tanh, 0.7978, delta=0.0001)
        if i < 0:
            return

        add_before_tanh = self.model.match_parent(mul_before_tanh, "Add", 0 if i == 1 else 1, output_name_to_node)
        if add_before_tanh is None:
            return

        mul_after_pow = self.model.match_parent(
            add_before_tanh,
            "Mul",
            None,
            output_name_to_node,
            exclude=[root_node] if root_node else [],
        )
        if mul_after_pow is None:
            return

        i = self.model.find_constant_input(mul_after_pow, 0.0447, delta=0.0001)
        if i < 0:
            return

        pow = self.model.match_parent(mul_after_pow, "Pow", 0 if i == 1 else 1, output_name_to_node)
        if pow is None:
            return

        if not self.model.has_constant_input(pow, 3.0):
            return

        if pow.input[0] != root_input:
            return

        subgraph_nodes = [
            mul_after_tanh,
            mul_half,
            add_after_tanh,
            tanh_node,
            mul_before_tanh,
            add_before_tanh,
            mul_after_pow,
            pow,
        ]
        if not self.model.is_safe_to_fuse_nodes(
            subgraph_nodes,
            [mul_after_tanh.output[0]],
            input_name_to_nodes,
            output_name_to_node,
        ):
            return

        self.nodes_to_remove.extend(subgraph_nodes)
        fused_node = helper.make_node(
            "FastGelu",
            inputs=[root_input],
            outputs=mul_after_tanh.output,
            name=self.model.create_node_name("FastGelu"),
        )
        fused_node.domain = "com.microsoft"
        self.nodes_to_add.append(fused_node)
        self.node_name_to_graph_name[fused_node.name] = self.this_graph_name
        return True

    def fuse_2(self, tanh_node, input_name_to_nodes: dict, output_name_to_node: dict) -> bool | None:
        """
        This pattern is from Tensorflow model.
        Fuse Gelu with tanh into one node:
              +---------------------------+
              |                           |
              |                           v
            [root] --> Pow --> Mul -----> Add  --> Mul --> Tanh --> Add --> Mul(B=0.5)-->Mul-->
              |       (Y=3)   (B=0.0447...)       (B=0.7978...)    (B=1)                  ^
              |                                                                           |
              +---------------------------------------------------------------------------+
        Note that constant input for Add and Mul could be first or second input: like either A=0.5 or B=0.5 is fine.
        """
        if tanh_node.output[0] not in input_name_to_nodes:
            return
        children = input_name_to_nodes[tanh_node.output[0]]
        if len(children) != 1 or children[0].op_type != "Add":
            return
        add_after_tanh = children[0]

        if not self.model.has_constant_input(add_after_tanh, 1.0):
            return

        if add_after_tanh.output[0] not in input_name_to_nodes:
            return
        children = input_name_to_nodes[add_after_tanh.output[0]]
        if len(children) != 1 or children[0].op_type != "Mul":
            return
        mul_half = children[0]

        i = self.model.find_constant_input(mul_half, 0.5)
        if i < 0:
            return

        if mul_half.output[0] not in input_name_to_nodes:
            return
        children = input_name_to_nodes[mul_half.output[0]]
        if len(children) != 1 or children[0].op_type != "Mul":
            return
        mul_after_mul_half = children[0]

        # root_node could be None when root_input is graph input
        root_node = self.model.get_parent(
            mul_after_mul_half,
            0 if mul_after_mul_half.input[1] == mul_half.output[0] else 1,
            output_name_to_node,
        )

        mul_before_tanh = self.model.match_parent(tanh_node, "Mul", 0, output_name_to_node)
        if mul_before_tanh is None:
            return

        i = self.model.find_constant_input(mul_before_tanh, 0.7978, delta=0.0001)
        if i < 0:
            return

        add_before_tanh = self.model.match_parent(mul_before_tanh, "Add", 0 if i == 1 else 1, output_name_to_node)
        if add_before_tanh is None:
            return

        mul_after_pow = self.model.match_parent(
            add_before_tanh,
            "Mul",
            None,
            output_name_to_node,
            exclude=[root_node] if root_node else [],
        )
        if mul_after_pow is None:
            return

        i = self.model.find_constant_input(mul_after_pow, 0.0447, delta=0.0001)
        if i < 0:
            return

        pow = self.model.match_parent(mul_after_pow, "Pow", 0 if i == 1 else 1, output_name_to_node)
        if pow is None:
            return

        if not self.model.has_constant_input(pow, 3.0):
            return

        root_input = mul_after_mul_half.input[0 if mul_after_mul_half.input[1] == mul_half.output[0] else 1]

        if pow.input[0] != root_input:
            return

        subgraph_nodes = [
            mul_after_mul_half,
            mul_half,
            add_after_tanh,
            tanh_node,
            mul_before_tanh,
            add_before_tanh,
            mul_after_pow,
            pow,
        ]
        if not self.model.is_safe_to_fuse_nodes(
            subgraph_nodes,
            [mul_after_mul_half.output[0]],
            input_name_to_nodes,
            output_name_to_node,
        ):
            return

        self.nodes_to_remove.extend(subgraph_nodes)
        fused_node = helper.make_node(
            "FastGelu",
            inputs=[root_input],
            outputs=mul_after_mul_half.output,
            name=self.model.create_node_name("FastGelu"),
        )
        fused_node.domain = "com.microsoft"
        self.nodes_to_add.append(fused_node)
        self.node_name_to_graph_name[fused_node.name] = self.this_graph_name
        return True

    def fuse_3(self, tanh_node, input_name_to_nodes: dict, output_name_to_node: dict) -> bool | None:
        """
        OpenAI's gelu implementation, also used in Megatron:
           Gelu(x) = x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1.0 + 0.044715 * x * x)))

        Fuse subgraph into a FastGelu node:
            +------------ Mul (B=0.79788456) -------------------+
            |                                                   |
            +-------------------------------+                   |
            |                               |                   |
            |                               v                   v
          [root] --> Mul (B=0.044715) --> Mul --> Add(B=1) --> Mul --> Tanh --> Add(B=1) --> Mul-->
            |                                                                                 ^
            |                                                                                 |
            +-----------> Mul (B=0.5) --------------------------------------------------------+
        """
        if tanh_node.output[0] not in input_name_to_nodes:
            return

        children = input_name_to_nodes[tanh_node.output[0]]
        if len(children) != 1 or children[0].op_type != "Add":
            return
        add_after_tanh = children[0]

        if not self.model.has_constant_input(add_after_tanh, 1.0):
            return

        if add_after_tanh.output[0] not in input_name_to_nodes:
            return
        children = input_name_to_nodes[add_after_tanh.output[0]]
        if len(children) != 1 or children[0].op_type != "Mul":
            return
        mul_last = children[0]

        mul_half = self.model.match_parent(mul_last, "Mul", None, output_name_to_node)
        if mul_half is None:
            return

        i = self.model.find_constant_input(mul_half, 0.5)
        if i < 0:
            return

        root_input = mul_half.input[0 if i == 1 else 1]

        mul_before_tanh = self.model.match_parent(tanh_node, "Mul", 0, output_name_to_node)
        if mul_before_tanh is None:
            return

        add_1 = self.model.match_parent(mul_before_tanh, "Add", None, output_name_to_node)
        if add_1 is None:
            return
        j = self.model.find_constant_input(add_1, 1.0)
        if j < 0:
            return

        mul_7978 = self.model.match_parent(mul_before_tanh, "Mul", None, output_name_to_node)
        if mul_7978 is None:
            return
        k = self.model.find_constant_input(mul_7978, 0.7978, delta=0.0001)
        if k < 0:
            return
        if mul_7978.input[0 if k == 1 else 1] != root_input:
            return

        mul_before_add_1 = self.model.match_parent(add_1, "Mul", 0 if j == 1 else 1, output_name_to_node)
        if mul_before_add_1 is None:
            return

        if mul_before_add_1.input[0] == root_input:
            another = 1
        elif mul_before_add_1.input[1] == root_input:
            another = 0
        else:
            return

        mul_0447 = self.model.match_parent(mul_before_add_1, "Mul", another, output_name_to_node)
        if mul_0447 is None:
            return
        m = self.model.find_constant_input(mul_0447, 0.0447, delta=0.0001)
        if m < 0:
            return

        if mul_0447.input[0 if m == 1 else 1] != root_input:
            return

        subgraph_nodes = [
            mul_0447,
            mul_before_add_1,
            add_1,
            mul_before_tanh,
            tanh_node,
            add_after_tanh,
            mul_7978,
            mul_half,
            mul_last,
        ]
        if not self.model.is_safe_to_fuse_nodes(
            subgraph_nodes,
            [mul_last.output[0]],
            input_name_to_nodes,
            output_name_to_node,
        ):
            return

        self.nodes_to_remove.extend(subgraph_nodes)
        fused_node = helper.make_node(
            "FastGelu",
            inputs=[root_input],
            outputs=mul_last.output,
            name=self.model.create_node_name("FastGelu"),
        )
        fused_node.domain = "com.microsoft"
        self.nodes_to_add.append(fused_node)
        self.node_name_to_graph_name[fused_node.name] = self.this_graph_name
        return True

    def fuse_4(self, tanh_node, input_name_to_nodes: dict, output_name_to_node: dict) -> bool | None:
        """
        PyTorch's gelu implementation with tanh approximation:
           Gelu(x) = 0.5 * x * (1 + torch.tanh(0.7978845834732056 * (x + 0.044714998453855515 * x * x * x)))

        Fuse Gelu with tanh into one node:
              +-----------------+------------------+
              |                 |                  |
              |                 v                  v
            [root] ==> Mul --> Mul --> Mul -----> Add  --> Mul --> Tanh --> Add -----> Mul --> Mul -->
              |                       (A=0.0447)          (A=0.7978)        (A=1)       ^     (A=0.5)
              |                                                                         |
              +-------------------------------------------------------------------------+
        Note that constant input for Add and Mul could be first or second input.
        """
        if tanh_node.output[0] not in input_name_to_nodes:
            return

        children = input_name_to_nodes[tanh_node.output[0]]
        if len(children) != 1 or children[0].op_type != "Add":
            return
        add_after_tanh = children[0]

        if not self.model.has_constant_input(add_after_tanh, 1.0):
            return

        if add_after_tanh.output[0] not in input_name_to_nodes:
            return
        children = input_name_to_nodes[add_after_tanh.output[0]]
        if len(children) != 1 or children[0].op_type != "Mul":
            return
        mul_after_tanh = children[0]

        if mul_after_tanh.output[0] not in input_name_to_nodes:
            return
        children = input_name_to_nodes[mul_after_tanh.output[0]]
        if len(children) != 1 or children[0].op_type != "Mul":
            return
        mul_half = children[0]

        if not self.model.has_constant_input(mul_half, 0.5):
            return

        root_input = mul_after_tanh.input[0 if mul_after_tanh.input[1] == add_after_tanh.output[0] else 1]

        mul_before_tanh = self.model.match_parent(tanh_node, "Mul", 0, output_name_to_node)
        if mul_before_tanh is None:
            return

        k = self.model.find_constant_input(mul_before_tanh, 0.7978, delta=0.01)
        if k < 0:
            return

        add_before_tanh = self.model.match_parent(mul_before_tanh, "Add", 0 if k == 1 else 1, output_name_to_node)
        if add_before_tanh is None:
            return

        if add_before_tanh.input[0] == root_input:
            another = 1
        elif add_before_tanh.input[1] == root_input:
            another = 0
        else:
            return

        mul_after_pow = self.model.match_parent(add_before_tanh, "Mul", another, output_name_to_node)
        if mul_after_pow is None:
            return

        m = self.model.find_constant_input(mul_after_pow, 0.0447, delta=0.01)
        if m < 0:
            return

        mul_cubed = self.model.match_parent(mul_after_pow, "Mul", 0 if m == 1 else 1, output_name_to_node)
        if mul_cubed is None:
            return

        if mul_cubed.input[0] == root_input:
            another = 1
        elif mul_cubed.input[1] == root_input:
            another = 0
        else:
            return

        mul_squared = self.model.match_parent(mul_cubed, "Mul", another, output_name_to_node)
        if mul_squared is None:
            return

        if mul_squared.input[0] != root_input or mul_squared.input[1] != root_input:
            return

        subgraph_nodes = [
            mul_squared,
            mul_cubed,
            mul_after_pow,
            add_before_tanh,
            mul_before_tanh,
            tanh_node,
            add_after_tanh,
            mul_after_tanh,
            mul_half,
        ]

        if not self.model.is_safe_to_fuse_nodes(
            subgraph_nodes,
            [mul_half.output[0]],
            input_name_to_nodes,
            output_name_to_node,
        ):
            return

        self.nodes_to_remove.extend(subgraph_nodes)
        fused_node = helper.make_node(
            "FastGelu",
            inputs=[root_input],
            outputs=mul_half.output,
            name=self.model.create_node_name("FastGelu"),
        )
        fused_node.domain = "com.microsoft"
        self.nodes_to_add.append(fused_node)
        self.node_name_to_graph_name[fused_node.name] = self.this_graph_name
        self.increase_counter("FastGelu")
        return True

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\plotting\_matplotlib\tools.py ===
# being a bit too dynamic
from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING
import warnings

from matplotlib import ticker
import matplotlib.table
import numpy as np

from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCIndex,
    ABCSeries,
)

if TYPE_CHECKING:
    from collections.abc import (
        Iterable,
        Sequence,
    )

    from matplotlib.axes import Axes
    from matplotlib.axis import Axis
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.table import Table

    from pandas import (
        DataFrame,
        Series,
    )


def do_adjust_figure(fig: Figure) -> bool:
    """Whether fig has constrained_layout enabled."""
    if not hasattr(fig, "get_constrained_layout"):
        return False
    return not fig.get_constrained_layout()


def maybe_adjust_figure(fig: Figure, *args, **kwargs) -> None:
    """Call fig.subplots_adjust unless fig has constrained_layout enabled."""
    if do_adjust_figure(fig):
        fig.subplots_adjust(*args, **kwargs)


def format_date_labels(ax: Axes, rot) -> None:
    # mini version of autofmt_xdate
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
        label.set_rotation(rot)
    fig = ax.get_figure()
    if fig is not None:
        # should always be a Figure but can technically be None
        maybe_adjust_figure(fig, bottom=0.2)  # type: ignore[arg-type]


def table(
    ax, data: DataFrame | Series, rowLabels=None, colLabels=None, **kwargs
) -> Table:
    if isinstance(data, ABCSeries):
        data = data.to_frame()
    elif isinstance(data, ABCDataFrame):
        pass
    else:
        raise ValueError("Input data must be DataFrame or Series")

    if rowLabels is None:
        rowLabels = data.index

    if colLabels is None:
        colLabels = data.columns

    cellText = data.values

    # error: Argument "cellText" to "table" has incompatible type "ndarray[Any,
    # Any]"; expected "Sequence[Sequence[str]] | None"
    return matplotlib.table.table(
        ax,
        cellText=cellText,  # type: ignore[arg-type]
        rowLabels=rowLabels,
        colLabels=colLabels,
        **kwargs,
    )


def _get_layout(
    nplots: int,
    layout: tuple[int, int] | None = None,
    layout_type: str = "box",
) -> tuple[int, int]:
    if layout is not None:
        if not isinstance(layout, (tuple, list)) or len(layout) != 2:
            raise ValueError("Layout must be a tuple of (rows, columns)")

        nrows, ncols = layout

        if nrows == -1 and ncols > 0:
            layout = nrows, ncols = (ceil(nplots / ncols), ncols)
        elif ncols == -1 and nrows > 0:
            layout = nrows, ncols = (nrows, ceil(nplots / nrows))
        elif ncols <= 0 and nrows <= 0:
            msg = "At least one dimension of layout must be positive"
            raise ValueError(msg)

        if nrows * ncols < nplots:
            raise ValueError(
                f"Layout of {nrows}x{ncols} must be larger than required size {nplots}"
            )

        return layout

    if layout_type == "single":
        return (1, 1)
    elif layout_type == "horizontal":
        return (1, nplots)
    elif layout_type == "vertical":
        return (nplots, 1)

    layouts = {1: (1, 1), 2: (1, 2), 3: (2, 2), 4: (2, 2)}
    try:
        return layouts[nplots]
    except KeyError:
        k = 1
        while k**2 < nplots:
            k += 1

        if (k - 1) * k >= nplots:
            return k, (k - 1)
        else:
            return k, k


# copied from matplotlib/pyplot.py and modified for pandas.plotting


def create_subplots(
    naxes: int,
    sharex: bool = False,
    sharey: bool = False,
    squeeze: bool = True,
    subplot_kw=None,
    ax=None,
    layout=None,
    layout_type: str = "box",
    **fig_kw,
):
    """
    Create a figure with a set of subplots already made.

    This utility wrapper makes it convenient to create common layouts of
    subplots, including the enclosing figure object, in a single call.

    Parameters
    ----------
    naxes : int
      Number of required axes. Exceeded axes are set invisible. Default is
      nrows * ncols.

    sharex : bool
      If True, the X axis will be shared amongst all subplots.

    sharey : bool
      If True, the Y axis will be shared amongst all subplots.

    squeeze : bool

      If True, extra dimensions are squeezed out from the returned axis object:
        - if only one subplot is constructed (nrows=ncols=1), the resulting
        single Axis object is returned as a scalar.
        - for Nx1 or 1xN subplots, the returned object is a 1-d numpy object
        array of Axis objects are returned as numpy 1-d arrays.
        - for NxM subplots with N>1 and M>1 are returned as a 2d array.

      If False, no squeezing is done: the returned axis object is always
      a 2-d array containing Axis instances, even if it ends up being 1x1.

    subplot_kw : dict
      Dict with keywords passed to the add_subplot() call used to create each
      subplots.

    ax : Matplotlib axis object, optional

    layout : tuple
      Number of rows and columns of the subplot grid.
      If not specified, calculated from naxes and layout_type

    layout_type : {'box', 'horizontal', 'vertical'}, default 'box'
      Specify how to layout the subplot grid.

    fig_kw : Other keyword arguments to be passed to the figure() call.
        Note that all keywords not recognized above will be
        automatically included here.

    Returns
    -------
    fig, ax : tuple
      - fig is the Matplotlib Figure object
      - ax can be either a single axis object or an array of axis objects if
      more than one subplot was created.  The dimensions of the resulting array
      can be controlled with the squeeze keyword, see above.

    Examples
    --------
    x = np.linspace(0, 2*np.pi, 400)
    y = np.sin(x**2)

    # Just a figure and one subplot
    f, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title('Simple plot')

    # Two subplots, unpack the output array immediately
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.plot(x, y)
    ax1.set_title('Sharing Y axis')
    ax2.scatter(x, y)

    # Four polar axes
    plt.subplots(2, 2, subplot_kw=dict(polar=True))
    """
    import matplotlib.pyplot as plt

    if subplot_kw is None:
        subplot_kw = {}

    if ax is None:
        fig = plt.figure(**fig_kw)
    else:
        if is_list_like(ax):
            if squeeze:
                ax = flatten_axes(ax)
            if layout is not None:
                warnings.warn(
                    "When passing multiple axes, layout keyword is ignored.",
                    UserWarning,
                    stacklevel=find_stack_level(),
                )
            if sharex or sharey:
                warnings.warn(
                    "When passing multiple axes, sharex and sharey "
                    "are ignored. These settings must be specified when creating axes.",
                    UserWarning,
                    stacklevel=find_stack_level(),
                )
            if ax.size == naxes:
                fig = ax.flat[0].get_figure()
                return fig, ax
            else:
                raise ValueError(
                    f"The number of passed axes must be {naxes}, the "
                    "same as the output plot"
                )

        fig = ax.get_figure()
        # if ax is passed and a number of subplots is 1, return ax as it is
        if naxes == 1:
            if squeeze:
                return fig, ax
            else:
                return fig, flatten_axes(ax)
        else:
            warnings.warn(
                "To output multiple subplots, the figure containing "
                "the passed axes is being cleared.",
                UserWarning,
                stacklevel=find_stack_level(),
            )
            fig.clear()

    nrows, ncols = _get_layout(naxes, layout=layout, layout_type=layout_type)
    nplots = nrows * ncols

    # Create empty object array to hold all axes.  It's easiest to make it 1-d
    # so we can just append subplots upon creation, and then
    axarr = np.empty(nplots, dtype=object)

    # Create first subplot separately, so we can share it if requested
    ax0 = fig.add_subplot(nrows, ncols, 1, **subplot_kw)

    if sharex:
        subplot_kw["sharex"] = ax0
    if sharey:
        subplot_kw["sharey"] = ax0
    axarr[0] = ax0

    # Note off-by-one counting because add_subplot uses the MATLAB 1-based
    # convention.
    for i in range(1, nplots):
        kwds = subplot_kw.copy()
        # Set sharex and sharey to None for blank/dummy axes, these can
        # interfere with proper axis limits on the visible axes if
        # they share axes e.g. issue #7528
        if i >= naxes:
            kwds["sharex"] = None
            kwds["sharey"] = None
        ax = fig.add_subplot(nrows, ncols, i + 1, **kwds)
        axarr[i] = ax

    if naxes != nplots:
        for ax in axarr[naxes:]:
            ax.set_visible(False)

    handle_shared_axes(axarr, nplots, naxes, nrows, ncols, sharex, sharey)

    if squeeze:
        # Reshape the array to have the final desired dimension (nrow,ncol),
        # though discarding unneeded dimensions that equal 1.  If we only have
        # one subplot, just return it instead of a 1-element array.
        if nplots == 1:
            axes = axarr[0]
        else:
            axes = axarr.reshape(nrows, ncols).squeeze()
    else:
        # returned axis array will be always 2-d, even if nrows=ncols=1
        axes = axarr.reshape(nrows, ncols)

    return fig, axes


def _remove_labels_from_axis(axis: Axis) -> None:
    for t in axis.get_majorticklabels():
        t.set_visible(False)

    # set_visible will not be effective if
    # minor axis has NullLocator and NullFormatter (default)
    if isinstance(axis.get_minor_locator(), ticker.NullLocator):
        axis.set_minor_locator(ticker.AutoLocator())
    if isinstance(axis.get_minor_formatter(), ticker.NullFormatter):
        axis.set_minor_formatter(ticker.FormatStrFormatter(""))
    for t in axis.get_minorticklabels():
        t.set_visible(False)

    axis.get_label().set_visible(False)


def _has_externally_shared_axis(ax1: Axes, compare_axis: str) -> bool:
    """
    Return whether an axis is externally shared.

    Parameters
    ----------
    ax1 : matplotlib.axes.Axes
        Axis to query.
    compare_axis : str
        `"x"` or `"y"` according to whether the X-axis or Y-axis is being
        compared.

    Returns
    -------
    bool
        `True` if the axis is externally shared. Otherwise `False`.

    Notes
    -----
    If two axes with different positions are sharing an axis, they can be
    referred to as *externally* sharing the common axis.

    If two axes sharing an axis also have the same position, they can be
    referred to as *internally* sharing the common axis (a.k.a twinning).

    _handle_shared_axes() is only interested in axes externally sharing an
    axis, regardless of whether either of the axes is also internally sharing
    with a third axis.
    """
    if compare_axis == "x":
        axes = ax1.get_shared_x_axes()
    elif compare_axis == "y":
        axes = ax1.get_shared_y_axes()
    else:
        raise ValueError(
            "_has_externally_shared_axis() needs 'x' or 'y' as a second parameter"
        )

    axes_siblings = axes.get_siblings(ax1)

    # Retain ax1 and any of its siblings which aren't in the same position as it
    ax1_points = ax1.get_position().get_points()

    for ax2 in axes_siblings:
        if not np.array_equal(ax1_points, ax2.get_position().get_points()):
            return True

    return False


def handle_shared_axes(
    axarr: Iterable[Axes],
    nplots: int,
    naxes: int,
    nrows: int,
    ncols: int,
    sharex: bool,
    sharey: bool,
) -> None:
    if nplots > 1:
        row_num = lambda x: x.get_subplotspec().rowspan.start
        col_num = lambda x: x.get_subplotspec().colspan.start

        is_first_col = lambda x: x.get_subplotspec().is_first_col()

        if nrows > 1:
            try:
                # first find out the ax layout,
                # so that we can correctly handle 'gaps"
                layout = np.zeros((nrows + 1, ncols + 1), dtype=np.bool_)
                for ax in axarr:
                    layout[row_num(ax), col_num(ax)] = ax.get_visible()

                for ax in axarr:
                    # only the last row of subplots should get x labels -> all
                    # other off layout handles the case that the subplot is
                    # the last in the column, because below is no subplot/gap.
                    if not layout[row_num(ax) + 1, col_num(ax)]:
                        continue
                    if sharex or _has_externally_shared_axis(ax, "x"):
                        _remove_labels_from_axis(ax.xaxis)

            except IndexError:
                # if gridspec is used, ax.rowNum and ax.colNum may different
                # from layout shape. in this case, use last_row logic
                is_last_row = lambda x: x.get_subplotspec().is_last_row()
                for ax in axarr:
                    if is_last_row(ax):
                        continue
                    if sharex or _has_externally_shared_axis(ax, "x"):
                        _remove_labels_from_axis(ax.xaxis)

        if ncols > 1:
            for ax in axarr:
                # only the first column should get y labels -> set all other to
                # off as we only have labels in the first column and we always
                # have a subplot there, we can skip the layout test
                if is_first_col(ax):
                    continue
                if sharey or _has_externally_shared_axis(ax, "y"):
                    _remove_labels_from_axis(ax.yaxis)


def flatten_axes(axes: Axes | Sequence[Axes]) -> np.ndarray:
    if not is_list_like(axes):
        return np.array([axes])
    elif isinstance(axes, (np.ndarray, ABCIndex)):
        return np.asarray(axes).ravel()
    return np.array(axes)


def set_ticks_props(
    axes: Axes | Sequence[Axes],
    xlabelsize: int | None = None,
    xrot=None,
    ylabelsize: int | None = None,
    yrot=None,
):
    import matplotlib.pyplot as plt

    for ax in flatten_axes(axes):
        if xlabelsize is not None:
            plt.setp(ax.get_xticklabels(), fontsize=xlabelsize)
        if xrot is not None:
            plt.setp(ax.get_xticklabels(), rotation=xrot)
        if ylabelsize is not None:
            plt.setp(ax.get_yticklabels(), fontsize=ylabelsize)
        if yrot is not None:
            plt.setp(ax.get_yticklabels(), rotation=yrot)
    return axes


def get_all_lines(ax: Axes) -> list[Line2D]:
    lines = ax.get_lines()

    if hasattr(ax, "right_ax"):
        lines += ax.right_ax.get_lines()

    if hasattr(ax, "left_ax"):
        lines += ax.left_ax.get_lines()

    return lines


def get_xlim(lines: Iterable[Line2D]) -> tuple[float, float]:
    left, right = np.inf, -np.inf
    for line in lines:
        x = line.get_xdata(orig=False)
        left = min(np.nanmin(x), left)
        right = max(np.nanmax(x), right)
    return left, right

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\bs4\builder\_lxml.py ===
# encoding: utf-8
from __future__ import annotations

# Use of this source code is governed by the MIT license.
__license__ = "MIT"

__all__ = [
    "LXMLTreeBuilderForXML",
    "LXMLTreeBuilder",
]


from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TYPE_CHECKING,
    Union,
)
from typing_extensions import TypeAlias

from io import BytesIO
from io import StringIO
from lxml import etree
from bs4.element import (
    AttributeDict,
    XMLAttributeDict,
    Comment,
    Doctype,
    NamespacedAttribute,
    ProcessingInstruction,
    XMLProcessingInstruction,
)
from bs4.builder import (
    DetectsXMLParsedAsHTML,
    FAST,
    HTML,
    HTMLTreeBuilder,
    PERMISSIVE,
    TreeBuilder,
    XML,
)
from bs4.dammit import EncodingDetector
from bs4.exceptions import ParserRejectedMarkup

if TYPE_CHECKING:
    from bs4._typing import (
        _Encoding,
        _Encodings,
        _NamespacePrefix,
        _NamespaceURL,
        _NamespaceMapping,
        _InvertedNamespaceMapping,
        _RawMarkup,
    )
    from bs4 import BeautifulSoup

LXML: str = "lxml"


def _invert(d: dict[Any, Any]) -> dict[Any, Any]:
    "Invert a dictionary."
    return dict((v, k) for k, v in list(d.items()))


_LXMLParser: TypeAlias = Union[etree.XMLParser, etree.HTMLParser]
_ParserOrParserClass: TypeAlias = Union[
    _LXMLParser, Type[etree.XMLParser], Type[etree.HTMLParser]
]


class LXMLTreeBuilderForXML(TreeBuilder):
    DEFAULT_PARSER_CLASS: Type[etree.XMLParser] = etree.XMLParser

    is_xml: bool = True

    processing_instruction_class: Type[ProcessingInstruction]

    NAME: str = "lxml-xml"
    ALTERNATE_NAMES: Iterable[str] = ["xml"]

    # Well, it's permissive by XML parser standards.
    features: Iterable[str] = [NAME, LXML, XML, FAST, PERMISSIVE]

    CHUNK_SIZE: int = 512

    # This namespace mapping is specified in the XML Namespace
    # standard.
    DEFAULT_NSMAPS: _NamespaceMapping = dict(xml="http://www.w3.org/XML/1998/namespace")

    DEFAULT_NSMAPS_INVERTED: _InvertedNamespaceMapping = _invert(DEFAULT_NSMAPS)

    nsmaps: List[Optional[_InvertedNamespaceMapping]]
    empty_element_tags: Set[str]
    parser: Any
    _default_parser: Optional[etree.XMLParser]

    # NOTE: If we parsed Element objects and looked at .sourceline,
    # we'd be able to see the line numbers from the original document.
    # But instead we build an XMLParser or HTMLParser object to serve
    # as the target of parse messages, and those messages don't include
    # line numbers.
    # See: https://bugs.launchpad.net/lxml/+bug/1846906

    def initialize_soup(self, soup: BeautifulSoup) -> None:
        """Let the BeautifulSoup object know about the standard namespace
        mapping.

        :param soup: A `BeautifulSoup`.
        """
        # Beyond this point, self.soup is set, so we can assume (and
        # assert) it's not None whenever necessary.
        super(LXMLTreeBuilderForXML, self).initialize_soup(soup)
        self._register_namespaces(self.DEFAULT_NSMAPS)

    def _register_namespaces(self, mapping: Dict[str, str]) -> None:
        """Let the BeautifulSoup object know about namespaces encountered
        while parsing the document.

        This might be useful later on when creating CSS selectors.

        This will track (almost) all namespaces, even ones that were
        only in scope for part of the document. If two namespaces have
        the same prefix, only the first one encountered will be
        tracked. Un-prefixed namespaces are not tracked.

        :param mapping: A dictionary mapping namespace prefixes to URIs.
        """
        assert self.soup is not None
        for key, value in list(mapping.items()):
            # This is 'if key' and not 'if key is not None' because we
            # don't track un-prefixed namespaces. Soupselect will
            # treat an un-prefixed namespace as the default, which
            # causes confusion in some cases.
            if key and key not in self.soup._namespaces:
                # Let the BeautifulSoup object know about a new namespace.
                # If there are multiple namespaces defined with the same
                # prefix, the first one in the document takes precedence.
                self.soup._namespaces[key] = value

    def default_parser(self, encoding: Optional[_Encoding]) -> _ParserOrParserClass:
        """Find the default parser for the given encoding.

        :return: Either a parser object or a class, which
          will be instantiated with default arguments.
        """
        if self._default_parser is not None:
            return self._default_parser
        return self.DEFAULT_PARSER_CLASS(target=self, recover=True, encoding=encoding)

    def parser_for(self, encoding: Optional[_Encoding]) -> _LXMLParser:
        """Instantiate an appropriate parser for the given encoding.

        :param encoding: A string.
        :return: A parser object such as an `etree.XMLParser`.
        """
        # Use the default parser.
        parser = self.default_parser(encoding)

        if callable(parser):
            # Instantiate the parser with default arguments
            parser = parser(target=self, recover=True, encoding=encoding)
        return parser

    def __init__(
        self,
        parser: Optional[etree.XMLParser] = None,
        empty_element_tags: Optional[Set[str]] = None,
        **kwargs: Any,
    ):
        # TODO: Issue a warning if parser is present but not a
        # callable, since that means there's no way to create new
        # parsers for different encodings.
        self._default_parser = parser
        self.soup = None
        self.nsmaps = [self.DEFAULT_NSMAPS_INVERTED]
        self.active_namespace_prefixes = [dict(self.DEFAULT_NSMAPS)]
        if self.is_xml:
            self.processing_instruction_class = XMLProcessingInstruction
        else:
            self.processing_instruction_class = ProcessingInstruction

        if "attribute_dict_class" not in kwargs:
            kwargs["attribute_dict_class"] = XMLAttributeDict
        super(LXMLTreeBuilderForXML, self).__init__(**kwargs)

    def _getNsTag(self, tag: str) -> Tuple[Optional[str], str]:
        # Split the namespace URL out of a fully-qualified lxml tag
        # name. Copied from lxml's src/lxml/sax.py.
        if tag[0] == "{":
            namespace, name = tag[1:].split("}", 1)
            return (namespace, name)
        else:
            return (None, tag)

    def prepare_markup(
        self,
        markup: _RawMarkup,
        user_specified_encoding: Optional[_Encoding] = None,
        document_declared_encoding: Optional[_Encoding] = None,
        exclude_encodings: Optional[_Encodings] = None,
    ) -> Iterable[
        Tuple[Union[str, bytes], Optional[_Encoding], Optional[_Encoding], bool]
    ]:
        """Run any preliminary steps necessary to make incoming markup
        acceptable to the parser.

        lxml really wants to get a bytestring and convert it to
        Unicode itself. So instead of using UnicodeDammit to convert
        the bytestring to Unicode using different encodings, this
        implementation uses EncodingDetector to iterate over the
        encodings, and tell lxml to try to parse the document as each
        one in turn.

        :param markup: Some markup -- hopefully a bytestring.
        :param user_specified_encoding: The user asked to try this encoding.
        :param document_declared_encoding: The markup itself claims to be
            in this encoding.
        :param exclude_encodings: The user asked _not_ to try any of
            these encodings.

        :yield: A series of 4-tuples: (markup, encoding, declared encoding,
            has undergone character replacement)

            Each 4-tuple represents a strategy for converting the
            document to Unicode and parsing it. Each strategy will be tried
            in turn.
        """
        if not self.is_xml:
            # We're in HTML mode, so if we're given XML, that's worth
            # noting.
            DetectsXMLParsedAsHTML.warn_if_markup_looks_like_xml(markup, stacklevel=3)

        if isinstance(markup, str):
            # We were given Unicode. Maybe lxml can parse Unicode on
            # this system?

            # TODO: This is a workaround for
            # https://bugs.launchpad.net/lxml/+bug/1948551.
            # We can remove it once the upstream issue is fixed.
            if len(markup) > 0 and markup[0] == "\N{BYTE ORDER MARK}":
                markup = markup[1:]
            yield markup, None, document_declared_encoding, False

        if isinstance(markup, str):
            # No, apparently not. Convert the Unicode to UTF-8 and
            # tell lxml to parse it as UTF-8.
            yield (markup.encode("utf8"), "utf8", document_declared_encoding, False)

            # Since the document was Unicode in the first place, there
            # is no need to try any more strategies; we know this will
            # work.
            return

        known_definite_encodings: List[_Encoding] = []
        if user_specified_encoding:
            # This was provided by the end-user; treat it as a known
            # definite encoding per the algorithm laid out in the
            # HTML5 spec. (See the EncodingDetector class for
            # details.)
            known_definite_encodings.append(user_specified_encoding)

        user_encodings: List[_Encoding] = []
        if document_declared_encoding:
            # This was found in the document; treat it as a slightly
            # lower-priority user encoding.
            user_encodings.append(document_declared_encoding)

        detector = EncodingDetector(
            markup,
            known_definite_encodings=known_definite_encodings,
            user_encodings=user_encodings,
            is_html=not self.is_xml,
            exclude_encodings=exclude_encodings,
        )
        for encoding in detector.encodings:
            yield (detector.markup, encoding, document_declared_encoding, False)

    def feed(self, markup: _RawMarkup) -> None:
        io: Union[BytesIO, StringIO]
        if isinstance(markup, bytes):
            io = BytesIO(markup)
        elif isinstance(markup, str):
            io = StringIO(markup)

        # initialize_soup is called before feed, so we know this
        # is not None.
        assert self.soup is not None

        # Call feed() at least once, even if the markup is empty,
        # or the parser won't be initialized.
        data = io.read(self.CHUNK_SIZE)
        try:
            self.parser = self.parser_for(self.soup.original_encoding)
            self.parser.feed(data)
            while len(data) != 0:
                # Now call feed() on the rest of the data, chunk by chunk.
                data = io.read(self.CHUNK_SIZE)
                if len(data) != 0:
                    self.parser.feed(data)
            self.parser.close()
        except (UnicodeDecodeError, LookupError, etree.ParserError) as e:
            raise ParserRejectedMarkup(e)

    def close(self) -> None:
        self.nsmaps = [self.DEFAULT_NSMAPS_INVERTED]

    def start(
        self,
        tag: str | bytes,
        attrs: Dict[str | bytes, str | bytes],
        nsmap: _NamespaceMapping = {},
    ) -> None:
        # This is called by lxml code as a result of calling
        # BeautifulSoup.feed(), and we know self.soup is set by the time feed()
        # is called.
        assert self.soup is not None
        assert isinstance(tag, str)

        # We need to recreate the attribute dict for three
        # reasons. First, for type checking, so we can assert there
        # are no bytestrings in the keys or values. Second, because we
        # need a mutable dict--lxml might send us an immutable
        # dictproxy. Third, so we can handle namespaced attribute
        # names by converting the keys to NamespacedAttributes.
        new_attrs: Dict[Union[str, NamespacedAttribute], str] = (
            self.attribute_dict_class()
        )
        for k, v in attrs.items():
            assert isinstance(k, str)
            assert isinstance(v, str)
            new_attrs[k] = v

        nsprefix: Optional[_NamespacePrefix] = None
        namespace: Optional[_NamespaceURL] = None
        # Invert each namespace map as it comes in.
        if len(nsmap) == 0 and len(self.nsmaps) > 1:
            # There are no new namespaces for this tag, but
            # non-default namespaces are in play, so we need a
            # separate tag stack to know when they end.
            self.nsmaps.append(None)
        elif len(nsmap) > 0:
            # A new namespace mapping has come into play.

            # First, Let the BeautifulSoup object know about it.
            self._register_namespaces(nsmap)

            # Then, add it to our running list of inverted namespace
            # mappings.
            self.nsmaps.append(_invert(nsmap))

            # The currently active namespace prefixes have
            # changed. Calculate the new mapping so it can be stored
            # with all Tag objects created while these prefixes are in
            # scope.
            current_mapping = dict(self.active_namespace_prefixes[-1])
            current_mapping.update(nsmap)

            # We should not track un-prefixed namespaces as we can only hold one
            # and it will be recognized as the default namespace by soupsieve,
            # which may be confusing in some situations.
            if "" in current_mapping:
                del current_mapping[""]
            self.active_namespace_prefixes.append(current_mapping)

            # Also treat the namespace mapping as a set of attributes on the
            # tag, so we can recreate it later.
            for prefix, namespace in list(nsmap.items()):
                attribute = NamespacedAttribute(
                    "xmlns", prefix, "http://www.w3.org/2000/xmlns/"
                )
                new_attrs[attribute] = namespace

        # Namespaces are in play. Find any attributes that came in
        # from lxml with namespaces attached to their names, and
        # turn then into NamespacedAttribute objects.
        final_attrs: AttributeDict = self.attribute_dict_class()
        for attr, value in list(new_attrs.items()):
            namespace, attr = self._getNsTag(attr)
            if namespace is None:
                final_attrs[attr] = value
            else:
                nsprefix = self._prefix_for_namespace(namespace)
                attr = NamespacedAttribute(nsprefix, attr, namespace)
                final_attrs[attr] = value

        namespace, tag = self._getNsTag(tag)
        nsprefix = self._prefix_for_namespace(namespace)
        self.soup.handle_starttag(
            tag,
            namespace,
            nsprefix,
            final_attrs,
            namespaces=self.active_namespace_prefixes[-1],
        )

    def _prefix_for_namespace(
        self, namespace: Optional[_NamespaceURL]
    ) -> Optional[_NamespacePrefix]:
        """Find the currently active prefix for the given namespace."""
        if namespace is None:
            return None
        for inverted_nsmap in reversed(self.nsmaps):
            if inverted_nsmap is not None and namespace in inverted_nsmap:
                return inverted_nsmap[namespace]
        return None

    def end(self, name: str | bytes) -> None:
        assert self.soup is not None
        assert isinstance(name, str)
        self.soup.endData()
        namespace, name = self._getNsTag(name)
        nsprefix = None
        if namespace is not None:
            for inverted_nsmap in reversed(self.nsmaps):
                if inverted_nsmap is not None and namespace in inverted_nsmap:
                    nsprefix = inverted_nsmap[namespace]
                    break
        self.soup.handle_endtag(name, nsprefix)
        if len(self.nsmaps) > 1:
            # This tag, or one of its parents, introduced a namespace
            # mapping, so pop it off the stack.
            out_of_scope_nsmap = self.nsmaps.pop()

            if out_of_scope_nsmap is not None:
                # This tag introduced a namespace mapping which is no
                # longer in scope. Recalculate the currently active
                # namespace prefixes.
                self.active_namespace_prefixes.pop()

    def pi(self, target: str, data: str) -> None:
        assert self.soup is not None
        self.soup.endData()
        data = target + " " + data
        self.soup.handle_data(data)
        self.soup.endData(self.processing_instruction_class)

    def data(self, data: str | bytes) -> None:
        assert self.soup is not None
        assert isinstance(data, str)
        self.soup.handle_data(data)

    def doctype(self, name: str, pubid: str, system: str) -> None:
        assert self.soup is not None
        self.soup.endData()
        doctype_string = Doctype._string_for_name_and_ids(name, pubid, system)
        self.soup.handle_data(doctype_string)
        self.soup.endData(containerClass=Doctype)

    def comment(self, text: str | bytes) -> None:
        "Handle comments as Comment objects."
        assert self.soup is not None
        assert isinstance(text, str)
        self.soup.endData()
        self.soup.handle_data(text)
        self.soup.endData(Comment)

    def test_fragment_to_document(self, fragment: str) -> str:
        """See `TreeBuilder`."""
        return '<?xml version="1.0" encoding="utf-8"?>\n%s' % fragment


class LXMLTreeBuilder(HTMLTreeBuilder, LXMLTreeBuilderForXML):
    NAME: str = LXML
    ALTERNATE_NAMES: Iterable[str] = ["lxml-html"]

    features: Iterable[str] = list(ALTERNATE_NAMES) + [NAME, HTML, FAST, PERMISSIVE]
    is_xml: bool = False

    def default_parser(self, encoding: Optional[_Encoding]) -> _ParserOrParserClass:
        return etree.HTMLParser

    def feed(self, markup: _RawMarkup) -> None:
        # We know self.soup is set by the time feed() is called.
        assert self.soup is not None
        encoding = self.soup.original_encoding
        try:
            self.parser = self.parser_for(encoding)
            self.parser.feed(markup)
            self.parser.close()
        except (UnicodeDecodeError, LookupError, etree.ParserError) as e:
            raise ParserRejectedMarkup(e)

    def test_fragment_to_document(self, fragment: str) -> str:
        """See `TreeBuilder`."""
        return "<html><body>%s</body></html>" % fragment

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v135\animation.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Animation (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import runtime


@dataclass
class Animation:
    '''
    Animation instance.
    '''
    #: ``Animation``'s id.
    id_: str

    #: ``Animation``'s name.
    name: str

    #: ``Animation``'s internal paused state.
    paused_state: bool

    #: ``Animation``'s play state.
    play_state: str

    #: ``Animation``'s playback rate.
    playback_rate: float

    #: ``Animation``'s start time.
    #: Milliseconds for time based animations and
    #: percentage [0 - 100] for scroll driven animations
    #: (i.e. when viewOrScrollTimeline exists).
    start_time: float

    #: ``Animation``'s current time.
    current_time: float

    #: Animation type of ``Animation``.
    type_: str

    #: ``Animation``'s source animation node.
    source: typing.Optional[AnimationEffect] = None

    #: A unique ID for ``Animation`` representing the sources that triggered this CSS
    #: animation/transition.
    css_id: typing.Optional[str] = None

    #: View or scroll timeline
    view_or_scroll_timeline: typing.Optional[ViewOrScrollTimeline] = None

    def to_json(self):
        json = dict()
        json['id'] = self.id_
        json['name'] = self.name
        json['pausedState'] = self.paused_state
        json['playState'] = self.play_state
        json['playbackRate'] = self.playback_rate
        json['startTime'] = self.start_time
        json['currentTime'] = self.current_time
        json['type'] = self.type_
        if self.source is not None:
            json['source'] = self.source.to_json()
        if self.css_id is not None:
            json['cssId'] = self.css_id
        if self.view_or_scroll_timeline is not None:
            json['viewOrScrollTimeline'] = self.view_or_scroll_timeline.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            id_=str(json['id']),
            name=str(json['name']),
            paused_state=bool(json['pausedState']),
            play_state=str(json['playState']),
            playback_rate=float(json['playbackRate']),
            start_time=float(json['startTime']),
            current_time=float(json['currentTime']),
            type_=str(json['type']),
            source=AnimationEffect.from_json(json['source']) if 'source' in json else None,
            css_id=str(json['cssId']) if 'cssId' in json else None,
            view_or_scroll_timeline=ViewOrScrollTimeline.from_json(json['viewOrScrollTimeline']) if 'viewOrScrollTimeline' in json else None,
        )


@dataclass
class ViewOrScrollTimeline:
    '''
    Timeline instance
    '''
    #: Orientation of the scroll
    axis: dom.ScrollOrientation

    #: Scroll container node
    source_node_id: typing.Optional[dom.BackendNodeId] = None

    #: Represents the starting scroll position of the timeline
    #: as a length offset in pixels from scroll origin.
    start_offset: typing.Optional[float] = None

    #: Represents the ending scroll position of the timeline
    #: as a length offset in pixels from scroll origin.
    end_offset: typing.Optional[float] = None

    #: The element whose principal box's visibility in the
    #: scrollport defined the progress of the timeline.
    #: Does not exist for animations with ScrollTimeline
    subject_node_id: typing.Optional[dom.BackendNodeId] = None

    def to_json(self):
        json = dict()
        json['axis'] = self.axis.to_json()
        if self.source_node_id is not None:
            json['sourceNodeId'] = self.source_node_id.to_json()
        if self.start_offset is not None:
            json['startOffset'] = self.start_offset
        if self.end_offset is not None:
            json['endOffset'] = self.end_offset
        if self.subject_node_id is not None:
            json['subjectNodeId'] = self.subject_node_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            axis=dom.ScrollOrientation.from_json(json['axis']),
            source_node_id=dom.BackendNodeId.from_json(json['sourceNodeId']) if 'sourceNodeId' in json else None,
            start_offset=float(json['startOffset']) if 'startOffset' in json else None,
            end_offset=float(json['endOffset']) if 'endOffset' in json else None,
            subject_node_id=dom.BackendNodeId.from_json(json['subjectNodeId']) if 'subjectNodeId' in json else None,
        )


@dataclass
class AnimationEffect:
    '''
    AnimationEffect instance
    '''
    #: ``AnimationEffect``'s delay.
    delay: float

    #: ``AnimationEffect``'s end delay.
    end_delay: float

    #: ``AnimationEffect``'s iteration start.
    iteration_start: float

    #: ``AnimationEffect``'s iterations.
    iterations: float

    #: ``AnimationEffect``'s iteration duration.
    #: Milliseconds for time based animations and
    #: percentage [0 - 100] for scroll driven animations
    #: (i.e. when viewOrScrollTimeline exists).
    duration: float

    #: ``AnimationEffect``'s playback direction.
    direction: str

    #: ``AnimationEffect``'s fill mode.
    fill: str

    #: ``AnimationEffect``'s timing function.
    easing: str

    #: ``AnimationEffect``'s target node.
    backend_node_id: typing.Optional[dom.BackendNodeId] = None

    #: ``AnimationEffect``'s keyframes.
    keyframes_rule: typing.Optional[KeyframesRule] = None

    def to_json(self):
        json = dict()
        json['delay'] = self.delay
        json['endDelay'] = self.end_delay
        json['iterationStart'] = self.iteration_start
        json['iterations'] = self.iterations
        json['duration'] = self.duration
        json['direction'] = self.direction
        json['fill'] = self.fill
        json['easing'] = self.easing
        if self.backend_node_id is not None:
            json['backendNodeId'] = self.backend_node_id.to_json()
        if self.keyframes_rule is not None:
            json['keyframesRule'] = self.keyframes_rule.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            delay=float(json['delay']),
            end_delay=float(json['endDelay']),
            iteration_start=float(json['iterationStart']),
            iterations=float(json['iterations']),
            duration=float(json['duration']),
            direction=str(json['direction']),
            fill=str(json['fill']),
            easing=str(json['easing']),
            backend_node_id=dom.BackendNodeId.from_json(json['backendNodeId']) if 'backendNodeId' in json else None,
            keyframes_rule=KeyframesRule.from_json(json['keyframesRule']) if 'keyframesRule' in json else None,
        )


@dataclass
class KeyframesRule:
    '''
    Keyframes Rule
    '''
    #: List of animation keyframes.
    keyframes: typing.List[KeyframeStyle]

    #: CSS keyframed animation's name.
    name: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['keyframes'] = [i.to_json() for i in self.keyframes]
        if self.name is not None:
            json['name'] = self.name
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            keyframes=[KeyframeStyle.from_json(i) for i in json['keyframes']],
            name=str(json['name']) if 'name' in json else None,
        )


@dataclass
class KeyframeStyle:
    '''
    Keyframe Style
    '''
    #: Keyframe's time offset.
    offset: str

    #: ``AnimationEffect``'s timing function.
    easing: str

    def to_json(self):
        json = dict()
        json['offset'] = self.offset
        json['easing'] = self.easing
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            offset=str(json['offset']),
            easing=str(json['easing']),
        )


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables animation domain notifications.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Animation.disable',
    }
    json = yield cmd_dict


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables animation domain notifications.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Animation.enable',
    }
    json = yield cmd_dict


def get_current_time(
        id_: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,float]:
    '''
    Returns the current time of the an animation.

    :param id_: Id of animation.
    :returns: Current time of the page.
    '''
    params: T_JSON_DICT = dict()
    params['id'] = id_
    cmd_dict: T_JSON_DICT = {
        'method': 'Animation.getCurrentTime',
        'params': params,
    }
    json = yield cmd_dict
    return float(json['currentTime'])


def get_playback_rate() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,float]:
    '''
    Gets the playback rate of the document timeline.

    :returns: Playback rate for animations on page.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Animation.getPlaybackRate',
    }
    json = yield cmd_dict
    return float(json['playbackRate'])


def release_animations(
        animations: typing.List[str]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Releases a set of animations to no longer be manipulated.

    :param animations: List of animation ids to seek.
    '''
    params: T_JSON_DICT = dict()
    params['animations'] = [i for i in animations]
    cmd_dict: T_JSON_DICT = {
        'method': 'Animation.releaseAnimations',
        'params': params,
    }
    json = yield cmd_dict


def resolve_animation(
        animation_id: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,runtime.RemoteObject]:
    '''
    Gets the remote object of the Animation.

    :param animation_id: Animation id.
    :returns: Corresponding remote object.
    '''
    params: T_JSON_DICT = dict()
    params['animationId'] = animation_id
    cmd_dict: T_JSON_DICT = {
        'method': 'Animation.resolveAnimation',
        'params': params,
    }
    json = yield cmd_dict
    return runtime.RemoteObject.from_json(json['remoteObject'])


def seek_animations(
        animations: typing.List[str],
        current_time: float
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Seek a set of animations to a particular time within each animation.

    :param animations: List of animation ids to seek.
    :param current_time: Set the current time of each animation.
    '''
    params: T_JSON_DICT = dict()
    params['animations'] = [i for i in animations]
    params['currentTime'] = current_time
    cmd_dict: T_JSON_DICT = {
        'method': 'Animation.seekAnimations',
        'params': params,
    }
    json = yield cmd_dict


def set_paused(
        animations: typing.List[str],
        paused: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets the paused state of a set of animations.

    :param animations: Animations to set the pause state of.
    :param paused: Paused state to set to.
    '''
    params: T_JSON_DICT = dict()
    params['animations'] = [i for i in animations]
    params['paused'] = paused
    cmd_dict: T_JSON_DICT = {
        'method': 'Animation.setPaused',
        'params': params,
    }
    json = yield cmd_dict


def set_playback_rate(
        playback_rate: float
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets the playback rate of the document timeline.

    :param playback_rate: Playback rate for animations on page
    '''
    params: T_JSON_DICT = dict()
    params['playbackRate'] = playback_rate
    cmd_dict: T_JSON_DICT = {
        'method': 'Animation.setPlaybackRate',
        'params': params,
    }
    json = yield cmd_dict


def set_timing(
        animation_id: str,
        duration: float,
        delay: float
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets the timing of an animation node.

    :param animation_id: Animation id.
    :param duration: Duration of the animation.
    :param delay: Delay of the animation.
    '''
    params: T_JSON_DICT = dict()
    params['animationId'] = animation_id
    params['duration'] = duration
    params['delay'] = delay
    cmd_dict: T_JSON_DICT = {
        'method': 'Animation.setTiming',
        'params': params,
    }
    json = yield cmd_dict


@event_class('Animation.animationCanceled')
@dataclass
class AnimationCanceled:
    '''
    Event for when an animation has been cancelled.
    '''
    #: Id of the animation that was cancelled.
    id_: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AnimationCanceled:
        return cls(
            id_=str(json['id'])
        )


@event_class('Animation.animationCreated')
@dataclass
class AnimationCreated:
    '''
    Event for each animation that has been created.
    '''
    #: Id of the animation that was created.
    id_: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AnimationCreated:
        return cls(
            id_=str(json['id'])
        )


@event_class('Animation.animationStarted')
@dataclass
class AnimationStarted:
    '''
    Event for animation that has been started.
    '''
    #: Animation that was started.
    animation: Animation

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AnimationStarted:
        return cls(
            animation=Animation.from_json(json['animation'])
        )


@event_class('Animation.animationUpdated')
@dataclass
class AnimationUpdated:
    '''
    Event for animation that has been updated.
    '''
    #: Animation that was updated.
    animation: Animation

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AnimationUpdated:
        return cls(
            animation=Animation.from_json(json['animation'])
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v136\animation.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Animation (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import runtime


@dataclass
class Animation:
    '''
    Animation instance.
    '''
    #: ``Animation``'s id.
    id_: str

    #: ``Animation``'s name.
    name: str

    #: ``Animation``'s internal paused state.
    paused_state: bool

    #: ``Animation``'s play state.
    play_state: str

    #: ``Animation``'s playback rate.
    playback_rate: float

    #: ``Animation``'s start time.
    #: Milliseconds for time based animations and
    #: percentage [0 - 100] for scroll driven animations
    #: (i.e. when viewOrScrollTimeline exists).
    start_time: float

    #: ``Animation``'s current time.
    current_time: float

    #: Animation type of ``Animation``.
    type_: str

    #: ``Animation``'s source animation node.
    source: typing.Optional[AnimationEffect] = None

    #: A unique ID for ``Animation`` representing the sources that triggered this CSS
    #: animation/transition.
    css_id: typing.Optional[str] = None

    #: View or scroll timeline
    view_or_scroll_timeline: typing.Optional[ViewOrScrollTimeline] = None

    def to_json(self):
        json = dict()
        json['id'] = self.id_
        json['name'] = self.name
        json['pausedState'] = self.paused_state
        json['playState'] = self.play_state
        json['playbackRate'] = self.playback_rate
        json['startTime'] = self.start_time
        json['currentTime'] = self.current_time
        json['type'] = self.type_
        if self.source is not None:
            json['source'] = self.source.to_json()
        if self.css_id is not None:
            json['cssId'] = self.css_id
        if self.view_or_scroll_timeline is not None:
            json['viewOrScrollTimeline'] = self.view_or_scroll_timeline.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            id_=str(json['id']),
            name=str(json['name']),
            paused_state=bool(json['pausedState']),
            play_state=str(json['playState']),
            playback_rate=float(json['playbackRate']),
            start_time=float(json['startTime']),
            current_time=float(json['currentTime']),
            type_=str(json['type']),
            source=AnimationEffect.from_json(json['source']) if 'source' in json else None,
            css_id=str(json['cssId']) if 'cssId' in json else None,
            view_or_scroll_timeline=ViewOrScrollTimeline.from_json(json['viewOrScrollTimeline']) if 'viewOrScrollTimeline' in json else None,
        )


@dataclass
class ViewOrScrollTimeline:
    '''
    Timeline instance
    '''
    #: Orientation of the scroll
    axis: dom.ScrollOrientation

    #: Scroll container node
    source_node_id: typing.Optional[dom.BackendNodeId] = None

    #: Represents the starting scroll position of the timeline
    #: as a length offset in pixels from scroll origin.
    start_offset: typing.Optional[float] = None

    #: Represents the ending scroll position of the timeline
    #: as a length offset in pixels from scroll origin.
    end_offset: typing.Optional[float] = None

    #: The element whose principal box's visibility in the
    #: scrollport defined the progress of the timeline.
    #: Does not exist for animations with ScrollTimeline
    subject_node_id: typing.Optional[dom.BackendNodeId] = None

    def to_json(self):
        json = dict()
        json['axis'] = self.axis.to_json()
        if self.source_node_id is not None:
            json['sourceNodeId'] = self.source_node_id.to_json()
        if self.start_offset is not None:
            json['startOffset'] = self.start_offset
        if self.end_offset is not None:
            json['endOffset'] = self.end_offset
        if self.subject_node_id is not None:
            json['subjectNodeId'] = self.subject_node_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            axis=dom.ScrollOrientation.from_json(json['axis']),
            source_node_id=dom.BackendNodeId.from_json(json['sourceNodeId']) if 'sourceNodeId' in json else None,
            start_offset=float(json['startOffset']) if 'startOffset' in json else None,
            end_offset=float(json['endOffset']) if 'endOffset' in json else None,
            subject_node_id=dom.BackendNodeId.from_json(json['subjectNodeId']) if 'subjectNodeId' in json else None,
        )


@dataclass
class AnimationEffect:
    '''
    AnimationEffect instance
    '''
    #: ``AnimationEffect``'s delay.
    delay: float

    #: ``AnimationEffect``'s end delay.
    end_delay: float

    #: ``AnimationEffect``'s iteration start.
    iteration_start: float

    #: ``AnimationEffect``'s iterations.
    iterations: float

    #: ``AnimationEffect``'s iteration duration.
    #: Milliseconds for time based animations and
    #: percentage [0 - 100] for scroll driven animations
    #: (i.e. when viewOrScrollTimeline exists).
    duration: float

    #: ``AnimationEffect``'s playback direction.
    direction: str

    #: ``AnimationEffect``'s fill mode.
    fill: str

    #: ``AnimationEffect``'s timing function.
    easing: str

    #: ``AnimationEffect``'s target node.
    backend_node_id: typing.Optional[dom.BackendNodeId] = None

    #: ``AnimationEffect``'s keyframes.
    keyframes_rule: typing.Optional[KeyframesRule] = None

    def to_json(self):
        json = dict()
        json['delay'] = self.delay
        json['endDelay'] = self.end_delay
        json['iterationStart'] = self.iteration_start
        json['iterations'] = self.iterations
        json['duration'] = self.duration
        json['direction'] = self.direction
        json['fill'] = self.fill
        json['easing'] = self.easing
        if self.backend_node_id is not None:
            json['backendNodeId'] = self.backend_node_id.to_json()
        if self.keyframes_rule is not None:
            json['keyframesRule'] = self.keyframes_rule.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            delay=float(json['delay']),
            end_delay=float(json['endDelay']),
            iteration_start=float(json['iterationStart']),
            iterations=float(json['iterations']),
            duration=float(json['duration']),
            direction=str(json['direction']),
            fill=str(json['fill']),
            easing=str(json['easing']),
            backend_node_id=dom.BackendNodeId.from_json(json['backendNodeId']) if 'backendNodeId' in json else None,
            keyframes_rule=KeyframesRule.from_json(json['keyframesRule']) if 'keyframesRule' in json else None,
        )


@dataclass
class KeyframesRule:
    '''
    Keyframes Rule
    '''
    #: List of animation keyframes.
    keyframes: typing.List[KeyframeStyle]

    #: CSS keyframed animation's name.
    name: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['keyframes'] = [i.to_json() for i in self.keyframes]
        if self.name is not None:
            json['name'] = self.name
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            keyframes=[KeyframeStyle.from_json(i) for i in json['keyframes']],
            name=str(json['name']) if 'name' in json else None,
        )


@dataclass
class KeyframeStyle:
    '''
    Keyframe Style
    '''
    #: Keyframe's time offset.
    offset: str

    #: ``AnimationEffect``'s timing function.
    easing: str

    def to_json(self):
        json = dict()
        json['offset'] = self.offset
        json['easing'] = self.easing
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            offset=str(json['offset']),
            easing=str(json['easing']),
        )


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables animation domain notifications.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Animation.disable',
    }
    json = yield cmd_dict


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables animation domain notifications.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Animation.enable',
    }
    json = yield cmd_dict


def get_current_time(
        id_: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,float]:
    '''
    Returns the current time of the an animation.

    :param id_: Id of animation.
    :returns: Current time of the page.
    '''
    params: T_JSON_DICT = dict()
    params['id'] = id_
    cmd_dict: T_JSON_DICT = {
        'method': 'Animation.getCurrentTime',
        'params': params,
    }
    json = yield cmd_dict
    return float(json['currentTime'])


def get_playback_rate() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,float]:
    '''
    Gets the playback rate of the document timeline.

    :returns: Playback rate for animations on page.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Animation.getPlaybackRate',
    }
    json = yield cmd_dict
    return float(json['playbackRate'])


def release_animations(
        animations: typing.List[str]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Releases a set of animations to no longer be manipulated.

    :param animations: List of animation ids to seek.
    '''
    params: T_JSON_DICT = dict()
    params['animations'] = [i for i in animations]
    cmd_dict: T_JSON_DICT = {
        'method': 'Animation.releaseAnimations',
        'params': params,
    }
    json = yield cmd_dict


def resolve_animation(
        animation_id: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,runtime.RemoteObject]:
    '''
    Gets the remote object of the Animation.

    :param animation_id: Animation id.
    :returns: Corresponding remote object.
    '''
    params: T_JSON_DICT = dict()
    params['animationId'] = animation_id
    cmd_dict: T_JSON_DICT = {
        'method': 'Animation.resolveAnimation',
        'params': params,
    }
    json = yield cmd_dict
    return runtime.RemoteObject.from_json(json['remoteObject'])


def seek_animations(
        animations: typing.List[str],
        current_time: float
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Seek a set of animations to a particular time within each animation.

    :param animations: List of animation ids to seek.
    :param current_time: Set the current time of each animation.
    '''
    params: T_JSON_DICT = dict()
    params['animations'] = [i for i in animations]
    params['currentTime'] = current_time
    cmd_dict: T_JSON_DICT = {
        'method': 'Animation.seekAnimations',
        'params': params,
    }
    json = yield cmd_dict


def set_paused(
        animations: typing.List[str],
        paused: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets the paused state of a set of animations.

    :param animations: Animations to set the pause state of.
    :param paused: Paused state to set to.
    '''
    params: T_JSON_DICT = dict()
    params['animations'] = [i for i in animations]
    params['paused'] = paused
    cmd_dict: T_JSON_DICT = {
        'method': 'Animation.setPaused',
        'params': params,
    }
    json = yield cmd_dict


def set_playback_rate(
        playback_rate: float
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets the playback rate of the document timeline.

    :param playback_rate: Playback rate for animations on page
    '''
    params: T_JSON_DICT = dict()
    params['playbackRate'] = playback_rate
    cmd_dict: T_JSON_DICT = {
        'method': 'Animation.setPlaybackRate',
        'params': params,
    }
    json = yield cmd_dict


def set_timing(
        animation_id: str,
        duration: float,
        delay: float
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets the timing of an animation node.

    :param animation_id: Animation id.
    :param duration: Duration of the animation.
    :param delay: Delay of the animation.
    '''
    params: T_JSON_DICT = dict()
    params['animationId'] = animation_id
    params['duration'] = duration
    params['delay'] = delay
    cmd_dict: T_JSON_DICT = {
        'method': 'Animation.setTiming',
        'params': params,
    }
    json = yield cmd_dict


@event_class('Animation.animationCanceled')
@dataclass
class AnimationCanceled:
    '''
    Event for when an animation has been cancelled.
    '''
    #: Id of the animation that was cancelled.
    id_: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AnimationCanceled:
        return cls(
            id_=str(json['id'])
        )


@event_class('Animation.animationCreated')
@dataclass
class AnimationCreated:
    '''
    Event for each animation that has been created.
    '''
    #: Id of the animation that was created.
    id_: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AnimationCreated:
        return cls(
            id_=str(json['id'])
        )


@event_class('Animation.animationStarted')
@dataclass
class AnimationStarted:
    '''
    Event for animation that has been started.
    '''
    #: Animation that was started.
    animation: Animation

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AnimationStarted:
        return cls(
            animation=Animation.from_json(json['animation'])
        )


@event_class('Animation.animationUpdated')
@dataclass
class AnimationUpdated:
    '''
    Event for animation that has been updated.
    '''
    #: Animation that was updated.
    animation: Animation

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AnimationUpdated:
        return cls(
            animation=Animation.from_json(json['animation'])
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v137\animation.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Animation (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import runtime


@dataclass
class Animation:
    '''
    Animation instance.
    '''
    #: ``Animation``'s id.
    id_: str

    #: ``Animation``'s name.
    name: str

    #: ``Animation``'s internal paused state.
    paused_state: bool

    #: ``Animation``'s play state.
    play_state: str

    #: ``Animation``'s playback rate.
    playback_rate: float

    #: ``Animation``'s start time.
    #: Milliseconds for time based animations and
    #: percentage [0 - 100] for scroll driven animations
    #: (i.e. when viewOrScrollTimeline exists).
    start_time: float

    #: ``Animation``'s current time.
    current_time: float

    #: Animation type of ``Animation``.
    type_: str

    #: ``Animation``'s source animation node.
    source: typing.Optional[AnimationEffect] = None

    #: A unique ID for ``Animation`` representing the sources that triggered this CSS
    #: animation/transition.
    css_id: typing.Optional[str] = None

    #: View or scroll timeline
    view_or_scroll_timeline: typing.Optional[ViewOrScrollTimeline] = None

    def to_json(self):
        json = dict()
        json['id'] = self.id_
        json['name'] = self.name
        json['pausedState'] = self.paused_state
        json['playState'] = self.play_state
        json['playbackRate'] = self.playback_rate
        json['startTime'] = self.start_time
        json['currentTime'] = self.current_time
        json['type'] = self.type_
        if self.source is not None:
            json['source'] = self.source.to_json()
        if self.css_id is not None:
            json['cssId'] = self.css_id
        if self.view_or_scroll_timeline is not None:
            json['viewOrScrollTimeline'] = self.view_or_scroll_timeline.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            id_=str(json['id']),
            name=str(json['name']),
            paused_state=bool(json['pausedState']),
            play_state=str(json['playState']),
            playback_rate=float(json['playbackRate']),
            start_time=float(json['startTime']),
            current_time=float(json['currentTime']),
            type_=str(json['type']),
            source=AnimationEffect.from_json(json['source']) if 'source' in json else None,
            css_id=str(json['cssId']) if 'cssId' in json else None,
            view_or_scroll_timeline=ViewOrScrollTimeline.from_json(json['viewOrScrollTimeline']) if 'viewOrScrollTimeline' in json else None,
        )


@dataclass
class ViewOrScrollTimeline:
    '''
    Timeline instance
    '''
    #: Orientation of the scroll
    axis: dom.ScrollOrientation

    #: Scroll container node
    source_node_id: typing.Optional[dom.BackendNodeId] = None

    #: Represents the starting scroll position of the timeline
    #: as a length offset in pixels from scroll origin.
    start_offset: typing.Optional[float] = None

    #: Represents the ending scroll position of the timeline
    #: as a length offset in pixels from scroll origin.
    end_offset: typing.Optional[float] = None

    #: The element whose principal box's visibility in the
    #: scrollport defined the progress of the timeline.
    #: Does not exist for animations with ScrollTimeline
    subject_node_id: typing.Optional[dom.BackendNodeId] = None

    def to_json(self):
        json = dict()
        json['axis'] = self.axis.to_json()
        if self.source_node_id is not None:
            json['sourceNodeId'] = self.source_node_id.to_json()
        if self.start_offset is not None:
            json['startOffset'] = self.start_offset
        if self.end_offset is not None:
            json['endOffset'] = self.end_offset
        if self.subject_node_id is not None:
            json['subjectNodeId'] = self.subject_node_id.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            axis=dom.ScrollOrientation.from_json(json['axis']),
            source_node_id=dom.BackendNodeId.from_json(json['sourceNodeId']) if 'sourceNodeId' in json else None,
            start_offset=float(json['startOffset']) if 'startOffset' in json else None,
            end_offset=float(json['endOffset']) if 'endOffset' in json else None,
            subject_node_id=dom.BackendNodeId.from_json(json['subjectNodeId']) if 'subjectNodeId' in json else None,
        )


@dataclass
class AnimationEffect:
    '''
    AnimationEffect instance
    '''
    #: ``AnimationEffect``'s delay.
    delay: float

    #: ``AnimationEffect``'s end delay.
    end_delay: float

    #: ``AnimationEffect``'s iteration start.
    iteration_start: float

    #: ``AnimationEffect``'s iterations.
    iterations: float

    #: ``AnimationEffect``'s iteration duration.
    #: Milliseconds for time based animations and
    #: percentage [0 - 100] for scroll driven animations
    #: (i.e. when viewOrScrollTimeline exists).
    duration: float

    #: ``AnimationEffect``'s playback direction.
    direction: str

    #: ``AnimationEffect``'s fill mode.
    fill: str

    #: ``AnimationEffect``'s timing function.
    easing: str

    #: ``AnimationEffect``'s target node.
    backend_node_id: typing.Optional[dom.BackendNodeId] = None

    #: ``AnimationEffect``'s keyframes.
    keyframes_rule: typing.Optional[KeyframesRule] = None

    def to_json(self):
        json = dict()
        json['delay'] = self.delay
        json['endDelay'] = self.end_delay
        json['iterationStart'] = self.iteration_start
        json['iterations'] = self.iterations
        json['duration'] = self.duration
        json['direction'] = self.direction
        json['fill'] = self.fill
        json['easing'] = self.easing
        if self.backend_node_id is not None:
            json['backendNodeId'] = self.backend_node_id.to_json()
        if self.keyframes_rule is not None:
            json['keyframesRule'] = self.keyframes_rule.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            delay=float(json['delay']),
            end_delay=float(json['endDelay']),
            iteration_start=float(json['iterationStart']),
            iterations=float(json['iterations']),
            duration=float(json['duration']),
            direction=str(json['direction']),
            fill=str(json['fill']),
            easing=str(json['easing']),
            backend_node_id=dom.BackendNodeId.from_json(json['backendNodeId']) if 'backendNodeId' in json else None,
            keyframes_rule=KeyframesRule.from_json(json['keyframesRule']) if 'keyframesRule' in json else None,
        )


@dataclass
class KeyframesRule:
    '''
    Keyframes Rule
    '''
    #: List of animation keyframes.
    keyframes: typing.List[KeyframeStyle]

    #: CSS keyframed animation's name.
    name: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['keyframes'] = [i.to_json() for i in self.keyframes]
        if self.name is not None:
            json['name'] = self.name
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            keyframes=[KeyframeStyle.from_json(i) for i in json['keyframes']],
            name=str(json['name']) if 'name' in json else None,
        )


@dataclass
class KeyframeStyle:
    '''
    Keyframe Style
    '''
    #: Keyframe's time offset.
    offset: str

    #: ``AnimationEffect``'s timing function.
    easing: str

    def to_json(self):
        json = dict()
        json['offset'] = self.offset
        json['easing'] = self.easing
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            offset=str(json['offset']),
            easing=str(json['easing']),
        )


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Disables animation domain notifications.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Animation.disable',
    }
    json = yield cmd_dict


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Enables animation domain notifications.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Animation.enable',
    }
    json = yield cmd_dict


def get_current_time(
        id_: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,float]:
    '''
    Returns the current time of the an animation.

    :param id_: Id of animation.
    :returns: Current time of the page.
    '''
    params: T_JSON_DICT = dict()
    params['id'] = id_
    cmd_dict: T_JSON_DICT = {
        'method': 'Animation.getCurrentTime',
        'params': params,
    }
    json = yield cmd_dict
    return float(json['currentTime'])


def get_playback_rate() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,float]:
    '''
    Gets the playback rate of the document timeline.

    :returns: Playback rate for animations on page.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Animation.getPlaybackRate',
    }
    json = yield cmd_dict
    return float(json['playbackRate'])


def release_animations(
        animations: typing.List[str]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Releases a set of animations to no longer be manipulated.

    :param animations: List of animation ids to seek.
    '''
    params: T_JSON_DICT = dict()
    params['animations'] = [i for i in animations]
    cmd_dict: T_JSON_DICT = {
        'method': 'Animation.releaseAnimations',
        'params': params,
    }
    json = yield cmd_dict


def resolve_animation(
        animation_id: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,runtime.RemoteObject]:
    '''
    Gets the remote object of the Animation.

    :param animation_id: Animation id.
    :returns: Corresponding remote object.
    '''
    params: T_JSON_DICT = dict()
    params['animationId'] = animation_id
    cmd_dict: T_JSON_DICT = {
        'method': 'Animation.resolveAnimation',
        'params': params,
    }
    json = yield cmd_dict
    return runtime.RemoteObject.from_json(json['remoteObject'])


def seek_animations(
        animations: typing.List[str],
        current_time: float
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Seek a set of animations to a particular time within each animation.

    :param animations: List of animation ids to seek.
    :param current_time: Set the current time of each animation.
    '''
    params: T_JSON_DICT = dict()
    params['animations'] = [i for i in animations]
    params['currentTime'] = current_time
    cmd_dict: T_JSON_DICT = {
        'method': 'Animation.seekAnimations',
        'params': params,
    }
    json = yield cmd_dict


def set_paused(
        animations: typing.List[str],
        paused: bool
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets the paused state of a set of animations.

    :param animations: Animations to set the pause state of.
    :param paused: Paused state to set to.
    '''
    params: T_JSON_DICT = dict()
    params['animations'] = [i for i in animations]
    params['paused'] = paused
    cmd_dict: T_JSON_DICT = {
        'method': 'Animation.setPaused',
        'params': params,
    }
    json = yield cmd_dict


def set_playback_rate(
        playback_rate: float
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets the playback rate of the document timeline.

    :param playback_rate: Playback rate for animations on page
    '''
    params: T_JSON_DICT = dict()
    params['playbackRate'] = playback_rate
    cmd_dict: T_JSON_DICT = {
        'method': 'Animation.setPlaybackRate',
        'params': params,
    }
    json = yield cmd_dict


def set_timing(
        animation_id: str,
        duration: float,
        delay: float
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets the timing of an animation node.

    :param animation_id: Animation id.
    :param duration: Duration of the animation.
    :param delay: Delay of the animation.
    '''
    params: T_JSON_DICT = dict()
    params['animationId'] = animation_id
    params['duration'] = duration
    params['delay'] = delay
    cmd_dict: T_JSON_DICT = {
        'method': 'Animation.setTiming',
        'params': params,
    }
    json = yield cmd_dict


@event_class('Animation.animationCanceled')
@dataclass
class AnimationCanceled:
    '''
    Event for when an animation has been cancelled.
    '''
    #: Id of the animation that was cancelled.
    id_: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AnimationCanceled:
        return cls(
            id_=str(json['id'])
        )


@event_class('Animation.animationCreated')
@dataclass
class AnimationCreated:
    '''
    Event for each animation that has been created.
    '''
    #: Id of the animation that was created.
    id_: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AnimationCreated:
        return cls(
            id_=str(json['id'])
        )


@event_class('Animation.animationStarted')
@dataclass
class AnimationStarted:
    '''
    Event for animation that has been started.
    '''
    #: Animation that was started.
    animation: Animation

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AnimationStarted:
        return cls(
            animation=Animation.from_json(json['animation'])
        )


@event_class('Animation.animationUpdated')
@dataclass
class AnimationUpdated:
    '''
    Event for animation that has been updated.
    '''
    #: Animation that was updated.
    animation: Animation

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> AnimationUpdated:
        return cls(
            animation=Animation.from_json(json['animation'])
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\wsproto\handshake.py ===
"""
wsproto/handshake
~~~~~~~~~~~~~~~~~~

An implementation of WebSocket handshakes.
"""
from collections import deque
from typing import (
    cast,
    Deque,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Sequence,
    Union,
)

import h11

from .connection import Connection, ConnectionState, ConnectionType
from .events import AcceptConnection, Event, RejectConnection, RejectData, Request
from .extensions import Extension
from .typing import Headers
from .utilities import (
    generate_accept_token,
    generate_nonce,
    LocalProtocolError,
    normed_header_dict,
    RemoteProtocolError,
    split_comma_header,
)

# RFC6455, Section 4.2.1/6 - Reading the Client's Opening Handshake
WEBSOCKET_VERSION = b"13"


class H11Handshake:
    """A Handshake implementation for HTTP/1.1 connections."""

    def __init__(self, connection_type: ConnectionType) -> None:
        self.client = connection_type is ConnectionType.CLIENT
        self._state = ConnectionState.CONNECTING

        if self.client:
            self._h11_connection = h11.Connection(h11.CLIENT)
        else:
            self._h11_connection = h11.Connection(h11.SERVER)

        self._connection: Optional[Connection] = None
        self._events: Deque[Event] = deque()
        self._initiating_request: Optional[Request] = None
        self._nonce: Optional[bytes] = None

    @property
    def state(self) -> ConnectionState:
        return self._state

    @property
    def connection(self) -> Optional[Connection]:
        """Return the established connection.

        This will either return the connection or raise a
        LocalProtocolError if the connection has not yet been
        established.

        :rtype: h11.Connection
        """
        return self._connection

    def initiate_upgrade_connection(
        self, headers: Headers, path: Union[bytes, str]
    ) -> None:
        """Initiate an upgrade connection.

        This should be used if the request has already be received and
        parsed.

        :param list headers: HTTP headers represented as a list of 2-tuples.
        :param str path: A URL path.
        """
        if self.client:
            raise LocalProtocolError(
                "Cannot initiate an upgrade connection when acting as the client"
            )
        upgrade_request = h11.Request(method=b"GET", target=path, headers=headers)
        h11_client = h11.Connection(h11.CLIENT)
        self.receive_data(h11_client.send(upgrade_request))

    def send(self, event: Event) -> bytes:
        """Send an event to the remote.

        This will return the bytes to send based on the event or raise
        a LocalProtocolError if the event is not valid given the
        state.

        :returns: Data to send to the WebSocket peer.
        :rtype: bytes
        """
        data = b""
        if isinstance(event, Request):
            data += self._initiate_connection(event)
        elif isinstance(event, AcceptConnection):
            data += self._accept(event)
        elif isinstance(event, RejectConnection):
            data += self._reject(event)
        elif isinstance(event, RejectData):
            data += self._send_reject_data(event)
        else:
            raise LocalProtocolError(
                f"Event {event} cannot be sent during the handshake"
            )
        return data

    def receive_data(self, data: Optional[bytes]) -> None:
        """Receive data from the remote.

        A list of events that the remote peer triggered by sending
        this data can be retrieved with :meth:`events`.

        :param bytes data: Data received from the WebSocket peer.
        """
        self._h11_connection.receive_data(data or b"")
        while True:
            try:
                event = self._h11_connection.next_event()
            except h11.RemoteProtocolError:
                raise RemoteProtocolError(
                    "Bad HTTP message", event_hint=RejectConnection()
                )
            if (
                isinstance(event, h11.ConnectionClosed)
                or event is h11.NEED_DATA
                or event is h11.PAUSED
            ):
                break

            if self.client:
                if isinstance(event, h11.InformationalResponse):
                    if event.status_code == 101:
                        self._events.append(self._establish_client_connection(event))
                    else:
                        self._events.append(
                            RejectConnection(
                                headers=list(event.headers),
                                status_code=event.status_code,
                                has_body=False,
                            )
                        )
                        self._state = ConnectionState.CLOSED
                elif isinstance(event, h11.Response):
                    self._state = ConnectionState.REJECTING
                    self._events.append(
                        RejectConnection(
                            headers=list(event.headers),
                            status_code=event.status_code,
                            has_body=True,
                        )
                    )
                elif isinstance(event, h11.Data):
                    self._events.append(
                        RejectData(data=event.data, body_finished=False)
                    )
                elif isinstance(event, h11.EndOfMessage):
                    self._events.append(RejectData(data=b"", body_finished=True))
                    self._state = ConnectionState.CLOSED
            else:
                if isinstance(event, h11.Request):
                    self._events.append(self._process_connection_request(event))

    def events(self) -> Generator[Event, None, None]:
        """Return a generator that provides any events that have been generated
        by protocol activity.

        :returns: a generator that yields H11 events.
        """
        while self._events:
            yield self._events.popleft()

    # Server mode methods

    def _process_connection_request(  # noqa: MC0001
        self, event: h11.Request
    ) -> Request:
        if event.method != b"GET":
            raise RemoteProtocolError(
                "Request method must be GET", event_hint=RejectConnection()
            )
        connection_tokens = None
        extensions: List[str] = []
        host = None
        key = None
        subprotocols: List[str] = []
        upgrade = b""
        version = None
        headers: Headers = []
        for name, value in event.headers:
            name = name.lower()
            if name == b"connection":
                connection_tokens = split_comma_header(value)
            elif name == b"host":
                host = value.decode("idna")
                continue  # Skip appending to headers
            elif name == b"sec-websocket-extensions":
                extensions.extend(split_comma_header(value))
                continue  # Skip appending to headers
            elif name == b"sec-websocket-key":
                key = value
            elif name == b"sec-websocket-protocol":
                subprotocols.extend(split_comma_header(value))
                continue  # Skip appending to headers
            elif name == b"sec-websocket-version":
                version = value
            elif name == b"upgrade":
                upgrade = value
            headers.append((name, value))
        if connection_tokens is None or not any(
            token.lower() == "upgrade" for token in connection_tokens
        ):
            raise RemoteProtocolError(
                "Missing header, 'Connection: Upgrade'", event_hint=RejectConnection()
            )
        if version != WEBSOCKET_VERSION:
            raise RemoteProtocolError(
                "Missing header, 'Sec-WebSocket-Version'",
                event_hint=RejectConnection(
                    headers=[(b"Sec-WebSocket-Version", WEBSOCKET_VERSION)],
                    status_code=426 if version else 400,
                ),
            )
        if key is None:
            raise RemoteProtocolError(
                "Missing header, 'Sec-WebSocket-Key'", event_hint=RejectConnection()
            )
        if upgrade.lower() != b"websocket":
            raise RemoteProtocolError(
                "Missing header, 'Upgrade: WebSocket'", event_hint=RejectConnection()
            )
        if host is None:
            raise RemoteProtocolError(
                "Missing header, 'Host'", event_hint=RejectConnection()
            )

        self._initiating_request = Request(
            extensions=extensions,
            extra_headers=headers,
            host=host,
            subprotocols=subprotocols,
            target=event.target.decode("ascii"),
        )
        return self._initiating_request

    def _accept(self, event: AcceptConnection) -> bytes:
        # _accept is always called after _process_connection_request.
        assert self._initiating_request is not None
        request_headers = normed_header_dict(self._initiating_request.extra_headers)

        nonce = request_headers[b"sec-websocket-key"]
        accept_token = generate_accept_token(nonce)

        headers = [
            (b"Upgrade", b"WebSocket"),
            (b"Connection", b"Upgrade"),
            (b"Sec-WebSocket-Accept", accept_token),
        ]

        if event.subprotocol is not None:
            if event.subprotocol not in self._initiating_request.subprotocols:
                raise LocalProtocolError(f"unexpected subprotocol {event.subprotocol}")
            headers.append(
                (b"Sec-WebSocket-Protocol", event.subprotocol.encode("ascii"))
            )

        if event.extensions:
            accepts = server_extensions_handshake(
                cast(Sequence[str], self._initiating_request.extensions),
                event.extensions,
            )
            if accepts:
                headers.append((b"Sec-WebSocket-Extensions", accepts))

        response = h11.InformationalResponse(
            status_code=101, headers=headers + event.extra_headers
        )
        self._connection = Connection(
            ConnectionType.CLIENT if self.client else ConnectionType.SERVER,
            event.extensions,
        )
        self._state = ConnectionState.OPEN
        return self._h11_connection.send(response) or b""

    def _reject(self, event: RejectConnection) -> bytes:
        if self.state != ConnectionState.CONNECTING:
            raise LocalProtocolError(
                "Connection cannot be rejected in state %s" % self.state
            )

        headers = list(event.headers)
        if not event.has_body:
            headers.append((b"content-length", b"0"))
        response = h11.Response(status_code=event.status_code, headers=headers)
        data = self._h11_connection.send(response) or b""
        self._state = ConnectionState.REJECTING
        if not event.has_body:
            data += self._h11_connection.send(h11.EndOfMessage()) or b""
            self._state = ConnectionState.CLOSED
        return data

    def _send_reject_data(self, event: RejectData) -> bytes:
        if self.state != ConnectionState.REJECTING:
            raise LocalProtocolError(
                f"Cannot send rejection data in state {self.state}"
            )

        data = self._h11_connection.send(h11.Data(data=event.data)) or b""
        if event.body_finished:
            data += self._h11_connection.send(h11.EndOfMessage()) or b""
            self._state = ConnectionState.CLOSED
        return data

    # Client mode methods

    def _initiate_connection(self, request: Request) -> bytes:
        self._initiating_request = request
        self._nonce = generate_nonce()

        headers = [
            (b"Host", request.host.encode("idna")),
            (b"Upgrade", b"WebSocket"),
            (b"Connection", b"Upgrade"),
            (b"Sec-WebSocket-Key", self._nonce),
            (b"Sec-WebSocket-Version", WEBSOCKET_VERSION),
        ]

        if request.subprotocols:
            headers.append(
                (
                    b"Sec-WebSocket-Protocol",
                    (", ".join(request.subprotocols)).encode("ascii"),
                )
            )

        if request.extensions:
            offers: Dict[str, Union[str, bool]] = {}
            for e in request.extensions:
                assert isinstance(e, Extension)
                offers[e.name] = e.offer()
            extensions = []
            for name, params in offers.items():
                bname = name.encode("ascii")
                if isinstance(params, bool):
                    if params:
                        extensions.append(bname)
                else:
                    extensions.append(b"%s; %s" % (bname, params.encode("ascii")))
            if extensions:
                headers.append((b"Sec-WebSocket-Extensions", b", ".join(extensions)))

        upgrade = h11.Request(
            method=b"GET",
            target=request.target.encode("ascii"),
            headers=headers + request.extra_headers,
        )
        return self._h11_connection.send(upgrade) or b""

    def _establish_client_connection(
        self, event: h11.InformationalResponse
    ) -> AcceptConnection:  # noqa: MC0001
        # _establish_client_connection is always called after _initiate_connection.
        assert self._initiating_request is not None
        assert self._nonce is not None

        accept = None
        connection_tokens = None
        accepts: List[str] = []
        subprotocol = None
        upgrade = b""
        headers: Headers = []
        for name, value in event.headers:
            name = name.lower()
            if name == b"connection":
                connection_tokens = split_comma_header(value)
                continue  # Skip appending to headers
            elif name == b"sec-websocket-extensions":
                accepts = split_comma_header(value)
                continue  # Skip appending to headers
            elif name == b"sec-websocket-accept":
                accept = value
                continue  # Skip appending to headers
            elif name == b"sec-websocket-protocol":
                subprotocol = value.decode("ascii")
                continue  # Skip appending to headers
            elif name == b"upgrade":
                upgrade = value
                continue  # Skip appending to headers
            headers.append((name, value))

        if connection_tokens is None or not any(
            token.lower() == "upgrade" for token in connection_tokens
        ):
            raise RemoteProtocolError(
                "Missing header, 'Connection: Upgrade'", event_hint=RejectConnection()
            )
        if upgrade.lower() != b"websocket":
            raise RemoteProtocolError(
                "Missing header, 'Upgrade: WebSocket'", event_hint=RejectConnection()
            )
        accept_token = generate_accept_token(self._nonce)
        if accept != accept_token:
            raise RemoteProtocolError("Bad accept token", event_hint=RejectConnection())
        if subprotocol is not None:
            if subprotocol not in self._initiating_request.subprotocols:
                raise RemoteProtocolError(
                    f"unrecognized subprotocol {subprotocol}",
                    event_hint=RejectConnection(),
                )
        extensions = client_extensions_handshake(
            accepts, cast(Sequence[Extension], self._initiating_request.extensions)
        )

        self._connection = Connection(
            ConnectionType.CLIENT if self.client else ConnectionType.SERVER,
            extensions,
            self._h11_connection.trailing_data[0],
        )
        self._state = ConnectionState.OPEN
        return AcceptConnection(
            extensions=extensions, extra_headers=headers, subprotocol=subprotocol
        )

    def __repr__(self) -> str:
        return "{}(client={}, state={})".format(
            self.__class__.__name__, self.client, self.state
        )


def server_extensions_handshake(
    requested: Iterable[str], supported: List[Extension]
) -> Optional[bytes]:
    """Agree on the extensions to use returning an appropriate header value.

    This returns None if there are no agreed extensions
    """
    accepts: Dict[str, Union[bool, bytes]] = {}
    for offer in requested:
        name = offer.split(";", 1)[0].strip()
        for extension in supported:
            if extension.name == name:
                accept = extension.accept(offer)
                if isinstance(accept, bool):
                    if accept:
                        accepts[extension.name] = True
                elif accept is not None:
                    accepts[extension.name] = accept.encode("ascii")

    if accepts:
        extensions: List[bytes] = []
        for name, params in accepts.items():
            name_bytes = name.encode("ascii")
            if isinstance(params, bool):
                assert params
                extensions.append(name_bytes)
            else:
                if params == b"":
                    extensions.append(b"%s" % (name_bytes))
                else:
                    extensions.append(b"%s; %s" % (name_bytes, params))
        return b", ".join(extensions)

    return None


def client_extensions_handshake(
    accepted: Iterable[str], supported: Sequence[Extension]
) -> List[Extension]:
    # This raises RemoteProtocolError is the accepted extension is not
    # supported.
    extensions = []
    for accept in accepted:
        name = accept.split(";", 1)[0].strip()
        for extension in supported:
            if extension.name == name:
                extension.finalize(accept)
                extensions.append(extension)
                break
        else:
            raise RemoteProtocolError(
                f"unrecognized extension {name}", event_hint=RejectConnection()
            )
    return extensions

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\domains\old_polynomialring.py ===
"""Implementation of :class:`PolynomialRing` class. """


from sympy.polys.agca.modules import FreeModulePolyRing
from sympy.polys.domains.compositedomain import CompositeDomain
from sympy.polys.domains.old_fractionfield import FractionField
from sympy.polys.domains.ring import Ring
from sympy.polys.orderings import monomial_key, build_product_order
from sympy.polys.polyclasses import DMP, DMF
from sympy.polys.polyerrors import (GeneratorsNeeded, PolynomialError,
        CoercionFailed, ExactQuotientFailed, NotReversible)
from sympy.polys.polyutils import dict_from_basic, basic_from_dict, _dict_reorder
from sympy.utilities import public
from sympy.utilities.iterables import iterable


@public
class PolynomialRingBase(Ring, CompositeDomain):
    """
    Base class for generalized polynomial rings.

    This base class should be used for uniform access to generalized polynomial
    rings. Subclasses only supply information about the element storage etc.

    Do not instantiate.
    """

    has_assoc_Ring = True
    has_assoc_Field = True

    default_order = "grevlex"

    def __init__(self, dom, *gens, **opts):
        if not gens:
            raise GeneratorsNeeded("generators not specified")

        lev = len(gens) - 1
        self.ngens = len(gens)

        self.zero = self.dtype.zero(lev, dom)
        self.one = self.dtype.one(lev, dom)

        self.domain = self.dom = dom
        self.symbols = self.gens = gens
        # NOTE 'order' may not be set if inject was called through CompositeDomain
        self.order = opts.get('order', monomial_key(self.default_order))

    def set_domain(self, dom):
        """Return a new polynomial ring with given domain. """
        return self.__class__(dom, *self.gens, order=self.order)

    def new(self, element):
        return self.dtype(element, self.dom, len(self.gens) - 1)

    def _ground_new(self, element):
        return self.one.ground_new(element)

    def _from_dict(self, element):
        return DMP.from_dict(element, len(self.gens) - 1, self.dom)

    def __str__(self):
        s_order = str(self.order)
        orderstr = (
            " order=" + s_order) if s_order != self.default_order else ""
        return str(self.dom) + '[' + ','.join(map(str, self.gens)) + orderstr + ']'

    def __hash__(self):
        return hash((self.__class__.__name__, self.dtype, self.dom,
                     self.gens, self.order))

    def __eq__(self, other):
        """Returns ``True`` if two domains are equivalent. """
        return isinstance(other, PolynomialRingBase) and \
            self.dtype == other.dtype and self.dom == other.dom and \
            self.gens == other.gens and self.order == other.order

    def from_ZZ(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
        return K1._ground_new(K1.dom.convert(a, K0))

    def from_ZZ_python(K1, a, K0):
        """Convert a Python ``int`` object to ``dtype``. """
        return K1._ground_new(K1.dom.convert(a, K0))

    def from_QQ(K1, a, K0):
        """Convert a Python ``Fraction`` object to ``dtype``. """
        return K1._ground_new(K1.dom.convert(a, K0))

    def from_QQ_python(K1, a, K0):
        """Convert a Python ``Fraction`` object to ``dtype``. """
        return K1._ground_new(K1.dom.convert(a, K0))

    def from_ZZ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpz`` object to ``dtype``. """
        return K1._ground_new(K1.dom.convert(a, K0))

    def from_QQ_gmpy(K1, a, K0):
        """Convert a GMPY ``mpq`` object to ``dtype``. """
        return K1._ground_new(K1.dom.convert(a, K0))

    def from_RealField(K1, a, K0):
        """Convert a mpmath ``mpf`` object to ``dtype``. """
        return K1._ground_new(K1.dom.convert(a, K0))

    def from_AlgebraicField(K1, a, K0):
        """Convert a ``ANP`` object to ``dtype``. """
        if K1.dom == K0:
            return K1._ground_new(a)

    def from_PolynomialRing(K1, a, K0):
        """Convert a ``PolyElement`` object to ``dtype``. """
        if K1.gens == K0.symbols:
            if K1.dom == K0.dom:
                return K1(dict(a))  # set the correct ring
            else:
                convert_dom = lambda c: K1.dom.convert_from(c, K0.dom)
                return K1._from_dict({m: convert_dom(c) for m, c in a.items()})
        else:
            monoms, coeffs = _dict_reorder(a.to_dict(), K0.symbols, K1.gens)

            if K1.dom != K0.dom:
                coeffs = [ K1.dom.convert(c, K0.dom) for c in coeffs ]

            return K1._from_dict(dict(zip(monoms, coeffs)))

    def from_GlobalPolynomialRing(K1, a, K0):
        """Convert a ``DMP`` object to ``dtype``. """
        if K1.gens == K0.gens:
            if K1.dom != K0.dom:
                a = a.convert(K1.dom)
            return K1(a.to_list())
        else:
            monoms, coeffs = _dict_reorder(a.to_dict(), K0.gens, K1.gens)

            if K1.dom != K0.dom:
                coeffs = [ K1.dom.convert(c, K0.dom) for c in coeffs ]

            return K1(dict(zip(monoms, coeffs)))

    def get_field(self):
        """Returns a field associated with ``self``. """
        return FractionField(self.dom, *self.gens)

    def poly_ring(self, *gens):
        """Returns a polynomial ring, i.e. ``K[X]``. """
        raise NotImplementedError('nested domains not allowed')

    def frac_field(self, *gens):
        """Returns a fraction field, i.e. ``K(X)``. """
        raise NotImplementedError('nested domains not allowed')

    def revert(self, a):
        try:
            return self.exquo(self.one, a)
        except (ExactQuotientFailed, ZeroDivisionError):
            raise NotReversible('%s is not a unit' % a)

    def gcdex(self, a, b):
        """Extended GCD of ``a`` and ``b``. """
        return a.gcdex(b)

    def gcd(self, a, b):
        """Returns GCD of ``a`` and ``b``. """
        return a.gcd(b)

    def lcm(self, a, b):
        """Returns LCM of ``a`` and ``b``. """
        return a.lcm(b)

    def factorial(self, a):
        """Returns factorial of ``a``. """
        return self.dtype(self.dom.factorial(a))

    def _vector_to_sdm(self, v, order):
        """
        For internal use by the modules class.

        Convert an iterable of elements of this ring into a sparse distributed
        module element.
        """
        raise NotImplementedError

    def _sdm_to_dics(self, s, n):
        """Helper for _sdm_to_vector."""
        from sympy.polys.distributedmodules import sdm_to_dict
        dic = sdm_to_dict(s)
        res = [{} for _ in range(n)]
        for k, v in dic.items():
            res[k[0]][k[1:]] = v
        return res

    def _sdm_to_vector(self, s, n):
        """
        For internal use by the modules class.

        Convert a sparse distributed module into a list of length ``n``.

        Examples
        ========

        >>> from sympy import QQ, ilex
        >>> from sympy.abc import x, y
        >>> R = QQ.old_poly_ring(x, y, order=ilex)
        >>> L = [((1, 1, 1), QQ(1)), ((0, 1, 0), QQ(1)), ((0, 0, 1), QQ(2))]
        >>> R._sdm_to_vector(L, 2)
        [DMF([[1], [2, 0]], [[1]], QQ), DMF([[1, 0], []], [[1]], QQ)]
        """
        dics = self._sdm_to_dics(s, n)
        # NOTE this works for global and local rings!
        return [self(x) for x in dics]

    def free_module(self, rank):
        """
        Generate a free module of rank ``rank`` over ``self``.

        Examples
        ========

        >>> from sympy.abc import x
        >>> from sympy import QQ
        >>> QQ.old_poly_ring(x).free_module(2)
        QQ[x]**2
        """
        return FreeModulePolyRing(self, rank)


def _vector_to_sdm_helper(v, order):
    """Helper method for common code in Global and Local poly rings."""
    from sympy.polys.distributedmodules import sdm_from_dict
    d = {}
    for i, e in enumerate(v):
        for key, value in e.to_dict().items():
            d[(i,) + key] = value
    return sdm_from_dict(d, order)


@public
class GlobalPolynomialRing(PolynomialRingBase):
    """A true polynomial ring, with objects DMP. """

    is_PolynomialRing = is_Poly = True
    dtype = DMP

    def new(self, element):
        if isinstance(element, dict):
            return DMP.from_dict(element, len(self.gens) - 1, self.dom)
        elif element in self.dom:
            return self._ground_new(self.dom.convert(element))
        else:
            return self.dtype(element, self.dom, len(self.gens) - 1)

    def from_FractionField(K1, a, K0):
        """
        Convert a ``DMF`` object to ``DMP``.

        Examples
        ========

        >>> from sympy.polys.polyclasses import DMP, DMF
        >>> from sympy.polys.domains import ZZ
        >>> from sympy.abc import x

        >>> f = DMF(([ZZ(1), ZZ(1)], [ZZ(1)]), ZZ)
        >>> K = ZZ.old_frac_field(x)

        >>> F = ZZ.old_poly_ring(x).from_FractionField(f, K)

        >>> F == DMP([ZZ(1), ZZ(1)], ZZ)
        True
        >>> type(F)  # doctest: +SKIP
        <class 'sympy.polys.polyclasses.DMP_Python'>

        """
        if a.denom().is_one:
            return K1.from_GlobalPolynomialRing(a.numer(), K0)

    def to_sympy(self, a):
        """Convert ``a`` to a SymPy object. """
        return basic_from_dict(a.to_sympy_dict(), *self.gens)

    def from_sympy(self, a):
        """Convert SymPy's expression to ``dtype``. """
        try:
            rep, _ = dict_from_basic(a, gens=self.gens)
        except PolynomialError:
            raise CoercionFailed("Cannot convert %s to type %s" % (a, self))

        for k, v in rep.items():
            rep[k] = self.dom.from_sympy(v)

        return DMP.from_dict(rep, self.ngens - 1, self.dom)

    def is_positive(self, a):
        """Returns True if ``LC(a)`` is positive. """
        return self.dom.is_positive(a.LC())

    def is_negative(self, a):
        """Returns True if ``LC(a)`` is negative. """
        return self.dom.is_negative(a.LC())

    def is_nonpositive(self, a):
        """Returns True if ``LC(a)`` is non-positive. """
        return self.dom.is_nonpositive(a.LC())

    def is_nonnegative(self, a):
        """Returns True if ``LC(a)`` is non-negative. """
        return self.dom.is_nonnegative(a.LC())

    def _vector_to_sdm(self, v, order):
        """
        Examples
        ========

        >>> from sympy import lex, QQ
        >>> from sympy.abc import x, y
        >>> R = QQ.old_poly_ring(x, y)
        >>> f = R.convert(x + 2*y)
        >>> g = R.convert(x * y)
        >>> R._vector_to_sdm([f, g], lex)
        [((1, 1, 1), 1), ((0, 1, 0), 1), ((0, 0, 1), 2)]
        """
        return _vector_to_sdm_helper(v, order)


class GeneralizedPolynomialRing(PolynomialRingBase):
    """A generalized polynomial ring, with objects DMF. """

    dtype = DMF

    def new(self, a):
        """Construct an element of ``self`` domain from ``a``. """
        res = self.dtype(a, self.dom, len(self.gens) - 1)

        # make sure res is actually in our ring
        if res.denom().terms(order=self.order)[0][0] != (0,)*len(self.gens):
            from sympy.printing.str import sstr
            raise CoercionFailed("denominator %s not allowed in %s"
                                 % (sstr(res), self))
        return res

    def __contains__(self, a):
        try:
            a = self.convert(a)
        except CoercionFailed:
            return False
        return a.denom().terms(order=self.order)[0][0] == (0,)*len(self.gens)

    def to_sympy(self, a):
        """Convert ``a`` to a SymPy object. """
        return (basic_from_dict(a.numer().to_sympy_dict(), *self.gens) /
                basic_from_dict(a.denom().to_sympy_dict(), *self.gens))

    def from_sympy(self, a):
        """Convert SymPy's expression to ``dtype``. """
        p, q = a.as_numer_denom()

        num, _ = dict_from_basic(p, gens=self.gens)
        den, _ = dict_from_basic(q, gens=self.gens)

        for k, v in num.items():
            num[k] = self.dom.from_sympy(v)

        for k, v in den.items():
            den[k] = self.dom.from_sympy(v)

        return self((num, den)).cancel()

    def exquo(self, a, b):
        """Exact quotient of ``a`` and ``b``. """
        # Elements are DMF that will always divide (except 0). The result is
        # not guaranteed to be in this ring, so we have to check that.
        r = a / b

        try:
            r = self.new((r.num, r.den))
        except CoercionFailed:
            raise ExactQuotientFailed(a, b, self)

        return r

    def from_FractionField(K1, a, K0):
        dmf = K1.get_field().from_FractionField(a, K0)
        return K1((dmf.num, dmf.den))

    def _vector_to_sdm(self, v, order):
        """
        Turn an iterable into a sparse distributed module.

        Note that the vector is multiplied by a unit first to make all entries
        polynomials.

        Examples
        ========

        >>> from sympy import ilex, QQ
        >>> from sympy.abc import x, y
        >>> R = QQ.old_poly_ring(x, y, order=ilex)
        >>> f = R.convert((x + 2*y) / (1 + x))
        >>> g = R.convert(x * y)
        >>> R._vector_to_sdm([f, g], ilex)
        [((0, 0, 1), 2), ((0, 1, 0), 1), ((1, 1, 1), 1), ((1,
          2, 1), 1)]
        """
        # NOTE this is quite inefficient...
        u = self.one.numer()
        for x in v:
            u *= x.denom()
        return _vector_to_sdm_helper([x.numer()*u/x.denom() for x in v], order)


@public
def PolynomialRing(dom, *gens, **opts):
    r"""
    Create a generalized multivariate polynomial ring.

    A generalized polynomial ring is defined by a ground field `K`, a set
    of generators (typically `x_1, \ldots, x_n`) and a monomial order `<`.
    The monomial order can be global, local or mixed. In any case it induces
    a total ordering on the monomials, and there exists for every (non-zero)
    polynomial `f \in K[x_1, \ldots, x_n]` a well-defined "leading monomial"
    `LM(f) = LM(f, >)`. One can then define a multiplicative subset
    `S = S_> = \{f \in K[x_1, \ldots, x_n] | LM(f) = 1\}`. The generalized
    polynomial ring corresponding to the monomial order is
    `R = S^{-1}K[x_1, \ldots, x_n]`.

    If `>` is a so-called global order, that is `1` is the smallest monomial,
    then we just have `S = K` and `R = K[x_1, \ldots, x_n]`.

    Examples
    ========

    A few examples may make this clearer.

    >>> from sympy.abc import x, y
    >>> from sympy import QQ

    Our first ring uses global lexicographic order.

    >>> R1 = QQ.old_poly_ring(x, y, order=(("lex", x, y),))

    The second ring uses local lexicographic order. Note that when using a
    single (non-product) order, you can just specify the name and omit the
    variables:

    >>> R2 = QQ.old_poly_ring(x, y, order="ilex")

    The third and fourth rings use a mixed orders:

    >>> o1 = (("ilex", x), ("lex", y))
    >>> o2 = (("lex", x), ("ilex", y))
    >>> R3 = QQ.old_poly_ring(x, y, order=o1)
    >>> R4 = QQ.old_poly_ring(x, y, order=o2)

    We will investigate what elements of `K(x, y)` are contained in the various
    rings.

    >>> L = [x, 1/x, y/(1 + x), 1/(1 + y), 1/(1 + x*y)]
    >>> test = lambda R: [f in R for f in L]

    The first ring is just `K[x, y]`:

    >>> test(R1)
    [True, False, False, False, False]

    The second ring is R1 localised at the maximal ideal (x, y):

    >>> test(R2)
    [True, False, True, True, True]

    The third ring is R1 localised at the prime ideal (x):

    >>> test(R3)
    [True, False, True, False, True]

    Finally the fourth ring is R1 localised at `S = K[x, y] \setminus yK[y]`:

    >>> test(R4)
    [True, False, False, True, False]
    """

    order = opts.get("order", GeneralizedPolynomialRing.default_order)
    if iterable(order):
        order = build_product_order(order, gens)
    order = monomial_key(order)
    opts['order'] = order

    if order.is_global:
        return GlobalPolynomialRing(dom, *gens, **opts)
    else:
        return GeneralizedPolynomialRing(dom, *gens, **opts)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\llvmjitcode.py ===
'''
Use llvmlite to create executable functions from SymPy expressions

This module requires llvmlite (https://github.com/numba/llvmlite).
'''

import ctypes

from sympy.external import import_module
from sympy.printing.printer import Printer
from sympy.core.singleton import S
from sympy.tensor.indexed import IndexedBase
from sympy.utilities.decorator import doctest_depends_on

llvmlite = import_module('llvmlite')
if llvmlite:
    ll = import_module('llvmlite.ir').ir
    llvm = import_module('llvmlite.binding').binding
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()


__doctest_requires__ = {('llvm_callable'): ['llvmlite']}


class LLVMJitPrinter(Printer):
    '''Convert expressions to LLVM IR'''
    def __init__(self, module, builder, fn, *args, **kwargs):
        self.func_arg_map = kwargs.pop("func_arg_map", {})
        if not llvmlite:
            raise ImportError("llvmlite is required for LLVMJITPrinter")
        super().__init__(*args, **kwargs)
        self.fp_type = ll.DoubleType()
        self.module = module
        self.builder = builder
        self.fn = fn
        self.ext_fn = {}  # keep track of wrappers to external functions
        self.tmp_var = {}

    def _add_tmp_var(self, name, value):
        self.tmp_var[name] = value

    def _print_Number(self, n):
        return ll.Constant(self.fp_type, float(n))

    def _print_Integer(self, expr):
        return ll.Constant(self.fp_type, float(expr.p))

    def _print_Symbol(self, s):
        val = self.tmp_var.get(s)
        if not val:
            # look up parameter with name s
            val = self.func_arg_map.get(s)
        if not val:
            raise LookupError("Symbol not found: %s" % s)
        return val

    def _print_Pow(self, expr):
        base0 = self._print(expr.base)
        if expr.exp == S.NegativeOne:
            return self.builder.fdiv(ll.Constant(self.fp_type, 1.0), base0)
        if expr.exp == S.Half:
            fn = self.ext_fn.get("sqrt")
            if not fn:
                fn_type = ll.FunctionType(self.fp_type, [self.fp_type])
                fn = ll.Function(self.module, fn_type, "sqrt")
                self.ext_fn["sqrt"] = fn
            return self.builder.call(fn, [base0], "sqrt")
        if expr.exp == 2:
            return self.builder.fmul(base0, base0)

        exp0 = self._print(expr.exp)
        fn = self.ext_fn.get("pow")
        if not fn:
            fn_type = ll.FunctionType(self.fp_type, [self.fp_type, self.fp_type])
            fn = ll.Function(self.module, fn_type, "pow")
            self.ext_fn["pow"] = fn
        return self.builder.call(fn, [base0, exp0], "pow")

    def _print_Mul(self, expr):
        nodes = [self._print(a) for a in expr.args]
        e = nodes[0]
        for node in nodes[1:]:
            e = self.builder.fmul(e, node)
        return e

    def _print_Add(self, expr):
        nodes = [self._print(a) for a in expr.args]
        e = nodes[0]
        for node in nodes[1:]:
            e = self.builder.fadd(e, node)
        return e

    # TODO - assumes all called functions take one double precision argument.
    #        Should have a list of math library functions to validate this.
    def _print_Function(self, expr):
        name = expr.func.__name__
        e0 = self._print(expr.args[0])
        fn = self.ext_fn.get(name)
        if not fn:
            fn_type = ll.FunctionType(self.fp_type, [self.fp_type])
            fn = ll.Function(self.module, fn_type, name)
            self.ext_fn[name] = fn
        return self.builder.call(fn, [e0], name)

    def emptyPrinter(self, expr):
        raise TypeError("Unsupported type for LLVM JIT conversion: %s"
                        % type(expr))


# Used when parameters are passed by array.  Often used in callbacks to
# handle a variable number of parameters.
class LLVMJitCallbackPrinter(LLVMJitPrinter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _print_Indexed(self, expr):
        array, idx = self.func_arg_map[expr.base]
        offset = int(expr.indices[0].evalf())
        array_ptr = self.builder.gep(array, [ll.Constant(ll.IntType(32), offset)])
        fp_array_ptr = self.builder.bitcast(array_ptr, ll.PointerType(self.fp_type))
        value = self.builder.load(fp_array_ptr)
        return value

    def _print_Symbol(self, s):
        val = self.tmp_var.get(s)
        if val:
            return val

        array, idx = self.func_arg_map.get(s, [None, 0])
        if not array:
            raise LookupError("Symbol not found: %s" % s)
        array_ptr = self.builder.gep(array, [ll.Constant(ll.IntType(32), idx)])
        fp_array_ptr = self.builder.bitcast(array_ptr,
                                            ll.PointerType(self.fp_type))
        value = self.builder.load(fp_array_ptr)
        return value


# ensure lifetime of the execution engine persists (else call to compiled
#   function will seg fault)
exe_engines = []

# ensure names for generated functions are unique
link_names = set()
current_link_suffix = 0


class LLVMJitCode:
    def __init__(self, signature):
        self.signature = signature
        self.fp_type = ll.DoubleType()
        self.module = ll.Module('mod1')
        self.fn = None
        self.llvm_arg_types = []
        self.llvm_ret_type = self.fp_type
        self.param_dict = {}  # map symbol name to LLVM function argument
        self.link_name = ''

    def _from_ctype(self, ctype):
        if ctype == ctypes.c_int:
            return ll.IntType(32)
        if ctype == ctypes.c_double:
            return self.fp_type
        if ctype == ctypes.POINTER(ctypes.c_double):
            return ll.PointerType(self.fp_type)
        if ctype == ctypes.c_void_p:
            return ll.PointerType(ll.IntType(32))
        if ctype == ctypes.py_object:
            return ll.PointerType(ll.IntType(32))

        print("Unhandled ctype = %s" % str(ctype))

    def _create_args(self, func_args):
        """Create types for function arguments"""
        self.llvm_ret_type = self._from_ctype(self.signature.ret_type)
        self.llvm_arg_types = \
            [self._from_ctype(a) for a in self.signature.arg_ctypes]

    def _create_function_base(self):
        """Create function with name and type signature"""
        global current_link_suffix
        default_link_name = 'jit_func'
        current_link_suffix += 1
        self.link_name = default_link_name + str(current_link_suffix)
        link_names.add(self.link_name)

        fn_type = ll.FunctionType(self.llvm_ret_type, self.llvm_arg_types)
        self.fn = ll.Function(self.module, fn_type, name=self.link_name)

    def _create_param_dict(self, func_args):
        """Mapping of symbolic values to function arguments"""
        for i, a in enumerate(func_args):
            self.fn.args[i].name = str(a)
            self.param_dict[a] = self.fn.args[i]

    def _create_function(self, expr):
        """Create function body and return LLVM IR"""
        bb_entry = self.fn.append_basic_block('entry')
        builder = ll.IRBuilder(bb_entry)

        lj = LLVMJitPrinter(self.module, builder, self.fn,
                            func_arg_map=self.param_dict)

        ret = self._convert_expr(lj, expr)
        lj.builder.ret(self._wrap_return(lj, ret))

        strmod = str(self.module)
        return strmod

    def _wrap_return(self, lj, vals):
        # Return a single double if there is one return value,
        #  else return a tuple of doubles.

        # Don't wrap return value in this case
        if self.signature.ret_type == ctypes.c_double:
            return vals[0]

        # Use this instead of a real PyObject*
        void_ptr = ll.PointerType(ll.IntType(32))

        # Create a wrapped double: PyObject* PyFloat_FromDouble(double v)
        wrap_type = ll.FunctionType(void_ptr, [self.fp_type])
        wrap_fn = ll.Function(lj.module, wrap_type, "PyFloat_FromDouble")

        wrapped_vals = [lj.builder.call(wrap_fn, [v]) for v in vals]
        if len(vals) == 1:
            final_val = wrapped_vals[0]
        else:
            # Create a tuple: PyObject* PyTuple_Pack(Py_ssize_t n, ...)

            # This should be Py_ssize_t
            tuple_arg_types = [ll.IntType(32)]

            tuple_arg_types.extend([void_ptr]*len(vals))
            tuple_type = ll.FunctionType(void_ptr, tuple_arg_types)
            tuple_fn = ll.Function(lj.module, tuple_type, "PyTuple_Pack")

            tuple_args = [ll.Constant(ll.IntType(32), len(wrapped_vals))]
            tuple_args.extend(wrapped_vals)

            final_val = lj.builder.call(tuple_fn, tuple_args)

        return final_val

    def _convert_expr(self, lj, expr):
        try:
            # Match CSE return data structure.
            if len(expr) == 2:
                tmp_exprs = expr[0]
                final_exprs = expr[1]
                if len(final_exprs) != 1 and self.signature.ret_type == ctypes.c_double:
                    raise NotImplementedError("Return of multiple expressions not supported for this callback")
                for name, e in tmp_exprs:
                    val = lj._print(e)
                    lj._add_tmp_var(name, val)
        except TypeError:
            final_exprs = [expr]

        vals = [lj._print(e) for e in final_exprs]

        return vals

    def _compile_function(self, strmod):
        llmod = llvm.parse_assembly(strmod)

        pmb = llvm.create_pass_manager_builder()
        pmb.opt_level = 2
        pass_manager = llvm.create_module_pass_manager()
        pmb.populate(pass_manager)

        pass_manager.run(llmod)

        target_machine = \
            llvm.Target.from_default_triple().create_target_machine()
        exe_eng = llvm.create_mcjit_compiler(llmod, target_machine)
        exe_eng.finalize_object()
        exe_engines.append(exe_eng)

        if False:
            print("Assembly")
            print(target_machine.emit_assembly(llmod))

        fptr = exe_eng.get_function_address(self.link_name)

        return fptr


class LLVMJitCodeCallback(LLVMJitCode):
    def __init__(self, signature):
        super().__init__(signature)

    def _create_param_dict(self, func_args):
        for i, a in enumerate(func_args):
            if isinstance(a, IndexedBase):
                self.param_dict[a] = (self.fn.args[i], i)
                self.fn.args[i].name = str(a)
            else:
                self.param_dict[a] = (self.fn.args[self.signature.input_arg],
                                      i)

    def _create_function(self, expr):
        """Create function body and return LLVM IR"""
        bb_entry = self.fn.append_basic_block('entry')
        builder = ll.IRBuilder(bb_entry)

        lj = LLVMJitCallbackPrinter(self.module, builder, self.fn,
                                    func_arg_map=self.param_dict)

        ret = self._convert_expr(lj, expr)

        if self.signature.ret_arg:
            output_fp_ptr = builder.bitcast(self.fn.args[self.signature.ret_arg],
                                            ll.PointerType(self.fp_type))
            for i, val in enumerate(ret):
                index = ll.Constant(ll.IntType(32), i)
                output_array_ptr = builder.gep(output_fp_ptr, [index])
                builder.store(val, output_array_ptr)
            builder.ret(ll.Constant(ll.IntType(32), 0))  # return success
        else:
            lj.builder.ret(self._wrap_return(lj, ret))

        strmod = str(self.module)
        return strmod


class CodeSignature:
    def __init__(self, ret_type):
        self.ret_type = ret_type
        self.arg_ctypes = []

        # Input argument array element index
        self.input_arg = 0

        # For the case output value is referenced through a parameter rather
        # than the return value
        self.ret_arg = None


def _llvm_jit_code(args, expr, signature, callback_type):
    """Create a native code function from a SymPy expression"""
    if callback_type is None:
        jit = LLVMJitCode(signature)
    else:
        jit = LLVMJitCodeCallback(signature)

    jit._create_args(args)
    jit._create_function_base()
    jit._create_param_dict(args)
    strmod = jit._create_function(expr)
    if False:
        print("LLVM IR")
        print(strmod)
    fptr = jit._compile_function(strmod)
    return fptr


@doctest_depends_on(modules=('llvmlite', 'scipy'))
def llvm_callable(args, expr, callback_type=None):
    '''Compile function from a SymPy expression

    Expressions are evaluated using double precision arithmetic.
    Some single argument math functions (exp, sin, cos, etc.) are supported
    in expressions.

    Parameters
    ==========

    args : List of Symbol
        Arguments to the generated function.  Usually the free symbols in
        the expression.  Currently each one is assumed to convert to
        a double precision scalar.
    expr : Expr, or (Replacements, Expr) as returned from 'cse'
        Expression to compile.
    callback_type : string
        Create function with signature appropriate to use as a callback.
        Currently supported:
           'scipy.integrate'
           'scipy.integrate.test'
           'cubature'

    Returns
    =======

    Compiled function that can evaluate the expression.

    Examples
    ========

    >>> import sympy.printing.llvmjitcode as jit
    >>> from sympy.abc import a
    >>> e = a*a + a + 1
    >>> e1 = jit.llvm_callable([a], e)
    >>> e.subs(a, 1.1)   # Evaluate via substitution
    3.31000000000000
    >>> e1(1.1)  # Evaluate using JIT-compiled code
    3.3100000000000005


    Callbacks for integration functions can be JIT compiled.

    >>> import sympy.printing.llvmjitcode as jit
    >>> from sympy.abc import a
    >>> from sympy import integrate
    >>> from scipy.integrate import quad
    >>> e = a*a
    >>> e1 = jit.llvm_callable([a], e, callback_type='scipy.integrate')
    >>> integrate(e, (a, 0.0, 2.0))
    2.66666666666667
    >>> quad(e1, 0.0, 2.0)[0]
    2.66666666666667

    The 'cubature' callback is for the Python wrapper around the
    cubature package ( https://github.com/saullocastro/cubature )
    and ( http://ab-initio.mit.edu/wiki/index.php/Cubature )

    There are two signatures for the SciPy integration callbacks.
    The first ('scipy.integrate') is the function to be passed to the
    integration routine, and will pass the signature checks.
    The second ('scipy.integrate.test') is only useful for directly calling
    the function using ctypes variables. It will not pass the signature checks
    for scipy.integrate.

    The return value from the cse module can also be compiled.  This
    can improve the performance of the compiled function.  If multiple
    expressions are given to cse, the compiled function returns a tuple.
    The 'cubature' callback handles multiple expressions (set `fdim`
    to match in the integration call.)

    >>> import sympy.printing.llvmjitcode as jit
    >>> from sympy import cse
    >>> from sympy.abc import x,y
    >>> e1 = x*x + y*y
    >>> e2 = 4*(x*x + y*y) + 8.0
    >>> after_cse = cse([e1,e2])
    >>> after_cse
    ([(x0, x**2), (x1, y**2)], [x0 + x1, 4*x0 + 4*x1 + 8.0])
    >>> j1 = jit.llvm_callable([x,y], after_cse)
    >>> j1(1.0, 2.0)
    (5.0, 28.0)
    '''

    if not llvmlite:
        raise ImportError("llvmlite is required for llvmjitcode")

    signature = CodeSignature(ctypes.py_object)

    arg_ctypes = []
    if callback_type is None:
        for _ in args:
            arg_ctype = ctypes.c_double
            arg_ctypes.append(arg_ctype)
    elif callback_type in ('scipy.integrate', 'scipy.integrate.test'):
        signature.ret_type = ctypes.c_double
        arg_ctypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double)]
        arg_ctypes_formal = [ctypes.c_int, ctypes.c_double]
        signature.input_arg = 1
    elif callback_type == 'cubature':
        arg_ctypes = [ctypes.c_int,
                      ctypes.POINTER(ctypes.c_double),
                      ctypes.c_void_p,
                      ctypes.c_int,
                      ctypes.POINTER(ctypes.c_double)
                      ]
        signature.ret_type = ctypes.c_int
        signature.input_arg = 1
        signature.ret_arg = 4
    else:
        raise ValueError("Unknown callback type: %s" % callback_type)

    signature.arg_ctypes = arg_ctypes

    fptr = _llvm_jit_code(args, expr, signature, callback_type)

    if callback_type and callback_type == 'scipy.integrate':
        arg_ctypes = arg_ctypes_formal

    # PYFUNCTYPE holds the GIL which is needed to prevent a segfault when
    # calling PyFloat_FromDouble on Python 3.10. Probably it is better to use
    # ctypes.c_double when returning a float rather than using ctypes.py_object
    # and returning a PyFloat from inside the jitted function (i.e. let ctypes
    # handle the conversion from double to PyFloat).
    if signature.ret_type == ctypes.py_object:
        FUNCTYPE = ctypes.PYFUNCTYPE
    else:
        FUNCTYPE = ctypes.CFUNCTYPE

    cfunc = FUNCTYPE(signature.ret_type, *arg_ctypes)(fptr)
    return cfunc

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\_ufunc_config.py ===
"""
Functions for changing global ufunc configuration

This provides helpers which wrap `_get_extobj_dict` and `_make_extobj`, and
`_extobj_contextvar` from umath.
"""
import functools

from numpy._utils import set_module

from .umath import _extobj_contextvar, _get_extobj_dict, _make_extobj

__all__ = [
    "seterr", "geterr", "setbufsize", "getbufsize", "seterrcall", "geterrcall",
    "errstate"
]


@set_module('numpy')
def seterr(all=None, divide=None, over=None, under=None, invalid=None):
    """
    Set how floating-point errors are handled.

    Note that operations on integer scalar types (such as `int16`) are
    handled like floating point, and are affected by these settings.

    Parameters
    ----------
    all : {'ignore', 'warn', 'raise', 'call', 'print', 'log'}, optional
        Set treatment for all types of floating-point errors at once:

        - ignore: Take no action when the exception occurs.
        - warn: Print a :exc:`RuntimeWarning` (via the Python `warnings`
          module).
        - raise: Raise a :exc:`FloatingPointError`.
        - call: Call a function specified using the `seterrcall` function.
        - print: Print a warning directly to ``stdout``.
        - log: Record error in a Log object specified by `seterrcall`.

        The default is not to change the current behavior.
    divide : {'ignore', 'warn', 'raise', 'call', 'print', 'log'}, optional
        Treatment for division by zero.
    over : {'ignore', 'warn', 'raise', 'call', 'print', 'log'}, optional
        Treatment for floating-point overflow.
    under : {'ignore', 'warn', 'raise', 'call', 'print', 'log'}, optional
        Treatment for floating-point underflow.
    invalid : {'ignore', 'warn', 'raise', 'call', 'print', 'log'}, optional
        Treatment for invalid floating-point operation.

    Returns
    -------
    old_settings : dict
        Dictionary containing the old settings.

    See also
    --------
    seterrcall : Set a callback function for the 'call' mode.
    geterr, geterrcall, errstate

    Notes
    -----
    The floating-point exceptions are defined in the IEEE 754 standard [1]_:

    - Division by zero: infinite result obtained from finite numbers.
    - Overflow: result too large to be expressed.
    - Underflow: result so close to zero that some precision
      was lost.
    - Invalid operation: result is not an expressible number, typically
      indicates that a NaN was produced.

    .. [1] https://en.wikipedia.org/wiki/IEEE_754

    Examples
    --------
    >>> import numpy as np
    >>> orig_settings = np.seterr(all='ignore')  # seterr to known value
    >>> np.int16(32000) * np.int16(3)
    np.int16(30464)
    >>> np.seterr(over='raise')
    {'divide': 'ignore', 'over': 'ignore', 'under': 'ignore', 'invalid': 'ignore'}
    >>> old_settings = np.seterr(all='warn', over='raise')
    >>> np.int16(32000) * np.int16(3)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    FloatingPointError: overflow encountered in scalar multiply

    >>> old_settings = np.seterr(all='print')
    >>> np.geterr()
    {'divide': 'print', 'over': 'print', 'under': 'print', 'invalid': 'print'}
    >>> np.int16(32000) * np.int16(3)
    np.int16(30464)
    >>> np.seterr(**orig_settings)  # restore original
    {'divide': 'print', 'over': 'print', 'under': 'print', 'invalid': 'print'}

    """

    old = _get_extobj_dict()
    # The errstate doesn't include call and bufsize, so pop them:
    old.pop("call", None)
    old.pop("bufsize", None)

    extobj = _make_extobj(
            all=all, divide=divide, over=over, under=under, invalid=invalid)
    _extobj_contextvar.set(extobj)
    return old


@set_module('numpy')
def geterr():
    """
    Get the current way of handling floating-point errors.

    Returns
    -------
    res : dict
        A dictionary with keys "divide", "over", "under", and "invalid",
        whose values are from the strings "ignore", "print", "log", "warn",
        "raise", and "call". The keys represent possible floating-point
        exceptions, and the values define how these exceptions are handled.

    See Also
    --------
    geterrcall, seterr, seterrcall

    Notes
    -----
    For complete documentation of the types of floating-point exceptions and
    treatment options, see `seterr`.

    Examples
    --------
    >>> import numpy as np
    >>> np.geterr()
    {'divide': 'warn', 'over': 'warn', 'under': 'ignore', 'invalid': 'warn'}
    >>> np.arange(3.) / np.arange(3.)  # doctest: +SKIP
    array([nan,  1.,  1.])
    RuntimeWarning: invalid value encountered in divide

    >>> oldsettings = np.seterr(all='warn', invalid='raise')
    >>> np.geterr()
    {'divide': 'warn', 'over': 'warn', 'under': 'warn', 'invalid': 'raise'}
    >>> np.arange(3.) / np.arange(3.)
    Traceback (most recent call last):
      ...
    FloatingPointError: invalid value encountered in divide
    >>> oldsettings = np.seterr(**oldsettings)  # restore original

    """
    res = _get_extobj_dict()
    # The "geterr" doesn't include call and bufsize,:
    res.pop("call", None)
    res.pop("bufsize", None)
    return res


@set_module('numpy')
def setbufsize(size):
    """
    Set the size of the buffer used in ufuncs.

    .. versionchanged:: 2.0
        The scope of setting the buffer is tied to the `numpy.errstate`
        context.  Exiting a ``with errstate():`` will also restore the bufsize.

    Parameters
    ----------
    size : int
        Size of buffer.

    Returns
    -------
    bufsize : int
        Previous size of ufunc buffer in bytes.

    Examples
    --------
    When exiting a `numpy.errstate` context manager the bufsize is restored:

    >>> import numpy as np
    >>> with np.errstate():
    ...     np.setbufsize(4096)
    ...     print(np.getbufsize())
    ...
    8192
    4096
    >>> np.getbufsize()
    8192

    """
    old = _get_extobj_dict()["bufsize"]
    extobj = _make_extobj(bufsize=size)
    _extobj_contextvar.set(extobj)
    return old


@set_module('numpy')
def getbufsize():
    """
    Return the size of the buffer used in ufuncs.

    Returns
    -------
    getbufsize : int
        Size of ufunc buffer in bytes.

    Examples
    --------
    >>> import numpy as np
    >>> np.getbufsize()
    8192

    """
    return _get_extobj_dict()["bufsize"]


@set_module('numpy')
def seterrcall(func):
    """
    Set the floating-point error callback function or log object.

    There are two ways to capture floating-point error messages.  The first
    is to set the error-handler to 'call', using `seterr`.  Then, set
    the function to call using this function.

    The second is to set the error-handler to 'log', using `seterr`.
    Floating-point errors then trigger a call to the 'write' method of
    the provided object.

    Parameters
    ----------
    func : callable f(err, flag) or object with write method
        Function to call upon floating-point errors ('call'-mode) or
        object whose 'write' method is used to log such message ('log'-mode).

        The call function takes two arguments. The first is a string describing
        the type of error (such as "divide by zero", "overflow", "underflow",
        or "invalid value"), and the second is the status flag.  The flag is a
        byte, whose four least-significant bits indicate the type of error, one
        of "divide", "over", "under", "invalid"::

          [0 0 0 0 divide over under invalid]

        In other words, ``flags = divide + 2*over + 4*under + 8*invalid``.

        If an object is provided, its write method should take one argument,
        a string.

    Returns
    -------
    h : callable, log instance or None
        The old error handler.

    See Also
    --------
    seterr, geterr, geterrcall

    Examples
    --------
    Callback upon error:

    >>> def err_handler(type, flag):
    ...     print("Floating point error (%s), with flag %s" % (type, flag))
    ...

    >>> import numpy as np

    >>> orig_handler = np.seterrcall(err_handler)
    >>> orig_err = np.seterr(all='call')

    >>> np.array([1, 2, 3]) / 0.0
    Floating point error (divide by zero), with flag 1
    array([inf, inf, inf])

    >>> np.seterrcall(orig_handler)
    <function err_handler at 0x...>
    >>> np.seterr(**orig_err)
    {'divide': 'call', 'over': 'call', 'under': 'call', 'invalid': 'call'}

    Log error message:

    >>> class Log:
    ...     def write(self, msg):
    ...         print("LOG: %s" % msg)
    ...

    >>> log = Log()
    >>> saved_handler = np.seterrcall(log)
    >>> save_err = np.seterr(all='log')

    >>> np.array([1, 2, 3]) / 0.0
    LOG: Warning: divide by zero encountered in divide
    array([inf, inf, inf])

    >>> np.seterrcall(orig_handler)
    <numpy.Log object at 0x...>
    >>> np.seterr(**orig_err)
    {'divide': 'log', 'over': 'log', 'under': 'log', 'invalid': 'log'}

    """
    old = _get_extobj_dict()["call"]
    extobj = _make_extobj(call=func)
    _extobj_contextvar.set(extobj)
    return old


@set_module('numpy')
def geterrcall():
    """
    Return the current callback function used on floating-point errors.

    When the error handling for a floating-point error (one of "divide",
    "over", "under", or "invalid") is set to 'call' or 'log', the function
    that is called or the log instance that is written to is returned by
    `geterrcall`. This function or log instance has been set with
    `seterrcall`.

    Returns
    -------
    errobj : callable, log instance or None
        The current error handler. If no handler was set through `seterrcall`,
        ``None`` is returned.

    See Also
    --------
    seterrcall, seterr, geterr

    Notes
    -----
    For complete documentation of the types of floating-point exceptions and
    treatment options, see `seterr`.

    Examples
    --------
    >>> import numpy as np
    >>> np.geterrcall()  # we did not yet set a handler, returns None

    >>> orig_settings = np.seterr(all='call')
    >>> def err_handler(type, flag):
    ...     print("Floating point error (%s), with flag %s" % (type, flag))
    >>> old_handler = np.seterrcall(err_handler)
    >>> np.array([1, 2, 3]) / 0.0
    Floating point error (divide by zero), with flag 1
    array([inf, inf, inf])

    >>> cur_handler = np.geterrcall()
    >>> cur_handler is err_handler
    True
    >>> old_settings = np.seterr(**orig_settings)  # restore original
    >>> old_handler = np.seterrcall(None)  # restore original

    """
    return _get_extobj_dict()["call"]


class _unspecified:
    pass


_Unspecified = _unspecified()


@set_module('numpy')
class errstate:
    """
    errstate(**kwargs)

    Context manager for floating-point error handling.

    Using an instance of `errstate` as a context manager allows statements in
    that context to execute with a known error handling behavior. Upon entering
    the context the error handling is set with `seterr` and `seterrcall`, and
    upon exiting it is reset to what it was before.

    ..  versionchanged:: 1.17.0
        `errstate` is also usable as a function decorator, saving
        a level of indentation if an entire function is wrapped.

    .. versionchanged:: 2.0
        `errstate` is now fully thread and asyncio safe, but may not be
        entered more than once.
        It is not safe to decorate async functions using ``errstate``.

    Parameters
    ----------
    kwargs : {divide, over, under, invalid}
        Keyword arguments. The valid keywords are the possible floating-point
        exceptions. Each keyword should have a string value that defines the
        treatment for the particular error. Possible values are
        {'ignore', 'warn', 'raise', 'call', 'print', 'log'}.

    See Also
    --------
    seterr, geterr, seterrcall, geterrcall

    Notes
    -----
    For complete documentation of the types of floating-point exceptions and
    treatment options, see `seterr`.

    Examples
    --------
    >>> import numpy as np
    >>> olderr = np.seterr(all='ignore')  # Set error handling to known state.

    >>> np.arange(3) / 0.
    array([nan, inf, inf])
    >>> with np.errstate(divide='ignore'):
    ...     np.arange(3) / 0.
    array([nan, inf, inf])

    >>> np.sqrt(-1)
    np.float64(nan)
    >>> with np.errstate(invalid='raise'):
    ...     np.sqrt(-1)
    Traceback (most recent call last):
      File "<stdin>", line 2, in <module>
    FloatingPointError: invalid value encountered in sqrt

    Outside the context the error handling behavior has not changed:

    >>> np.geterr()
    {'divide': 'ignore', 'over': 'ignore', 'under': 'ignore', 'invalid': 'ignore'}
    >>> olderr = np.seterr(**olderr)  # restore original state

    """
    __slots__ = (
        "_all",
        "_call",
        "_divide",
        "_invalid",
        "_over",
        "_token",
        "_under",
    )

    def __init__(self, *, call=_Unspecified,
                 all=None, divide=None, over=None, under=None, invalid=None):
        self._token = None
        self._call = call
        self._all = all
        self._divide = divide
        self._over = over
        self._under = under
        self._invalid = invalid

    def __enter__(self):
        # Note that __call__ duplicates much of this logic
        if self._token is not None:
            raise TypeError("Cannot enter `np.errstate` twice.")
        if self._call is _Unspecified:
            extobj = _make_extobj(
                    all=self._all, divide=self._divide, over=self._over,
                    under=self._under, invalid=self._invalid)
        else:
            extobj = _make_extobj(
                    call=self._call,
                    all=self._all, divide=self._divide, over=self._over,
                    under=self._under, invalid=self._invalid)

        self._token = _extobj_contextvar.set(extobj)

    def __exit__(self, *exc_info):
        _extobj_contextvar.reset(self._token)

    def __call__(self, func):
        # We need to customize `__call__` compared to `ContextDecorator`
        # because we must store the token per-thread so cannot store it on
        # the instance (we could create a new instance for this).
        # This duplicates the code from `__enter__`.
        @functools.wraps(func)
        def inner(*args, **kwargs):
            if self._call is _Unspecified:
                extobj = _make_extobj(
                        all=self._all, divide=self._divide, over=self._over,
                        under=self._under, invalid=self._invalid)
            else:
                extobj = _make_extobj(
                        call=self._call,
                        all=self._all, divide=self._divide, over=self._over,
                        under=self._under, invalid=self._invalid)

            _token = _extobj_contextvar.set(extobj)
            try:
                # Call the original, decorated, function:
                return func(*args, **kwargs)
            finally:
                _extobj_contextvar.reset(_token)

        return inner

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\fusion_layernorm.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from logging import getLogger

from fusion_base import Fusion
from onnx import TensorProto, helper
from onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionLayerNormalization(Fusion):
    def __init__(self, model: OnnxModel, check_constant_and_dimension: bool = True, force: bool = False):
        super().__init__(model, "LayerNormalization", "ReduceMean")
        self.check_constant_and_dimension = check_constant_and_dimension
        self.force = force

    def fuse(self, node, input_name_to_nodes: dict, output_name_to_node: dict):
        """
        Fuse Layer Normalization subgraph into one node LayerNormalization:
              +----------------------+
              |                      |
              |                      v
          [Root] --> ReduceMean -->  Sub  --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul --> Add
                     (axis=2 or -1)  |      (Y=2)   (axis=2 or -1)  (B=E-6 or E-12)    ^
                                     |                                                 |
                                     +-------------------------------------------------+

         It also handles cases of duplicated sub nodes exported from older version of PyTorch:
              +----------------------+
              |                      v
              |           +-------> Sub-----------------------------------------------+
              |           |                                                           |
              |           |                                                           v
          [Root] --> ReduceMean -->  Sub  --> Pow --> ReduceMean --> Add --> Sqrt --> Div  --> Mul --> Add
              |                      ^
              |                      |
              +----------------------+
        """
        subgraph_nodes = []
        children = self.model.get_children(node, input_name_to_nodes)
        if len(children) == 0 or len(children) > 2:
            return

        root_input = node.input[0]

        if children[0].op_type != "Sub" or children[0].input[0] != root_input:
            return

        if len(children) == 2:
            if children[1].op_type != "Sub" or children[1].input[0] != root_input:
                return

        div_node = None
        for child in children:
            # Check if Sub --> Div exists
            div_node_1 = self.model.find_first_child_by_type(child, "Div", input_name_to_nodes, recursive=False)
            if div_node_1 is not None:
                div_node = div_node_1
                break
            else:
                # Check if Sub --> Cast --> Div
                div_node_2 = self.model.match_child_path(child, ["Cast", "Div"])
                if div_node_2 is not None:
                    div_node = div_node_2[-1]
                    break

        if div_node is None:
            return

        _path_id, parent_nodes, _ = self.model.match_parent_paths(
            div_node,
            [
                (["Sqrt", "Add", "ReduceMean", "Pow", "Sub"], [1, 0, 0, 0, 0]),
                (["Sqrt", "Add", "ReduceMean", "Pow", "Cast", "Sub"], [1, 0, 0, 0, 0, 0]),
            ],
            output_name_to_node,
        )
        if parent_nodes is None:
            return

        sub_node = parent_nodes[-1]
        if sub_node not in children:
            return

        add_eps_node = parent_nodes[1]
        i, epsilon = self.model.get_constant_input(add_eps_node)
        if epsilon is None or epsilon <= 0 or epsilon > 1.0e-4:
            logger.debug(f"skip SkipLayerNormalization fusion since epsilon value is not expected: {epsilon}")
            return

        pow_node = parent_nodes[3]
        if self.model.find_constant_input(pow_node, 2.0) != 1:
            return

        if div_node.output[0] not in input_name_to_nodes:
            return

        # In MMDit model, Div might have two Mul+Add children paths.
        div_children = input_name_to_nodes[div_node.output[0]]
        for temp_node in div_children:
            if temp_node.op_type == "Cast":
                # Div --> Cast --> Mul
                subgraph_nodes.append(temp_node)  # add Cast node to list of subgraph nodes
                if temp_node.output[0] not in input_name_to_nodes:
                    continue
                mul_node = input_name_to_nodes[temp_node.output[0]][0]
            else:
                # Div --> Mul
                mul_node = temp_node
            if mul_node.op_type != "Mul":
                continue

            if mul_node.output[0] not in input_name_to_nodes:
                continue
            last_add_node = input_name_to_nodes[mul_node.output[0]][0]
            if last_add_node.op_type != "Add":
                continue

            subgraph_nodes.append(node)
            subgraph_nodes.extend(children)
            subgraph_nodes.extend(parent_nodes[:-1])

            subgraph_nodes.extend([last_add_node, mul_node, div_node])

            node_before_weight = div_node if temp_node.op_type != "Cast" else temp_node
            weight_input = mul_node.input[1 - self.model.input_index(node_before_weight.output[0], mul_node)]
            if self.check_constant_and_dimension and not self.model.is_constant_with_specified_dimension(
                weight_input, 1, "layernorm weight"
            ):
                continue

            bias_input = last_add_node.input[1 - self.model.input_index(mul_node.output[0], last_add_node)]
            if self.check_constant_and_dimension and not self.model.is_constant_with_specified_dimension(
                bias_input, 1, "layernorm bias"
            ):
                continue

            layer_norm_output = last_add_node.output[0]
            if not self.model.is_safe_to_fuse_nodes(
                subgraph_nodes,
                last_add_node.output,
                input_name_to_nodes,
                output_name_to_node,
            ):
                # If it is not safe to fuse, somce computation may be duplicated if we force to fuse it.
                # It it unknown that force fusion might bring performance gain/loss.
                # User need test performance impact to see whether forcing fusion can help.
                if self.force:
                    self.prune_graph = True
                else:
                    logger.debug("It is not safe to fuse LayerNormalization node. Skip")
                    continue
            else:
                self.nodes_to_remove.extend(subgraph_nodes)

            normalize_node = helper.make_node(
                "LayerNormalization",
                inputs=[node.input[0], weight_input, bias_input],
                outputs=[layer_norm_output],
                name=self.model.create_node_name("LayerNormalization", name_prefix="LayerNorm"),
            )
            normalize_node.attribute.extend([helper.make_attribute("epsilon", float(epsilon))])
            self.nodes_to_add.append(normalize_node)
            self.node_name_to_graph_name[normalize_node.name] = self.this_graph_name


class FusionLayerNormalizationNCHW(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "LayerNormalization", "ReduceMean")

    def get_weight_or_bias(self, output_name, description):
        value = self.model.get_constant_value(output_name)
        if value is None:
            logger.debug(f"{description} {output_name} is not initializer.")
            return None

        if len(value.shape) != 3 or value.shape[1] != 1 or value.shape[2] != 1:
            logger.debug(f"{description} {output_name} shall have 3 dimensions Cx1x1. Got shape {value.shape}")
            return None

        return value.reshape([value.shape[0]])

    def create_transpose_node(self, input_name: str, perm: list[int], output_name=None):
        """Append a Transpose node after an input"""
        node_name = self.model.create_node_name("Transpose")

        if output_name is None:
            output_name = node_name + "_out" + "-" + input_name

        transpose_node = helper.make_node("Transpose", inputs=[input_name], outputs=[output_name], name=node_name)
        transpose_node.attribute.extend([helper.make_attribute("perm", perm)])

        return transpose_node

    def fuse(self, node, input_name_to_nodes: dict, output_name_to_node: dict):
        """
        Fuse Layer Normalization subgraph into one node LayerNormalization:
              +----------------------+
              | NxCxHxW              |
              |                      v                                                     (Cx1x1)  (Cx1x1)
          [Root] --> ReduceMean -->  Sub --> Pow --> ReduceMean --> Add --> Sqrt --> Div --> Mul --> Add -->
                     (axes=1)        |      (Y=2)     (axes=1)     (E-6)             ^
                                     |                                               |
                                     +-----------------------------------------------+

        Fused subgraph:
                       (0,2,3,1)                            (0,3,1,2)
            [Root] --> Transpose --> LayerNormalization --> Transpose -->
        """
        axes = OnnxModel.get_node_attribute(node, "axes")
        if (not isinstance(axes, list)) or axes != [1]:
            return

        subgraph_nodes = []
        children = self.model.get_children(node, input_name_to_nodes)
        if len(children) != 1:
            return

        root_input = node.input[0]

        if children[0].op_type != "Sub" or children[0].input[0] != root_input:
            return
        sub = children[0]

        div_node = self.model.find_first_child_by_type(sub, "Div", input_name_to_nodes, recursive=False)
        if div_node is None:
            return

        parent_nodes = self.model.match_parent_path(
            div_node,
            ["Sqrt", "Add", "ReduceMean", "Pow", "Sub"],
            [1, 0, 0, 0, 0],
            output_name_to_node,
        )
        if parent_nodes is None:
            return

        _sqrt_node, second_add_node, reduce_mean_node, pow_node, sub_node = parent_nodes
        if sub != sub_node:
            return

        i, epsilon = self.model.get_constant_input(second_add_node)
        if epsilon is None or epsilon <= 0 or epsilon > 1.0e-4:
            logger.debug(f"skip SkipLayerNormalization fusion since epsilon value is not expected: {epsilon}")
            return

        axes = OnnxModel.get_node_attribute(reduce_mean_node, "axes")
        assert isinstance(axes, list)
        if axes != [1]:
            return

        if self.model.find_constant_input(pow_node, 2.0) != 1:
            return

        temp_node = input_name_to_nodes[div_node.output[0]][0]
        mul_node = temp_node
        if mul_node.op_type != "Mul":
            return

        last_add_node = input_name_to_nodes[mul_node.output[0]][0]
        if last_add_node.op_type != "Add":
            return

        subgraph_nodes.append(node)
        subgraph_nodes.extend(parent_nodes)
        subgraph_nodes.extend([last_add_node, mul_node, div_node])

        if not self.model.is_safe_to_fuse_nodes(
            subgraph_nodes,
            last_add_node.output,
            input_name_to_nodes,
            output_name_to_node,
        ):
            logger.debug("It is not safe to fuse LayerNormalization node. Skip")
            return

        node_before_weight = div_node if temp_node.op_type != "Cast" else temp_node
        weight_input = mul_node.input[1 - self.model.input_index(node_before_weight.output[0], mul_node)]
        weight = self.get_weight_or_bias(weight_input, "layernorm weight")
        if weight is None:
            return

        bias_input = last_add_node.input[1 - self.model.input_index(mul_node.output[0], last_add_node)]
        bias = self.get_weight_or_bias(bias_input, "layernorm bias")
        if bias is None:
            return

        weight_nhwc = helper.make_tensor(weight_input + "_NHWC", TensorProto.FLOAT, weight.shape, weight)

        bias_nhwc = helper.make_tensor(bias_input + "_NHWC", TensorProto.FLOAT, weight.shape, weight)
        self.model.add_initializer(weight_nhwc, self.this_graph_name)
        self.model.add_initializer(bias_nhwc, self.this_graph_name)

        self.nodes_to_remove.extend(subgraph_nodes)

        transpose_input = self.create_transpose_node(node.input[0], [0, 2, 3, 1])

        layernorm_node_name = self.model.create_node_name("LayerNormalization", name_prefix="LayerNorm")

        transpose_output = self.create_transpose_node(
            layernorm_node_name + "_out_nhwc", [0, 3, 1, 2], last_add_node.output[0]
        )

        normalize_node = helper.make_node(
            "LayerNormalization",
            inputs=[transpose_input.output[0], weight_input + "_NHWC", bias_input + "_NHWC"],
            outputs=[layernorm_node_name + "_out_nhwc"],
            name=layernorm_node_name,
        )
        normalize_node.attribute.extend([helper.make_attribute("epsilon", float(epsilon))])

        self.nodes_to_add.append(transpose_input)
        self.nodes_to_add.append(normalize_node)
        self.nodes_to_add.append(transpose_output)
        self.node_name_to_graph_name[transpose_input.name] = self.this_graph_name
        self.node_name_to_graph_name[normalize_node.name] = self.this_graph_name
        self.node_name_to_graph_name[transpose_output.name] = self.this_graph_name

        counter_name = "LayerNormalization(NHWC)"
        self.increase_counter(counter_name)


class FusionLayerNormalizationTF(Fusion):
    def __init__(self, model: OnnxModel):
        super().__init__(model, "LayerNormalization", "Add", "TF")

    def fuse(self, node, input_name_to_nodes: dict, output_name_to_node: dict):
        """
         Layer Norm from Tensorflow model(using keras2onnx or tf2onnx):
          +------------------------------------+
          |                                    |
          |                                    |
        (Cast_1)                               |
          |                                    |
          |                                    v                                           (B)                             (B)             (A)
         Add --> (Cast_1) --> ReduceMean -->  Sub  --> Mul --> ReduceMean --> (Cast_3) --> Add --> Sqrt --> Reciprocol --> Mul --> Mul --> Sub --> Add
          |                       |                                                                                         |       ^              ^
          |                       |                                                                                         |       |              |
          |                       +--------------------------------------------------(Cast_2)-------------------------------|-------+              |
          |                                                                                                                 v                      |
          +---------------------------------------------------------------------------------------------------------------> Mul--------------------+
        """
        return_indice = []
        _, parent_nodes, return_indice = self.model.match_parent_paths(
            node,
            [
                (
                    [
                        "Sub",
                        "Mul",
                        "Mul",
                        "Reciprocal",
                        "Sqrt",
                        "Add",
                        "ReduceMean",
                        "Mul",
                        "Sub",
                        "ReduceMean",
                    ],
                    [1, 1, None, 0, 0, 0, None, 0, 0, None],
                ),
                (
                    [
                        "Sub",
                        "Mul",
                        "Mul",
                        "Reciprocal",
                        "Sqrt",
                        "Add",
                        "Cast",
                        "ReduceMean",
                        "Mul",
                        "Sub",
                        "ReduceMean",
                    ],
                    [1, 1, None, 0, 0, 0, 0, None, 0, 0, None],
                ),
            ],
            output_name_to_node,
        )

        if parent_nodes is None:
            return

        assert len(return_indice) == 3
        if not (return_indice[0] in [0, 1] and return_indice[1] in [0, 1] and return_indice[2] in [0, 1]):
            logger.debug("return indice is exepected in [0, 1], but got {return_indice}")
            return

        (
            sub_node_0,
            mul_node_0,
            mul_node_1,
            reciprocol_node,
            sqrt_node,
            add_node_0,
        ) = parent_nodes[:6]
        reduce_mean_node_0, mul_node_2, sub_node_1, reduce_mean_node_1 = parent_nodes[-4:]

        cast_node_3 = None
        if len(parent_nodes) == 11:
            cast_node_3 = parent_nodes[6]
            assert cast_node_3.op_type == "Cast"

        mul_node_3 = self.model.match_parent(node, "Mul", 0, output_name_to_node)
        if mul_node_3 is None:
            logger.debug("mul_node_3 not found")
            return

        node_before_reduce = self.model.get_parent(reduce_mean_node_1, 0, output_name_to_node)
        root_node = (
            node_before_reduce
            if cast_node_3 is None
            else self.model.get_parent(node_before_reduce, 0, output_name_to_node)
        )
        if root_node is None:
            logger.debug("root node is none")
            return

        i, epsilon = self.model.get_constant_input(add_node_0)
        if epsilon is None or epsilon <= 0 or (epsilon > 1.0e-5 and cast_node_3 is None):
            logger.debug("epsilon is not matched")
            return

        if cast_node_3 is None and (
            reduce_mean_node_1.input[0] not in mul_node_3.input or reduce_mean_node_1.input[0] not in sub_node_1.input
        ):
            logger.debug("reduce_mean_node_1 and mul_node_3 shall link from root node")
            return

        if cast_node_3 is not None and (
            node_before_reduce.input[0] not in mul_node_3.input or reduce_mean_node_1.input[0] not in sub_node_1.input
        ):
            logger.debug("reduce_mean_node_1 and mul_node_3 shall link from root node")
            return

        if mul_node_2.input[0] != mul_node_2.input[1]:
            logger.debug("mul_node_2 shall have two same inputs")
            return

        subgraph_nodes = [
            node,
            sub_node_0,
            mul_node_0,
            mul_node_1,
            reciprocol_node,
            sqrt_node,
            add_node_0,
            reduce_mean_node_0,
            mul_node_2,
            sub_node_1,
            reduce_mean_node_1,
            mul_node_3,
        ]

        if cast_node_3 is not None:
            cast_node_2 = self.model.match_parent(mul_node_0, "Cast", 0, output_name_to_node)
            if cast_node_2 is None:
                logger.debug("cast_node_2 not found")
                return
            subgraph_nodes.extend([node_before_reduce, cast_node_2, cast_node_3])

        if not self.model.is_safe_to_fuse_nodes(
            subgraph_nodes,
            node.output,
            self.model.input_name_to_nodes(),
            self.model.output_name_to_node(),
        ):
            logger.debug("not safe to fuse layer normalization")
            return

        self.nodes_to_remove.extend(subgraph_nodes)

        weight_input = mul_node_1.input[1]
        bias_input = sub_node_0.input[0]

        # TODO: add epsilon attribute
        fused_node = helper.make_node(
            "LayerNormalization",
            inputs=[mul_node_3.input[0], weight_input, bias_input],
            outputs=[node.output[0]],
            name=self.model.create_node_name("LayerNormalization", name_prefix="LayerNorm"),
        )
        fused_node.attribute.extend([helper.make_attribute("epsilon", float(epsilon))])
        self.nodes_to_add.append(fused_node)
        self.node_name_to_graph_name[fused_node.name] = self.this_graph_name

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\remote\remote_connection.py ===
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

import logging
import platform
import string
import warnings
from base64 import b64encode
from typing import Optional
from urllib import parse
from urllib.parse import urlparse

import urllib3

from selenium import __version__

from . import utils
from .client_config import ClientConfig
from .command import Command
from .errorhandler import ErrorCode

LOGGER = logging.getLogger(__name__)

remote_commands = {
    Command.NEW_SESSION: ("POST", "/session"),
    Command.QUIT: ("DELETE", "/session/$sessionId"),
    Command.W3C_GET_CURRENT_WINDOW_HANDLE: ("GET", "/session/$sessionId/window"),
    Command.W3C_GET_WINDOW_HANDLES: ("GET", "/session/$sessionId/window/handles"),
    Command.GET: ("POST", "/session/$sessionId/url"),
    Command.GO_FORWARD: ("POST", "/session/$sessionId/forward"),
    Command.GO_BACK: ("POST", "/session/$sessionId/back"),
    Command.REFRESH: ("POST", "/session/$sessionId/refresh"),
    Command.W3C_EXECUTE_SCRIPT: ("POST", "/session/$sessionId/execute/sync"),
    Command.W3C_EXECUTE_SCRIPT_ASYNC: ("POST", "/session/$sessionId/execute/async"),
    Command.GET_CURRENT_URL: ("GET", "/session/$sessionId/url"),
    Command.GET_TITLE: ("GET", "/session/$sessionId/title"),
    Command.GET_PAGE_SOURCE: ("GET", "/session/$sessionId/source"),
    Command.SCREENSHOT: ("GET", "/session/$sessionId/screenshot"),
    Command.ELEMENT_SCREENSHOT: ("GET", "/session/$sessionId/element/$id/screenshot"),
    Command.FIND_ELEMENT: ("POST", "/session/$sessionId/element"),
    Command.FIND_ELEMENTS: ("POST", "/session/$sessionId/elements"),
    Command.W3C_GET_ACTIVE_ELEMENT: ("GET", "/session/$sessionId/element/active"),
    Command.FIND_CHILD_ELEMENT: ("POST", "/session/$sessionId/element/$id/element"),
    Command.FIND_CHILD_ELEMENTS: ("POST", "/session/$sessionId/element/$id/elements"),
    Command.CLICK_ELEMENT: ("POST", "/session/$sessionId/element/$id/click"),
    Command.CLEAR_ELEMENT: ("POST", "/session/$sessionId/element/$id/clear"),
    Command.GET_ELEMENT_TEXT: ("GET", "/session/$sessionId/element/$id/text"),
    Command.SEND_KEYS_TO_ELEMENT: ("POST", "/session/$sessionId/element/$id/value"),
    Command.GET_ELEMENT_TAG_NAME: ("GET", "/session/$sessionId/element/$id/name"),
    Command.IS_ELEMENT_SELECTED: ("GET", "/session/$sessionId/element/$id/selected"),
    Command.IS_ELEMENT_ENABLED: ("GET", "/session/$sessionId/element/$id/enabled"),
    Command.GET_ELEMENT_RECT: ("GET", "/session/$sessionId/element/$id/rect"),
    Command.GET_ELEMENT_ATTRIBUTE: ("GET", "/session/$sessionId/element/$id/attribute/$name"),
    Command.GET_ELEMENT_PROPERTY: ("GET", "/session/$sessionId/element/$id/property/$name"),
    Command.GET_ELEMENT_ARIA_ROLE: ("GET", "/session/$sessionId/element/$id/computedrole"),
    Command.GET_ELEMENT_ARIA_LABEL: ("GET", "/session/$sessionId/element/$id/computedlabel"),
    Command.GET_SHADOW_ROOT: ("GET", "/session/$sessionId/element/$id/shadow"),
    Command.FIND_ELEMENT_FROM_SHADOW_ROOT: ("POST", "/session/$sessionId/shadow/$shadowId/element"),
    Command.FIND_ELEMENTS_FROM_SHADOW_ROOT: ("POST", "/session/$sessionId/shadow/$shadowId/elements"),
    Command.GET_ALL_COOKIES: ("GET", "/session/$sessionId/cookie"),
    Command.ADD_COOKIE: ("POST", "/session/$sessionId/cookie"),
    Command.GET_COOKIE: ("GET", "/session/$sessionId/cookie/$name"),
    Command.DELETE_ALL_COOKIES: ("DELETE", "/session/$sessionId/cookie"),
    Command.DELETE_COOKIE: ("DELETE", "/session/$sessionId/cookie/$name"),
    Command.SWITCH_TO_FRAME: ("POST", "/session/$sessionId/frame"),
    Command.SWITCH_TO_PARENT_FRAME: ("POST", "/session/$sessionId/frame/parent"),
    Command.SWITCH_TO_WINDOW: ("POST", "/session/$sessionId/window"),
    Command.NEW_WINDOW: ("POST", "/session/$sessionId/window/new"),
    Command.CLOSE: ("DELETE", "/session/$sessionId/window"),
    Command.GET_ELEMENT_VALUE_OF_CSS_PROPERTY: ("GET", "/session/$sessionId/element/$id/css/$propertyName"),
    Command.EXECUTE_ASYNC_SCRIPT: ("POST", "/session/$sessionId/execute_async"),
    Command.SET_TIMEOUTS: ("POST", "/session/$sessionId/timeouts"),
    Command.GET_TIMEOUTS: ("GET", "/session/$sessionId/timeouts"),
    Command.W3C_DISMISS_ALERT: ("POST", "/session/$sessionId/alert/dismiss"),
    Command.W3C_ACCEPT_ALERT: ("POST", "/session/$sessionId/alert/accept"),
    Command.W3C_SET_ALERT_VALUE: ("POST", "/session/$sessionId/alert/text"),
    Command.W3C_GET_ALERT_TEXT: ("GET", "/session/$sessionId/alert/text"),
    Command.W3C_ACTIONS: ("POST", "/session/$sessionId/actions"),
    Command.W3C_CLEAR_ACTIONS: ("DELETE", "/session/$sessionId/actions"),
    Command.SET_WINDOW_RECT: ("POST", "/session/$sessionId/window/rect"),
    Command.GET_WINDOW_RECT: ("GET", "/session/$sessionId/window/rect"),
    Command.W3C_MAXIMIZE_WINDOW: ("POST", "/session/$sessionId/window/maximize"),
    Command.SET_SCREEN_ORIENTATION: ("POST", "/session/$sessionId/orientation"),
    Command.GET_SCREEN_ORIENTATION: ("GET", "/session/$sessionId/orientation"),
    Command.GET_NETWORK_CONNECTION: ("GET", "/session/$sessionId/network_connection"),
    Command.SET_NETWORK_CONNECTION: ("POST", "/session/$sessionId/network_connection"),
    Command.GET_LOG: ("POST", "/session/$sessionId/se/log"),
    Command.GET_AVAILABLE_LOG_TYPES: ("GET", "/session/$sessionId/se/log/types"),
    Command.CURRENT_CONTEXT_HANDLE: ("GET", "/session/$sessionId/context"),
    Command.CONTEXT_HANDLES: ("GET", "/session/$sessionId/contexts"),
    Command.SWITCH_TO_CONTEXT: ("POST", "/session/$sessionId/context"),
    Command.FULLSCREEN_WINDOW: ("POST", "/session/$sessionId/window/fullscreen"),
    Command.MINIMIZE_WINDOW: ("POST", "/session/$sessionId/window/minimize"),
    Command.PRINT_PAGE: ("POST", "/session/$sessionId/print"),
    Command.ADD_VIRTUAL_AUTHENTICATOR: ("POST", "/session/$sessionId/webauthn/authenticator"),
    Command.REMOVE_VIRTUAL_AUTHENTICATOR: (
        "DELETE",
        "/session/$sessionId/webauthn/authenticator/$authenticatorId",
    ),
    Command.ADD_CREDENTIAL: ("POST", "/session/$sessionId/webauthn/authenticator/$authenticatorId/credential"),
    Command.GET_CREDENTIALS: ("GET", "/session/$sessionId/webauthn/authenticator/$authenticatorId/credentials"),
    Command.REMOVE_CREDENTIAL: (
        "DELETE",
        "/session/$sessionId/webauthn/authenticator/$authenticatorId/credentials/$credentialId",
    ),
    Command.REMOVE_ALL_CREDENTIALS: (
        "DELETE",
        "/session/$sessionId/webauthn/authenticator/$authenticatorId/credentials",
    ),
    Command.SET_USER_VERIFIED: ("POST", "/session/$sessionId/webauthn/authenticator/$authenticatorId/uv"),
    Command.UPLOAD_FILE: ("POST", "/session/$sessionId/se/file"),
    Command.GET_DOWNLOADABLE_FILES: ("GET", "/session/$sessionId/se/files"),
    Command.DOWNLOAD_FILE: ("POST", "/session/$sessionId/se/files"),
    Command.DELETE_DOWNLOADABLE_FILES: ("DELETE", "/session/$sessionId/se/files"),
    # Federated Credential Management (FedCM)
    Command.GET_FEDCM_TITLE: ("GET", "/session/$sessionId/fedcm/gettitle"),
    Command.GET_FEDCM_DIALOG_TYPE: ("GET", "/session/$sessionId/fedcm/getdialogtype"),
    Command.GET_FEDCM_ACCOUNT_LIST: ("GET", "/session/$sessionId/fedcm/accountlist"),
    Command.CLICK_FEDCM_DIALOG_BUTTON: ("POST", "/session/$sessionId/fedcm/clickdialogbutton"),
    Command.CANCEL_FEDCM_DIALOG: ("POST", "/session/$sessionId/fedcm/canceldialog"),
    Command.SELECT_FEDCM_ACCOUNT: ("POST", "/session/$sessionId/fedcm/selectaccount"),
    Command.SET_FEDCM_DELAY: ("POST", "/session/$sessionId/fedcm/setdelayenabled"),
    Command.RESET_FEDCM_COOLDOWN: ("POST", "/session/$sessionId/fedcm/resetcooldown"),
}


class RemoteConnection:
    """A connection with the Remote WebDriver server.

    Communicates with the server using the WebDriver wire protocol:
    https://github.com/SeleniumHQ/selenium/wiki/JsonWireProtocol
    """

    browser_name = None
    # Keep backward compatibility for AppiumConnection - https://github.com/SeleniumHQ/selenium/issues/14694
    import os
    import socket

    import certifi

    _timeout = socket.getdefaulttimeout()
    _ca_certs = os.getenv("REQUESTS_CA_BUNDLE") if "REQUESTS_CA_BUNDLE" in os.environ else certifi.where()
    _client_config: Optional[ClientConfig] = None

    system = platform.system().lower()
    if system == "darwin":
        system = "mac"

    # Class variables for headers
    extra_headers = None
    user_agent = f"selenium/{__version__} (python {system})"

    @property
    def client_config(self):
        return self._client_config

    @classmethod
    def get_timeout(cls):
        """:Returns:

        Timeout value in seconds for all http requests made to the
        Remote Connection
        """
        warnings.warn(
            "get_timeout() in RemoteConnection is deprecated, get timeout from client_config instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls._client_config.timeout

    @classmethod
    def set_timeout(cls, timeout):
        """Override the default timeout.

        :Args:
            - timeout - timeout value for http requests in seconds
        """
        warnings.warn(
            "set_timeout() in RemoteConnection is deprecated, set timeout in client_config instead",
            DeprecationWarning,
            stacklevel=2,
        )
        cls._client_config.timeout = timeout

    @classmethod
    def reset_timeout(cls):
        """Reset the http request timeout to socket._GLOBAL_DEFAULT_TIMEOUT."""
        warnings.warn(
            "reset_timeout() in RemoteConnection is deprecated, use reset_timeout() in client_config instead",
            DeprecationWarning,
            stacklevel=2,
        )
        cls._client_config.reset_timeout()

    @classmethod
    def get_certificate_bundle_path(cls):
        """:Returns:

        Paths of the .pem encoded certificate to verify connection to
        command executor. Defaults to certifi.where() or
        REQUESTS_CA_BUNDLE env variable if set.
        """
        warnings.warn(
            "get_certificate_bundle_path() in RemoteConnection is deprecated, get ca_certs from client_config instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls._client_config.ca_certs

    @classmethod
    def set_certificate_bundle_path(cls, path):
        """Set the path to the certificate bundle to verify connection to
        command executor. Can also be set to None to disable certificate
        validation.

        :Args:
            - path - path of a .pem encoded certificate chain.
        """
        warnings.warn(
            "set_certificate_bundle_path() in RemoteConnection is deprecated, set ca_certs in client_config instead",
            DeprecationWarning,
            stacklevel=2,
        )
        cls._client_config.ca_certs = path

    @classmethod
    def get_remote_connection_headers(cls, parsed_url, keep_alive=False):
        """Get headers for remote request.

        :Args:
         - parsed_url - The parsed url
         - keep_alive (Boolean) - Is this a keep-alive connection (default: False)
        """

        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json;charset=UTF-8",
            "User-Agent": cls.user_agent,
        }

        if parsed_url.username:
            warnings.warn(
                "Embedding username and password in URL could be insecure, use ClientConfig instead", stacklevel=2
            )
            base64string = b64encode(f"{parsed_url.username}:{parsed_url.password}".encode())
            headers.update({"Authorization": f"Basic {base64string.decode()}"})

        if keep_alive:
            headers.update({"Connection": "keep-alive"})

        if cls.extra_headers:
            headers.update(cls.extra_headers)

        return headers

    def _identify_http_proxy_auth(self):
        parsed_url = urlparse(self._proxy_url)
        if parsed_url.username and parsed_url.password:
            return True

    def _separate_http_proxy_auth(self):
        parsed_url = urlparse(self._proxy_url)
        proxy_without_auth = f"{parsed_url.scheme}://{parsed_url.hostname}:{parsed_url.port}"
        auth = f"{parsed_url.username}:{parsed_url.password}"
        return proxy_without_auth, auth

    def _get_connection_manager(self):
        pool_manager_init_args = {"timeout": self._client_config.timeout}
        pool_manager_init_args.update(
            self._client_config.init_args_for_pool_manager.get("init_args_for_pool_manager", {})
        )

        if self._client_config.ignore_certificates:
            pool_manager_init_args["cert_reqs"] = "CERT_NONE"
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        elif self._client_config.ca_certs:
            pool_manager_init_args["cert_reqs"] = "CERT_REQUIRED"
            pool_manager_init_args["ca_certs"] = self._client_config.ca_certs

        if self._proxy_url:
            if self._proxy_url.lower().startswith("sock"):
                from urllib3.contrib.socks import SOCKSProxyManager

                return SOCKSProxyManager(self._proxy_url, **pool_manager_init_args)
            if self._identify_http_proxy_auth():
                self._proxy_url, self._basic_proxy_auth = self._separate_http_proxy_auth()
                pool_manager_init_args["proxy_headers"] = urllib3.make_headers(proxy_basic_auth=self._basic_proxy_auth)
            return urllib3.ProxyManager(self._proxy_url, **pool_manager_init_args)

        return urllib3.PoolManager(**pool_manager_init_args)

    def __init__(
        self,
        remote_server_addr: Optional[str] = None,
        keep_alive: Optional[bool] = True,
        ignore_proxy: Optional[bool] = False,
        ignore_certificates: Optional[bool] = False,
        init_args_for_pool_manager: Optional[dict] = None,
        client_config: Optional[ClientConfig] = None,
    ):
        self._client_config = client_config or ClientConfig(
            remote_server_addr=remote_server_addr,
            keep_alive=keep_alive,
            ignore_certificates=ignore_certificates,
            init_args_for_pool_manager=init_args_for_pool_manager,
        )

        # Keep backward compatibility for AppiumConnection - https://github.com/SeleniumHQ/selenium/issues/14694
        RemoteConnection._timeout = self._client_config.timeout
        RemoteConnection._ca_certs = self._client_config.ca_certs
        RemoteConnection._client_config = self._client_config
        RemoteConnection.extra_headers = self._client_config.extra_headers or RemoteConnection.extra_headers
        RemoteConnection.user_agent = self._client_config.user_agent or RemoteConnection.user_agent

        if remote_server_addr:
            warnings.warn(
                "setting remote_server_addr in RemoteConnection() is deprecated, set in client_config instead",
                DeprecationWarning,
                stacklevel=2,
            )

        if not keep_alive:
            warnings.warn(
                "setting keep_alive in RemoteConnection() is deprecated, set in client_config instead",
                DeprecationWarning,
                stacklevel=2,
            )

        if ignore_certificates:
            warnings.warn(
                "setting ignore_certificates in RemoteConnection() is deprecated, set in client_config instead",
                DeprecationWarning,
                stacklevel=2,
            )

        if init_args_for_pool_manager:
            warnings.warn(
                "setting init_args_for_pool_manager in RemoteConnection() is deprecated, set in client_config instead",
                DeprecationWarning,
                stacklevel=2,
            )

        if ignore_proxy:
            warnings.warn(
                "setting ignore_proxy in RemoteConnection() is deprecated, set in client_config instead",
                DeprecationWarning,
                stacklevel=2,
            )
            self._proxy_url = None
        else:
            self._proxy_url = self._client_config.get_proxy_url()

        if self._client_config.keep_alive:
            self._conn = self._get_connection_manager()
        self._commands = remote_commands

    extra_commands = {}

    def add_command(self, name, method, url):
        """Register a new command."""
        self._commands[name] = (method, url)

    def get_command(self, name: str):
        """Retrieve a command if it exists."""
        return self._commands.get(name)

    def execute(self, command, params):
        """Send a command to the remote server.

        Any path substitutions required for the URL mapped to the command should be
        included in the command parameters.

        :Args:
         - command - A string specifying the command to execute.
         - params - A dictionary of named parameters to send with the command as
           its JSON payload.
        """
        command_info = self._commands.get(command) or self.extra_commands.get(command)
        assert command_info is not None, f"Unrecognised command {command}"
        path_string = command_info[1]
        path = string.Template(path_string).substitute(params)
        substitute_params = {word[1:] for word in path_string.split("/") if word.startswith("$")}  # remove dollar sign
        if isinstance(params, dict) and substitute_params:
            for word in substitute_params:
                del params[word]
        data = utils.dump_json(params)
        url = f"{self._client_config.remote_server_addr}{path}"
        trimmed = self._trim_large_entries(params)
        LOGGER.debug("%s %s %s", command_info[0], url, str(trimmed))
        return self._request(command_info[0], url, body=data)

    def _request(self, method, url, body=None):
        """Send an HTTP request to the remote server.

        :Args:
         - method - A string for the HTTP method to send the request with.
         - url - A string for the URL to send the request to.
         - body - A string for request body. Ignored unless method is POST or PUT.

        :Returns:
          A dictionary with the server's parsed JSON response.
        """
        parsed_url = parse.urlparse(url)
        headers = self.get_remote_connection_headers(parsed_url, self._client_config.keep_alive)
        auth_header = self._client_config.get_auth_header()

        if auth_header:
            headers.update(auth_header)

        if body and method not in ("POST", "PUT"):
            body = None

        if self._client_config.keep_alive:
            response = self._conn.request(method, url, body=body, headers=headers, timeout=self._client_config.timeout)
            statuscode = response.status
        else:
            conn = self._get_connection_manager()
            with conn as http:
                response = http.request(method, url, body=body, headers=headers, timeout=self._client_config.timeout)
            statuscode = response.status
        data = response.data.decode("UTF-8")
        LOGGER.debug("Remote response: status=%s | data=%s | headers=%s", response.status, data, response.headers)
        try:
            if 300 <= statuscode < 304:
                return self._request("GET", response.headers.get("location", None))
            if 399 < statuscode <= 500:
                if statuscode == 401:
                    return {"status": statuscode, "value": "Authorization Required"}
                return {"status": statuscode, "value": str(statuscode) if not data else data.strip()}
            content_type = []
            if response.headers.get("Content-Type", None):
                content_type = response.headers.get("Content-Type", None).split(";")
            if not any([x.startswith("image/png") for x in content_type]):
                try:
                    data = utils.load_json(data.strip())
                except ValueError:
                    if 199 < statuscode < 300:
                        status = ErrorCode.SUCCESS
                    else:
                        status = ErrorCode.UNKNOWN_ERROR
                    return {"status": status, "value": data.strip()}

                # Some drivers incorrectly return a response
                # with no 'value' field when they should return null.
                if "value" not in data:
                    data["value"] = None
                return data
            data = {"status": 0, "value": data}
            return data
        finally:
            LOGGER.debug("Finished Request")
            response.close()

    def close(self):
        """Clean up resources when finished with the remote_connection."""
        if hasattr(self, "_conn"):
            self._conn.clear()

    def _trim_large_entries(self, input_dict, max_length=100):
        """Truncate string values in a dictionary if they exceed max_length.

        :param dict: Dictionary with potentially large values
        :param max_length: Maximum allowed length of string values
        :return: Dictionary with truncated string values
        """
        output_dictionary = {}
        for key, value in input_dict.items():
            if isinstance(value, dict):
                output_dictionary[key] = self._trim_large_entries(value, max_length)
            elif isinstance(value, str) and len(value) > max_length:
                output_dictionary[key] = value[:max_length] + "..."
            else:
                output_dictionary[key] = value

        return output_dictionary

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\onnx_model_bert.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from logging import getLogger

from convert_to_packing_mode import PackingMode
from fusion_attention import AttentionMask, FusionAttention
from fusion_bart_attention import FusionBartAttention
from fusion_biasgelu import FusionBiasGelu
from fusion_constant_fold import FusionConstantFold
from fusion_embedlayer import FusionEmbedLayerNormalization
from fusion_fastgelu import FusionFastGelu
from fusion_gelu import FusionGelu
from fusion_gelu_approximation import FusionGeluApproximation
from fusion_gemmfastgelu import FusionGemmFastGelu
from fusion_layernorm import FusionLayerNormalization, FusionLayerNormalizationTF
from fusion_options import AttentionMaskFormat, FusionOptions
from fusion_qordered_attention import FusionQOrderedAttention
from fusion_qordered_gelu import FusionQOrderedGelu
from fusion_qordered_layernorm import FusionQOrderedLayerNormalization
from fusion_qordered_matmul import FusionQOrderedMatMul
from fusion_quickgelu import FusionQuickGelu
from fusion_reshape import FusionReshape
from fusion_rotary_attention import FusionRotaryEmbeddings
from fusion_shape import FusionShape
from fusion_simplified_layernorm import FusionSimplifiedLayerNormalization, FusionSkipSimplifiedLayerNormalization
from fusion_skiplayernorm import FusionBiasSkipLayerNormalization, FusionSkipLayerNormalization
from fusion_utils import FusionUtils
from onnx import ModelProto, TensorProto, helper
from onnx_model import OnnxModel

logger = getLogger(__name__)


class BertOnnxModel(OnnxModel):
    def __init__(self, model: ModelProto, num_heads: int = 0, hidden_size: int = 0):
        """Initialize BERT ONNX Model.

        Args:
            model (ModelProto): the ONNX model
            num_heads (int, optional): number of attention heads. Defaults to 0 (detect the parameter automatically).
            hidden_size (int, optional): hidden dimension. Defaults to 0 (detect the parameter automatically).
        """
        assert (num_heads == 0 and hidden_size == 0) or (num_heads > 0 and hidden_size % num_heads == 0)

        super().__init__(model)
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        self.attention_mask = AttentionMask(self)
        self.attention_fusion = FusionAttention(self, self.hidden_size, self.num_heads, self.attention_mask)
        self.qordered_attention_fusion = FusionQOrderedAttention(
            self, self.hidden_size, self.num_heads, self.attention_mask
        )
        self.utils = FusionUtils(self)

    def fuse_constant_fold(self):
        fusion = FusionConstantFold(self)
        fusion.apply()

    def fuse_attention(self):
        self.attention_fusion.apply()
        # Only relevant in models with Q-DQ nodes
        self.qordered_attention_fusion.apply()

    def fuse_gelu(self):
        fusion = FusionGelu(self)
        fusion.apply()
        fusion = FusionFastGelu(self)
        fusion.apply()
        fusion = FusionQuickGelu(self)
        fusion.apply()
        # Only relevant in models with Q-DQ nodes
        fusion = FusionQOrderedGelu(self)
        fusion.apply()

    def fuse_bias_gelu(self, is_fastgelu):
        fusion = FusionBiasGelu(self, is_fastgelu)
        fusion.apply()

    def gelu_approximation(self):
        fusion = FusionGeluApproximation(self)
        fusion.apply()

    def fuse_gemm_fast_gelu(self):
        fusion = FusionGemmFastGelu(self)
        fusion.apply()

    def fuse_add_bias_skip_layer_norm(self):
        fusion = FusionBiasSkipLayerNormalization(self)
        fusion.apply()

    def fuse_reshape(self):
        fusion = FusionReshape(self)
        fusion.apply()

    def fuse_shape(self):
        fusion = FusionShape(self)
        fusion.apply()

    def fuse_embed_layer(self, use_mask_index):
        fusion = FusionEmbedLayerNormalization(self, use_mask_index)
        fusion.apply()

    def fuse_layer_norm(self):
        fusion = FusionLayerNormalization(self)
        fusion.apply()

        fusion = FusionLayerNormalizationTF(self)
        fusion.apply()

        # Only relevant in models with Q-DQ nodes
        fusion = FusionQOrderedLayerNormalization(self)
        fusion.apply()

    def fuse_simplified_layer_norm(self):
        fusion = FusionSimplifiedLayerNormalization(self)
        fusion.apply()

    def fuse_skip_layer_norm(self, shape_infer=True):
        fusion = FusionSkipLayerNormalization(self, shape_infer=shape_infer)
        fusion.apply()

    def fuse_skip_simplified_layer_norm(self):
        fusion = FusionSkipSimplifiedLayerNormalization(self)
        fusion.apply()

    def fuse_rotary_embeddings(self):
        fusion = FusionRotaryEmbeddings(self)
        fusion.apply()
        # Remove non-MS domain functions
        rot_emb_nodes = list(
            filter(
                lambda node: node.op_type == "RotaryEmbedding" and node.domain != "com.microsoft",
                self.model.graph.node,
            )
        )
        non_ms_domains_to_keep = {node.domain for node in rot_emb_nodes}
        i = 0
        while i < len(self.model.functions):
            fn = self.model.functions[i]
            if "RotaryEmbedding" in fn.name and fn.domain not in non_ms_domains_to_keep:
                self.model.functions.remove(fn)
            else:
                i += 1

    # Only relevant in models with Q-DQ nodes
    def fuse_qordered_mamtul(self):
        fusion = FusionQOrderedMatMul(self)
        fusion.apply()

    def get_graph_inputs_from_node_type(self, op_type: str, input_indices: list[int], casted: bool):
        """
        Get graph inputs that feed into node type (like EmbedLayerNormalization or Attention).
        Returns a list of the graph input names based on the filter whether it is casted or not.
        """
        graph_inputs = []

        output_name_to_node = self.output_name_to_node()
        nodes = self.get_nodes_by_op_type(op_type)
        for node in nodes:
            bert_inputs = [node.input[i] for i in input_indices if i < len(node.input)]
            for bert_input in bert_inputs:
                if self.find_graph_input(bert_input):
                    if not casted:
                        graph_inputs.append(bert_input)
                elif bert_input in output_name_to_node:
                    parent = output_name_to_node[bert_input]
                    if parent.op_type == "Cast" and self.find_graph_input(parent.input[0]) is not None:
                        if casted:
                            graph_inputs.append(parent.input[0])
        return graph_inputs

    def get_graph_inputs_from_fused_nodes(self, casted: bool):
        inputs = self.get_graph_inputs_from_node_type("EmbedLayerNormalization", [0, 1, 7], casted)
        inputs += self.get_graph_inputs_from_node_type("Attention", [3], casted)
        return inputs

    def change_graph_inputs_to_int32(self):
        """Change data type of all graph inputs to int32 type, and add Cast node if needed."""
        graph = self.graph()
        add_cast_count = 0
        remove_cast_count = 0
        for graph_input in graph.input:
            new_node, removed_nodes = self.change_graph_input_type(graph_input, TensorProto.INT32)
            if new_node:
                add_cast_count += 1
            remove_cast_count += len(removed_nodes)
        logger.info(
            f"Graph inputs are changed to int32. Added {add_cast_count} Cast nodes, and removed {remove_cast_count} Cast nodes."
        )

    def use_dynamic_axes(self, dynamic_batch_dim="batch_size", dynamic_seq_len="max_seq_len"):
        """
        Update input and output shape to use dynamic axes.
        """
        bert_graph_inputs = self.get_graph_inputs_from_fused_nodes(
            casted=True
        ) + self.get_graph_inputs_from_fused_nodes(casted=False)

        for input in self.model.graph.input:
            if input.name in bert_graph_inputs:
                dim_proto = input.type.tensor_type.shape.dim[0]
                dim_proto.dim_param = dynamic_batch_dim
                if dynamic_seq_len is not None:
                    dim_proto = input.type.tensor_type.shape.dim[1]
                    dim_proto.dim_param = dynamic_seq_len

        for output in self.model.graph.output:
            dim_proto = output.type.tensor_type.shape.dim[0]
            dim_proto.dim_param = dynamic_batch_dim

    def preprocess(self):
        self.adjust_reshape_and_expand()
        return

    def adjust_reshape_and_expand(self):
        nodes_to_remove = []
        for node in self.nodes():
            if node.op_type == "Reshape":
                # Clean up unnecessary reshape nodes.
                # Find reshape nodes with no actually data in "shape" attribute and remove.
                reshape_shape = self.get_constant_value(node.input[1])
                if reshape_shape is not None and reshape_shape.size == 0:
                    nodes_to_remove.extend([node])
                    self.replace_input_of_all_nodes(node.output[0], node.input[0])
                    continue

                # Find path "Slice" -> "Reshape" -> "Expand" -> "Expand" -> current "Reshape", simplify the graph by
                # changing current reshape's input to output of slice.
                reshape_path = self.match_parent_path(
                    node,
                    ["Expand", "Expand", "Reshape", "Slice"],
                    [0, 0, 0, 0],
                    self.output_name_to_node(),
                )
                if reshape_path is not None:
                    expand_node = reshape_path[-3]
                    expand_shape_value = self.get_constant_value(expand_node.input[1])

                    reshape_before_expand = reshape_path[-2]
                    shape_value = self.get_constant_value(reshape_before_expand.input[1])

                    slice_node = reshape_path[-1]
                    if (
                        expand_shape_value is not None
                        and shape_value is not None
                        and len(expand_shape_value) == 2
                        and len(shape_value) == 1
                        and expand_shape_value[1] == shape_value[0]
                    ):
                        node.input[0] = slice_node.output[0]

        if nodes_to_remove:
            self.remove_nodes(nodes_to_remove)
            logger.info(f"Removed Reshape and Expand count: {len(nodes_to_remove)}")

    def clean_graph(self):
        output_name_to_node = self.output_name_to_node()
        nodes_to_remove = []
        for node in self.nodes():
            # Before:
            #  input_ids --> Shape --> Gather(indices=0) --> Unsqueeze ------+
            #          |                                                     |
            #          |                                                     v
            #          +----> Shape --> Gather(indices=1) --> Unsqueeze--->  Concat --> ConstantOfShape -->Cast --> EmbedLayerNormaliation/ReduceSum
            # After:
            #  input_ids --> Shape                                                  --> ConstantOfShape -->Cast --> EmbedLayerNormaliation/ReduceSum
            # TODO: merge ConstantOfShape -->Cast to ConstantOfShape (need update the data type of value)
            op_input_id = {"EmbedLayerNormalization": 1, "ReduceSum": 0, "Attention": 3}
            if node.op_type in op_input_id:
                i = op_input_id[node.op_type]
                parent_nodes = self.match_parent_path(
                    node,
                    [
                        "Cast",
                        "ConstantOfShape",
                        "Concat",
                        "Unsqueeze",
                        "Gather",
                        "Shape",
                    ],
                    [i, 0, 0, 0, 0, 0],
                    output_name_to_node,
                )
                if parent_nodes is not None:
                    (
                        cast,
                        constantOfShape,  # noqa: N806
                        concat,
                        unsqueeze,
                        gather,
                        shape,
                    ) = parent_nodes
                    if shape.input[0] == self.graph().input[0].name:
                        constantOfShape.input[0] = shape.output[0]
                        output_name_to_node = self.output_name_to_node()

            if node.op_type == "Attention":
                # Before:
                #   input_ids --> Shape -->ConstantOfShape -->Cast --> ReduceSum --> Attention
                # After:
                #   remove this path, and remove the optional mask_index input of Attention node.
                parent_nodes = self.match_parent_path(
                    node,
                    ["ReduceSum", "Cast", "ConstantOfShape", "Shape"],
                    [3, 0, 0, 0],
                    output_name_to_node,
                )
                if parent_nodes is not None:
                    if parent_nodes[-1].input[0] == self.graph().input[0].name:
                        attention_node = helper.make_node(
                            "Attention",
                            inputs=node.input[0 : len(node.input) - 1],
                            outputs=node.output,
                            name=node.name + "_remove_mask",
                        )
                        attention_node.domain = "com.microsoft"
                        attention_node.attribute.extend([helper.make_attribute("num_heads", self.num_heads)])
                        self.add_node(attention_node, self.get_graph_by_node(attention_node).name)
                        nodes_to_remove.append(node)
        self.remove_nodes(nodes_to_remove)

    def postprocess(self):
        self.clean_graph()
        self.prune_graph()

    def optimize(self, options: FusionOptions | None = None, add_dynamic_axes: bool = False):
        if (options is not None) and not options.enable_shape_inference:
            self.disable_shape_inference()

        self.utils.remove_identity_nodes()

        # Remove cast nodes that having same data type of input and output based on symbolic shape inference.
        self.utils.remove_useless_cast_nodes()

        # Apply any missed constant-folding model optimizations (e.g. for Dynamo-exported models)
        self.fuse_constant_fold()

        if (options is None) or options.enable_layer_norm:
            self.fuse_layer_norm()
            self.fuse_simplified_layer_norm()

        if (options is None) or options.enable_gelu:
            self.fuse_gelu()

        self.preprocess()

        self.fuse_reshape()

        if (options is None) or options.enable_skip_layer_norm:
            self.fuse_skip_layer_norm(options.enable_shape_inference)
            self.fuse_skip_simplified_layer_norm()

        if (options is None) or options.enable_rotary_embeddings:
            self.fuse_rotary_embeddings()

        if options is not None:
            self.attention_mask.set_mask_format(options.attention_mask_format)
            if options.use_multi_head_attention and not isinstance(self.attention_fusion, FusionBartAttention):
                self.attention_fusion = FusionAttention(
                    self,
                    self.hidden_size,
                    self.num_heads,
                    self.attention_mask,
                    options.use_multi_head_attention,
                )

        if (options is None) or options.enable_attention:
            self.fuse_attention()

        # Perform the MatMul fusion after the Attention fusion as we do not
        # want to fuse the MatMuls inside the Attention subgraphs
        if (options is None) or options.enable_qordered_matmul:
            self.fuse_qordered_mamtul()

        self.fuse_shape()

        if (options is None) or options.enable_embed_layer_norm:
            use_mask_index = options.attention_mask_format == AttentionMaskFormat.MaskIndexEnd
            self.fuse_embed_layer(use_mask_index)

        # Remove reshape nodes that having same shape of input and output based on symbolic shape inference.
        self.utils.remove_useless_reshape_nodes()

        self.postprocess()

        # Bias fusion is done after postprocess to avoid extra Reshape between bias and Gelu/FastGelu/SkipLayerNormalization
        if (options is None) or options.enable_bias_gelu:
            # Fuse Gelu and Add Bias before it.
            self.fuse_bias_gelu(is_fastgelu=True)
            self.fuse_bias_gelu(is_fastgelu=False)

        if (options is None) or options.enable_bias_skip_layer_norm:
            # Fuse SkipLayerNormalization and Add Bias before it.
            self.fuse_add_bias_skip_layer_norm()

        if options is not None and options.enable_gelu_approximation:
            self.gelu_approximation()

        if options is not None and options.enable_gemm_fast_gelu:
            self.fuse_gemm_fast_gelu()

        self.remove_unused_constant()

        # Use symbolic batch dimension in input and output.
        if add_dynamic_axes:
            self.use_dynamic_axes()

        logger.info(f"opset version: {self.get_opset_version()}")

    def get_fused_operator_statistics(self):
        """
        Returns node count of fused operators.
        """
        op_count = {}
        ops = [
            "EmbedLayerNormalization",
            "Attention",
            "MultiHeadAttention",
            "Gelu",
            "FastGelu",
            "BiasGelu",
            "GemmFastGelu",
            "LayerNormalization",
            "SimplifiedLayerNormalization",
            "SkipLayerNormalization",
            "SkipSimplifiedLayerNormalization",
            "RotaryEmbedding",
        ]
        q_ops = [
            "QOrderedAttention",
            "QOrderedGelu",
            "QOrderedLayerNormalization",
            "QOrderedMatMul",
        ]
        for op in ops + q_ops:
            nodes = self.get_nodes_by_op_type(op)
            op_count[op] = len(nodes)

        logger.info(f"Optimized operators: {op_count}")
        return op_count

    def is_fully_optimized(self, fused_op_count=None):
        """
        Returns True when the model is fully optimized.
        """
        if fused_op_count is None:
            fused_op_count = self.get_fused_operator_statistics()

        def op_count(op_name: str):
            return fused_op_count.get(op_name) or 0

        embed = op_count("EmbedLayerNormalization")
        attention = op_count("Attention") + op_count("MultiHeadAttention") + op_count("QOrderedAttention")
        gelu = op_count("Gelu") + op_count("BiasGelu") + op_count("FastGelu")
        layer_norm = op_count("LayerNormalization") + op_count("SkipLayerNormalization")
        simple_layer_norm = op_count("SimplifiedLayerNormalization") + op_count("SkipSimplifiedLayerNormalization")

        is_perfect = (
            (embed > 0)
            and (attention > 0)
            and (attention == gelu)
            and ((layer_norm >= 2 * attention) or (simple_layer_norm >= 2 * attention))
        )

        if layer_norm == 0:
            logger.debug("Layer Normalization not fused")

        if simple_layer_norm == 0:
            logger.debug("Simple Layer Normalization not fused")

        if gelu == 0:
            logger.debug("Gelu (or FastGelu) not fused")

        if embed == 0:
            logger.debug("EmbedLayerNormalization not fused")

        if attention == 0:
            logger.warning("Attention (or MultiHeadAttention) not fused")

        return is_perfect

    def convert_to_packing_mode(self, use_symbolic_shape_infer: bool = False):
        packing_mode = PackingMode(self)
        packing_mode.convert(use_symbolic_shape_infer)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\llama\benchmark_all.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
import argparse
import datetime
import json
import logging
import os
import subprocess

import torch
from benchmark_helper import setup_logger
from metrics import BenchmarkRecord

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-b",
        "--batch-sizes",
        type=str,
        default="1 2",
    )

    parser.add_argument(
        "-s",
        "--sequence-lengths",
        type=str,
        default="8 16 32 64 128 256 512",
    )

    parser.add_argument(
        "-w",
        "--warmup-runs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "-n",
        "--num-runs",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--hf-pt-eager",
        default=False,
        action="store_true",
        help="Benchmark in PyTorch without `torch.compile`",
    )

    parser.add_argument(
        "--hf-pt-compile",
        default=False,
        action="store_true",
        help="Benchmark in PyTorch with `torch.compile`",
    )

    parser.add_argument(
        "--hf-ort-dir-path",
        type=str,
        default="",
        help="Path to folder containing ONNX models for Optimum + ORT benchmarking",
    )

    parser.add_argument(
        "--ort-msft-model-path",
        type=str,
        default="",
        help="Path to ONNX model from https://github.com/microsoft/Llama-2-Onnx",
    )

    parser.add_argument(
        "--ort-convert-to-onnx-model-path",
        type=str,
        default="",
        help="Path to ONNX model from convert_to_onnx",
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./model_cache",
        help="Cache dir where Hugging Face files are stored",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name in Hugging Face",
    )

    parser.add_argument(
        "--precision",
        type=str,
        required=True,
        choices=["int4", "int8", "fp16", "fp32"],
        help="Precision to run model",
    )

    parser.add_argument(
        "--device",
        type=str,
        required=True,
        choices=["cpu", "cuda", "rocm"],
        help="Device to benchmark models",
    )

    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="GPU device ID",
    )

    parser.add_argument(
        "--verbose",
        default=False,
        action="store_true",
        help="Print detailed logs",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=10,
        help="Number of mins to attempt the benchmark before moving on",
    )

    parser.add_argument(
        "--log-folder",
        type=str,
        default=None,
        help="Path to folder to save logs and results",
    )

    args = parser.parse_args()

    setattr(args, "model_size", args.model_name.split("/")[-1].replace(".", "-"))  # noqa: B010
    log_folder_name = f"./{args.model_size}_{args.precision}"
    if not args.log_folder:
        args.log_folder = log_folder_name
    os.makedirs(args.log_folder, exist_ok=True)

    # Convert timeout value to secs
    args.timeout *= 60

    return args


def process_log_file(device_id, log_file, base_results):
    entries = []
    batch_size, sequence_length, step = None, None, None
    latency_s, latency_ms, throughput, memory = None, None, None, None

    batch_pattern = "Batch Size: "
    sequence_pattern = "Sequence Length: "
    prompt_step_pattern = "to get past_key_values"
    per_token_step_pattern = "with past_key_values"
    latency_pattern = "Latency: "
    throughput_pattern = "Throughput: "
    memory_pattern = "peak="

    with open(log_file) as f:
        for input_line in f:
            line = input_line.replace("\n", "")

            if batch_pattern in line:
                batch_size = int(line[len(batch_pattern) :])
            elif sequence_pattern in line:
                sequence_length = int(line[len(sequence_pattern) :])
            elif prompt_step_pattern in line:
                step = "prompt"
            elif per_token_step_pattern in line:
                step = "per-token"
            elif latency_pattern in line:
                latency_s = float(line[len(latency_pattern) : line.rfind(" ")])
                latency_ms = latency_s * 1000
            elif throughput_pattern in line:
                throughput = float(line[len(throughput_pattern) : line.rfind(" ")])
            elif memory_pattern in line:
                if "CPU" in line:
                    # Example format for log entry:
                    # CPU memory usage: before=1000.0 MB, peak=2000.0 MB
                    memory = float(line[line.rfind("=") + 1 : line.rfind(" MB")]) / 1000
                else:
                    # Example format for log entry:
                    # GPU memory usage: before=[{'device_id': 0, 'name': 'NVIDIA A100-SXM4-80GB', 'max_used_MB': 69637.25}, {'device_id': 1, 'name': 'NVIDIA A100-SXM4-80GB', 'max_used_MB': 890.625}]  peak=[{'device_id': 0, 'name': 'NVIDIA A100-SXM4-80GB', 'max_used_MB': 73861.25}, {'device_id': 1, 'name': 'NVIDIA A100-SXM4-80GB', 'max_used_MB': 890.625}]
                    peak = line[line.find(memory_pattern) + len(memory_pattern) :].replace("'", '"')
                    usage = json.loads(peak)[device_id]["max_used_MB"]
                    memory = float(usage) / 1000

                # Append log entry to list of entries
                entry = base_results + [  # noqa: RUF005
                    batch_size,
                    sequence_length,
                    step,
                    latency_s,
                    latency_ms,
                    throughput,
                    memory,
                ]
                entries.append(entry)

    return entries


def save_results(results, filename):
    import pandas as pd

    df = pd.DataFrame(
        results,
        columns=[
            "Warmup Runs",
            "Measured Runs",
            "Model Name",
            "Engine",
            "Precision",
            "Device",
            "Batch Size",
            "Sequence Length",
            "Step",
            "Latency (s)",
            "Latency (ms)",
            "Throughput (tps)",
            "Memory (GB)",
        ],
    )

    # Set column types
    df["Warmup Runs"] = df["Warmup Runs"].astype("int")
    df["Measured Runs"] = df["Measured Runs"].astype("int")
    df["Batch Size"] = df["Batch Size"].astype("int")
    df["Sequence Length"] = df["Sequence Length"].astype("int")
    df["Latency (s)"] = df["Latency (s)"].astype("float")
    df["Latency (ms)"] = df["Latency (ms)"].astype("float")
    df["Throughput (tps)"] = df["Throughput (tps)"].astype("float")
    df["Memory (GB)"] = df["Memory (GB)"].astype("float")

    # get package name and version
    import pkg_resources

    installed_packages = pkg_resources.working_set
    installed_packages_list = sorted(
        [f"{i.key}=={i.version}" for i in installed_packages if i.key in ["onnxruntime", "onnxruntime-gpu"]]
    )

    ort_pkg_name = ""
    ort_pkg_version = ""
    if installed_packages_list:
        ort_pkg_name = installed_packages_list[0].split("==")[0]
        ort_pkg_version = installed_packages_list[0].split("==")[1]

    # Save results to csv with standard format
    records = []
    for _, row in df.iterrows():
        if row["Engine"] in ["optimum-ort", "onnxruntime"]:
            record = BenchmarkRecord(
                row["Model Name"], row["Precision"], "onnxruntime", row["Device"], ort_pkg_name, ort_pkg_version
            )
        elif row["Engine"] in ["pytorch-eager", "pytorch-compile"]:
            record = BenchmarkRecord(
                row["Model Name"], row["Precision"], "pytorch", row["Device"], torch.__name__, torch.__version__
            )
        else:
            record = BenchmarkRecord(row["Model Name"], row["Precision"], row["Engine"], row["Device"], "", "")
        record.config.warmup_runs = row["Warmup Runs"]
        record.config.measured_runs = row["Measured Runs"]
        record.config.batch_size = row["Batch Size"]
        record.config.seq_length = row["Sequence Length"]
        record.config.customized["measure_step"] = row["Step"]
        record.config.customized["engine"] = row["Engine"]
        record.metrics.customized["latency_s_mean"] = row["Latency (s)"]
        record.metrics.latency_ms_mean = row["Latency (ms)"]
        record.metrics.customized["throughput_tps"] = row["Throughput (tps)"]
        record.metrics.max_memory_usage_GB = row["Memory (GB)"]

        records.append(record)

    BenchmarkRecord.save_as_csv(filename, records)
    BenchmarkRecord.save_as_json(filename.replace(".csv", ".json"), records)
    logger.info(f"Results saved in {filename}!")


def benchmark(args, benchmark_cmd, engine):
    log_filename = f"{engine}_{datetime.datetime.now():%Y-%m-%d_%H:%M:%S}.log"
    log_path = os.path.join(args.log_folder, log_filename)
    with open(log_path, "w") as log_file:
        process = subprocess.Popen(benchmark_cmd, stdout=log_file, stderr=log_file)
        try:
            process.wait(args.timeout)
        except subprocess.TimeoutExpired:
            process.kill()

    # Create entries for csv
    logger.info("Gathering data from log files...")
    base_results = [args.warmup_runs, args.num_runs, args.model_name, engine, args.precision, args.device]
    results = process_log_file(args.device_id, log_path, base_results)

    return results


def main():
    args = get_args()
    setup_logger(args.verbose)
    logger.info(args.__dict__)
    torch.backends.cudnn.benchmark = True

    all_results = []
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)

    # Benchmark PyTorch without torch.compile
    if args.hf_pt_eager:
        benchmark_cmd = [
            "python",
            "-m",
            "models.llama.benchmark",
            "--benchmark-type",
            "hf-pt-eager",
            "--model-name",
            args.model_name,
            "--precision",
            args.precision,
            "--batch-sizes",
            args.batch_sizes,
            "--sequence-lengths",
            args.sequence_lengths,
            "--device",
            args.device,
            "--warmup-runs",
            str(args.warmup_runs),
            "--num-runs",
            str(args.num_runs),
            "--log-folder",
            args.log_folder,
            "--cache-dir",
            args.cache_dir,
            "--auth",
        ]
        logger.info("Benchmark PyTorch without torch.compile")
        results = benchmark(args, benchmark_cmd, "pytorch-eager")
        all_results.extend(results)

    # Benchmark PyTorch with torch.compile
    if args.hf_pt_compile:
        benchmark_cmd = [
            "python",
            "-m",
            "models.llama.benchmark",
            "--benchmark-type",
            "hf-pt-compile",
            "--model-name",
            args.model_name,
            "--precision",
            args.precision,
            "--batch-sizes",
            args.batch_sizes,
            "--sequence-lengths",
            args.sequence_lengths,
            "--device",
            args.device,
            "--warmup-runs",
            str(args.warmup_runs),
            "--num-runs",
            str(args.num_runs),
            "--log-folder",
            args.log_folder,
            "--cache-dir",
            args.cache_dir,
            "--auth",
        ]
        logger.info("Benchmark PyTorch with torch.compile")
        results = benchmark(args, benchmark_cmd, "pytorch-compile")
        all_results.extend(results)

    # Benchmark Optimum + ONNX Runtime
    if args.hf_ort_dir_path:
        benchmark_cmd = [
            "python",
            "-m",
            "models.llama.benchmark",
            "--benchmark-type",
            "hf-ort",
            "--hf-ort-dir-path",
            args.hf_ort_dir_path,
            "--model-name",
            args.model_name,
            "--precision",
            args.precision,
            "--batch-sizes",
            args.batch_sizes,
            "--sequence-lengths",
            args.sequence_lengths,
            "--device",
            args.device,
            "--warmup-runs",
            str(args.warmup_runs),
            "--num-runs",
            str(args.num_runs),
            "--log-folder",
            args.log_folder,
            "--cache-dir",
            args.cache_dir,
            "--auth",
        ]
        logger.info("Benchmark Optimum + ONNX Runtime")
        results = benchmark(args, benchmark_cmd, "optimum-ort")
        all_results.extend(results)

    # Benchmark Microsoft model in ONNX Runtime
    if args.ort_msft_model_path:
        benchmark_cmd = [
            "python",
            "-m",
            "models.llama.benchmark",
            "--benchmark-type",
            "ort-msft",
            "--ort-model-path",
            args.ort_msft_model_path,
            "--model-name",
            args.model_name,
            "--precision",
            args.precision,
            "--batch-sizes",
            args.batch_sizes,
            "--sequence-lengths",
            args.sequence_lengths,
            "--device",
            args.device,
            "--warmup-runs",
            str(args.warmup_runs),
            "--num-runs",
            str(args.num_runs),
            "--log-folder",
            args.log_folder,
            "--cache-dir",
            args.cache_dir,
        ]
        logger.info("Benchmark Microsoft model in ONNX Runtime")
        results = benchmark(args, benchmark_cmd, "ort-msft")
        all_results.extend(results)

    # Benchmark convert_to_onnx model in ONNX Runtime
    if args.ort_convert_to_onnx_model_path:
        benchmark_cmd = [
            "python",
            "-m",
            "models.llama.benchmark",
            "--benchmark-type",
            "ort-convert-to-onnx",
            "--ort-model-path",
            args.ort_convert_to_onnx_model_path,
            "--model-name",
            args.model_name,
            "--precision",
            args.precision,
            "--batch-sizes",
            args.batch_sizes,
            "--sequence-lengths",
            args.sequence_lengths,
            "--device",
            args.device,
            "--warmup-runs",
            str(args.warmup_runs),
            "--num-runs",
            str(args.num_runs),
            "--log-folder",
            args.log_folder,
            "--cache-dir",
            args.cache_dir,
        ]
        logger.info("Benchmark convert_to_onnx model in ONNX Runtime")
        results = benchmark(args, benchmark_cmd, "onnxruntime")
        all_results.extend(results)

    csv_file = f"{args.model_size}_{args.precision}_{datetime.datetime.now():%Y-%m-%d_%H:%M:%S}.csv"
    save_results(all_results, os.path.join(args.log_folder, csv_file))


if __name__ == "__main__":
    main()