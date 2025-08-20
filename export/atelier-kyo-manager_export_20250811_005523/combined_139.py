
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\longformer\generate_test_data.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# Generate test data for a longformer model, so that we can use onnxruntime_perf_test.exe to evaluate the inference latency.

import argparse
import os
import random
from pathlib import Path

import numpy as np
from bert_test_data import fake_input_ids_data, fake_input_mask_data, output_test_data
from onnx import ModelProto, TensorProto
from onnx_model import OnnxModel


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True, type=str, help="bert onnx model path.")

    parser.add_argument(
        "--output_dir",
        required=False,
        type=str,
        default=None,
        help="output test data path. If not specified, .",
    )

    parser.add_argument("--batch_size", required=False, type=int, default=1, help="batch size of input")

    parser.add_argument(
        "--sequence_length",
        required=False,
        type=int,
        default=128,
        help="maximum sequence length of input",
    )

    parser.add_argument(
        "-a",
        "--average_sequence_length",
        default=-1,
        type=int,
        help="average sequence length excluding padding",
    )

    parser.add_argument(
        "-r",
        "--random_sequence_length",
        required=False,
        action="store_true",
        help="use uniform random instead of fixed sequence length",
    )
    parser.set_defaults(random_sequence_length=False)

    parser.add_argument(
        "--global_tokens",
        required=False,
        type=int,
        default=10,
        help="number of global tokens",
    )

    parser.add_argument(
        "--input_ids_name",
        required=False,
        type=str,
        default=None,
        help="input name for input ids",
    )

    parser.add_argument(
        "--input_mask_name",
        required=False,
        type=str,
        default=None,
        help="input name for attention mask",
    )

    parser.add_argument(
        "--global_mask_name",
        required=False,
        type=str,
        default=None,
        help="input name for global attention mask",
    )

    parser.add_argument(
        "--samples",
        required=False,
        type=int,
        default=1,
        help="number of test cases to be generated",
    )

    parser.add_argument("--seed", required=False, type=int, default=3, help="random seed")

    parser.add_argument(
        "--verbose",
        required=False,
        action="store_true",
        help="print verbose information",
    )
    parser.set_defaults(verbose=False)

    args = parser.parse_args()
    return args


def get_longformer_inputs(onnx_file, input_ids_name=None, input_mask_name=None, global_mask_name=None):
    """
    Get graph inputs for longformer model.
    """
    model = ModelProto()
    with open(onnx_file, "rb") as f:
        model.ParseFromString(f.read())

    onnx_model = OnnxModel(model)
    graph_inputs = onnx_model.get_graph_inputs_excluding_initializers()

    if input_ids_name is not None:
        input_ids = onnx_model.find_graph_input(input_ids_name)
        if input_ids is None:
            raise ValueError(f"Graph does not have input named {input_ids_name}")

        input_mask = None
        if input_mask_name:
            input_mask = onnx_model.find_graph_input(input_mask_name)
            if input_mask is None:
                raise ValueError(f"Graph does not have input named {input_mask_name}")

        global_mask = None
        if global_mask_name:
            global_mask = onnx_model.find_graph_input(global_mask_name)
            if global_mask is None:
                raise ValueError(f"Graph does not have input named {global_mask_name}")

        expected_inputs = 1 + (1 if input_mask else 0) + (1 if global_mask else 0)
        if len(graph_inputs) != expected_inputs:
            raise ValueError(f"Expect the graph to have {expected_inputs} inputs. Got {len(graph_inputs)}")

        return input_ids, input_mask, global_mask

    if len(graph_inputs) != 3:
        raise ValueError(f"Expect the graph to have 3 inputs. Got {len(graph_inputs)}")

    # Try guess the inputs based on naming.
    input_ids = None
    input_mask = None
    global_mask = None
    for input in graph_inputs:
        input_name_lower = input.name.lower()
        if "global" in input_name_lower:
            global_mask = input
        elif "mask" in input_name_lower:
            input_mask = input
        else:
            input_ids = input

    if input_ids and input_mask and global_mask:
        return input_ids, input_mask, global_mask

    raise ValueError("Fail to assign 3 inputs. You might try rename the graph inputs.")


def fake_global_mask_data(global_mask, batch_size, sequence_length, num_global_tokens):
    """
    Fake data based on the graph input of segment_ids.
    Args:
        segment_ids (TensorProto): graph input of input tensor.
    Returns:
        data (np.array): the data for input tensor
    """
    data_type = global_mask.type.tensor_type.elem_type
    assert data_type in [TensorProto.FLOAT, TensorProto.INT32, TensorProto.INT64]

    if num_global_tokens > 0:
        assert num_global_tokens <= sequence_length
        data = np.zeros((batch_size, sequence_length), dtype=np.int32)
        temp = np.ones((batch_size, num_global_tokens), dtype=np.int32)
        data[: temp.shape[0], : temp.shape[1]] = temp
    else:
        data = np.zeros((batch_size, sequence_length), dtype=np.int32)

    if data_type == TensorProto.FLOAT:
        data = np.float32(data)
    elif data_type == TensorProto.INT64:
        data = np.int64(data)

    return data


def fake_test_data(
    batch_size,
    sequence_length,
    test_cases,
    dictionary_size,
    verbose,
    random_seed,
    input_ids,
    input_mask,
    global_mask,
    num_global_tokens,
    average_sequence_length,
    random_sequence_length,
):
    """
    Generate fake input data for test.
    """
    assert input_ids is not None

    np.random.seed(random_seed)
    random.seed(random_seed)

    all_inputs = []
    for _ in range(test_cases):
        input_1 = fake_input_ids_data(input_ids, batch_size, sequence_length, dictionary_size)
        inputs = {input_ids.name: input_1}

        if input_mask:
            inputs[input_mask.name] = fake_input_mask_data(
                input_mask, batch_size, sequence_length, average_sequence_length, random_sequence_length
            )

        if global_mask:
            inputs[global_mask.name] = fake_global_mask_data(
                global_mask, batch_size, sequence_length, num_global_tokens
            )

        if verbose and len(all_inputs) == 0:
            print("Example inputs", inputs)
        all_inputs.append(inputs)

    return all_inputs


def generate_test_data(
    batch_size,
    sequence_length,
    test_cases,
    seed,
    verbose,
    input_ids,
    input_mask,
    global_mask,
    num_global_tokens,
    average_sequence_length,
    random_sequence_length,
):
    dictionary_size = 10000
    all_inputs = fake_test_data(
        batch_size,
        sequence_length,
        test_cases,
        dictionary_size,
        verbose,
        seed,
        input_ids,
        input_mask,
        global_mask,
        num_global_tokens,
        average_sequence_length,
        random_sequence_length,
    )
    if len(all_inputs) != test_cases:
        print("Failed to create test data for test.")
    return all_inputs


def create_longformer_test_data(
    model,
    output_dir,
    batch_size,
    sequence_length,
    test_cases,
    seed,
    verbose,
    input_ids_name,
    input_mask_name,
    global_mask_name,
    num_global_tokens,
    average_sequence_length,
    random_sequence_length,
):
    input_ids, input_mask, global_mask = get_longformer_inputs(model, input_ids_name, input_mask_name, global_mask_name)
    all_inputs = generate_test_data(
        batch_size,
        sequence_length,
        test_cases,
        seed,
        verbose,
        input_ids,
        input_mask,
        global_mask,
        num_global_tokens,
        average_sequence_length,
        random_sequence_length,
    )

    for i, inputs in enumerate(all_inputs):
        output_test_data(output_dir, i, inputs)


def main():
    args = parse_arguments()

    output_dir = args.output_dir
    if output_dir is None:
        # Default output directory is a sub-directory under the directory of model.
        output_dir = os.path.join(
            Path(args.model).parent,
            f"b{args.batch_size}_s{args.sequence_length}_g{args.global_tokens}",
        )

    if output_dir is not None:
        # create the output directory if not existed
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
    else:
        print("Directory existed. test data files will be overwritten.")

    if args.average_sequence_length <= 0:
        args.average_sequence_length = args.sequence_length

    create_longformer_test_data(
        args.model,
        output_dir,
        args.batch_size,
        args.sequence_length,
        args.samples,
        args.seed,
        args.verbose,
        args.input_ids_name,
        args.input_mask_name,
        args.global_mask_name,
        args.global_tokens,
        args.average_sequence_length,
    )

    print("Test data is saved to directory:", output_dir)


if __name__ == "__main__":
    main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\cli\req_command.py ===
"""Contains the RequirementCommand base class.

This class is in a separate module so the commands that do not always
need PackageFinder capability don't unnecessarily import the
PackageFinder machinery and all its vendored dependencies, etc.
"""

import logging
from functools import partial
from optparse import Values
from typing import Any, List, Optional, Tuple

from pip._internal.cache import WheelCache
from pip._internal.cli import cmdoptions
from pip._internal.cli.index_command import IndexGroupCommand
from pip._internal.cli.index_command import SessionCommandMixin as SessionCommandMixin
from pip._internal.exceptions import CommandError, PreviousBuildDirError
from pip._internal.index.collector import LinkCollector
from pip._internal.index.package_finder import PackageFinder
from pip._internal.models.selection_prefs import SelectionPreferences
from pip._internal.models.target_python import TargetPython
from pip._internal.network.session import PipSession
from pip._internal.operations.build.build_tracker import BuildTracker
from pip._internal.operations.prepare import RequirementPreparer
from pip._internal.req.constructors import (
    install_req_from_editable,
    install_req_from_line,
    install_req_from_parsed_requirement,
    install_req_from_req_string,
)
from pip._internal.req.req_dependency_group import parse_dependency_groups
from pip._internal.req.req_file import parse_requirements
from pip._internal.req.req_install import InstallRequirement
from pip._internal.resolution.base import BaseResolver
from pip._internal.utils.temp_dir import (
    TempDirectory,
    TempDirectoryTypeRegistry,
    tempdir_kinds,
)

logger = logging.getLogger(__name__)


KEEPABLE_TEMPDIR_TYPES = [
    tempdir_kinds.BUILD_ENV,
    tempdir_kinds.EPHEM_WHEEL_CACHE,
    tempdir_kinds.REQ_BUILD,
]


def with_cleanup(func: Any) -> Any:
    """Decorator for common logic related to managing temporary
    directories.
    """

    def configure_tempdir_registry(registry: TempDirectoryTypeRegistry) -> None:
        for t in KEEPABLE_TEMPDIR_TYPES:
            registry.set_delete(t, False)

    def wrapper(
        self: RequirementCommand, options: Values, args: List[Any]
    ) -> Optional[int]:
        assert self.tempdir_registry is not None
        if options.no_clean:
            configure_tempdir_registry(self.tempdir_registry)

        try:
            return func(self, options, args)
        except PreviousBuildDirError:
            # This kind of conflict can occur when the user passes an explicit
            # build directory with a pre-existing folder. In that case we do
            # not want to accidentally remove it.
            configure_tempdir_registry(self.tempdir_registry)
            raise

    return wrapper


class RequirementCommand(IndexGroupCommand):
    def __init__(self, *args: Any, **kw: Any) -> None:
        super().__init__(*args, **kw)

        self.cmd_opts.add_option(cmdoptions.dependency_groups())
        self.cmd_opts.add_option(cmdoptions.no_clean())

    @staticmethod
    def determine_resolver_variant(options: Values) -> str:
        """Determines which resolver should be used, based on the given options."""
        if "legacy-resolver" in options.deprecated_features_enabled:
            return "legacy"

        return "resolvelib"

    @classmethod
    def make_requirement_preparer(
        cls,
        temp_build_dir: TempDirectory,
        options: Values,
        build_tracker: BuildTracker,
        session: PipSession,
        finder: PackageFinder,
        use_user_site: bool,
        download_dir: Optional[str] = None,
        verbosity: int = 0,
    ) -> RequirementPreparer:
        """
        Create a RequirementPreparer instance for the given parameters.
        """
        temp_build_dir_path = temp_build_dir.path
        assert temp_build_dir_path is not None
        legacy_resolver = False

        resolver_variant = cls.determine_resolver_variant(options)
        if resolver_variant == "resolvelib":
            lazy_wheel = "fast-deps" in options.features_enabled
            if lazy_wheel:
                logger.warning(
                    "pip is using lazily downloaded wheels using HTTP "
                    "range requests to obtain dependency information. "
                    "This experimental feature is enabled through "
                    "--use-feature=fast-deps and it is not ready for "
                    "production."
                )
        else:
            legacy_resolver = True
            lazy_wheel = False
            if "fast-deps" in options.features_enabled:
                logger.warning(
                    "fast-deps has no effect when used with the legacy resolver."
                )

        return RequirementPreparer(
            build_dir=temp_build_dir_path,
            src_dir=options.src_dir,
            download_dir=download_dir,
            build_isolation=options.build_isolation,
            check_build_deps=options.check_build_deps,
            build_tracker=build_tracker,
            session=session,
            progress_bar=options.progress_bar,
            finder=finder,
            require_hashes=options.require_hashes,
            use_user_site=use_user_site,
            lazy_wheel=lazy_wheel,
            verbosity=verbosity,
            legacy_resolver=legacy_resolver,
            resume_retries=options.resume_retries,
        )

    @classmethod
    def make_resolver(
        cls,
        preparer: RequirementPreparer,
        finder: PackageFinder,
        options: Values,
        wheel_cache: Optional[WheelCache] = None,
        use_user_site: bool = False,
        ignore_installed: bool = True,
        ignore_requires_python: bool = False,
        force_reinstall: bool = False,
        upgrade_strategy: str = "to-satisfy-only",
        use_pep517: Optional[bool] = None,
        py_version_info: Optional[Tuple[int, ...]] = None,
    ) -> BaseResolver:
        """
        Create a Resolver instance for the given parameters.
        """
        make_install_req = partial(
            install_req_from_req_string,
            isolated=options.isolated_mode,
            use_pep517=use_pep517,
        )
        resolver_variant = cls.determine_resolver_variant(options)
        # The long import name and duplicated invocation is needed to convince
        # Mypy into correctly typechecking. Otherwise it would complain the
        # "Resolver" class being redefined.
        if resolver_variant == "resolvelib":
            import pip._internal.resolution.resolvelib.resolver

            return pip._internal.resolution.resolvelib.resolver.Resolver(
                preparer=preparer,
                finder=finder,
                wheel_cache=wheel_cache,
                make_install_req=make_install_req,
                use_user_site=use_user_site,
                ignore_dependencies=options.ignore_dependencies,
                ignore_installed=ignore_installed,
                ignore_requires_python=ignore_requires_python,
                force_reinstall=force_reinstall,
                upgrade_strategy=upgrade_strategy,
                py_version_info=py_version_info,
            )
        import pip._internal.resolution.legacy.resolver

        return pip._internal.resolution.legacy.resolver.Resolver(
            preparer=preparer,
            finder=finder,
            wheel_cache=wheel_cache,
            make_install_req=make_install_req,
            use_user_site=use_user_site,
            ignore_dependencies=options.ignore_dependencies,
            ignore_installed=ignore_installed,
            ignore_requires_python=ignore_requires_python,
            force_reinstall=force_reinstall,
            upgrade_strategy=upgrade_strategy,
            py_version_info=py_version_info,
        )

    def get_requirements(
        self,
        args: List[str],
        options: Values,
        finder: PackageFinder,
        session: PipSession,
    ) -> List[InstallRequirement]:
        """
        Parse command-line arguments into the corresponding requirements.
        """
        requirements: List[InstallRequirement] = []
        for filename in options.constraints:
            for parsed_req in parse_requirements(
                filename,
                constraint=True,
                finder=finder,
                options=options,
                session=session,
            ):
                req_to_add = install_req_from_parsed_requirement(
                    parsed_req,
                    isolated=options.isolated_mode,
                    user_supplied=False,
                )
                requirements.append(req_to_add)

        for req in args:
            req_to_add = install_req_from_line(
                req,
                comes_from=None,
                isolated=options.isolated_mode,
                use_pep517=options.use_pep517,
                user_supplied=True,
                config_settings=getattr(options, "config_settings", None),
            )
            requirements.append(req_to_add)

        if options.dependency_groups:
            for req in parse_dependency_groups(options.dependency_groups):
                req_to_add = install_req_from_req_string(
                    req,
                    isolated=options.isolated_mode,
                    use_pep517=options.use_pep517,
                    user_supplied=True,
                )
                requirements.append(req_to_add)

        for req in options.editables:
            req_to_add = install_req_from_editable(
                req,
                user_supplied=True,
                isolated=options.isolated_mode,
                use_pep517=options.use_pep517,
                config_settings=getattr(options, "config_settings", None),
            )
            requirements.append(req_to_add)

        # NOTE: options.require_hashes may be set if --require-hashes is True
        for filename in options.requirements:
            for parsed_req in parse_requirements(
                filename, finder=finder, options=options, session=session
            ):
                req_to_add = install_req_from_parsed_requirement(
                    parsed_req,
                    isolated=options.isolated_mode,
                    use_pep517=options.use_pep517,
                    user_supplied=True,
                    config_settings=(
                        parsed_req.options.get("config_settings")
                        if parsed_req.options
                        else None
                    ),
                )
                requirements.append(req_to_add)

        # If any requirement has hash options, enable hash checking.
        if any(req.has_hash_options for req in requirements):
            options.require_hashes = True

        if not (
            args
            or options.editables
            or options.requirements
            or options.dependency_groups
        ):
            opts = {"name": self.name}
            if options.find_links:
                raise CommandError(
                    "You must give at least one requirement to {name} "
                    '(maybe you meant "pip {name} {links}"?)'.format(
                        **dict(opts, links=" ".join(options.find_links))
                    )
                )
            else:
                raise CommandError(
                    "You must give at least one requirement to {name} "
                    '(see "pip help {name}")'.format(**opts)
                )

        return requirements

    @staticmethod
    def trace_basic_info(finder: PackageFinder) -> None:
        """
        Trace basic information about the provided objects.
        """
        # Display where finder is looking for packages
        search_scope = finder.search_scope
        locations = search_scope.get_formatted_locations()
        if locations:
            logger.info(locations)

    def _build_package_finder(
        self,
        options: Values,
        session: PipSession,
        target_python: Optional[TargetPython] = None,
        ignore_requires_python: Optional[bool] = None,
    ) -> PackageFinder:
        """
        Create a package finder appropriate to this requirement command.

        :param ignore_requires_python: Whether to ignore incompatible
            "Requires-Python" values in links. Defaults to False.
        """
        link_collector = LinkCollector.create(session, options=options)
        selection_prefs = SelectionPreferences(
            allow_yanked=True,
            format_control=options.format_control,
            allow_all_prereleases=options.pre,
            prefer_binary=options.prefer_binary,
            ignore_requires_python=ignore_requires_python,
        )

        return PackageFinder.create(
            link_collector=link_collector,
            selection_prefs=selection_prefs,
            target_python=target_python,
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\fortran\fortran_parser.py ===
from sympy.external import import_module

lfortran = import_module('lfortran')

if lfortran:
    from sympy.codegen.ast import (Variable, IntBaseType, FloatBaseType, String,
                                   Return, FunctionDefinition, Assignment)
    from sympy.core import Add, Mul, Integer, Float
    from sympy.core.symbol import Symbol

    asr_mod = lfortran.asr
    asr = lfortran.asr.asr
    src_to_ast = lfortran.ast.src_to_ast
    ast_to_asr = lfortran.semantic.ast_to_asr.ast_to_asr

    """
    This module contains all the necessary Classes and Function used to Parse
    Fortran code into SymPy expression

    The module and its API are currently under development and experimental.
    It is also dependent on LFortran for the ASR that is converted to SymPy syntax
    which is also under development.
    The module only supports the features currently supported by the LFortran ASR
    which will be updated as the development of LFortran and this module progresses

    You might find unexpected bugs and exceptions while using the module, feel free
    to report them to the SymPy Issue Tracker

    The API for the module might also change while in development if better and
    more effective ways are discovered for the process

    Features Supported
    ==================

    - Variable Declarations (integers and reals)
    - Function Definitions
    - Assignments and Basic Binary Operations


    Notes
    =====

    The module depends on an external dependency

    LFortran : Required to parse Fortran source code into ASR


    References
    ==========

    .. [1] https://github.com/sympy/sympy/issues
    .. [2] https://gitlab.com/lfortran/lfortran
    .. [3] https://docs.lfortran.org/

    """


    class ASR2PyVisitor(asr.ASTVisitor):  # type: ignore
        """
        Visitor Class for LFortran ASR

        It is a Visitor class derived from asr.ASRVisitor which visits all the
        nodes of the LFortran ASR and creates corresponding AST node for each
        ASR node

        """

        def __init__(self):
            """Initialize the Parser"""
            self._py_ast = []

        def visit_TranslationUnit(self, node):
            """
            Function to visit all the elements of the Translation Unit
            created by LFortran ASR
            """
            for s in node.global_scope.symbols:
                sym = node.global_scope.symbols[s]
                self.visit(sym)
            for item in node.items:
                self.visit(item)

        def visit_Assignment(self, node):
            """Visitor Function for Assignment

            Visits each Assignment is the LFortran ASR and creates corresponding
            assignment for SymPy.

            Notes
            =====

            The function currently only supports variable assignment and binary
            operation assignments of varying multitudes. Any type of numberS or
            array is not supported.

            Raises
            ======

            NotImplementedError() when called for Numeric assignments or Arrays

            """
            # TODO: Arithmetic Assignment
            if isinstance(node.target, asr.Variable):
                target = node.target
                value = node.value
                if isinstance(value, asr.Variable):
                    new_node = Assignment(
                            Variable(
                                    target.name
                                ),
                            Variable(
                                    value.name
                                )
                        )
                elif (type(value) == asr.BinOp):
                    exp_ast = call_visitor(value)
                    for expr in exp_ast:
                        new_node = Assignment(
                                Variable(target.name),
                                expr
                            )
                else:
                    raise NotImplementedError("Numeric assignments not supported")
            else:
                raise NotImplementedError("Arrays not supported")
            self._py_ast.append(new_node)

        def visit_BinOp(self, node):
            """Visitor Function for Binary Operations

            Visits each binary operation present in the LFortran ASR like addition,
            subtraction, multiplication, division and creates the corresponding
            operation node in SymPy's AST

            In case of more than one binary operations, the function calls the
            call_visitor() function on the child nodes of the binary operations
            recursively until all the operations have been processed.

            Notes
            =====

            The function currently only supports binary operations with Variables
            or other binary operations. Numerics are not supported as of yet.

            Raises
            ======

            NotImplementedError() when called for Numeric assignments

            """
            # TODO: Integer Binary Operations
            op = node.op
            lhs = node.left
            rhs = node.right

            if (type(lhs) == asr.Variable):
                left_value = Symbol(lhs.name)
            elif(type(lhs) == asr.BinOp):
                l_exp_ast = call_visitor(lhs)
                for exp in l_exp_ast:
                    left_value = exp
            else:
                raise NotImplementedError("Numbers Currently not supported")

            if (type(rhs) == asr.Variable):
                right_value = Symbol(rhs.name)
            elif(type(rhs) == asr.BinOp):
                r_exp_ast = call_visitor(rhs)
                for exp in r_exp_ast:
                    right_value = exp
            else:
                raise NotImplementedError("Numbers Currently not supported")

            if isinstance(op, asr.Add):
                new_node = Add(left_value, right_value)
            elif isinstance(op, asr.Sub):
                new_node = Add(left_value, -right_value)
            elif isinstance(op, asr.Div):
                new_node = Mul(left_value, 1/right_value)
            elif isinstance(op, asr.Mul):
                new_node = Mul(left_value, right_value)

            self._py_ast.append(new_node)

        def visit_Variable(self, node):
            """Visitor Function for Variable Declaration

            Visits each variable declaration present in the ASR and creates a
            Symbol declaration for each variable

            Notes
            =====

            The functions currently only support declaration of integer and
            real variables. Other data types are still under development.

            Raises
            ======

            NotImplementedError() when called for unsupported data types

            """
            if isinstance(node.type, asr.Integer):
                var_type = IntBaseType(String('integer'))
                value = Integer(0)
            elif isinstance(node.type, asr.Real):
                var_type = FloatBaseType(String('real'))
                value = Float(0.0)
            else:
                raise NotImplementedError("Data type not supported")

            if not (node.intent == 'in'):
                new_node = Variable(
                    node.name
                ).as_Declaration(
                    type = var_type,
                    value = value
                )
                self._py_ast.append(new_node)

        def visit_Sequence(self, seq):
            """Visitor Function for code sequence

            Visits a code sequence/ block and calls the visitor function on all the
            children of the code block to create corresponding code in python

            """
            if seq is not None:
                for node in seq:
                    self._py_ast.append(call_visitor(node))

        def visit_Num(self, node):
            """Visitor Function for Numbers in ASR

            This function is currently under development and will be updated
            with improvements in the LFortran ASR

            """
            # TODO:Numbers when the LFortran ASR is updated
            # self._py_ast.append(Integer(node.n))
            pass

        def visit_Function(self, node):
            """Visitor Function for function Definitions

            Visits each function definition present in the ASR and creates a
            function definition node in the Python AST with all the elements of the
            given function

            The functions declare all the variables required as SymPy symbols in
            the function before the function definition

            This function also the call_visior_function to parse the contents of
            the function body

            """
            # TODO: Return statement, variable declaration
            fn_args = [Variable(arg_iter.name) for arg_iter in node.args]
            fn_body = []
            fn_name = node.name
            for i in node.body:
                fn_ast = call_visitor(i)
            try:
                fn_body_expr = fn_ast
            except UnboundLocalError:
                fn_body_expr = []
            for sym in node.symtab.symbols:
                decl = call_visitor(node.symtab.symbols[sym])
                for symbols in decl:
                    fn_body.append(symbols)
            for elem in fn_body_expr:
                fn_body.append(elem)
            fn_body.append(
                Return(
                    Variable(
                        node.return_var.name
                    )
                )
            )
            if isinstance(node.return_var.type, asr.Integer):
                ret_type = IntBaseType(String('integer'))
            elif isinstance(node.return_var.type, asr.Real):
                ret_type = FloatBaseType(String('real'))
            else:
                raise NotImplementedError("Data type not supported")
            new_node = FunctionDefinition(
                        return_type = ret_type,
                        name = fn_name,
                        parameters = fn_args,
                        body = fn_body
                    )
            self._py_ast.append(new_node)

        def ret_ast(self):
            """Returns the AST nodes"""
            return self._py_ast
else:
    class ASR2PyVisitor():  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError('lfortran not available')

def call_visitor(fort_node):
    """Calls the AST Visitor on the Module

    This function is used to call the AST visitor for a program or module
    It imports all the required modules and calls the visit() function
    on the given node

    Parameters
    ==========

    fort_node : LFortran ASR object
        Node for the operation for which the NodeVisitor is called

    Returns
    =======

    res_ast : list
        list of SymPy AST Nodes

    """
    v = ASR2PyVisitor()
    v.visit(fort_node)
    res_ast = v.ret_ast()
    return res_ast


def src_to_sympy(src):
    """Wrapper function to convert the given Fortran source code to SymPy Expressions

    Parameters
    ==========

    src : string
        A string with the Fortran source code

    Returns
    =======

    py_src : string
        A string with the Python source code compatible with SymPy

    """
    a_ast = src_to_ast(src, translation_unit=False)
    a = ast_to_asr(a_ast)
    py_src = call_visitor(a)
    return py_src

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_array_api_info.py ===
"""
Array API Inspection namespace

This is the namespace for inspection functions as defined by the array API
standard. See
https://data-apis.org/array-api/latest/API_specification/inspection.html for
more details.

"""
from numpy._core import (
    bool,
    complex64,
    complex128,
    dtype,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    intp,
    uint8,
    uint16,
    uint32,
    uint64,
)


class __array_namespace_info__:
    """
    Get the array API inspection namespace for NumPy.

    The array API inspection namespace defines the following functions:

    - capabilities()
    - default_device()
    - default_dtypes()
    - dtypes()
    - devices()

    See
    https://data-apis.org/array-api/latest/API_specification/inspection.html
    for more details.

    Returns
    -------
    info : ModuleType
        The array API inspection namespace for NumPy.

    Examples
    --------
    >>> info = np.__array_namespace_info__()
    >>> info.default_dtypes()
    {'real floating': numpy.float64,
     'complex floating': numpy.complex128,
     'integral': numpy.int64,
     'indexing': numpy.int64}

    """

    __module__ = 'numpy'

    def capabilities(self):
        """
        Return a dictionary of array API library capabilities.

        The resulting dictionary has the following keys:

        - **"boolean indexing"**: boolean indicating whether an array library
          supports boolean indexing. Always ``True`` for NumPy.

        - **"data-dependent shapes"**: boolean indicating whether an array
          library supports data-dependent output shapes. Always ``True`` for
          NumPy.

        See
        https://data-apis.org/array-api/latest/API_specification/generated/array_api.info.capabilities.html
        for more details.

        See Also
        --------
        __array_namespace_info__.default_device,
        __array_namespace_info__.default_dtypes,
        __array_namespace_info__.dtypes,
        __array_namespace_info__.devices

        Returns
        -------
        capabilities : dict
            A dictionary of array API library capabilities.

        Examples
        --------
        >>> info = np.__array_namespace_info__()
        >>> info.capabilities()
        {'boolean indexing': True,
         'data-dependent shapes': True}

        """
        return {
            "boolean indexing": True,
            "data-dependent shapes": True,
            # 'max rank' will be part of the 2024.12 standard
            # "max rank": 64,
        }

    def default_device(self):
        """
        The default device used for new NumPy arrays.

        For NumPy, this always returns ``'cpu'``.

        See Also
        --------
        __array_namespace_info__.capabilities,
        __array_namespace_info__.default_dtypes,
        __array_namespace_info__.dtypes,
        __array_namespace_info__.devices

        Returns
        -------
        device : str
            The default device used for new NumPy arrays.

        Examples
        --------
        >>> info = np.__array_namespace_info__()
        >>> info.default_device()
        'cpu'

        """
        return "cpu"

    def default_dtypes(self, *, device=None):
        """
        The default data types used for new NumPy arrays.

        For NumPy, this always returns the following dictionary:

        - **"real floating"**: ``numpy.float64``
        - **"complex floating"**: ``numpy.complex128``
        - **"integral"**: ``numpy.intp``
        - **"indexing"**: ``numpy.intp``

        Parameters
        ----------
        device : str, optional
            The device to get the default data types for. For NumPy, only
            ``'cpu'`` is allowed.

        Returns
        -------
        dtypes : dict
            A dictionary describing the default data types used for new NumPy
            arrays.

        See Also
        --------
        __array_namespace_info__.capabilities,
        __array_namespace_info__.default_device,
        __array_namespace_info__.dtypes,
        __array_namespace_info__.devices

        Examples
        --------
        >>> info = np.__array_namespace_info__()
        >>> info.default_dtypes()
        {'real floating': numpy.float64,
         'complex floating': numpy.complex128,
         'integral': numpy.int64,
         'indexing': numpy.int64}

        """
        if device not in ["cpu", None]:
            raise ValueError(
                'Device not understood. Only "cpu" is allowed, but received:'
                f' {device}'
            )
        return {
            "real floating": dtype(float64),
            "complex floating": dtype(complex128),
            "integral": dtype(intp),
            "indexing": dtype(intp),
        }

    def dtypes(self, *, device=None, kind=None):
        """
        The array API data types supported by NumPy.

        Note that this function only returns data types that are defined by
        the array API.

        Parameters
        ----------
        device : str, optional
            The device to get the data types for. For NumPy, only ``'cpu'`` is
            allowed.
        kind : str or tuple of str, optional
            The kind of data types to return. If ``None``, all data types are
            returned. If a string, only data types of that kind are returned.
            If a tuple, a dictionary containing the union of the given kinds
            is returned. The following kinds are supported:

            - ``'bool'``: boolean data types (i.e., ``bool``).
            - ``'signed integer'``: signed integer data types (i.e., ``int8``,
              ``int16``, ``int32``, ``int64``).
            - ``'unsigned integer'``: unsigned integer data types (i.e.,
              ``uint8``, ``uint16``, ``uint32``, ``uint64``).
            - ``'integral'``: integer data types. Shorthand for ``('signed
              integer', 'unsigned integer')``.
            - ``'real floating'``: real-valued floating-point data types
              (i.e., ``float32``, ``float64``).
            - ``'complex floating'``: complex floating-point data types (i.e.,
              ``complex64``, ``complex128``).
            - ``'numeric'``: numeric data types. Shorthand for ``('integral',
              'real floating', 'complex floating')``.

        Returns
        -------
        dtypes : dict
            A dictionary mapping the names of data types to the corresponding
            NumPy data types.

        See Also
        --------
        __array_namespace_info__.capabilities,
        __array_namespace_info__.default_device,
        __array_namespace_info__.default_dtypes,
        __array_namespace_info__.devices

        Examples
        --------
        >>> info = np.__array_namespace_info__()
        >>> info.dtypes(kind='signed integer')
        {'int8': numpy.int8,
         'int16': numpy.int16,
         'int32': numpy.int32,
         'int64': numpy.int64}

        """
        if device not in ["cpu", None]:
            raise ValueError(
                'Device not understood. Only "cpu" is allowed, but received:'
                f' {device}'
            )
        if kind is None:
            return {
                "bool": dtype(bool),
                "int8": dtype(int8),
                "int16": dtype(int16),
                "int32": dtype(int32),
                "int64": dtype(int64),
                "uint8": dtype(uint8),
                "uint16": dtype(uint16),
                "uint32": dtype(uint32),
                "uint64": dtype(uint64),
                "float32": dtype(float32),
                "float64": dtype(float64),
                "complex64": dtype(complex64),
                "complex128": dtype(complex128),
            }
        if kind == "bool":
            return {"bool": bool}
        if kind == "signed integer":
            return {
                "int8": dtype(int8),
                "int16": dtype(int16),
                "int32": dtype(int32),
                "int64": dtype(int64),
            }
        if kind == "unsigned integer":
            return {
                "uint8": dtype(uint8),
                "uint16": dtype(uint16),
                "uint32": dtype(uint32),
                "uint64": dtype(uint64),
            }
        if kind == "integral":
            return {
                "int8": dtype(int8),
                "int16": dtype(int16),
                "int32": dtype(int32),
                "int64": dtype(int64),
                "uint8": dtype(uint8),
                "uint16": dtype(uint16),
                "uint32": dtype(uint32),
                "uint64": dtype(uint64),
            }
        if kind == "real floating":
            return {
                "float32": dtype(float32),
                "float64": dtype(float64),
            }
        if kind == "complex floating":
            return {
                "complex64": dtype(complex64),
                "complex128": dtype(complex128),
            }
        if kind == "numeric":
            return {
                "int8": dtype(int8),
                "int16": dtype(int16),
                "int32": dtype(int32),
                "int64": dtype(int64),
                "uint8": dtype(uint8),
                "uint16": dtype(uint16),
                "uint32": dtype(uint32),
                "uint64": dtype(uint64),
                "float32": dtype(float32),
                "float64": dtype(float64),
                "complex64": dtype(complex64),
                "complex128": dtype(complex128),
            }
        if isinstance(kind, tuple):
            res = {}
            for k in kind:
                res.update(self.dtypes(kind=k))
            return res
        raise ValueError(f"unsupported kind: {kind!r}")

    def devices(self):
        """
        The devices supported by NumPy.

        For NumPy, this always returns ``['cpu']``.

        Returns
        -------
        devices : list of str
            The devices supported by NumPy.

        See Also
        --------
        __array_namespace_info__.capabilities,
        __array_namespace_info__.default_device,
        __array_namespace_info__.default_dtypes,
        __array_namespace_info__.dtypes

        Examples
        --------
        >>> info = np.__array_namespace_info__()
        >>> info.devices()
        ['cpu']

        """
        return ["cpu"]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\__init__.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""
ONNX Runtime is a performance-focused scoring engine for Open Neural Network Exchange (ONNX) models.
For more information on ONNX Runtime, please see `aka.ms/onnxruntime <https://aka.ms/onnxruntime/>`_
or the `Github project <https://github.com/microsoft/onnxruntime/>`_.
"""

__version__ = "1.22.0"
__author__ = "Microsoft"

# we need to do device version validation (for example to check Cuda version for an onnxruntime-training package).
# in order to know whether the onnxruntime package is for training it needs
# to do import onnxruntime.training.ortmodule first.
# onnxruntime.capi._pybind_state is required before import onnxruntime.training.ortmodule.
# however, import onnxruntime.capi._pybind_state will already raise an exception if a required Cuda version
# is not found.
# here we need to save the exception and continue with Cuda version validation in order to post
# meaningful messages to the user.
# the saved exception is raised after device version validation.
try:
    from onnxruntime.capi._pybind_state import (
        ExecutionMode,  # noqa: F401
        ExecutionOrder,  # noqa: F401
        GraphOptimizationLevel,  # noqa: F401
        LoraAdapter,  # noqa: F401
        ModelMetadata,  # noqa: F401
        NodeArg,  # noqa: F401
        OrtAllocatorType,  # noqa: F401
        OrtArenaCfg,  # noqa: F401
        OrtMemoryInfo,  # noqa: F401
        OrtMemType,  # noqa: F401
        OrtSparseFormat,  # noqa: F401
        RunOptions,  # noqa: F401
        SessionIOBinding,  # noqa: F401
        SessionOptions,  # noqa: F401
        create_and_register_allocator,  # noqa: F401
        create_and_register_allocator_v2,  # noqa: F401
        disable_telemetry_events,  # noqa: F401
        enable_telemetry_events,  # noqa: F401
        get_all_providers,  # noqa: F401
        get_available_providers,  # noqa: F401
        get_build_info,  # noqa: F401
        get_device,  # noqa: F401
        get_version_string,  # noqa: F401
        has_collective_ops,  # noqa: F401
        set_default_logger_severity,  # noqa: F401
        set_default_logger_verbosity,  # noqa: F401
        set_seed,  # noqa: F401
    )

    import_capi_exception = None
except Exception as e:
    import_capi_exception = e

from onnxruntime.capi import onnxruntime_validation

if import_capi_exception:
    raise import_capi_exception

from onnxruntime.capi.onnxruntime_inference_collection import (
    AdapterFormat,  # noqa: F401
    InferenceSession,  # noqa: F401
    IOBinding,  # noqa: F401
    OrtDevice,  # noqa: F401
    OrtValue,  # noqa: F401
    SparseTensor,  # noqa: F401
)

# TODO: thiagofc: Temporary experimental namespace for new PyTorch front-end
try:  # noqa: SIM105
    from . import experimental  # noqa: F401
except ImportError:
    pass


package_name, version, cuda_version = onnxruntime_validation.get_package_name_and_version_info()

if version:
    __version__ = version

onnxruntime_validation.check_distro_info()


def _get_package_version(package_name: str):
    from importlib.metadata import PackageNotFoundError, version

    try:
        package_version = version(package_name)
    except PackageNotFoundError:
        package_version = None
    return package_version


def _get_package_root(package_name: str, directory_name: str | None = None):
    from importlib.metadata import PackageNotFoundError, distribution

    root_directory_name = directory_name or package_name
    try:
        dist = distribution(package_name)
        files = dist.files or []

        for file in files:
            if file.name.endswith("__init__.py") and root_directory_name in file.parts:
                return file.locate().parent

        # Fallback to the first __init__.py
        if not directory_name:
            for file in files:
                if file.name.endswith("__init__.py"):
                    return file.locate().parent
    except PackageNotFoundError:
        # package not found, do nothing
        pass

    return None


def _get_nvidia_dll_paths(is_windows: bool, cuda: bool = True, cudnn: bool = True):
    if is_windows:
        # Path is relative to site-packages directory.
        cuda_dll_paths = [
            ("nvidia", "cublas", "bin", "cublasLt64_12.dll"),
            ("nvidia", "cublas", "bin", "cublas64_12.dll"),
            ("nvidia", "cufft", "bin", "cufft64_11.dll"),
            ("nvidia", "cuda_runtime", "bin", "cudart64_12.dll"),
        ]
        cudnn_dll_paths = [
            ("nvidia", "cudnn", "bin", "cudnn_engines_runtime_compiled64_9.dll"),
            ("nvidia", "cudnn", "bin", "cudnn_engines_precompiled64_9.dll"),
            ("nvidia", "cudnn", "bin", "cudnn_heuristic64_9.dll"),
            ("nvidia", "cudnn", "bin", "cudnn_ops64_9.dll"),
            ("nvidia", "cudnn", "bin", "cudnn_adv64_9.dll"),
            ("nvidia", "cudnn", "bin", "cudnn_graph64_9.dll"),
            ("nvidia", "cudnn", "bin", "cudnn64_9.dll"),
        ]
    else:  # Linux
        # cublas64 depends on cublasLt64, so cublasLt64 should be loaded first.
        cuda_dll_paths = [
            ("nvidia", "cublas", "lib", "libcublasLt.so.12"),
            ("nvidia", "cublas", "lib", "libcublas.so.12"),
            ("nvidia", "cuda_nvrtc", "lib", "libnvrtc.so.12"),
            ("nvidia", "curand", "lib", "libcurand.so.10"),
            ("nvidia", "cufft", "lib", "libcufft.so.11"),
            ("nvidia", "cuda_runtime", "lib", "libcudart.so.12"),
        ]

        # Do not load cudnn sub DLLs (they will be dynamically loaded later) to be consistent with PyTorch in Linux.
        cudnn_dll_paths = [
            ("nvidia", "cudnn", "lib", "libcudnn.so.9"),
        ]

    return (cuda_dll_paths if cuda else []) + (cudnn_dll_paths if cudnn else [])


def print_debug_info():
    """Print information to help debugging."""
    import importlib.util
    import os
    import platform
    from importlib.metadata import distributions

    print(f"{package_name} version: {__version__}")
    if cuda_version:
        print(f"CUDA version used in build: {cuda_version}")
    print("platform:", platform.platform())

    print("\nPython package, version and location:")
    ort_packages = []
    for dist in distributions():
        package = dist.metadata["Name"]
        if package == "onnxruntime" or package.startswith(("onnxruntime-", "ort-")):
            # Exclude packages whose root directory name is not onnxruntime.
            location = _get_package_root(package, "onnxruntime")
            if location and (package not in ort_packages):
                ort_packages.append(package)
                print(f"{package}=={dist.version} at {location}")

    if len(ort_packages) > 1:
        print(
            "\033[33mWARNING: multiple onnxruntime packages are installed to the same location. "
            "Please 'pip uninstall` all above packages, then `pip install` only one of them.\033[0m"
        )

    if cuda_version:
        # Print version of installed packages that is related to CUDA or cuDNN DLLs.
        packages = [
            "torch",
            "nvidia-cuda-runtime-cu12",
            "nvidia-cudnn-cu12",
            "nvidia-cublas-cu12",
            "nvidia-cufft-cu12",
            "nvidia-curand-cu12",
            "nvidia-cuda-nvrtc-cu12",
            "nvidia-nvjitlink-cu12",
        ]
        for package in packages:
            directory_name = "nvidia" if package.startswith("nvidia-") else None
            version = _get_package_version(package)
            if version:
                print(f"{package}=={version} at {_get_package_root(package, directory_name)}")
            else:
                print(f"{package} not installed")

    if platform.system() == "Windows":
        print(f"\nEnvironment variable:\nPATH={os.environ['PATH']}")
    elif platform.system() == "Linux":
        print(f"\nEnvironment variable:\nLD_LIBRARY_PATH={os.environ['LD_LIBRARY_PATH']}")

    if importlib.util.find_spec("psutil"):

        def is_target_dll(path: str):
            target_keywords = ["vcruntime140", "msvcp140"]
            if cuda_version:
                target_keywords = ["cufft", "cublas", "cudart", "nvrtc", "curand", "cudnn", *target_keywords]
            return any(keyword in path for keyword in target_keywords)

        import psutil

        p = psutil.Process(os.getpid())

        print("\nList of loaded DLLs:")
        for lib in p.memory_maps():
            if is_target_dll(lib.path.lower()):
                print(lib.path)

        if cuda_version:
            if importlib.util.find_spec("cpuinfo") and importlib.util.find_spec("py3nvml"):
                from .transformers.machine_info import get_device_info

                print("\nDevice information:")
                print(get_device_info())
            else:
                print("please `pip install py-cpuinfo py3nvml` to show device information.")
    else:
        print("please `pip install psutil` to show loaded DLLs.")


def preload_dlls(cuda: bool = True, cudnn: bool = True, msvc: bool = True, directory=None):
    """Preload CUDA 12.x and cuDNN 9.x DLLs in Windows or Linux, and MSVC runtime DLLs in Windows.

       When the installed PyTorch is compatible (using same major version of CUDA and cuDNN),
       there is no need to call this function if `import torch` is done before `import onnxruntime`.

    Args:
        cuda (bool, optional): enable loading CUDA DLLs. Defaults to True.
        cudnn (bool, optional): enable loading cuDNN DLLs. Defaults to True.
        msvc (bool, optional): enable loading MSVC DLLs in Windows. Defaults to True.
        directory(str, optional): a directory contains CUDA or cuDNN DLLs. It can be an absolute path,
           or a path relative to the directory of this file.
           If directory is None (default value), the search order: the lib directory of compatible PyTorch in Windows,
            nvidia site packages, default DLL loading paths.
           If directory is empty string (""), the search order: nvidia site packages, default DLL loading paths.
           If directory is a path, the search order: the directory, default DLL loading paths.
    """
    import ctypes
    import os
    import platform
    import sys

    if platform.system() not in ["Windows", "Linux"]:
        return

    is_windows = platform.system() == "Windows"
    if is_windows and msvc:
        try:
            ctypes.CDLL("vcruntime140.dll")
            ctypes.CDLL("msvcp140.dll")
            if platform.machine() != "ARM64":
                ctypes.CDLL("vcruntime140_1.dll")
        except OSError:
            print("Microsoft Visual C++ Redistributable is not installed, this may lead to the DLL load failure.")
            print("It can be downloaded at https://aka.ms/vs/17/release/vc_redist.x64.exe.")

    if not (cuda_version and cuda_version.startswith("12.")) and (cuda or cudnn):
        print(
            f"\033[33mWARNING: {package_name} is not built with CUDA 12.x support. "
            "Please install a version that supports CUDA 12.x, or call preload_dlls with cuda=False and cudnn=False.\033[0m"
        )
        return

    if not (cuda_version and cuda_version.startswith("12.") and (cuda or cudnn)):
        return

    is_cuda_cudnn_imported_by_torch = False

    if is_windows:
        torch_version = _get_package_version("torch")
        is_torch_for_cuda_12 = torch_version and "+cu12" in torch_version
        if "torch" in sys.modules:
            is_cuda_cudnn_imported_by_torch = is_torch_for_cuda_12
            if (torch_version and "+cu" in torch_version) and not is_torch_for_cuda_12:
                print(
                    f"\033[33mWARNING: The installed PyTorch {torch_version} does not support CUDA 12.x. "
                    f"Please install PyTorch for CUDA 12.x to be compatible with {package_name}.\033[0m"
                )

        if is_torch_for_cuda_12 and directory is None:
            torch_root = _get_package_root("torch", "torch")
            if torch_root:
                directory = os.path.join(torch_root, "lib")

    base_directory = directory or ".."
    if not os.path.isabs(base_directory):
        base_directory = os.path.join(os.path.dirname(__file__), base_directory)
    base_directory = os.path.normpath(base_directory)
    if not os.path.isdir(base_directory):
        raise RuntimeError(f"Invalid parameter of directory={directory}. The directory does not exist!")

    if is_cuda_cudnn_imported_by_torch:
        # In Windows, PyTorch has loaded CUDA and cuDNN DLLs during `import torch`, no need to load them again.
        print("Skip loading CUDA and cuDNN DLLs since torch is imported.")
        return

    # Try load DLLs from nvidia site packages.
    dll_paths = _get_nvidia_dll_paths(is_windows, cuda, cudnn)
    loaded_dlls = []
    for relative_path in dll_paths:
        dll_path = (
            os.path.join(base_directory, relative_path[-1])
            if directory
            else os.path.join(base_directory, *relative_path)
        )
        if os.path.isfile(dll_path):
            try:
                _ = ctypes.CDLL(dll_path)
                loaded_dlls.append(relative_path[-1])
            except Exception as e:
                print(f"Failed to load {dll_path}: {e}")

    # Try load DLLs with default path settings.
    has_failure = False
    for relative_path in dll_paths:
        dll_filename = relative_path[-1]
        if dll_filename not in loaded_dlls:
            try:
                _ = ctypes.CDLL(dll_filename)
            except Exception as e:
                has_failure = True
                print(f"Failed to load {dll_filename}: {e}")

    if has_failure:
        print("Please follow https://onnxruntime.ai/docs/install/#cuda-and-cudnn to install CUDA and CuDNN.")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\ext\indexable.py ===
# ext/indexable.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors

"""Define attributes on ORM-mapped classes that have "index" attributes for
columns with :class:`_types.Indexable` types.

"index" means the attribute is associated with an element of an
:class:`_types.Indexable` column with the predefined index to access it.
The :class:`_types.Indexable` types include types such as
:class:`_types.ARRAY`, :class:`_types.JSON` and
:class:`_postgresql.HSTORE`.



The :mod:`~sqlalchemy.ext.indexable` extension provides
:class:`_schema.Column`-like interface for any element of an
:class:`_types.Indexable` typed column. In simple cases, it can be
treated as a :class:`_schema.Column` - mapped attribute.

Synopsis
========

Given ``Person`` as a model with a primary key and JSON data field.
While this field may have any number of elements encoded within it,
we would like to refer to the element called ``name`` individually
as a dedicated attribute which behaves like a standalone column::

    from sqlalchemy import Column, JSON, Integer
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.ext.indexable import index_property

    Base = declarative_base()


    class Person(Base):
        __tablename__ = "person"

        id = Column(Integer, primary_key=True)
        data = Column(JSON)

        name = index_property("data", "name")

Above, the ``name`` attribute now behaves like a mapped column.   We
can compose a new ``Person`` and set the value of ``name``::

    >>> person = Person(name="Alchemist")

The value is now accessible::

    >>> person.name
    'Alchemist'

Behind the scenes, the JSON field was initialized to a new blank dictionary
and the field was set::

    >>> person.data
    {'name': 'Alchemist'}

The field is mutable in place::

    >>> person.name = "Renamed"
    >>> person.name
    'Renamed'
    >>> person.data
    {'name': 'Renamed'}

When using :class:`.index_property`, the change that we make to the indexable
structure is also automatically tracked as history; we no longer need
to use :class:`~.mutable.MutableDict` in order to track this change
for the unit of work.

Deletions work normally as well::

    >>> del person.name
    >>> person.data
    {}

Above, deletion of ``person.name`` deletes the value from the dictionary,
but not the dictionary itself.

A missing key will produce ``AttributeError``::

    >>> person = Person()
    >>> person.name
    AttributeError: 'name'

Unless you set a default value::

    >>> class Person(Base):
    ...     __tablename__ = "person"
    ...
    ...     id = Column(Integer, primary_key=True)
    ...     data = Column(JSON)
    ...
    ...     name = index_property("data", "name", default=None)  # See default

    >>> person = Person()
    >>> print(person.name)
    None


The attributes are also accessible at the class level.
Below, we illustrate ``Person.name`` used to generate
an indexed SQL criteria::

    >>> from sqlalchemy.orm import Session
    >>> session = Session()
    >>> query = session.query(Person).filter(Person.name == "Alchemist")

The above query is equivalent to::

    >>> query = session.query(Person).filter(Person.data["name"] == "Alchemist")

Multiple :class:`.index_property` objects can be chained to produce
multiple levels of indexing::

    from sqlalchemy import Column, JSON, Integer
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.ext.indexable import index_property

    Base = declarative_base()


    class Person(Base):
        __tablename__ = "person"

        id = Column(Integer, primary_key=True)
        data = Column(JSON)

        birthday = index_property("data", "birthday")
        year = index_property("birthday", "year")
        month = index_property("birthday", "month")
        day = index_property("birthday", "day")

Above, a query such as::

    q = session.query(Person).filter(Person.year == "1980")

On a PostgreSQL backend, the above query will render as:

.. sourcecode:: sql

    SELECT person.id, person.data
    FROM person
    WHERE person.data -> %(data_1)s -> %(param_1)s = %(param_2)s

Default Values
==============

:class:`.index_property` includes special behaviors for when the indexed
data structure does not exist, and a set operation is called:

* For an :class:`.index_property` that is given an integer index value,
  the default data structure will be a Python list of ``None`` values,
  at least as long as the index value; the value is then set at its
  place in the list.  This means for an index value of zero, the list
  will be initialized to ``[None]`` before setting the given value,
  and for an index value of five, the list will be initialized to
  ``[None, None, None, None, None]`` before setting the fifth element
  to the given value.   Note that an existing list is **not** extended
  in place to receive a value.

* for an :class:`.index_property` that is given any other kind of index
  value (e.g. strings usually), a Python dictionary is used as the
  default data structure.

* The default data structure can be set to any Python callable using the
  :paramref:`.index_property.datatype` parameter, overriding the previous
  rules.


Subclassing
===========

:class:`.index_property` can be subclassed, in particular for the common
use case of providing coercion of values or SQL expressions as they are
accessed.  Below is a common recipe for use with a PostgreSQL JSON type,
where we want to also include automatic casting plus ``astext()``::

    class pg_json_property(index_property):
        def __init__(self, attr_name, index, cast_type):
            super(pg_json_property, self).__init__(attr_name, index)
            self.cast_type = cast_type

        def expr(self, model):
            expr = super(pg_json_property, self).expr(model)
            return expr.astext.cast(self.cast_type)

The above subclass can be used with the PostgreSQL-specific
version of :class:`_postgresql.JSON`::

    from sqlalchemy import Column, Integer
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.dialects.postgresql import JSON

    Base = declarative_base()


    class Person(Base):
        __tablename__ = "person"

        id = Column(Integer, primary_key=True)
        data = Column(JSON)

        age = pg_json_property("data", "age", Integer)

The ``age`` attribute at the instance level works as before; however
when rendering SQL, PostgreSQL's ``->>`` operator will be used
for indexed access, instead of the usual index operator of ``->``::

    >>> query = session.query(Person).filter(Person.age < 20)

The above query will render:
.. sourcecode:: sql

    SELECT person.id, person.data
    FROM person
    WHERE CAST(person.data ->> %(data_1)s AS INTEGER) < %(param_1)s

"""  # noqa
from .. import inspect
from ..ext.hybrid import hybrid_property
from ..orm.attributes import flag_modified


__all__ = ["index_property"]


class index_property(hybrid_property):  # noqa
    """A property generator. The generated property describes an object
    attribute that corresponds to an :class:`_types.Indexable`
    column.

    .. seealso::

        :mod:`sqlalchemy.ext.indexable`

    """

    _NO_DEFAULT_ARGUMENT = object()

    def __init__(
        self,
        attr_name,
        index,
        default=_NO_DEFAULT_ARGUMENT,
        datatype=None,
        mutable=True,
        onebased=True,
    ):
        """Create a new :class:`.index_property`.

        :param attr_name:
            An attribute name of an `Indexable` typed column, or other
            attribute that returns an indexable structure.
        :param index:
            The index to be used for getting and setting this value.  This
            should be the Python-side index value for integers.
        :param default:
            A value which will be returned instead of `AttributeError`
            when there is not a value at given index.
        :param datatype: default datatype to use when the field is empty.
            By default, this is derived from the type of index used; a
            Python list for an integer index, or a Python dictionary for
            any other style of index.   For a list, the list will be
            initialized to a list of None values that is at least
            ``index`` elements long.
        :param mutable: if False, writes and deletes to the attribute will
            be disallowed.
        :param onebased: assume the SQL representation of this value is
            one-based; that is, the first index in SQL is 1, not zero.
        """

        if mutable:
            super().__init__(self.fget, self.fset, self.fdel, self.expr)
        else:
            super().__init__(self.fget, None, None, self.expr)
        self.attr_name = attr_name
        self.index = index
        self.default = default
        is_numeric = isinstance(index, int)
        onebased = is_numeric and onebased

        if datatype is not None:
            self.datatype = datatype
        else:
            if is_numeric:
                self.datatype = lambda: [None for x in range(index + 1)]
            else:
                self.datatype = dict
        self.onebased = onebased

    def _fget_default(self, err=None):
        if self.default == self._NO_DEFAULT_ARGUMENT:
            raise AttributeError(self.attr_name) from err
        else:
            return self.default

    def fget(self, instance):
        attr_name = self.attr_name
        column_value = getattr(instance, attr_name)
        if column_value is None:
            return self._fget_default()
        try:
            value = column_value[self.index]
        except (KeyError, IndexError) as err:
            return self._fget_default(err)
        else:
            return value

    def fset(self, instance, value):
        attr_name = self.attr_name
        column_value = getattr(instance, attr_name, None)
        if column_value is None:
            column_value = self.datatype()
            setattr(instance, attr_name, column_value)
        column_value[self.index] = value
        setattr(instance, attr_name, column_value)
        if attr_name in inspect(instance).mapper.attrs:
            flag_modified(instance, attr_name)

    def fdel(self, instance):
        attr_name = self.attr_name
        column_value = getattr(instance, attr_name)
        if column_value is None:
            raise AttributeError(self.attr_name)
        try:
            del column_value[self.index]
        except KeyError as err:
            raise AttributeError(self.attr_name) from err
        else:
            setattr(instance, attr_name, column_value)
            flag_modified(instance, attr_name)

    def expr(self, model):
        column = getattr(model, self.attr_name)
        index = self.index
        if self.onebased:
            index += 1
        return column[index]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\grover.py ===
"""Grover's algorithm and helper functions.

Todo:

* W gate construction (or perhaps -W gate based on Mermin's book)
* Generalize the algorithm for an unknown function that returns 1 on multiple
  qubit states, not just one.
* Implement _represent_ZGate in OracleGate
"""

from sympy.core.numbers import pi
from sympy.core.sympify import sympify
from sympy.core.basic import Atom
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import eye
from sympy.core.numbers import NegativeOne
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.qexpr import QuantumError
from sympy.physics.quantum.hilbert import ComplexSpace
from sympy.physics.quantum.operator import UnitaryOperator
from sympy.physics.quantum.gate import Gate
from sympy.physics.quantum.qubit import IntQubit

__all__ = [
    'OracleGate',
    'WGate',
    'superposition_basis',
    'grover_iteration',
    'apply_grover'
]


def superposition_basis(nqubits):
    """Creates an equal superposition of the computational basis.

    Parameters
    ==========

    nqubits : int
        The number of qubits.

    Returns
    =======

    state : Qubit
        An equal superposition of the computational basis with nqubits.

    Examples
    ========

    Create an equal superposition of 2 qubits::

        >>> from sympy.physics.quantum.grover import superposition_basis
        >>> superposition_basis(2)
        |0>/2 + |1>/2 + |2>/2 + |3>/2
    """

    amp = 1/sqrt(2**nqubits)
    return sum(amp*IntQubit(n, nqubits=nqubits) for n in range(2**nqubits))

class OracleGateFunction(Atom):
    """Wrapper for python functions used in `OracleGate`s"""

    def __new__(cls, function):
        if not callable(function):
            raise TypeError('Callable expected, got: %r' % function)
        obj = Atom.__new__(cls)
        obj.function = function
        return obj

    def _hashable_content(self):
        return type(self), self.function

    def __call__(self, *args):
        return self.function(*args)


class OracleGate(Gate):
    """A black box gate.

    The gate marks the desired qubits of an unknown function by flipping
    the sign of the qubits.  The unknown function returns true when it
    finds its desired qubits and false otherwise.

    Parameters
    ==========

    qubits : int
        Number of qubits.

    oracle : callable
        A callable function that returns a boolean on a computational basis.

    Examples
    ========

    Apply an Oracle gate that flips the sign of ``|2>`` on different qubits::

        >>> from sympy.physics.quantum.qubit import IntQubit
        >>> from sympy.physics.quantum.qapply import qapply
        >>> from sympy.physics.quantum.grover import OracleGate
        >>> f = lambda qubits: qubits == IntQubit(2)
        >>> v = OracleGate(2, f)
        >>> qapply(v*IntQubit(2))
        -|2>
        >>> qapply(v*IntQubit(3))
        |3>
    """

    gate_name = 'V'
    gate_name_latex = 'V'

    #-------------------------------------------------------------------------
    # Initialization/creation
    #-------------------------------------------------------------------------

    @classmethod
    def _eval_args(cls, args):
        if len(args) != 2:
            raise QuantumError(
                'Insufficient/excessive arguments to Oracle.  Please ' +
                'supply the number of qubits and an unknown function.'
            )
        sub_args = (args[0],)
        sub_args = UnitaryOperator._eval_args(sub_args)
        if not sub_args[0].is_Integer:
            raise TypeError('Integer expected, got: %r' % sub_args[0])

        function = args[1]
        if not isinstance(function, OracleGateFunction):
            function = OracleGateFunction(function)

        return (sub_args[0], function)

    @classmethod
    def _eval_hilbert_space(cls, args):
        """This returns the smallest possible Hilbert space."""
        return ComplexSpace(2)**args[0]

    #-------------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------------

    @property
    def search_function(self):
        """The unknown function that helps find the sought after qubits."""
        return self.label[1]

    @property
    def targets(self):
        """A tuple of target qubits."""
        return sympify(tuple(range(self.args[0])))

    #-------------------------------------------------------------------------
    # Apply
    #-------------------------------------------------------------------------

    def _apply_operator_Qubit(self, qubits, **options):
        """Apply this operator to a Qubit subclass.

        Parameters
        ==========

        qubits : Qubit
            The qubit subclass to apply this operator to.

        Returns
        =======

        state : Expr
            The resulting quantum state.
        """
        if qubits.nqubits != self.nqubits:
            raise QuantumError(
                'OracleGate operates on %r qubits, got: %r'
                % (self.nqubits, qubits.nqubits)
            )
        # If function returns 1 on qubits
            # return the negative of the qubits (flip the sign)
        if self.search_function(qubits):
            return -qubits
        else:
            return qubits

    #-------------------------------------------------------------------------
    # Represent
    #-------------------------------------------------------------------------

    def _represent_ZGate(self, basis, **options):
        """
        Represent the OracleGate in the computational basis.
        """
        nbasis = 2**self.nqubits  # compute it only once
        matrixOracle = eye(nbasis)
        # Flip the sign given the output of the oracle function
        for i in range(nbasis):
            if self.search_function(IntQubit(i, nqubits=self.nqubits)):
                matrixOracle[i, i] = NegativeOne()
        return matrixOracle


class WGate(Gate):
    """General n qubit W Gate in Grover's algorithm.

    The gate performs the operation ``2|phi><phi| - 1`` on some qubits.
    ``|phi> = (tensor product of n Hadamards)*(|0> with n qubits)``

    Parameters
    ==========

    nqubits : int
        The number of qubits to operate on

    """

    gate_name = 'W'
    gate_name_latex = 'W'

    @classmethod
    def _eval_args(cls, args):
        if len(args) != 1:
            raise QuantumError(
                'Insufficient/excessive arguments to W gate.  Please ' +
                'supply the number of qubits to operate on.'
            )
        args = UnitaryOperator._eval_args(args)
        if not args[0].is_Integer:
            raise TypeError('Integer expected, got: %r' % args[0])
        return args

    #-------------------------------------------------------------------------
    # Properties
    #-------------------------------------------------------------------------

    @property
    def targets(self):
        return sympify(tuple(reversed(range(self.args[0]))))

    #-------------------------------------------------------------------------
    # Apply
    #-------------------------------------------------------------------------

    def _apply_operator_Qubit(self, qubits, **options):
        """
        qubits: a set of qubits (Qubit)
        Returns: quantum object (quantum expression - QExpr)
        """
        if qubits.nqubits != self.nqubits:
            raise QuantumError(
                'WGate operates on %r qubits, got: %r'
                % (self.nqubits, qubits.nqubits)
            )

        # See 'Quantum Computer Science' by David Mermin p.92 -> W|a> result
        # Return (2/(sqrt(2^n)))|phi> - |a> where |a> is the current basis
        # state and phi is the superposition of basis states (see function
        # create_computational_basis above)
        basis_states = superposition_basis(self.nqubits)
        change_to_basis = (2/sqrt(2**self.nqubits))*basis_states
        return change_to_basis - qubits


def grover_iteration(qstate, oracle):
    """Applies one application of the Oracle and W Gate, WV.

    Parameters
    ==========

    qstate : Qubit
        A superposition of qubits.
    oracle : OracleGate
        The black box operator that flips the sign of the desired basis qubits.

    Returns
    =======

    Qubit : The qubits after applying the Oracle and W gate.

    Examples
    ========

    Perform one iteration of grover's algorithm to see a phase change::

        >>> from sympy.physics.quantum.qapply import qapply
        >>> from sympy.physics.quantum.qubit import IntQubit
        >>> from sympy.physics.quantum.grover import OracleGate
        >>> from sympy.physics.quantum.grover import superposition_basis
        >>> from sympy.physics.quantum.grover import grover_iteration
        >>> numqubits = 2
        >>> basis_states = superposition_basis(numqubits)
        >>> f = lambda qubits: qubits == IntQubit(2)
        >>> v = OracleGate(numqubits, f)
        >>> qapply(grover_iteration(basis_states, v))
        |2>

    """
    wgate = WGate(oracle.nqubits)
    return wgate*oracle*qstate


def apply_grover(oracle, nqubits, iterations=None):
    """Applies grover's algorithm.

    Parameters
    ==========

    oracle : callable
        The unknown callable function that returns true when applied to the
        desired qubits and false otherwise.

    Returns
    =======

    state : Expr
        The resulting state after Grover's algorithm has been iterated.

    Examples
    ========

    Apply grover's algorithm to an even superposition of 2 qubits::

        >>> from sympy.physics.quantum.qapply import qapply
        >>> from sympy.physics.quantum.qubit import IntQubit
        >>> from sympy.physics.quantum.grover import apply_grover
        >>> f = lambda qubits: qubits == IntQubit(2)
        >>> qapply(apply_grover(f, 2))
        |2>

    """
    if nqubits <= 0:
        raise QuantumError(
            'Grover\'s algorithm needs nqubits > 0, received %r qubits'
            % nqubits
        )
    if iterations is None:
        iterations = floor(sqrt(2**nqubits)*(pi/4))

    v = OracleGate(nqubits, oracle)
    iterated = superposition_basis(nqubits)
    for iter in range(iterations):
        iterated = grover_iteration(iterated, v)
        iterated = qapply(iterated)

    return iterated

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\orthopolys.py ===
"""Efficient functions for generating orthogonal polynomials."""
from sympy.core.symbol import Dummy
from sympy.polys.densearith import (dup_mul, dup_mul_ground,
    dup_lshift, dup_sub, dup_add, dup_sub_term, dup_sub_ground, dup_sqr)
from sympy.polys.domains import ZZ, QQ
from sympy.polys.polytools import named_poly
from sympy.utilities import public

def dup_jacobi(n, a, b, K):
    """Low-level implementation of Jacobi polynomials."""
    if n < 1:
        return [K.one]
    m2, m1 = [K.one], [(a+b)/K(2) + K.one, (a-b)/K(2)]
    for i in range(2, n+1):
        den = K(i)*(a + b + i)*(a + b + K(2)*i - K(2))
        f0 = (a + b + K(2)*i - K.one) * (a*a - b*b) / (K(2)*den)
        f1 = (a + b + K(2)*i - K.one) * (a + b + K(2)*i - K(2)) * (a + b + K(2)*i) / (K(2)*den)
        f2 = (a + i - K.one)*(b + i - K.one)*(a + b + K(2)*i) / den
        p0 = dup_mul_ground(m1, f0, K)
        p1 = dup_mul_ground(dup_lshift(m1, 1, K), f1, K)
        p2 = dup_mul_ground(m2, f2, K)
        m2, m1 = m1, dup_sub(dup_add(p0, p1, K), p2, K)
    return m1

@public
def jacobi_poly(n, a, b, x=None, polys=False):
    r"""Generates the Jacobi polynomial `P_n^{(a,b)}(x)`.

    Parameters
    ==========

    n : int
        Degree of the polynomial.
    a
        Lower limit of minimal domain for the list of coefficients.
    b
        Upper limit of minimal domain for the list of coefficients.
    x : optional
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.
    """
    return named_poly(n, dup_jacobi, None, "Jacobi polynomial", (x, a, b), polys)

def dup_gegenbauer(n, a, K):
    """Low-level implementation of Gegenbauer polynomials."""
    if n < 1:
        return [K.one]
    m2, m1 = [K.one], [K(2)*a, K.zero]
    for i in range(2, n+1):
        p1 = dup_mul_ground(dup_lshift(m1, 1, K), K(2)*(a-K.one)/K(i) + K(2), K)
        p2 = dup_mul_ground(m2, K(2)*(a-K.one)/K(i) + K.one, K)
        m2, m1 = m1, dup_sub(p1, p2, K)
    return m1

def gegenbauer_poly(n, a, x=None, polys=False):
    r"""Generates the Gegenbauer polynomial `C_n^{(a)}(x)`.

    Parameters
    ==========

    n : int
        Degree of the polynomial.
    x : optional
    a
        Decides minimal domain for the list of coefficients.
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.
    """
    return named_poly(n, dup_gegenbauer, None, "Gegenbauer polynomial", (x, a), polys)

def dup_chebyshevt(n, K):
    """Low-level implementation of Chebyshev polynomials of the first kind."""
    if n < 1:
        return [K.one]
    # When n is small, it is faster to directly calculate the recurrence relation.
    if n < 64: # The threshold serves as a heuristic
        return _dup_chebyshevt_rec(n, K)
    return _dup_chebyshevt_prod(n, K)

def _dup_chebyshevt_rec(n, K):
    r""" Chebyshev polynomials of the first kind using recurrence.

    Explanation
    ===========

    Chebyshev polynomials of the first kind are defined by the recurrence
    relation:

    .. math::
        T_0(x) &= 1\\
        T_1(x) &= x\\
        T_n(x) &= 2xT_{n-1}(x) - T_{n-2}(x)

    This function calculates the Chebyshev polynomial of the first kind using
    the above recurrence relation.

    Parameters
    ==========

    n : int
        n is a nonnegative integer.
    K : domain

    """
    m2, m1 = [K.one], [K.one, K.zero]
    for _ in range(n - 1):
        m2, m1 = m1, dup_sub(dup_mul_ground(dup_lshift(m1, 1, K), K(2), K), m2, K)
    return m1

def _dup_chebyshevt_prod(n, K):
    r""" Chebyshev polynomials of the first kind using recursive products.

    Explanation
    ===========

    Computes Chebyshev polynomials of the first kind using

    .. math::
        T_{2n}(x) &= 2T_n^2(x) - 1\\
        T_{2n+1}(x) &= 2T_{n+1}(x)T_n(x) - x

    This is faster than ``_dup_chebyshevt_rec`` for large ``n``.

    Parameters
    ==========

    n : int
        n is a nonnegative integer.
    K : domain

    """
    m2, m1 = [K.one, K.zero], [K(2), K.zero, -K.one]
    for i in bin(n)[3:]:
        c = dup_sub_term(dup_mul_ground(dup_mul(m1, m2, K), K(2), K), K.one, 1, K)
        if  i  == '1':
            m2, m1 = c, dup_sub_ground(dup_mul_ground(dup_sqr(m1, K), K(2), K), K.one, K)
        else:
            m2, m1 = dup_sub_ground(dup_mul_ground(dup_sqr(m2, K), K(2), K), K.one, K), c
    return m2

def dup_chebyshevu(n, K):
    """Low-level implementation of Chebyshev polynomials of the second kind."""
    if n < 1:
        return [K.one]
    m2, m1 = [K.one], [K(2), K.zero]
    for i in range(2, n+1):
        m2, m1 = m1, dup_sub(dup_mul_ground(dup_lshift(m1, 1, K), K(2), K), m2, K)
    return m1

@public
def chebyshevt_poly(n, x=None, polys=False):
    r"""Generates the Chebyshev polynomial of the first kind `T_n(x)`.

    Parameters
    ==========

    n : int
        Degree of the polynomial.
    x : optional
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.
    """
    return named_poly(n, dup_chebyshevt, ZZ,
            "Chebyshev polynomial of the first kind", (x,), polys)

@public
def chebyshevu_poly(n, x=None, polys=False):
    r"""Generates the Chebyshev polynomial of the second kind `U_n(x)`.

    Parameters
    ==========

    n : int
        Degree of the polynomial.
    x : optional
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.
    """
    return named_poly(n, dup_chebyshevu, ZZ,
            "Chebyshev polynomial of the second kind", (x,), polys)

def dup_hermite(n, K):
    """Low-level implementation of Hermite polynomials."""
    if n < 1:
        return [K.one]
    m2, m1 = [K.one], [K(2), K.zero]
    for i in range(2, n+1):
        a = dup_lshift(m1, 1, K)
        b = dup_mul_ground(m2, K(i-1), K)
        m2, m1 = m1, dup_mul_ground(dup_sub(a, b, K), K(2), K)
    return m1

def dup_hermite_prob(n, K):
    """Low-level implementation of probabilist's Hermite polynomials."""
    if n < 1:
        return [K.one]
    m2, m1 = [K.one], [K.one, K.zero]
    for i in range(2, n+1):
        a = dup_lshift(m1, 1, K)
        b = dup_mul_ground(m2, K(i-1), K)
        m2, m1 = m1, dup_sub(a, b, K)
    return m1

@public
def hermite_poly(n, x=None, polys=False):
    r"""Generates the Hermite polynomial `H_n(x)`.

    Parameters
    ==========

    n : int
        Degree of the polynomial.
    x : optional
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.
    """
    return named_poly(n, dup_hermite, ZZ, "Hermite polynomial", (x,), polys)

@public
def hermite_prob_poly(n, x=None, polys=False):
    r"""Generates the probabilist's Hermite polynomial `He_n(x)`.

    Parameters
    ==========

    n : int
        Degree of the polynomial.
    x : optional
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.
    """
    return named_poly(n, dup_hermite_prob, ZZ,
            "probabilist's Hermite polynomial", (x,), polys)

def dup_legendre(n, K):
    """Low-level implementation of Legendre polynomials."""
    if n < 1:
        return [K.one]
    m2, m1 = [K.one], [K.one, K.zero]
    for i in range(2, n+1):
        a = dup_mul_ground(dup_lshift(m1, 1, K), K(2*i-1, i), K)
        b = dup_mul_ground(m2, K(i-1, i), K)
        m2, m1 = m1, dup_sub(a, b, K)
    return m1

@public
def legendre_poly(n, x=None, polys=False):
    r"""Generates the Legendre polynomial `P_n(x)`.

    Parameters
    ==========

    n : int
        Degree of the polynomial.
    x : optional
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.
    """
    return named_poly(n, dup_legendre, QQ, "Legendre polynomial", (x,), polys)

def dup_laguerre(n, alpha, K):
    """Low-level implementation of Laguerre polynomials."""
    m2, m1 = [K.zero], [K.one]
    for i in range(1, n+1):
        a = dup_mul(m1, [-K.one/K(i), (alpha-K.one)/K(i) + K(2)], K)
        b = dup_mul_ground(m2, (alpha-K.one)/K(i) + K.one, K)
        m2, m1 = m1, dup_sub(a, b, K)
    return m1

@public
def laguerre_poly(n, x=None, alpha=0, polys=False):
    r"""Generates the Laguerre polynomial `L_n^{(\alpha)}(x)`.

    Parameters
    ==========

    n : int
        Degree of the polynomial.
    x : optional
    alpha : optional
        Decides minimal domain for the list of coefficients.
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.
    """
    return named_poly(n, dup_laguerre, None, "Laguerre polynomial", (x, alpha), polys)

def dup_spherical_bessel_fn(n, K):
    """Low-level implementation of fn(n, x)."""
    if n < 1:
        return [K.one, K.zero]
    m2, m1 = [K.one], [K.one, K.zero]
    for i in range(2, n+1):
        m2, m1 = m1, dup_sub(dup_mul_ground(dup_lshift(m1, 1, K), K(2*i-1), K), m2, K)
    return dup_lshift(m1, 1, K)

def dup_spherical_bessel_fn_minus(n, K):
    """Low-level implementation of fn(-n, x)."""
    m2, m1 = [K.one, K.zero], [K.zero]
    for i in range(2, n+1):
        m2, m1 = m1, dup_sub(dup_mul_ground(dup_lshift(m1, 1, K), K(3-2*i), K), m2, K)
    return m1

def spherical_bessel_fn(n, x=None, polys=False):
    """
    Coefficients for the spherical Bessel functions.

    These are only needed in the jn() function.

    The coefficients are calculated from:

    fn(0, z) = 1/z
    fn(1, z) = 1/z**2
    fn(n-1, z) + fn(n+1, z) == (2*n+1)/z * fn(n, z)

    Parameters
    ==========

    n : int
        Degree of the polynomial.
    x : optional
    polys : bool, optional
        If True, return a Poly, otherwise (default) return an expression.

    Examples
    ========

    >>> from sympy.polys.orthopolys import spherical_bessel_fn as fn
    >>> from sympy import Symbol
    >>> z = Symbol("z")
    >>> fn(1, z)
    z**(-2)
    >>> fn(2, z)
    -1/z + 3/z**3
    >>> fn(3, z)
    -6/z**2 + 15/z**4
    >>> fn(4, z)
    1/z - 45/z**3 + 105/z**5

    """
    if x is None:
        x = Dummy("x")
    f = dup_spherical_bessel_fn_minus if n < 0 else dup_spherical_bessel_fn
    return named_poly(abs(n), f, ZZ, "", (QQ(1)/x,), polys)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\isympy.py ===
"""
Python shell for SymPy.

This is just a normal Python shell (IPython shell if you have the
IPython package installed), that executes the following commands for
the user:

    >>> from __future__ import division
    >>> from sympy import *
    >>> x, y, z, t = symbols('x y z t')
    >>> k, m, n = symbols('k m n', integer=True)
    >>> f, g, h = symbols('f g h', cls=Function)
    >>> init_printing()

So starting 'isympy' is equivalent to starting Python (or IPython) and
executing the above commands by hand.  It is intended for easy and quick
experimentation with SymPy.  isympy is a good way to use SymPy as an
interactive calculator. If you have IPython and Matplotlib installed, then
interactive plotting is enabled by default.

COMMAND LINE OPTIONS
--------------------

-c CONSOLE, --console=CONSOLE

     Use the specified shell (Python or IPython) shell as the console
     backend instead of the default one (IPython if present, Python
     otherwise), e.g.:

        $isympy -c python

    CONSOLE must be one of 'ipython' or 'python'

-p PRETTY, --pretty PRETTY

    Setup pretty-printing in SymPy. When pretty-printing is enabled,
    expressions can be printed with Unicode or ASCII. The default is
    to use pretty-printing (with Unicode if the terminal supports it).
    When this option is 'no', expressions will not be pretty-printed
    and ASCII will be used:

        $isympy -p no

    PRETTY must be one of 'unicode', 'ascii', or 'no'

-t TYPES, --types=TYPES

    Setup the ground types for the polys.  By default, gmpy ground types
    are used if gmpy2 or gmpy is installed, otherwise it falls back to python
    ground types, which are a little bit slower.  You can manually
    choose python ground types even if gmpy is installed (e.g., for
    testing purposes):

        $isympy -t python

    TYPES must be one of 'gmpy', 'gmpy1' or 'python'

    Note that the ground type gmpy1 is primarily intended for testing; it
    forces the use of gmpy version 1 even if gmpy2 is available.

    This is the same as setting the environment variable
    SYMPY_GROUND_TYPES to the given ground type (e.g.,
    SYMPY_GROUND_TYPES='gmpy')

    The ground types can be determined interactively from the variable
    sympy.polys.domains.GROUND_TYPES.

-o ORDER, --order ORDER

    Setup the ordering of terms for printing.  The default is lex, which
    orders terms lexicographically (e.g., x**2 + x + 1). You can choose
    other orderings, such as rev-lex, which will use reverse
    lexicographic ordering (e.g., 1 + x + x**2):

        $isympy -o rev-lex

    ORDER must be one of 'lex', 'rev-lex', 'grlex', 'rev-grlex',
    'grevlex', 'rev-grevlex', 'old', or 'none'.

    Note that for very large expressions, ORDER='none' may speed up
    printing considerably but the terms will have no canonical order.

-q, --quiet

    Print only Python's and SymPy's versions to stdout at startup.

-d, --doctest

    Use the same format that should be used for doctests.  This is
    equivalent to -c python -p no.

-C, --no-cache

    Disable the caching mechanism.  Disabling the cache may slow certain
    operations down considerably.  This is useful for testing the cache,
    or for benchmarking, as the cache can result in deceptive timings.

    This is equivalent to setting the environment variable
    SYMPY_USE_CACHE to 'no'.

-a, --auto-symbols (requires at least IPython 0.11)

    Automatically create missing symbols.  Normally, typing a name of a
    Symbol that has not been instantiated first would raise NameError,
    but with this option enabled, any undefined name will be
    automatically created as a Symbol.

    Note that this is intended only for interactive, calculator style
    usage. In a script that uses SymPy, Symbols should be instantiated
    at the top, so that it's clear what they are.

    This will not override any names that are already defined, which
    includes the single character letters represented by the mnemonic
    QCOSINE (see the "Gotchas and Pitfalls" document in the
    documentation). You can delete existing names by executing "del
    name".  If a name is defined, typing "'name' in dir()" will return True.

    The Symbols that are created using this have default assumptions.
    If you want to place assumptions on symbols, you should create them
    using symbols() or var().

    Finally, this only works in the top level namespace. So, for
    example, if you define a function in isympy with an undefined
    Symbol, it will not work.

    See also the -i and -I options.

-i, --int-to-Integer (requires at least IPython 0.11)

    Automatically wrap int literals with Integer.  This makes it so that
    things like 1/2 will come out as Rational(1, 2), rather than 0.5.  This
    works by preprocessing the source and wrapping all int literals with
    Integer.  Note that this will not change the behavior of int literals
    assigned to variables, and it also won't change the behavior of functions
    that return int literals.

    If you want an int, you can wrap the literal in int(), e.g. int(3)/int(2)
    gives 1.5 (with division imported from __future__).

-I, --interactive (requires at least IPython 0.11)

    This is equivalent to --auto-symbols --int-to-Integer.  Future options
    designed for ease of interactive use may be added to this.

-D, --debug

    Enable debugging output.  This is the same as setting the
    environment variable SYMPY_DEBUG to 'True'.  The debug status is set
    in the variable SYMPY_DEBUG within isympy.

-- IPython options

    Additionally you can pass command line options directly to the IPython
    interpreter (the standard Python shell is not supported).  However you
    need to add the '--' separator between two types of options, e.g the
    startup banner option and the colors option. You need to enter the
    options as required by the version of IPython that you are using, too:

    in IPython 0.11,

        $isympy -q -- --colors=NoColor

    or older versions of IPython,

        $isympy -q -- -colors NoColor

See also isympy --help.
"""

import os
import sys

# DO NOT IMPORT SYMPY HERE! Or the setting of the sympy environment variables
# by the command line will break.

def main() -> None:
    from argparse import ArgumentParser, RawDescriptionHelpFormatter

    VERSION = None
    if '--version' in sys.argv:
        # We cannot import sympy before this is run, because flags like -C and
        # -t set environment variables that must be set before SymPy is
        # imported. The only thing we need to import it for is to get the
        # version, which only matters with the --version flag.
        import sympy
        VERSION = sympy.__version__

    usage = 'isympy [options] -- [ipython options]'
    parser = ArgumentParser(
        usage=usage,
        description=__doc__,
        formatter_class=RawDescriptionHelpFormatter,
    )

    parser.add_argument('--version', action='version', version=VERSION)

    parser.add_argument(
        '-c', '--console',
        dest='console',
        action='store',
        default=None,
        choices=['ipython', 'python'],
        metavar='CONSOLE',
        help='select type of interactive session: ipython | python; defaults '
        'to ipython if IPython is installed, otherwise python')

    parser.add_argument(
        '-p', '--pretty',
        dest='pretty',
        action='store',
        default=None,
        metavar='PRETTY',
        choices=['unicode', 'ascii', 'no'],
        help='setup pretty printing: unicode | ascii | no; defaults to '
        'unicode printing if the terminal supports it, otherwise ascii')

    parser.add_argument(
        '-t', '--types',
        dest='types',
        action='store',
        default=None,
        metavar='TYPES',
        choices=['gmpy', 'gmpy1', 'python'],
        help='setup ground types: gmpy | gmpy1 | python; defaults to gmpy if gmpy2 '
        'or gmpy is installed, otherwise python')

    parser.add_argument(
        '-o', '--order',
        dest='order',
        action='store',
        default=None,
        metavar='ORDER',
        choices=['lex', 'grlex', 'grevlex', 'rev-lex', 'rev-grlex', 'rev-grevlex', 'old', 'none'],
        help='setup ordering of terms: [rev-]lex | [rev-]grlex | [rev-]grevlex | old | none; defaults to lex')

    parser.add_argument(
        '-q', '--quiet',
        dest='quiet',
        action='store_true',
        default=False,
        help='print only version information at startup')

    parser.add_argument(
        '-d', '--doctest',
        dest='doctest',
        action='store_true',
        default=False,
        help='use the doctest format for output (you can just copy and paste it)')

    parser.add_argument(
        '-C', '--no-cache',
        dest='cache',
        action='store_false',
        default=True,
        help='disable caching mechanism')

    parser.add_argument(
        '-a', '--auto-symbols',
        dest='auto_symbols',
        action='store_true',
        default=False,
        help='automatically construct missing symbols')

    parser.add_argument(
        '-i', '--int-to-Integer',
        dest='auto_int_to_Integer',
        action='store_true',
        default=False,
        help="automatically wrap int literals with Integer")

    parser.add_argument(
        '-I', '--interactive',
        dest='interactive',
        action='store_true',
        default=False,
        help="equivalent to -a -i")

    parser.add_argument(
        '-D', '--debug',
        dest='debug',
        action='store_true',
        default=False,
        help='enable debugging output')

    (options, ipy_args) = parser.parse_known_args()
    if '--' in ipy_args:
        ipy_args.remove('--')

    if not options.cache:
        os.environ['SYMPY_USE_CACHE'] = 'no'

    if options.types:
        os.environ['SYMPY_GROUND_TYPES'] = options.types

    if options.debug:
        os.environ['SYMPY_DEBUG'] = str(options.debug)

    if options.doctest:
        options.pretty = 'no'
        options.console = 'python'

    session = options.console

    if session is not None:
        ipython = session == 'ipython'
    else:
        try:
            import IPython # noqa: F401
            ipython = True
        except ImportError:
            if not options.quiet:
                from sympy.interactive.session import no_ipython
                print(no_ipython)
            ipython = False

    args = {
        'pretty_print': True,
        'use_unicode':  None,
        'use_latex':    None,
        'order':        None,
        'argv':         ipy_args,
    }

    if options.pretty == 'unicode':
        args['use_unicode'] = True
    elif options.pretty == 'ascii':
        args['use_unicode'] = False
    elif options.pretty == 'no':
        args['pretty_print'] = False

    if options.order is not None:
        args['order'] = options.order

    args['quiet'] = options.quiet
    args['auto_symbols'] = options.auto_symbols or options.interactive
    args['auto_int_to_Integer'] = options.auto_int_to_Integer or options.interactive

    from sympy.interactive import init_session
    init_session(ipython, **args)

if __name__ == "__main__":
    main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\llama\llama_parity.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from __future__ import annotations

import argparse
import logging
import os
import time

import numpy as np
import packaging.version as pv
import torch
import transformers
from benchmark_helper import setup_logger
from dist_settings import get_rank, get_size
from llama_inputs import (
    add_io_bindings_as_ortvalues,
    convert_inputs_for_ort,
    get_merged_sample_with_past_kv_inputs,
    get_sample_inputs,
    get_sample_with_past_kv_inputs,
    verify_ort_inputs,
)
from llama_torch import setup_torch_model
from models.torch_export_patches.cache_helper import make_dynamic_cache
from transformers import AutoConfig
from transformers.cache_utils import DynamicCache

import onnxruntime as ort

logger = logging.getLogger("")


def get_sequence_lengths(args: argparse.Namespace, config: AutoConfig):
    past_sequence_length, curr_sequence_length = (8, 1) if args.use_past_kv else (0, 8)
    max_sequence_length = config.max_position_embeddings
    return past_sequence_length, curr_sequence_length, max_sequence_length


def get_inputs(args: argparse.Namespace, config: AutoConfig):
    # Dummy values for parity
    world_size = get_size()
    batch_size = 2
    past_sequence_length, sequence_length, max_sequence_length = get_sequence_lengths(args, config)

    if args.merged:
        inputs = get_merged_sample_with_past_kv_inputs(
            config,
            args.device,
            batch_size,
            seq_len=sequence_length,
            past_seq_len=past_sequence_length,
            max_seq_len=max_sequence_length,
            use_fp16=args.use_fp16,
            use_buffer_share=args.use_buffer_share,
            return_dict=True,
            world_size=world_size,
        )
    elif args.use_past_kv:
        inputs = get_sample_with_past_kv_inputs(
            config,
            args.device,
            batch_size,
            sequence_length,
            use_fp16=args.use_fp16,
            return_dict=True,
            world_size=world_size,
        )
    else:
        inputs = get_sample_inputs(config, args.device, batch_size, sequence_length, return_dict=True)

    return inputs


def torch_deepcopy(value):
    if isinstance(value, (int, float, str)):
        return value
    if isinstance(value, tuple):
        return tuple(torch_deepcopy(v) for v in value)
    if isinstance(value, list):
        return [torch_deepcopy(v) for v in value]
    if isinstance(value, set):
        return {torch_deepcopy(v) for v in value}
    if isinstance(value, dict):
        return {k: torch_deepcopy(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value.copy()
    if hasattr(value, "clone"):
        return value.clone()
    if isinstance(value, DynamicCache):
        return make_dynamic_cache(torch_deepcopy(list(zip(value.key_cache, value.value_cache, strict=False))))
    # We should have a code using serialization, deserialization assuming a model
    # cannot be exported without them.
    raise NotImplementedError(f"torch_deepcopy not implemented for type {type(value)}")


def verify_parity(
    args: argparse.Namespace,
    location: str,
    use_auth_token: bool,
    kv_cache_ortvalues: dict,
    pytorch_model: None | torch.nn.Module = None,
    config: None | AutoConfig = None,
):
    # If it's running in a machine which GPU memory < 36GB, it should unload the llama in GPU in time and free the GPU memory for ORT.
    py_model = pytorch_model
    if py_model is None:
        config, py_model = setup_torch_model(
            args,
            location,
            use_auth_token,
            torch_dtype=(torch.float16 if args.use_fp16 else torch.float32),
            device=args.device,
        )

    inputs = get_inputs(args, config)

    if "past_key_values" in inputs and pv.Version(transformers.__version__) >= pv.Version("4.45"):
        # Using DynamicCache
        inputs["past_key_values"] = make_dynamic_cache(inputs["past_key_values"])

    # Run inference with PyTorch
    if args.execution_provider != "cpu":
        torch.cuda.synchronize()
    start_time = time.time()
    # If there is a cache in the inputs, we need to make a copy as the model modify them inplace.
    # DynamicCache inherits from torch.nn.Module in some version of transformers.
    # We need to make the copy manually.
    pt_outputs = py_model(**torch_deepcopy(inputs)).logits.detach().cpu().numpy()
    if args.execution_provider != "cpu":
        torch.cuda.synchronize()
    end_time = time.time()
    logger.info(f"PyTorch took {end_time - start_time} s")

    if args.small_gpu and py_model is not None:
        del py_model
        torch.cuda.empty_cache()

    # Run inference with ORT
    past_sequence_length, _, max_sequence_length = get_sequence_lengths(args, config)
    inputs = convert_inputs_for_ort(
        inputs,
        use_buffer_share=args.use_buffer_share,
        past_seq_len=past_sequence_length,
        max_seq_len=max_sequence_length,
    )

    ep = f"{args.execution_provider.upper()}ExecutionProvider"
    if ep == "CUDAExecutionProvider":
        ep = (ep, {"device_id": args.rank})
    ort_model = ort.InferenceSession(
        args.onnx_model_path,
        sess_options=ort.SessionOptions(),
        providers=[ep],
    )
    inputs = verify_ort_inputs(ort_model, inputs)

    # Add IO bindings for non-CPU execution providers
    if args.execution_provider != "cpu":
        io_binding, kv_cache_ortvalues = add_io_bindings_as_ortvalues(
            ort_model,
            ort_inputs=inputs,
            device=args.execution_provider,
            device_id=int(args.rank),
            use_buffer_share=args.use_buffer_share,
            kv_cache_ortvalues=kv_cache_ortvalues,
        )

        io_binding.synchronize_inputs()
        start_time = time.time()
        ort_model.run_with_iobinding(io_binding)
        io_binding.synchronize_outputs()
        end_time = time.time()

        ort_outputs = io_binding.copy_outputs_to_cpu()[0]  # Get logits
        del ort_model

    else:
        start_time = time.time()
        ort_outputs = ort_model.run(None, inputs)
        end_time = time.time()

        ort_outputs = ort_outputs[0]  # Get logits

    logger.info(f"ONNX Runtime took {end_time - start_time} s")

    # Compare PyTorch and ONNX Runtime accuracy
    tol = 2e1 if "int4" in args.onnx_model_path or "int8" in args.onnx_model_path else 5e-1
    parity = np.allclose(pt_outputs, ort_outputs, rtol=tol, atol=tol)
    logger.warning(f"Are PyTorch and ONNX Runtime results close? {parity}")
    if not parity:
        logger.warning(f"Max diff: {np.max(pt_outputs - ort_outputs)}")
    return kv_cache_ortvalues


def get_args(argv: list[str]):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model_name",
        required=False,
        help="Model name in Hugging Face",
    )

    parser.add_argument(
        "-t",
        "--torch_model_directory",
        required=False,
        default=os.path.join("."),
        help="Path to folder containing PyTorch model and associated files if saved on disk",
    )

    parser.add_argument(
        "-o",
        "--onnx_model_path",
        required=True,
        default=os.path.join("."),
        help="Path to ONNX model (with external data files saved in the same folder as the model)",
    )

    parser.add_argument(
        "-ep",
        "--execution_provider",
        required=False,
        default="cpu",
        choices=["cpu", "cuda", "rocm"],
        help="Execution provider to verify parity with",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print verbose logs",
    )
    parser.set_defaults(verbose=False)

    parser.add_argument(
        "-p",
        "--use_past_kv",
        action="store_true",
        help="Use past key and past value as inputs to the model. Necessary for decoder_with_past_model.onnx models.",
    )
    parser.set_defaults(use_past_kv=False)

    parser.add_argument(
        "-g",
        "--use_buffer_share",
        action="store_true",
        help="Use if model has GroupQueryAttention and you want to enable past-present buffer sharing",
    )
    parser.set_defaults(use_buffer_share=False)

    parser.add_argument(
        "--merged",
        action="store_true",
        help="Use merged model (i.e. decoder_merged_model.onnx).",
    )
    parser.set_defaults(merged=False)

    parser.add_argument(
        "-fp",
        "--precision",
        required=True,
        choices=["int4", "int8", "fp16", "fp32"],
        help="Precision of model",
    )

    parser.add_argument(
        "--cache_dir",
        required=False,
        type=str,
        default="./model_cache",
        help="model cache dir to override default HF cache dir to avoid overflood the /home dir",
    )

    # The argument is used for CI mainly, because the CI machine has 24G GPU memory at most.
    parser.add_argument(
        "--small_gpu",
        action="store_true",
        help="Load the llama in GPU every time for parity_check if it's running in a machine which GPU memory < 36GB. ",
    )

    args = parser.parse_args() if argv == [] else parser.parse_args(argv)

    # Use FP32 precision for FP32, INT8, INT4 CPU models, use FP16 precision for FP16 and INT4 GPU models
    args.precision = (
        "fp32"
        if args.precision in {"int8", "fp32"} or (args.precision == "int4" and args.execution_provider == "cpu")
        else "fp16"
    )
    return args


def main(argv: list[str] = []):  # noqa: B006
    args = get_args(argv)
    setup_logger(args.verbose)
    logger.info(f"Arguments: {args}")
    rank = get_rank()

    # Load model and config
    setattr(args, "use_fp16", args.precision == "fp16")  # noqa: B010
    args.rank = rank
    setattr(args, "device_name", "cpu" if args.execution_provider == "cpu" else f"cuda:{rank}")  # noqa: B010
    setattr(args, "device", torch.device(args.device_name))  # noqa: B010
    use_auth_token = args.torch_model_directory == os.path.join(".")
    location = args.model_name if use_auth_token else args.torch_model_directory

    kv_cache_ortvalues = {}
    if not args.merged:
        verify_parity(args, location, use_auth_token, kv_cache_ortvalues)
    else:
        config = llama = None
        if not args.small_gpu:
            config, llama = setup_torch_model(
                args,
                location,
                use_auth_token,
                torch_dtype=(torch.float16 if args.use_fp16 else torch.float32),
                device=args.device,
            )

        # Verify prompt processing in merged model (decoder_model.onnx)
        args.use_past_kv = False
        kv_cache_ortvalues = verify_parity(
            args, location, use_auth_token, kv_cache_ortvalues, pytorch_model=llama, config=config
        )

        # Verify token generation in merged model (decoder_with_past_model.onnx)
        args.use_past_kv = True
        verify_parity(args, location, use_auth_token, kv_cache_ortvalues, pytorch_model=llama, config=config)


if __name__ == "__main__":
    seed = 2
    np.random.seed(seed)
    torch.manual_seed(seed)
    main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\external\gmpy.py ===
from __future__ import annotations
import os
from ctypes import c_long, sizeof
from functools import reduce
from typing import Type
from warnings import warn

from sympy.external import import_module

from .pythonmpq import PythonMPQ

from .ntheory import (
    bit_scan1 as python_bit_scan1,
    bit_scan0 as python_bit_scan0,
    remove as python_remove,
    factorial as python_factorial,
    sqrt as python_sqrt,
    sqrtrem as python_sqrtrem,
    gcd as python_gcd,
    lcm as python_lcm,
    gcdext as python_gcdext,
    is_square as python_is_square,
    invert as python_invert,
    legendre as python_legendre,
    jacobi as python_jacobi,
    kronecker as python_kronecker,
    iroot as python_iroot,
    is_fermat_prp as python_is_fermat_prp,
    is_euler_prp as python_is_euler_prp,
    is_strong_prp as python_is_strong_prp,
    is_fibonacci_prp as python_is_fibonacci_prp,
    is_lucas_prp as python_is_lucas_prp,
    is_selfridge_prp as python_is_selfridge_prp,
    is_strong_lucas_prp as python_is_strong_lucas_prp,
    is_strong_selfridge_prp as python_is_strong_selfridge_prp,
    is_bpsw_prp as python_is_bpsw_prp,
    is_strong_bpsw_prp as python_is_strong_bpsw_prp,
)


__all__ = [
    # GROUND_TYPES is either 'gmpy' or 'python' depending on which is used. If
    # gmpy is installed then it will be used unless the environment variable
    # SYMPY_GROUND_TYPES is set to something other than 'auto', 'gmpy', or
    # 'gmpy2'.
    'GROUND_TYPES',

    # If HAS_GMPY is 0, no supported version of gmpy is available. Otherwise,
    # HAS_GMPY will be 2 for gmpy2 if GROUND_TYPES is 'gmpy'. It used to be
    # possible for HAS_GMPY to be 1 for gmpy but gmpy is no longer supported.
    'HAS_GMPY',

    # SYMPY_INTS is a tuple containing the base types for valid integer types.
    # This is either (int,) or (int, type(mpz(0))) depending on GROUND_TYPES.
    'SYMPY_INTS',

    # MPQ is either gmpy.mpq or the Python equivalent from
    # sympy.external.pythonmpq
    'MPQ',

    # MPZ is either gmpy.mpz or int.
    'MPZ',

    'bit_scan1',
    'bit_scan0',
    'remove',
    'factorial',
    'sqrt',
    'is_square',
    'sqrtrem',
    'gcd',
    'lcm',
    'gcdext',
    'invert',
    'legendre',
    'jacobi',
    'kronecker',
    'iroot',
    'is_fermat_prp',
    'is_euler_prp',
    'is_strong_prp',
    'is_fibonacci_prp',
    'is_lucas_prp',
    'is_selfridge_prp',
    'is_strong_lucas_prp',
    'is_strong_selfridge_prp',
    'is_bpsw_prp',
    'is_strong_bpsw_prp',
]


#
# Tested python-flint version. Future versions might work but we will only use
# them if explicitly requested by SYMPY_GROUND_TYPES=flint.
#
_PYTHON_FLINT_VERSION_NEEDED = ["0.6", "0.7", "0.8", "0.9", "0.10"]


def _flint_version_okay(flint_version):
    major, minor = flint_version.split('.')[:2]
    flint_ver = f'{major}.{minor}'
    return flint_ver in _PYTHON_FLINT_VERSION_NEEDED

#
# We will only use gmpy2 >= 2.0.0
#
_GMPY2_MIN_VERSION = '2.0.0'


def _get_flint(sympy_ground_types):
    if sympy_ground_types not in ('auto', 'flint'):
        return None

    try:
        import flint
        # Earlier versions of python-flint may not have __version__.
        from flint import __version__ as _flint_version
    except ImportError:
        if sympy_ground_types == 'flint':
            warn("SYMPY_GROUND_TYPES was set to flint but python-flint is not "
                 "installed. Falling back to other ground types.")
        return None

    if _flint_version_okay(_flint_version):
        return flint
    elif sympy_ground_types == 'auto':
        return None
    else:
        warn(f"Using python-flint {_flint_version} because SYMPY_GROUND_TYPES "
             f"is set to flint but this version of SymPy is only tested "
             f"with python-flint versions {_PYTHON_FLINT_VERSION_NEEDED}.")
        return flint


def _get_gmpy2(sympy_ground_types):
    if sympy_ground_types not in ('auto', 'gmpy', 'gmpy2'):
        return None

    gmpy = import_module('gmpy2', min_module_version=_GMPY2_MIN_VERSION,
            module_version_attr='version', module_version_attr_call_args=())

    if sympy_ground_types != 'auto' and gmpy is None:
        warn("gmpy2 library is not installed, switching to 'python' ground types")

    return gmpy


#
# SYMPY_GROUND_TYPES can be flint, gmpy, gmpy2, python or auto (default)
#
_SYMPY_GROUND_TYPES = os.environ.get('SYMPY_GROUND_TYPES', 'auto').lower()
_flint = None
_gmpy = None

#
# First handle auto-detection of flint/gmpy2. We will prefer flint if available
# or otherwise gmpy2 if available and then lastly the python types.
#
if _SYMPY_GROUND_TYPES in ('auto', 'flint'):
    _flint = _get_flint(_SYMPY_GROUND_TYPES)
    if _flint is not None:
        _SYMPY_GROUND_TYPES = 'flint'
    else:
        _SYMPY_GROUND_TYPES = 'auto'

if _SYMPY_GROUND_TYPES in ('auto', 'gmpy', 'gmpy2'):
    _gmpy = _get_gmpy2(_SYMPY_GROUND_TYPES)
    if _gmpy is not None:
        _SYMPY_GROUND_TYPES = 'gmpy'
    else:
        _SYMPY_GROUND_TYPES = 'python'

if _SYMPY_GROUND_TYPES not in ('flint', 'gmpy', 'python'):
    warn("SYMPY_GROUND_TYPES environment variable unrecognised. "
         "Should be 'auto', 'flint', 'gmpy', 'gmpy2' or 'python'.")
    _SYMPY_GROUND_TYPES = 'python'

#
# At this point _SYMPY_GROUND_TYPES is either flint, gmpy or python. The blocks
# below define the values exported by this module in each case.
#

#
# In gmpy2 and flint, there are functions that take a long (or unsigned long)
# argument. That is, it is not possible to input a value larger than that.
#
LONG_MAX = (1 << (8*sizeof(c_long) - 1)) - 1

#
# Type checkers are confused by what SYMPY_INTS is. There may be a better type
# hint for this like Type[Integral] or something.
#
SYMPY_INTS: tuple[Type, ...]

if _SYMPY_GROUND_TYPES == 'gmpy':

    assert _gmpy is not None

    flint = None
    gmpy = _gmpy

    HAS_GMPY = 2
    GROUND_TYPES = 'gmpy'
    SYMPY_INTS = (int, type(gmpy.mpz(0)))
    MPZ = gmpy.mpz
    MPQ = gmpy.mpq

    bit_scan1 = gmpy.bit_scan1
    bit_scan0 = gmpy.bit_scan0
    remove = gmpy.remove
    factorial = gmpy.fac
    sqrt = gmpy.isqrt
    is_square = gmpy.is_square
    sqrtrem = gmpy.isqrt_rem
    gcd = gmpy.gcd
    lcm = gmpy.lcm
    gcdext = gmpy.gcdext
    invert = gmpy.invert
    legendre = gmpy.legendre
    jacobi = gmpy.jacobi
    kronecker = gmpy.kronecker

    def iroot(x, n):
        # In the latest gmpy2, the threshold for n is ULONG_MAX,
        # but adjust to the older one.
        if n <= LONG_MAX:
            return gmpy.iroot(x, n)
        return python_iroot(x, n)

    is_fermat_prp = gmpy.is_fermat_prp
    is_euler_prp = gmpy.is_euler_prp
    is_strong_prp = gmpy.is_strong_prp
    is_fibonacci_prp = gmpy.is_fibonacci_prp
    is_lucas_prp = gmpy.is_lucas_prp
    is_selfridge_prp = gmpy.is_selfridge_prp
    is_strong_lucas_prp = gmpy.is_strong_lucas_prp
    is_strong_selfridge_prp = gmpy.is_strong_selfridge_prp
    is_bpsw_prp = gmpy.is_bpsw_prp
    is_strong_bpsw_prp = gmpy.is_strong_bpsw_prp

elif _SYMPY_GROUND_TYPES == 'flint':

    assert _flint is not None

    flint = _flint
    gmpy = None

    HAS_GMPY = 0
    GROUND_TYPES = 'flint'
    SYMPY_INTS = (int, flint.fmpz) # type: ignore
    MPZ = flint.fmpz # type: ignore
    MPQ = flint.fmpq # type: ignore

    bit_scan1 = python_bit_scan1
    bit_scan0 = python_bit_scan0
    remove = python_remove
    factorial = python_factorial

    def sqrt(x):
        return flint.fmpz(x).isqrt()

    def is_square(x):
        if x < 0:
            return False
        return flint.fmpz(x).sqrtrem()[1] == 0

    def sqrtrem(x):
        return flint.fmpz(x).sqrtrem()

    def gcd(*args):
        return reduce(flint.fmpz.gcd, args, flint.fmpz(0))

    def lcm(*args):
        return reduce(flint.fmpz.lcm, args, flint.fmpz(1))

    gcdext = python_gcdext
    invert = python_invert
    legendre = python_legendre

    def jacobi(x, y):
        if y <= 0 or not y % 2:
            raise ValueError("y should be an odd positive integer")
        return flint.fmpz(x).jacobi(y)

    kronecker = python_kronecker

    def iroot(x, n):
        if n <= LONG_MAX:
            y = flint.fmpz(x).root(n)
            return y, y**n == x
        return python_iroot(x, n)

    is_fermat_prp = python_is_fermat_prp
    is_euler_prp = python_is_euler_prp
    is_strong_prp = python_is_strong_prp
    is_fibonacci_prp = python_is_fibonacci_prp
    is_lucas_prp = python_is_lucas_prp
    is_selfridge_prp = python_is_selfridge_prp
    is_strong_lucas_prp = python_is_strong_lucas_prp
    is_strong_selfridge_prp = python_is_strong_selfridge_prp
    is_bpsw_prp = python_is_bpsw_prp
    is_strong_bpsw_prp = python_is_strong_bpsw_prp

elif _SYMPY_GROUND_TYPES == 'python':

    flint = None
    gmpy = None

    HAS_GMPY = 0
    GROUND_TYPES = 'python'
    SYMPY_INTS = (int,)
    MPZ = int
    MPQ = PythonMPQ

    bit_scan1 = python_bit_scan1
    bit_scan0 = python_bit_scan0
    remove = python_remove
    factorial = python_factorial
    sqrt = python_sqrt
    is_square = python_is_square
    sqrtrem = python_sqrtrem
    gcd = python_gcd
    lcm = python_lcm
    gcdext = python_gcdext
    invert = python_invert
    legendre = python_legendre
    jacobi = python_jacobi
    kronecker = python_kronecker
    iroot = python_iroot
    is_fermat_prp = python_is_fermat_prp
    is_euler_prp = python_is_euler_prp
    is_strong_prp = python_is_strong_prp
    is_fibonacci_prp = python_is_fibonacci_prp
    is_lucas_prp = python_is_lucas_prp
    is_selfridge_prp = python_is_selfridge_prp
    is_strong_lucas_prp = python_is_strong_lucas_prp
    is_strong_selfridge_prp = python_is_strong_selfridge_prp
    is_bpsw_prp = python_is_bpsw_prp
    is_strong_bpsw_prp = python_is_strong_bpsw_prp

else:
    assert False

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\holonomic\recurrence.py ===
"""Recurrence Operators"""

from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.printing import sstr
from sympy.core.sympify import sympify


def RecurrenceOperators(base, generator):
    """
    Returns an Algebra of Recurrence Operators and the operator for
    shifting i.e. the `Sn` operator.
    The first argument needs to be the base polynomial ring for the algebra
    and the second argument must be a generator which can be either a
    noncommutative Symbol or a string.

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy import symbols
    >>> from sympy.holonomic.recurrence import RecurrenceOperators
    >>> n = symbols('n', integer=True)
    >>> R, Sn = RecurrenceOperators(ZZ.old_poly_ring(n), 'Sn')
    """

    ring = RecurrenceOperatorAlgebra(base, generator)
    return (ring, ring.shift_operator)


class RecurrenceOperatorAlgebra:
    """
    A Recurrence Operator Algebra is a set of noncommutative polynomials
    in intermediate `Sn` and coefficients in a base ring A. It follows the
    commutation rule:
    Sn * a(n) = a(n + 1) * Sn

    This class represents a Recurrence Operator Algebra and serves as the parent ring
    for Recurrence Operators.

    Examples
    ========

    >>> from sympy import ZZ
    >>> from sympy import symbols
    >>> from sympy.holonomic.recurrence import RecurrenceOperators
    >>> n = symbols('n', integer=True)
    >>> R, Sn = RecurrenceOperators(ZZ.old_poly_ring(n), 'Sn')
    >>> R
    Univariate Recurrence Operator Algebra in intermediate Sn over the base ring
    ZZ[n]

    See Also
    ========

    RecurrenceOperator
    """

    def __init__(self, base, generator):
        # the base ring for the algebra
        self.base = base
        # the operator representing shift i.e. `Sn`
        self.shift_operator = RecurrenceOperator(
            [base.zero, base.one], self)

        if generator is None:
            self.gen_symbol = symbols('Sn', commutative=False)
        else:
            if isinstance(generator, str):
                self.gen_symbol = symbols(generator, commutative=False)
            elif isinstance(generator, Symbol):
                self.gen_symbol = generator

    def __str__(self):
        string = 'Univariate Recurrence Operator Algebra in intermediate '\
            + sstr(self.gen_symbol) + ' over the base ring ' + \
            (self.base).__str__()

        return string

    __repr__ = __str__

    def __eq__(self, other):
        if self.base == other.base and self.gen_symbol == other.gen_symbol:
            return True
        else:
            return False


def _add_lists(list1, list2):
    if len(list1) <= len(list2):
        sol = [a + b for a, b in zip(list1, list2)] + list2[len(list1):]
    else:
        sol = [a + b for a, b in zip(list1, list2)] + list1[len(list2):]
    return sol


class RecurrenceOperator:
    """
    The Recurrence Operators are defined by a list of polynomials
    in the base ring and the parent ring of the Operator.

    Explanation
    ===========

    Takes a list of polynomials for each power of Sn and the
    parent ring which must be an instance of RecurrenceOperatorAlgebra.

    A Recurrence Operator can be created easily using
    the operator `Sn`. See examples below.

    Examples
    ========

    >>> from sympy.holonomic.recurrence import RecurrenceOperator, RecurrenceOperators
    >>> from sympy import ZZ
    >>> from sympy import symbols
    >>> n = symbols('n', integer=True)
    >>> R, Sn = RecurrenceOperators(ZZ.old_poly_ring(n),'Sn')

    >>> RecurrenceOperator([0, 1, n**2], R)
    (1)Sn + (n**2)Sn**2

    >>> Sn*n
    (n + 1)Sn

    >>> n*Sn*n + 1 - Sn**2*n
    (1) + (n**2 + n)Sn + (-n - 2)Sn**2

    See Also
    ========

    DifferentialOperatorAlgebra
    """

    _op_priority = 20

    def __init__(self, list_of_poly, parent):
        # the parent ring for this operator
        # must be an RecurrenceOperatorAlgebra object
        self.parent = parent
        # sequence of polynomials in n for each power of Sn
        # represents the operator
        # convert the expressions into ring elements using from_sympy
        if isinstance(list_of_poly, list):
            for i, j in enumerate(list_of_poly):
                if isinstance(j, int):
                    list_of_poly[i] = self.parent.base.from_sympy(S(j))
                elif not isinstance(j, self.parent.base.dtype):
                    list_of_poly[i] = self.parent.base.from_sympy(j)

            self.listofpoly = list_of_poly
        self.order = len(self.listofpoly) - 1

    def __mul__(self, other):
        """
        Multiplies two Operators and returns another
        RecurrenceOperator instance using the commutation rule
        Sn * a(n) = a(n + 1) * Sn
        """

        listofself = self.listofpoly
        base = self.parent.base

        if not isinstance(other, RecurrenceOperator):
            if not isinstance(other, self.parent.base.dtype):
                listofother = [self.parent.base.from_sympy(sympify(other))]

            else:
                listofother = [other]
        else:
            listofother = other.listofpoly
        # multiply a polynomial `b` with a list of polynomials

        def _mul_dmp_diffop(b, listofother):
            if isinstance(listofother, list):
                return [i * b for i in listofother]
            return [b * listofother]

        sol = _mul_dmp_diffop(listofself[0], listofother)

        # compute Sn^i * b
        def _mul_Sni_b(b):
            sol = [base.zero]

            if isinstance(b, list):
                for i in b:
                    j = base.to_sympy(i).subs(base.gens[0], base.gens[0] + S.One)
                    sol.append(base.from_sympy(j))

            else:
                j = b.subs(base.gens[0], base.gens[0] + S.One)
                sol.append(base.from_sympy(j))

            return sol

        for i in range(1, len(listofself)):
            # find Sn^i * b in ith iteration
            listofother = _mul_Sni_b(listofother)
            # solution = solution + listofself[i] * (Sn^i * b)
            sol = _add_lists(sol, _mul_dmp_diffop(listofself[i], listofother))

        return RecurrenceOperator(sol, self.parent)

    def __rmul__(self, other):
        if not isinstance(other, RecurrenceOperator):

            if isinstance(other, int):
                other = S(other)

            if not isinstance(other, self.parent.base.dtype):
                other = (self.parent.base).from_sympy(other)

            sol = [other * j for j in self.listofpoly]
            return RecurrenceOperator(sol, self.parent)

    def __add__(self, other):
        if isinstance(other, RecurrenceOperator):

            sol = _add_lists(self.listofpoly, other.listofpoly)
            return RecurrenceOperator(sol, self.parent)

        else:

            if isinstance(other, int):
                other = S(other)
            list_self = self.listofpoly
            if not isinstance(other, self.parent.base.dtype):
                list_other = [((self.parent).base).from_sympy(other)]
            else:
                list_other = [other]
            sol = [list_self[0] + list_other[0]] + list_self[1:]

            return RecurrenceOperator(sol, self.parent)

    __radd__ = __add__

    def __sub__(self, other):
        return self + (-1) * other

    def __rsub__(self, other):
        return (-1) * self + other

    def __pow__(self, n):
        if n == 1:
            return self
        result = RecurrenceOperator([self.parent.base.one], self.parent)
        if n == 0:
            return result
        # if self is `Sn`
        if self.listofpoly == self.parent.shift_operator.listofpoly:
            sol = [self.parent.base.zero] * n + [self.parent.base.one]
            return RecurrenceOperator(sol, self.parent)
        x = self
        while True:
            if n % 2:
                result *= x
            n >>= 1
            if not n:
                break
            x *= x
        return result

    def __str__(self):
        listofpoly = self.listofpoly
        print_str = ''

        for i, j in enumerate(listofpoly):
            if j == self.parent.base.zero:
                continue

            j = self.parent.base.to_sympy(j)

            if i == 0:
                print_str += '(' + sstr(j) + ')'
                continue

            if print_str:
                print_str += ' + '

            if i == 1:
                print_str += '(' + sstr(j) + ')Sn'
                continue

            print_str += '(' + sstr(j) + ')' + 'Sn**' + sstr(i)

        return print_str

    __repr__ = __str__

    def __eq__(self, other):
        if isinstance(other, RecurrenceOperator):
            if self.listofpoly == other.listofpoly and self.parent == other.parent:
                return True
            else:
                return False
        return self.listofpoly[0] == other and \
            all(i is self.parent.base.zero for i in self.listofpoly[1:])


class HolonomicSequence:
    """
    A Holonomic Sequence is a type of sequence satisfying a linear homogeneous
    recurrence relation with Polynomial coefficients. Alternatively, A sequence
    is Holonomic if and only if its generating function is a Holonomic Function.
    """

    def __init__(self, recurrence, u0=[]):
        self.recurrence = recurrence
        if not isinstance(u0, list):
            self.u0 = [u0]
        else:
            self.u0 = u0

        if len(self.u0) == 0:
            self._have_init_cond = False
        else:
            self._have_init_cond = True
        self.n = recurrence.parent.base.gens[0]

    def __repr__(self):
        str_sol = 'HolonomicSequence(%s, %s)' % ((self.recurrence).__repr__(), sstr(self.n))
        if not self._have_init_cond:
            return str_sol
        else:
            cond_str = ''
            seq_str = 0
            for i in self.u0:
                cond_str += ', u(%s) = %s' % (sstr(seq_str), sstr(i))
                seq_str += 1

            sol = str_sol + cond_str
            return sol

    __str__ = __repr__

    def __eq__(self, other):
        if self.recurrence != other.recurrence or self.n != other.n:
            return False
        if self._have_init_cond and other._have_init_cond:
            return self.u0 == other.u0
        return True

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\humanfriendly\tables.py ===
# Human friendly input/output in Python.
#
# Author: Peter Odding <peter@peterodding.com>
# Last Change: February 16, 2020
# URL: https://humanfriendly.readthedocs.io

"""
Functions that render ASCII tables.

Some generic notes about the table formatting functions in this module:

- These functions were not written with performance in mind (*at all*) because
  they're intended to format tabular data to be presented on a terminal. If
  someone were to run into a performance problem using these functions, they'd
  be printing so much tabular data to the terminal that a human wouldn't be
  able to digest the tabular data anyway, so the point is moot :-).

- These functions ignore ANSI escape sequences (at least the ones generated by
  the :mod:`~humanfriendly.terminal` module) in the calculation of columns
  widths. On reason for this is that column names are highlighted in color when
  connected to a terminal. It also means that you can use ANSI escape sequences
  to highlight certain column's values if you feel like it (for example to
  highlight deviations from the norm in an overview of calculated values).
"""

# Standard library modules.
import collections
import re

# Modules included in our package.
from humanfriendly.compat import coerce_string
from humanfriendly.terminal import (
    ansi_strip,
    ansi_width,
    ansi_wrap,
    terminal_supports_colors,
    find_terminal_size,
    HIGHLIGHT_COLOR,
)

# Public identifiers that require documentation.
__all__ = (
    'format_pretty_table',
    'format_robust_table',
    'format_rst_table',
    'format_smart_table',
)

# Compiled regular expression pattern to recognize table columns containing
# numeric data (integer and/or floating point numbers). Used to right-align the
# contents of such columns.
#
# Pre-emptive snarky comment: This pattern doesn't match every possible
# floating point number notation!?!1!1
#
# Response: I know, that's intentional. The use of this regular expression
# pattern has a very high DWIM level and weird floating point notations do not
# fall under the DWIM umbrella :-).
NUMERIC_DATA_PATTERN = re.compile(r'^\d+(\.\d+)?$')


def format_smart_table(data, column_names):
    """
    Render tabular data using the most appropriate representation.

    :param data: An iterable (e.g. a :func:`tuple` or :class:`list`)
                 containing the rows of the table, where each row is an
                 iterable containing the columns of the table (strings).
    :param column_names: An iterable of column names (strings).
    :returns: The rendered table (a string).

    If you want an easy way to render tabular data on a terminal in a human
    friendly format then this function is for you! It works as follows:

    - If the input data doesn't contain any line breaks the function
      :func:`format_pretty_table()` is used to render a pretty table. If the
      resulting table fits in the terminal without wrapping the rendered pretty
      table is returned.

    - If the input data does contain line breaks or if a pretty table would
      wrap (given the width of the terminal) then the function
      :func:`format_robust_table()` is used to render a more robust table that
      can deal with data containing line breaks and long text.
    """
    # Normalize the input in case we fall back from a pretty table to a robust
    # table (in which case we'll definitely iterate the input more than once).
    data = [normalize_columns(r) for r in data]
    column_names = normalize_columns(column_names)
    # Make sure the input data doesn't contain any line breaks (because pretty
    # tables break horribly when a column's text contains a line break :-).
    if not any(any('\n' in c for c in r) for r in data):
        # Render a pretty table.
        pretty_table = format_pretty_table(data, column_names)
        # Check if the pretty table fits in the terminal.
        table_width = max(map(ansi_width, pretty_table.splitlines()))
        num_rows, num_columns = find_terminal_size()
        if table_width <= num_columns:
            # The pretty table fits in the terminal without wrapping!
            return pretty_table
    # Fall back to a robust table when a pretty table won't work.
    return format_robust_table(data, column_names)


def format_pretty_table(data, column_names=None, horizontal_bar='-', vertical_bar='|'):
    """
    Render a table using characters like dashes and vertical bars to emulate borders.

    :param data: An iterable (e.g. a :func:`tuple` or :class:`list`)
                 containing the rows of the table, where each row is an
                 iterable containing the columns of the table (strings).
    :param column_names: An iterable of column names (strings).
    :param horizontal_bar: The character used to represent a horizontal bar (a
                           string).
    :param vertical_bar: The character used to represent a vertical bar (a
                         string).
    :returns: The rendered table (a string).

    Here's an example:

    >>> from humanfriendly.tables import format_pretty_table
    >>> column_names = ['Version', 'Uploaded on', 'Downloads']
    >>> humanfriendly_releases = [
    ... ['1.23', '2015-05-25', '218'],
    ... ['1.23.1', '2015-05-26', '1354'],
    ... ['1.24', '2015-05-26', '223'],
    ... ['1.25', '2015-05-26', '4319'],
    ... ['1.25.1', '2015-06-02', '197'],
    ... ]
    >>> print(format_pretty_table(humanfriendly_releases, column_names))
    -------------------------------------
    | Version | Uploaded on | Downloads |
    -------------------------------------
    | 1.23    | 2015-05-25  |       218 |
    | 1.23.1  | 2015-05-26  |      1354 |
    | 1.24    | 2015-05-26  |       223 |
    | 1.25    | 2015-05-26  |      4319 |
    | 1.25.1  | 2015-06-02  |       197 |
    -------------------------------------

    Notes about the resulting table:

    - If a column contains numeric data (integer and/or floating point
      numbers) in all rows (ignoring column names of course) then the content
      of that column is right-aligned, as can be seen in the example above. The
      idea here is to make it easier to compare the numbers in different
      columns to each other.

    - The column names are highlighted in color so they stand out a bit more
      (see also :data:`.HIGHLIGHT_COLOR`). The following screen shot shows what
      that looks like (my terminals are always set to white text on a black
      background):

      .. image:: images/pretty-table.png
    """
    # Normalize the input because we'll have to iterate it more than once.
    data = [normalize_columns(r, expandtabs=True) for r in data]
    if column_names is not None:
        column_names = normalize_columns(column_names)
        if column_names:
            if terminal_supports_colors():
                column_names = [highlight_column_name(n) for n in column_names]
            data.insert(0, column_names)
    # Calculate the maximum width of each column.
    widths = collections.defaultdict(int)
    numeric_data = collections.defaultdict(list)
    for row_index, row in enumerate(data):
        for column_index, column in enumerate(row):
            widths[column_index] = max(widths[column_index], ansi_width(column))
            if not (column_names and row_index == 0):
                numeric_data[column_index].append(bool(NUMERIC_DATA_PATTERN.match(ansi_strip(column))))
    # Create a horizontal bar of dashes as a delimiter.
    line_delimiter = horizontal_bar * (sum(widths.values()) + len(widths) * 3 + 1)
    # Start the table with a vertical bar.
    lines = [line_delimiter]
    # Format the rows and columns.
    for row_index, row in enumerate(data):
        line = [vertical_bar]
        for column_index, column in enumerate(row):
            padding = ' ' * (widths[column_index] - ansi_width(column))
            if all(numeric_data[column_index]):
                line.append(' ' + padding + column + ' ')
            else:
                line.append(' ' + column + padding + ' ')
            line.append(vertical_bar)
        lines.append(u''.join(line))
        if column_names and row_index == 0:
            lines.append(line_delimiter)
    # End the table with a vertical bar.
    lines.append(line_delimiter)
    # Join the lines, returning a single string.
    return u'\n'.join(lines)


def format_robust_table(data, column_names):
    """
    Render tabular data with one column per line (allowing columns with line breaks).

    :param data: An iterable (e.g. a :func:`tuple` or :class:`list`)
                 containing the rows of the table, where each row is an
                 iterable containing the columns of the table (strings).
    :param column_names: An iterable of column names (strings).
    :returns: The rendered table (a string).

    Here's an example:

    >>> from humanfriendly.tables import format_robust_table
    >>> column_names = ['Version', 'Uploaded on', 'Downloads']
    >>> humanfriendly_releases = [
    ... ['1.23', '2015-05-25', '218'],
    ... ['1.23.1', '2015-05-26', '1354'],
    ... ['1.24', '2015-05-26', '223'],
    ... ['1.25', '2015-05-26', '4319'],
    ... ['1.25.1', '2015-06-02', '197'],
    ... ]
    >>> print(format_robust_table(humanfriendly_releases, column_names))
    -----------------------
    Version: 1.23
    Uploaded on: 2015-05-25
    Downloads: 218
    -----------------------
    Version: 1.23.1
    Uploaded on: 2015-05-26
    Downloads: 1354
    -----------------------
    Version: 1.24
    Uploaded on: 2015-05-26
    Downloads: 223
    -----------------------
    Version: 1.25
    Uploaded on: 2015-05-26
    Downloads: 4319
    -----------------------
    Version: 1.25.1
    Uploaded on: 2015-06-02
    Downloads: 197
    -----------------------

    The column names are highlighted in bold font and color so they stand out a
    bit more (see :data:`.HIGHLIGHT_COLOR`).
    """
    blocks = []
    column_names = ["%s:" % n for n in normalize_columns(column_names)]
    if terminal_supports_colors():
        column_names = [highlight_column_name(n) for n in column_names]
    # Convert each row into one or more `name: value' lines (one per column)
    # and group each `row of lines' into a block (i.e. rows become blocks).
    for row in data:
        lines = []
        for column_index, column_text in enumerate(normalize_columns(row)):
            stripped_column = column_text.strip()
            if '\n' not in stripped_column:
                # Columns without line breaks are formatted inline.
                lines.append("%s %s" % (column_names[column_index], stripped_column))
            else:
                # Columns with line breaks could very well contain indented
                # lines, so we'll put the column name on a separate line. This
                # way any indentation remains intact, and it's easier to
                # copy/paste the text.
                lines.append(column_names[column_index])
                lines.extend(column_text.rstrip().splitlines())
        blocks.append(lines)
    # Calculate the width of the row delimiter.
    num_rows, num_columns = find_terminal_size()
    longest_line = max(max(map(ansi_width, lines)) for lines in blocks)
    delimiter = u"\n%s\n" % ('-' * min(longest_line, num_columns))
    # Force a delimiter at the start and end of the table.
    blocks.insert(0, "")
    blocks.append("")
    # Embed the row delimiter between every two blocks.
    return delimiter.join(u"\n".join(b) for b in blocks).strip()


def format_rst_table(data, column_names=None):
    """
    Render a table in reStructuredText_ format.

    :param data: An iterable (e.g. a :func:`tuple` or :class:`list`)
                 containing the rows of the table, where each row is an
                 iterable containing the columns of the table (strings).
    :param column_names: An iterable of column names (strings).
    :returns: The rendered table (a string).

    Here's an example:

    >>> from humanfriendly.tables import format_rst_table
    >>> column_names = ['Version', 'Uploaded on', 'Downloads']
    >>> humanfriendly_releases = [
    ... ['1.23', '2015-05-25', '218'],
    ... ['1.23.1', '2015-05-26', '1354'],
    ... ['1.24', '2015-05-26', '223'],
    ... ['1.25', '2015-05-26', '4319'],
    ... ['1.25.1', '2015-06-02', '197'],
    ... ]
    >>> print(format_rst_table(humanfriendly_releases, column_names))
    =======  ===========  =========
    Version  Uploaded on  Downloads
    =======  ===========  =========
    1.23     2015-05-25   218
    1.23.1   2015-05-26   1354
    1.24     2015-05-26   223
    1.25     2015-05-26   4319
    1.25.1   2015-06-02   197
    =======  ===========  =========

    .. _reStructuredText: https://en.wikipedia.org/wiki/ReStructuredText
    """
    data = [normalize_columns(r) for r in data]
    if column_names:
        data.insert(0, normalize_columns(column_names))
    # Calculate the maximum width of each column.
    widths = collections.defaultdict(int)
    for row in data:
        for index, column in enumerate(row):
            widths[index] = max(widths[index], len(column))
    # Pad the columns using whitespace.
    for row in data:
        for index, column in enumerate(row):
            if index < (len(row) - 1):
                row[index] = column.ljust(widths[index])
    # Add table markers.
    delimiter = ['=' * w for i, w in sorted(widths.items())]
    if column_names:
        data.insert(1, delimiter)
    data.insert(0, delimiter)
    data.append(delimiter)
    # Join the lines and columns together.
    return '\n'.join('  '.join(r) for r in data)


def normalize_columns(row, expandtabs=False):
    results = []
    for value in row:
        text = coerce_string(value)
        if expandtabs:
            text = text.expandtabs()
        results.append(text)
    return results


def highlight_column_name(name):
    return ansi_wrap(name, bold=True, color=HIGHLIGHT_COLOR)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\external\pythonmpq.py ===
"""
PythonMPQ: Rational number type based on Python integers.

This class is intended as a pure Python fallback for when gmpy2 is not
installed. If gmpy2 is installed then its mpq type will be used instead. The
mpq type is around 20x faster. We could just use the stdlib Fraction class
here but that is slower:

    from fractions import Fraction
    from sympy.external.pythonmpq import PythonMPQ
    nums = range(1000)
    dens = range(5, 1005)
    rats = [Fraction(n, d) for n, d in zip(nums, dens)]
    sum(rats) # <--- 24 milliseconds
    rats = [PythonMPQ(n, d) for n, d in zip(nums, dens)]
    sum(rats) # <---  7 milliseconds

Both mpq and Fraction have some awkward features like the behaviour of
division with // and %:

    >>> from fractions import Fraction
    >>> Fraction(2, 3) % Fraction(1, 4)
    1/6

For the QQ domain we do not want this behaviour because there should be no
remainder when dividing rational numbers. SymPy does not make use of this
aspect of mpq when gmpy2 is installed. Since this class is a fallback for that
case we do not bother implementing e.g. __mod__ so that we can be sure we
are not using it when gmpy2 is installed either.
"""

from __future__ import annotations
import operator
from math import gcd
from decimal import Decimal
from fractions import Fraction
import sys
from typing import Type


# Used for __hash__
_PyHASH_MODULUS = sys.hash_info.modulus
_PyHASH_INF = sys.hash_info.inf


class PythonMPQ:
    """Rational number implementation that is intended to be compatible with
    gmpy2's mpq.

    Also slightly faster than fractions.Fraction.

    PythonMPQ should be treated as immutable although no effort is made to
    prevent mutation (since that might slow down calculations).
    """
    __slots__ = ('numerator', 'denominator')

    def __new__(cls, numerator, denominator=None):
        """Construct PythonMPQ with gcd computation and checks"""
        if denominator is not None:
            #
            # PythonMPQ(n, d): require n and d to be int and d != 0
            #
            if isinstance(numerator, int) and isinstance(denominator, int):
                # This is the slow part:
                divisor = gcd(numerator, denominator)
                numerator //= divisor
                denominator //= divisor
                return cls._new_check(numerator, denominator)
        else:
            #
            # PythonMPQ(q)
            #
            # Here q can be PythonMPQ, int, Decimal, float, Fraction or str
            #
            if isinstance(numerator, int):
                return cls._new(numerator, 1)
            elif isinstance(numerator, PythonMPQ):
                return cls._new(numerator.numerator, numerator.denominator)

            # Let Fraction handle Decimal/float conversion and str parsing
            if isinstance(numerator, (Decimal, float, str)):
                numerator = Fraction(numerator)
            if isinstance(numerator, Fraction):
                return cls._new(numerator.numerator, numerator.denominator)
        #
        # Reject everything else. This is more strict than mpq which allows
        # things like mpq(Fraction, Fraction) or mpq(Decimal, any). The mpq
        # behaviour is somewhat inconsistent so we choose to accept only a
        # more strict subset of what mpq allows.
        #
        raise TypeError("PythonMPQ() requires numeric or string argument")

    @classmethod
    def _new_check(cls, numerator, denominator):
        """Construct PythonMPQ, check divide by zero and canonicalize signs"""
        if not denominator:
            raise ZeroDivisionError(f'Zero divisor {numerator}/{denominator}')
        elif denominator < 0:
            numerator = -numerator
            denominator = -denominator
        return cls._new(numerator, denominator)

    @classmethod
    def _new(cls, numerator, denominator):
        """Construct PythonMPQ efficiently (no checks)"""
        obj = super().__new__(cls)
        obj.numerator = numerator
        obj.denominator = denominator
        return obj

    def __int__(self):
        """Convert to int (truncates towards zero)"""
        p, q = self.numerator, self.denominator
        if p < 0:
            return -(-p//q)
        return p//q

    def __float__(self):
        """Convert to float (approximately)"""
        return self.numerator / self.denominator

    def __bool__(self):
        """True/False if nonzero/zero"""
        return bool(self.numerator)

    def __eq__(self, other):
        """Compare equal with PythonMPQ, int, float, Decimal or Fraction"""
        if isinstance(other, PythonMPQ):
            return (self.numerator == other.numerator
                and self.denominator == other.denominator)
        elif isinstance(other, self._compatible_types):
            return self.__eq__(PythonMPQ(other))
        else:
            return NotImplemented

    def __hash__(self):
        """hash - same as mpq/Fraction"""
        try:
            dinv = pow(self.denominator, -1, _PyHASH_MODULUS)
        except ValueError:
            hash_ = _PyHASH_INF
        else:
            hash_ = hash(hash(abs(self.numerator)) * dinv)
        result = hash_ if self.numerator >= 0 else -hash_
        return -2 if result == -1 else result

    def __reduce__(self):
        """Deconstruct for pickling"""
        return type(self), (self.numerator, self.denominator)

    def __str__(self):
        """Convert to string"""
        if self.denominator != 1:
            return f"{self.numerator}/{self.denominator}"
        else:
            return f"{self.numerator}"

    def __repr__(self):
        """Convert to string"""
        return f"MPQ({self.numerator},{self.denominator})"

    def _cmp(self, other, op):
        """Helper for lt/le/gt/ge"""
        if not isinstance(other, self._compatible_types):
            return NotImplemented
        lhs = self.numerator * other.denominator
        rhs = other.numerator * self.denominator
        return op(lhs, rhs)

    def __lt__(self, other):
        """self < other"""
        return self._cmp(other, operator.lt)

    def __le__(self, other):
        """self <= other"""
        return self._cmp(other, operator.le)

    def __gt__(self, other):
        """self > other"""
        return self._cmp(other, operator.gt)

    def __ge__(self, other):
        """self >= other"""
        return self._cmp(other, operator.ge)

    def __abs__(self):
        """abs(q)"""
        return self._new(abs(self.numerator), self.denominator)

    def __pos__(self):
        """+q"""
        return self

    def __neg__(self):
        """-q"""
        return self._new(-self.numerator, self.denominator)

    def __add__(self, other):
        """q1 + q2"""
        if isinstance(other, PythonMPQ):
            #
            # This is much faster than the naive method used in the stdlib
            # fractions module. Not sure where this method comes from
            # though...
            #
            # Compare timings for something like:
            #   nums = range(1000)
            #   rats = [PythonMPQ(n, d) for n, d in zip(nums[:-5], nums[5:])]
            #   sum(rats) # <-- time this
            #
            ap, aq = self.numerator, self.denominator
            bp, bq = other.numerator, other.denominator
            g = gcd(aq, bq)
            if g == 1:
                p = ap*bq + aq*bp
                q = bq*aq
            else:
                q1, q2 = aq//g, bq//g
                p, q = ap*q2 + bp*q1, q1*q2
                g2 = gcd(p, g)
                p, q = (p // g2), q * (g // g2)

        elif isinstance(other, int):
            p = self.numerator + self.denominator * other
            q = self.denominator
        else:
            return NotImplemented

        return self._new(p, q)

    def __radd__(self, other):
        """z1 + q2"""
        if isinstance(other, int):
            p = self.numerator + self.denominator * other
            q = self.denominator
            return self._new(p, q)
        else:
            return NotImplemented

    def __sub__(self ,other):
        """q1 - q2"""
        if isinstance(other, PythonMPQ):
            ap, aq = self.numerator, self.denominator
            bp, bq = other.numerator, other.denominator
            g = gcd(aq, bq)
            if g == 1:
                p = ap*bq - aq*bp
                q = bq*aq
            else:
                q1, q2 = aq//g, bq//g
                p, q = ap*q2 - bp*q1, q1*q2
                g2 = gcd(p, g)
                p, q = (p // g2), q * (g // g2)
        elif isinstance(other, int):
            p = self.numerator - self.denominator*other
            q = self.denominator
        else:
            return NotImplemented

        return self._new(p, q)

    def __rsub__(self, other):
        """z1 - q2"""
        if isinstance(other, int):
            p = self.denominator * other - self.numerator
            q = self.denominator
            return self._new(p, q)
        else:
            return NotImplemented

    def __mul__(self, other):
        """q1 * q2"""
        if isinstance(other, PythonMPQ):
            ap, aq = self.numerator, self.denominator
            bp, bq = other.numerator, other.denominator
            x1 = gcd(ap, bq)
            x2 = gcd(bp, aq)
            p, q = ((ap//x1)*(bp//x2), (aq//x2)*(bq//x1))
        elif isinstance(other, int):
            x = gcd(other, self.denominator)
            p = self.numerator*(other//x)
            q = self.denominator//x
        else:
            return NotImplemented

        return self._new(p, q)

    def __rmul__(self, other):
        """z1 * q2"""
        if isinstance(other, int):
            x = gcd(self.denominator, other)
            p = self.numerator*(other//x)
            q = self.denominator//x
            return self._new(p, q)
        else:
            return NotImplemented

    def __pow__(self, exp):
        """q ** z"""
        p, q = self.numerator, self.denominator

        if exp < 0:
            p, q, exp = q, p, -exp

        return self._new_check(p**exp, q**exp)

    def __truediv__(self, other):
        """q1 / q2"""
        if isinstance(other, PythonMPQ):
            ap, aq = self.numerator, self.denominator
            bp, bq = other.numerator, other.denominator
            x1 = gcd(ap, bp)
            x2 = gcd(bq, aq)
            p, q = ((ap//x1)*(bq//x2), (aq//x2)*(bp//x1))
        elif isinstance(other, int):
            x = gcd(other, self.numerator)
            p = self.numerator//x
            q = self.denominator*(other//x)
        else:
            return NotImplemented

        return self._new_check(p, q)

    def __rtruediv__(self, other):
        """z / q"""
        if isinstance(other, int):
            x = gcd(self.numerator, other)
            p = self.denominator*(other//x)
            q = self.numerator//x
            return self._new_check(p, q)
        else:
            return NotImplemented

    _compatible_types: tuple[Type, ...] = ()

#
# These are the types that PythonMPQ will interoperate with for operations
# and comparisons such as ==, + etc. We define this down here so that we can
# include PythonMPQ in the list as well.
#
PythonMPQ._compatible_types = (PythonMPQ, int, Decimal, Fraction)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\cartesian.py ===
"""Operators and states for 1D cartesian position and momentum.

TODO:

* Add 3D classes to mappings in operatorset.py

"""

from sympy.core.numbers import (I, pi)
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.special.delta_functions import DiracDelta
from sympy.sets.sets import Interval

from sympy.physics.quantum.constants import hbar
from sympy.physics.quantum.hilbert import L2
from sympy.physics.quantum.operator import DifferentialOperator, HermitianOperator
from sympy.physics.quantum.state import Ket, Bra, State

__all__ = [
    'XOp',
    'YOp',
    'ZOp',
    'PxOp',
    'X',
    'Y',
    'Z',
    'Px',
    'XKet',
    'XBra',
    'PxKet',
    'PxBra',
    'PositionState3D',
    'PositionKet3D',
    'PositionBra3D'
]

#-------------------------------------------------------------------------
# Position operators
#-------------------------------------------------------------------------


class XOp(HermitianOperator):
    """1D cartesian position operator."""

    @classmethod
    def default_args(self):
        return ("X",)

    @classmethod
    def _eval_hilbert_space(self, args):
        return L2(Interval(S.NegativeInfinity, S.Infinity))

    def _eval_commutator_PxOp(self, other):
        return I*hbar

    def _apply_operator_XKet(self, ket, **options):
        return ket.position*ket

    def _apply_operator_PositionKet3D(self, ket, **options):
        return ket.position_x*ket

    def _represent_PxKet(self, basis, *, index=1, **options):
        states = basis._enumerate_state(2, start_index=index)
        coord1 = states[0].momentum
        coord2 = states[1].momentum
        d = DifferentialOperator(coord1)
        delta = DiracDelta(coord1 - coord2)

        return I*hbar*(d*delta)


class YOp(HermitianOperator):
    """ Y cartesian coordinate operator (for 2D or 3D systems) """

    @classmethod
    def default_args(self):
        return ("Y",)

    @classmethod
    def _eval_hilbert_space(self, args):
        return L2(Interval(S.NegativeInfinity, S.Infinity))

    def _apply_operator_PositionKet3D(self, ket, **options):
        return ket.position_y*ket


class ZOp(HermitianOperator):
    """ Z cartesian coordinate operator (for 3D systems) """

    @classmethod
    def default_args(self):
        return ("Z",)

    @classmethod
    def _eval_hilbert_space(self, args):
        return L2(Interval(S.NegativeInfinity, S.Infinity))

    def _apply_operator_PositionKet3D(self, ket, **options):
        return ket.position_z*ket

#-------------------------------------------------------------------------
# Momentum operators
#-------------------------------------------------------------------------


class PxOp(HermitianOperator):
    """1D cartesian momentum operator."""

    @classmethod
    def default_args(self):
        return ("Px",)

    @classmethod
    def _eval_hilbert_space(self, args):
        return L2(Interval(S.NegativeInfinity, S.Infinity))

    def _apply_operator_PxKet(self, ket, **options):
        return ket.momentum*ket

    def _represent_XKet(self, basis, *, index=1, **options):
        states = basis._enumerate_state(2, start_index=index)
        coord1 = states[0].position
        coord2 = states[1].position
        d = DifferentialOperator(coord1)
        delta = DiracDelta(coord1 - coord2)

        return -I*hbar*(d*delta)

X = XOp('X')
Y = YOp('Y')
Z = ZOp('Z')
Px = PxOp('Px')

#-------------------------------------------------------------------------
# Position eigenstates
#-------------------------------------------------------------------------


class XKet(Ket):
    """1D cartesian position eigenket."""

    @classmethod
    def _operators_to_state(self, op, **options):
        return self.__new__(self, *_lowercase_labels(op), **options)

    def _state_to_operators(self, op_class, **options):
        return op_class.__new__(op_class,
                                *_uppercase_labels(self), **options)

    @classmethod
    def default_args(self):
        return ("x",)

    @classmethod
    def dual_class(self):
        return XBra

    @property
    def position(self):
        """The position of the state."""
        return self.label[0]

    def _enumerate_state(self, num_states, **options):
        return _enumerate_continuous_1D(self, num_states, **options)

    def _eval_innerproduct_XBra(self, bra, **hints):
        return DiracDelta(self.position - bra.position)

    def _eval_innerproduct_PxBra(self, bra, **hints):
        return exp(-I*self.position*bra.momentum/hbar)/sqrt(2*pi*hbar)


class XBra(Bra):
    """1D cartesian position eigenbra."""

    @classmethod
    def default_args(self):
        return ("x",)

    @classmethod
    def dual_class(self):
        return XKet

    @property
    def position(self):
        """The position of the state."""
        return self.label[0]


class PositionState3D(State):
    """ Base class for 3D cartesian position eigenstates """

    @classmethod
    def _operators_to_state(self, op, **options):
        return self.__new__(self, *_lowercase_labels(op), **options)

    def _state_to_operators(self, op_class, **options):
        return op_class.__new__(op_class,
                                *_uppercase_labels(self), **options)

    @classmethod
    def default_args(self):
        return ("x", "y", "z")

    @property
    def position_x(self):
        """ The x coordinate of the state """
        return self.label[0]

    @property
    def position_y(self):
        """ The y coordinate of the state """
        return self.label[1]

    @property
    def position_z(self):
        """ The z coordinate of the state """
        return self.label[2]


class PositionKet3D(Ket, PositionState3D):
    """ 3D cartesian position eigenket """

    def _eval_innerproduct_PositionBra3D(self, bra, **options):
        x_diff = self.position_x - bra.position_x
        y_diff = self.position_y - bra.position_y
        z_diff = self.position_z - bra.position_z

        return DiracDelta(x_diff)*DiracDelta(y_diff)*DiracDelta(z_diff)

    @classmethod
    def dual_class(self):
        return PositionBra3D


# XXX: The type:ignore here is because mypy gives Definition of
# "_state_to_operators" in base class "PositionState3D" is incompatible with
# definition in base class "BraBase"
class PositionBra3D(Bra, PositionState3D):  # type: ignore
    """ 3D cartesian position eigenbra """

    @classmethod
    def dual_class(self):
        return PositionKet3D

#-------------------------------------------------------------------------
# Momentum eigenstates
#-------------------------------------------------------------------------


class PxKet(Ket):
    """1D cartesian momentum eigenket."""

    @classmethod
    def _operators_to_state(self, op, **options):
        return self.__new__(self, *_lowercase_labels(op), **options)

    def _state_to_operators(self, op_class, **options):
        return op_class.__new__(op_class,
                                *_uppercase_labels(self), **options)

    @classmethod
    def default_args(self):
        return ("px",)

    @classmethod
    def dual_class(self):
        return PxBra

    @property
    def momentum(self):
        """The momentum of the state."""
        return self.label[0]

    def _enumerate_state(self, *args, **options):
        return _enumerate_continuous_1D(self, *args, **options)

    def _eval_innerproduct_XBra(self, bra, **hints):
        return exp(I*self.momentum*bra.position/hbar)/sqrt(2*pi*hbar)

    def _eval_innerproduct_PxBra(self, bra, **hints):
        return DiracDelta(self.momentum - bra.momentum)


class PxBra(Bra):
    """1D cartesian momentum eigenbra."""

    @classmethod
    def default_args(self):
        return ("px",)

    @classmethod
    def dual_class(self):
        return PxKet

    @property
    def momentum(self):
        """The momentum of the state."""
        return self.label[0]

#-------------------------------------------------------------------------
# Global helper functions
#-------------------------------------------------------------------------


def _enumerate_continuous_1D(*args, **options):
    state = args[0]
    num_states = args[1]
    state_class = state.__class__
    index_list = options.pop('index_list', [])

    if len(index_list) == 0:
        start_index = options.pop('start_index', 1)
        index_list = list(range(start_index, start_index + num_states))

    enum_states = [0 for i in range(len(index_list))]

    for i, ind in enumerate(index_list):
        label = state.args[0]
        enum_states[i] = state_class(str(label) + "_" + str(ind), **options)

    return enum_states


def _lowercase_labels(ops):
    if not isinstance(ops, set):
        ops = [ops]

    return [str(arg.label[0]).lower() for arg in ops]


def _uppercase_labels(ops):
    if not isinstance(ops, set):
        ops = [ops]

    new_args = [str(arg.label[0])[0].upper() +
                str(arg.label[0])[1:] for arg in ops]

    return new_args

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\urllib3\fields.py ===
from __future__ import annotations

import email.utils
import mimetypes
import typing

_TYPE_FIELD_VALUE = typing.Union[str, bytes]
_TYPE_FIELD_VALUE_TUPLE = typing.Union[
    _TYPE_FIELD_VALUE,
    tuple[str, _TYPE_FIELD_VALUE],
    tuple[str, _TYPE_FIELD_VALUE, str],
]


def guess_content_type(
    filename: str | None, default: str = "application/octet-stream"
) -> str:
    """
    Guess the "Content-Type" of a file.

    :param filename:
        The filename to guess the "Content-Type" of using :mod:`mimetypes`.
    :param default:
        If no "Content-Type" can be guessed, default to `default`.
    """
    if filename:
        return mimetypes.guess_type(filename)[0] or default
    return default


def format_header_param_rfc2231(name: str, value: _TYPE_FIELD_VALUE) -> str:
    """
    Helper function to format and quote a single header parameter using the
    strategy defined in RFC 2231.

    Particularly useful for header parameters which might contain
    non-ASCII values, like file names. This follows
    `RFC 2388 Section 4.4 <https://tools.ietf.org/html/rfc2388#section-4.4>`_.

    :param name:
        The name of the parameter, a string expected to be ASCII only.
    :param value:
        The value of the parameter, provided as ``bytes`` or `str``.
    :returns:
        An RFC-2231-formatted unicode string.

    .. deprecated:: 2.0.0
        Will be removed in urllib3 v2.1.0. This is not valid for
        ``multipart/form-data`` header parameters.
    """
    import warnings

    warnings.warn(
        "'format_header_param_rfc2231' is deprecated and will be "
        "removed in urllib3 v2.1.0. This is not valid for "
        "multipart/form-data header parameters.",
        DeprecationWarning,
        stacklevel=2,
    )

    if isinstance(value, bytes):
        value = value.decode("utf-8")

    if not any(ch in value for ch in '"\\\r\n'):
        result = f'{name}="{value}"'
        try:
            result.encode("ascii")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
        else:
            return result

    value = email.utils.encode_rfc2231(value, "utf-8")
    value = f"{name}*={value}"

    return value


def format_multipart_header_param(name: str, value: _TYPE_FIELD_VALUE) -> str:
    """
    Format and quote a single multipart header parameter.

    This follows the `WHATWG HTML Standard`_ as of 2021/06/10, matching
    the behavior of current browser and curl versions. Values are
    assumed to be UTF-8. The ``\\n``, ``\\r``, and ``"`` characters are
    percent encoded.

    .. _WHATWG HTML Standard:
        https://html.spec.whatwg.org/multipage/
        form-control-infrastructure.html#multipart-form-data

    :param name:
        The name of the parameter, an ASCII-only ``str``.
    :param value:
        The value of the parameter, a ``str`` or UTF-8 encoded
        ``bytes``.
    :returns:
        A string ``name="value"`` with the escaped value.

    .. versionchanged:: 2.0.0
        Matches the WHATWG HTML Standard as of 2021/06/10. Control
        characters are no longer percent encoded.

    .. versionchanged:: 2.0.0
        Renamed from ``format_header_param_html5`` and
        ``format_header_param``. The old names will be removed in
        urllib3 v2.1.0.
    """
    if isinstance(value, bytes):
        value = value.decode("utf-8")

    # percent encode \n \r "
    value = value.translate({10: "%0A", 13: "%0D", 34: "%22"})
    return f'{name}="{value}"'


def format_header_param_html5(name: str, value: _TYPE_FIELD_VALUE) -> str:
    """
    .. deprecated:: 2.0.0
        Renamed to :func:`format_multipart_header_param`. Will be
        removed in urllib3 v2.1.0.
    """
    import warnings

    warnings.warn(
        "'format_header_param_html5' has been renamed to "
        "'format_multipart_header_param'. The old name will be "
        "removed in urllib3 v2.1.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    return format_multipart_header_param(name, value)


def format_header_param(name: str, value: _TYPE_FIELD_VALUE) -> str:
    """
    .. deprecated:: 2.0.0
        Renamed to :func:`format_multipart_header_param`. Will be
        removed in urllib3 v2.1.0.
    """
    import warnings

    warnings.warn(
        "'format_header_param' has been renamed to "
        "'format_multipart_header_param'. The old name will be "
        "removed in urllib3 v2.1.0.",
        DeprecationWarning,
        stacklevel=2,
    )
    return format_multipart_header_param(name, value)


class RequestField:
    """
    A data container for request body parameters.

    :param name:
        The name of this request field. Must be unicode.
    :param data:
        The data/value body.
    :param filename:
        An optional filename of the request field. Must be unicode.
    :param headers:
        An optional dict-like object of headers to initially use for the field.

    .. versionchanged:: 2.0.0
        The ``header_formatter`` parameter is deprecated and will
        be removed in urllib3 v2.1.0.
    """

    def __init__(
        self,
        name: str,
        data: _TYPE_FIELD_VALUE,
        filename: str | None = None,
        headers: typing.Mapping[str, str] | None = None,
        header_formatter: typing.Callable[[str, _TYPE_FIELD_VALUE], str] | None = None,
    ):
        self._name = name
        self._filename = filename
        self.data = data
        self.headers: dict[str, str | None] = {}
        if headers:
            self.headers = dict(headers)

        if header_formatter is not None:
            import warnings

            warnings.warn(
                "The 'header_formatter' parameter is deprecated and "
                "will be removed in urllib3 v2.1.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.header_formatter = header_formatter
        else:
            self.header_formatter = format_multipart_header_param

    @classmethod
    def from_tuples(
        cls,
        fieldname: str,
        value: _TYPE_FIELD_VALUE_TUPLE,
        header_formatter: typing.Callable[[str, _TYPE_FIELD_VALUE], str] | None = None,
    ) -> RequestField:
        """
        A :class:`~urllib3.fields.RequestField` factory from old-style tuple parameters.

        Supports constructing :class:`~urllib3.fields.RequestField` from
        parameter of key/value strings AND key/filetuple. A filetuple is a
        (filename, data, MIME type) tuple where the MIME type is optional.
        For example::

            'foo': 'bar',
            'fakefile': ('foofile.txt', 'contents of foofile'),
            'realfile': ('barfile.txt', open('realfile').read()),
            'typedfile': ('bazfile.bin', open('bazfile').read(), 'image/jpeg'),
            'nonamefile': 'contents of nonamefile field',

        Field names and filenames must be unicode.
        """
        filename: str | None
        content_type: str | None
        data: _TYPE_FIELD_VALUE

        if isinstance(value, tuple):
            if len(value) == 3:
                filename, data, content_type = value
            else:
                filename, data = value
                content_type = guess_content_type(filename)
        else:
            filename = None
            content_type = None
            data = value

        request_param = cls(
            fieldname, data, filename=filename, header_formatter=header_formatter
        )
        request_param.make_multipart(content_type=content_type)

        return request_param

    def _render_part(self, name: str, value: _TYPE_FIELD_VALUE) -> str:
        """
        Override this method to change how each multipart header
        parameter is formatted. By default, this calls
        :func:`format_multipart_header_param`.

        :param name:
            The name of the parameter, an ASCII-only ``str``.
        :param value:
            The value of the parameter, a ``str`` or UTF-8 encoded
            ``bytes``.

        :meta public:
        """
        return self.header_formatter(name, value)

    def _render_parts(
        self,
        header_parts: (
            dict[str, _TYPE_FIELD_VALUE | None]
            | typing.Sequence[tuple[str, _TYPE_FIELD_VALUE | None]]
        ),
    ) -> str:
        """
        Helper function to format and quote a single header.

        Useful for single headers that are composed of multiple items. E.g.,
        'Content-Disposition' fields.

        :param header_parts:
            A sequence of (k, v) tuples or a :class:`dict` of (k, v) to format
            as `k1="v1"; k2="v2"; ...`.
        """
        iterable: typing.Iterable[tuple[str, _TYPE_FIELD_VALUE | None]]

        parts = []
        if isinstance(header_parts, dict):
            iterable = header_parts.items()
        else:
            iterable = header_parts

        for name, value in iterable:
            if value is not None:
                parts.append(self._render_part(name, value))

        return "; ".join(parts)

    def render_headers(self) -> str:
        """
        Renders the headers for this request field.
        """
        lines = []

        sort_keys = ["Content-Disposition", "Content-Type", "Content-Location"]
        for sort_key in sort_keys:
            if self.headers.get(sort_key, False):
                lines.append(f"{sort_key}: {self.headers[sort_key]}")

        for header_name, header_value in self.headers.items():
            if header_name not in sort_keys:
                if header_value:
                    lines.append(f"{header_name}: {header_value}")

        lines.append("\r\n")
        return "\r\n".join(lines)

    def make_multipart(
        self,
        content_disposition: str | None = None,
        content_type: str | None = None,
        content_location: str | None = None,
    ) -> None:
        """
        Makes this request field into a multipart request field.

        This method overrides "Content-Disposition", "Content-Type" and
        "Content-Location" headers to the request parameter.

        :param content_disposition:
            The 'Content-Disposition' of the request body. Defaults to 'form-data'
        :param content_type:
            The 'Content-Type' of the request body.
        :param content_location:
            The 'Content-Location' of the request body.

        """
        content_disposition = (content_disposition or "form-data") + "; ".join(
            [
                "",
                self._render_parts(
                    (("name", self._name), ("filename", self._filename))
                ),
            ]
        )

        self.headers["Content-Disposition"] = content_disposition
        self.headers["Content-Type"] = content_type
        self.headers["Content-Location"] = content_location

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\fusion_options.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from argparse import ArgumentParser
from enum import Enum


class AttentionMaskFormat:
    # Build 1D mask indice (sequence length). It requires right side padding! Recommended for BERT model to get best performance.
    MaskIndexEnd = 0

    # For experiment only. Do not use it in production.
    MaskIndexEndAndStart = 1

    # Raw attention mask with 0 means padding (or no attention) and 1 otherwise.
    AttentionMask = 2

    # No attention mask
    NoMask = 3


class AttentionOpType(Enum):
    Attention = "Attention"
    MultiHeadAttention = "MultiHeadAttention"
    GroupQueryAttention = "GroupQueryAttention"
    PagedAttention = "PagedAttention"

    def __str__(self):
        return self.value

    # Override __eq__ to return string comparison
    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        return other.value == self.value


class FusionOptions:
    """Options of fusion in graph optimization"""

    def __init__(self, model_type):
        self.enable_gelu = True
        self.enable_layer_norm = True
        self.enable_attention = True
        self.enable_rotary_embeddings = True

        # Use MultiHeadAttention instead of Attention operator. The difference:
        # (1) Attention has merged weights for Q/K/V projection, which might be faster in some cases since 3 MatMul is
        #     merged into one.
        # (2) Attention could only handle self attention; MultiHeadAttention could handle both self and cross attention.
        self.use_multi_head_attention = False
        self.disable_multi_head_attention_bias = False

        self.enable_skip_layer_norm = True
        self.enable_embed_layer_norm = True
        self.enable_bias_skip_layer_norm = True
        self.enable_bias_gelu = True
        self.enable_gelu_approximation = False
        self.enable_qordered_matmul = True

        self.enable_shape_inference = True
        self.enable_gemm_fast_gelu = False
        self.group_norm_channels_last = True

        if model_type == "clip":
            self.enable_embed_layer_norm = False

        # Set default to sequence length for BERT model to use fused attention to speed up.
        # Note that embed layer normalization will convert 2D mask to 1D when mask type is MaskIndexEnd.
        self.attention_mask_format = AttentionMaskFormat.AttentionMask
        if model_type == "bert":
            self.attention_mask_format = AttentionMaskFormat.MaskIndexEnd
        elif model_type == "vit":
            self.attention_mask_format = AttentionMaskFormat.NoMask

        self.attention_op_type = None

        # options for stable diffusion
        if model_type in ["unet", "vae", "clip"]:
            self.enable_nhwc_conv = True
            self.enable_group_norm = True
            self.enable_skip_group_norm = True
            self.enable_bias_splitgelu = True
            self.enable_packed_qkv = True
            self.enable_packed_kv = True
            self.enable_bias_add = True

    def use_raw_attention_mask(self, use_raw_mask=True):
        if use_raw_mask:
            self.attention_mask_format = AttentionMaskFormat.AttentionMask
        else:
            self.attention_mask_format = AttentionMaskFormat.MaskIndexEnd

    def disable_attention_mask(self):
        self.attention_mask_format = AttentionMaskFormat.NoMask

    def set_attention_op_type(self, attn_op_type: AttentionOpType):
        self.attention_op_type = attn_op_type

    @staticmethod
    def parse(args):
        options = FusionOptions(args.model_type)
        if args.disable_gelu:
            options.enable_gelu = False
        if args.disable_layer_norm:
            options.enable_layer_norm = False
        if args.disable_rotary_embeddings:
            options.enable_rotary_embeddings = False
        if args.disable_attention:
            options.enable_attention = False
        if args.use_multi_head_attention:
            options.use_multi_head_attention = True
        if args.disable_skip_layer_norm:
            options.enable_skip_layer_norm = False
        if args.disable_embed_layer_norm:
            options.enable_embed_layer_norm = False
        if args.disable_bias_skip_layer_norm:
            options.enable_bias_skip_layer_norm = False
        if args.disable_bias_gelu:
            options.enable_bias_gelu = False
        if args.enable_gelu_approximation:
            options.enable_gelu_approximation = True
        if args.disable_shape_inference:
            options.enable_shape_inference = False
        if args.enable_gemm_fast_gelu:
            options.enable_gemm_fast_gelu = True
        if args.use_mask_index:
            options.use_raw_attention_mask(False)
        if args.use_raw_attention_mask:
            options.use_raw_attention_mask(True)
        if args.no_attention_mask:
            options.disable_attention_mask()

        if args.model_type in ["unet", "vae", "clip"]:
            if args.use_group_norm_channels_first:
                options.group_norm_channels_last = False
            if args.disable_nhwc_conv:
                options.enable_nhwc_conv = False
            if args.disable_group_norm:
                options.enable_group_norm = False
            if args.disable_skip_group_norm:
                options.enable_skip_group_norm = False
            if args.disable_bias_splitgelu:
                options.enable_bias_splitgelu = False
            if args.disable_packed_qkv:
                options.enable_packed_qkv = False
            if args.disable_packed_kv:
                options.enable_packed_kv = False
            if args.disable_bias_add:
                options.enable_bias_add = False

        return options

    @staticmethod
    def add_arguments(parser: ArgumentParser):
        parser.add_argument(
            "--disable_attention",
            required=False,
            action="store_true",
            help="disable Attention fusion",
        )
        parser.set_defaults(disable_attention=False)

        parser.add_argument(
            "--disable_skip_layer_norm",
            required=False,
            action="store_true",
            help="disable SkipLayerNormalization fusion",
        )
        parser.set_defaults(disable_skip_layer_norm=False)

        parser.add_argument(
            "--disable_embed_layer_norm",
            required=False,
            action="store_true",
            help="disable EmbedLayerNormalization fusion",
        )
        parser.set_defaults(disable_embed_layer_norm=False)

        parser.add_argument(
            "--disable_bias_skip_layer_norm",
            required=False,
            action="store_true",
            help="disable Add Bias and SkipLayerNormalization fusion",
        )
        parser.set_defaults(disable_bias_skip_layer_norm=False)

        parser.add_argument(
            "--disable_bias_gelu",
            required=False,
            action="store_true",
            help="disable Add Bias and Gelu/FastGelu fusion",
        )
        parser.set_defaults(disable_bias_gelu=False)

        parser.add_argument(
            "--disable_layer_norm",
            required=False,
            action="store_true",
            help="disable LayerNormalization fusion",
        )
        parser.set_defaults(disable_layer_norm=False)

        parser.add_argument(
            "--disable_gelu",
            required=False,
            action="store_true",
            help="disable Gelu fusion",
        )
        parser.set_defaults(disable_gelu=False)

        parser.add_argument(
            "--enable_gelu_approximation",
            required=False,
            action="store_true",
            help="enable Gelu/BiasGelu to FastGelu conversion",
        )
        parser.set_defaults(enable_gelu_approximation=False)

        parser.add_argument(
            "--disable_shape_inference",
            required=False,
            action="store_true",
            help="disable symbolic shape inference",
        )
        parser.set_defaults(disable_shape_inference=False)

        parser.add_argument(
            "--enable_gemm_fast_gelu",
            required=False,
            action="store_true",
            help="enable GemmfastGelu fusion",
        )
        parser.set_defaults(enable_gemm_fast_gelu=False)

        parser.add_argument(
            "--use_mask_index",
            required=False,
            action="store_true",
            help="use mask index to activate fused attention to speed up. It requires right-side padding!",
        )
        parser.set_defaults(use_mask_index=False)

        parser.add_argument(
            "--use_raw_attention_mask",
            required=False,
            action="store_true",
            help="use raw attention mask. Use this option if your input is not right-side padding. This might deactivate fused attention and get worse performance.",
        )
        parser.set_defaults(use_raw_attention_mask=False)

        parser.add_argument(
            "--no_attention_mask",
            required=False,
            action="store_true",
            help="no attention mask. Only works for model_type=bert",
        )
        parser.set_defaults(no_attention_mask=False)

        parser.add_argument(
            "--use_multi_head_attention",
            required=False,
            action="store_true",
            help="Use MultiHeadAttention instead of Attention operator for testing purpose. "
            "Note that MultiHeadAttention might be slower than Attention when qkv are not packed. ",
        )
        parser.set_defaults(use_multi_head_attention=False)

        parser.add_argument(
            "--disable_group_norm",
            required=False,
            action="store_true",
            help="not fuse GroupNorm. Only works for model_type=unet or vae",
        )
        parser.set_defaults(disable_group_norm=False)

        parser.add_argument(
            "--disable_skip_group_norm",
            required=False,
            action="store_true",
            help="not fuse Add + GroupNorm to SkipGroupNorm. Only works for model_type=unet or vae",
        )
        parser.set_defaults(disable_skip_group_norm=False)

        parser.add_argument(
            "--disable_packed_kv",
            required=False,
            action="store_true",
            help="not use packed kv for cross attention in MultiHeadAttention. Only works for model_type=unet",
        )
        parser.set_defaults(disable_packed_kv=False)

        parser.add_argument(
            "--disable_packed_qkv",
            required=False,
            action="store_true",
            help="not use packed qkv for self attention in MultiHeadAttention. Only works for model_type=unet",
        )
        parser.set_defaults(disable_packed_qkv=False)

        parser.add_argument(
            "--disable_bias_add",
            required=False,
            action="store_true",
            help="not fuse BiasAdd. Only works for model_type=unet",
        )
        parser.set_defaults(disable_bias_add=False)

        parser.add_argument(
            "--disable_bias_splitgelu",
            required=False,
            action="store_true",
            help="not fuse BiasSplitGelu. Only works for model_type=unet",
        )
        parser.set_defaults(disable_bias_splitgelu=False)

        parser.add_argument(
            "--disable_nhwc_conv",
            required=False,
            action="store_true",
            help="Do not use NhwcConv. Only works for model_type=unet or vae",
        )
        parser.set_defaults(disable_nhwc_conv=False)

        parser.add_argument(
            "--use_group_norm_channels_first",
            required=False,
            action="store_true",
            help="Use channels_first (NCHW) instead of channels_last (NHWC) for GroupNorm. Only works for model_type=unet or vae",
        )
        parser.set_defaults(use_group_norm_channels_first=False)

        parser.add_argument(
            "--disable_rotary_embeddings",
            required=False,
            action="store_true",
            help="Do not fuse rotary embeddings into RotaryEmbedding op",
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\accessor.py ===
"""

accessor.py contains base classes for implementing accessor properties
that can be mixed into or pinned onto other pandas classes.

"""
from __future__ import annotations

from typing import (
    Callable,
    final,
)
import warnings

from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level


class DirNamesMixin:
    _accessors: set[str] = set()
    _hidden_attrs: frozenset[str] = frozenset()

    @final
    def _dir_deletions(self) -> set[str]:
        """
        Delete unwanted __dir__ for this object.
        """
        return self._accessors | self._hidden_attrs

    def _dir_additions(self) -> set[str]:
        """
        Add additional __dir__ for this object.
        """
        return {accessor for accessor in self._accessors if hasattr(self, accessor)}

    def __dir__(self) -> list[str]:
        """
        Provide method name lookup and completion.

        Notes
        -----
        Only provide 'public' methods.
        """
        rv = set(super().__dir__())
        rv = (rv - self._dir_deletions()) | self._dir_additions()
        return sorted(rv)


class PandasDelegate:
    """
    Abstract base class for delegating methods/properties.
    """

    def _delegate_property_get(self, name: str, *args, **kwargs):
        raise TypeError(f"You cannot access the property {name}")

    def _delegate_property_set(self, name: str, value, *args, **kwargs):
        raise TypeError(f"The property {name} cannot be set")

    def _delegate_method(self, name: str, *args, **kwargs):
        raise TypeError(f"You cannot call method {name}")

    @classmethod
    def _add_delegate_accessors(
        cls,
        delegate,
        accessors: list[str],
        typ: str,
        overwrite: bool = False,
        accessor_mapping: Callable[[str], str] = lambda x: x,
        raise_on_missing: bool = True,
    ) -> None:
        """
        Add accessors to cls from the delegate class.

        Parameters
        ----------
        cls
            Class to add the methods/properties to.
        delegate
            Class to get methods/properties and doc-strings.
        accessors : list of str
            List of accessors to add.
        typ : {'property', 'method'}
        overwrite : bool, default False
            Overwrite the method/property in the target class if it exists.
        accessor_mapping: Callable, default lambda x: x
            Callable to map the delegate's function to the cls' function.
        raise_on_missing: bool, default True
            Raise if an accessor does not exist on delegate.
            False skips the missing accessor.
        """

        def _create_delegator_property(name: str):
            def _getter(self):
                return self._delegate_property_get(name)

            def _setter(self, new_values):
                return self._delegate_property_set(name, new_values)

            _getter.__name__ = name
            _setter.__name__ = name

            return property(
                fget=_getter,
                fset=_setter,
                doc=getattr(delegate, accessor_mapping(name)).__doc__,
            )

        def _create_delegator_method(name: str):
            def f(self, *args, **kwargs):
                return self._delegate_method(name, *args, **kwargs)

            f.__name__ = name
            f.__doc__ = getattr(delegate, accessor_mapping(name)).__doc__

            return f

        for name in accessors:
            if (
                not raise_on_missing
                and getattr(delegate, accessor_mapping(name), None) is None
            ):
                continue

            if typ == "property":
                f = _create_delegator_property(name)
            else:
                f = _create_delegator_method(name)

            # don't overwrite existing methods/properties
            if overwrite or not hasattr(cls, name):
                setattr(cls, name, f)


def delegate_names(
    delegate,
    accessors: list[str],
    typ: str,
    overwrite: bool = False,
    accessor_mapping: Callable[[str], str] = lambda x: x,
    raise_on_missing: bool = True,
):
    """
    Add delegated names to a class using a class decorator.  This provides
    an alternative usage to directly calling `_add_delegate_accessors`
    below a class definition.

    Parameters
    ----------
    delegate : object
        The class to get methods/properties & doc-strings.
    accessors : Sequence[str]
        List of accessor to add.
    typ : {'property', 'method'}
    overwrite : bool, default False
       Overwrite the method/property in the target class if it exists.
    accessor_mapping: Callable, default lambda x: x
        Callable to map the delegate's function to the cls' function.
    raise_on_missing: bool, default True
        Raise if an accessor does not exist on delegate.
        False skips the missing accessor.

    Returns
    -------
    callable
        A class decorator.

    Examples
    --------
    @delegate_names(Categorical, ["categories", "ordered"], "property")
    class CategoricalAccessor(PandasDelegate):
        [...]
    """

    def add_delegate_accessors(cls):
        cls._add_delegate_accessors(
            delegate,
            accessors,
            typ,
            overwrite=overwrite,
            accessor_mapping=accessor_mapping,
            raise_on_missing=raise_on_missing,
        )
        return cls

    return add_delegate_accessors


# Ported with modifications from xarray; licence at LICENSES/XARRAY_LICENSE
# https://github.com/pydata/xarray/blob/master/xarray/core/extensions.py
# 1. We don't need to catch and re-raise AttributeErrors as RuntimeErrors
# 2. We use a UserWarning instead of a custom Warning


class CachedAccessor:
    """
    Custom property-like object.

    A descriptor for caching accessors.

    Parameters
    ----------
    name : str
        Namespace that will be accessed under, e.g. ``df.foo``.
    accessor : cls
        Class with the extension methods.

    Notes
    -----
    For accessor, The class's __init__ method assumes that one of
    ``Series``, ``DataFrame`` or ``Index`` as the
    single argument ``data``.
    """

    def __init__(self, name: str, accessor) -> None:
        self._name = name
        self._accessor = accessor

    def __get__(self, obj, cls):
        if obj is None:
            # we're accessing the attribute of the class, i.e., Dataset.geo
            return self._accessor
        accessor_obj = self._accessor(obj)
        # Replace the property with the accessor object. Inspired by:
        # https://www.pydanny.com/cached-property.html
        # We need to use object.__setattr__ because we overwrite __setattr__ on
        # NDFrame
        object.__setattr__(obj, self._name, accessor_obj)
        return accessor_obj


@doc(klass="", others="")
def _register_accessor(name: str, cls):
    """
    Register a custom accessor on {klass} objects.

    Parameters
    ----------
    name : str
        Name under which the accessor should be registered. A warning is issued
        if this name conflicts with a preexisting attribute.

    Returns
    -------
    callable
        A class decorator.

    See Also
    --------
    register_dataframe_accessor : Register a custom accessor on DataFrame objects.
    register_series_accessor : Register a custom accessor on Series objects.
    register_index_accessor : Register a custom accessor on Index objects.

    Notes
    -----
    When accessed, your accessor will be initialized with the pandas object
    the user is interacting with. So the signature must be

    .. code-block:: python

        def __init__(self, pandas_object):  # noqa: E999
            ...

    For consistency with pandas methods, you should raise an ``AttributeError``
    if the data passed to your accessor has an incorrect dtype.

    >>> pd.Series(['a', 'b']).dt
    Traceback (most recent call last):
    ...
    AttributeError: Can only use .dt accessor with datetimelike values

    Examples
    --------
    In your library code::

        import pandas as pd

        @pd.api.extensions.register_dataframe_accessor("geo")
        class GeoAccessor:
            def __init__(self, pandas_obj):
                self._obj = pandas_obj

            @property
            def center(self):
                # return the geographic center point of this DataFrame
                lat = self._obj.latitude
                lon = self._obj.longitude
                return (float(lon.mean()), float(lat.mean()))

            def plot(self):
                # plot this array's data on a map, e.g., using Cartopy
                pass

    Back in an interactive IPython session:

        .. code-block:: ipython

            In [1]: ds = pd.DataFrame({{"longitude": np.linspace(0, 10),
               ...:                    "latitude": np.linspace(0, 20)}})
            In [2]: ds.geo.center
            Out[2]: (5.0, 10.0)
            In [3]: ds.geo.plot()  # plots data on a map
    """

    def decorator(accessor):
        if hasattr(cls, name):
            warnings.warn(
                f"registration of accessor {repr(accessor)} under name "
                f"{repr(name)} for type {repr(cls)} is overriding a preexisting "
                f"attribute with the same name.",
                UserWarning,
                stacklevel=find_stack_level(),
            )
        setattr(cls, name, CachedAccessor(name, accessor))
        cls._accessors.add(name)
        return accessor

    return decorator


@doc(_register_accessor, klass="DataFrame")
def register_dataframe_accessor(name: str):
    from pandas import DataFrame

    return _register_accessor(name, DataFrame)


@doc(_register_accessor, klass="Series")
def register_series_accessor(name: str):
    from pandas import Series

    return _register_accessor(name, Series)


@doc(_register_accessor, klass="Index")
def register_index_accessor(name: str):
    from pandas import Index

    return _register_accessor(name, Index)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\bidi\network.py ===
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

from selenium.webdriver.common.bidi.common import command_builder


class NetworkEvent:
    """Represents a network event."""

    def __init__(self, event_class, **kwargs):
        self.event_class = event_class
        self.params = kwargs

    @classmethod
    def from_json(cls, json):
        return cls(event_class=json.get("event_class"), **json)


class Network:
    EVENTS = {
        "before_request": "network.beforeRequestSent",
        "response_started": "network.responseStarted",
        "response_completed": "network.responseCompleted",
        "auth_required": "network.authRequired",
        "fetch_error": "network.fetchError",
        "continue_request": "network.continueRequest",
        "continue_auth": "network.continueWithAuth",
    }

    PHASES = {
        "before_request": "beforeRequestSent",
        "response_started": "responseStarted",
        "auth_required": "authRequired",
    }

    def __init__(self, conn):
        self.conn = conn
        self.intercepts = []
        self.callbacks = {}
        self.subscriptions = {}

    def _add_intercept(self, phases=[], contexts=None, url_patterns=None):
        """Add an intercept to the network.

        Parameters:
        ----------
            phases (list, optional): A list of phases to intercept.
                Default is empty list.
            contexts (list, optional): A list of contexts to intercept.
                Default is None.
            url_patterns (list, optional): A list of URL patterns to intercept.
                Default is None.

        Returns:
        -------
            str : intercept id
        """
        params = {}
        if contexts is not None:
            params["contexts"] = contexts
        if url_patterns is not None:
            params["urlPatterns"] = url_patterns
        if len(phases) > 0:
            params["phases"] = phases
        else:
            params["phases"] = ["beforeRequestSent"]
        cmd = command_builder("network.addIntercept", params)

        result = self.conn.execute(cmd)
        self.intercepts.append(result["intercept"])
        return result

    def _remove_intercept(self, intercept=None):
        """Remove a specific intercept, or all intercepts.

        Parameters:
        ----------
            intercept (str, optional): The intercept to remove.
                Default is None.

        Raises:
        ------
            Exception: If intercept is not found.

        Notes:
        -----
            If intercept is None, all intercepts will be removed.
        """
        if intercept is None:
            intercepts_to_remove = self.intercepts.copy()  # create a copy before iterating
            for intercept_id in intercepts_to_remove:  # remove all intercepts
                self.conn.execute(command_builder("network.removeIntercept", {"intercept": intercept_id}))
                self.intercepts.remove(intercept_id)
        else:
            try:
                self.conn.execute(command_builder("network.removeIntercept", {"intercept": intercept}))
                self.intercepts.remove(intercept)
            except Exception as e:
                raise Exception(f"Exception: {e}")

    def _on_request(self, event_name, callback):
        """Set a callback function to subscribe to a network event.

        Parameters:
        ----------
            event_name (str): The event to subscribe to.
            callback (function): The callback function to execute on event.
                Takes Request object as argument.

        Returns:
        -------
            int : callback id
        """

        event = NetworkEvent(event_name)

        def _callback(event_data):
            request = Request(
                network=self,
                request_id=event_data.params["request"].get("request", None),
                body_size=event_data.params["request"].get("bodySize", None),
                cookies=event_data.params["request"].get("cookies", None),
                resource_type=event_data.params["request"].get("goog:resourceType", None),
                headers=event_data.params["request"].get("headers", None),
                headers_size=event_data.params["request"].get("headersSize", None),
                timings=event_data.params["request"].get("timings", None),
                url=event_data.params["request"].get("url", None),
            )
            callback(request)

        callback_id = self.conn.add_callback(event, _callback)

        if event_name in self.callbacks:
            self.callbacks[event_name].append(callback_id)
        else:
            self.callbacks[event_name] = [callback_id]

        return callback_id

    def add_request_handler(self, event, callback, url_patterns=None, contexts=None):
        """Add a request handler to the network.

        Parameters:
        ----------
            event (str): The event to subscribe to.
            url_patterns (list, optional): A list of URL patterns to intercept.
                Default is None.
            contexts (list, optional): A list of contexts to intercept.
                Default is None.
            callback (function): The callback function to execute on request interception
                Takes Request object as argument.

        Returns:
        -------
            int : callback id
        """

        try:
            event_name = self.EVENTS[event]
            phase_name = self.PHASES[event]
        except KeyError:
            raise Exception(f"Event {event} not found")

        result = self._add_intercept(phases=[phase_name], url_patterns=url_patterns, contexts=contexts)
        callback_id = self._on_request(event_name, callback)

        if event_name in self.subscriptions:
            self.subscriptions[event_name].append(callback_id)
        else:
            params = {}
            params["events"] = [event_name]
            self.conn.execute(command_builder("session.subscribe", params))
            self.subscriptions[event_name] = [callback_id]

        self.callbacks[callback_id] = result["intercept"]
        return callback_id

    def remove_request_handler(self, event, callback_id):
        """Remove a request handler from the network.

        Parameters:
        ----------
            event_name (str): The event to unsubscribe from.
            callback_id (int): The callback id to remove.
        """
        try:
            event_name = self.EVENTS[event]
        except KeyError:
            raise Exception(f"Event {event} not found")

        net_event = NetworkEvent(event_name)

        self.conn.remove_callback(net_event, callback_id)
        self._remove_intercept(self.callbacks[callback_id])
        del self.callbacks[callback_id]
        self.subscriptions[event_name].remove(callback_id)
        if len(self.subscriptions[event_name]) == 0:
            params = {}
            params["events"] = [event_name]
            self.conn.execute(command_builder("session.unsubscribe", params))
            del self.subscriptions[event_name]

    def clear_request_handlers(self):
        """Clear all request handlers from the network."""

        for event_name in self.subscriptions:
            net_event = NetworkEvent(event_name)
            for callback_id in self.subscriptions[event_name]:
                self.conn.remove_callback(net_event, callback_id)
                self._remove_intercept(self.callbacks[callback_id])
                del self.callbacks[callback_id]
            params = {}
            params["events"] = [event_name]
            self.conn.execute(command_builder("session.unsubscribe", params))
        self.subscriptions = {}

    def add_auth_handler(self, username, password):
        """Add an authentication handler to the network.

        Parameters:
        ----------
            username (str): The username to authenticate with.
            password (str): The password to authenticate with.

        Returns:
        -------
            int : callback id
        """
        event = "auth_required"

        def _callback(request):
            request._continue_with_auth(username, password)

        return self.add_request_handler(event, _callback)

    def remove_auth_handler(self, callback_id):
        """Remove an authentication handler from the network.

        Parameters:
        ----------
            callback_id (int): The callback id to remove.
        """
        event = "auth_required"
        self.remove_request_handler(event, callback_id)


class Request:
    """Represents an intercepted network request."""

    def __init__(
        self,
        network: Network,
        request_id,
        body_size=None,
        cookies=None,
        resource_type=None,
        headers=None,
        headers_size=None,
        method=None,
        timings=None,
        url=None,
    ):
        self.network = network
        self.request_id = request_id
        self.body_size = body_size
        self.cookies = cookies
        self.resource_type = resource_type
        self.headers = headers
        self.headers_size = headers_size
        self.method = method
        self.timings = timings
        self.url = url

    def fail_request(self):
        """Fail this request."""

        if not self.request_id:
            raise ValueError("Request not found.")

        params = {"request": self.request_id}
        self.network.conn.execute(command_builder("network.failRequest", params))

    def continue_request(self, body=None, method=None, headers=None, cookies=None, url=None):
        """Continue after intercepting this request."""

        if not self.request_id:
            raise ValueError("Request not found.")

        params = {"request": self.request_id}
        if body is not None:
            params["body"] = body
        if method is not None:
            params["method"] = method
        if headers is not None:
            params["headers"] = headers
        if cookies is not None:
            params["cookies"] = cookies
        if url is not None:
            params["url"] = url

        self.network.conn.execute(command_builder("network.continueRequest", params))

    def _continue_with_auth(self, username=None, password=None):
        """Continue with authentication.

        Parameters:
        ----------
            request (Request): The request to continue with.
            username (str): The username to authenticate with.
            password (str): The password to authenticate with.

        Notes:
        -----
            If username or password is None, it attempts auth with no credentials
        """

        params = {}
        params["request"] = self.request_id

        if not username or not password:  # no credentials is valid option
            params["action"] = "default"
        else:
            params["action"] = "provideCredentials"
            params["credentials"] = {"type": "password", "username": username, "password": password}

        self.network.conn.execute(command_builder("network.continueWithAuth", params))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\logic\inference.py ===
"""Inference in propositional logic"""

from sympy.logic.boolalg import And, Not, conjuncts, to_cnf, BooleanFunction
from sympy.core.sorting import ordered
from sympy.core.sympify import sympify
from sympy.external.importtools import import_module


def literal_symbol(literal):
    """
    The symbol in this literal (without the negation).

    Examples
    ========

    >>> from sympy.abc import A
    >>> from sympy.logic.inference import literal_symbol
    >>> literal_symbol(A)
    A
    >>> literal_symbol(~A)
    A

    """

    if literal is True or literal is False:
        return literal
    elif literal.is_Symbol:
        return literal
    elif literal.is_Not:
        return literal_symbol(literal.args[0])
    else:
        raise ValueError("Argument must be a boolean literal.")


def satisfiable(expr, algorithm=None, all_models=False, minimal=False, use_lra_theory=False):
    """
    Check satisfiability of a propositional sentence.
    Returns a model when it succeeds.
    Returns {true: true} for trivially true expressions.

    On setting all_models to True, if given expr is satisfiable then
    returns a generator of models. However, if expr is unsatisfiable
    then returns a generator containing the single element False.

    Examples
    ========

    >>> from sympy.abc import A, B
    >>> from sympy.logic.inference import satisfiable
    >>> satisfiable(A & ~B)
    {A: True, B: False}
    >>> satisfiable(A & ~A)
    False
    >>> satisfiable(True)
    {True: True}
    >>> next(satisfiable(A & ~A, all_models=True))
    False
    >>> models = satisfiable((A >> B) & B, all_models=True)
    >>> next(models)
    {A: False, B: True}
    >>> next(models)
    {A: True, B: True}
    >>> def use_models(models):
    ...     for model in models:
    ...         if model:
    ...             # Do something with the model.
    ...             print(model)
    ...         else:
    ...             # Given expr is unsatisfiable.
    ...             print("UNSAT")
    >>> use_models(satisfiable(A >> ~A, all_models=True))
    {A: False}
    >>> use_models(satisfiable(A ^ A, all_models=True))
    UNSAT

    """
    if use_lra_theory:
        if algorithm is not None and algorithm != "dpll2":
            raise ValueError(f"Currently only dpll2 can handle using lra theory. {algorithm} is not handled.")
        algorithm = "dpll2"

    if algorithm is None or algorithm == "pycosat":
        pycosat = import_module('pycosat')
        if pycosat is not None:
            algorithm = "pycosat"
        else:
            if algorithm == "pycosat":
                raise ImportError("pycosat module is not present")
            # Silently fall back to dpll2 if pycosat
            # is not installed
            algorithm = "dpll2"

    if algorithm=="minisat22":
        pysat = import_module('pysat')
        if pysat is None:
            algorithm = "dpll2"

    if algorithm=="z3":
        z3 = import_module('z3')
        if z3 is None:
            algorithm = "dpll2"

    if algorithm == "dpll":
        from sympy.logic.algorithms.dpll import dpll_satisfiable
        return dpll_satisfiable(expr)
    elif algorithm == "dpll2":
        from sympy.logic.algorithms.dpll2 import dpll_satisfiable
        return dpll_satisfiable(expr, all_models, use_lra_theory=use_lra_theory)
    elif algorithm == "pycosat":
        from sympy.logic.algorithms.pycosat_wrapper import pycosat_satisfiable
        return pycosat_satisfiable(expr, all_models)
    elif algorithm == "minisat22":
        from sympy.logic.algorithms.minisat22_wrapper import minisat22_satisfiable
        return minisat22_satisfiable(expr, all_models, minimal)
    elif algorithm == "z3":
        from sympy.logic.algorithms.z3_wrapper import z3_satisfiable
        return z3_satisfiable(expr, all_models)

    raise NotImplementedError


def valid(expr):
    """
    Check validity of a propositional sentence.
    A valid propositional sentence is True under every assignment.

    Examples
    ========

    >>> from sympy.abc import A, B
    >>> from sympy.logic.inference import valid
    >>> valid(A | ~A)
    True
    >>> valid(A | B)
    False

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Validity

    """
    return not satisfiable(Not(expr))


def pl_true(expr, model=None, deep=False):
    """
    Returns whether the given assignment is a model or not.

    If the assignment does not specify the value for every proposition,
    this may return None to indicate 'not obvious'.

    Parameters
    ==========

    model : dict, optional, default: {}
        Mapping of symbols to boolean values to indicate assignment.
    deep: boolean, optional, default: False
        Gives the value of the expression under partial assignments
        correctly. May still return None to indicate 'not obvious'.


    Examples
    ========

    >>> from sympy.abc import A, B
    >>> from sympy.logic.inference import pl_true
    >>> pl_true( A & B, {A: True, B: True})
    True
    >>> pl_true(A & B, {A: False})
    False
    >>> pl_true(A & B, {A: True})
    >>> pl_true(A & B, {A: True}, deep=True)
    >>> pl_true(A >> (B >> A))
    >>> pl_true(A >> (B >> A), deep=True)
    True
    >>> pl_true(A & ~A)
    >>> pl_true(A & ~A, deep=True)
    False
    >>> pl_true(A & B & (~A | ~B), {A: True})
    >>> pl_true(A & B & (~A | ~B), {A: True}, deep=True)
    False

    """

    from sympy.core.symbol import Symbol

    boolean = (True, False)

    def _validate(expr):
        if isinstance(expr, Symbol) or expr in boolean:
            return True
        if not isinstance(expr, BooleanFunction):
            return False
        return all(_validate(arg) for arg in expr.args)

    if expr in boolean:
        return expr
    expr = sympify(expr)
    if not _validate(expr):
        raise ValueError("%s is not a valid boolean expression" % expr)
    if not model:
        model = {}
    model = {k: v for k, v in model.items() if v in boolean}
    result = expr.subs(model)
    if result in boolean:
        return bool(result)
    if deep:
        model = dict.fromkeys(result.atoms(), True)
        if pl_true(result, model):
            if valid(result):
                return True
        else:
            if not satisfiable(result):
                return False
    return None


def entails(expr, formula_set=None):
    """
    Check whether the given expr_set entail an expr.
    If formula_set is empty then it returns the validity of expr.

    Examples
    ========

    >>> from sympy.abc import A, B, C
    >>> from sympy.logic.inference import entails
    >>> entails(A, [A >> B, B >> C])
    False
    >>> entails(C, [A >> B, B >> C, A])
    True
    >>> entails(A >> B)
    False
    >>> entails(A >> (B >> A))
    True

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Logical_consequence

    """
    if formula_set:
        formula_set = list(formula_set)
    else:
        formula_set = []
    formula_set.append(Not(expr))
    return not satisfiable(And(*formula_set))


class KB:
    """Base class for all knowledge bases"""
    def __init__(self, sentence=None):
        self.clauses_ = set()
        if sentence:
            self.tell(sentence)

    def tell(self, sentence):
        raise NotImplementedError

    def ask(self, query):
        raise NotImplementedError

    def retract(self, sentence):
        raise NotImplementedError

    @property
    def clauses(self):
        return list(ordered(self.clauses_))


class PropKB(KB):
    """A KB for Propositional Logic.  Inefficient, with no indexing."""

    def tell(self, sentence):
        """Add the sentence's clauses to the KB

        Examples
        ========

        >>> from sympy.logic.inference import PropKB
        >>> from sympy.abc import x, y
        >>> l = PropKB()
        >>> l.clauses
        []

        >>> l.tell(x | y)
        >>> l.clauses
        [x | y]

        >>> l.tell(y)
        >>> l.clauses
        [y, x | y]

        """
        for c in conjuncts(to_cnf(sentence)):
            self.clauses_.add(c)

    def ask(self, query):
        """Checks if the query is true given the set of clauses.

        Examples
        ========

        >>> from sympy.logic.inference import PropKB
        >>> from sympy.abc import x, y
        >>> l = PropKB()
        >>> l.tell(x & ~y)
        >>> l.ask(x)
        True
        >>> l.ask(y)
        False

        """
        return entails(query, self.clauses_)

    def retract(self, sentence):
        """Remove the sentence's clauses from the KB

        Examples
        ========

        >>> from sympy.logic.inference import PropKB
        >>> from sympy.abc import x, y
        >>> l = PropKB()
        >>> l.clauses
        []

        >>> l.tell(x | y)
        >>> l.clauses
        [x | y]

        >>> l.retract(x | y)
        >>> l.clauses
        []

        """
        for c in conjuncts(to_cnf(sentence)):
            self.clauses_.discard(c)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\optics\waves.py ===
"""
This module has all the classes and functions related to waves in optics.

**Contains**

* TWave
"""

__all__ = ['TWave']

from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.function import Derivative, Function
from sympy.core.numbers import (Number, pi, I)
from sympy.core.singleton import S
from sympy.core.symbol import (Symbol, symbols)
from sympy.core.sympify import _sympify, sympify
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (atan2, cos, sin)
from sympy.physics.units import speed_of_light, meter, second


c = speed_of_light.convert_to(meter/second)


class TWave(Expr):

    r"""
    This is a simple transverse sine wave travelling in a one-dimensional space.
    Basic properties are required at the time of creation of the object,
    but they can be changed later with respective methods provided.

    Explanation
    ===========

    It is represented as :math:`A \times cos(k*x - \omega \times t + \phi )`,
    where :math:`A` is the amplitude, :math:`\omega` is the angular frequency,
    :math:`k` is the wavenumber (spatial frequency), :math:`x` is a spatial variable
    to represent the position on the dimension on which the wave propagates,
    and :math:`\phi` is the phase angle of the wave.


    Arguments
    =========

    amplitude : Sympifyable
        Amplitude of the wave.
    frequency : Sympifyable
        Frequency of the wave.
    phase : Sympifyable
        Phase angle of the wave.
    time_period : Sympifyable
        Time period of the wave.
    n : Sympifyable
        Refractive index of the medium.

    Raises
    =======

    ValueError : When neither frequency nor time period is provided
        or they are not consistent.
    TypeError : When anything other than TWave objects is added.


    Examples
    ========

    >>> from sympy import symbols
    >>> from sympy.physics.optics import TWave
    >>> A1, phi1, A2, phi2, f = symbols('A1, phi1, A2, phi2, f')
    >>> w1 = TWave(A1, f, phi1)
    >>> w2 = TWave(A2, f, phi2)
    >>> w3 = w1 + w2  # Superposition of two waves
    >>> w3
    TWave(sqrt(A1**2 + 2*A1*A2*cos(phi1 - phi2) + A2**2), f,
        atan2(A1*sin(phi1) + A2*sin(phi2), A1*cos(phi1) + A2*cos(phi2)), 1/f, n)
    >>> w3.amplitude
    sqrt(A1**2 + 2*A1*A2*cos(phi1 - phi2) + A2**2)
    >>> w3.phase
    atan2(A1*sin(phi1) + A2*sin(phi2), A1*cos(phi1) + A2*cos(phi2))
    >>> w3.speed
    299792458*meter/(second*n)
    >>> w3.angular_velocity
    2*pi*f

    """

    def __new__(
            cls,
            amplitude,
            frequency=None,
            phase=S.Zero,
            time_period=None,
            n=Symbol('n')):
        if time_period is not None:
            time_period = _sympify(time_period)
            _frequency = S.One/time_period
        if frequency is not None:
            frequency = _sympify(frequency)
            _time_period = S.One/frequency
            if time_period is not None:
                if frequency != S.One/time_period:
                    raise ValueError("frequency and time_period should be consistent.")
        if frequency is None and time_period is None:
            raise ValueError("Either frequency or time period is needed.")
        if frequency is None:
            frequency = _frequency
        if time_period is None:
            time_period = _time_period

        amplitude = _sympify(amplitude)
        phase = _sympify(phase)
        n = sympify(n)
        obj = Basic.__new__(cls, amplitude, frequency, phase, time_period, n)
        return obj

    @property
    def amplitude(self):
        """
        Returns the amplitude of the wave.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.optics import TWave
        >>> A, phi, f = symbols('A, phi, f')
        >>> w = TWave(A, f, phi)
        >>> w.amplitude
        A
        """
        return self.args[0]

    @property
    def frequency(self):
        """
        Returns the frequency of the wave,
        in cycles per second.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.optics import TWave
        >>> A, phi, f = symbols('A, phi, f')
        >>> w = TWave(A, f, phi)
        >>> w.frequency
        f
        """
        return self.args[1]

    @property
    def phase(self):
        """
        Returns the phase angle of the wave,
        in radians.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.optics import TWave
        >>> A, phi, f = symbols('A, phi, f')
        >>> w = TWave(A, f, phi)
        >>> w.phase
        phi
        """
        return self.args[2]

    @property
    def time_period(self):
        """
        Returns the temporal period of the wave,
        in seconds per cycle.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.optics import TWave
        >>> A, phi, f = symbols('A, phi, f')
        >>> w = TWave(A, f, phi)
        >>> w.time_period
        1/f
        """
        return self.args[3]

    @property
    def n(self):
        """
        Returns the refractive index of the medium
        """
        return self.args[4]

    @property
    def wavelength(self):
        """
        Returns the wavelength (spatial period) of the wave,
        in meters per cycle.
        It depends on the medium of the wave.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.optics import TWave
        >>> A, phi, f = symbols('A, phi, f')
        >>> w = TWave(A, f, phi)
        >>> w.wavelength
        299792458*meter/(second*f*n)
        """
        return c/(self.frequency*self.n)


    @property
    def speed(self):
        """
        Returns the propagation speed of the wave,
        in meters per second.
        It is dependent on the propagation medium.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.optics import TWave
        >>> A, phi, f = symbols('A, phi, f')
        >>> w = TWave(A, f, phi)
        >>> w.speed
        299792458*meter/(second*n)
        """
        return self.wavelength*self.frequency

    @property
    def angular_velocity(self):
        """
        Returns the angular velocity of the wave,
        in radians per second.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.optics import TWave
        >>> A, phi, f = symbols('A, phi, f')
        >>> w = TWave(A, f, phi)
        >>> w.angular_velocity
        2*pi*f
        """
        return 2*pi*self.frequency

    @property
    def wavenumber(self):
        """
        Returns the wavenumber of the wave,
        in radians per meter.

        Examples
        ========

        >>> from sympy import symbols
        >>> from sympy.physics.optics import TWave
        >>> A, phi, f = symbols('A, phi, f')
        >>> w = TWave(A, f, phi)
        >>> w.wavenumber
        pi*second*f*n/(149896229*meter)
        """
        return 2*pi/self.wavelength

    def __str__(self):
        """String representation of a TWave."""
        from sympy.printing import sstr
        return type(self).__name__ + sstr(self.args)

    __repr__ = __str__

    def __add__(self, other):
        """
        Addition of two waves will result in their superposition.
        The type of interference will depend on their phase angles.
        """
        if isinstance(other, TWave):
            if self.frequency == other.frequency and self.wavelength == other.wavelength:
                return TWave(sqrt(self.amplitude**2 + other.amplitude**2 + 2 *
                                  self.amplitude*other.amplitude*cos(
                                      self.phase - other.phase)),
                             self.frequency,
                             atan2(self.amplitude*sin(self.phase)
                             + other.amplitude*sin(other.phase),
                             self.amplitude*cos(self.phase)
                             + other.amplitude*cos(other.phase))
                             )
            else:
                raise NotImplementedError("Interference of waves with different frequencies"
                    " has not been implemented.")
        else:
            raise TypeError(type(other).__name__ + " and TWave objects cannot be added.")

    def __mul__(self, other):
        """
        Multiplying a wave by a scalar rescales the amplitude of the wave.
        """
        other = sympify(other)
        if isinstance(other, Number):
            return TWave(self.amplitude*other, *self.args[1:])
        else:
            raise TypeError(type(other).__name__ + " and TWave objects cannot be multiplied.")

    def __sub__(self, other):
        return self.__add__(-1*other)

    def __neg__(self):
        return self.__mul__(-1)

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rsub__(self, other):
        return (-self).__radd__(other)

    def _eval_rewrite_as_sin(self, *args, **kwargs):
        return self.amplitude*sin(self.wavenumber*Symbol('x')
            - self.angular_velocity*Symbol('t') + self.phase + pi/2, evaluate=False)

    def _eval_rewrite_as_cos(self, *args, **kwargs):
        return self.amplitude*cos(self.wavenumber*Symbol('x')
            - self.angular_velocity*Symbol('t') + self.phase)

    def _eval_rewrite_as_pde(self, *args, **kwargs):
        mu, epsilon, x, t = symbols('mu, epsilon, x, t')
        E = Function('E')
        return Derivative(E(x, t), x, 2) + mu*epsilon*Derivative(E(x, t), t, 2)

    def _eval_rewrite_as_exp(self, *args, **kwargs):
        return self.amplitude*exp(I*(self.wavenumber*Symbol('x')
            - self.angular_velocity*Symbol('t') + self.phase))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\specialpolys.py ===
"""Functions for generating interesting polynomials, e.g. for benchmarking. """


from sympy.core import Add, Mul, Symbol, sympify, Dummy, symbols
from sympy.core.containers import Tuple
from sympy.core.singleton import S
from sympy.ntheory import nextprime
from sympy.polys.densearith import (
    dmp_add_term, dmp_neg, dmp_mul, dmp_sqr
)
from sympy.polys.densebasic import (
    dmp_zero, dmp_one, dmp_ground,
    dup_from_raw_dict, dmp_raise, dup_random
)
from sympy.polys.domains import ZZ
from sympy.polys.factortools import dup_zz_cyclotomic_poly
from sympy.polys.polyclasses import DMP
from sympy.polys.polytools import Poly, PurePoly
from sympy.polys.polyutils import _analyze_gens
from sympy.utilities import subsets, public, filldedent


@public
def swinnerton_dyer_poly(n, x=None, polys=False):
    """Generates n-th Swinnerton-Dyer polynomial in `x`.

    Parameters
    ----------
    n : int
        `n` decides the order of polynomial
    x : optional
    polys : bool, optional
        ``polys=True`` returns an expression, otherwise
        (default) returns an expression.
    """
    if n <= 0:
        raise ValueError(
            "Cannot generate Swinnerton-Dyer polynomial of order %s" % n)

    if x is not None:
        sympify(x)
    else:
        x = Dummy('x')

    if n > 3:
        from sympy.functions.elementary.miscellaneous import sqrt
        from .numberfields import minimal_polynomial
        p = 2
        a = [sqrt(2)]
        for i in range(2, n + 1):
            p = nextprime(p)
            a.append(sqrt(p))
        return minimal_polynomial(Add(*a), x, polys=polys)

    if n == 1:
        ex = x**2 - 2
    elif n == 2:
        ex = x**4 - 10*x**2 + 1
    elif n == 3:
        ex = x**8 - 40*x**6 + 352*x**4 - 960*x**2 + 576

    return PurePoly(ex, x) if polys else ex


@public
def cyclotomic_poly(n, x=None, polys=False):
    """Generates cyclotomic polynomial of order `n` in `x`.

    Parameters
    ----------
    n : int
        `n` decides the order of polynomial
    x : optional
    polys : bool, optional
        ``polys=True`` returns an expression, otherwise
        (default) returns an expression.
    """
    if n <= 0:
        raise ValueError(
            "Cannot generate cyclotomic polynomial of order %s" % n)

    poly = DMP(dup_zz_cyclotomic_poly(int(n), ZZ), ZZ)

    if x is not None:
        poly = Poly.new(poly, x)
    else:
        poly = PurePoly.new(poly, Dummy('x'))

    return poly if polys else poly.as_expr()


@public
def symmetric_poly(n, *gens, polys=False):
    """
    Generates symmetric polynomial of order `n`.

    Parameters
    ==========

    polys: bool, optional (default: False)
        Returns a Poly object when ``polys=True``, otherwise
        (default) returns an expression.
    """
    gens = _analyze_gens(gens)

    if n < 0 or n > len(gens) or not gens:
        raise ValueError("Cannot generate symmetric polynomial of order %s for %s" % (n, gens))
    elif not n:
        poly = S.One
    else:
        poly = Add(*[Mul(*s) for s in subsets(gens, int(n))])

    return Poly(poly, *gens) if polys else poly


@public
def random_poly(x, n, inf, sup, domain=ZZ, polys=False):
    """Generates a polynomial of degree ``n`` with coefficients in
    ``[inf, sup]``.

    Parameters
    ----------
    x
        `x` is the independent term of polynomial
    n : int
        `n` decides the order of polynomial
    inf
        Lower limit of range in which coefficients lie
    sup
        Upper limit of range in which coefficients lie
    domain : optional
         Decides what ring the coefficients are supposed
         to belong. Default is set to Integers.
    polys : bool, optional
        ``polys=True`` returns an expression, otherwise
        (default) returns an expression.
    """
    poly = Poly(dup_random(n, inf, sup, domain), x, domain=domain)

    return poly if polys else poly.as_expr()


@public
def interpolating_poly(n, x, X='x', Y='y'):
    """Construct Lagrange interpolating polynomial for ``n``
    data points. If a sequence of values are given for ``X`` and ``Y``
    then the first ``n`` values will be used.
    """
    ok = getattr(x, 'free_symbols', None)

    if isinstance(X, str):
        X = symbols("%s:%s" % (X, n))
    elif ok and ok & Tuple(*X).free_symbols:
        ok = False

    if isinstance(Y, str):
        Y = symbols("%s:%s" % (Y, n))
    elif ok and ok & Tuple(*Y).free_symbols:
        ok = False

    if not ok:
        raise ValueError(filldedent('''
            Expecting symbol for x that does not appear in X or Y.
            Use `interpolate(list(zip(X, Y)), x)` instead.'''))

    coeffs = []
    numert = Mul(*[x - X[i] for i in range(n)])

    for i in range(n):
        numer = numert/(x - X[i])
        denom = Mul(*[(X[i] - X[j]) for j in range(n) if i != j])
        coeffs.append(numer/denom)

    return Add(*[coeff*y for coeff, y in zip(coeffs, Y)])


def fateman_poly_F_1(n):
    """Fateman's GCD benchmark: trivial GCD """
    Y = [Symbol('y_' + str(i)) for i in range(n + 1)]

    y_0, y_1 = Y[0], Y[1]

    u = y_0 + Add(*Y[1:])
    v = y_0**2 + Add(*[y**2 for y in Y[1:]])

    F = ((u + 1)*(u + 2)).as_poly(*Y)
    G = ((v + 1)*(-3*y_1*y_0**2 + y_1**2 - 1)).as_poly(*Y)

    H = Poly(1, *Y)

    return F, G, H


def dmp_fateman_poly_F_1(n, K):
    """Fateman's GCD benchmark: trivial GCD """
    u = [K(1), K(0)]

    for i in range(n):
        u = [dmp_one(i, K), u]

    v = [K(1), K(0), K(0)]

    for i in range(0, n):
        v = [dmp_one(i, K), dmp_zero(i), v]

    m = n - 1

    U = dmp_add_term(u, dmp_ground(K(1), m), 0, n, K)
    V = dmp_add_term(u, dmp_ground(K(2), m), 0, n, K)

    f = [[-K(3), K(0)], [], [K(1), K(0), -K(1)]]

    W = dmp_add_term(v, dmp_ground(K(1), m), 0, n, K)
    Y = dmp_raise(f, m, 1, K)

    F = dmp_mul(U, V, n, K)
    G = dmp_mul(W, Y, n, K)

    H = dmp_one(n, K)

    return F, G, H


def fateman_poly_F_2(n):
    """Fateman's GCD benchmark: linearly dense quartic inputs """
    Y = [Symbol('y_' + str(i)) for i in range(n + 1)]

    y_0 = Y[0]

    u = Add(*Y[1:])

    H = Poly((y_0 + u + 1)**2, *Y)

    F = Poly((y_0 - u - 2)**2, *Y)
    G = Poly((y_0 + u + 2)**2, *Y)

    return H*F, H*G, H


def dmp_fateman_poly_F_2(n, K):
    """Fateman's GCD benchmark: linearly dense quartic inputs """
    u = [K(1), K(0)]

    for i in range(n - 1):
        u = [dmp_one(i, K), u]

    m = n - 1

    v = dmp_add_term(u, dmp_ground(K(2), m - 1), 0, n, K)

    f = dmp_sqr([dmp_one(m, K), dmp_neg(v, m, K)], n, K)
    g = dmp_sqr([dmp_one(m, K), v], n, K)

    v = dmp_add_term(u, dmp_one(m - 1, K), 0, n, K)

    h = dmp_sqr([dmp_one(m, K), v], n, K)

    return dmp_mul(f, h, n, K), dmp_mul(g, h, n, K), h


def fateman_poly_F_3(n):
    """Fateman's GCD benchmark: sparse inputs (deg f ~ vars f) """
    Y = [Symbol('y_' + str(i)) for i in range(n + 1)]

    y_0 = Y[0]

    u = Add(*[y**(n + 1) for y in Y[1:]])

    H = Poly((y_0**(n + 1) + u + 1)**2, *Y)

    F = Poly((y_0**(n + 1) - u - 2)**2, *Y)
    G = Poly((y_0**(n + 1) + u + 2)**2, *Y)

    return H*F, H*G, H


def dmp_fateman_poly_F_3(n, K):
    """Fateman's GCD benchmark: sparse inputs (deg f ~ vars f) """
    u = dup_from_raw_dict({n + 1: K.one}, K)

    for i in range(0, n - 1):
        u = dmp_add_term([u], dmp_one(i, K), n + 1, i + 1, K)

    v = dmp_add_term(u, dmp_ground(K(2), n - 2), 0, n, K)

    f = dmp_sqr(
        dmp_add_term([dmp_neg(v, n - 1, K)], dmp_one(n - 1, K), n + 1, n, K), n, K)
    g = dmp_sqr(dmp_add_term([v], dmp_one(n - 1, K), n + 1, n, K), n, K)

    v = dmp_add_term(u, dmp_one(n - 2, K), 0, n - 1, K)

    h = dmp_sqr(dmp_add_term([v], dmp_one(n - 1, K), n + 1, n, K), n, K)

    return dmp_mul(f, h, n, K), dmp_mul(g, h, n, K), h

# A few useful polynomials from Wang's paper ('78).

from sympy.polys.rings import ring

def _f_0():
    R, x, y, z = ring("x,y,z", ZZ)
    return x**2*y*z**2 + 2*x**2*y*z + 3*x**2*y + 2*x**2 + 3*x + 4*y**2*z**2 + 5*y**2*z + 6*y**2 + y*z**2 + 2*y*z + y + 1

def _f_1():
    R, x, y, z = ring("x,y,z", ZZ)
    return x**3*y*z + x**2*y**2*z**2 + x**2*y**2 + 20*x**2*y*z + 30*x**2*y + x**2*z**2 + 10*x**2*z + x*y**3*z + 30*x*y**2*z + 20*x*y**2 + x*y*z**3 + 10*x*y*z**2 + x*y*z + 610*x*y + 20*x*z**2 + 230*x*z + 300*x + y**2*z**2 + 10*y**2*z + 30*y*z**2 + 320*y*z + 200*y + 600*z + 6000

def _f_2():
    R, x, y, z = ring("x,y,z", ZZ)
    return x**5*y**3 + x**5*y**2*z + x**5*y*z**2 + x**5*z**3 + x**3*y**2 + x**3*y*z + 90*x**3*y + 90*x**3*z + x**2*y**2*z - 11*x**2*y**2 + x**2*z**3 - 11*x**2*z**2 + y*z - 11*y + 90*z - 990

def _f_3():
    R, x, y, z = ring("x,y,z", ZZ)
    return x**5*y**2 + x**4*z**4 + x**4 + x**3*y**3*z + x**3*z + x**2*y**4 + x**2*y**3*z**3 + x**2*y*z**5 + x**2*y*z + x*y**2*z**4 + x*y**2 + x*y*z**7 + x*y*z**3 + x*y*z**2 + y**2*z + y*z**4

def _f_4():
    R, x, y, z = ring("x,y,z", ZZ)
    return -x**9*y**8*z - x**8*y**5*z**3 - x**7*y**12*z**2 - 5*x**7*y**8 - x**6*y**9*z**4 + x**6*y**7*z**3 + 3*x**6*y**7*z - 5*x**6*y**5*z**2 - x**6*y**4*z**3 + x**5*y**4*z**5 + 3*x**5*y**4*z**3 - x**5*y*z**5 + x**4*y**11*z**4 + 3*x**4*y**11*z**2 - x**4*y**8*z**4 + 5*x**4*y**7*z**2 + 15*x**4*y**7 - 5*x**4*y**4*z**2 + x**3*y**8*z**6 + 3*x**3*y**8*z**4 - x**3*y**5*z**6 + 5*x**3*y**4*z**4 + 15*x**3*y**4*z**2 + x**3*y**3*z**5 + 3*x**3*y**3*z**3 - 5*x**3*y*z**4 + x**2*z**7 + 3*x**2*z**5 + x*y**7*z**6 + 3*x*y**7*z**4 + 5*x*y**3*z**4 + 15*x*y**3*z**2 + y**4*z**8 + 3*y**4*z**6 + 5*z**6 + 15*z**4

def _f_5():
    R, x, y, z = ring("x,y,z", ZZ)
    return -x**3 - 3*x**2*y + 3*x**2*z - 3*x*y**2 + 6*x*y*z - 3*x*z**2 - y**3 + 3*y**2*z - 3*y*z**2 + z**3

def _f_6():
    R, x, y, z, t = ring("x,y,z,t", ZZ)
    return 2115*x**4*y + 45*x**3*z**3*t**2 - 45*x**3*t**2 - 423*x*y**4 - 47*x*y**3 + 141*x*y*z**3 + 94*x*y*z*t - 9*y**3*z**3*t**2 + 9*y**3*t**2 - y**2*z**3*t**2 + y**2*t**2 + 3*z**6*t**2 + 2*z**4*t**3 - 3*z**3*t**2 - 2*z*t**3

def _w_1():
    R, x, y, z = ring("x,y,z", ZZ)
    return 4*x**6*y**4*z**2 + 4*x**6*y**3*z**3 - 4*x**6*y**2*z**4 - 4*x**6*y*z**5 + x**5*y**4*z**3 + 12*x**5*y**3*z - x**5*y**2*z**5 + 12*x**5*y**2*z**2 - 12*x**5*y*z**3 - 12*x**5*z**4 + 8*x**4*y**4 + 6*x**4*y**3*z**2 + 8*x**4*y**3*z - 4*x**4*y**2*z**4 + 4*x**4*y**2*z**3 - 8*x**4*y**2*z**2 - 4*x**4*y*z**5 - 2*x**4*y*z**4 - 8*x**4*y*z**3 + 2*x**3*y**4*z + x**3*y**3*z**3 - x**3*y**2*z**5 - 2*x**3*y**2*z**3 + 9*x**3*y**2*z - 12*x**3*y*z**3 + 12*x**3*y*z**2 - 12*x**3*z**4 + 3*x**3*z**3 + 6*x**2*y**3 - 6*x**2*y**2*z**2 + 8*x**2*y**2*z - 2*x**2*y*z**4 - 8*x**2*y*z**3 + 2*x**2*y*z**2 + 2*x*y**3*z - 2*x*y**2*z**3 - 3*x*y*z + 3*x*z**3 - 2*y**2 + 2*y*z**2

def _w_2():
    R, x, y = ring("x,y", ZZ)
    return 24*x**8*y**3 + 48*x**8*y**2 + 24*x**7*y**5 - 72*x**7*y**2 + 25*x**6*y**4 + 2*x**6*y**3 + 4*x**6*y + 8*x**6 + x**5*y**6 + x**5*y**3 - 12*x**5 + x**4*y**5 - x**4*y**4 - 2*x**4*y**3 + 292*x**4*y**2 - x**3*y**6 + 3*x**3*y**3 - x**2*y**5 + 12*x**2*y**3 + 48*x**2 - 12*y**3

def f_polys():
    return _f_0(), _f_1(), _f_2(), _f_3(), _f_4(), _f_5(), _f_6()

def w_polys():
    return _w_1(), _w_2()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\utilities\matchpy_connector.py ===
"""
The objects in this module allow the usage of the MatchPy pattern matching
library on SymPy expressions.
"""
import re
from typing import List, Callable, NamedTuple, Any, Dict

from sympy.core.sympify import _sympify
from sympy.external import import_module
from sympy.functions import (log, sin, cos, tan, cot, csc, sec, erf, gamma, uppergamma)
from sympy.functions.elementary.hyperbolic import acosh, asinh, atanh, acoth, acsch, asech, cosh, sinh, tanh, coth, sech, csch
from sympy.functions.elementary.trigonometric import atan, acsc, asin, acot, acos, asec
from sympy.functions.special.error_functions import fresnelc, fresnels, erfc, erfi, Ei
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.relational import (Equality, Unequality)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp
from sympy.integrals.integrals import Integral
from sympy.printing.repr import srepr
from sympy.utilities.decorator import doctest_depends_on


matchpy = import_module("matchpy")


__doctest_requires__ = {('*',): ['matchpy']}


if matchpy:
    from matchpy import Operation, CommutativeOperation, AssociativeOperation, OneIdentityOperation
    from matchpy.expressions.functions import op_iter, create_operation_expression, op_len

    Operation.register(Integral)
    Operation.register(Pow)
    OneIdentityOperation.register(Pow)

    Operation.register(Add)
    OneIdentityOperation.register(Add)
    CommutativeOperation.register(Add)
    AssociativeOperation.register(Add)

    Operation.register(Mul)
    OneIdentityOperation.register(Mul)
    CommutativeOperation.register(Mul)
    AssociativeOperation.register(Mul)

    Operation.register(Equality)
    CommutativeOperation.register(Equality)
    Operation.register(Unequality)
    CommutativeOperation.register(Unequality)

    Operation.register(exp)
    Operation.register(log)
    Operation.register(gamma)
    Operation.register(uppergamma)
    Operation.register(fresnels)
    Operation.register(fresnelc)
    Operation.register(erf)
    Operation.register(Ei)
    Operation.register(erfc)
    Operation.register(erfi)
    Operation.register(sin)
    Operation.register(cos)
    Operation.register(tan)
    Operation.register(cot)
    Operation.register(csc)
    Operation.register(sec)
    Operation.register(sinh)
    Operation.register(cosh)
    Operation.register(tanh)
    Operation.register(coth)
    Operation.register(csch)
    Operation.register(sech)
    Operation.register(asin)
    Operation.register(acos)
    Operation.register(atan)
    Operation.register(acot)
    Operation.register(acsc)
    Operation.register(asec)
    Operation.register(asinh)
    Operation.register(acosh)
    Operation.register(atanh)
    Operation.register(acoth)
    Operation.register(acsch)
    Operation.register(asech)

    @op_iter.register(Integral)  # type: ignore
    def _(operation):
        return iter((operation._args[0],) + operation._args[1])

    @op_iter.register(Basic)  # type: ignore
    def _(operation):
        return iter(operation._args)

    @op_len.register(Integral)  # type: ignore
    def _(operation):
        return 1 + len(operation._args[1])

    @op_len.register(Basic)  # type: ignore
    def _(operation):
        return len(operation._args)

    @create_operation_expression.register(Basic)
    def sympy_op_factory(old_operation, new_operands, variable_name=True):
        return type(old_operation)(*new_operands)


if matchpy:
    from matchpy import Wildcard
else:
    class Wildcard: # type: ignore
        def __init__(self, min_length, fixed_size, variable_name, optional):
            self.min_count = min_length
            self.fixed_size = fixed_size
            self.variable_name = variable_name
            self.optional = optional


@doctest_depends_on(modules=('matchpy',))
class _WildAbstract(Wildcard, Symbol):
    min_length: int  # abstract field required in subclasses
    fixed_size: bool  # abstract field required in subclasses

    def __init__(self, variable_name=None, optional=None, **assumptions):
        min_length = self.min_length
        fixed_size = self.fixed_size
        if optional is not None:
            optional = _sympify(optional)
        Wildcard.__init__(self, min_length, fixed_size, str(variable_name), optional)

    def __getstate__(self):
        return {
            "min_length": self.min_length,
            "fixed_size": self.fixed_size,
            "min_count": self.min_count,
            "variable_name": self.variable_name,
            "optional": self.optional,
        }

    def __new__(cls, variable_name=None, optional=None, **assumptions):
        cls._sanitize(assumptions, cls)
        return _WildAbstract.__xnew__(cls, variable_name, optional, **assumptions)

    def __getnewargs__(self):
        return self.variable_name, self.optional

    @staticmethod
    def __xnew__(cls, variable_name=None, optional=None, **assumptions):
        obj = Symbol.__xnew__(cls, variable_name, **assumptions)
        return obj

    def _hashable_content(self):
        if self.optional:
            return super()._hashable_content() + (self.min_count, self.fixed_size, self.variable_name, self.optional)
        else:
            return super()._hashable_content() + (self.min_count, self.fixed_size, self.variable_name)

    def __copy__(self) -> '_WildAbstract':
        return type(self)(variable_name=self.variable_name, optional=self.optional)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.name


@doctest_depends_on(modules=('matchpy',))
class WildDot(_WildAbstract):
    min_length = 1
    fixed_size = True


@doctest_depends_on(modules=('matchpy',))
class WildPlus(_WildAbstract):
    min_length = 1
    fixed_size = False


@doctest_depends_on(modules=('matchpy',))
class WildStar(_WildAbstract):
    min_length = 0
    fixed_size = False


def _get_srepr(expr):
    s = srepr(expr)
    s = re.sub(r"WildDot\('(\w+)'\)", r"\1", s)
    s = re.sub(r"WildPlus\('(\w+)'\)", r"*\1", s)
    s = re.sub(r"WildStar\('(\w+)'\)", r"*\1", s)
    return s


class ReplacementInfo(NamedTuple):
    replacement: Any
    info: Any


@doctest_depends_on(modules=('matchpy',))
class Replacer:
    """
    Replacer object to perform multiple pattern matching and subexpression
    replacements in SymPy expressions.

    Examples
    ========

    Example to construct a simple first degree equation solver:

    >>> from sympy.utilities.matchpy_connector import WildDot, Replacer
    >>> from sympy import Equality, Symbol
    >>> x = Symbol("x")
    >>> a_ = WildDot("a_", optional=1)
    >>> b_ = WildDot("b_", optional=0)

    The lines above have defined two wildcards, ``a_`` and ``b_``, the
    coefficients of the equation `a x + b = 0`. The optional values specified
    indicate which expression to return in case no match is found, they are
    necessary in equations like `a x = 0` and `x + b = 0`.

    Create two constraints to make sure that ``a_`` and ``b_`` will not match
    any expression containing ``x``:

    >>> from matchpy import CustomConstraint
    >>> free_x_a = CustomConstraint(lambda a_: not a_.has(x))
    >>> free_x_b = CustomConstraint(lambda b_: not b_.has(x))

    Now create the rule replacer with the constraints:

    >>> replacer = Replacer(common_constraints=[free_x_a, free_x_b])

    Add the matching rule:

    >>> replacer.add(Equality(a_*x + b_, 0), -b_/a_)

    Let's try it:

    >>> replacer.replace(Equality(3*x + 4, 0))
    -4/3

    Notice that it will not match equations expressed with other patterns:

    >>> eq = Equality(3*x, 4)
    >>> replacer.replace(eq)
    Eq(3*x, 4)

    In order to extend the matching patterns, define another one (we also need
    to clear the cache, because the previous result has already been memorized
    and the pattern matcher will not iterate again if given the same expression)

    >>> replacer.add(Equality(a_*x, b_), b_/a_)
    >>> replacer._matcher.clear()
    >>> replacer.replace(eq)
    4/3
    """

    def __init__(self, common_constraints: list = [], lambdify: bool = False, info: bool = False):
        self._matcher = matchpy.ManyToOneMatcher()
        self._common_constraint = common_constraints
        self._lambdify = lambdify
        self._info = info
        self._wildcards: Dict[str, Wildcard] = {}

    def _get_lambda(self, lambda_str: str) -> Callable[..., Expr]:
        exec("from sympy import *")
        return eval(lambda_str, locals())

    def _get_custom_constraint(self, constraint_expr: Expr, condition_template: str) -> Callable[..., Expr]:
        wilds = [x.name for x in constraint_expr.atoms(_WildAbstract)]
        lambdaargs = ', '.join(wilds)
        fullexpr = _get_srepr(constraint_expr)
        condition = condition_template.format(fullexpr)
        return matchpy.CustomConstraint(
            self._get_lambda(f"lambda {lambdaargs}: ({condition})"))

    def _get_custom_constraint_nonfalse(self, constraint_expr: Expr) -> Callable[..., Expr]:
        return self._get_custom_constraint(constraint_expr, "({}) != False")

    def _get_custom_constraint_true(self, constraint_expr: Expr) -> Callable[..., Expr]:
        return self._get_custom_constraint(constraint_expr, "({}) == True")

    def add(self, expr: Expr, replacement, conditions_true: List[Expr] = [],
            conditions_nonfalse: List[Expr] = [], info: Any = None) -> None:
        expr = _sympify(expr)
        replacement = _sympify(replacement)
        constraints = self._common_constraint[:]
        constraint_conditions_true = [
            self._get_custom_constraint_true(cond) for cond in conditions_true]
        constraint_conditions_nonfalse = [
            self._get_custom_constraint_nonfalse(cond) for cond in conditions_nonfalse]
        constraints.extend(constraint_conditions_true)
        constraints.extend(constraint_conditions_nonfalse)
        pattern = matchpy.Pattern(expr, *constraints)
        if self._lambdify:
            lambda_str = f"lambda {', '.join((x.name for x in expr.atoms(_WildAbstract)))}: {_get_srepr(replacement)}"
            lambda_expr = self._get_lambda(lambda_str)
            replacement = lambda_expr
        else:
            self._wildcards.update({str(i): i for i in expr.atoms(Wildcard)})
        if self._info:
            replacement = ReplacementInfo(replacement, info)
        self._matcher.add(pattern, replacement)

    def replace(self, expression, max_count: int = -1):
        # This method partly rewrites the .replace method of ManyToOneReplacer
        # in MatchPy.
        # License: https://github.com/HPAC/matchpy/blob/master/LICENSE
        infos = []
        replaced = True
        replace_count = 0
        while replaced and (max_count < 0 or replace_count < max_count):
            replaced = False
            for subexpr, pos in matchpy.preorder_iter_with_position(expression):
                try:
                    replacement_data, subst = next(iter(self._matcher.match(subexpr)))
                    if self._info:
                        replacement = replacement_data.replacement
                        infos.append(replacement_data.info)
                    else:
                        replacement = replacement_data

                    if self._lambdify:
                        result = replacement(**subst)
                    else:
                        result = replacement.xreplace({self._wildcards[k]: v for k, v in subst.items()})

                    expression = matchpy.functions.replace(expression, pos, result)
                    replaced = True
                    break
                except StopIteration:
                    pass
            replace_count += 1
        if self._info:
            return expression, infos
        else:
            return expression

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\util\hashing.py ===
"""
data hash pandas / numpy objects
"""
from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np

from pandas._libs.hashing import hash_object_array

from pandas.core.dtypes.common import is_list_like
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCExtensionArray,
    ABCIndex,
    ABCMultiIndex,
    ABCSeries,
)

if TYPE_CHECKING:
    from collections.abc import (
        Hashable,
        Iterable,
        Iterator,
    )

    from pandas._typing import (
        ArrayLike,
        npt,
    )

    from pandas import (
        DataFrame,
        Index,
        MultiIndex,
        Series,
    )


# 16 byte long hashing key
_default_hash_key = "0123456789123456"


def combine_hash_arrays(
    arrays: Iterator[np.ndarray], num_items: int
) -> npt.NDArray[np.uint64]:
    """
    Parameters
    ----------
    arrays : Iterator[np.ndarray]
    num_items : int

    Returns
    -------
    np.ndarray[uint64]

    Should be the same as CPython's tupleobject.c
    """
    try:
        first = next(arrays)
    except StopIteration:
        return np.array([], dtype=np.uint64)

    arrays = itertools.chain([first], arrays)

    mult = np.uint64(1000003)
    out = np.zeros_like(first) + np.uint64(0x345678)
    last_i = 0
    for i, a in enumerate(arrays):
        inverse_i = num_items - i
        out ^= a
        out *= mult
        mult += np.uint64(82520 + inverse_i + inverse_i)
        last_i = i
    assert last_i + 1 == num_items, "Fed in wrong num_items"
    out += np.uint64(97531)
    return out


def hash_pandas_object(
    obj: Index | DataFrame | Series,
    index: bool = True,
    encoding: str = "utf8",
    hash_key: str | None = _default_hash_key,
    categorize: bool = True,
) -> Series:
    """
    Return a data hash of the Index/Series/DataFrame.

    Parameters
    ----------
    obj : Index, Series, or DataFrame
    index : bool, default True
        Include the index in the hash (if Series/DataFrame).
    encoding : str, default 'utf8'
        Encoding for data & key when strings.
    hash_key : str, default _default_hash_key
        Hash_key for string key to encode.
    categorize : bool, default True
        Whether to first categorize object arrays before hashing. This is more
        efficient when the array contains duplicate values.

    Returns
    -------
    Series of uint64, same length as the object

    Examples
    --------
    >>> pd.util.hash_pandas_object(pd.Series([1, 2, 3]))
    0    14639053686158035780
    1     3869563279212530728
    2      393322362522515241
    dtype: uint64
    """
    from pandas import Series

    if hash_key is None:
        hash_key = _default_hash_key

    if isinstance(obj, ABCMultiIndex):
        return Series(hash_tuples(obj, encoding, hash_key), dtype="uint64", copy=False)

    elif isinstance(obj, ABCIndex):
        h = hash_array(obj._values, encoding, hash_key, categorize).astype(
            "uint64", copy=False
        )
        ser = Series(h, index=obj, dtype="uint64", copy=False)

    elif isinstance(obj, ABCSeries):
        h = hash_array(obj._values, encoding, hash_key, categorize).astype(
            "uint64", copy=False
        )
        if index:
            index_iter = (
                hash_pandas_object(
                    obj.index,
                    index=False,
                    encoding=encoding,
                    hash_key=hash_key,
                    categorize=categorize,
                )._values
                for _ in [None]
            )
            arrays = itertools.chain([h], index_iter)
            h = combine_hash_arrays(arrays, 2)

        ser = Series(h, index=obj.index, dtype="uint64", copy=False)

    elif isinstance(obj, ABCDataFrame):
        hashes = (
            hash_array(series._values, encoding, hash_key, categorize)
            for _, series in obj.items()
        )
        num_items = len(obj.columns)
        if index:
            index_hash_generator = (
                hash_pandas_object(
                    obj.index,
                    index=False,
                    encoding=encoding,
                    hash_key=hash_key,
                    categorize=categorize,
                )._values
                for _ in [None]
            )
            num_items += 1

            # keep `hashes` specifically a generator to keep mypy happy
            _hashes = itertools.chain(hashes, index_hash_generator)
            hashes = (x for x in _hashes)
        h = combine_hash_arrays(hashes, num_items)

        ser = Series(h, index=obj.index, dtype="uint64", copy=False)
    else:
        raise TypeError(f"Unexpected type for hashing {type(obj)}")

    return ser


def hash_tuples(
    vals: MultiIndex | Iterable[tuple[Hashable, ...]],
    encoding: str = "utf8",
    hash_key: str = _default_hash_key,
) -> npt.NDArray[np.uint64]:
    """
    Hash an MultiIndex / listlike-of-tuples efficiently.

    Parameters
    ----------
    vals : MultiIndex or listlike-of-tuples
    encoding : str, default 'utf8'
    hash_key : str, default _default_hash_key

    Returns
    -------
    ndarray[np.uint64] of hashed values
    """
    if not is_list_like(vals):
        raise TypeError("must be convertible to a list-of-tuples")

    from pandas import (
        Categorical,
        MultiIndex,
    )

    if not isinstance(vals, ABCMultiIndex):
        mi = MultiIndex.from_tuples(vals)
    else:
        mi = vals

    # create a list-of-Categoricals
    cat_vals = [
        Categorical._simple_new(
            mi.codes[level],
            CategoricalDtype(categories=mi.levels[level], ordered=False),
        )
        for level in range(mi.nlevels)
    ]

    # hash the list-of-ndarrays
    hashes = (
        cat._hash_pandas_object(encoding=encoding, hash_key=hash_key, categorize=False)
        for cat in cat_vals
    )
    h = combine_hash_arrays(hashes, len(cat_vals))

    return h


def hash_array(
    vals: ArrayLike,
    encoding: str = "utf8",
    hash_key: str = _default_hash_key,
    categorize: bool = True,
) -> npt.NDArray[np.uint64]:
    """
    Given a 1d array, return an array of deterministic integers.

    Parameters
    ----------
    vals : ndarray or ExtensionArray
    encoding : str, default 'utf8'
        Encoding for data & key when strings.
    hash_key : str, default _default_hash_key
        Hash_key for string key to encode.
    categorize : bool, default True
        Whether to first categorize object arrays before hashing. This is more
        efficient when the array contains duplicate values.

    Returns
    -------
    ndarray[np.uint64, ndim=1]
        Hashed values, same length as the vals.

    Examples
    --------
    >>> pd.util.hash_array(np.array([1, 2, 3]))
    array([ 6238072747940578789, 15839785061582574730,  2185194620014831856],
      dtype=uint64)
    """
    if not hasattr(vals, "dtype"):
        raise TypeError("must pass a ndarray-like")

    if isinstance(vals, ABCExtensionArray):
        return vals._hash_pandas_object(
            encoding=encoding, hash_key=hash_key, categorize=categorize
        )

    if not isinstance(vals, np.ndarray):
        # GH#42003
        raise TypeError(
            "hash_array requires np.ndarray or ExtensionArray, not "
            f"{type(vals).__name__}. Use hash_pandas_object instead."
        )

    return _hash_ndarray(vals, encoding, hash_key, categorize)


def _hash_ndarray(
    vals: np.ndarray,
    encoding: str = "utf8",
    hash_key: str = _default_hash_key,
    categorize: bool = True,
) -> npt.NDArray[np.uint64]:
    """
    See hash_array.__doc__.
    """
    dtype = vals.dtype

    # _hash_ndarray only takes 64-bit values, so handle 128-bit by parts
    if np.issubdtype(dtype, np.complex128):
        hash_real = _hash_ndarray(vals.real, encoding, hash_key, categorize)
        hash_imag = _hash_ndarray(vals.imag, encoding, hash_key, categorize)
        return hash_real + 23 * hash_imag

    # First, turn whatever array this is into unsigned 64-bit ints, if we can
    # manage it.
    if dtype == bool:
        vals = vals.astype("u8")
    elif issubclass(dtype.type, (np.datetime64, np.timedelta64)):
        vals = vals.view("i8").astype("u8", copy=False)
    elif issubclass(dtype.type, np.number) and dtype.itemsize <= 8:
        vals = vals.view(f"u{vals.dtype.itemsize}").astype("u8")
    else:
        # With repeated values, its MUCH faster to categorize object dtypes,
        # then hash and rename categories. We allow skipping the categorization
        # when the values are known/likely to be unique.
        if categorize:
            from pandas import (
                Categorical,
                Index,
                factorize,
            )

            codes, categories = factorize(vals, sort=False)
            dtype = CategoricalDtype(categories=Index(categories), ordered=False)
            cat = Categorical._simple_new(codes, dtype)
            return cat._hash_pandas_object(
                encoding=encoding, hash_key=hash_key, categorize=False
            )

        try:
            vals = hash_object_array(vals, hash_key, encoding)
        except TypeError:
            # we have mixed types
            vals = hash_object_array(
                vals.astype(str).astype(object), hash_key, encoding
            )

    # Then, redistribute these 64-bit ints within the space of 64-bit ints
    vals ^= vals >> 30
    vals *= np.uint64(0xBF58476D1CE4E5B9)
    vals ^= vals >> 27
    vals *= np.uint64(0x94D049BB133111EB)
    vals ^= vals >> 31
    return vals

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\mysql\asyncmy.py ===
# dialects/mysql/asyncmy.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors <see AUTHORS
# file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors

r"""
.. dialect:: mysql+asyncmy
    :name: asyncmy
    :dbapi: asyncmy
    :connectstring: mysql+asyncmy://user:password@host:port/dbname[?key=value&key=value...]
    :url: https://github.com/long2ice/asyncmy

Using a special asyncio mediation layer, the asyncmy dialect is usable
as the backend for the :ref:`SQLAlchemy asyncio <asyncio_toplevel>`
extension package.

This dialect should normally be used only with the
:func:`_asyncio.create_async_engine` engine creation function::

    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(
        "mysql+asyncmy://user:pass@hostname/dbname?charset=utf8mb4"
    )

"""  # noqa
from collections import deque
from contextlib import asynccontextmanager

from .pymysql import MySQLDialect_pymysql
from ... import pool
from ... import util
from ...engine import AdaptedConnection
from ...util.concurrency import asyncio
from ...util.concurrency import await_fallback
from ...util.concurrency import await_only


class AsyncAdapt_asyncmy_cursor:
    # TODO: base on connectors/asyncio.py
    # see #10415
    server_side = False
    __slots__ = (
        "_adapt_connection",
        "_connection",
        "await_",
        "_cursor",
        "_rows",
    )

    def __init__(self, adapt_connection):
        self._adapt_connection = adapt_connection
        self._connection = adapt_connection._connection
        self.await_ = adapt_connection.await_

        cursor = self._connection.cursor()

        self._cursor = self.await_(cursor.__aenter__())
        self._rows = deque()

    @property
    def description(self):
        return self._cursor.description

    @property
    def rowcount(self):
        return self._cursor.rowcount

    @property
    def arraysize(self):
        return self._cursor.arraysize

    @arraysize.setter
    def arraysize(self, value):
        self._cursor.arraysize = value

    @property
    def lastrowid(self):
        return self._cursor.lastrowid

    def close(self):
        # note we aren't actually closing the cursor here,
        # we are just letting GC do it.   to allow this to be async
        # we would need the Result to change how it does "Safe close cursor".
        # MySQL "cursors" don't actually have state to be "closed" besides
        # exhausting rows, which we already have done for sync cursor.
        # another option would be to emulate aiosqlite dialect and assign
        # cursor only if we are doing server side cursor operation.
        self._rows.clear()

    def execute(self, operation, parameters=None):
        return self.await_(self._execute_async(operation, parameters))

    def executemany(self, operation, seq_of_parameters):
        return self.await_(
            self._executemany_async(operation, seq_of_parameters)
        )

    async def _execute_async(self, operation, parameters):
        async with self._adapt_connection._mutex_and_adapt_errors():
            if parameters is None:
                result = await self._cursor.execute(operation)
            else:
                result = await self._cursor.execute(operation, parameters)

            if not self.server_side:
                # asyncmy has a "fake" async result, so we have to pull it out
                # of that here since our default result is not async.
                # we could just as easily grab "_rows" here and be done with it
                # but this is safer.
                self._rows = deque(await self._cursor.fetchall())
            return result

    async def _executemany_async(self, operation, seq_of_parameters):
        async with self._adapt_connection._mutex_and_adapt_errors():
            return await self._cursor.executemany(operation, seq_of_parameters)

    def setinputsizes(self, *inputsizes):
        pass

    def __iter__(self):
        while self._rows:
            yield self._rows.popleft()

    def fetchone(self):
        if self._rows:
            return self._rows.popleft()
        else:
            return None

    def fetchmany(self, size=None):
        if size is None:
            size = self.arraysize

        rr = self._rows
        return [rr.popleft() for _ in range(min(size, len(rr)))]

    def fetchall(self):
        retval = list(self._rows)
        self._rows.clear()
        return retval


class AsyncAdapt_asyncmy_ss_cursor(AsyncAdapt_asyncmy_cursor):
    # TODO: base on connectors/asyncio.py
    # see #10415
    __slots__ = ()
    server_side = True

    def __init__(self, adapt_connection):
        self._adapt_connection = adapt_connection
        self._connection = adapt_connection._connection
        self.await_ = adapt_connection.await_

        cursor = self._connection.cursor(
            adapt_connection.dbapi.asyncmy.cursors.SSCursor
        )

        self._cursor = self.await_(cursor.__aenter__())

    def close(self):
        if self._cursor is not None:
            self.await_(self._cursor.close())
            self._cursor = None

    def fetchone(self):
        return self.await_(self._cursor.fetchone())

    def fetchmany(self, size=None):
        return self.await_(self._cursor.fetchmany(size=size))

    def fetchall(self):
        return self.await_(self._cursor.fetchall())


class AsyncAdapt_asyncmy_connection(AdaptedConnection):
    # TODO: base on connectors/asyncio.py
    # see #10415
    await_ = staticmethod(await_only)
    __slots__ = ("dbapi", "_execute_mutex")

    def __init__(self, dbapi, connection):
        self.dbapi = dbapi
        self._connection = connection
        self._execute_mutex = asyncio.Lock()

    @asynccontextmanager
    async def _mutex_and_adapt_errors(self):
        async with self._execute_mutex:
            try:
                yield
            except AttributeError:
                raise self.dbapi.InternalError(
                    "network operation failed due to asyncmy attribute error"
                )

    def ping(self, reconnect):
        assert not reconnect
        return self.await_(self._do_ping())

    async def _do_ping(self):
        async with self._mutex_and_adapt_errors():
            return await self._connection.ping(False)

    def character_set_name(self):
        return self._connection.character_set_name()

    def autocommit(self, value):
        self.await_(self._connection.autocommit(value))

    def cursor(self, server_side=False):
        if server_side:
            return AsyncAdapt_asyncmy_ss_cursor(self)
        else:
            return AsyncAdapt_asyncmy_cursor(self)

    def rollback(self):
        self.await_(self._connection.rollback())

    def commit(self):
        self.await_(self._connection.commit())

    def terminate(self):
        # it's not awaitable.
        self._connection.close()

    def close(self) -> None:
        self.await_(self._connection.ensure_closed())


class AsyncAdaptFallback_asyncmy_connection(AsyncAdapt_asyncmy_connection):
    __slots__ = ()

    await_ = staticmethod(await_fallback)


def _Binary(x):
    """Return x as a binary type."""
    return bytes(x)


class AsyncAdapt_asyncmy_dbapi:
    def __init__(self, asyncmy):
        self.asyncmy = asyncmy
        self.paramstyle = "format"
        self._init_dbapi_attributes()

    def _init_dbapi_attributes(self):
        for name in (
            "Warning",
            "Error",
            "InterfaceError",
            "DataError",
            "DatabaseError",
            "OperationalError",
            "InterfaceError",
            "IntegrityError",
            "ProgrammingError",
            "InternalError",
            "NotSupportedError",
        ):
            setattr(self, name, getattr(self.asyncmy.errors, name))

    STRING = util.symbol("STRING")
    NUMBER = util.symbol("NUMBER")
    BINARY = util.symbol("BINARY")
    DATETIME = util.symbol("DATETIME")
    TIMESTAMP = util.symbol("TIMESTAMP")
    Binary = staticmethod(_Binary)

    def connect(self, *arg, **kw):
        async_fallback = kw.pop("async_fallback", False)
        creator_fn = kw.pop("async_creator_fn", self.asyncmy.connect)

        if util.asbool(async_fallback):
            return AsyncAdaptFallback_asyncmy_connection(
                self,
                await_fallback(creator_fn(*arg, **kw)),
            )
        else:
            return AsyncAdapt_asyncmy_connection(
                self,
                await_only(creator_fn(*arg, **kw)),
            )


class MySQLDialect_asyncmy(MySQLDialect_pymysql):
    driver = "asyncmy"
    supports_statement_cache = True

    supports_server_side_cursors = True
    _sscursor = AsyncAdapt_asyncmy_ss_cursor

    is_async = True
    has_terminate = True

    @classmethod
    def import_dbapi(cls):
        return AsyncAdapt_asyncmy_dbapi(__import__("asyncmy"))

    @classmethod
    def get_pool_class(cls, url):
        async_fallback = url.query.get("async_fallback", False)

        if util.asbool(async_fallback):
            return pool.FallbackAsyncAdaptedQueuePool
        else:
            return pool.AsyncAdaptedQueuePool

    def do_terminate(self, dbapi_connection) -> None:
        dbapi_connection.terminate()

    def create_connect_args(self, url):
        return super().create_connect_args(
            url, _translate_args=dict(username="user", database="db")
        )

    def is_disconnect(self, e, connection, cursor):
        if super().is_disconnect(e, connection, cursor):
            return True
        else:
            str_e = str(e).lower()
            return (
                "not connected" in str_e or "network operation failed" in str_e
            )

    def _found_rows_client_flag(self):
        from asyncmy.constants import CLIENT

        return CLIENT.FOUND_ROWS

    def get_driver_connection(self, connection):
        return connection._connection


dialect = MySQLDialect_asyncmy

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\postgresql\dml.py ===
# dialects/postgresql/dml.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
from __future__ import annotations

from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from . import ext
from .._typing import _OnConflictConstraintT
from .._typing import _OnConflictIndexElementsT
from .._typing import _OnConflictIndexWhereT
from .._typing import _OnConflictSetT
from .._typing import _OnConflictWhereT
from ... import util
from ...sql import coercions
from ...sql import roles
from ...sql import schema
from ...sql._typing import _DMLTableArgument
from ...sql.base import _exclusive_against
from ...sql.base import _generative
from ...sql.base import ColumnCollection
from ...sql.base import ReadOnlyColumnCollection
from ...sql.dml import Insert as StandardInsert
from ...sql.elements import ClauseElement
from ...sql.elements import ColumnElement
from ...sql.elements import KeyedColumnElement
from ...sql.elements import TextClause
from ...sql.expression import alias
from ...util.typing import Self


__all__ = ("Insert", "insert")


def insert(table: _DMLTableArgument) -> Insert:
    """Construct a PostgreSQL-specific variant :class:`_postgresql.Insert`
    construct.

    .. container:: inherited_member

        The :func:`sqlalchemy.dialects.postgresql.insert` function creates
        a :class:`sqlalchemy.dialects.postgresql.Insert`.  This class is based
        on the dialect-agnostic :class:`_sql.Insert` construct which may
        be constructed using the :func:`_sql.insert` function in
        SQLAlchemy Core.

    The :class:`_postgresql.Insert` construct includes additional methods
    :meth:`_postgresql.Insert.on_conflict_do_update`,
    :meth:`_postgresql.Insert.on_conflict_do_nothing`.

    """
    return Insert(table)


class Insert(StandardInsert):
    """PostgreSQL-specific implementation of INSERT.

    Adds methods for PG-specific syntaxes such as ON CONFLICT.

    The :class:`_postgresql.Insert` object is created using the
    :func:`sqlalchemy.dialects.postgresql.insert` function.

    """

    stringify_dialect = "postgresql"
    inherit_cache = False

    @util.memoized_property
    def excluded(
        self,
    ) -> ReadOnlyColumnCollection[str, KeyedColumnElement[Any]]:
        """Provide the ``excluded`` namespace for an ON CONFLICT statement

        PG's ON CONFLICT clause allows reference to the row that would
        be inserted, known as ``excluded``.  This attribute provides
        all columns in this row to be referenceable.

        .. tip::  The :attr:`_postgresql.Insert.excluded` attribute is an
            instance of :class:`_expression.ColumnCollection`, which provides
            an interface the same as that of the :attr:`_schema.Table.c`
            collection described at :ref:`metadata_tables_and_columns`.
            With this collection, ordinary names are accessible like attributes
            (e.g. ``stmt.excluded.some_column``), but special names and
            dictionary method names should be accessed using indexed access,
            such as ``stmt.excluded["column name"]`` or
            ``stmt.excluded["values"]``.   See the docstring for
            :class:`_expression.ColumnCollection` for further examples.

        .. seealso::

            :ref:`postgresql_insert_on_conflict` - example of how
            to use :attr:`_expression.Insert.excluded`

        """
        return alias(self.table, name="excluded").columns

    _on_conflict_exclusive = _exclusive_against(
        "_post_values_clause",
        msgs={
            "_post_values_clause": "This Insert construct already has "
            "an ON CONFLICT clause established"
        },
    )

    @_generative
    @_on_conflict_exclusive
    def on_conflict_do_update(
        self,
        constraint: _OnConflictConstraintT = None,
        index_elements: _OnConflictIndexElementsT = None,
        index_where: _OnConflictIndexWhereT = None,
        set_: _OnConflictSetT = None,
        where: _OnConflictWhereT = None,
    ) -> Self:
        r"""
        Specifies a DO UPDATE SET action for ON CONFLICT clause.

        Either the ``constraint`` or ``index_elements`` argument is
        required, but only one of these can be specified.

        :param constraint:
         The name of a unique or exclusion constraint on the table,
         or the constraint object itself if it has a .name attribute.

        :param index_elements:
         A sequence consisting of string column names, :class:`_schema.Column`
         objects, or other column expression objects that will be used
         to infer a target index.

        :param index_where:
         Additional WHERE criterion that can be used to infer a
         conditional target index.

        :param set\_:
         A dictionary or other mapping object
         where the keys are either names of columns in the target table,
         or :class:`_schema.Column` objects or other ORM-mapped columns
         matching that of the target table, and expressions or literals
         as values, specifying the ``SET`` actions to take.

         .. versionadded:: 1.4 The
            :paramref:`_postgresql.Insert.on_conflict_do_update.set_`
            parameter supports :class:`_schema.Column` objects from the target
            :class:`_schema.Table` as keys.

         .. warning:: This dictionary does **not** take into account
            Python-specified default UPDATE values or generation functions,
            e.g. those specified using :paramref:`_schema.Column.onupdate`.
            These values will not be exercised for an ON CONFLICT style of
            UPDATE, unless they are manually specified in the
            :paramref:`.Insert.on_conflict_do_update.set_` dictionary.

        :param where:
         Optional argument. An expression object representing a ``WHERE``
         clause that restricts the rows affected by ``DO UPDATE SET``. Rows not
         meeting the ``WHERE`` condition will not be updated (effectively a
         ``DO NOTHING`` for those rows).


        .. seealso::

            :ref:`postgresql_insert_on_conflict`

        """
        self._post_values_clause = OnConflictDoUpdate(
            constraint, index_elements, index_where, set_, where
        )
        return self

    @_generative
    @_on_conflict_exclusive
    def on_conflict_do_nothing(
        self,
        constraint: _OnConflictConstraintT = None,
        index_elements: _OnConflictIndexElementsT = None,
        index_where: _OnConflictIndexWhereT = None,
    ) -> Self:
        """
        Specifies a DO NOTHING action for ON CONFLICT clause.

        The ``constraint`` and ``index_elements`` arguments
        are optional, but only one of these can be specified.

        :param constraint:
         The name of a unique or exclusion constraint on the table,
         or the constraint object itself if it has a .name attribute.

        :param index_elements:
         A sequence consisting of string column names, :class:`_schema.Column`
         objects, or other column expression objects that will be used
         to infer a target index.

        :param index_where:
         Additional WHERE criterion that can be used to infer a
         conditional target index.

        .. seealso::

            :ref:`postgresql_insert_on_conflict`

        """
        self._post_values_clause = OnConflictDoNothing(
            constraint, index_elements, index_where
        )
        return self


class OnConflictClause(ClauseElement):
    stringify_dialect = "postgresql"

    constraint_target: Optional[str]
    inferred_target_elements: Optional[List[Union[str, schema.Column[Any]]]]
    inferred_target_whereclause: Optional[
        Union[ColumnElement[Any], TextClause]
    ]

    def __init__(
        self,
        constraint: _OnConflictConstraintT = None,
        index_elements: _OnConflictIndexElementsT = None,
        index_where: _OnConflictIndexWhereT = None,
    ):
        if constraint is not None:
            if not isinstance(constraint, str) and isinstance(
                constraint,
                (schema.Constraint, ext.ExcludeConstraint),
            ):
                constraint = getattr(constraint, "name") or constraint

        if constraint is not None:
            if index_elements is not None:
                raise ValueError(
                    "'constraint' and 'index_elements' are mutually exclusive"
                )

            if isinstance(constraint, str):
                self.constraint_target = constraint
                self.inferred_target_elements = None
                self.inferred_target_whereclause = None
            elif isinstance(constraint, schema.Index):
                index_elements = constraint.expressions
                index_where = constraint.dialect_options["postgresql"].get(
                    "where"
                )
            elif isinstance(constraint, ext.ExcludeConstraint):
                index_elements = constraint.columns
                index_where = constraint.where
            else:
                index_elements = constraint.columns
                index_where = constraint.dialect_options["postgresql"].get(
                    "where"
                )

        if index_elements is not None:
            self.constraint_target = None
            self.inferred_target_elements = [
                coercions.expect(roles.DDLConstraintColumnRole, column)
                for column in index_elements
            ]

            self.inferred_target_whereclause = (
                coercions.expect(
                    (
                        roles.StatementOptionRole
                        if isinstance(constraint, ext.ExcludeConstraint)
                        else roles.WhereHavingRole
                    ),
                    index_where,
                )
                if index_where is not None
                else None
            )

        elif constraint is None:
            self.constraint_target = self.inferred_target_elements = (
                self.inferred_target_whereclause
            ) = None


class OnConflictDoNothing(OnConflictClause):
    __visit_name__ = "on_conflict_do_nothing"


class OnConflictDoUpdate(OnConflictClause):
    __visit_name__ = "on_conflict_do_update"

    update_values_to_set: List[Tuple[Union[schema.Column[Any], str], Any]]
    update_whereclause: Optional[ColumnElement[Any]]

    def __init__(
        self,
        constraint: _OnConflictConstraintT = None,
        index_elements: _OnConflictIndexElementsT = None,
        index_where: _OnConflictIndexWhereT = None,
        set_: _OnConflictSetT = None,
        where: _OnConflictWhereT = None,
    ):
        super().__init__(
            constraint=constraint,
            index_elements=index_elements,
            index_where=index_where,
        )

        if (
            self.inferred_target_elements is None
            and self.constraint_target is None
        ):
            raise ValueError(
                "Either constraint or index_elements, "
                "but not both, must be specified unless DO NOTHING"
            )

        if isinstance(set_, dict):
            if not set_:
                raise ValueError("set parameter dictionary must not be empty")
        elif isinstance(set_, ColumnCollection):
            set_ = dict(set_)
        else:
            raise ValueError(
                "set parameter must be a non-empty dictionary "
                "or a ColumnCollection such as the `.c.` collection "
                "of a Table object"
            )
        self.update_values_to_set = [
            (coercions.expect(roles.DMLColumnRole, key), value)
            for key, value in set_.items()
        ]
        self.update_whereclause = (
            coercions.expect(roles.WhereHavingRole, where)
            if where is not None
            else None
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\repr.py ===
"""
A Printer for generating executable code.

The most important function here is srepr that returns a string so that the
relation eval(srepr(expr))=expr holds in an appropriate environment.
"""

from __future__ import annotations
from typing import Any

from sympy.core.function import AppliedUndef
from sympy.core.mul import Mul
from mpmath.libmp import repr_dps, to_str as mlib_to_str

from .printer import Printer, print_function


class ReprPrinter(Printer):
    printmethod = "_sympyrepr"

    _default_settings: dict[str, Any] = {
        "order": None,
        "perm_cyclic" : True,
    }

    def reprify(self, args, sep):
        """
        Prints each item in `args` and joins them with `sep`.
        """
        return sep.join([self.doprint(item) for item in args])

    def emptyPrinter(self, expr):
        """
        The fallback printer.
        """
        if isinstance(expr, str):
            return expr
        elif hasattr(expr, "__srepr__"):
            return expr.__srepr__()
        elif hasattr(expr, "args") and hasattr(expr.args, "__iter__"):
            l = []
            for o in expr.args:
                l.append(self._print(o))
            return expr.__class__.__name__ + '(%s)' % ', '.join(l)
        elif hasattr(expr, "__module__") and hasattr(expr, "__name__"):
            return "<'%s.%s'>" % (expr.__module__, expr.__name__)
        else:
            return str(expr)

    def _print_Add(self, expr, order=None):
        args = self._as_ordered_terms(expr, order=order)
        args = map(self._print, args)
        clsname = type(expr).__name__
        return clsname + "(%s)" % ", ".join(args)

    def _print_Cycle(self, expr):
        return expr.__repr__()

    def _print_Permutation(self, expr):
        from sympy.combinatorics.permutations import Permutation, Cycle
        from sympy.utilities.exceptions import sympy_deprecation_warning

        perm_cyclic = Permutation.print_cyclic
        if perm_cyclic is not None:
            sympy_deprecation_warning(
                f"""
                Setting Permutation.print_cyclic is deprecated. Instead use
                init_printing(perm_cyclic={perm_cyclic}).
                """,
                deprecated_since_version="1.6",
                active_deprecations_target="deprecated-permutation-print_cyclic",
                stacklevel=7,
            )
        else:
            perm_cyclic = self._settings.get("perm_cyclic", True)

        if perm_cyclic:
            if not expr.size:
                return 'Permutation()'
            # before taking Cycle notation, see if the last element is
            # a singleton and move it to the head of the string
            s = Cycle(expr)(expr.size - 1).__repr__()[len('Cycle'):]
            last = s.rfind('(')
            if not last == 0 and ',' not in s[last:]:
                s = s[last:] + s[:last]
            return 'Permutation%s' %s
        else:
            s = expr.support()
            if not s:
                if expr.size < 5:
                    return 'Permutation(%s)' % str(expr.array_form)
                return 'Permutation([], size=%s)' % expr.size
            trim = str(expr.array_form[:s[-1] + 1]) + ', size=%s' % expr.size
            use = full = str(expr.array_form)
            if len(trim) < len(full):
                use = trim
            return 'Permutation(%s)' % use

    def _print_Function(self, expr):
        r = self._print(expr.func)
        r += '(%s)' % ', '.join([self._print(a) for a in expr.args])
        return r

    def _print_Heaviside(self, expr):
        # Same as _print_Function but uses pargs to suppress default value for
        # 2nd arg.
        r = self._print(expr.func)
        r += '(%s)' % ', '.join([self._print(a) for a in expr.pargs])
        return r

    def _print_FunctionClass(self, expr):
        if issubclass(expr, AppliedUndef):
            return 'Function(%r)' % (expr.__name__)
        else:
            return expr.__name__

    def _print_Half(self, expr):
        return 'Rational(1, 2)'

    def _print_RationalConstant(self, expr):
        return str(expr)

    def _print_AtomicExpr(self, expr):
        return str(expr)

    def _print_NumberSymbol(self, expr):
        return str(expr)

    def _print_Integer(self, expr):
        return 'Integer(%i)' % expr.p

    def _print_Complexes(self, expr):
        return 'Complexes'

    def _print_Integers(self, expr):
        return 'Integers'

    def _print_Naturals(self, expr):
        return 'Naturals'

    def _print_Naturals0(self, expr):
        return 'Naturals0'

    def _print_Rationals(self, expr):
        return 'Rationals'

    def _print_Reals(self, expr):
        return 'Reals'

    def _print_EmptySet(self, expr):
        return 'EmptySet'

    def _print_UniversalSet(self, expr):
        return 'UniversalSet'

    def _print_EmptySequence(self, expr):
        return 'EmptySequence'

    def _print_list(self, expr):
        return "[%s]" % self.reprify(expr, ", ")

    def _print_dict(self, expr):
        sep = ", "
        dict_kvs = ["%s: %s" % (self.doprint(key), self.doprint(value)) for key, value in expr.items()]
        return "{%s}" % sep.join(dict_kvs)

    def _print_set(self, expr):
        if not expr:
            return "set()"
        return "{%s}" % self.reprify(expr, ", ")

    def _print_MatrixBase(self, expr):
        # special case for some empty matrices
        if (expr.rows == 0) ^ (expr.cols == 0):
            return '%s(%s, %s, %s)' % (expr.__class__.__name__,
                                       self._print(expr.rows),
                                       self._print(expr.cols),
                                       self._print([]))
        l = []
        for i in range(expr.rows):
            l.append([])
            for j in range(expr.cols):
                l[-1].append(expr[i, j])
        return '%s(%s)' % (expr.__class__.__name__, self._print(l))

    def _print_BooleanTrue(self, expr):
        return "true"

    def _print_BooleanFalse(self, expr):
        return "false"

    def _print_NaN(self, expr):
        return "nan"

    def _print_Mul(self, expr, order=None):
        if self.order not in ('old', 'none'):
            args = expr.as_ordered_factors()
        else:
            # use make_args in case expr was something like -x -> x
            args = Mul.make_args(expr)

        args = map(self._print, args)
        clsname = type(expr).__name__
        return clsname + "(%s)" % ", ".join(args)

    def _print_Rational(self, expr):
        return 'Rational(%s, %s)' % (self._print(expr.p), self._print(expr.q))

    def _print_PythonRational(self, expr):
        return "%s(%d, %d)" % (expr.__class__.__name__, expr.p, expr.q)

    def _print_Fraction(self, expr):
        return 'Fraction(%s, %s)' % (self._print(expr.numerator), self._print(expr.denominator))

    def _print_Float(self, expr):
        r = mlib_to_str(expr._mpf_, repr_dps(expr._prec))
        return "%s('%s', precision=%i)" % (expr.__class__.__name__, r, expr._prec)

    def _print_Sum2(self, expr):
        return "Sum2(%s, (%s, %s, %s))" % (self._print(expr.f), self._print(expr.i),
                                           self._print(expr.a), self._print(expr.b))

    def _print_Str(self, s):
        return "%s(%s)" % (s.__class__.__name__, self._print(s.name))

    def _print_Symbol(self, expr):
        d = expr._assumptions_orig
        # print the dummy_index like it was an assumption
        if expr.is_Dummy:
            d = d.copy()
            d['dummy_index'] = expr.dummy_index

        if d == {}:
            return "%s(%s)" % (expr.__class__.__name__, self._print(expr.name))
        else:
            attr = ['%s=%s' % (k, v) for k, v in d.items()]
            return "%s(%s, %s)" % (expr.__class__.__name__,
                                   self._print(expr.name), ', '.join(attr))

    def _print_CoordinateSymbol(self, expr):
        d = expr._assumptions.generator

        if d == {}:
            return "%s(%s, %s)" % (
                expr.__class__.__name__,
                self._print(expr.coord_sys),
                self._print(expr.index)
            )
        else:
            attr = ['%s=%s' % (k, v) for k, v in d.items()]
            return "%s(%s, %s, %s)" % (
                expr.__class__.__name__,
                self._print(expr.coord_sys),
                self._print(expr.index),
                ', '.join(attr)
            )

    def _print_Predicate(self, expr):
        return "Q.%s" % expr.name

    def _print_AppliedPredicate(self, expr):
        # will be changed to just expr.args when args overriding is removed
        args = expr._args
        return "%s(%s)" % (expr.__class__.__name__, self.reprify(args, ", "))

    def _print_str(self, expr):
        return repr(expr)

    def _print_tuple(self, expr):
        if len(expr) == 1:
            return "(%s,)" % self._print(expr[0])
        else:
            return "(%s)" % self.reprify(expr, ", ")

    def _print_WildFunction(self, expr):
        return "%s('%s')" % (expr.__class__.__name__, expr.name)

    def _print_AlgebraicNumber(self, expr):
        return "%s(%s, %s)" % (expr.__class__.__name__,
            self._print(expr.root), self._print(expr.coeffs()))

    def _print_PolyRing(self, ring):
        return "%s(%s, %s, %s)" % (ring.__class__.__name__,
            self._print(ring.symbols), self._print(ring.domain), self._print(ring.order))

    def _print_FracField(self, field):
        return "%s(%s, %s, %s)" % (field.__class__.__name__,
            self._print(field.symbols), self._print(field.domain), self._print(field.order))

    def _print_PolyElement(self, poly):
        terms = list(poly.terms())
        terms.sort(key=poly.ring.order, reverse=True)
        return "%s(%s, %s)" % (poly.__class__.__name__, self._print(poly.ring), self._print(terms))

    def _print_FracElement(self, frac):
        numer_terms = list(frac.numer.terms())
        numer_terms.sort(key=frac.field.order, reverse=True)
        denom_terms = list(frac.denom.terms())
        denom_terms.sort(key=frac.field.order, reverse=True)
        numer = self._print(numer_terms)
        denom = self._print(denom_terms)
        return "%s(%s, %s, %s)" % (frac.__class__.__name__, self._print(frac.field), numer, denom)

    def _print_FractionField(self, domain):
        cls = domain.__class__.__name__
        field = self._print(domain.field)
        return "%s(%s)" % (cls, field)

    def _print_PolynomialRingBase(self, ring):
        cls = ring.__class__.__name__
        dom = self._print(ring.domain)
        gens = ', '.join(map(self._print, ring.gens))
        order = str(ring.order)
        if order != ring.default_order:
            orderstr = ", order=" + order
        else:
            orderstr = ""
        return "%s(%s, %s%s)" % (cls, dom, gens, orderstr)

    def _print_DMP(self, p):
        cls = p.__class__.__name__
        rep = self._print(p.to_list())
        dom = self._print(p.dom)
        return "%s(%s, %s)" % (cls, rep, dom)

    def _print_MonogenicFiniteExtension(self, ext):
        # The expanded tree shown by srepr(ext.modulus)
        # is not practical.
        return "FiniteExtension(%s)" % str(ext.modulus)

    def _print_ExtensionElement(self, f):
        rep = self._print(f.rep)
        ext = self._print(f.ext)
        return "ExtElem(%s, %s)" % (rep, ext)

@print_function(ReprPrinter)
def srepr(expr, **settings):
    """return expr in repr form"""
    return ReprPrinter(settings).doprint(expr)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\utilities\decorator.py ===
"""Useful utility decorators. """

from typing import TypeVar
import sys
import types
import inspect
from functools import wraps, update_wrapper

from sympy.utilities.exceptions import sympy_deprecation_warning


T = TypeVar('T')
"""A generic type"""


def threaded_factory(func, use_add):
    """A factory for ``threaded`` decorators. """
    from sympy.core import sympify
    from sympy.matrices import MatrixBase
    from sympy.utilities.iterables import iterable

    @wraps(func)
    def threaded_func(expr, *args, **kwargs):
        if isinstance(expr, MatrixBase):
            return expr.applyfunc(lambda f: func(f, *args, **kwargs))
        elif iterable(expr):
            try:
                return expr.__class__([func(f, *args, **kwargs) for f in expr])
            except TypeError:
                return expr
        else:
            expr = sympify(expr)

            if use_add and expr.is_Add:
                return expr.__class__(*[ func(f, *args, **kwargs) for f in expr.args ])
            elif expr.is_Relational:
                return expr.__class__(func(expr.lhs, *args, **kwargs),
                                      func(expr.rhs, *args, **kwargs))
            else:
                return func(expr, *args, **kwargs)

    return threaded_func


def threaded(func):
    """Apply ``func`` to sub--elements of an object, including :class:`~.Add`.

    This decorator is intended to make it uniformly possible to apply a
    function to all elements of composite objects, e.g. matrices, lists, tuples
    and other iterable containers, or just expressions.

    This version of :func:`threaded` decorator allows threading over
    elements of :class:`~.Add` class. If this behavior is not desirable
    use :func:`xthreaded` decorator.

    Functions using this decorator must have the following signature::

      @threaded
      def function(expr, *args, **kwargs):

    """
    return threaded_factory(func, True)


def xthreaded(func):
    """Apply ``func`` to sub--elements of an object, excluding :class:`~.Add`.

    This decorator is intended to make it uniformly possible to apply a
    function to all elements of composite objects, e.g. matrices, lists, tuples
    and other iterable containers, or just expressions.

    This version of :func:`threaded` decorator disallows threading over
    elements of :class:`~.Add` class. If this behavior is not desirable
    use :func:`threaded` decorator.

    Functions using this decorator must have the following signature::

      @xthreaded
      def function(expr, *args, **kwargs):

    """
    return threaded_factory(func, False)


def conserve_mpmath_dps(func):
    """After the function finishes, resets the value of ``mpmath.mp.dps`` to
    the value it had before the function was run."""
    import mpmath

    def func_wrapper(*args, **kwargs):
        dps = mpmath.mp.dps
        try:
            return func(*args, **kwargs)
        finally:
            mpmath.mp.dps = dps

    func_wrapper = update_wrapper(func_wrapper, func)
    return func_wrapper


class no_attrs_in_subclass:
    """Don't 'inherit' certain attributes from a base class

    >>> from sympy.utilities.decorator import no_attrs_in_subclass

    >>> class A(object):
    ...     x = 'test'

    >>> A.x = no_attrs_in_subclass(A, A.x)

    >>> class B(A):
    ...     pass

    >>> hasattr(A, 'x')
    True
    >>> hasattr(B, 'x')
    False

    """
    def __init__(self, cls, f):
        self.cls = cls
        self.f = f

    def __get__(self, instance, owner=None):
        if owner == self.cls:
            if hasattr(self.f, '__get__'):
                return self.f.__get__(instance, owner)
            return self.f
        raise AttributeError


def doctest_depends_on(exe=None, modules=None, disable_viewers=None,
                       python_version=None, ground_types=None):
    """
    Adds metadata about the dependencies which need to be met for doctesting
    the docstrings of the decorated objects.

    ``exe`` should be a list of executables

    ``modules`` should be a list of modules

    ``disable_viewers`` should be a list of viewers for :func:`~sympy.printing.preview.preview` to disable

    ``python_version`` should be the minimum Python version required, as a tuple
    (like ``(3, 0)``)
    """
    dependencies = {}
    if exe is not None:
        dependencies['executables'] = exe
    if modules is not None:
        dependencies['modules'] = modules
    if disable_viewers is not None:
        dependencies['disable_viewers'] = disable_viewers
    if python_version is not None:
        dependencies['python_version'] = python_version
    if ground_types is not None:
        dependencies['ground_types'] = ground_types

    def skiptests():
        from sympy.testing.runtests import DependencyError, SymPyDocTests, PyTestReporter # lazy import
        r = PyTestReporter()
        t = SymPyDocTests(r, None)
        try:
            t._check_dependencies(**dependencies)
        except DependencyError:
            return True  # Skip doctests
        else:
            return False # Run doctests

    def depends_on_deco(fn):
        fn._doctest_depends_on = dependencies
        fn.__doctest_skip__ = skiptests

        if inspect.isclass(fn):
            fn._doctest_depdends_on = no_attrs_in_subclass(
                fn, fn._doctest_depends_on)
            fn.__doctest_skip__ = no_attrs_in_subclass(
                fn, fn.__doctest_skip__)
        return fn

    return depends_on_deco


def public(obj: T) -> T:
    """
    Append ``obj``'s name to global ``__all__`` variable (call site).

    By using this decorator on functions or classes you achieve the same goal
    as by filling ``__all__`` variables manually, you just do not have to repeat
    yourself (object's name). You also know if object is public at definition
    site, not at some random location (where ``__all__`` was set).

    Note that in multiple decorator setup (in almost all cases) ``@public``
    decorator must be applied before any other decorators, because it relies
    on the pointer to object's global namespace. If you apply other decorators
    first, ``@public`` may end up modifying the wrong namespace.

    Examples
    ========

    >>> from sympy.utilities.decorator import public

    >>> __all__ # noqa: F821
    Traceback (most recent call last):
    ...
    NameError: name '__all__' is not defined

    >>> @public
    ... def some_function():
    ...     pass

    >>> __all__ # noqa: F821
    ['some_function']

    """
    if isinstance(obj, types.FunctionType):
        ns = obj.__globals__
        name = obj.__name__
    elif isinstance(obj, (type(type), type)):
        ns = sys.modules[obj.__module__].__dict__
        name = obj.__name__
    else:
        raise TypeError("expected a function or a class, got %s" % obj)

    if "__all__" not in ns:
        ns["__all__"] = [name]
    else:
        ns["__all__"].append(name)

    return obj


def memoize_property(propfunc):
    """Property decorator that caches the value of potentially expensive
    ``propfunc`` after the first evaluation. The cached value is stored in
    the corresponding property name with an attached underscore."""
    attrname = '_' + propfunc.__name__
    sentinel = object()

    @wraps(propfunc)
    def accessor(self):
        val = getattr(self, attrname, sentinel)
        if val is sentinel:
            val = propfunc(self)
            setattr(self, attrname, val)
        return val

    return property(accessor)


def deprecated(message, *, deprecated_since_version,
               active_deprecations_target, stacklevel=3):
    '''
    Mark a function as deprecated.

    This decorator should be used if an entire function or class is
    deprecated. If only a certain functionality is deprecated, you should use
    :func:`~.warns_deprecated_sympy` directly. This decorator is just a
    convenience. There is no functional difference between using this
    decorator and calling ``warns_deprecated_sympy()`` at the top of the
    function.

    The decorator takes the same arguments as
    :func:`~.warns_deprecated_sympy`. See its
    documentation for details on what the keywords to this decorator do.

    See the :ref:`deprecation-policy` document for details on when and how
    things should be deprecated in SymPy.

    Examples
    ========

    >>> from sympy.utilities.decorator import deprecated
    >>> from sympy import simplify
    >>> @deprecated("""\
    ... The simplify_this(expr) function is deprecated. Use simplify(expr)
    ... instead.""", deprecated_since_version="1.1",
    ... active_deprecations_target='simplify-this-deprecation')
    ... def simplify_this(expr):
    ...     """
    ...     Simplify ``expr``.
    ...
    ...     .. deprecated:: 1.1
    ...
    ...        The ``simplify_this`` function is deprecated. Use :func:`simplify`
    ...        instead. See its documentation for more information. See
    ...        :ref:`simplify-this-deprecation` for details.
    ...
    ...     """
    ...     return simplify(expr)
    >>> from sympy.abc import x
    >>> simplify_this(x*(x + 1) - x**2) # doctest: +SKIP
    <stdin>:1: SymPyDeprecationWarning:
    <BLANKLINE>
    The simplify_this(expr) function is deprecated. Use simplify(expr)
    instead.
    <BLANKLINE>
    See https://docs.sympy.org/latest/explanation/active-deprecations.html#simplify-this-deprecation
    for details.
    <BLANKLINE>
    This has been deprecated since SymPy version 1.1. It
    will be removed in a future version of SymPy.
    <BLANKLINE>
      simplify_this(x)
    x

    See Also
    ========
    sympy.utilities.exceptions.SymPyDeprecationWarning
    sympy.utilities.exceptions.sympy_deprecation_warning
    sympy.utilities.exceptions.ignore_warnings
    sympy.testing.pytest.warns_deprecated_sympy

    '''
    decorator_kwargs = {"deprecated_since_version": deprecated_since_version,
               "active_deprecations_target": active_deprecations_target}
    def deprecated_decorator(wrapped):
        if hasattr(wrapped, '__mro__'):  # wrapped is actually a class
            class wrapper(wrapped):
                __doc__ = wrapped.__doc__
                __module__ = wrapped.__module__
                _sympy_deprecated_func = wrapped
                if '__new__' in wrapped.__dict__:
                    def __new__(cls, *args, **kwargs):
                        sympy_deprecation_warning(message, **decorator_kwargs, stacklevel=stacklevel)
                        return super().__new__(cls, *args, **kwargs)
                else:
                    def __init__(self, *args, **kwargs):
                        sympy_deprecation_warning(message, **decorator_kwargs, stacklevel=stacklevel)
                        super().__init__(*args, **kwargs)
            wrapper.__name__ = wrapped.__name__
        else:
            @wraps(wrapped)
            def wrapper(*args, **kwargs):
                sympy_deprecation_warning(message, **decorator_kwargs, stacklevel=stacklevel)
                return wrapped(*args, **kwargs)
            wrapper._sympy_deprecated_func = wrapped
        return wrapper
    return deprecated_decorator

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\bs4\css.py ===
"""Integration code for CSS selectors using `Soup Sieve <https://facelessuser.github.io/soupsieve/>`_ (pypi: ``soupsieve``).

Acquire a `CSS` object through the `element.Tag.css` attribute of
the starting point of your CSS selector, or (if you want to run a
selector against the entire document) of the `BeautifulSoup` object
itself.

The main advantage of doing this instead of using ``soupsieve``
functions is that you don't need to keep passing the `element.Tag` to be
selected against, since the `CSS` object is permanently scoped to that
`element.Tag`.

"""

from __future__ import annotations

from types import ModuleType
from typing import (
    Any,
    cast,
    Iterable,
    Iterator,
    Optional,
    TYPE_CHECKING,
)
import warnings
from bs4._typing import _NamespaceMapping

if TYPE_CHECKING:
    from soupsieve import SoupSieve
    from bs4 import element
    from bs4.element import ResultSet, Tag

soupsieve: Optional[ModuleType]
try:
    import soupsieve
except ImportError:
    soupsieve = None
    warnings.warn(
        "The soupsieve package is not installed. CSS selectors cannot be used."
    )


class CSS(object):
    """A proxy object against the ``soupsieve`` library, to simplify its
    CSS selector API.

    You don't need to instantiate this class yourself; instead, use
    `element.Tag.css`.

    :param tag: All CSS selectors run by this object will use this as
        their starting point.

    :param api: An optional drop-in replacement for the ``soupsieve`` module,
        intended for use in unit tests.
    """

    def __init__(self, tag: element.Tag, api: Optional[ModuleType] = None):
        if api is None:
            api = soupsieve
        if api is None:
            raise NotImplementedError(
                "Cannot execute CSS selectors because the soupsieve package is not installed."
            )
        self.api = api
        self.tag = tag

    def escape(self, ident: str) -> str:
        """Escape a CSS identifier.

        This is a simple wrapper around `soupsieve.escape() <https://facelessuser.github.io/soupsieve/api/#soupsieveescape>`_. See the
        documentation for that function for more information.
        """
        if soupsieve is None:
            raise NotImplementedError(
                "Cannot escape CSS identifiers because the soupsieve package is not installed."
            )
        return cast(str, self.api.escape(ident))

    def _ns(
        self, ns: Optional[_NamespaceMapping], select: str
    ) -> Optional[_NamespaceMapping]:
        """Normalize a dictionary of namespaces."""
        if not isinstance(select, self.api.SoupSieve) and ns is None:
            # If the selector is a precompiled pattern, it already has
            # a namespace context compiled in, which cannot be
            # replaced.
            ns = self.tag._namespaces
        return ns

    def _rs(self, results: Iterable[Tag]) -> ResultSet[Tag]:
        """Normalize a list of results to a py:class:`ResultSet`.

        A py:class:`ResultSet` is more consistent with the rest of
        Beautiful Soup's API, and :py:meth:`ResultSet.__getattr__` has
        a helpful error message if you try to treat a list of results
        as a single result (a common mistake).
        """
        # Import here to avoid circular import
        from bs4 import ResultSet

        return ResultSet(None, results)

    def compile(
        self,
        select: str,
        namespaces: Optional[_NamespaceMapping] = None,
        flags: int = 0,
        **kwargs: Any,
    ) -> SoupSieve:
        """Pre-compile a selector and return the compiled object.

        :param selector: A CSS selector.

        :param namespaces: A dictionary mapping namespace prefixes
           used in the CSS selector to namespace URIs. By default,
           Beautiful Soup will use the prefixes it encountered while
           parsing the document.

        :param flags: Flags to be passed into Soup Sieve's
            `soupsieve.compile() <https://facelessuser.github.io/soupsieve/api/#soupsievecompile>`_ method.

        :param kwargs: Keyword arguments to be passed into Soup Sieve's
           `soupsieve.compile() <https://facelessuser.github.io/soupsieve/api/#soupsievecompile>`_ method.

        :return: A precompiled selector object.
        :rtype: soupsieve.SoupSieve
        """
        return self.api.compile(select, self._ns(namespaces, select), flags, **kwargs)

    def select_one(
        self,
        select: str,
        namespaces: Optional[_NamespaceMapping] = None,
        flags: int = 0,
        **kwargs: Any,
    ) -> element.Tag | None:
        """Perform a CSS selection operation on the current Tag and return the
        first result, if any.

        This uses the Soup Sieve library. For more information, see
        that library's documentation for the `soupsieve.select_one() <https://facelessuser.github.io/soupsieve/api/#soupsieveselect_one>`_ method.

        :param selector: A CSS selector.

        :param namespaces: A dictionary mapping namespace prefixes
           used in the CSS selector to namespace URIs. By default,
           Beautiful Soup will use the prefixes it encountered while
           parsing the document.

        :param flags: Flags to be passed into Soup Sieve's
            `soupsieve.select_one() <https://facelessuser.github.io/soupsieve/api/#soupsieveselect_one>`_ method.

        :param kwargs: Keyword arguments to be passed into Soup Sieve's
           `soupsieve.select_one() <https://facelessuser.github.io/soupsieve/api/#soupsieveselect_one>`_ method.
        """
        return self.api.select_one(
            select, self.tag, self._ns(namespaces, select), flags, **kwargs
        )

    def select(
        self,
        select: str,
        namespaces: Optional[_NamespaceMapping] = None,
        limit: int = 0,
        flags: int = 0,
        **kwargs: Any,
    ) -> ResultSet[element.Tag]:
        """Perform a CSS selection operation on the current `element.Tag`.

        This uses the Soup Sieve library. For more information, see
        that library's documentation for the `soupsieve.select() <https://facelessuser.github.io/soupsieve/api/#soupsieveselect>`_ method.

        :param selector: A CSS selector.

        :param namespaces: A dictionary mapping namespace prefixes
            used in the CSS selector to namespace URIs. By default,
            Beautiful Soup will pass in the prefixes it encountered while
            parsing the document.

        :param limit: After finding this number of results, stop looking.

        :param flags: Flags to be passed into Soup Sieve's
            `soupsieve.select() <https://facelessuser.github.io/soupsieve/api/#soupsieveselect>`_ method.

        :param kwargs: Keyword arguments to be passed into Soup Sieve's
           `soupsieve.select() <https://facelessuser.github.io/soupsieve/api/#soupsieveselect>`_ method.
        """
        if limit is None:
            limit = 0

        return self._rs(
            self.api.select(
                select, self.tag, self._ns(namespaces, select), limit, flags, **kwargs
            )
        )

    def iselect(
        self,
        select: str,
        namespaces: Optional[_NamespaceMapping] = None,
        limit: int = 0,
        flags: int = 0,
        **kwargs: Any,
    ) -> Iterator[element.Tag]:
        """Perform a CSS selection operation on the current `element.Tag`.

        This uses the Soup Sieve library. For more information, see
        that library's documentation for the `soupsieve.iselect()
        <https://facelessuser.github.io/soupsieve/api/#soupsieveiselect>`_
        method. It is the same as select(), but it returns a generator
        instead of a list.

        :param selector: A string containing a CSS selector.

        :param namespaces: A dictionary mapping namespace prefixes
            used in the CSS selector to namespace URIs. By default,
            Beautiful Soup will pass in the prefixes it encountered while
            parsing the document.

        :param limit: After finding this number of results, stop looking.

        :param flags: Flags to be passed into Soup Sieve's
            `soupsieve.iselect() <https://facelessuser.github.io/soupsieve/api/#soupsieveiselect>`_ method.

        :param kwargs: Keyword arguments to be passed into Soup Sieve's
           `soupsieve.iselect() <https://facelessuser.github.io/soupsieve/api/#soupsieveiselect>`_ method.
        """
        return self.api.iselect(
            select, self.tag, self._ns(namespaces, select), limit, flags, **kwargs
        )

    def closest(
        self,
        select: str,
        namespaces: Optional[_NamespaceMapping] = None,
        flags: int = 0,
        **kwargs: Any,
    ) -> Optional[element.Tag]:
        """Find the `element.Tag` closest to this one that matches the given selector.

        This uses the Soup Sieve library. For more information, see
        that library's documentation for the `soupsieve.closest()
        <https://facelessuser.github.io/soupsieve/api/#soupsieveclosest>`_
        method.

        :param selector: A string containing a CSS selector.

        :param namespaces: A dictionary mapping namespace prefixes
            used in the CSS selector to namespace URIs. By default,
            Beautiful Soup will pass in the prefixes it encountered while
            parsing the document.

        :param flags: Flags to be passed into Soup Sieve's
            `soupsieve.closest() <https://facelessuser.github.io/soupsieve/api/#soupsieveclosest>`_ method.

        :param kwargs: Keyword arguments to be passed into Soup Sieve's
           `soupsieve.closest() <https://facelessuser.github.io/soupsieve/api/#soupsieveclosest>`_ method.

        """
        return self.api.closest(
            select, self.tag, self._ns(namespaces, select), flags, **kwargs
        )

    def match(
        self,
        select: str,
        namespaces: Optional[_NamespaceMapping] = None,
        flags: int = 0,
        **kwargs: Any,
    ) -> bool:
        """Check whether or not this `element.Tag` matches the given CSS selector.

        This uses the Soup Sieve library. For more information, see
        that library's documentation for the `soupsieve.match()
        <https://facelessuser.github.io/soupsieve/api/#soupsievematch>`_
        method.

        :param: a CSS selector.

        :param namespaces: A dictionary mapping namespace prefixes
            used in the CSS selector to namespace URIs. By default,
            Beautiful Soup will pass in the prefixes it encountered while
            parsing the document.

        :param flags: Flags to be passed into Soup Sieve's
            `soupsieve.match()
            <https://facelessuser.github.io/soupsieve/api/#soupsievematch>`_
            method.

        :param kwargs: Keyword arguments to be passed into SoupSieve's
            `soupsieve.match()
            <https://facelessuser.github.io/soupsieve/api/#soupsievematch>`_
            method.
        """
        return cast(
            bool,
            self.api.match(
                select, self.tag, self._ns(namespaces, select), flags, **kwargs
            ),
        )

    def filter(
        self,
        select: str,
        namespaces: Optional[_NamespaceMapping] = None,
        flags: int = 0,
        **kwargs: Any,
    ) -> ResultSet[element.Tag]:
        """Filter this `element.Tag`'s direct children based on the given CSS selector.

        This uses the Soup Sieve library. It works the same way as
        passing a `element.Tag` into that library's `soupsieve.filter()
        <https://facelessuser.github.io/soupsieve/api/#soupsievefilter>`_
        method. For more information, see the documentation for
        `soupsieve.filter()
        <https://facelessuser.github.io/soupsieve/api/#soupsievefilter>`_.

        :param namespaces: A dictionary mapping namespace prefixes
            used in the CSS selector to namespace URIs. By default,
            Beautiful Soup will pass in the prefixes it encountered while
            parsing the document.

        :param flags: Flags to be passed into Soup Sieve's
            `soupsieve.filter()
            <https://facelessuser.github.io/soupsieve/api/#soupsievefilter>`_
            method.

        :param kwargs: Keyword arguments to be passed into SoupSieve's
            `soupsieve.filter()
            <https://facelessuser.github.io/soupsieve/api/#soupsievefilter>`_
            method.
        """
        return self._rs(
            self.api.filter(
                select, self.tag, self._ns(namespaces, select), flags, **kwargs
            )
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\ort_format_model\ort_flatbuffers_py\fbs\Attribute.py ===
# automatically generated by the FlatBuffers compiler, do not modify

# namespace: fbs

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Attribute(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Attribute()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsAttribute(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def AttributeBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x4F\x52\x54\x4D", size_prefixed=size_prefixed)

    # Attribute
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Attribute
    def Name(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Attribute
    def DocString(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Attribute
    def Type(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # Attribute
    def F(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return 0.0

    # Attribute
    def I(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int64Flags, o + self._tab.Pos)
        return 0

    # Attribute
    def S(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Attribute
    def T(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from ort_flatbuffers_py.fbs.Tensor import Tensor
            obj = Tensor()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Attribute
    def G(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from ort_flatbuffers_py.fbs.Graph import Graph
            obj = Graph()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Attribute
    def Floats(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Float32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # Attribute
    def FloatsAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Float32Flags, o)
        return 0

    # Attribute
    def FloatsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Attribute
    def FloatsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        return o == 0

    # Attribute
    def Ints(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int64Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 8))
        return 0

    # Attribute
    def IntsAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int64Flags, o)
        return 0

    # Attribute
    def IntsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Attribute
    def IntsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        return o == 0

    # Attribute
    def Strings(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(24))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.String(a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return ""

    # Attribute
    def StringsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(24))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Attribute
    def StringsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(24))
        return o == 0

    # Attribute
    def Tensors(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(26))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from ort_flatbuffers_py.fbs.Tensor import Tensor
            obj = Tensor()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Attribute
    def TensorsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(26))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Attribute
    def TensorsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(26))
        return o == 0

    # Attribute
    def Graphs(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(28))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from ort_flatbuffers_py.fbs.Graph import Graph
            obj = Graph()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Attribute
    def GraphsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(28))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Attribute
    def GraphsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(28))
        return o == 0

def AttributeStart(builder):
    builder.StartObject(13)

def Start(builder):
    AttributeStart(builder)

def AttributeAddName(builder, name):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0)

def AddName(builder, name):
    AttributeAddName(builder, name)

def AttributeAddDocString(builder, docString):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(docString), 0)

def AddDocString(builder, docString):
    AttributeAddDocString(builder, docString)

def AttributeAddType(builder, type):
    builder.PrependInt32Slot(2, type, 0)

def AddType(builder, type):
    AttributeAddType(builder, type)

def AttributeAddF(builder, f):
    builder.PrependFloat32Slot(3, f, 0.0)

def AddF(builder, f):
    AttributeAddF(builder, f)

def AttributeAddI(builder, i):
    builder.PrependInt64Slot(4, i, 0)

def AddI(builder, i):
    AttributeAddI(builder, i)

def AttributeAddS(builder, s):
    builder.PrependUOffsetTRelativeSlot(5, flatbuffers.number_types.UOffsetTFlags.py_type(s), 0)

def AddS(builder, s):
    AttributeAddS(builder, s)

def AttributeAddT(builder, t):
    builder.PrependUOffsetTRelativeSlot(6, flatbuffers.number_types.UOffsetTFlags.py_type(t), 0)

def AddT(builder, t):
    AttributeAddT(builder, t)

def AttributeAddG(builder, g):
    builder.PrependUOffsetTRelativeSlot(7, flatbuffers.number_types.UOffsetTFlags.py_type(g), 0)

def AddG(builder, g):
    AttributeAddG(builder, g)

def AttributeAddFloats(builder, floats):
    builder.PrependUOffsetTRelativeSlot(8, flatbuffers.number_types.UOffsetTFlags.py_type(floats), 0)

def AddFloats(builder, floats):
    AttributeAddFloats(builder, floats)

def AttributeStartFloatsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartFloatsVector(builder, numElems: int) -> int:
    return AttributeStartFloatsVector(builder, numElems)

def AttributeAddInts(builder, ints):
    builder.PrependUOffsetTRelativeSlot(9, flatbuffers.number_types.UOffsetTFlags.py_type(ints), 0)

def AddInts(builder, ints):
    AttributeAddInts(builder, ints)

def AttributeStartIntsVector(builder, numElems):
    return builder.StartVector(8, numElems, 8)

def StartIntsVector(builder, numElems: int) -> int:
    return AttributeStartIntsVector(builder, numElems)

def AttributeAddStrings(builder, strings):
    builder.PrependUOffsetTRelativeSlot(10, flatbuffers.number_types.UOffsetTFlags.py_type(strings), 0)

def AddStrings(builder, strings):
    AttributeAddStrings(builder, strings)

def AttributeStartStringsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartStringsVector(builder, numElems: int) -> int:
    return AttributeStartStringsVector(builder, numElems)

def AttributeAddTensors(builder, tensors):
    builder.PrependUOffsetTRelativeSlot(11, flatbuffers.number_types.UOffsetTFlags.py_type(tensors), 0)

def AddTensors(builder, tensors):
    AttributeAddTensors(builder, tensors)

def AttributeStartTensorsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartTensorsVector(builder, numElems: int) -> int:
    return AttributeStartTensorsVector(builder, numElems)

def AttributeAddGraphs(builder, graphs):
    builder.PrependUOffsetTRelativeSlot(12, flatbuffers.number_types.UOffsetTFlags.py_type(graphs), 0)

def AddGraphs(builder, graphs):
    AttributeAddGraphs(builder, graphs)

def AttributeStartGraphsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartGraphsVector(builder, numElems: int) -> int:
    return AttributeStartGraphsVector(builder, numElems)

def AttributeEnd(builder):
    return builder.EndObject()

def End(builder):
    return AttributeEnd(builder)