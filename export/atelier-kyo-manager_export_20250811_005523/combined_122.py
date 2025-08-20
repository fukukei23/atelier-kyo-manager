
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\req\constructors.py ===
"""Backing implementation for InstallRequirement's various constructors

The idea here is that these formed a major chunk of InstallRequirement's size
so, moving them and support code dedicated to them outside of that class
helps creates for better understandability for the rest of the code.

These are meant to be used elsewhere within pip to create instances of
InstallRequirement.
"""

import copy
import logging
import os
import re
from dataclasses import dataclass
from typing import Collection, Dict, List, Optional, Set, Tuple, Union

from pip._vendor.packaging.markers import Marker
from pip._vendor.packaging.requirements import InvalidRequirement, Requirement
from pip._vendor.packaging.specifiers import Specifier

from pip._internal.exceptions import InstallationError
from pip._internal.models.index import PyPI, TestPyPI
from pip._internal.models.link import Link
from pip._internal.models.wheel import Wheel
from pip._internal.req.req_file import ParsedRequirement
from pip._internal.req.req_install import InstallRequirement
from pip._internal.utils.filetypes import is_archive_file
from pip._internal.utils.misc import is_installable_dir
from pip._internal.utils.packaging import get_requirement
from pip._internal.utils.urls import path_to_url
from pip._internal.vcs import is_url, vcs

__all__ = [
    "install_req_from_editable",
    "install_req_from_line",
    "parse_editable",
]

logger = logging.getLogger(__name__)
operators = Specifier._operators.keys()


def _strip_extras(path: str) -> Tuple[str, Optional[str]]:
    m = re.match(r"^(.+)(\[[^\]]+\])$", path)
    extras = None
    if m:
        path_no_extras = m.group(1)
        extras = m.group(2)
    else:
        path_no_extras = path

    return path_no_extras, extras


def convert_extras(extras: Optional[str]) -> Set[str]:
    if not extras:
        return set()
    return get_requirement("placeholder" + extras.lower()).extras


def _set_requirement_extras(req: Requirement, new_extras: Set[str]) -> Requirement:
    """
    Returns a new requirement based on the given one, with the supplied extras. If the
    given requirement already has extras those are replaced (or dropped if no new extras
    are given).
    """
    match: Optional[re.Match[str]] = re.fullmatch(
        # see https://peps.python.org/pep-0508/#complete-grammar
        r"([\w\t .-]+)(\[[^\]]*\])?(.*)",
        str(req),
        flags=re.ASCII,
    )
    # ireq.req is a valid requirement so the regex should always match
    assert (
        match is not None
    ), f"regex match on requirement {req} failed, this should never happen"
    pre: Optional[str] = match.group(1)
    post: Optional[str] = match.group(3)
    assert (
        pre is not None and post is not None
    ), f"regex group selection for requirement {req} failed, this should never happen"
    extras: str = "[{}]".format(",".join(sorted(new_extras)) if new_extras else "")
    return get_requirement(f"{pre}{extras}{post}")


def parse_editable(editable_req: str) -> Tuple[Optional[str], str, Set[str]]:
    """Parses an editable requirement into:
        - a requirement name
        - an URL
        - extras
        - editable options
    Accepted requirements:
        svn+http://blahblah@rev#egg=Foobar[baz]&subdirectory=version_subdir
        .[some_extra]
    """

    url = editable_req

    # If a file path is specified with extras, strip off the extras.
    url_no_extras, extras = _strip_extras(url)

    if os.path.isdir(url_no_extras):
        # Treating it as code that has already been checked out
        url_no_extras = path_to_url(url_no_extras)

    if url_no_extras.lower().startswith("file:"):
        package_name = Link(url_no_extras).egg_fragment
        if extras:
            return (
                package_name,
                url_no_extras,
                get_requirement("placeholder" + extras.lower()).extras,
            )
        else:
            return package_name, url_no_extras, set()

    for version_control in vcs:
        if url.lower().startswith(f"{version_control}:"):
            url = f"{version_control}+{url}"
            break

    link = Link(url)

    if not link.is_vcs:
        backends = ", ".join(vcs.all_schemes)
        raise InstallationError(
            f"{editable_req} is not a valid editable requirement. "
            f"It should either be a path to a local project or a VCS URL "
            f"(beginning with {backends})."
        )

    package_name = link.egg_fragment
    if not package_name:
        raise InstallationError(
            f"Could not detect requirement name for '{editable_req}', "
            "please specify one with #egg=your_package_name"
        )
    return package_name, url, set()


def check_first_requirement_in_file(filename: str) -> None:
    """Check if file is parsable as a requirements file.

    This is heavily based on ``pkg_resources.parse_requirements``, but
    simplified to just check the first meaningful line.

    :raises InvalidRequirement: If the first meaningful line cannot be parsed
        as an requirement.
    """
    with open(filename, encoding="utf-8", errors="ignore") as f:
        # Create a steppable iterator, so we can handle \-continuations.
        lines = (
            line
            for line in (line.strip() for line in f)
            if line and not line.startswith("#")  # Skip blank lines/comments.
        )

        for line in lines:
            # Drop comments -- a hash without a space may be in a URL.
            if " #" in line:
                line = line[: line.find(" #")]
            # If there is a line continuation, drop it, and append the next line.
            if line.endswith("\\"):
                line = line[:-2].strip() + next(lines, "")
            get_requirement(line)
            return


def deduce_helpful_msg(req: str) -> str:
    """Returns helpful msg in case requirements file does not exist,
    or cannot be parsed.

    :params req: Requirements file path
    """
    if not os.path.exists(req):
        return f" File '{req}' does not exist."
    msg = " The path does exist. "
    # Try to parse and check if it is a requirements file.
    try:
        check_first_requirement_in_file(req)
    except InvalidRequirement:
        logger.debug("Cannot parse '%s' as requirements file", req)
    else:
        msg += (
            f"The argument you provided "
            f"({req}) appears to be a"
            f" requirements file. If that is the"
            f" case, use the '-r' flag to install"
            f" the packages specified within it."
        )
    return msg


@dataclass(frozen=True)
class RequirementParts:
    requirement: Optional[Requirement]
    link: Optional[Link]
    markers: Optional[Marker]
    extras: Set[str]


def parse_req_from_editable(editable_req: str) -> RequirementParts:
    name, url, extras_override = parse_editable(editable_req)

    if name is not None:
        try:
            req: Optional[Requirement] = get_requirement(name)
        except InvalidRequirement as exc:
            raise InstallationError(f"Invalid requirement: {name!r}: {exc}")
    else:
        req = None

    link = Link(url)

    return RequirementParts(req, link, None, extras_override)


# ---- The actual constructors follow ----


def install_req_from_editable(
    editable_req: str,
    comes_from: Optional[Union[InstallRequirement, str]] = None,
    *,
    use_pep517: Optional[bool] = None,
    isolated: bool = False,
    global_options: Optional[List[str]] = None,
    hash_options: Optional[Dict[str, List[str]]] = None,
    constraint: bool = False,
    user_supplied: bool = False,
    permit_editable_wheels: bool = False,
    config_settings: Optional[Dict[str, Union[str, List[str]]]] = None,
) -> InstallRequirement:
    parts = parse_req_from_editable(editable_req)

    return InstallRequirement(
        parts.requirement,
        comes_from=comes_from,
        user_supplied=user_supplied,
        editable=True,
        permit_editable_wheels=permit_editable_wheels,
        link=parts.link,
        constraint=constraint,
        use_pep517=use_pep517,
        isolated=isolated,
        global_options=global_options,
        hash_options=hash_options,
        config_settings=config_settings,
        extras=parts.extras,
    )


def _looks_like_path(name: str) -> bool:
    """Checks whether the string "looks like" a path on the filesystem.

    This does not check whether the target actually exists, only judge from the
    appearance.

    Returns true if any of the following conditions is true:
    * a path separator is found (either os.path.sep or os.path.altsep);
    * a dot is found (which represents the current directory).
    """
    if os.path.sep in name:
        return True
    if os.path.altsep is not None and os.path.altsep in name:
        return True
    if name.startswith("."):
        return True
    return False


def _get_url_from_path(path: str, name: str) -> Optional[str]:
    """
    First, it checks whether a provided path is an installable directory. If it
    is, returns the path.

    If false, check if the path is an archive file (such as a .whl).
    The function checks if the path is a file. If false, if the path has
    an @, it will treat it as a PEP 440 URL requirement and return the path.
    """
    if _looks_like_path(name) and os.path.isdir(path):
        if is_installable_dir(path):
            return path_to_url(path)
        # TODO: The is_installable_dir test here might not be necessary
        #       now that it is done in load_pyproject_toml too.
        raise InstallationError(
            f"Directory {name!r} is not installable. Neither 'setup.py' "
            "nor 'pyproject.toml' found."
        )
    if not is_archive_file(path):
        return None
    if os.path.isfile(path):
        return path_to_url(path)
    urlreq_parts = name.split("@", 1)
    if len(urlreq_parts) >= 2 and not _looks_like_path(urlreq_parts[0]):
        # If the path contains '@' and the part before it does not look
        # like a path, try to treat it as a PEP 440 URL req instead.
        return None
    logger.warning(
        "Requirement %r looks like a filename, but the file does not exist",
        name,
    )
    return path_to_url(path)


def parse_req_from_line(name: str, line_source: Optional[str]) -> RequirementParts:
    if is_url(name):
        marker_sep = "; "
    else:
        marker_sep = ";"
    if marker_sep in name:
        name, markers_as_string = name.split(marker_sep, 1)
        markers_as_string = markers_as_string.strip()
        if not markers_as_string:
            markers = None
        else:
            markers = Marker(markers_as_string)
    else:
        markers = None
    name = name.strip()
    req_as_string = None
    path = os.path.normpath(os.path.abspath(name))
    link = None
    extras_as_string = None

    if is_url(name):
        link = Link(name)
    else:
        p, extras_as_string = _strip_extras(path)
        url = _get_url_from_path(p, name)
        if url is not None:
            link = Link(url)

    # it's a local file, dir, or url
    if link:
        # Handle relative file URLs
        if link.scheme == "file" and re.search(r"\.\./", link.url):
            link = Link(path_to_url(os.path.normpath(os.path.abspath(link.path))))
        # wheel file
        if link.is_wheel:
            wheel = Wheel(link.filename)  # can raise InvalidWheelFilename
            req_as_string = f"{wheel.name}=={wheel.version}"
        else:
            # set the req to the egg fragment.  when it's not there, this
            # will become an 'unnamed' requirement
            req_as_string = link.egg_fragment

    # a requirement specifier
    else:
        req_as_string = name

    extras = convert_extras(extras_as_string)

    def with_source(text: str) -> str:
        if not line_source:
            return text
        return f"{text} (from {line_source})"

    def _parse_req_string(req_as_string: str) -> Requirement:
        try:
            return get_requirement(req_as_string)
        except InvalidRequirement as exc:
            if os.path.sep in req_as_string:
                add_msg = "It looks like a path."
                add_msg += deduce_helpful_msg(req_as_string)
            elif "=" in req_as_string and not any(
                op in req_as_string for op in operators
            ):
                add_msg = "= is not a valid operator. Did you mean == ?"
            else:
                add_msg = ""
            msg = with_source(f"Invalid requirement: {req_as_string!r}: {exc}")
            if add_msg:
                msg += f"\nHint: {add_msg}"
            raise InstallationError(msg)

    if req_as_string is not None:
        req: Optional[Requirement] = _parse_req_string(req_as_string)
    else:
        req = None

    return RequirementParts(req, link, markers, extras)


def install_req_from_line(
    name: str,
    comes_from: Optional[Union[str, InstallRequirement]] = None,
    *,
    use_pep517: Optional[bool] = None,
    isolated: bool = False,
    global_options: Optional[List[str]] = None,
    hash_options: Optional[Dict[str, List[str]]] = None,
    constraint: bool = False,
    line_source: Optional[str] = None,
    user_supplied: bool = False,
    config_settings: Optional[Dict[str, Union[str, List[str]]]] = None,
) -> InstallRequirement:
    """Creates an InstallRequirement from a name, which might be a
    requirement, directory containing 'setup.py', filename, or URL.

    :param line_source: An optional string describing where the line is from,
        for logging purposes in case of an error.
    """
    parts = parse_req_from_line(name, line_source)

    return InstallRequirement(
        parts.requirement,
        comes_from,
        link=parts.link,
        markers=parts.markers,
        use_pep517=use_pep517,
        isolated=isolated,
        global_options=global_options,
        hash_options=hash_options,
        config_settings=config_settings,
        constraint=constraint,
        extras=parts.extras,
        user_supplied=user_supplied,
    )


def install_req_from_req_string(
    req_string: str,
    comes_from: Optional[InstallRequirement] = None,
    isolated: bool = False,
    use_pep517: Optional[bool] = None,
    user_supplied: bool = False,
) -> InstallRequirement:
    try:
        req = get_requirement(req_string)
    except InvalidRequirement as exc:
        raise InstallationError(f"Invalid requirement: {req_string!r}: {exc}")

    domains_not_allowed = [
        PyPI.file_storage_domain,
        TestPyPI.file_storage_domain,
    ]
    if (
        req.url
        and comes_from
        and comes_from.link
        and comes_from.link.netloc in domains_not_allowed
    ):
        # Explicitly disallow pypi packages that depend on external urls
        raise InstallationError(
            "Packages installed from PyPI cannot depend on packages "
            "which are not also hosted on PyPI.\n"
            f"{comes_from.name} depends on {req} "
        )

    return InstallRequirement(
        req,
        comes_from,
        isolated=isolated,
        use_pep517=use_pep517,
        user_supplied=user_supplied,
    )


def install_req_from_parsed_requirement(
    parsed_req: ParsedRequirement,
    isolated: bool = False,
    use_pep517: Optional[bool] = None,
    user_supplied: bool = False,
    config_settings: Optional[Dict[str, Union[str, List[str]]]] = None,
) -> InstallRequirement:
    if parsed_req.is_editable:
        req = install_req_from_editable(
            parsed_req.requirement,
            comes_from=parsed_req.comes_from,
            use_pep517=use_pep517,
            constraint=parsed_req.constraint,
            isolated=isolated,
            user_supplied=user_supplied,
            config_settings=config_settings,
        )

    else:
        req = install_req_from_line(
            parsed_req.requirement,
            comes_from=parsed_req.comes_from,
            use_pep517=use_pep517,
            isolated=isolated,
            global_options=(
                parsed_req.options.get("global_options", [])
                if parsed_req.options
                else []
            ),
            hash_options=(
                parsed_req.options.get("hashes", {}) if parsed_req.options else {}
            ),
            constraint=parsed_req.constraint,
            line_source=parsed_req.line_source,
            user_supplied=user_supplied,
            config_settings=config_settings,
        )
    return req


def install_req_from_link_and_ireq(
    link: Link, ireq: InstallRequirement
) -> InstallRequirement:
    return InstallRequirement(
        req=ireq.req,
        comes_from=ireq.comes_from,
        editable=ireq.editable,
        link=link,
        markers=ireq.markers,
        use_pep517=ireq.use_pep517,
        isolated=ireq.isolated,
        global_options=ireq.global_options,
        hash_options=ireq.hash_options,
        config_settings=ireq.config_settings,
        user_supplied=ireq.user_supplied,
    )


def install_req_drop_extras(ireq: InstallRequirement) -> InstallRequirement:
    """
    Creates a new InstallationRequirement using the given template but without
    any extras. Sets the original requirement as the new one's parent
    (comes_from).
    """
    return InstallRequirement(
        req=(
            _set_requirement_extras(ireq.req, set()) if ireq.req is not None else None
        ),
        comes_from=ireq,
        editable=ireq.editable,
        link=ireq.link,
        markers=ireq.markers,
        use_pep517=ireq.use_pep517,
        isolated=ireq.isolated,
        global_options=ireq.global_options,
        hash_options=ireq.hash_options,
        constraint=ireq.constraint,
        extras=[],
        config_settings=ireq.config_settings,
        user_supplied=ireq.user_supplied,
        permit_editable_wheels=ireq.permit_editable_wheels,
    )


def install_req_extend_extras(
    ireq: InstallRequirement,
    extras: Collection[str],
) -> InstallRequirement:
    """
    Returns a copy of an installation requirement with some additional extras.
    Makes a shallow copy of the ireq object.
    """
    result = copy.copy(ireq)
    result.extras = {*ireq.extras, *extras}
    result.req = (
        _set_requirement_extras(ireq.req, result.extras)
        if ireq.req is not None
        else None
    )
    return result

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\models\gpt2\convert_to_onnx.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""
This converts GPT2 model to onnx. Examples:
(1) Convert pretrained model 'gpt2' to ONNX
   python convert_to_onnx.py -m gpt2 --output gpt2.onnx
(2) Convert pretrained model 'distilgpt2' to ONNX, and use optimizer to get float16 model.
   python convert_to_onnx.py -m distilgpt2 --output distilgpt2_fp16.onnx -o -p fp16
(3) Convert a model check point to ONNX, and run optimization and int8 quantization
   python convert_to_onnx.py -m ./my_model_checkpoint/ --output my_model_int8.onnx -o -p int8

"""

import argparse
import csv
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy
import torch
from benchmark_helper import (
    Precision,
    create_onnxruntime_session,
    get_ort_environment_variables,
    prepare_environment,
    setup_logger,
)
from gpt2_helper import DEFAULT_TOLERANCE, MODEL_CLASSES, PRETRAINED_GPT2_MODELS, Gpt2Helper
from gpt2_tester import Gpt2Tester
from packaging import version
from quantize_helper import QuantizeHelper
from transformers import AutoConfig
from transformers import __version__ as transformers_version

from onnxruntime import __version__ as ort_version

logger = logging.getLogger("")


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model_name_or_path",
        required=True,
        type=str,
        help="Model path, or pretrained model name in the list: " + ", ".join(PRETRAINED_GPT2_MODELS),
    )

    parser.add_argument(
        "--model_class",
        required=False,
        type=str,
        default="GPT2LMHeadModel",
        choices=list(MODEL_CLASSES.keys()),
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )

    parser.add_argument(
        "--cache_dir",
        required=False,
        type=str,
        default=os.path.join(".", "cache_models"),
        help="Directory to cache pre-trained models",
    )

    parser.add_argument(
        "--output",
        required=False,
        type=str,
        default=os.path.join(".", "onnx_models"),
        help="Output directory, or model path ends with .onnx",
    )

    parser.add_argument(
        "-o",
        "--optimize_onnx",
        required=False,
        action="store_true",
        help="Use optimizer.py to optimize onnx model",
    )
    parser.set_defaults(optimize_onnx=False)

    parser.add_argument("--use_gpu", required=False, action="store_true", help="use GPU for inference")
    parser.set_defaults(use_gpu=False)

    parser.add_argument(
        "--provider",
        required=False,
        default=None,
        choices=["dml", "rocm", "migraphx", "cuda", "tensorrt"],
        help="use dml, rocm, cuda, tensorrt or migraphx for respective backend",
    )

    parser.add_argument(
        "--tolerance",
        required=False,
        type=float,
        default=0,
        help="the absolute and relative tolerance for parity verification",
    )

    parser.add_argument(
        "--input_test_file",
        "-i",
        required=False,
        type=str,
        default="",
        help="Path to the file with inputs to test with",
    )

    parser.add_argument(
        "-p",
        "--precision",
        required=False,
        type=Precision,
        default=Precision.FLOAT32,
        choices=list(Precision),
        help="Precision of model to run. fp32 for full precision, fp16 for half or mixed precision, and int8 for quantization",
    )

    parser.add_argument(
        "-t",
        "--test_cases",
        required=False,
        type=int,
        default=1000,
        help="Number of test cases per run for parity",
    )
    parser.add_argument(
        "-r",
        "--test_runs",
        required=False,
        type=int,
        default=10,
        help="Number of runs for parity. It is used for significance test.",
    )

    parser.add_argument("--verbose", required=False, action="store_true")
    parser.set_defaults(verbose=False)

    parser.add_argument("-e", "--use_external_data_format", required=False, action="store_true")
    parser.set_defaults(use_external_data_format=False)

    parser.add_argument("--overwrite", required=False, action="store_true")
    parser.set_defaults(overwrite=False)

    parser.add_argument(
        "--use_int64_inputs",
        required=False,
        action="store_true",
        help="Use int32 instead of int64 for input_ids, position_ids and attention_mask.",
    )
    parser.set_defaults(use_int64_inputs=False)

    parser.add_argument(
        "-s",
        "--stage",
        type=int,
        default=0,
        required=False,
        choices=[0, 1, 2],
        help="Stage in generation: 1 (initial decoder), 2 (decoder), 0 (both). "
        "1 - decode the first token when past_sequence_length is zero; "
        "2 - decode the remaining tokens when past_sequence_length is not zero; "
        "0 - one onnx model for both stages 1 and 2. "
        "Note that we will optimize 1 and 2 differently for best performance.",
    )

    fp16_option_group = parser.add_argument_group(
        'float to float16 conversion parameters that works when "--precision fp16" is specified'
    )

    fp16_option_group.add_argument(
        "-a",
        "--auto_mixed_precision",
        required=False,
        action="store_true",
        help="Convert to mixed precision automatically. Other float16 conversion parameters will be ignored.",
    )
    fp16_option_group.set_defaults(auto_mixed_precision=False)

    fp16_option_group.add_argument(
        "--keep_io_types",
        required=False,
        action="store_true",
        help="Use float32 for past inputs, present and logits outputs.",
    )
    fp16_option_group.set_defaults(keep_io_types=False)

    fp16_option_group.add_argument(
        "--io_block_list",
        nargs="+",
        default=[],
        help="List of inputs or outputs in float32 instead of float16",
    )

    fp16_option_group.add_argument(
        "--op_block_list",
        nargs="+",
        default=[],
        help="List of operators (like Add LayerNormalization SkipLayerNormalization EmbedLayerNormalization FastGelu) "
        "to compute in float32 instead of float16.",
    )

    fp16_option_group.add_argument(
        "--node_block_list",
        nargs="+",
        default=[],
        help="List of node names to compute in float32 instead of float16.",
    )

    fp16_option_group.add_argument(
        "--force_fp16_initializers",
        required=False,
        action="store_true",
        help="Convert all float initializers to float16.",
    )
    fp16_option_group.set_defaults(force_fp16_initializers=False)

    args = parser.parse_args(argv)

    return args


def get_onnx_model_size(onnx_path: str, use_external_data_format: bool):
    if not use_external_data_format:
        return os.path.getsize(onnx_path)
    else:
        return sum([f.stat().st_size for f in Path(onnx_path).parent.rglob("*")])


def get_latency_name(batch_size, sequence_length, past_sequence_length):
    return f"average_latency(batch_size={batch_size},sequence_length={sequence_length},past_sequence_length={past_sequence_length})"


def main(argv=None, experiment_name: str = "", run_id: str = "0", csv_filename: str = "gpt2_parity_results.csv"):
    result = {}
    if version.parse(transformers_version) < version.parse(
        "3.1.0"
    ):  # past_key_values name does not exist in 3.0.2 or older
        raise RuntimeError("This tool requires transformers 3.1.0 or later.")

    args = parse_arguments(argv)
    setup_logger(args.verbose)

    if not experiment_name:
        experiment_name = " ".join(argv if argv else sys.argv[1:])

    if args.tolerance == 0:
        args.tolerance = DEFAULT_TOLERANCE[args.precision]

    logger.info(f"Arguments:{args}")

    cache_dir = args.cache_dir
    output_dir = args.output if not args.output.endswith(".onnx") else os.path.dirname(args.output)
    prepare_environment(cache_dir, output_dir, args.use_gpu)

    if args.precision != Precision.FLOAT32:
        assert args.optimize_onnx, "fp16/int8 requires --optimize_onnx"

    if args.precision == Precision.FLOAT16:
        assert args.use_gpu, "fp16 requires --use_gpu"

    if args.precision == Precision.INT8:
        assert not args.use_gpu, "quantization only supports CPU"

    model_class = MODEL_CLASSES[args.model_class][0]
    use_padding = MODEL_CLASSES[args.model_class][2]

    gpt2helper = Gpt2Helper
    config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=cache_dir)
    model = model_class.from_pretrained(args.model_name_or_path, config=config, cache_dir=cache_dir)

    device = torch.device("cuda:0" if args.use_gpu else "cpu")
    model.eval().to(device)

    if (not args.use_external_data_format) and (config.n_layer > 24):
        logger.info("Try --use_external_data_format when model size > 2GB")

    onnx_model_paths = gpt2helper.get_onnx_paths(
        output_dir,
        args.model_name_or_path,
        args.model_class,
        new_folder=(args.precision == Precision.INT8),
        remove_existing=["fp32", "fp16", "int8"],
    )  # Do not remove raw model to save time in parity test

    raw_onnx_model = onnx_model_paths["raw"]

    int_data_type = torch.int64 if args.use_int64_inputs else torch.int32

    if os.path.exists(raw_onnx_model) and not args.overwrite:
        logger.warning(f"Skip exporting ONNX model since it existed: {raw_onnx_model}")
    else:
        logger.info(f"Exporting ONNX model to {raw_onnx_model}")
        gpt2helper.export_onnx(
            model,
            device,
            raw_onnx_model,
            args.verbose,
            args.use_external_data_format,
            has_position_ids=use_padding,
            has_attention_mask=use_padding,
            input_ids_dtype=int_data_type,
            position_ids_dtype=int_data_type,
            attention_mask_dtype=int_data_type,
        )

    fp16_params = {"keep_io_types": args.keep_io_types}
    if args.io_block_list:
        fp16_params["keep_io_types"] = args.io_block_list
    if args.node_block_list:
        fp16_params["node_block_list"] = args.node_block_list
    if args.op_block_list:
        fp16_params["op_block_list"] = args.op_block_list
    if args.force_fp16_initializers:
        fp16_params["force_fp16_initializers"] = args.force_fp16_initializers

    is_io_float16 = args.precision == Precision.FLOAT16 and not args.keep_io_types

    optimized_ops = ""
    all_ops = ""
    if args.optimize_onnx or args.precision != Precision.FLOAT32:
        output_path = onnx_model_paths[str(args.precision) if args.precision != Precision.INT8 else "fp32"]

        logger.info(f"Optimizing model to {output_path}")
        m = gpt2helper.optimize_onnx(
            raw_onnx_model,
            output_path,
            args.precision == Precision.FLOAT16,
            model.config.num_attention_heads,
            model.config.hidden_size,
            args.use_external_data_format,
            auto_mixed_precision=args.auto_mixed_precision,
            stage=args.stage,
            **fp16_params,
        )

        nodes = m.nodes()
        op_list = {node.op_type for node in nodes}
        all_ops = ",".join(op_list)

        # print optimized operators
        optimized_op_counter = m.get_fused_operator_statistics()
        if optimized_op_counter:
            optimized_ops = ",".join([key for key in optimized_op_counter if optimized_op_counter[key] > 0])
    else:
        output_path = raw_onnx_model

    if args.precision == Precision.INT8:
        logger.info("quantizing model...")
        QuantizeHelper.quantize_onnx_model(output_path, onnx_model_paths["int8"], args.use_external_data_format)
        model = QuantizeHelper.quantize_torch_model(model)
        logger.info("finished quantizing model")
        output_path = onnx_model_paths["int8"]

    if args.output.endswith(".onnx") and output_path != args.output and not args.use_external_data_format:
        shutil.move(output_path, args.output)
        output_path = args.output

    logger.info(f"Output path: {output_path}")
    model_size_in_MB = int(get_onnx_model_size(output_path, args.use_external_data_format) / 1024 / 1024)  # noqa: N806

    provider = args.provider
    session = create_onnxruntime_session(
        output_path, args.use_gpu, provider, enable_all_optimization=True, verbose=args.verbose
    )
    if args.model_class == "GPT2LMHeadModel" and session is not None:
        parity_result = gpt2helper.test_parity(
            session,
            model,
            device,
            is_io_float16,
            rtol=args.tolerance,
            atol=args.tolerance,
            model_class=args.model_class,
            has_position_ids=use_padding,
            has_attention_mask=use_padding,
            input_ids_dtype=int_data_type,
            position_ids_dtype=int_data_type,
            attention_mask_dtype=int_data_type,
            test_cases_per_run=args.test_cases,
            total_runs=args.test_runs,
            stage=args.stage,
            verbose=args.verbose,
        )

        # An example configuration for testing performance
        batch_size = 8
        sequence_length = 32 if args.stage == 1 else 1
        past_sequence_length = 0 if args.stage == 1 else 32

        latency = gpt2helper.test_performance(
            session,
            model,
            device,
            is_io_float16,
            total_runs=100,
            use_io_binding=True,
            model_class=args.model_class,
            has_position_ids=use_padding,
            has_attention_mask=use_padding,
            input_ids_dtype=int_data_type,
            position_ids_dtype=int_data_type,
            attention_mask_dtype=int_data_type,
            batch_size=batch_size,
            sequence_length=sequence_length,
            past_sequence_length=past_sequence_length,
        )

        if args.precision == Precision.FLOAT16:
            logger.info(f"fp16 conversion parameters:{fp16_params}")

        # Write results to file
        latency_name = get_latency_name(batch_size, sequence_length, past_sequence_length)
        csv_file_existed = os.path.exists(csv_filename)
        with open(csv_filename, mode="a", newline="") as csv_file:
            column_names = [
                "experiment",
                "run_id",
                "model_name",
                "model_class",
                "stage",
                "gpu",
                "precision",
                "optimizer",
                "test_cases",
                "runs",
                "keep_io_types",
                "io_block_list",
                "op_block_list",
                "node_block_list",
                "force_fp16_initializers",
                "auto_mixed_precision",
                "optimized_operators",
                "operators",
                "environment_variables",
                "onnxruntime",
                latency_name,
                "top1_match_rate",
                "onnx_size_in_MB",
                "diff_50_percentile",
                "diff_90_percentile",
                "diff_95_percentile",
                "diff_99_percentile",
                "diff_pass_rate",
                "nan_rate",
                "top1_match_rate_per_run",
            ]
            csv_writer = csv.DictWriter(csv_file, fieldnames=column_names)
            if not csv_file_existed:
                csv_writer.writeheader()
            row = {
                "experiment": experiment_name,
                "run_id": run_id,
                "model_name": args.model_name_or_path,
                "model_class": args.model_class,
                "stage": args.stage,
                "gpu": args.use_gpu,
                "precision": args.precision,
                "optimizer": args.optimize_onnx,
                "test_cases": args.test_cases,
                "runs": args.test_runs,
                "keep_io_types": args.keep_io_types,
                "io_block_list": args.io_block_list,
                "op_block_list": args.op_block_list,
                "node_block_list": args.node_block_list,
                "force_fp16_initializers": args.force_fp16_initializers,
                "auto_mixed_precision": args.auto_mixed_precision,
                "optimized_operators": optimized_ops,
                "operators": all_ops,
                "environment_variables": get_ort_environment_variables(),
                "onnxruntime": ort_version,
                latency_name: f"{latency:.2f}",
                "diff_50_percentile": parity_result["max_diff_percentile_50"],
                "diff_90_percentile": parity_result["max_diff_percentile_90"],
                "diff_95_percentile": parity_result["max_diff_percentile_95"],
                "diff_99_percentile": parity_result["max_diff_percentile_99"],
                "diff_pass_rate": parity_result["diff_pass_rate"],
                "nan_rate": parity_result["nan_rate"],
                "top1_match_rate": parity_result["top1_match_rate"],
                "top1_match_rate_per_run": parity_result["top1_match_rate_per_run"],
                "onnx_size_in_MB": f"{model_size_in_MB}",
            }
            logger.info(f"result: {row}")
            result.update(row)
            csv_writer.writerow(row)

    if args.input_test_file:
        test_inputs = []
        # Each line of test file is a JSON string like:
        # {"input_ids": [[14698, 257, 1310, 13688, 319, 326]]}
        with open(args.input_test_file) as read_f:
            for _, line in enumerate(read_f):
                line = line.rstrip()  # noqa: PLW2901
                data = json.loads(line)
                input_ids = torch.from_numpy(numpy.asarray(data["input_ids"], dtype=numpy.int64)).to(device)

                if use_padding:
                    if "attention_mask" in data:
                        numpy_float = numpy.float16 if is_io_float16 else numpy.float32
                        attention_mask = torch.from_numpy(numpy.asarray(data["attention_mask"], dtype=numpy_float)).to(
                            device
                        )
                    else:
                        padding = -1
                        attention_mask = (input_ids != padding).type(torch.float16 if is_io_float16 else torch.float32)
                        input_ids.masked_fill_(input_ids == padding, 0)

                    if "position_ids" in data:
                        position_ids = torch.from_numpy(numpy.asarray(data["position_ids"], dtype=numpy.int64)).to(
                            device
                        )
                    else:
                        position_ids = attention_mask.long().cumsum(-1) - 1
                        position_ids.masked_fill_(position_ids < 0, 0)

                    inputs = {
                        "input_ids": input_ids.to(int_data_type),
                        "position_ids": position_ids.to(int_data_type),
                        "attention_mask": attention_mask.to(int_data_type),
                    }
                else:
                    inputs = {"input_ids": input_ids.to(int_data_type)}

                test_inputs.append(inputs)

        Gpt2Tester.test_generation(
            session,
            model,
            device,
            test_inputs,
            precision=args.precision,
            model_class=args.model_class,
            top_k=20,
            top_k_no_order=True,
            max_steps=24,
            max_inputs=0,
            verbose=args.verbose,
            save_test_data=3,
            save_test_data_dir=Path(output_path).parent,
        )

    logger.info(f"Done. Output model: {output_path}")
    return result


if __name__ == "__main__":
    main()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v136\preload.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Preload (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page


class RuleSetId(str):
    '''
    Unique id
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> RuleSetId:
        return cls(json)

    def __repr__(self):
        return 'RuleSetId({})'.format(super().__repr__())


@dataclass
class RuleSet:
    '''
    Corresponds to SpeculationRuleSet
    '''
    id_: RuleSetId

    #: Identifies a document which the rule set is associated with.
    loader_id: network.LoaderId

    #: Source text of JSON representing the rule set. If it comes from
    #: ``script`` tag, it is the textContent of the node. Note that it is
    #: a JSON for valid case.
    #: 
    #: See also:
    #: - https://wicg.github.io/nav-speculation/speculation-rules.html
    #: - https://github.com/WICG/nav-speculation/blob/main/triggers.md
    source_text: str

    #: A speculation rule set is either added through an inline
    #: ``script`` tag or through an external resource via the
    #: 'Speculation-Rules' HTTP header. For the first case, we include
    #: the BackendNodeId of the relevant ``script`` tag. For the second
    #: case, we include the external URL where the rule set was loaded
    #: from, and also RequestId if Network domain is enabled.
    #: 
    #: See also:
    #: - https://wicg.github.io/nav-speculation/speculation-rules.html#speculation-rules-script
    #: - https://wicg.github.io/nav-speculation/speculation-rules.html#speculation-rules-header
    backend_node_id: typing.Optional[dom.BackendNodeId] = None

    url: typing.Optional[str] = None

    request_id: typing.Optional[network.RequestId] = None

    #: Error information
    #: ``errorMessage`` is null iff ``errorType`` is null.
    error_type: typing.Optional[RuleSetErrorType] = None

    #: TODO(https://crbug.com/1425354): Replace this property with structured error.
    error_message: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['id'] = self.id_.to_json()
        json['loaderId'] = self.loader_id.to_json()
        json['sourceText'] = self.source_text
        if self.backend_node_id is not None:
            json['backendNodeId'] = self.backend_node_id.to_json()
        if self.url is not None:
            json['url'] = self.url
        if self.request_id is not None:
            json['requestId'] = self.request_id.to_json()
        if self.error_type is not None:
            json['errorType'] = self.error_type.to_json()
        if self.error_message is not None:
            json['errorMessage'] = self.error_message
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            id_=RuleSetId.from_json(json['id']),
            loader_id=network.LoaderId.from_json(json['loaderId']),
            source_text=str(json['sourceText']),
            backend_node_id=dom.BackendNodeId.from_json(json['backendNodeId']) if 'backendNodeId' in json else None,
            url=str(json['url']) if 'url' in json else None,
            request_id=network.RequestId.from_json(json['requestId']) if 'requestId' in json else None,
            error_type=RuleSetErrorType.from_json(json['errorType']) if 'errorType' in json else None,
            error_message=str(json['errorMessage']) if 'errorMessage' in json else None,
        )


class RuleSetErrorType(enum.Enum):
    SOURCE_IS_NOT_JSON_OBJECT = "SourceIsNotJsonObject"
    INVALID_RULES_SKIPPED = "InvalidRulesSkipped"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class SpeculationAction(enum.Enum):
    '''
    The type of preloading attempted. It corresponds to
    mojom::SpeculationAction (although PrefetchWithSubresources is omitted as it
    isn't being used by clients).
    '''
    PREFETCH = "Prefetch"
    PRERENDER = "Prerender"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class SpeculationTargetHint(enum.Enum):
    '''
    Corresponds to mojom::SpeculationTargetHint.
    See https://github.com/WICG/nav-speculation/blob/main/triggers.md#window-name-targeting-hints
    '''
    BLANK = "Blank"
    SELF = "Self"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class PreloadingAttemptKey:
    '''
    A key that identifies a preloading attempt.

    The url used is the url specified by the trigger (i.e. the initial URL), and
    not the final url that is navigated to. For example, prerendering allows
    same-origin main frame navigations during the attempt, but the attempt is
    still keyed with the initial URL.
    '''
    loader_id: network.LoaderId

    action: SpeculationAction

    url: str

    target_hint: typing.Optional[SpeculationTargetHint] = None

    def to_json(self):
        json = dict()
        json['loaderId'] = self.loader_id.to_json()
        json['action'] = self.action.to_json()
        json['url'] = self.url
        if self.target_hint is not None:
            json['targetHint'] = self.target_hint.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            loader_id=network.LoaderId.from_json(json['loaderId']),
            action=SpeculationAction.from_json(json['action']),
            url=str(json['url']),
            target_hint=SpeculationTargetHint.from_json(json['targetHint']) if 'targetHint' in json else None,
        )


@dataclass
class PreloadingAttemptSource:
    '''
    Lists sources for a preloading attempt, specifically the ids of rule sets
    that had a speculation rule that triggered the attempt, and the
    BackendNodeIds of <a href> or <area href> elements that triggered the
    attempt (in the case of attempts triggered by a document rule). It is
    possible for multiple rule sets and links to trigger a single attempt.
    '''
    key: PreloadingAttemptKey

    rule_set_ids: typing.List[RuleSetId]

    node_ids: typing.List[dom.BackendNodeId]

    def to_json(self):
        json = dict()
        json['key'] = self.key.to_json()
        json['ruleSetIds'] = [i.to_json() for i in self.rule_set_ids]
        json['nodeIds'] = [i.to_json() for i in self.node_ids]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            key=PreloadingAttemptKey.from_json(json['key']),
            rule_set_ids=[RuleSetId.from_json(i) for i in json['ruleSetIds']],
            node_ids=[dom.BackendNodeId.from_json(i) for i in json['nodeIds']],
        )


class PreloadPipelineId(str):
    '''
    Chrome manages different types of preloads together using a
    concept of preloading pipeline. For example, if a site uses a
    SpeculationRules for prerender, Chrome first starts a prefetch and
    then upgrades it to prerender.

    CDP events for them are emitted separately but they share
    ``PreloadPipelineId``.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> PreloadPipelineId:
        return cls(json)

    def __repr__(self):
        return 'PreloadPipelineId({})'.format(super().__repr__())


class PrerenderFinalStatus(enum.Enum):
    '''
    List of FinalStatus reasons for Prerender2.
    '''
    ACTIVATED = "Activated"
    DESTROYED = "Destroyed"
    LOW_END_DEVICE = "LowEndDevice"
    INVALID_SCHEME_REDIRECT = "InvalidSchemeRedirect"
    INVALID_SCHEME_NAVIGATION = "InvalidSchemeNavigation"
    NAVIGATION_REQUEST_BLOCKED_BY_CSP = "NavigationRequestBlockedByCsp"
    MAIN_FRAME_NAVIGATION = "MainFrameNavigation"
    MOJO_BINDER_POLICY = "MojoBinderPolicy"
    RENDERER_PROCESS_CRASHED = "RendererProcessCrashed"
    RENDERER_PROCESS_KILLED = "RendererProcessKilled"
    DOWNLOAD = "Download"
    TRIGGER_DESTROYED = "TriggerDestroyed"
    NAVIGATION_NOT_COMMITTED = "NavigationNotCommitted"
    NAVIGATION_BAD_HTTP_STATUS = "NavigationBadHttpStatus"
    CLIENT_CERT_REQUESTED = "ClientCertRequested"
    NAVIGATION_REQUEST_NETWORK_ERROR = "NavigationRequestNetworkError"
    CANCEL_ALL_HOSTS_FOR_TESTING = "CancelAllHostsForTesting"
    DID_FAIL_LOAD = "DidFailLoad"
    STOP = "Stop"
    SSL_CERTIFICATE_ERROR = "SslCertificateError"
    LOGIN_AUTH_REQUESTED = "LoginAuthRequested"
    UA_CHANGE_REQUIRES_RELOAD = "UaChangeRequiresReload"
    BLOCKED_BY_CLIENT = "BlockedByClient"
    AUDIO_OUTPUT_DEVICE_REQUESTED = "AudioOutputDeviceRequested"
    MIXED_CONTENT = "MixedContent"
    TRIGGER_BACKGROUNDED = "TriggerBackgrounded"
    MEMORY_LIMIT_EXCEEDED = "MemoryLimitExceeded"
    DATA_SAVER_ENABLED = "DataSaverEnabled"
    TRIGGER_URL_HAS_EFFECTIVE_URL = "TriggerUrlHasEffectiveUrl"
    ACTIVATED_BEFORE_STARTED = "ActivatedBeforeStarted"
    INACTIVE_PAGE_RESTRICTION = "InactivePageRestriction"
    START_FAILED = "StartFailed"
    TIMEOUT_BACKGROUNDED = "TimeoutBackgrounded"
    CROSS_SITE_REDIRECT_IN_INITIAL_NAVIGATION = "CrossSiteRedirectInInitialNavigation"
    CROSS_SITE_NAVIGATION_IN_INITIAL_NAVIGATION = "CrossSiteNavigationInInitialNavigation"
    SAME_SITE_CROSS_ORIGIN_REDIRECT_NOT_OPT_IN_IN_INITIAL_NAVIGATION = "SameSiteCrossOriginRedirectNotOptInInInitialNavigation"
    SAME_SITE_CROSS_ORIGIN_NAVIGATION_NOT_OPT_IN_IN_INITIAL_NAVIGATION = "SameSiteCrossOriginNavigationNotOptInInInitialNavigation"
    ACTIVATION_NAVIGATION_PARAMETER_MISMATCH = "ActivationNavigationParameterMismatch"
    ACTIVATED_IN_BACKGROUND = "ActivatedInBackground"
    EMBEDDER_HOST_DISALLOWED = "EmbedderHostDisallowed"
    ACTIVATION_NAVIGATION_DESTROYED_BEFORE_SUCCESS = "ActivationNavigationDestroyedBeforeSuccess"
    TAB_CLOSED_BY_USER_GESTURE = "TabClosedByUserGesture"
    TAB_CLOSED_WITHOUT_USER_GESTURE = "TabClosedWithoutUserGesture"
    PRIMARY_MAIN_FRAME_RENDERER_PROCESS_CRASHED = "PrimaryMainFrameRendererProcessCrashed"
    PRIMARY_MAIN_FRAME_RENDERER_PROCESS_KILLED = "PrimaryMainFrameRendererProcessKilled"
    ACTIVATION_FRAME_POLICY_NOT_COMPATIBLE = "ActivationFramePolicyNotCompatible"
    PRELOADING_DISABLED = "PreloadingDisabled"
    BATTERY_SAVER_ENABLED = "BatterySaverEnabled"
    ACTIVATED_DURING_MAIN_FRAME_NAVIGATION = "ActivatedDuringMainFrameNavigation"
    PRELOADING_UNSUPPORTED_BY_WEB_CONTENTS = "PreloadingUnsupportedByWebContents"
    CROSS_SITE_REDIRECT_IN_MAIN_FRAME_NAVIGATION = "CrossSiteRedirectInMainFrameNavigation"
    CROSS_SITE_NAVIGATION_IN_MAIN_FRAME_NAVIGATION = "CrossSiteNavigationInMainFrameNavigation"
    SAME_SITE_CROSS_ORIGIN_REDIRECT_NOT_OPT_IN_IN_MAIN_FRAME_NAVIGATION = "SameSiteCrossOriginRedirectNotOptInInMainFrameNavigation"
    SAME_SITE_CROSS_ORIGIN_NAVIGATION_NOT_OPT_IN_IN_MAIN_FRAME_NAVIGATION = "SameSiteCrossOriginNavigationNotOptInInMainFrameNavigation"
    MEMORY_PRESSURE_ON_TRIGGER = "MemoryPressureOnTrigger"
    MEMORY_PRESSURE_AFTER_TRIGGERED = "MemoryPressureAfterTriggered"
    PRERENDERING_DISABLED_BY_DEV_TOOLS = "PrerenderingDisabledByDevTools"
    SPECULATION_RULE_REMOVED = "SpeculationRuleRemoved"
    ACTIVATED_WITH_AUXILIARY_BROWSING_CONTEXTS = "ActivatedWithAuxiliaryBrowsingContexts"
    MAX_NUM_OF_RUNNING_EAGER_PRERENDERS_EXCEEDED = "MaxNumOfRunningEagerPrerendersExceeded"
    MAX_NUM_OF_RUNNING_NON_EAGER_PRERENDERS_EXCEEDED = "MaxNumOfRunningNonEagerPrerendersExceeded"
    MAX_NUM_OF_RUNNING_EMBEDDER_PRERENDERS_EXCEEDED = "MaxNumOfRunningEmbedderPrerendersExceeded"
    PRERENDERING_URL_HAS_EFFECTIVE_URL = "PrerenderingUrlHasEffectiveUrl"
    REDIRECTED_PRERENDERING_URL_HAS_EFFECTIVE_URL = "RedirectedPrerenderingUrlHasEffectiveUrl"
    ACTIVATION_URL_HAS_EFFECTIVE_URL = "ActivationUrlHasEffectiveUrl"
    JAVA_SCRIPT_INTERFACE_ADDED = "JavaScriptInterfaceAdded"
    JAVA_SCRIPT_INTERFACE_REMOVED = "JavaScriptInterfaceRemoved"
    ALL_PRERENDERING_CANCELED = "AllPrerenderingCanceled"
    WINDOW_CLOSED = "WindowClosed"
    SLOW_NETWORK = "SlowNetwork"
    OTHER_PRERENDERED_PAGE_ACTIVATED = "OtherPrerenderedPageActivated"
    V8_OPTIMIZER_DISABLED = "V8OptimizerDisabled"
    PRERENDER_FAILED_DURING_PREFETCH = "PrerenderFailedDuringPrefetch"
    BROWSING_DATA_REMOVED = "BrowsingDataRemoved"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class PreloadingStatus(enum.Enum):
    '''
    Preloading status values, see also PreloadingTriggeringOutcome. This
    status is shared by prefetchStatusUpdated and prerenderStatusUpdated.
    '''
    PENDING = "Pending"
    RUNNING = "Running"
    READY = "Ready"
    SUCCESS = "Success"
    FAILURE = "Failure"
    NOT_SUPPORTED = "NotSupported"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class PrefetchStatus(enum.Enum):
    '''
    TODO(https://crbug.com/1384419): revisit the list of PrefetchStatus and
    filter out the ones that aren't necessary to the developers.
    '''
    PREFETCH_ALLOWED = "PrefetchAllowed"
    PREFETCH_FAILED_INELIGIBLE_REDIRECT = "PrefetchFailedIneligibleRedirect"
    PREFETCH_FAILED_INVALID_REDIRECT = "PrefetchFailedInvalidRedirect"
    PREFETCH_FAILED_MIME_NOT_SUPPORTED = "PrefetchFailedMIMENotSupported"
    PREFETCH_FAILED_NET_ERROR = "PrefetchFailedNetError"
    PREFETCH_FAILED_NON2_XX = "PrefetchFailedNon2XX"
    PREFETCH_EVICTED_AFTER_BROWSING_DATA_REMOVED = "PrefetchEvictedAfterBrowsingDataRemoved"
    PREFETCH_EVICTED_AFTER_CANDIDATE_REMOVED = "PrefetchEvictedAfterCandidateRemoved"
    PREFETCH_EVICTED_FOR_NEWER_PREFETCH = "PrefetchEvictedForNewerPrefetch"
    PREFETCH_HELDBACK = "PrefetchHeldback"
    PREFETCH_INELIGIBLE_RETRY_AFTER = "PrefetchIneligibleRetryAfter"
    PREFETCH_IS_PRIVACY_DECOY = "PrefetchIsPrivacyDecoy"
    PREFETCH_IS_STALE = "PrefetchIsStale"
    PREFETCH_NOT_ELIGIBLE_BROWSER_CONTEXT_OFF_THE_RECORD = "PrefetchNotEligibleBrowserContextOffTheRecord"
    PREFETCH_NOT_ELIGIBLE_DATA_SAVER_ENABLED = "PrefetchNotEligibleDataSaverEnabled"
    PREFETCH_NOT_ELIGIBLE_EXISTING_PROXY = "PrefetchNotEligibleExistingProxy"
    PREFETCH_NOT_ELIGIBLE_HOST_IS_NON_UNIQUE = "PrefetchNotEligibleHostIsNonUnique"
    PREFETCH_NOT_ELIGIBLE_NON_DEFAULT_STORAGE_PARTITION = "PrefetchNotEligibleNonDefaultStoragePartition"
    PREFETCH_NOT_ELIGIBLE_SAME_SITE_CROSS_ORIGIN_PREFETCH_REQUIRED_PROXY = "PrefetchNotEligibleSameSiteCrossOriginPrefetchRequiredProxy"
    PREFETCH_NOT_ELIGIBLE_SCHEME_IS_NOT_HTTPS = "PrefetchNotEligibleSchemeIsNotHttps"
    PREFETCH_NOT_ELIGIBLE_USER_HAS_COOKIES = "PrefetchNotEligibleUserHasCookies"
    PREFETCH_NOT_ELIGIBLE_USER_HAS_SERVICE_WORKER = "PrefetchNotEligibleUserHasServiceWorker"
    PREFETCH_NOT_ELIGIBLE_USER_HAS_SERVICE_WORKER_NO_FETCH_HANDLER = "PrefetchNotEligibleUserHasServiceWorkerNoFetchHandler"
    PREFETCH_NOT_ELIGIBLE_REDIRECT_FROM_SERVICE_WORKER = "PrefetchNotEligibleRedirectFromServiceWorker"
    PREFETCH_NOT_ELIGIBLE_REDIRECT_TO_SERVICE_WORKER = "PrefetchNotEligibleRedirectToServiceWorker"
    PREFETCH_NOT_ELIGIBLE_BATTERY_SAVER_ENABLED = "PrefetchNotEligibleBatterySaverEnabled"
    PREFETCH_NOT_ELIGIBLE_PRELOADING_DISABLED = "PrefetchNotEligiblePreloadingDisabled"
    PREFETCH_NOT_FINISHED_IN_TIME = "PrefetchNotFinishedInTime"
    PREFETCH_NOT_STARTED = "PrefetchNotStarted"
    PREFETCH_NOT_USED_COOKIES_CHANGED = "PrefetchNotUsedCookiesChanged"
    PREFETCH_PROXY_NOT_AVAILABLE = "PrefetchProxyNotAvailable"
    PREFETCH_RESPONSE_USED = "PrefetchResponseUsed"
    PREFETCH_SUCCESSFUL_BUT_NOT_USED = "PrefetchSuccessfulButNotUsed"
    PREFETCH_NOT_USED_PROBE_FAILED = "PrefetchNotUsedProbeFailed"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class PrerenderMismatchedHeaders:
    '''
    Information of headers to be displayed when the header mismatch occurred.
    '''
    header_name: str

    initial_value: typing.Optional[str] = None

    activation_value: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['headerName'] = self.header_name
        if self.initial_value is not None:
            json['initialValue'] = self.initial_value
        if self.activation_value is not None:
            json['activationValue'] = self.activation_value
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            header_name=str(json['headerName']),
            initial_value=str(json['initialValue']) if 'initialValue' in json else None,
            activation_value=str(json['activationValue']) if 'activationValue' in json else None,
        )


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:

    cmd_dict: T_JSON_DICT = {
        'method': 'Preload.enable',
    }
    json = yield cmd_dict


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:

    cmd_dict: T_JSON_DICT = {
        'method': 'Preload.disable',
    }
    json = yield cmd_dict


@event_class('Preload.ruleSetUpdated')
@dataclass
class RuleSetUpdated:
    '''
    Upsert. Currently, it is only emitted when a rule set added.
    '''
    rule_set: RuleSet

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> RuleSetUpdated:
        return cls(
            rule_set=RuleSet.from_json(json['ruleSet'])
        )


@event_class('Preload.ruleSetRemoved')
@dataclass
class RuleSetRemoved:
    id_: RuleSetId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> RuleSetRemoved:
        return cls(
            id_=RuleSetId.from_json(json['id'])
        )


@event_class('Preload.preloadEnabledStateUpdated')
@dataclass
class PreloadEnabledStateUpdated:
    '''
    Fired when a preload enabled state is updated.
    '''
    disabled_by_preference: bool
    disabled_by_data_saver: bool
    disabled_by_battery_saver: bool
    disabled_by_holdback_prefetch_speculation_rules: bool
    disabled_by_holdback_prerender_speculation_rules: bool

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> PreloadEnabledStateUpdated:
        return cls(
            disabled_by_preference=bool(json['disabledByPreference']),
            disabled_by_data_saver=bool(json['disabledByDataSaver']),
            disabled_by_battery_saver=bool(json['disabledByBatterySaver']),
            disabled_by_holdback_prefetch_speculation_rules=bool(json['disabledByHoldbackPrefetchSpeculationRules']),
            disabled_by_holdback_prerender_speculation_rules=bool(json['disabledByHoldbackPrerenderSpeculationRules'])
        )


@event_class('Preload.prefetchStatusUpdated')
@dataclass
class PrefetchStatusUpdated:
    '''
    Fired when a prefetch attempt is updated.
    '''
    key: PreloadingAttemptKey
    pipeline_id: PreloadPipelineId
    #: The frame id of the frame initiating prefetch.
    initiating_frame_id: page.FrameId
    prefetch_url: str
    status: PreloadingStatus
    prefetch_status: PrefetchStatus
    request_id: network.RequestId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> PrefetchStatusUpdated:
        return cls(
            key=PreloadingAttemptKey.from_json(json['key']),
            pipeline_id=PreloadPipelineId.from_json(json['pipelineId']),
            initiating_frame_id=page.FrameId.from_json(json['initiatingFrameId']),
            prefetch_url=str(json['prefetchUrl']),
            status=PreloadingStatus.from_json(json['status']),
            prefetch_status=PrefetchStatus.from_json(json['prefetchStatus']),
            request_id=network.RequestId.from_json(json['requestId'])
        )


@event_class('Preload.prerenderStatusUpdated')
@dataclass
class PrerenderStatusUpdated:
    '''
    Fired when a prerender attempt is updated.
    '''
    key: PreloadingAttemptKey
    pipeline_id: PreloadPipelineId
    status: PreloadingStatus
    prerender_status: typing.Optional[PrerenderFinalStatus]
    #: This is used to give users more information about the name of Mojo interface
    #: that is incompatible with prerender and has caused the cancellation of the attempt.
    disallowed_mojo_interface: typing.Optional[str]
    mismatched_headers: typing.Optional[typing.List[PrerenderMismatchedHeaders]]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> PrerenderStatusUpdated:
        return cls(
            key=PreloadingAttemptKey.from_json(json['key']),
            pipeline_id=PreloadPipelineId.from_json(json['pipelineId']),
            status=PreloadingStatus.from_json(json['status']),
            prerender_status=PrerenderFinalStatus.from_json(json['prerenderStatus']) if 'prerenderStatus' in json else None,
            disallowed_mojo_interface=str(json['disallowedMojoInterface']) if 'disallowedMojoInterface' in json else None,
            mismatched_headers=[PrerenderMismatchedHeaders.from_json(i) for i in json['mismatchedHeaders']] if 'mismatchedHeaders' in json else None
        )


@event_class('Preload.preloadingAttemptSourcesUpdated')
@dataclass
class PreloadingAttemptSourcesUpdated:
    '''
    Send a list of sources for all preloading attempts in a document.
    '''
    loader_id: network.LoaderId
    preloading_attempt_sources: typing.List[PreloadingAttemptSource]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> PreloadingAttemptSourcesUpdated:
        return cls(
            loader_id=network.LoaderId.from_json(json['loaderId']),
            preloading_attempt_sources=[PreloadingAttemptSource.from_json(i) for i in json['preloadingAttemptSources']]
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v137\preload.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Preload (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page


class RuleSetId(str):
    '''
    Unique id
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> RuleSetId:
        return cls(json)

    def __repr__(self):
        return 'RuleSetId({})'.format(super().__repr__())


@dataclass
class RuleSet:
    '''
    Corresponds to SpeculationRuleSet
    '''
    id_: RuleSetId

    #: Identifies a document which the rule set is associated with.
    loader_id: network.LoaderId

    #: Source text of JSON representing the rule set. If it comes from
    #: ``script`` tag, it is the textContent of the node. Note that it is
    #: a JSON for valid case.
    #: 
    #: See also:
    #: - https://wicg.github.io/nav-speculation/speculation-rules.html
    #: - https://github.com/WICG/nav-speculation/blob/main/triggers.md
    source_text: str

    #: A speculation rule set is either added through an inline
    #: ``script`` tag or through an external resource via the
    #: 'Speculation-Rules' HTTP header. For the first case, we include
    #: the BackendNodeId of the relevant ``script`` tag. For the second
    #: case, we include the external URL where the rule set was loaded
    #: from, and also RequestId if Network domain is enabled.
    #: 
    #: See also:
    #: - https://wicg.github.io/nav-speculation/speculation-rules.html#speculation-rules-script
    #: - https://wicg.github.io/nav-speculation/speculation-rules.html#speculation-rules-header
    backend_node_id: typing.Optional[dom.BackendNodeId] = None

    url: typing.Optional[str] = None

    request_id: typing.Optional[network.RequestId] = None

    #: Error information
    #: ``errorMessage`` is null iff ``errorType`` is null.
    error_type: typing.Optional[RuleSetErrorType] = None

    #: TODO(https://crbug.com/1425354): Replace this property with structured error.
    error_message: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['id'] = self.id_.to_json()
        json['loaderId'] = self.loader_id.to_json()
        json['sourceText'] = self.source_text
        if self.backend_node_id is not None:
            json['backendNodeId'] = self.backend_node_id.to_json()
        if self.url is not None:
            json['url'] = self.url
        if self.request_id is not None:
            json['requestId'] = self.request_id.to_json()
        if self.error_type is not None:
            json['errorType'] = self.error_type.to_json()
        if self.error_message is not None:
            json['errorMessage'] = self.error_message
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            id_=RuleSetId.from_json(json['id']),
            loader_id=network.LoaderId.from_json(json['loaderId']),
            source_text=str(json['sourceText']),
            backend_node_id=dom.BackendNodeId.from_json(json['backendNodeId']) if 'backendNodeId' in json else None,
            url=str(json['url']) if 'url' in json else None,
            request_id=network.RequestId.from_json(json['requestId']) if 'requestId' in json else None,
            error_type=RuleSetErrorType.from_json(json['errorType']) if 'errorType' in json else None,
            error_message=str(json['errorMessage']) if 'errorMessage' in json else None,
        )


class RuleSetErrorType(enum.Enum):
    SOURCE_IS_NOT_JSON_OBJECT = "SourceIsNotJsonObject"
    INVALID_RULES_SKIPPED = "InvalidRulesSkipped"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class SpeculationAction(enum.Enum):
    '''
    The type of preloading attempted. It corresponds to
    mojom::SpeculationAction (although PrefetchWithSubresources is omitted as it
    isn't being used by clients).
    '''
    PREFETCH = "Prefetch"
    PRERENDER = "Prerender"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class SpeculationTargetHint(enum.Enum):
    '''
    Corresponds to mojom::SpeculationTargetHint.
    See https://github.com/WICG/nav-speculation/blob/main/triggers.md#window-name-targeting-hints
    '''
    BLANK = "Blank"
    SELF = "Self"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class PreloadingAttemptKey:
    '''
    A key that identifies a preloading attempt.

    The url used is the url specified by the trigger (i.e. the initial URL), and
    not the final url that is navigated to. For example, prerendering allows
    same-origin main frame navigations during the attempt, but the attempt is
    still keyed with the initial URL.
    '''
    loader_id: network.LoaderId

    action: SpeculationAction

    url: str

    target_hint: typing.Optional[SpeculationTargetHint] = None

    def to_json(self):
        json = dict()
        json['loaderId'] = self.loader_id.to_json()
        json['action'] = self.action.to_json()
        json['url'] = self.url
        if self.target_hint is not None:
            json['targetHint'] = self.target_hint.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            loader_id=network.LoaderId.from_json(json['loaderId']),
            action=SpeculationAction.from_json(json['action']),
            url=str(json['url']),
            target_hint=SpeculationTargetHint.from_json(json['targetHint']) if 'targetHint' in json else None,
        )


@dataclass
class PreloadingAttemptSource:
    '''
    Lists sources for a preloading attempt, specifically the ids of rule sets
    that had a speculation rule that triggered the attempt, and the
    BackendNodeIds of <a href> or <area href> elements that triggered the
    attempt (in the case of attempts triggered by a document rule). It is
    possible for multiple rule sets and links to trigger a single attempt.
    '''
    key: PreloadingAttemptKey

    rule_set_ids: typing.List[RuleSetId]

    node_ids: typing.List[dom.BackendNodeId]

    def to_json(self):
        json = dict()
        json['key'] = self.key.to_json()
        json['ruleSetIds'] = [i.to_json() for i in self.rule_set_ids]
        json['nodeIds'] = [i.to_json() for i in self.node_ids]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            key=PreloadingAttemptKey.from_json(json['key']),
            rule_set_ids=[RuleSetId.from_json(i) for i in json['ruleSetIds']],
            node_ids=[dom.BackendNodeId.from_json(i) for i in json['nodeIds']],
        )


class PreloadPipelineId(str):
    '''
    Chrome manages different types of preloads together using a
    concept of preloading pipeline. For example, if a site uses a
    SpeculationRules for prerender, Chrome first starts a prefetch and
    then upgrades it to prerender.

    CDP events for them are emitted separately but they share
    ``PreloadPipelineId``.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> PreloadPipelineId:
        return cls(json)

    def __repr__(self):
        return 'PreloadPipelineId({})'.format(super().__repr__())


class PrerenderFinalStatus(enum.Enum):
    '''
    List of FinalStatus reasons for Prerender2.
    '''
    ACTIVATED = "Activated"
    DESTROYED = "Destroyed"
    LOW_END_DEVICE = "LowEndDevice"
    INVALID_SCHEME_REDIRECT = "InvalidSchemeRedirect"
    INVALID_SCHEME_NAVIGATION = "InvalidSchemeNavigation"
    NAVIGATION_REQUEST_BLOCKED_BY_CSP = "NavigationRequestBlockedByCsp"
    MAIN_FRAME_NAVIGATION = "MainFrameNavigation"
    MOJO_BINDER_POLICY = "MojoBinderPolicy"
    RENDERER_PROCESS_CRASHED = "RendererProcessCrashed"
    RENDERER_PROCESS_KILLED = "RendererProcessKilled"
    DOWNLOAD = "Download"
    TRIGGER_DESTROYED = "TriggerDestroyed"
    NAVIGATION_NOT_COMMITTED = "NavigationNotCommitted"
    NAVIGATION_BAD_HTTP_STATUS = "NavigationBadHttpStatus"
    CLIENT_CERT_REQUESTED = "ClientCertRequested"
    NAVIGATION_REQUEST_NETWORK_ERROR = "NavigationRequestNetworkError"
    CANCEL_ALL_HOSTS_FOR_TESTING = "CancelAllHostsForTesting"
    DID_FAIL_LOAD = "DidFailLoad"
    STOP = "Stop"
    SSL_CERTIFICATE_ERROR = "SslCertificateError"
    LOGIN_AUTH_REQUESTED = "LoginAuthRequested"
    UA_CHANGE_REQUIRES_RELOAD = "UaChangeRequiresReload"
    BLOCKED_BY_CLIENT = "BlockedByClient"
    AUDIO_OUTPUT_DEVICE_REQUESTED = "AudioOutputDeviceRequested"
    MIXED_CONTENT = "MixedContent"
    TRIGGER_BACKGROUNDED = "TriggerBackgrounded"
    MEMORY_LIMIT_EXCEEDED = "MemoryLimitExceeded"
    DATA_SAVER_ENABLED = "DataSaverEnabled"
    TRIGGER_URL_HAS_EFFECTIVE_URL = "TriggerUrlHasEffectiveUrl"
    ACTIVATED_BEFORE_STARTED = "ActivatedBeforeStarted"
    INACTIVE_PAGE_RESTRICTION = "InactivePageRestriction"
    START_FAILED = "StartFailed"
    TIMEOUT_BACKGROUNDED = "TimeoutBackgrounded"
    CROSS_SITE_REDIRECT_IN_INITIAL_NAVIGATION = "CrossSiteRedirectInInitialNavigation"
    CROSS_SITE_NAVIGATION_IN_INITIAL_NAVIGATION = "CrossSiteNavigationInInitialNavigation"
    SAME_SITE_CROSS_ORIGIN_REDIRECT_NOT_OPT_IN_IN_INITIAL_NAVIGATION = "SameSiteCrossOriginRedirectNotOptInInInitialNavigation"
    SAME_SITE_CROSS_ORIGIN_NAVIGATION_NOT_OPT_IN_IN_INITIAL_NAVIGATION = "SameSiteCrossOriginNavigationNotOptInInInitialNavigation"
    ACTIVATION_NAVIGATION_PARAMETER_MISMATCH = "ActivationNavigationParameterMismatch"
    ACTIVATED_IN_BACKGROUND = "ActivatedInBackground"
    EMBEDDER_HOST_DISALLOWED = "EmbedderHostDisallowed"
    ACTIVATION_NAVIGATION_DESTROYED_BEFORE_SUCCESS = "ActivationNavigationDestroyedBeforeSuccess"
    TAB_CLOSED_BY_USER_GESTURE = "TabClosedByUserGesture"
    TAB_CLOSED_WITHOUT_USER_GESTURE = "TabClosedWithoutUserGesture"
    PRIMARY_MAIN_FRAME_RENDERER_PROCESS_CRASHED = "PrimaryMainFrameRendererProcessCrashed"
    PRIMARY_MAIN_FRAME_RENDERER_PROCESS_KILLED = "PrimaryMainFrameRendererProcessKilled"
    ACTIVATION_FRAME_POLICY_NOT_COMPATIBLE = "ActivationFramePolicyNotCompatible"
    PRELOADING_DISABLED = "PreloadingDisabled"
    BATTERY_SAVER_ENABLED = "BatterySaverEnabled"
    ACTIVATED_DURING_MAIN_FRAME_NAVIGATION = "ActivatedDuringMainFrameNavigation"
    PRELOADING_UNSUPPORTED_BY_WEB_CONTENTS = "PreloadingUnsupportedByWebContents"
    CROSS_SITE_REDIRECT_IN_MAIN_FRAME_NAVIGATION = "CrossSiteRedirectInMainFrameNavigation"
    CROSS_SITE_NAVIGATION_IN_MAIN_FRAME_NAVIGATION = "CrossSiteNavigationInMainFrameNavigation"
    SAME_SITE_CROSS_ORIGIN_REDIRECT_NOT_OPT_IN_IN_MAIN_FRAME_NAVIGATION = "SameSiteCrossOriginRedirectNotOptInInMainFrameNavigation"
    SAME_SITE_CROSS_ORIGIN_NAVIGATION_NOT_OPT_IN_IN_MAIN_FRAME_NAVIGATION = "SameSiteCrossOriginNavigationNotOptInInMainFrameNavigation"
    MEMORY_PRESSURE_ON_TRIGGER = "MemoryPressureOnTrigger"
    MEMORY_PRESSURE_AFTER_TRIGGERED = "MemoryPressureAfterTriggered"
    PRERENDERING_DISABLED_BY_DEV_TOOLS = "PrerenderingDisabledByDevTools"
    SPECULATION_RULE_REMOVED = "SpeculationRuleRemoved"
    ACTIVATED_WITH_AUXILIARY_BROWSING_CONTEXTS = "ActivatedWithAuxiliaryBrowsingContexts"
    MAX_NUM_OF_RUNNING_EAGER_PRERENDERS_EXCEEDED = "MaxNumOfRunningEagerPrerendersExceeded"
    MAX_NUM_OF_RUNNING_NON_EAGER_PRERENDERS_EXCEEDED = "MaxNumOfRunningNonEagerPrerendersExceeded"
    MAX_NUM_OF_RUNNING_EMBEDDER_PRERENDERS_EXCEEDED = "MaxNumOfRunningEmbedderPrerendersExceeded"
    PRERENDERING_URL_HAS_EFFECTIVE_URL = "PrerenderingUrlHasEffectiveUrl"
    REDIRECTED_PRERENDERING_URL_HAS_EFFECTIVE_URL = "RedirectedPrerenderingUrlHasEffectiveUrl"
    ACTIVATION_URL_HAS_EFFECTIVE_URL = "ActivationUrlHasEffectiveUrl"
    JAVA_SCRIPT_INTERFACE_ADDED = "JavaScriptInterfaceAdded"
    JAVA_SCRIPT_INTERFACE_REMOVED = "JavaScriptInterfaceRemoved"
    ALL_PRERENDERING_CANCELED = "AllPrerenderingCanceled"
    WINDOW_CLOSED = "WindowClosed"
    SLOW_NETWORK = "SlowNetwork"
    OTHER_PRERENDERED_PAGE_ACTIVATED = "OtherPrerenderedPageActivated"
    V8_OPTIMIZER_DISABLED = "V8OptimizerDisabled"
    PRERENDER_FAILED_DURING_PREFETCH = "PrerenderFailedDuringPrefetch"
    BROWSING_DATA_REMOVED = "BrowsingDataRemoved"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class PreloadingStatus(enum.Enum):
    '''
    Preloading status values, see also PreloadingTriggeringOutcome. This
    status is shared by prefetchStatusUpdated and prerenderStatusUpdated.
    '''
    PENDING = "Pending"
    RUNNING = "Running"
    READY = "Ready"
    SUCCESS = "Success"
    FAILURE = "Failure"
    NOT_SUPPORTED = "NotSupported"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class PrefetchStatus(enum.Enum):
    '''
    TODO(https://crbug.com/1384419): revisit the list of PrefetchStatus and
    filter out the ones that aren't necessary to the developers.
    '''
    PREFETCH_ALLOWED = "PrefetchAllowed"
    PREFETCH_FAILED_INELIGIBLE_REDIRECT = "PrefetchFailedIneligibleRedirect"
    PREFETCH_FAILED_INVALID_REDIRECT = "PrefetchFailedInvalidRedirect"
    PREFETCH_FAILED_MIME_NOT_SUPPORTED = "PrefetchFailedMIMENotSupported"
    PREFETCH_FAILED_NET_ERROR = "PrefetchFailedNetError"
    PREFETCH_FAILED_NON2_XX = "PrefetchFailedNon2XX"
    PREFETCH_EVICTED_AFTER_BROWSING_DATA_REMOVED = "PrefetchEvictedAfterBrowsingDataRemoved"
    PREFETCH_EVICTED_AFTER_CANDIDATE_REMOVED = "PrefetchEvictedAfterCandidateRemoved"
    PREFETCH_EVICTED_FOR_NEWER_PREFETCH = "PrefetchEvictedForNewerPrefetch"
    PREFETCH_HELDBACK = "PrefetchHeldback"
    PREFETCH_INELIGIBLE_RETRY_AFTER = "PrefetchIneligibleRetryAfter"
    PREFETCH_IS_PRIVACY_DECOY = "PrefetchIsPrivacyDecoy"
    PREFETCH_IS_STALE = "PrefetchIsStale"
    PREFETCH_NOT_ELIGIBLE_BROWSER_CONTEXT_OFF_THE_RECORD = "PrefetchNotEligibleBrowserContextOffTheRecord"
    PREFETCH_NOT_ELIGIBLE_DATA_SAVER_ENABLED = "PrefetchNotEligibleDataSaverEnabled"
    PREFETCH_NOT_ELIGIBLE_EXISTING_PROXY = "PrefetchNotEligibleExistingProxy"
    PREFETCH_NOT_ELIGIBLE_HOST_IS_NON_UNIQUE = "PrefetchNotEligibleHostIsNonUnique"
    PREFETCH_NOT_ELIGIBLE_NON_DEFAULT_STORAGE_PARTITION = "PrefetchNotEligibleNonDefaultStoragePartition"
    PREFETCH_NOT_ELIGIBLE_SAME_SITE_CROSS_ORIGIN_PREFETCH_REQUIRED_PROXY = "PrefetchNotEligibleSameSiteCrossOriginPrefetchRequiredProxy"
    PREFETCH_NOT_ELIGIBLE_SCHEME_IS_NOT_HTTPS = "PrefetchNotEligibleSchemeIsNotHttps"
    PREFETCH_NOT_ELIGIBLE_USER_HAS_COOKIES = "PrefetchNotEligibleUserHasCookies"
    PREFETCH_NOT_ELIGIBLE_USER_HAS_SERVICE_WORKER = "PrefetchNotEligibleUserHasServiceWorker"
    PREFETCH_NOT_ELIGIBLE_USER_HAS_SERVICE_WORKER_NO_FETCH_HANDLER = "PrefetchNotEligibleUserHasServiceWorkerNoFetchHandler"
    PREFETCH_NOT_ELIGIBLE_REDIRECT_FROM_SERVICE_WORKER = "PrefetchNotEligibleRedirectFromServiceWorker"
    PREFETCH_NOT_ELIGIBLE_REDIRECT_TO_SERVICE_WORKER = "PrefetchNotEligibleRedirectToServiceWorker"
    PREFETCH_NOT_ELIGIBLE_BATTERY_SAVER_ENABLED = "PrefetchNotEligibleBatterySaverEnabled"
    PREFETCH_NOT_ELIGIBLE_PRELOADING_DISABLED = "PrefetchNotEligiblePreloadingDisabled"
    PREFETCH_NOT_FINISHED_IN_TIME = "PrefetchNotFinishedInTime"
    PREFETCH_NOT_STARTED = "PrefetchNotStarted"
    PREFETCH_NOT_USED_COOKIES_CHANGED = "PrefetchNotUsedCookiesChanged"
    PREFETCH_PROXY_NOT_AVAILABLE = "PrefetchProxyNotAvailable"
    PREFETCH_RESPONSE_USED = "PrefetchResponseUsed"
    PREFETCH_SUCCESSFUL_BUT_NOT_USED = "PrefetchSuccessfulButNotUsed"
    PREFETCH_NOT_USED_PROBE_FAILED = "PrefetchNotUsedProbeFailed"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class PrerenderMismatchedHeaders:
    '''
    Information of headers to be displayed when the header mismatch occurred.
    '''
    header_name: str

    initial_value: typing.Optional[str] = None

    activation_value: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['headerName'] = self.header_name
        if self.initial_value is not None:
            json['initialValue'] = self.initial_value
        if self.activation_value is not None:
            json['activationValue'] = self.activation_value
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            header_name=str(json['headerName']),
            initial_value=str(json['initialValue']) if 'initialValue' in json else None,
            activation_value=str(json['activationValue']) if 'activationValue' in json else None,
        )


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:

    cmd_dict: T_JSON_DICT = {
        'method': 'Preload.enable',
    }
    json = yield cmd_dict


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:

    cmd_dict: T_JSON_DICT = {
        'method': 'Preload.disable',
    }
    json = yield cmd_dict


@event_class('Preload.ruleSetUpdated')
@dataclass
class RuleSetUpdated:
    '''
    Upsert. Currently, it is only emitted when a rule set added.
    '''
    rule_set: RuleSet

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> RuleSetUpdated:
        return cls(
            rule_set=RuleSet.from_json(json['ruleSet'])
        )


@event_class('Preload.ruleSetRemoved')
@dataclass
class RuleSetRemoved:
    id_: RuleSetId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> RuleSetRemoved:
        return cls(
            id_=RuleSetId.from_json(json['id'])
        )


@event_class('Preload.preloadEnabledStateUpdated')
@dataclass
class PreloadEnabledStateUpdated:
    '''
    Fired when a preload enabled state is updated.
    '''
    disabled_by_preference: bool
    disabled_by_data_saver: bool
    disabled_by_battery_saver: bool
    disabled_by_holdback_prefetch_speculation_rules: bool
    disabled_by_holdback_prerender_speculation_rules: bool

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> PreloadEnabledStateUpdated:
        return cls(
            disabled_by_preference=bool(json['disabledByPreference']),
            disabled_by_data_saver=bool(json['disabledByDataSaver']),
            disabled_by_battery_saver=bool(json['disabledByBatterySaver']),
            disabled_by_holdback_prefetch_speculation_rules=bool(json['disabledByHoldbackPrefetchSpeculationRules']),
            disabled_by_holdback_prerender_speculation_rules=bool(json['disabledByHoldbackPrerenderSpeculationRules'])
        )


@event_class('Preload.prefetchStatusUpdated')
@dataclass
class PrefetchStatusUpdated:
    '''
    Fired when a prefetch attempt is updated.
    '''
    key: PreloadingAttemptKey
    pipeline_id: PreloadPipelineId
    #: The frame id of the frame initiating prefetch.
    initiating_frame_id: page.FrameId
    prefetch_url: str
    status: PreloadingStatus
    prefetch_status: PrefetchStatus
    request_id: network.RequestId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> PrefetchStatusUpdated:
        return cls(
            key=PreloadingAttemptKey.from_json(json['key']),
            pipeline_id=PreloadPipelineId.from_json(json['pipelineId']),
            initiating_frame_id=page.FrameId.from_json(json['initiatingFrameId']),
            prefetch_url=str(json['prefetchUrl']),
            status=PreloadingStatus.from_json(json['status']),
            prefetch_status=PrefetchStatus.from_json(json['prefetchStatus']),
            request_id=network.RequestId.from_json(json['requestId'])
        )


@event_class('Preload.prerenderStatusUpdated')
@dataclass
class PrerenderStatusUpdated:
    '''
    Fired when a prerender attempt is updated.
    '''
    key: PreloadingAttemptKey
    pipeline_id: PreloadPipelineId
    status: PreloadingStatus
    prerender_status: typing.Optional[PrerenderFinalStatus]
    #: This is used to give users more information about the name of Mojo interface
    #: that is incompatible with prerender and has caused the cancellation of the attempt.
    disallowed_mojo_interface: typing.Optional[str]
    mismatched_headers: typing.Optional[typing.List[PrerenderMismatchedHeaders]]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> PrerenderStatusUpdated:
        return cls(
            key=PreloadingAttemptKey.from_json(json['key']),
            pipeline_id=PreloadPipelineId.from_json(json['pipelineId']),
            status=PreloadingStatus.from_json(json['status']),
            prerender_status=PrerenderFinalStatus.from_json(json['prerenderStatus']) if 'prerenderStatus' in json else None,
            disallowed_mojo_interface=str(json['disallowedMojoInterface']) if 'disallowedMojoInterface' in json else None,
            mismatched_headers=[PrerenderMismatchedHeaders.from_json(i) for i in json['mismatchedHeaders']] if 'mismatchedHeaders' in json else None
        )


@event_class('Preload.preloadingAttemptSourcesUpdated')
@dataclass
class PreloadingAttemptSourcesUpdated:
    '''
    Send a list of sources for all preloading attempts in a document.
    '''
    loader_id: network.LoaderId
    preloading_attempt_sources: typing.List[PreloadingAttemptSource]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> PreloadingAttemptSourcesUpdated:
        return cls(
            loader_id=network.LoaderId.from_json(json['loaderId']),
            preloading_attempt_sources=[PreloadingAttemptSource.from_json(i) for i in json['preloadingAttemptSources']]
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\codegen\cfunctions.py ===
"""
This module contains SymPy functions mathcin corresponding to special math functions in the
C standard library (since C99, also available in C++11).

The functions defined in this module allows the user to express functions such as ``expm1``
as a SymPy function for symbolic manipulation.

"""
from sympy.core.function import ArgumentIndexError, Function
from sympy.core.numbers import Rational
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.logic.boolalg import BooleanFunction, true, false

def _expm1(x):
    return exp(x) - S.One


class expm1(Function):
    """
    Represents the exponential function minus one.

    Explanation
    ===========

    The benefit of using ``expm1(x)`` over ``exp(x) - 1``
    is that the latter is prone to cancellation under finite precision
    arithmetic when x is close to zero.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.codegen.cfunctions import expm1
    >>> '%.0e' % expm1(1e-99).evalf()
    '1e-99'
    >>> from math import exp
    >>> exp(1e-99) - 1
    0.0
    >>> expm1(x).diff(x)
    exp(x)

    See Also
    ========

    log1p
    """
    nargs = 1

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex == 1:
            return exp(*self.args)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_expand_func(self, **hints):
        return _expm1(*self.args)

    def _eval_rewrite_as_exp(self, arg, **kwargs):
        return exp(arg) - S.One

    _eval_rewrite_as_tractable = _eval_rewrite_as_exp

    @classmethod
    def eval(cls, arg):
        exp_arg = exp.eval(arg)
        if exp_arg is not None:
            return exp_arg - S.One

    def _eval_is_real(self):
        return self.args[0].is_real

    def _eval_is_finite(self):
        return self.args[0].is_finite


def _log1p(x):
    return log(x + S.One)


class log1p(Function):
    """
    Represents the natural logarithm of a number plus one.

    Explanation
    ===========

    The benefit of using ``log1p(x)`` over ``log(x + 1)``
    is that the latter is prone to cancellation under finite precision
    arithmetic when x is close to zero.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.codegen.cfunctions import log1p
    >>> from sympy import expand_log
    >>> '%.0e' % expand_log(log1p(1e-99)).evalf()
    '1e-99'
    >>> from math import log
    >>> log(1 + 1e-99)
    0.0
    >>> log1p(x).diff(x)
    1/(x + 1)

    See Also
    ========

    expm1
    """
    nargs = 1


    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex == 1:
            return S.One/(self.args[0] + S.One)
        else:
            raise ArgumentIndexError(self, argindex)


    def _eval_expand_func(self, **hints):
        return _log1p(*self.args)

    def _eval_rewrite_as_log(self, arg, **kwargs):
        return _log1p(arg)

    _eval_rewrite_as_tractable = _eval_rewrite_as_log

    @classmethod
    def eval(cls, arg):
        if arg.is_Rational:
            return log(arg + S.One)
        elif not arg.is_Float:  # not safe to add 1 to Float
            return log.eval(arg + S.One)
        elif arg.is_number:
            return log(Rational(arg) + S.One)

    def _eval_is_real(self):
        return (self.args[0] + S.One).is_nonnegative

    def _eval_is_finite(self):
        if (self.args[0] + S.One).is_zero:
            return False
        return self.args[0].is_finite

    def _eval_is_positive(self):
        return self.args[0].is_positive

    def _eval_is_zero(self):
        return self.args[0].is_zero

    def _eval_is_nonnegative(self):
        return self.args[0].is_nonnegative

_Two = S(2)

def _exp2(x):
    return Pow(_Two, x)

class exp2(Function):
    """
    Represents the exponential function with base two.

    Explanation
    ===========

    The benefit of using ``exp2(x)`` over ``2**x``
    is that the latter is not as efficient under finite precision
    arithmetic.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.codegen.cfunctions import exp2
    >>> exp2(2).evalf() == 4.0
    True
    >>> exp2(x).diff(x)
    log(2)*exp2(x)

    See Also
    ========

    log2
    """
    nargs = 1


    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex == 1:
            return self*log(_Two)
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_rewrite_as_Pow(self, arg, **kwargs):
        return _exp2(arg)

    _eval_rewrite_as_tractable = _eval_rewrite_as_Pow

    def _eval_expand_func(self, **hints):
        return _exp2(*self.args)

    @classmethod
    def eval(cls, arg):
        if arg.is_number:
            return _exp2(arg)


def _log2(x):
    return log(x)/log(_Two)


class log2(Function):
    """
    Represents the logarithm function with base two.

    Explanation
    ===========

    The benefit of using ``log2(x)`` over ``log(x)/log(2)``
    is that the latter is not as efficient under finite precision
    arithmetic.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.codegen.cfunctions import log2
    >>> log2(4).evalf() == 2.0
    True
    >>> log2(x).diff(x)
    1/(x*log(2))

    See Also
    ========

    exp2
    log10
    """
    nargs = 1

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex == 1:
            return S.One/(log(_Two)*self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)


    @classmethod
    def eval(cls, arg):
        if arg.is_number:
            result = log.eval(arg, base=_Two)
            if result.is_Atom:
                return result
        elif arg.is_Pow and arg.base == _Two:
            return arg.exp

    def _eval_evalf(self, *args, **kwargs):
        return self.rewrite(log).evalf(*args, **kwargs)

    def _eval_expand_func(self, **hints):
        return _log2(*self.args)

    def _eval_rewrite_as_log(self, arg, **kwargs):
        return _log2(arg)

    _eval_rewrite_as_tractable = _eval_rewrite_as_log


def _fma(x, y, z):
    return x*y + z


class fma(Function):
    """
    Represents "fused multiply add".

    Explanation
    ===========

    The benefit of using ``fma(x, y, z)`` over ``x*y + z``
    is that, under finite precision arithmetic, the former is
    supported by special instructions on some CPUs.

    Examples
    ========

    >>> from sympy.abc import x, y, z
    >>> from sympy.codegen.cfunctions import fma
    >>> fma(x, y, z).diff(x)
    y

    """
    nargs = 3

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex in (1, 2):
            return self.args[2 - argindex]
        elif argindex == 3:
            return S.One
        else:
            raise ArgumentIndexError(self, argindex)


    def _eval_expand_func(self, **hints):
        return _fma(*self.args)

    def _eval_rewrite_as_tractable(self, arg, limitvar=None, **kwargs):
        return _fma(arg)


_Ten = S(10)


def _log10(x):
    return log(x)/log(_Ten)


class log10(Function):
    """
    Represents the logarithm function with base ten.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.codegen.cfunctions import log10
    >>> log10(100).evalf() == 2.0
    True
    >>> log10(x).diff(x)
    1/(x*log(10))

    See Also
    ========

    log2
    """
    nargs = 1

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex == 1:
            return S.One/(log(_Ten)*self.args[0])
        else:
            raise ArgumentIndexError(self, argindex)


    @classmethod
    def eval(cls, arg):
        if arg.is_number:
            result = log.eval(arg, base=_Ten)
            if result.is_Atom:
                return result
        elif arg.is_Pow and arg.base == _Ten:
            return arg.exp

    def _eval_expand_func(self, **hints):
        return _log10(*self.args)

    def _eval_rewrite_as_log(self, arg, **kwargs):
        return _log10(arg)

    _eval_rewrite_as_tractable = _eval_rewrite_as_log


def _Sqrt(x):
    return Pow(x, S.Half)


class Sqrt(Function):  # 'sqrt' already defined in sympy.functions.elementary.miscellaneous
    """
    Represents the square root function.

    Explanation
    ===========

    The reason why one would use ``Sqrt(x)`` over ``sqrt(x)``
    is that the latter is internally represented as ``Pow(x, S.Half)`` which
    may not be what one wants when doing code-generation.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.codegen.cfunctions import Sqrt
    >>> Sqrt(x)
    Sqrt(x)
    >>> Sqrt(x).diff(x)
    1/(2*sqrt(x))

    See Also
    ========

    Cbrt
    """
    nargs = 1

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex == 1:
            return Pow(self.args[0], Rational(-1, 2))/_Two
        else:
            raise ArgumentIndexError(self, argindex)

    def _eval_expand_func(self, **hints):
        return _Sqrt(*self.args)

    def _eval_rewrite_as_Pow(self, arg, **kwargs):
        return _Sqrt(arg)

    _eval_rewrite_as_tractable = _eval_rewrite_as_Pow


def _Cbrt(x):
    return Pow(x, Rational(1, 3))


class Cbrt(Function):  # 'cbrt' already defined in sympy.functions.elementary.miscellaneous
    """
    Represents the cube root function.

    Explanation
    ===========

    The reason why one would use ``Cbrt(x)`` over ``cbrt(x)``
    is that the latter is internally represented as ``Pow(x, Rational(1, 3))`` which
    may not be what one wants when doing code-generation.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.codegen.cfunctions import Cbrt
    >>> Cbrt(x)
    Cbrt(x)
    >>> Cbrt(x).diff(x)
    1/(3*x**(2/3))

    See Also
    ========

    Sqrt
    """
    nargs = 1

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex == 1:
            return Pow(self.args[0], Rational(-_Two/3))/3
        else:
            raise ArgumentIndexError(self, argindex)


    def _eval_expand_func(self, **hints):
        return _Cbrt(*self.args)

    def _eval_rewrite_as_Pow(self, arg, **kwargs):
        return _Cbrt(arg)

    _eval_rewrite_as_tractable = _eval_rewrite_as_Pow


def _hypot(x, y):
    return sqrt(Pow(x, 2) + Pow(y, 2))


class hypot(Function):
    """
    Represents the hypotenuse function.

    Explanation
    ===========

    The hypotenuse function is provided by e.g. the math library
    in the C99 standard, hence one may want to represent the function
    symbolically when doing code-generation.

    Examples
    ========

    >>> from sympy.abc import x, y
    >>> from sympy.codegen.cfunctions import hypot
    >>> hypot(3, 4).evalf() == 5.0
    True
    >>> hypot(x, y)
    hypot(x, y)
    >>> hypot(x, y).diff(x)
    x/hypot(x, y)

    """
    nargs = 2

    def fdiff(self, argindex=1):
        """
        Returns the first derivative of this function.
        """
        if argindex in (1, 2):
            return 2*self.args[argindex-1]/(_Two*self.func(*self.args))
        else:
            raise ArgumentIndexError(self, argindex)


    def _eval_expand_func(self, **hints):
        return _hypot(*self.args)

    def _eval_rewrite_as_Pow(self, arg, **kwargs):
        return _hypot(arg)

    _eval_rewrite_as_tractable = _eval_rewrite_as_Pow


class isnan(BooleanFunction):
    nargs = 1

    @classmethod
    def eval(cls, arg):
        if arg is S.NaN:
            return true
        elif arg.is_number:
            return false
        else:
            return None


class isinf(BooleanFunction):
    nargs = 1

    @classmethod
    def eval(cls, arg):
        if arg.is_infinite:
            return true
        elif arg.is_finite:
            return false
        else:
            return None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\interchange\from_dataframe.py ===
from __future__ import annotations

import ctypes
import re
from typing import Any

import numpy as np

from pandas._config import using_string_dtype

from pandas.compat._optional import import_optional_dependency
from pandas.errors import SettingWithCopyError

import pandas as pd
from pandas.core.interchange.dataframe_protocol import (
    Buffer,
    Column,
    ColumnNullType,
    DataFrame as DataFrameXchg,
    DtypeKind,
)
from pandas.core.interchange.utils import (
    ArrowCTypes,
    Endianness,
)

_NP_DTYPES: dict[DtypeKind, dict[int, Any]] = {
    DtypeKind.INT: {8: np.int8, 16: np.int16, 32: np.int32, 64: np.int64},
    DtypeKind.UINT: {8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64},
    DtypeKind.FLOAT: {32: np.float32, 64: np.float64},
    DtypeKind.BOOL: {1: bool, 8: bool},
}


def from_dataframe(df, allow_copy: bool = True) -> pd.DataFrame:
    """
    Build a ``pd.DataFrame`` from any DataFrame supporting the interchange protocol.

    .. note::

       For new development, we highly recommend using the Arrow C Data Interface
       alongside the Arrow PyCapsule Interface instead of the interchange protocol.
       From pandas 2.3 onwards, `from_dataframe` uses the PyCapsule Interface,
       only falling back to the interchange protocol if that fails.

    .. warning::

        Due to severe implementation issues, we recommend only considering using the
        interchange protocol in the following cases:

        - converting to pandas: for pandas >= 2.0.3
        - converting from pandas: for pandas >= 3.0.0

    Parameters
    ----------
    df : DataFrameXchg
        Object supporting the interchange protocol, i.e. `__dataframe__` method.
    allow_copy : bool, default: True
        Whether to allow copying the memory to perform the conversion
        (if false then zero-copy approach is requested).

    Returns
    -------
    pd.DataFrame

    Examples
    --------
    >>> df_not_necessarily_pandas = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
    >>> interchange_object = df_not_necessarily_pandas.__dataframe__()
    >>> interchange_object.column_names()
    Index(['A', 'B'], dtype='object')
    >>> df_pandas = (pd.api.interchange.from_dataframe
    ...              (interchange_object.select_columns_by_name(['A'])))
    >>> df_pandas
         A
    0    1
    1    2

    These methods (``column_names``, ``select_columns_by_name``) should work
    for any dataframe library which implements the interchange protocol.
    """
    if isinstance(df, pd.DataFrame):
        return df

    if hasattr(df, "__arrow_c_stream__"):
        try:
            pa = import_optional_dependency("pyarrow", min_version="14.0.0")
        except ImportError:
            # fallback to _from_dataframe
            pass
        else:
            try:
                return pa.table(df).to_pandas(zero_copy_only=not allow_copy)
            except pa.ArrowInvalid as e:
                raise RuntimeError(e) from e

    if not hasattr(df, "__dataframe__"):
        raise ValueError("`df` does not support __dataframe__")

    return _from_dataframe(
        df.__dataframe__(allow_copy=allow_copy), allow_copy=allow_copy
    )


def _from_dataframe(df: DataFrameXchg, allow_copy: bool = True):
    """
    Build a ``pd.DataFrame`` from the DataFrame interchange object.

    Parameters
    ----------
    df : DataFrameXchg
        Object supporting the interchange protocol, i.e. `__dataframe__` method.
    allow_copy : bool, default: True
        Whether to allow copying the memory to perform the conversion
        (if false then zero-copy approach is requested).

    Returns
    -------
    pd.DataFrame
    """
    pandas_dfs = []
    for chunk in df.get_chunks():
        pandas_df = protocol_df_chunk_to_pandas(chunk)
        pandas_dfs.append(pandas_df)

    if not allow_copy and len(pandas_dfs) > 1:
        raise RuntimeError(
            "To join chunks a copy is required which is forbidden by allow_copy=False"
        )
    if not pandas_dfs:
        pandas_df = protocol_df_chunk_to_pandas(df)
    elif len(pandas_dfs) == 1:
        pandas_df = pandas_dfs[0]
    else:
        pandas_df = pd.concat(pandas_dfs, axis=0, ignore_index=True, copy=False)

    index_obj = df.metadata.get("pandas.index", None)
    if index_obj is not None:
        pandas_df.index = index_obj

    return pandas_df


def protocol_df_chunk_to_pandas(df: DataFrameXchg) -> pd.DataFrame:
    """
    Convert interchange protocol chunk to ``pd.DataFrame``.

    Parameters
    ----------
    df : DataFrameXchg

    Returns
    -------
    pd.DataFrame
    """
    columns: dict[str, Any] = {}
    buffers = []  # hold on to buffers, keeps memory alive
    for name in df.column_names():
        if not isinstance(name, str):
            raise ValueError(f"Column {name} is not a string")
        if name in columns:
            raise ValueError(f"Column {name} is not unique")
        col = df.get_column_by_name(name)
        dtype = col.dtype[0]
        if dtype in (
            DtypeKind.INT,
            DtypeKind.UINT,
            DtypeKind.FLOAT,
            DtypeKind.BOOL,
        ):
            columns[name], buf = primitive_column_to_ndarray(col)
        elif dtype == DtypeKind.CATEGORICAL:
            columns[name], buf = categorical_column_to_series(col)
        elif dtype == DtypeKind.STRING:
            columns[name], buf = string_column_to_ndarray(col)
        elif dtype == DtypeKind.DATETIME:
            columns[name], buf = datetime_column_to_ndarray(col)
        else:
            raise NotImplementedError(f"Data type {dtype} not handled yet")

        buffers.append(buf)

    pandas_df = pd.DataFrame(columns)
    pandas_df.attrs["_INTERCHANGE_PROTOCOL_BUFFERS"] = buffers
    return pandas_df


def primitive_column_to_ndarray(col: Column) -> tuple[np.ndarray, Any]:
    """
    Convert a column holding one of the primitive dtypes to a NumPy array.

    A primitive type is one of: int, uint, float, bool.

    Parameters
    ----------
    col : Column

    Returns
    -------
    tuple
        Tuple of np.ndarray holding the data and the memory owner object
        that keeps the memory alive.
    """
    buffers = col.get_buffers()

    data_buff, data_dtype = buffers["data"]
    data = buffer_to_ndarray(
        data_buff, data_dtype, offset=col.offset, length=col.size()
    )

    data = set_nulls(data, col, buffers["validity"])
    return data, buffers


def categorical_column_to_series(col: Column) -> tuple[pd.Series, Any]:
    """
    Convert a column holding categorical data to a pandas Series.

    Parameters
    ----------
    col : Column

    Returns
    -------
    tuple
        Tuple of pd.Series holding the data and the memory owner object
        that keeps the memory alive.
    """
    categorical = col.describe_categorical

    if not categorical["is_dictionary"]:
        raise NotImplementedError("Non-dictionary categoricals not supported yet")

    cat_column = categorical["categories"]
    if hasattr(cat_column, "_col"):
        # Item "Column" of "Optional[Column]" has no attribute "_col"
        # Item "None" of "Optional[Column]" has no attribute "_col"
        categories = np.array(cat_column._col)  # type: ignore[union-attr]
    else:
        raise NotImplementedError(
            "Interchanging categorical columns isn't supported yet, and our "
            "fallback of using the `col._col` attribute (a ndarray) failed."
        )
    buffers = col.get_buffers()

    codes_buff, codes_dtype = buffers["data"]
    codes = buffer_to_ndarray(
        codes_buff, codes_dtype, offset=col.offset, length=col.size()
    )

    # Doing module in order to not get ``IndexError`` for
    # out-of-bounds sentinel values in `codes`
    if len(categories) > 0:
        values = categories[codes % len(categories)]
    else:
        values = codes

    cat = pd.Categorical(
        values, categories=categories, ordered=categorical["is_ordered"]
    )
    data = pd.Series(cat)

    data = set_nulls(data, col, buffers["validity"])
    return data, buffers


def string_column_to_ndarray(col: Column) -> tuple[np.ndarray, Any]:
    """
    Convert a column holding string data to a NumPy array.

    Parameters
    ----------
    col : Column

    Returns
    -------
    tuple
        Tuple of np.ndarray holding the data and the memory owner object
        that keeps the memory alive.
    """
    null_kind, sentinel_val = col.describe_null

    if null_kind not in (
        ColumnNullType.NON_NULLABLE,
        ColumnNullType.USE_BITMASK,
        ColumnNullType.USE_BYTEMASK,
    ):
        raise NotImplementedError(
            f"{null_kind} null kind is not yet supported for string columns."
        )

    buffers = col.get_buffers()

    assert buffers["offsets"], "String buffers must contain offsets"
    # Retrieve the data buffer containing the UTF-8 code units
    data_buff, _ = buffers["data"]
    # We're going to reinterpret the buffer as uint8, so make sure we can do it safely
    assert col.dtype[2] in (
        ArrowCTypes.STRING,
        ArrowCTypes.LARGE_STRING,
    )  # format_str == utf-8
    # Convert the buffers to NumPy arrays. In order to go from STRING to
    # an equivalent ndarray, we claim that the buffer is uint8 (i.e., a byte array)
    data_dtype = (
        DtypeKind.UINT,
        8,
        ArrowCTypes.UINT8,
        Endianness.NATIVE,
    )
    # Specify zero offset as we don't want to chunk the string data
    data = buffer_to_ndarray(data_buff, data_dtype, offset=0, length=data_buff.bufsize)

    # Retrieve the offsets buffer containing the index offsets demarcating
    # the beginning and the ending of each string
    offset_buff, offset_dtype = buffers["offsets"]
    # Offsets buffer contains start-stop positions of strings in the data buffer,
    # meaning that it has more elements than in the data buffer, do `col.size() + 1`
    # here to pass a proper offsets buffer size
    offsets = buffer_to_ndarray(
        offset_buff, offset_dtype, offset=col.offset, length=col.size() + 1
    )

    null_pos = None
    if null_kind in (ColumnNullType.USE_BITMASK, ColumnNullType.USE_BYTEMASK):
        validity = buffers["validity"]
        if validity is not None:
            valid_buff, valid_dtype = validity
            null_pos = buffer_to_ndarray(
                valid_buff, valid_dtype, offset=col.offset, length=col.size()
            )
            if sentinel_val == 0:
                null_pos = ~null_pos

    # Assemble the strings from the code units
    str_list: list[None | float | str] = [None] * col.size()
    for i in range(col.size()):
        # Check for missing values
        if null_pos is not None and null_pos[i]:
            str_list[i] = np.nan
            continue

        # Extract a range of code units
        units = data[offsets[i] : offsets[i + 1]]

        # Convert the list of code units to bytes
        str_bytes = bytes(units)

        # Create the string
        string = str_bytes.decode(encoding="utf-8")

        # Add to our list of strings
        str_list[i] = string

    if using_string_dtype():
        res = pd.Series(str_list, dtype="str")
    else:
        res = np.asarray(str_list, dtype="object")  # type: ignore[assignment]

    return res, buffers  # type: ignore[return-value]


def parse_datetime_format_str(format_str, data) -> pd.Series | np.ndarray:
    """Parse datetime `format_str` to interpret the `data`."""
    # timestamp 'ts{unit}:tz'
    timestamp_meta = re.match(r"ts([smun]):(.*)", format_str)
    if timestamp_meta:
        unit, tz = timestamp_meta.group(1), timestamp_meta.group(2)
        if unit != "s":
            # the format string describes only a first letter of the unit, so
            # add one extra letter to convert the unit to numpy-style:
            # 'm' -> 'ms', 'u' -> 'us', 'n' -> 'ns'
            unit += "s"
        data = data.astype(f"datetime64[{unit}]")
        if tz != "":
            data = pd.Series(data).dt.tz_localize("UTC").dt.tz_convert(tz)
        return data

    # date 'td{Days/Ms}'
    date_meta = re.match(r"td([Dm])", format_str)
    if date_meta:
        unit = date_meta.group(1)
        if unit == "D":
            # NumPy doesn't support DAY unit, so converting days to seconds
            # (converting to uint64 to avoid overflow)
            data = (data.astype(np.uint64) * (24 * 60 * 60)).astype("datetime64[s]")
        elif unit == "m":
            data = data.astype("datetime64[ms]")
        else:
            raise NotImplementedError(f"Date unit is not supported: {unit}")
        return data

    raise NotImplementedError(f"DateTime kind is not supported: {format_str}")


def datetime_column_to_ndarray(col: Column) -> tuple[np.ndarray | pd.Series, Any]:
    """
    Convert a column holding DateTime data to a NumPy array.

    Parameters
    ----------
    col : Column

    Returns
    -------
    tuple
        Tuple of np.ndarray holding the data and the memory owner object
        that keeps the memory alive.
    """
    buffers = col.get_buffers()

    _, col_bit_width, format_str, _ = col.dtype
    dbuf, _ = buffers["data"]
    # Consider dtype being `uint` to get number of units passed since the 01.01.1970

    data = buffer_to_ndarray(
        dbuf,
        (
            DtypeKind.INT,
            col_bit_width,
            getattr(ArrowCTypes, f"INT{col_bit_width}"),
            Endianness.NATIVE,
        ),
        offset=col.offset,
        length=col.size(),
    )

    data = parse_datetime_format_str(format_str, data)  # type: ignore[assignment]
    data = set_nulls(data, col, buffers["validity"])
    return data, buffers


def buffer_to_ndarray(
    buffer: Buffer,
    dtype: tuple[DtypeKind, int, str, str],
    *,
    length: int,
    offset: int = 0,
) -> np.ndarray:
    """
    Build a NumPy array from the passed buffer.

    Parameters
    ----------
    buffer : Buffer
        Buffer to build a NumPy array from.
    dtype : tuple
        Data type of the buffer conforming protocol dtypes format.
    offset : int, default: 0
        Number of elements to offset from the start of the buffer.
    length : int, optional
        If the buffer is a bit-mask, specifies a number of bits to read
        from the buffer. Has no effect otherwise.

    Returns
    -------
    np.ndarray

    Notes
    -----
    The returned array doesn't own the memory. The caller of this function is
    responsible for keeping the memory owner object alive as long as
    the returned NumPy array is being used.
    """
    kind, bit_width, _, _ = dtype

    column_dtype = _NP_DTYPES.get(kind, {}).get(bit_width, None)
    if column_dtype is None:
        raise NotImplementedError(f"Conversion for {dtype} is not yet supported.")

    # TODO: No DLPack yet, so need to construct a new ndarray from the data pointer
    # and size in the buffer plus the dtype on the column. Use DLPack as NumPy supports
    # it since https://github.com/numpy/numpy/pull/19083
    ctypes_type = np.ctypeslib.as_ctypes_type(column_dtype)

    if bit_width == 1:
        assert length is not None, "`length` must be specified for a bit-mask buffer."
        pa = import_optional_dependency("pyarrow")
        arr = pa.BooleanArray.from_buffers(
            pa.bool_(),
            length,
            [None, pa.foreign_buffer(buffer.ptr, length)],
            offset=offset,
        )
        return np.asarray(arr)
    else:
        data_pointer = ctypes.cast(
            buffer.ptr + (offset * bit_width // 8), ctypes.POINTER(ctypes_type)
        )
        if length > 0:
            return np.ctypeslib.as_array(data_pointer, shape=(length,))
        return np.array([], dtype=ctypes_type)


def set_nulls(
    data: np.ndarray | pd.Series,
    col: Column,
    validity: tuple[Buffer, tuple[DtypeKind, int, str, str]] | None,
    allow_modify_inplace: bool = True,
):
    """
    Set null values for the data according to the column null kind.

    Parameters
    ----------
    data : np.ndarray or pd.Series
        Data to set nulls in.
    col : Column
        Column object that describes the `data`.
    validity : tuple(Buffer, dtype) or None
        The return value of ``col.buffers()``. We do not access the ``col.buffers()``
        here to not take the ownership of the memory of buffer objects.
    allow_modify_inplace : bool, default: True
        Whether to modify the `data` inplace when zero-copy is possible (True) or always
        modify a copy of the `data` (False).

    Returns
    -------
    np.ndarray or pd.Series
        Data with the nulls being set.
    """
    if validity is None:
        return data
    null_kind, sentinel_val = col.describe_null
    null_pos = None

    if null_kind == ColumnNullType.USE_SENTINEL:
        null_pos = pd.Series(data) == sentinel_val
    elif null_kind in (ColumnNullType.USE_BITMASK, ColumnNullType.USE_BYTEMASK):
        assert validity, "Expected to have a validity buffer for the mask"
        valid_buff, valid_dtype = validity
        null_pos = buffer_to_ndarray(
            valid_buff, valid_dtype, offset=col.offset, length=col.size()
        )
        if sentinel_val == 0:
            null_pos = ~null_pos
    elif null_kind in (ColumnNullType.NON_NULLABLE, ColumnNullType.USE_NAN):
        pass
    else:
        raise NotImplementedError(f"Null kind {null_kind} is not yet supported.")

    if null_pos is not None and np.any(null_pos):
        if not allow_modify_inplace:
            data = data.copy()
        try:
            data[null_pos] = None
        except TypeError:
            # TypeError happens if the `data` dtype appears to be non-nullable
            # in numpy notation (bool, int, uint). If this happens,
            # cast the `data` to nullable float dtype.
            data = data.astype(float)
            data[null_pos] = None
        except SettingWithCopyError:
            # `SettingWithCopyError` may happen for datetime-like with missing values.
            data = data.copy()
            data[null_pos] = None

    return data

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\orm\mapped_collection.py ===
# orm/mapped_collection.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

from __future__ import annotations

import operator
from typing import Any
from typing import Callable
from typing import Dict
from typing import Generic
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

from . import base
from .collections import collection
from .collections import collection_adapter
from .. import exc as sa_exc
from .. import util
from ..sql import coercions
from ..sql import expression
from ..sql import roles
from ..util.langhelpers import Missing
from ..util.langhelpers import MissingOr
from ..util.typing import Literal

if TYPE_CHECKING:
    from . import AttributeEventToken
    from . import Mapper
    from .collections import CollectionAdapter
    from ..sql.elements import ColumnElement

_KT = TypeVar("_KT", bound=Any)
_VT = TypeVar("_VT", bound=Any)


class _PlainColumnGetter(Generic[_KT]):
    """Plain column getter, stores collection of Column objects
    directly.

    Serializes to a :class:`._SerializableColumnGetterV2`
    which has more expensive __call__() performance
    and some rare caveats.

    """

    __slots__ = ("cols", "composite")

    def __init__(self, cols: Sequence[ColumnElement[_KT]]) -> None:
        self.cols = cols
        self.composite = len(cols) > 1

    def __reduce__(
        self,
    ) -> Tuple[
        Type[_SerializableColumnGetterV2[_KT]],
        Tuple[Sequence[Tuple[Optional[str], Optional[str]]]],
    ]:
        return _SerializableColumnGetterV2._reduce_from_cols(self.cols)

    def _cols(self, mapper: Mapper[_KT]) -> Sequence[ColumnElement[_KT]]:
        return self.cols

    def __call__(self, value: _KT) -> MissingOr[Union[_KT, Tuple[_KT, ...]]]:
        state = base.instance_state(value)
        m = base._state_mapper(state)

        key: List[_KT] = [
            m._get_state_attr_by_column(state, state.dict, col)
            for col in self._cols(m)
        ]
        if self.composite:
            return tuple(key)
        else:
            obj = key[0]
            if obj is None:
                return Missing
            else:
                return obj


class _SerializableColumnGetterV2(_PlainColumnGetter[_KT]):
    """Updated serializable getter which deals with
    multi-table mapped classes.

    Two extremely unusual cases are not supported.
    Mappings which have tables across multiple metadata
    objects, or which are mapped to non-Table selectables
    linked across inheriting mappers may fail to function
    here.

    """

    __slots__ = ("colkeys",)

    def __init__(
        self, colkeys: Sequence[Tuple[Optional[str], Optional[str]]]
    ) -> None:
        self.colkeys = colkeys
        self.composite = len(colkeys) > 1

    def __reduce__(
        self,
    ) -> Tuple[
        Type[_SerializableColumnGetterV2[_KT]],
        Tuple[Sequence[Tuple[Optional[str], Optional[str]]]],
    ]:
        return self.__class__, (self.colkeys,)

    @classmethod
    def _reduce_from_cols(cls, cols: Sequence[ColumnElement[_KT]]) -> Tuple[
        Type[_SerializableColumnGetterV2[_KT]],
        Tuple[Sequence[Tuple[Optional[str], Optional[str]]]],
    ]:
        def _table_key(c: ColumnElement[_KT]) -> Optional[str]:
            if not isinstance(c.table, expression.TableClause):
                return None
            else:
                return c.table.key  # type: ignore

        colkeys = [(c.key, _table_key(c)) for c in cols]
        return _SerializableColumnGetterV2, (colkeys,)

    def _cols(self, mapper: Mapper[_KT]) -> Sequence[ColumnElement[_KT]]:
        cols: List[ColumnElement[_KT]] = []
        metadata = getattr(mapper.local_table, "metadata", None)
        for ckey, tkey in self.colkeys:
            if tkey is None or metadata is None or tkey not in metadata:
                cols.append(mapper.local_table.c[ckey])  # type: ignore
            else:
                cols.append(metadata.tables[tkey].c[ckey])
        return cols


def column_keyed_dict(
    mapping_spec: Union[Type[_KT], Callable[[_KT], _VT]],
    *,
    ignore_unpopulated_attribute: bool = False,
) -> Type[KeyFuncDict[_KT, _KT]]:
    """A dictionary-based collection type with column-based keying.

    .. versionchanged:: 2.0 Renamed :data:`.column_mapped_collection` to
       :class:`.column_keyed_dict`.

    Returns a :class:`.KeyFuncDict` factory which will produce new
    dictionary keys based on the value of a particular :class:`.Column`-mapped
    attribute on ORM mapped instances to be added to the dictionary.

    .. note:: the value of the target attribute must be assigned with its
       value at the time that the object is being added to the
       dictionary collection.   Additionally, changes to the key attribute
       are **not tracked**, which means the key in the dictionary is not
       automatically synchronized with the key value on the target object
       itself.  See :ref:`key_collections_mutations` for further details.

    .. seealso::

        :ref:`orm_dictionary_collection` - background on use

    :param mapping_spec: a :class:`_schema.Column` object that is expected
     to be mapped by the target mapper to a particular attribute on the
     mapped class, the value of which on a particular instance is to be used
     as the key for a new dictionary entry for that instance.
    :param ignore_unpopulated_attribute:  if True, and the mapped attribute
     indicated by the given :class:`_schema.Column` target attribute
     on an object is not populated at all, the operation will be silently
     skipped.  By default, an error is raised.

     .. versionadded:: 2.0 an error is raised by default if the attribute
        being used for the dictionary key is determined that it was never
        populated with any value.  The
        :paramref:`_orm.column_keyed_dict.ignore_unpopulated_attribute`
        parameter may be set which will instead indicate that this condition
        should be ignored, and the append operation silently skipped.
        This is in contrast to the behavior of the 1.x series which would
        erroneously populate the value in the dictionary with an arbitrary key
        value of ``None``.


    """
    cols = [
        coercions.expect(roles.ColumnArgumentRole, q, argname="mapping_spec")
        for q in util.to_list(mapping_spec)
    ]
    keyfunc = _PlainColumnGetter(cols)
    return _mapped_collection_cls(
        keyfunc,
        ignore_unpopulated_attribute=ignore_unpopulated_attribute,
    )


class _AttrGetter:
    __slots__ = ("attr_name", "getter")

    def __init__(self, attr_name: str):
        self.attr_name = attr_name
        self.getter = operator.attrgetter(attr_name)

    def __call__(self, mapped_object: Any) -> Any:
        obj = self.getter(mapped_object)
        if obj is None:
            state = base.instance_state(mapped_object)
            mp = state.mapper
            if self.attr_name in mp.attrs:
                dict_ = state.dict
                obj = dict_.get(self.attr_name, base.NO_VALUE)
                if obj is None:
                    return Missing
            else:
                return Missing

        return obj

    def __reduce__(self) -> Tuple[Type[_AttrGetter], Tuple[str]]:
        return _AttrGetter, (self.attr_name,)


def attribute_keyed_dict(
    attr_name: str, *, ignore_unpopulated_attribute: bool = False
) -> Type[KeyFuncDict[Any, Any]]:
    """A dictionary-based collection type with attribute-based keying.

    .. versionchanged:: 2.0 Renamed :data:`.attribute_mapped_collection` to
       :func:`.attribute_keyed_dict`.

    Returns a :class:`.KeyFuncDict` factory which will produce new
    dictionary keys based on the value of a particular named attribute on
    ORM mapped instances to be added to the dictionary.

    .. note:: the value of the target attribute must be assigned with its
       value at the time that the object is being added to the
       dictionary collection.   Additionally, changes to the key attribute
       are **not tracked**, which means the key in the dictionary is not
       automatically synchronized with the key value on the target object
       itself.  See :ref:`key_collections_mutations` for further details.

    .. seealso::

        :ref:`orm_dictionary_collection` - background on use

    :param attr_name: string name of an ORM-mapped attribute
     on the mapped class, the value of which on a particular instance
     is to be used as the key for a new dictionary entry for that instance.
    :param ignore_unpopulated_attribute:  if True, and the target attribute
     on an object is not populated at all, the operation will be silently
     skipped.  By default, an error is raised.

     .. versionadded:: 2.0 an error is raised by default if the attribute
        being used for the dictionary key is determined that it was never
        populated with any value.  The
        :paramref:`_orm.attribute_keyed_dict.ignore_unpopulated_attribute`
        parameter may be set which will instead indicate that this condition
        should be ignored, and the append operation silently skipped.
        This is in contrast to the behavior of the 1.x series which would
        erroneously populate the value in the dictionary with an arbitrary key
        value of ``None``.


    """

    return _mapped_collection_cls(
        _AttrGetter(attr_name),
        ignore_unpopulated_attribute=ignore_unpopulated_attribute,
    )


def keyfunc_mapping(
    keyfunc: Callable[[Any], Any],
    *,
    ignore_unpopulated_attribute: bool = False,
) -> Type[KeyFuncDict[_KT, Any]]:
    """A dictionary-based collection type with arbitrary keying.

    .. versionchanged:: 2.0 Renamed :data:`.mapped_collection` to
       :func:`.keyfunc_mapping`.

    Returns a :class:`.KeyFuncDict` factory with a keying function
    generated from keyfunc, a callable that takes an entity and returns a
    key value.

    .. note:: the given keyfunc is called only once at the time that the
       target object is being added to the collection.   Changes to the
       effective value returned by the function are not tracked.


    .. seealso::

        :ref:`orm_dictionary_collection` - background on use

    :param keyfunc: a callable that will be passed the ORM-mapped instance
     which should then generate a new key to use in the dictionary.
     If the value returned is :attr:`.LoaderCallableStatus.NO_VALUE`, an error
     is raised.
    :param ignore_unpopulated_attribute:  if True, and the callable returns
     :attr:`.LoaderCallableStatus.NO_VALUE` for a particular instance, the
     operation will be silently skipped.  By default, an error is raised.

     .. versionadded:: 2.0 an error is raised by default if the callable
        being used for the dictionary key returns
        :attr:`.LoaderCallableStatus.NO_VALUE`, which in an ORM attribute
        context indicates an attribute that was never populated with any value.
        The :paramref:`_orm.mapped_collection.ignore_unpopulated_attribute`
        parameter may be set which will instead indicate that this condition
        should be ignored, and the append operation silently skipped. This is
        in contrast to the behavior of the 1.x series which would erroneously
        populate the value in the dictionary with an arbitrary key value of
        ``None``.


    """
    return _mapped_collection_cls(
        keyfunc, ignore_unpopulated_attribute=ignore_unpopulated_attribute
    )


class KeyFuncDict(Dict[_KT, _VT]):
    """Base for ORM mapped dictionary classes.

    Extends the ``dict`` type with additional methods needed by SQLAlchemy ORM
    collection classes. Use of :class:`_orm.KeyFuncDict` is most directly
    by using the :func:`.attribute_keyed_dict` or
    :func:`.column_keyed_dict` class factories.
    :class:`_orm.KeyFuncDict` may also serve as the base for user-defined
    custom dictionary classes.

    .. versionchanged:: 2.0 Renamed :class:`.MappedCollection` to
       :class:`.KeyFuncDict`.

    .. seealso::

        :func:`_orm.attribute_keyed_dict`

        :func:`_orm.column_keyed_dict`

        :ref:`orm_dictionary_collection`

        :ref:`orm_custom_collection`


    """

    def __init__(
        self,
        keyfunc: Callable[[Any], Any],
        *dict_args: Any,
        ignore_unpopulated_attribute: bool = False,
    ) -> None:
        """Create a new collection with keying provided by keyfunc.

        keyfunc may be any callable that takes an object and returns an object
        for use as a dictionary key.

        The keyfunc will be called every time the ORM needs to add a member by
        value-only (such as when loading instances from the database) or
        remove a member.  The usual cautions about dictionary keying apply-
        ``keyfunc(object)`` should return the same output for the life of the
        collection.  Keying based on mutable properties can result in
        unreachable instances "lost" in the collection.

        """
        self.keyfunc = keyfunc
        self.ignore_unpopulated_attribute = ignore_unpopulated_attribute
        super().__init__(*dict_args)

    @classmethod
    def _unreduce(
        cls,
        keyfunc: Callable[[Any], Any],
        values: Dict[_KT, _KT],
        adapter: Optional[CollectionAdapter] = None,
    ) -> "KeyFuncDict[_KT, _KT]":
        mp: KeyFuncDict[_KT, _KT] = KeyFuncDict(keyfunc)
        mp.update(values)
        # note that the adapter sets itself up onto this collection
        # when its `__setstate__` method is called
        return mp

    def __reduce__(
        self,
    ) -> Tuple[
        Callable[[_KT, _KT], KeyFuncDict[_KT, _KT]],
        Tuple[Any, Union[Dict[_KT, _KT], Dict[_KT, _KT]], CollectionAdapter],
    ]:
        return (
            KeyFuncDict._unreduce,
            (
                self.keyfunc,
                dict(self),
                collection_adapter(self),
            ),
        )

    @util.preload_module("sqlalchemy.orm.attributes")
    def _raise_for_unpopulated(
        self,
        value: _KT,
        initiator: Union[AttributeEventToken, Literal[None, False]] = None,
        *,
        warn_only: bool,
    ) -> None:
        mapper = base.instance_state(value).mapper

        attributes = util.preloaded.orm_attributes

        if not isinstance(initiator, attributes.AttributeEventToken):
            relationship = "unknown relationship"
        elif initiator.key in mapper.attrs:
            relationship = f"{mapper.attrs[initiator.key]}"
        else:
            relationship = initiator.key

        if warn_only:
            util.warn(
                f"Attribute keyed dictionary value for "
                f"attribute '{relationship}' was None; this will raise "
                "in a future release. "
                f"To skip this assignment entirely, "
                f'Set the "ignore_unpopulated_attribute=True" '
                f"parameter on the mapped collection factory."
            )
        else:
            raise sa_exc.InvalidRequestError(
                "In event triggered from population of "
                f"attribute '{relationship}' "
                "(potentially from a backref), "
                f"can't populate value in KeyFuncDict; "
                "dictionary key "
                f"derived from {base.instance_str(value)} is not "
                f"populated. Ensure appropriate state is set up on "
                f"the {base.instance_str(value)} object "
                f"before assigning to the {relationship} attribute. "
                f"To skip this assignment entirely, "
                f'Set the "ignore_unpopulated_attribute=True" '
                f"parameter on the mapped collection factory."
            )

    @collection.appender  # type: ignore[misc]
    @collection.internally_instrumented  # type: ignore[misc]
    def set(
        self,
        value: _KT,
        _sa_initiator: Union[AttributeEventToken, Literal[None, False]] = None,
    ) -> None:
        """Add an item by value, consulting the keyfunc for the key."""

        key = self.keyfunc(value)

        if key is base.NO_VALUE:
            if not self.ignore_unpopulated_attribute:
                self._raise_for_unpopulated(
                    value, _sa_initiator, warn_only=False
                )
            else:
                return
        elif key is Missing:
            if not self.ignore_unpopulated_attribute:
                self._raise_for_unpopulated(
                    value, _sa_initiator, warn_only=True
                )
                key = None
            else:
                return

        self.__setitem__(key, value, _sa_initiator)  # type: ignore[call-arg]

    @collection.remover  # type: ignore[misc]
    @collection.internally_instrumented  # type: ignore[misc]
    def remove(
        self,
        value: _KT,
        _sa_initiator: Union[AttributeEventToken, Literal[None, False]] = None,
    ) -> None:
        """Remove an item by value, consulting the keyfunc for the key."""

        key = self.keyfunc(value)

        if key is base.NO_VALUE:
            if not self.ignore_unpopulated_attribute:
                self._raise_for_unpopulated(
                    value, _sa_initiator, warn_only=False
                )
            return
        elif key is Missing:
            if not self.ignore_unpopulated_attribute:
                self._raise_for_unpopulated(
                    value, _sa_initiator, warn_only=True
                )
                key = None
            else:
                return

        # Let self[key] raise if key is not in this collection
        # testlib.pragma exempt:__ne__
        if self[key] != value:
            raise sa_exc.InvalidRequestError(
                "Can not remove '%s': collection holds '%s' for key '%s'. "
                "Possible cause: is the KeyFuncDict key function "
                "based on mutable properties or properties that only obtain "
                "values after flush?" % (value, self[key], key)
            )
        self.__delitem__(key, _sa_initiator)  # type: ignore[call-arg]


def _mapped_collection_cls(
    keyfunc: Callable[[Any], Any], ignore_unpopulated_attribute: bool
) -> Type[KeyFuncDict[_KT, _KT]]:
    class _MKeyfuncMapped(KeyFuncDict[_KT, _KT]):
        def __init__(self, *dict_args: Any) -> None:
            super().__init__(
                keyfunc,
                *dict_args,
                ignore_unpopulated_attribute=ignore_unpopulated_attribute,
            )

    return _MKeyfuncMapped


MappedCollection = KeyFuncDict
"""A synonym for :class:`.KeyFuncDict`.

.. versionchanged:: 2.0 Renamed :class:`.MappedCollection` to
   :class:`.KeyFuncDict`.

"""

mapped_collection = keyfunc_mapping
"""A synonym for :func:`_orm.keyfunc_mapping`.

.. versionchanged:: 2.0 Renamed :data:`.mapped_collection` to
   :func:`_orm.keyfunc_mapping`

"""

attribute_mapped_collection = attribute_keyed_dict
"""A synonym for :func:`_orm.attribute_keyed_dict`.

.. versionchanged:: 2.0 Renamed :data:`.attribute_mapped_collection` to
   :func:`_orm.attribute_keyed_dict`

"""

column_mapped_collection = column_keyed_dict
"""A synonym for :func:`_orm.column_keyed_dict.

.. versionchanged:: 2.0 Renamed :func:`.column_mapped_collection` to
   :func:`_orm.column_keyed_dict`

"""

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pycparser\c_lexer.py ===
#------------------------------------------------------------------------------
# pycparser: c_lexer.py
#
# CLexer class: lexer for the C language
#
# Eli Bendersky [https://eli.thegreenplace.net/]
# License: BSD
#------------------------------------------------------------------------------
import re

from .ply import lex
from .ply.lex import TOKEN


class CLexer(object):
    """ A lexer for the C language. After building it, set the
        input text with input(), and call token() to get new
        tokens.

        The public attribute filename can be set to an initial
        filename, but the lexer will update it upon #line
        directives.
    """
    def __init__(self, error_func, on_lbrace_func, on_rbrace_func,
                 type_lookup_func):
        """ Create a new Lexer.

            error_func:
                An error function. Will be called with an error
                message, line and column as arguments, in case of
                an error during lexing.

            on_lbrace_func, on_rbrace_func:
                Called when an LBRACE or RBRACE is encountered
                (likely to push/pop type_lookup_func's scope)

            type_lookup_func:
                A type lookup function. Given a string, it must
                return True IFF this string is a name of a type
                that was defined with a typedef earlier.
        """
        self.error_func = error_func
        self.on_lbrace_func = on_lbrace_func
        self.on_rbrace_func = on_rbrace_func
        self.type_lookup_func = type_lookup_func
        self.filename = ''

        # Keeps track of the last token returned from self.token()
        self.last_token = None

        # Allow either "# line" or "# <num>" to support GCC's
        # cpp output
        #
        self.line_pattern = re.compile(r'([ \t]*line\W)|([ \t]*\d+)')
        self.pragma_pattern = re.compile(r'[ \t]*pragma\W')

    def build(self, **kwargs):
        """ Builds the lexer from the specification. Must be
            called after the lexer object is created.

            This method exists separately, because the PLY
            manual warns against calling lex.lex inside
            __init__
        """
        self.lexer = lex.lex(object=self, **kwargs)

    def reset_lineno(self):
        """ Resets the internal line number counter of the lexer.
        """
        self.lexer.lineno = 1

    def input(self, text):
        self.lexer.input(text)

    def token(self):
        self.last_token = self.lexer.token()
        return self.last_token

    def find_tok_column(self, token):
        """ Find the column of the token in its line.
        """
        last_cr = self.lexer.lexdata.rfind('\n', 0, token.lexpos)
        return token.lexpos - last_cr

    ######################--   PRIVATE   --######################

    ##
    ## Internal auxiliary methods
    ##
    def _error(self, msg, token):
        location = self._make_tok_location(token)
        self.error_func(msg, location[0], location[1])
        self.lexer.skip(1)

    def _make_tok_location(self, token):
        return (token.lineno, self.find_tok_column(token))

    ##
    ## Reserved keywords
    ##
    keywords = (
        'AUTO', 'BREAK', 'CASE', 'CHAR', 'CONST',
        'CONTINUE', 'DEFAULT', 'DO', 'DOUBLE', 'ELSE', 'ENUM', 'EXTERN',
        'FLOAT', 'FOR', 'GOTO', 'IF', 'INLINE', 'INT', 'LONG',
        'REGISTER', 'OFFSETOF',
        'RESTRICT', 'RETURN', 'SHORT', 'SIGNED', 'SIZEOF', 'STATIC', 'STRUCT',
        'SWITCH', 'TYPEDEF', 'UNION', 'UNSIGNED', 'VOID',
        'VOLATILE', 'WHILE', '__INT128',
    )

    keywords_new = (
        '_BOOL', '_COMPLEX',
        '_NORETURN', '_THREAD_LOCAL', '_STATIC_ASSERT',
        '_ATOMIC', '_ALIGNOF', '_ALIGNAS',
        '_PRAGMA',
        )

    keyword_map = {}

    for keyword in keywords:
        keyword_map[keyword.lower()] = keyword

    for keyword in keywords_new:
        keyword_map[keyword[:2].upper() + keyword[2:].lower()] = keyword

    ##
    ## All the tokens recognized by the lexer
    ##
    tokens = keywords + keywords_new + (
        # Identifiers
        'ID',

        # Type identifiers (identifiers previously defined as
        # types with typedef)
        'TYPEID',

        # constants
        'INT_CONST_DEC', 'INT_CONST_OCT', 'INT_CONST_HEX', 'INT_CONST_BIN', 'INT_CONST_CHAR',
        'FLOAT_CONST', 'HEX_FLOAT_CONST',
        'CHAR_CONST',
        'WCHAR_CONST',
        'U8CHAR_CONST',
        'U16CHAR_CONST',
        'U32CHAR_CONST',

        # String literals
        'STRING_LITERAL',
        'WSTRING_LITERAL',
        'U8STRING_LITERAL',
        'U16STRING_LITERAL',
        'U32STRING_LITERAL',

        # Operators
        'PLUS', 'MINUS', 'TIMES', 'DIVIDE', 'MOD',
        'OR', 'AND', 'NOT', 'XOR', 'LSHIFT', 'RSHIFT',
        'LOR', 'LAND', 'LNOT',
        'LT', 'LE', 'GT', 'GE', 'EQ', 'NE',

        # Assignment
        'EQUALS', 'TIMESEQUAL', 'DIVEQUAL', 'MODEQUAL',
        'PLUSEQUAL', 'MINUSEQUAL',
        'LSHIFTEQUAL','RSHIFTEQUAL', 'ANDEQUAL', 'XOREQUAL',
        'OREQUAL',

        # Increment/decrement
        'PLUSPLUS', 'MINUSMINUS',

        # Structure dereference (->)
        'ARROW',

        # Conditional operator (?)
        'CONDOP',

        # Delimiters
        'LPAREN', 'RPAREN',         # ( )
        'LBRACKET', 'RBRACKET',     # [ ]
        'LBRACE', 'RBRACE',         # { }
        'COMMA', 'PERIOD',          # . ,
        'SEMI', 'COLON',            # ; :

        # Ellipsis (...)
        'ELLIPSIS',

        # pre-processor
        'PPHASH',       # '#'
        'PPPRAGMA',     # 'pragma'
        'PPPRAGMASTR',
    )

    ##
    ## Regexes for use in tokens
    ##
    ##

    # valid C identifiers (K&R2: A.2.3), plus '$' (supported by some compilers)
    identifier = r'[a-zA-Z_$][0-9a-zA-Z_$]*'

    hex_prefix = '0[xX]'
    hex_digits = '[0-9a-fA-F]+'
    bin_prefix = '0[bB]'
    bin_digits = '[01]+'

    # integer constants (K&R2: A.2.5.1)
    integer_suffix_opt = r'(([uU]ll)|([uU]LL)|(ll[uU]?)|(LL[uU]?)|([uU][lL])|([lL][uU]?)|[uU])?'
    decimal_constant = '(0'+integer_suffix_opt+')|([1-9][0-9]*'+integer_suffix_opt+')'
    octal_constant = '0[0-7]*'+integer_suffix_opt
    hex_constant = hex_prefix+hex_digits+integer_suffix_opt
    bin_constant = bin_prefix+bin_digits+integer_suffix_opt

    bad_octal_constant = '0[0-7]*[89]'

    # character constants (K&R2: A.2.5.2)
    # Note: a-zA-Z and '.-~^_!=&;,' are allowed as escape chars to support #line
    # directives with Windows paths as filenames (..\..\dir\file)
    # For the same reason, decimal_escape allows all digit sequences. We want to
    # parse all correct code, even if it means to sometimes parse incorrect
    # code.
    #
    # The original regexes were taken verbatim from the C syntax definition,
    # and were later modified to avoid worst-case exponential running time.
    #
    #   simple_escape = r"""([a-zA-Z._~!=&\^\-\\?'"])"""
    #   decimal_escape = r"""(\d+)"""
    #   hex_escape = r"""(x[0-9a-fA-F]+)"""
    #   bad_escape = r"""([\\][^a-zA-Z._~^!=&\^\-\\?'"x0-7])"""
    #
    # The following modifications were made to avoid the ambiguity that allowed backtracking:
    # (https://github.com/eliben/pycparser/issues/61)
    #
    # - \x was removed from simple_escape, unless it was not followed by a hex digit, to avoid ambiguity with hex_escape.
    # - hex_escape allows one or more hex characters, but requires that the next character(if any) is not hex
    # - decimal_escape allows one or more decimal characters, but requires that the next character(if any) is not a decimal
    # - bad_escape does not allow any decimals (8-9), to avoid conflicting with the permissive decimal_escape.
    #
    # Without this change, python's `re` module would recursively try parsing each ambiguous escape sequence in multiple ways.
    # e.g. `\123` could be parsed as `\1`+`23`, `\12`+`3`, and `\123`.

    simple_escape = r"""([a-wyzA-Z._~!=&\^\-\\?'"]|x(?![0-9a-fA-F]))"""
    decimal_escape = r"""(\d+)(?!\d)"""
    hex_escape = r"""(x[0-9a-fA-F]+)(?![0-9a-fA-F])"""
    bad_escape = r"""([\\][^a-zA-Z._~^!=&\^\-\\?'"x0-9])"""

    escape_sequence = r"""(\\("""+simple_escape+'|'+decimal_escape+'|'+hex_escape+'))'

    # This complicated regex with lookahead might be slow for strings, so because all of the valid escapes (including \x) allowed
    # 0 or more non-escaped characters after the first character, simple_escape+decimal_escape+hex_escape got simplified to

    escape_sequence_start_in_string = r"""(\\[0-9a-zA-Z._~!=&\^\-\\?'"])"""

    cconst_char = r"""([^'\\\n]|"""+escape_sequence+')'
    char_const = "'"+cconst_char+"'"
    wchar_const = 'L'+char_const
    u8char_const = 'u8'+char_const
    u16char_const = 'u'+char_const
    u32char_const = 'U'+char_const
    multicharacter_constant = "'"+cconst_char+"{2,4}'"
    unmatched_quote = "('"+cconst_char+"*\\n)|('"+cconst_char+"*$)"
    bad_char_const = r"""('"""+cconst_char+"""[^'\n]+')|('')|('"""+bad_escape+r"""[^'\n]*')"""

    # string literals (K&R2: A.2.6)
    string_char = r"""([^"\\\n]|"""+escape_sequence_start_in_string+')'
    string_literal = '"'+string_char+'*"'
    wstring_literal = 'L'+string_literal
    u8string_literal = 'u8'+string_literal
    u16string_literal = 'u'+string_literal
    u32string_literal = 'U'+string_literal
    bad_string_literal = '"'+string_char+'*'+bad_escape+string_char+'*"'

    # floating constants (K&R2: A.2.5.3)
    exponent_part = r"""([eE][-+]?[0-9]+)"""
    fractional_constant = r"""([0-9]*\.[0-9]+)|([0-9]+\.)"""
    floating_constant = '(((('+fractional_constant+')'+exponent_part+'?)|([0-9]+'+exponent_part+'))[FfLl]?)'
    binary_exponent_part = r'''([pP][+-]?[0-9]+)'''
    hex_fractional_constant = '((('+hex_digits+r""")?\."""+hex_digits+')|('+hex_digits+r"""\.))"""
    hex_floating_constant = '('+hex_prefix+'('+hex_digits+'|'+hex_fractional_constant+')'+binary_exponent_part+'[FfLl]?)'

    ##
    ## Lexer states: used for preprocessor \n-terminated directives
    ##
    states = (
        # ppline: preprocessor line directives
        #
        ('ppline', 'exclusive'),

        # pppragma: pragma
        #
        ('pppragma', 'exclusive'),
    )

    def t_PPHASH(self, t):
        r'[ \t]*\#'
        if self.line_pattern.match(t.lexer.lexdata, pos=t.lexer.lexpos):
            t.lexer.begin('ppline')
            self.pp_line = self.pp_filename = None
        elif self.pragma_pattern.match(t.lexer.lexdata, pos=t.lexer.lexpos):
            t.lexer.begin('pppragma')
        else:
            t.type = 'PPHASH'
            return t

    ##
    ## Rules for the ppline state
    ##
    @TOKEN(string_literal)
    def t_ppline_FILENAME(self, t):
        if self.pp_line is None:
            self._error('filename before line number in #line', t)
        else:
            self.pp_filename = t.value.lstrip('"').rstrip('"')

    @TOKEN(decimal_constant)
    def t_ppline_LINE_NUMBER(self, t):
        if self.pp_line is None:
            self.pp_line = t.value
        else:
            # Ignore: GCC's cpp sometimes inserts a numeric flag
            # after the file name
            pass

    def t_ppline_NEWLINE(self, t):
        r'\n'
        if self.pp_line is None:
            self._error('line number missing in #line', t)
        else:
            self.lexer.lineno = int(self.pp_line)

            if self.pp_filename is not None:
                self.filename = self.pp_filename

        t.lexer.begin('INITIAL')

    def t_ppline_PPLINE(self, t):
        r'line'
        pass

    t_ppline_ignore = ' \t'

    def t_ppline_error(self, t):
        self._error('invalid #line directive', t)

    ##
    ## Rules for the pppragma state
    ##
    def t_pppragma_NEWLINE(self, t):
        r'\n'
        t.lexer.lineno += 1
        t.lexer.begin('INITIAL')

    def t_pppragma_PPPRAGMA(self, t):
        r'pragma'
        return t

    t_pppragma_ignore = ' \t'

    def t_pppragma_STR(self, t):
        '.+'
        t.type = 'PPPRAGMASTR'
        return t

    def t_pppragma_error(self, t):
        self._error('invalid #pragma directive', t)

    ##
    ## Rules for the normal state
    ##
    t_ignore = ' \t'

    # Newlines
    def t_NEWLINE(self, t):
        r'\n+'
        t.lexer.lineno += t.value.count("\n")

    # Operators
    t_PLUS              = r'\+'
    t_MINUS             = r'-'
    t_TIMES             = r'\*'
    t_DIVIDE            = r'/'
    t_MOD               = r'%'
    t_OR                = r'\|'
    t_AND               = r'&'
    t_NOT               = r'~'
    t_XOR               = r'\^'
    t_LSHIFT            = r'<<'
    t_RSHIFT            = r'>>'
    t_LOR               = r'\|\|'
    t_LAND              = r'&&'
    t_LNOT              = r'!'
    t_LT                = r'<'
    t_GT                = r'>'
    t_LE                = r'<='
    t_GE                = r'>='
    t_EQ                = r'=='
    t_NE                = r'!='

    # Assignment operators
    t_EQUALS            = r'='
    t_TIMESEQUAL        = r'\*='
    t_DIVEQUAL          = r'/='
    t_MODEQUAL          = r'%='
    t_PLUSEQUAL         = r'\+='
    t_MINUSEQUAL        = r'-='
    t_LSHIFTEQUAL       = r'<<='
    t_RSHIFTEQUAL       = r'>>='
    t_ANDEQUAL          = r'&='
    t_OREQUAL           = r'\|='
    t_XOREQUAL          = r'\^='

    # Increment/decrement
    t_PLUSPLUS          = r'\+\+'
    t_MINUSMINUS        = r'--'

    # ->
    t_ARROW             = r'->'

    # ?
    t_CONDOP            = r'\?'

    # Delimiters
    t_LPAREN            = r'\('
    t_RPAREN            = r'\)'
    t_LBRACKET          = r'\['
    t_RBRACKET          = r'\]'
    t_COMMA             = r','
    t_PERIOD            = r'\.'
    t_SEMI              = r';'
    t_COLON             = r':'
    t_ELLIPSIS          = r'\.\.\.'

    # Scope delimiters
    # To see why on_lbrace_func is needed, consider:
    #   typedef char TT;
    #   void foo(int TT) { TT = 10; }
    #   TT x = 5;
    # Outside the function, TT is a typedef, but inside (starting and ending
    # with the braces) it's a parameter.  The trouble begins with yacc's
    # lookahead token.  If we open a new scope in brace_open, then TT has
    # already been read and incorrectly interpreted as TYPEID.  So, we need
    # to open and close scopes from within the lexer.
    # Similar for the TT immediately outside the end of the function.
    #
    @TOKEN(r'\{')
    def t_LBRACE(self, t):
        self.on_lbrace_func()
        return t
    @TOKEN(r'\}')
    def t_RBRACE(self, t):
        self.on_rbrace_func()
        return t

    t_STRING_LITERAL = string_literal

    # The following floating and integer constants are defined as
    # functions to impose a strict order (otherwise, decimal
    # is placed before the others because its regex is longer,
    # and this is bad)
    #
    @TOKEN(floating_constant)
    def t_FLOAT_CONST(self, t):
        return t

    @TOKEN(hex_floating_constant)
    def t_HEX_FLOAT_CONST(self, t):
        return t

    @TOKEN(hex_constant)
    def t_INT_CONST_HEX(self, t):
        return t

    @TOKEN(bin_constant)
    def t_INT_CONST_BIN(self, t):
        return t

    @TOKEN(bad_octal_constant)
    def t_BAD_CONST_OCT(self, t):
        msg = "Invalid octal constant"
        self._error(msg, t)

    @TOKEN(octal_constant)
    def t_INT_CONST_OCT(self, t):
        return t

    @TOKEN(decimal_constant)
    def t_INT_CONST_DEC(self, t):
        return t

    # Must come before bad_char_const, to prevent it from
    # catching valid char constants as invalid
    #
    @TOKEN(multicharacter_constant)
    def t_INT_CONST_CHAR(self, t):
        return t

    @TOKEN(char_const)
    def t_CHAR_CONST(self, t):
        return t

    @TOKEN(wchar_const)
    def t_WCHAR_CONST(self, t):
        return t

    @TOKEN(u8char_const)
    def t_U8CHAR_CONST(self, t):
        return t

    @TOKEN(u16char_const)
    def t_U16CHAR_CONST(self, t):
        return t

    @TOKEN(u32char_const)
    def t_U32CHAR_CONST(self, t):
        return t

    @TOKEN(unmatched_quote)
    def t_UNMATCHED_QUOTE(self, t):
        msg = "Unmatched '"
        self._error(msg, t)

    @TOKEN(bad_char_const)
    def t_BAD_CHAR_CONST(self, t):
        msg = "Invalid char constant %s" % t.value
        self._error(msg, t)

    @TOKEN(wstring_literal)
    def t_WSTRING_LITERAL(self, t):
        return t

    @TOKEN(u8string_literal)
    def t_U8STRING_LITERAL(self, t):
        return t

    @TOKEN(u16string_literal)
    def t_U16STRING_LITERAL(self, t):
        return t

    @TOKEN(u32string_literal)
    def t_U32STRING_LITERAL(self, t):
        return t

    # unmatched string literals are caught by the preprocessor

    @TOKEN(bad_string_literal)
    def t_BAD_STRING_LITERAL(self, t):
        msg = "String contains invalid escape code"
        self._error(msg, t)

    @TOKEN(identifier)
    def t_ID(self, t):
        t.type = self.keyword_map.get(t.value, "ID")
        if t.type == 'ID' and self.type_lookup_func(t.value):
            t.type = "TYPEID"
        return t

    def t_error(self, t):
        msg = 'Illegal character %s' % repr(t.value[0])
        self._error(msg, t)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v135\preload.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Preload (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page


class RuleSetId(str):
    '''
    Unique id
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> RuleSetId:
        return cls(json)

    def __repr__(self):
        return 'RuleSetId({})'.format(super().__repr__())


@dataclass
class RuleSet:
    '''
    Corresponds to SpeculationRuleSet
    '''
    id_: RuleSetId

    #: Identifies a document which the rule set is associated with.
    loader_id: network.LoaderId

    #: Source text of JSON representing the rule set. If it comes from
    #: ``script`` tag, it is the textContent of the node. Note that it is
    #: a JSON for valid case.
    #: 
    #: See also:
    #: - https://wicg.github.io/nav-speculation/speculation-rules.html
    #: - https://github.com/WICG/nav-speculation/blob/main/triggers.md
    source_text: str

    #: A speculation rule set is either added through an inline
    #: ``script`` tag or through an external resource via the
    #: 'Speculation-Rules' HTTP header. For the first case, we include
    #: the BackendNodeId of the relevant ``script`` tag. For the second
    #: case, we include the external URL where the rule set was loaded
    #: from, and also RequestId if Network domain is enabled.
    #: 
    #: See also:
    #: - https://wicg.github.io/nav-speculation/speculation-rules.html#speculation-rules-script
    #: - https://wicg.github.io/nav-speculation/speculation-rules.html#speculation-rules-header
    backend_node_id: typing.Optional[dom.BackendNodeId] = None

    url: typing.Optional[str] = None

    request_id: typing.Optional[network.RequestId] = None

    #: Error information
    #: ``errorMessage`` is null iff ``errorType`` is null.
    error_type: typing.Optional[RuleSetErrorType] = None

    #: TODO(https://crbug.com/1425354): Replace this property with structured error.
    error_message: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['id'] = self.id_.to_json()
        json['loaderId'] = self.loader_id.to_json()
        json['sourceText'] = self.source_text
        if self.backend_node_id is not None:
            json['backendNodeId'] = self.backend_node_id.to_json()
        if self.url is not None:
            json['url'] = self.url
        if self.request_id is not None:
            json['requestId'] = self.request_id.to_json()
        if self.error_type is not None:
            json['errorType'] = self.error_type.to_json()
        if self.error_message is not None:
            json['errorMessage'] = self.error_message
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            id_=RuleSetId.from_json(json['id']),
            loader_id=network.LoaderId.from_json(json['loaderId']),
            source_text=str(json['sourceText']),
            backend_node_id=dom.BackendNodeId.from_json(json['backendNodeId']) if 'backendNodeId' in json else None,
            url=str(json['url']) if 'url' in json else None,
            request_id=network.RequestId.from_json(json['requestId']) if 'requestId' in json else None,
            error_type=RuleSetErrorType.from_json(json['errorType']) if 'errorType' in json else None,
            error_message=str(json['errorMessage']) if 'errorMessage' in json else None,
        )


class RuleSetErrorType(enum.Enum):
    SOURCE_IS_NOT_JSON_OBJECT = "SourceIsNotJsonObject"
    INVALID_RULES_SKIPPED = "InvalidRulesSkipped"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class SpeculationAction(enum.Enum):
    '''
    The type of preloading attempted. It corresponds to
    mojom::SpeculationAction (although PrefetchWithSubresources is omitted as it
    isn't being used by clients).
    '''
    PREFETCH = "Prefetch"
    PRERENDER = "Prerender"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class SpeculationTargetHint(enum.Enum):
    '''
    Corresponds to mojom::SpeculationTargetHint.
    See https://github.com/WICG/nav-speculation/blob/main/triggers.md#window-name-targeting-hints
    '''
    BLANK = "Blank"
    SELF = "Self"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class PreloadingAttemptKey:
    '''
    A key that identifies a preloading attempt.

    The url used is the url specified by the trigger (i.e. the initial URL), and
    not the final url that is navigated to. For example, prerendering allows
    same-origin main frame navigations during the attempt, but the attempt is
    still keyed with the initial URL.
    '''
    loader_id: network.LoaderId

    action: SpeculationAction

    url: str

    target_hint: typing.Optional[SpeculationTargetHint] = None

    def to_json(self):
        json = dict()
        json['loaderId'] = self.loader_id.to_json()
        json['action'] = self.action.to_json()
        json['url'] = self.url
        if self.target_hint is not None:
            json['targetHint'] = self.target_hint.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            loader_id=network.LoaderId.from_json(json['loaderId']),
            action=SpeculationAction.from_json(json['action']),
            url=str(json['url']),
            target_hint=SpeculationTargetHint.from_json(json['targetHint']) if 'targetHint' in json else None,
        )


@dataclass
class PreloadingAttemptSource:
    '''
    Lists sources for a preloading attempt, specifically the ids of rule sets
    that had a speculation rule that triggered the attempt, and the
    BackendNodeIds of <a href> or <area href> elements that triggered the
    attempt (in the case of attempts triggered by a document rule). It is
    possible for multiple rule sets and links to trigger a single attempt.
    '''
    key: PreloadingAttemptKey

    rule_set_ids: typing.List[RuleSetId]

    node_ids: typing.List[dom.BackendNodeId]

    def to_json(self):
        json = dict()
        json['key'] = self.key.to_json()
        json['ruleSetIds'] = [i.to_json() for i in self.rule_set_ids]
        json['nodeIds'] = [i.to_json() for i in self.node_ids]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            key=PreloadingAttemptKey.from_json(json['key']),
            rule_set_ids=[RuleSetId.from_json(i) for i in json['ruleSetIds']],
            node_ids=[dom.BackendNodeId.from_json(i) for i in json['nodeIds']],
        )


class PreloadPipelineId(str):
    '''
    Chrome manages different types of preloads together using a
    concept of preloading pipeline. For example, if a site uses a
    SpeculationRules for prerender, Chrome first starts a prefetch and
    then upgrades it to prerender.

    CDP events for them are emitted separately but they share
    ``PreloadPipelineId``.
    '''
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> PreloadPipelineId:
        return cls(json)

    def __repr__(self):
        return 'PreloadPipelineId({})'.format(super().__repr__())


class PrerenderFinalStatus(enum.Enum):
    '''
    List of FinalStatus reasons for Prerender2.
    '''
    ACTIVATED = "Activated"
    DESTROYED = "Destroyed"
    LOW_END_DEVICE = "LowEndDevice"
    INVALID_SCHEME_REDIRECT = "InvalidSchemeRedirect"
    INVALID_SCHEME_NAVIGATION = "InvalidSchemeNavigation"
    NAVIGATION_REQUEST_BLOCKED_BY_CSP = "NavigationRequestBlockedByCsp"
    MAIN_FRAME_NAVIGATION = "MainFrameNavigation"
    MOJO_BINDER_POLICY = "MojoBinderPolicy"
    RENDERER_PROCESS_CRASHED = "RendererProcessCrashed"
    RENDERER_PROCESS_KILLED = "RendererProcessKilled"
    DOWNLOAD = "Download"
    TRIGGER_DESTROYED = "TriggerDestroyed"
    NAVIGATION_NOT_COMMITTED = "NavigationNotCommitted"
    NAVIGATION_BAD_HTTP_STATUS = "NavigationBadHttpStatus"
    CLIENT_CERT_REQUESTED = "ClientCertRequested"
    NAVIGATION_REQUEST_NETWORK_ERROR = "NavigationRequestNetworkError"
    CANCEL_ALL_HOSTS_FOR_TESTING = "CancelAllHostsForTesting"
    DID_FAIL_LOAD = "DidFailLoad"
    STOP = "Stop"
    SSL_CERTIFICATE_ERROR = "SslCertificateError"
    LOGIN_AUTH_REQUESTED = "LoginAuthRequested"
    UA_CHANGE_REQUIRES_RELOAD = "UaChangeRequiresReload"
    BLOCKED_BY_CLIENT = "BlockedByClient"
    AUDIO_OUTPUT_DEVICE_REQUESTED = "AudioOutputDeviceRequested"
    MIXED_CONTENT = "MixedContent"
    TRIGGER_BACKGROUNDED = "TriggerBackgrounded"
    MEMORY_LIMIT_EXCEEDED = "MemoryLimitExceeded"
    DATA_SAVER_ENABLED = "DataSaverEnabled"
    TRIGGER_URL_HAS_EFFECTIVE_URL = "TriggerUrlHasEffectiveUrl"
    ACTIVATED_BEFORE_STARTED = "ActivatedBeforeStarted"
    INACTIVE_PAGE_RESTRICTION = "InactivePageRestriction"
    START_FAILED = "StartFailed"
    TIMEOUT_BACKGROUNDED = "TimeoutBackgrounded"
    CROSS_SITE_REDIRECT_IN_INITIAL_NAVIGATION = "CrossSiteRedirectInInitialNavigation"
    CROSS_SITE_NAVIGATION_IN_INITIAL_NAVIGATION = "CrossSiteNavigationInInitialNavigation"
    SAME_SITE_CROSS_ORIGIN_REDIRECT_NOT_OPT_IN_IN_INITIAL_NAVIGATION = "SameSiteCrossOriginRedirectNotOptInInInitialNavigation"
    SAME_SITE_CROSS_ORIGIN_NAVIGATION_NOT_OPT_IN_IN_INITIAL_NAVIGATION = "SameSiteCrossOriginNavigationNotOptInInInitialNavigation"
    ACTIVATION_NAVIGATION_PARAMETER_MISMATCH = "ActivationNavigationParameterMismatch"
    ACTIVATED_IN_BACKGROUND = "ActivatedInBackground"
    EMBEDDER_HOST_DISALLOWED = "EmbedderHostDisallowed"
    ACTIVATION_NAVIGATION_DESTROYED_BEFORE_SUCCESS = "ActivationNavigationDestroyedBeforeSuccess"
    TAB_CLOSED_BY_USER_GESTURE = "TabClosedByUserGesture"
    TAB_CLOSED_WITHOUT_USER_GESTURE = "TabClosedWithoutUserGesture"
    PRIMARY_MAIN_FRAME_RENDERER_PROCESS_CRASHED = "PrimaryMainFrameRendererProcessCrashed"
    PRIMARY_MAIN_FRAME_RENDERER_PROCESS_KILLED = "PrimaryMainFrameRendererProcessKilled"
    ACTIVATION_FRAME_POLICY_NOT_COMPATIBLE = "ActivationFramePolicyNotCompatible"
    PRELOADING_DISABLED = "PreloadingDisabled"
    BATTERY_SAVER_ENABLED = "BatterySaverEnabled"
    ACTIVATED_DURING_MAIN_FRAME_NAVIGATION = "ActivatedDuringMainFrameNavigation"
    PRELOADING_UNSUPPORTED_BY_WEB_CONTENTS = "PreloadingUnsupportedByWebContents"
    CROSS_SITE_REDIRECT_IN_MAIN_FRAME_NAVIGATION = "CrossSiteRedirectInMainFrameNavigation"
    CROSS_SITE_NAVIGATION_IN_MAIN_FRAME_NAVIGATION = "CrossSiteNavigationInMainFrameNavigation"
    SAME_SITE_CROSS_ORIGIN_REDIRECT_NOT_OPT_IN_IN_MAIN_FRAME_NAVIGATION = "SameSiteCrossOriginRedirectNotOptInInMainFrameNavigation"
    SAME_SITE_CROSS_ORIGIN_NAVIGATION_NOT_OPT_IN_IN_MAIN_FRAME_NAVIGATION = "SameSiteCrossOriginNavigationNotOptInInMainFrameNavigation"
    MEMORY_PRESSURE_ON_TRIGGER = "MemoryPressureOnTrigger"
    MEMORY_PRESSURE_AFTER_TRIGGERED = "MemoryPressureAfterTriggered"
    PRERENDERING_DISABLED_BY_DEV_TOOLS = "PrerenderingDisabledByDevTools"
    SPECULATION_RULE_REMOVED = "SpeculationRuleRemoved"
    ACTIVATED_WITH_AUXILIARY_BROWSING_CONTEXTS = "ActivatedWithAuxiliaryBrowsingContexts"
    MAX_NUM_OF_RUNNING_EAGER_PRERENDERS_EXCEEDED = "MaxNumOfRunningEagerPrerendersExceeded"
    MAX_NUM_OF_RUNNING_NON_EAGER_PRERENDERS_EXCEEDED = "MaxNumOfRunningNonEagerPrerendersExceeded"
    MAX_NUM_OF_RUNNING_EMBEDDER_PRERENDERS_EXCEEDED = "MaxNumOfRunningEmbedderPrerendersExceeded"
    PRERENDERING_URL_HAS_EFFECTIVE_URL = "PrerenderingUrlHasEffectiveUrl"
    REDIRECTED_PRERENDERING_URL_HAS_EFFECTIVE_URL = "RedirectedPrerenderingUrlHasEffectiveUrl"
    ACTIVATION_URL_HAS_EFFECTIVE_URL = "ActivationUrlHasEffectiveUrl"
    JAVA_SCRIPT_INTERFACE_ADDED = "JavaScriptInterfaceAdded"
    JAVA_SCRIPT_INTERFACE_REMOVED = "JavaScriptInterfaceRemoved"
    ALL_PRERENDERING_CANCELED = "AllPrerenderingCanceled"
    WINDOW_CLOSED = "WindowClosed"
    SLOW_NETWORK = "SlowNetwork"
    OTHER_PRERENDERED_PAGE_ACTIVATED = "OtherPrerenderedPageActivated"
    V8_OPTIMIZER_DISABLED = "V8OptimizerDisabled"
    PRERENDER_FAILED_DURING_PREFETCH = "PrerenderFailedDuringPrefetch"
    BROWSING_DATA_REMOVED = "BrowsingDataRemoved"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class PreloadingStatus(enum.Enum):
    '''
    Preloading status values, see also PreloadingTriggeringOutcome. This
    status is shared by prefetchStatusUpdated and prerenderStatusUpdated.
    '''
    PENDING = "Pending"
    RUNNING = "Running"
    READY = "Ready"
    SUCCESS = "Success"
    FAILURE = "Failure"
    NOT_SUPPORTED = "NotSupported"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class PrefetchStatus(enum.Enum):
    '''
    TODO(https://crbug.com/1384419): revisit the list of PrefetchStatus and
    filter out the ones that aren't necessary to the developers.
    '''
    PREFETCH_ALLOWED = "PrefetchAllowed"
    PREFETCH_FAILED_INELIGIBLE_REDIRECT = "PrefetchFailedIneligibleRedirect"
    PREFETCH_FAILED_INVALID_REDIRECT = "PrefetchFailedInvalidRedirect"
    PREFETCH_FAILED_MIME_NOT_SUPPORTED = "PrefetchFailedMIMENotSupported"
    PREFETCH_FAILED_NET_ERROR = "PrefetchFailedNetError"
    PREFETCH_FAILED_NON2_XX = "PrefetchFailedNon2XX"
    PREFETCH_EVICTED_AFTER_CANDIDATE_REMOVED = "PrefetchEvictedAfterCandidateRemoved"
    PREFETCH_EVICTED_FOR_NEWER_PREFETCH = "PrefetchEvictedForNewerPrefetch"
    PREFETCH_HELDBACK = "PrefetchHeldback"
    PREFETCH_INELIGIBLE_RETRY_AFTER = "PrefetchIneligibleRetryAfter"
    PREFETCH_IS_PRIVACY_DECOY = "PrefetchIsPrivacyDecoy"
    PREFETCH_IS_STALE = "PrefetchIsStale"
    PREFETCH_NOT_ELIGIBLE_BROWSER_CONTEXT_OFF_THE_RECORD = "PrefetchNotEligibleBrowserContextOffTheRecord"
    PREFETCH_NOT_ELIGIBLE_DATA_SAVER_ENABLED = "PrefetchNotEligibleDataSaverEnabled"
    PREFETCH_NOT_ELIGIBLE_EXISTING_PROXY = "PrefetchNotEligibleExistingProxy"
    PREFETCH_NOT_ELIGIBLE_HOST_IS_NON_UNIQUE = "PrefetchNotEligibleHostIsNonUnique"
    PREFETCH_NOT_ELIGIBLE_NON_DEFAULT_STORAGE_PARTITION = "PrefetchNotEligibleNonDefaultStoragePartition"
    PREFETCH_NOT_ELIGIBLE_SAME_SITE_CROSS_ORIGIN_PREFETCH_REQUIRED_PROXY = "PrefetchNotEligibleSameSiteCrossOriginPrefetchRequiredProxy"
    PREFETCH_NOT_ELIGIBLE_SCHEME_IS_NOT_HTTPS = "PrefetchNotEligibleSchemeIsNotHttps"
    PREFETCH_NOT_ELIGIBLE_USER_HAS_COOKIES = "PrefetchNotEligibleUserHasCookies"
    PREFETCH_NOT_ELIGIBLE_USER_HAS_SERVICE_WORKER = "PrefetchNotEligibleUserHasServiceWorker"
    PREFETCH_NOT_ELIGIBLE_BATTERY_SAVER_ENABLED = "PrefetchNotEligibleBatterySaverEnabled"
    PREFETCH_NOT_ELIGIBLE_PRELOADING_DISABLED = "PrefetchNotEligiblePreloadingDisabled"
    PREFETCH_NOT_FINISHED_IN_TIME = "PrefetchNotFinishedInTime"
    PREFETCH_NOT_STARTED = "PrefetchNotStarted"
    PREFETCH_NOT_USED_COOKIES_CHANGED = "PrefetchNotUsedCookiesChanged"
    PREFETCH_PROXY_NOT_AVAILABLE = "PrefetchProxyNotAvailable"
    PREFETCH_RESPONSE_USED = "PrefetchResponseUsed"
    PREFETCH_SUCCESSFUL_BUT_NOT_USED = "PrefetchSuccessfulButNotUsed"
    PREFETCH_NOT_USED_PROBE_FAILED = "PrefetchNotUsedProbeFailed"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class PrerenderMismatchedHeaders:
    '''
    Information of headers to be displayed when the header mismatch occurred.
    '''
    header_name: str

    initial_value: typing.Optional[str] = None

    activation_value: typing.Optional[str] = None

    def to_json(self):
        json = dict()
        json['headerName'] = self.header_name
        if self.initial_value is not None:
            json['initialValue'] = self.initial_value
        if self.activation_value is not None:
            json['activationValue'] = self.activation_value
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            header_name=str(json['headerName']),
            initial_value=str(json['initialValue']) if 'initialValue' in json else None,
            activation_value=str(json['activationValue']) if 'activationValue' in json else None,
        )


def enable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:

    cmd_dict: T_JSON_DICT = {
        'method': 'Preload.enable',
    }
    json = yield cmd_dict


def disable() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:

    cmd_dict: T_JSON_DICT = {
        'method': 'Preload.disable',
    }
    json = yield cmd_dict


@event_class('Preload.ruleSetUpdated')
@dataclass
class RuleSetUpdated:
    '''
    Upsert. Currently, it is only emitted when a rule set added.
    '''
    rule_set: RuleSet

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> RuleSetUpdated:
        return cls(
            rule_set=RuleSet.from_json(json['ruleSet'])
        )


@event_class('Preload.ruleSetRemoved')
@dataclass
class RuleSetRemoved:
    id_: RuleSetId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> RuleSetRemoved:
        return cls(
            id_=RuleSetId.from_json(json['id'])
        )


@event_class('Preload.preloadEnabledStateUpdated')
@dataclass
class PreloadEnabledStateUpdated:
    '''
    Fired when a preload enabled state is updated.
    '''
    disabled_by_preference: bool
    disabled_by_data_saver: bool
    disabled_by_battery_saver: bool
    disabled_by_holdback_prefetch_speculation_rules: bool
    disabled_by_holdback_prerender_speculation_rules: bool

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> PreloadEnabledStateUpdated:
        return cls(
            disabled_by_preference=bool(json['disabledByPreference']),
            disabled_by_data_saver=bool(json['disabledByDataSaver']),
            disabled_by_battery_saver=bool(json['disabledByBatterySaver']),
            disabled_by_holdback_prefetch_speculation_rules=bool(json['disabledByHoldbackPrefetchSpeculationRules']),
            disabled_by_holdback_prerender_speculation_rules=bool(json['disabledByHoldbackPrerenderSpeculationRules'])
        )


@event_class('Preload.prefetchStatusUpdated')
@dataclass
class PrefetchStatusUpdated:
    '''
    Fired when a prefetch attempt is updated.
    '''
    key: PreloadingAttemptKey
    pipeline_id: PreloadPipelineId
    #: The frame id of the frame initiating prefetch.
    initiating_frame_id: page.FrameId
    prefetch_url: str
    status: PreloadingStatus
    prefetch_status: PrefetchStatus
    request_id: network.RequestId

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> PrefetchStatusUpdated:
        return cls(
            key=PreloadingAttemptKey.from_json(json['key']),
            pipeline_id=PreloadPipelineId.from_json(json['pipelineId']),
            initiating_frame_id=page.FrameId.from_json(json['initiatingFrameId']),
            prefetch_url=str(json['prefetchUrl']),
            status=PreloadingStatus.from_json(json['status']),
            prefetch_status=PrefetchStatus.from_json(json['prefetchStatus']),
            request_id=network.RequestId.from_json(json['requestId'])
        )


@event_class('Preload.prerenderStatusUpdated')
@dataclass
class PrerenderStatusUpdated:
    '''
    Fired when a prerender attempt is updated.
    '''
    key: PreloadingAttemptKey
    pipeline_id: PreloadPipelineId
    status: PreloadingStatus
    prerender_status: typing.Optional[PrerenderFinalStatus]
    #: This is used to give users more information about the name of Mojo interface
    #: that is incompatible with prerender and has caused the cancellation of the attempt.
    disallowed_mojo_interface: typing.Optional[str]
    mismatched_headers: typing.Optional[typing.List[PrerenderMismatchedHeaders]]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> PrerenderStatusUpdated:
        return cls(
            key=PreloadingAttemptKey.from_json(json['key']),
            pipeline_id=PreloadPipelineId.from_json(json['pipelineId']),
            status=PreloadingStatus.from_json(json['status']),
            prerender_status=PrerenderFinalStatus.from_json(json['prerenderStatus']) if 'prerenderStatus' in json else None,
            disallowed_mojo_interface=str(json['disallowedMojoInterface']) if 'disallowedMojoInterface' in json else None,
            mismatched_headers=[PrerenderMismatchedHeaders.from_json(i) for i in json['mismatchedHeaders']] if 'mismatchedHeaders' in json else None
        )


@event_class('Preload.preloadingAttemptSourcesUpdated')
@dataclass
class PreloadingAttemptSourcesUpdated:
    '''
    Send a list of sources for all preloading attempts in a document.
    '''
    loader_id: network.LoaderId
    preloading_attempt_sources: typing.List[PreloadingAttemptSource]

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> PreloadingAttemptSourcesUpdated:
        return cls(
            loader_id=network.LoaderId.from_json(json['loaderId']),
            preloading_attempt_sources=[PreloadingAttemptSource.from_json(i) for i in json['preloadingAttemptSources']]
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\indexers\utils.py ===
"""
Low-dependency indexing utilities.
"""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np

from pandas._libs import lib

from pandas.core.dtypes.common import (
    is_array_like,
    is_bool_dtype,
    is_integer,
    is_integer_dtype,
    is_list_like,
)
from pandas.core.dtypes.dtypes import ExtensionDtype
from pandas.core.dtypes.generic import (
    ABCIndex,
    ABCSeries,
)

if TYPE_CHECKING:
    from pandas._typing import AnyArrayLike

    from pandas.core.frame import DataFrame
    from pandas.core.indexes.base import Index

# -----------------------------------------------------------
# Indexer Identification


def is_valid_positional_slice(slc: slice) -> bool:
    """
    Check if a slice object can be interpreted as a positional indexer.

    Parameters
    ----------
    slc : slice

    Returns
    -------
    bool

    Notes
    -----
    A valid positional slice may also be interpreted as a label-based slice
    depending on the index being sliced.
    """
    return (
        lib.is_int_or_none(slc.start)
        and lib.is_int_or_none(slc.stop)
        and lib.is_int_or_none(slc.step)
    )


def is_list_like_indexer(key) -> bool:
    """
    Check if we have a list-like indexer that is *not* a NamedTuple.

    Parameters
    ----------
    key : object

    Returns
    -------
    bool
    """
    # allow a list_like, but exclude NamedTuples which can be indexers
    return is_list_like(key) and not (isinstance(key, tuple) and type(key) is not tuple)


def is_scalar_indexer(indexer, ndim: int) -> bool:
    """
    Return True if we are all scalar indexers.

    Parameters
    ----------
    indexer : object
    ndim : int
        Number of dimensions in the object being indexed.

    Returns
    -------
    bool
    """
    if ndim == 1 and is_integer(indexer):
        # GH37748: allow indexer to be an integer for Series
        return True
    if isinstance(indexer, tuple) and len(indexer) == ndim:
        return all(is_integer(x) for x in indexer)
    return False


def is_empty_indexer(indexer) -> bool:
    """
    Check if we have an empty indexer.

    Parameters
    ----------
    indexer : object

    Returns
    -------
    bool
    """
    if is_list_like(indexer) and not len(indexer):
        return True
    if not isinstance(indexer, tuple):
        indexer = (indexer,)
    return any(isinstance(idx, np.ndarray) and len(idx) == 0 for idx in indexer)


# -----------------------------------------------------------
# Indexer Validation


def check_setitem_lengths(indexer, value, values) -> bool:
    """
    Validate that value and indexer are the same length.

    An special-case is allowed for when the indexer is a boolean array
    and the number of true values equals the length of ``value``. In
    this case, no exception is raised.

    Parameters
    ----------
    indexer : sequence
        Key for the setitem.
    value : array-like
        Value for the setitem.
    values : array-like
        Values being set into.

    Returns
    -------
    bool
        Whether this is an empty listlike setting which is a no-op.

    Raises
    ------
    ValueError
        When the indexer is an ndarray or list and the lengths don't match.
    """
    no_op = False

    if isinstance(indexer, (np.ndarray, list)):
        # We can ignore other listlikes because they are either
        #  a) not necessarily 1-D indexers, e.g. tuple
        #  b) boolean indexers e.g. BoolArray
        if is_list_like(value):
            if len(indexer) != len(value) and values.ndim == 1:
                # boolean with truth values == len of the value is ok too
                if isinstance(indexer, list):
                    indexer = np.array(indexer)
                if not (
                    isinstance(indexer, np.ndarray)
                    and indexer.dtype == np.bool_
                    and indexer.sum() == len(value)
                ):
                    raise ValueError(
                        "cannot set using a list-like indexer "
                        "with a different length than the value"
                    )
            if not len(indexer):
                no_op = True

    elif isinstance(indexer, slice):
        if is_list_like(value):
            if len(value) != length_of_indexer(indexer, values) and values.ndim == 1:
                # In case of two dimensional value is used row-wise and broadcasted
                raise ValueError(
                    "cannot set using a slice indexer with a "
                    "different length than the value"
                )
            if not len(value):
                no_op = True

    return no_op


def validate_indices(indices: np.ndarray, n: int) -> None:
    """
    Perform bounds-checking for an indexer.

    -1 is allowed for indicating missing values.

    Parameters
    ----------
    indices : ndarray
    n : int
        Length of the array being indexed.

    Raises
    ------
    ValueError

    Examples
    --------
    >>> validate_indices(np.array([1, 2]), 3) # OK

    >>> validate_indices(np.array([1, -2]), 3)
    Traceback (most recent call last):
        ...
    ValueError: negative dimensions are not allowed

    >>> validate_indices(np.array([1, 2, 3]), 3)
    Traceback (most recent call last):
        ...
    IndexError: indices are out-of-bounds

    >>> validate_indices(np.array([-1, -1]), 0) # OK

    >>> validate_indices(np.array([0, 1]), 0)
    Traceback (most recent call last):
        ...
    IndexError: indices are out-of-bounds
    """
    if len(indices):
        min_idx = indices.min()
        if min_idx < -1:
            msg = f"'indices' contains values less than allowed ({min_idx} < -1)"
            raise ValueError(msg)

        max_idx = indices.max()
        if max_idx >= n:
            raise IndexError("indices are out-of-bounds")


# -----------------------------------------------------------
# Indexer Conversion


def maybe_convert_indices(indices, n: int, verify: bool = True) -> np.ndarray:
    """
    Attempt to convert indices into valid, positive indices.

    If we have negative indices, translate to positive here.
    If we have indices that are out-of-bounds, raise an IndexError.

    Parameters
    ----------
    indices : array-like
        Array of indices that we are to convert.
    n : int
        Number of elements in the array that we are indexing.
    verify : bool, default True
        Check that all entries are between 0 and n - 1, inclusive.

    Returns
    -------
    array-like
        An array-like of positive indices that correspond to the ones
        that were passed in initially to this function.

    Raises
    ------
    IndexError
        One of the converted indices either exceeded the number of,
        elements (specified by `n`), or was still negative.
    """
    if isinstance(indices, list):
        indices = np.array(indices)
        if len(indices) == 0:
            # If `indices` is empty, np.array will return a float,
            # and will cause indexing errors.
            return np.empty(0, dtype=np.intp)

    mask = indices < 0
    if mask.any():
        indices = indices.copy()
        indices[mask] += n

    if verify:
        mask = (indices >= n) | (indices < 0)
        if mask.any():
            raise IndexError("indices are out-of-bounds")
    return indices


# -----------------------------------------------------------
# Unsorted


def length_of_indexer(indexer, target=None) -> int:
    """
    Return the expected length of target[indexer]

    Returns
    -------
    int
    """
    if target is not None and isinstance(indexer, slice):
        target_len = len(target)
        start = indexer.start
        stop = indexer.stop
        step = indexer.step
        if start is None:
            start = 0
        elif start < 0:
            start += target_len
        if stop is None or stop > target_len:
            stop = target_len
        elif stop < 0:
            stop += target_len
        if step is None:
            step = 1
        elif step < 0:
            start, stop = stop + 1, start + 1
            step = -step
        return (stop - start + step - 1) // step
    elif isinstance(indexer, (ABCSeries, ABCIndex, np.ndarray, list)):
        if isinstance(indexer, list):
            indexer = np.array(indexer)

        if indexer.dtype == bool:
            # GH#25774
            return indexer.sum()
        return len(indexer)
    elif isinstance(indexer, range):
        return (indexer.stop - indexer.start) // indexer.step
    elif not is_list_like_indexer(indexer):
        return 1
    raise AssertionError("cannot find the length of the indexer")


def disallow_ndim_indexing(result) -> None:
    """
    Helper function to disallow multi-dimensional indexing on 1D Series/Index.

    GH#27125 indexer like idx[:, None] expands dim, but we cannot do that
    and keep an index, so we used to return ndarray, which was deprecated
    in GH#30588.
    """
    if np.ndim(result) > 1:
        raise ValueError(
            "Multi-dimensional indexing (e.g. `obj[:, None]`) is no longer "
            "supported. Convert to a numpy array before indexing instead."
        )


def unpack_1tuple(tup):
    """
    If we have a length-1 tuple/list that contains a slice, unpack to just
    the slice.

    Notes
    -----
    The list case is deprecated.
    """
    if len(tup) == 1 and isinstance(tup[0], slice):
        # if we don't have a MultiIndex, we may still be able to handle
        #  a 1-tuple.  see test_1tuple_without_multiindex

        if isinstance(tup, list):
            # GH#31299
            raise ValueError(
                "Indexing with a single-item list containing a "
                "slice is not allowed. Pass a tuple instead.",
            )

        return tup[0]
    return tup


def check_key_length(columns: Index, key, value: DataFrame) -> None:
    """
    Checks if a key used as indexer has the same length as the columns it is
    associated with.

    Parameters
    ----------
    columns : Index The columns of the DataFrame to index.
    key : A list-like of keys to index with.
    value : DataFrame The value to set for the keys.

    Raises
    ------
    ValueError: If the length of key is not equal to the number of columns in value
                or if the number of columns referenced by key is not equal to number
                of columns.
    """
    if columns.is_unique:
        if len(value.columns) != len(key):
            raise ValueError("Columns must be same length as key")
    else:
        # Missing keys in columns are represented as -1
        if len(columns.get_indexer_non_unique(key)[0]) != len(value.columns):
            raise ValueError("Columns must be same length as key")


def unpack_tuple_and_ellipses(item: tuple):
    """
    Possibly unpack arr[..., n] to arr[n]
    """
    if len(item) > 1:
        # Note: we are assuming this indexing is being done on a 1D arraylike
        if item[0] is Ellipsis:
            item = item[1:]
        elif item[-1] is Ellipsis:
            item = item[:-1]

    if len(item) > 1:
        raise IndexError("too many indices for array.")

    item = item[0]
    return item


# -----------------------------------------------------------
# Public indexer validation


def check_array_indexer(array: AnyArrayLike, indexer: Any) -> Any:
    """
    Check if `indexer` is a valid array indexer for `array`.

    For a boolean mask, `array` and `indexer` are checked to have the same
    length. The dtype is validated, and if it is an integer or boolean
    ExtensionArray, it is checked if there are missing values present, and
    it is converted to the appropriate numpy array. Other dtypes will raise
    an error.

    Non-array indexers (integer, slice, Ellipsis, tuples, ..) are passed
    through as is.

    Parameters
    ----------
    array : array-like
        The array that is being indexed (only used for the length).
    indexer : array-like or list-like
        The array-like that's used to index. List-like input that is not yet
        a numpy array or an ExtensionArray is converted to one. Other input
        types are passed through as is.

    Returns
    -------
    numpy.ndarray
        The validated indexer as a numpy array that can be used to index.

    Raises
    ------
    IndexError
        When the lengths don't match.
    ValueError
        When `indexer` cannot be converted to a numpy ndarray to index
        (e.g. presence of missing values).

    See Also
    --------
    api.types.is_bool_dtype : Check if `key` is of boolean dtype.

    Examples
    --------
    When checking a boolean mask, a boolean ndarray is returned when the
    arguments are all valid.

    >>> mask = pd.array([True, False])
    >>> arr = pd.array([1, 2])
    >>> pd.api.indexers.check_array_indexer(arr, mask)
    array([ True, False])

    An IndexError is raised when the lengths don't match.

    >>> mask = pd.array([True, False, True])
    >>> pd.api.indexers.check_array_indexer(arr, mask)
    Traceback (most recent call last):
    ...
    IndexError: Boolean index has wrong length: 3 instead of 2.

    NA values in a boolean array are treated as False.

    >>> mask = pd.array([True, pd.NA])
    >>> pd.api.indexers.check_array_indexer(arr, mask)
    array([ True, False])

    A numpy boolean mask will get passed through (if the length is correct):

    >>> mask = np.array([True, False])
    >>> pd.api.indexers.check_array_indexer(arr, mask)
    array([ True, False])

    Similarly for integer indexers, an integer ndarray is returned when it is
    a valid indexer, otherwise an error is  (for integer indexers, a matching
    length is not required):

    >>> indexer = pd.array([0, 2], dtype="Int64")
    >>> arr = pd.array([1, 2, 3])
    >>> pd.api.indexers.check_array_indexer(arr, indexer)
    array([0, 2])

    >>> indexer = pd.array([0, pd.NA], dtype="Int64")
    >>> pd.api.indexers.check_array_indexer(arr, indexer)
    Traceback (most recent call last):
    ...
    ValueError: Cannot index with an integer indexer containing NA values

    For non-integer/boolean dtypes, an appropriate error is raised:

    >>> indexer = np.array([0., 2.], dtype="float64")
    >>> pd.api.indexers.check_array_indexer(arr, indexer)
    Traceback (most recent call last):
    ...
    IndexError: arrays used as indices must be of integer or boolean type
    """
    from pandas.core.construction import array as pd_array

    # whatever is not an array-like is returned as-is (possible valid array
    # indexers that are not array-like: integer, slice, Ellipsis, None)
    # In this context, tuples are not considered as array-like, as they have
    # a specific meaning in indexing (multi-dimensional indexing)
    if is_list_like(indexer):
        if isinstance(indexer, tuple):
            return indexer
    else:
        return indexer

    # convert list-likes to array
    if not is_array_like(indexer):
        indexer = pd_array(indexer)
        if len(indexer) == 0:
            # empty list is converted to float array by pd.array
            indexer = np.array([], dtype=np.intp)

    dtype = indexer.dtype
    if is_bool_dtype(dtype):
        if isinstance(dtype, ExtensionDtype):
            indexer = indexer.to_numpy(dtype=bool, na_value=False)
        else:
            indexer = np.asarray(indexer, dtype=bool)

        # GH26658
        if len(indexer) != len(array):
            raise IndexError(
                f"Boolean index has wrong length: "
                f"{len(indexer)} instead of {len(array)}"
            )
    elif is_integer_dtype(dtype):
        try:
            indexer = np.asarray(indexer, dtype=np.intp)
        except ValueError as err:
            raise ValueError(
                "Cannot index with an integer indexer containing NA values"
            ) from err
    else:
        raise IndexError("arrays used as indices must be of integer or boolean type")

    return indexer

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\sql\default_comparator.py ===
# sql/default_comparator.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

"""Default implementation of SQL comparison operations.
"""

from __future__ import annotations

import typing
from typing import Any
from typing import Callable
from typing import Dict
from typing import NoReturn
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

from . import coercions
from . import operators
from . import roles
from . import type_api
from .elements import and_
from .elements import BinaryExpression
from .elements import ClauseElement
from .elements import CollationClause
from .elements import CollectionAggregate
from .elements import ExpressionClauseList
from .elements import False_
from .elements import Null
from .elements import OperatorExpression
from .elements import or_
from .elements import True_
from .elements import UnaryExpression
from .operators import OperatorType
from .. import exc
from .. import util

_T = typing.TypeVar("_T", bound=Any)

if typing.TYPE_CHECKING:
    from .elements import ColumnElement
    from .operators import custom_op
    from .type_api import TypeEngine


def _boolean_compare(
    expr: ColumnElement[Any],
    op: OperatorType,
    obj: Any,
    *,
    negate_op: Optional[OperatorType] = None,
    reverse: bool = False,
    _python_is_types: Tuple[Type[Any], ...] = (type(None), bool),
    result_type: Optional[TypeEngine[bool]] = None,
    **kwargs: Any,
) -> OperatorExpression[bool]:
    if result_type is None:
        result_type = type_api.BOOLEANTYPE

    if isinstance(obj, _python_is_types + (Null, True_, False_)):
        # allow x ==/!= True/False to be treated as a literal.
        # this comes out to "== / != true/false" or "1/0" if those
        # constants aren't supported and works on all platforms
        if op in (operators.eq, operators.ne) and isinstance(
            obj, (bool, True_, False_)
        ):
            return OperatorExpression._construct_for_op(
                expr,
                coercions.expect(roles.ConstExprRole, obj),
                op,
                type_=result_type,
                negate=negate_op,
                modifiers=kwargs,
            )
        elif op in (
            operators.is_distinct_from,
            operators.is_not_distinct_from,
        ):
            return OperatorExpression._construct_for_op(
                expr,
                coercions.expect(roles.ConstExprRole, obj),
                op,
                type_=result_type,
                negate=negate_op,
                modifiers=kwargs,
            )
        elif expr._is_collection_aggregate:
            obj = coercions.expect(
                roles.ConstExprRole, element=obj, operator=op, expr=expr
            )
        else:
            # all other None uses IS, IS NOT
            if op in (operators.eq, operators.is_):
                return OperatorExpression._construct_for_op(
                    expr,
                    coercions.expect(roles.ConstExprRole, obj),
                    operators.is_,
                    negate=operators.is_not,
                    type_=result_type,
                )
            elif op in (operators.ne, operators.is_not):
                return OperatorExpression._construct_for_op(
                    expr,
                    coercions.expect(roles.ConstExprRole, obj),
                    operators.is_not,
                    negate=operators.is_,
                    type_=result_type,
                )
            else:
                raise exc.ArgumentError(
                    "Only '=', '!=', 'is_()', 'is_not()', "
                    "'is_distinct_from()', 'is_not_distinct_from()' "
                    "operators can be used with None/True/False"
                )
    else:
        obj = coercions.expect(
            roles.BinaryElementRole, element=obj, operator=op, expr=expr
        )

    if reverse:
        return OperatorExpression._construct_for_op(
            obj,
            expr,
            op,
            type_=result_type,
            negate=negate_op,
            modifiers=kwargs,
        )
    else:
        return OperatorExpression._construct_for_op(
            expr,
            obj,
            op,
            type_=result_type,
            negate=negate_op,
            modifiers=kwargs,
        )


def _custom_op_operate(
    expr: ColumnElement[Any],
    op: custom_op[Any],
    obj: Any,
    reverse: bool = False,
    result_type: Optional[TypeEngine[Any]] = None,
    **kw: Any,
) -> ColumnElement[Any]:
    if result_type is None:
        if op.return_type:
            result_type = op.return_type
        elif op.is_comparison:
            result_type = type_api.BOOLEANTYPE

    return _binary_operate(
        expr, op, obj, reverse=reverse, result_type=result_type, **kw
    )


def _binary_operate(
    expr: ColumnElement[Any],
    op: OperatorType,
    obj: roles.BinaryElementRole[Any],
    *,
    reverse: bool = False,
    result_type: Optional[TypeEngine[_T]] = None,
    **kw: Any,
) -> OperatorExpression[_T]:
    coerced_obj = coercions.expect(
        roles.BinaryElementRole, obj, expr=expr, operator=op
    )

    if reverse:
        left, right = coerced_obj, expr
    else:
        left, right = expr, coerced_obj

    if result_type is None:
        op, result_type = left.comparator._adapt_expression(
            op, right.comparator
        )

    return OperatorExpression._construct_for_op(
        left, right, op, type_=result_type, modifiers=kw
    )


def _conjunction_operate(
    expr: ColumnElement[Any], op: OperatorType, other: Any, **kw: Any
) -> ColumnElement[Any]:
    if op is operators.and_:
        return and_(expr, other)
    elif op is operators.or_:
        return or_(expr, other)
    else:
        raise NotImplementedError()


def _scalar(
    expr: ColumnElement[Any],
    op: OperatorType,
    fn: Callable[[ColumnElement[Any]], ColumnElement[Any]],
    **kw: Any,
) -> ColumnElement[Any]:
    return fn(expr)


def _in_impl(
    expr: ColumnElement[Any],
    op: OperatorType,
    seq_or_selectable: ClauseElement,
    negate_op: OperatorType,
    **kw: Any,
) -> ColumnElement[Any]:
    seq_or_selectable = coercions.expect(
        roles.InElementRole, seq_or_selectable, expr=expr, operator=op
    )
    if "in_ops" in seq_or_selectable._annotations:
        op, negate_op = seq_or_selectable._annotations["in_ops"]

    return _boolean_compare(
        expr, op, seq_or_selectable, negate_op=negate_op, **kw
    )


def _getitem_impl(
    expr: ColumnElement[Any], op: OperatorType, other: Any, **kw: Any
) -> ColumnElement[Any]:
    if (
        isinstance(expr.type, type_api.INDEXABLE)
        or isinstance(expr.type, type_api.TypeDecorator)
        and isinstance(expr.type.impl_instance, type_api.INDEXABLE)
    ):
        other = coercions.expect(
            roles.BinaryElementRole, other, expr=expr, operator=op
        )
        return _binary_operate(expr, op, other, **kw)
    else:
        _unsupported_impl(expr, op, other, **kw)


def _unsupported_impl(
    expr: ColumnElement[Any], op: OperatorType, *arg: Any, **kw: Any
) -> NoReturn:
    raise NotImplementedError(
        "Operator '%s' is not supported on this expression" % op.__name__
    )


def _inv_impl(
    expr: ColumnElement[Any], op: OperatorType, **kw: Any
) -> ColumnElement[Any]:
    """See :meth:`.ColumnOperators.__inv__`."""

    # undocumented element currently used by the ORM for
    # relationship.contains()
    if hasattr(expr, "negation_clause"):
        return expr.negation_clause
    else:
        return expr._negate()


def _neg_impl(
    expr: ColumnElement[Any], op: OperatorType, **kw: Any
) -> ColumnElement[Any]:
    """See :meth:`.ColumnOperators.__neg__`."""
    return UnaryExpression(expr, operator=operators.neg, type_=expr.type)


def _bitwise_not_impl(
    expr: ColumnElement[Any], op: OperatorType, **kw: Any
) -> ColumnElement[Any]:
    """See :meth:`.ColumnOperators.bitwise_not`."""

    return UnaryExpression(
        expr, operator=operators.bitwise_not_op, type_=expr.type
    )


def _match_impl(
    expr: ColumnElement[Any], op: OperatorType, other: Any, **kw: Any
) -> ColumnElement[Any]:
    """See :meth:`.ColumnOperators.match`."""

    return _boolean_compare(
        expr,
        operators.match_op,
        coercions.expect(
            roles.BinaryElementRole,
            other,
            expr=expr,
            operator=operators.match_op,
        ),
        result_type=type_api.MATCHTYPE,
        negate_op=(
            operators.not_match_op
            if op is operators.match_op
            else operators.match_op
        ),
        **kw,
    )


def _distinct_impl(
    expr: ColumnElement[Any], op: OperatorType, **kw: Any
) -> ColumnElement[Any]:
    """See :meth:`.ColumnOperators.distinct`."""
    return UnaryExpression(
        expr, operator=operators.distinct_op, type_=expr.type
    )


def _between_impl(
    expr: ColumnElement[Any],
    op: OperatorType,
    cleft: Any,
    cright: Any,
    **kw: Any,
) -> ColumnElement[Any]:
    """See :meth:`.ColumnOperators.between`."""
    return BinaryExpression(
        expr,
        ExpressionClauseList._construct_for_list(
            operators.and_,
            type_api.NULLTYPE,
            coercions.expect(
                roles.BinaryElementRole,
                cleft,
                expr=expr,
                operator=operators.and_,
            ),
            coercions.expect(
                roles.BinaryElementRole,
                cright,
                expr=expr,
                operator=operators.and_,
            ),
            group=False,
        ),
        op,
        negate=(
            operators.not_between_op
            if op is operators.between_op
            else operators.between_op
        ),
        modifiers=kw,
    )


def _collate_impl(
    expr: ColumnElement[str], op: OperatorType, collation: str, **kw: Any
) -> ColumnElement[str]:
    return CollationClause._create_collation_expression(expr, collation)


def _regexp_match_impl(
    expr: ColumnElement[str],
    op: OperatorType,
    pattern: Any,
    flags: Optional[str],
    **kw: Any,
) -> ColumnElement[Any]:
    return BinaryExpression(
        expr,
        coercions.expect(
            roles.BinaryElementRole,
            pattern,
            expr=expr,
            operator=operators.comma_op,
        ),
        op,
        negate=operators.not_regexp_match_op,
        modifiers={"flags": flags},
    )


def _regexp_replace_impl(
    expr: ColumnElement[Any],
    op: OperatorType,
    pattern: Any,
    replacement: Any,
    flags: Optional[str],
    **kw: Any,
) -> ColumnElement[Any]:
    return BinaryExpression(
        expr,
        ExpressionClauseList._construct_for_list(
            operators.comma_op,
            type_api.NULLTYPE,
            coercions.expect(
                roles.BinaryElementRole,
                pattern,
                expr=expr,
                operator=operators.comma_op,
            ),
            coercions.expect(
                roles.BinaryElementRole,
                replacement,
                expr=expr,
                operator=operators.comma_op,
            ),
            group=False,
        ),
        op,
        modifiers={"flags": flags},
    )


# a mapping of operators with the method they use, along with
# additional keyword arguments to be passed
operator_lookup: Dict[
    str,
    Tuple[
        Callable[..., ColumnElement[Any]],
        util.immutabledict[
            str, Union[OperatorType, Callable[..., ColumnElement[Any]]]
        ],
    ],
] = {
    "and_": (_conjunction_operate, util.EMPTY_DICT),
    "or_": (_conjunction_operate, util.EMPTY_DICT),
    "inv": (_inv_impl, util.EMPTY_DICT),
    "add": (_binary_operate, util.EMPTY_DICT),
    "mul": (_binary_operate, util.EMPTY_DICT),
    "sub": (_binary_operate, util.EMPTY_DICT),
    "div": (_binary_operate, util.EMPTY_DICT),
    "mod": (_binary_operate, util.EMPTY_DICT),
    "bitwise_xor_op": (_binary_operate, util.EMPTY_DICT),
    "bitwise_or_op": (_binary_operate, util.EMPTY_DICT),
    "bitwise_and_op": (_binary_operate, util.EMPTY_DICT),
    "bitwise_not_op": (_bitwise_not_impl, util.EMPTY_DICT),
    "bitwise_lshift_op": (_binary_operate, util.EMPTY_DICT),
    "bitwise_rshift_op": (_binary_operate, util.EMPTY_DICT),
    "truediv": (_binary_operate, util.EMPTY_DICT),
    "floordiv": (_binary_operate, util.EMPTY_DICT),
    "custom_op": (_custom_op_operate, util.EMPTY_DICT),
    "json_path_getitem_op": (_binary_operate, util.EMPTY_DICT),
    "json_getitem_op": (_binary_operate, util.EMPTY_DICT),
    "concat_op": (_binary_operate, util.EMPTY_DICT),
    "any_op": (
        _scalar,
        util.immutabledict({"fn": CollectionAggregate._create_any}),
    ),
    "all_op": (
        _scalar,
        util.immutabledict({"fn": CollectionAggregate._create_all}),
    ),
    "lt": (_boolean_compare, util.immutabledict({"negate_op": operators.ge})),
    "le": (_boolean_compare, util.immutabledict({"negate_op": operators.gt})),
    "ne": (_boolean_compare, util.immutabledict({"negate_op": operators.eq})),
    "gt": (_boolean_compare, util.immutabledict({"negate_op": operators.le})),
    "ge": (_boolean_compare, util.immutabledict({"negate_op": operators.lt})),
    "eq": (_boolean_compare, util.immutabledict({"negate_op": operators.ne})),
    "is_distinct_from": (
        _boolean_compare,
        util.immutabledict({"negate_op": operators.is_not_distinct_from}),
    ),
    "is_not_distinct_from": (
        _boolean_compare,
        util.immutabledict({"negate_op": operators.is_distinct_from}),
    ),
    "like_op": (
        _boolean_compare,
        util.immutabledict({"negate_op": operators.not_like_op}),
    ),
    "ilike_op": (
        _boolean_compare,
        util.immutabledict({"negate_op": operators.not_ilike_op}),
    ),
    "not_like_op": (
        _boolean_compare,
        util.immutabledict({"negate_op": operators.like_op}),
    ),
    "not_ilike_op": (
        _boolean_compare,
        util.immutabledict({"negate_op": operators.ilike_op}),
    ),
    "contains_op": (
        _boolean_compare,
        util.immutabledict({"negate_op": operators.not_contains_op}),
    ),
    "icontains_op": (
        _boolean_compare,
        util.immutabledict({"negate_op": operators.not_icontains_op}),
    ),
    "startswith_op": (
        _boolean_compare,
        util.immutabledict({"negate_op": operators.not_startswith_op}),
    ),
    "istartswith_op": (
        _boolean_compare,
        util.immutabledict({"negate_op": operators.not_istartswith_op}),
    ),
    "endswith_op": (
        _boolean_compare,
        util.immutabledict({"negate_op": operators.not_endswith_op}),
    ),
    "iendswith_op": (
        _boolean_compare,
        util.immutabledict({"negate_op": operators.not_iendswith_op}),
    ),
    "desc_op": (
        _scalar,
        util.immutabledict({"fn": UnaryExpression._create_desc}),
    ),
    "asc_op": (
        _scalar,
        util.immutabledict({"fn": UnaryExpression._create_asc}),
    ),
    "nulls_first_op": (
        _scalar,
        util.immutabledict({"fn": UnaryExpression._create_nulls_first}),
    ),
    "nulls_last_op": (
        _scalar,
        util.immutabledict({"fn": UnaryExpression._create_nulls_last}),
    ),
    "in_op": (
        _in_impl,
        util.immutabledict({"negate_op": operators.not_in_op}),
    ),
    "not_in_op": (
        _in_impl,
        util.immutabledict({"negate_op": operators.in_op}),
    ),
    "is_": (
        _boolean_compare,
        util.immutabledict({"negate_op": operators.is_}),
    ),
    "is_not": (
        _boolean_compare,
        util.immutabledict({"negate_op": operators.is_not}),
    ),
    "collate": (_collate_impl, util.EMPTY_DICT),
    "match_op": (_match_impl, util.EMPTY_DICT),
    "not_match_op": (_match_impl, util.EMPTY_DICT),
    "distinct_op": (_distinct_impl, util.EMPTY_DICT),
    "between_op": (_between_impl, util.EMPTY_DICT),
    "not_between_op": (_between_impl, util.EMPTY_DICT),
    "neg": (_neg_impl, util.EMPTY_DICT),
    "getitem": (_getitem_impl, util.EMPTY_DICT),
    "lshift": (_unsupported_impl, util.EMPTY_DICT),
    "rshift": (_unsupported_impl, util.EMPTY_DICT),
    "contains": (_unsupported_impl, util.EMPTY_DICT),
    "regexp_match_op": (_regexp_match_impl, util.EMPTY_DICT),
    "not_regexp_match_op": (_regexp_match_impl, util.EMPTY_DICT),
    "regexp_replace_op": (_regexp_replace_impl, util.EMPTY_DICT),
}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\click\decorators.py ===
from __future__ import annotations

import inspect
import typing as t
from functools import update_wrapper
from gettext import gettext as _

from .core import Argument
from .core import Command
from .core import Context
from .core import Group
from .core import Option
from .core import Parameter
from .globals import get_current_context
from .utils import echo

if t.TYPE_CHECKING:
    import typing_extensions as te

    P = te.ParamSpec("P")

R = t.TypeVar("R")
T = t.TypeVar("T")
_AnyCallable = t.Callable[..., t.Any]
FC = t.TypeVar("FC", bound="_AnyCallable | Command")


def pass_context(f: t.Callable[te.Concatenate[Context, P], R]) -> t.Callable[P, R]:
    """Marks a callback as wanting to receive the current context
    object as first argument.
    """

    def new_func(*args: P.args, **kwargs: P.kwargs) -> R:
        return f(get_current_context(), *args, **kwargs)

    return update_wrapper(new_func, f)


def pass_obj(f: t.Callable[te.Concatenate[T, P], R]) -> t.Callable[P, R]:
    """Similar to :func:`pass_context`, but only pass the object on the
    context onwards (:attr:`Context.obj`).  This is useful if that object
    represents the state of a nested system.
    """

    def new_func(*args: P.args, **kwargs: P.kwargs) -> R:
        return f(get_current_context().obj, *args, **kwargs)

    return update_wrapper(new_func, f)


def make_pass_decorator(
    object_type: type[T], ensure: bool = False
) -> t.Callable[[t.Callable[te.Concatenate[T, P], R]], t.Callable[P, R]]:
    """Given an object type this creates a decorator that will work
    similar to :func:`pass_obj` but instead of passing the object of the
    current context, it will find the innermost context of type
    :func:`object_type`.

    This generates a decorator that works roughly like this::

        from functools import update_wrapper

        def decorator(f):
            @pass_context
            def new_func(ctx, *args, **kwargs):
                obj = ctx.find_object(object_type)
                return ctx.invoke(f, obj, *args, **kwargs)
            return update_wrapper(new_func, f)
        return decorator

    :param object_type: the type of the object to pass.
    :param ensure: if set to `True`, a new object will be created and
                   remembered on the context if it's not there yet.
    """

    def decorator(f: t.Callable[te.Concatenate[T, P], R]) -> t.Callable[P, R]:
        def new_func(*args: P.args, **kwargs: P.kwargs) -> R:
            ctx = get_current_context()

            obj: T | None
            if ensure:
                obj = ctx.ensure_object(object_type)
            else:
                obj = ctx.find_object(object_type)

            if obj is None:
                raise RuntimeError(
                    "Managed to invoke callback without a context"
                    f" object of type {object_type.__name__!r}"
                    " existing."
                )

            return ctx.invoke(f, obj, *args, **kwargs)

        return update_wrapper(new_func, f)

    return decorator


def pass_meta_key(
    key: str, *, doc_description: str | None = None
) -> t.Callable[[t.Callable[te.Concatenate[T, P], R]], t.Callable[P, R]]:
    """Create a decorator that passes a key from
    :attr:`click.Context.meta` as the first argument to the decorated
    function.

    :param key: Key in ``Context.meta`` to pass.
    :param doc_description: Description of the object being passed,
        inserted into the decorator's docstring. Defaults to "the 'key'
        key from Context.meta".

    .. versionadded:: 8.0
    """

    def decorator(f: t.Callable[te.Concatenate[T, P], R]) -> t.Callable[P, R]:
        def new_func(*args: P.args, **kwargs: P.kwargs) -> R:
            ctx = get_current_context()
            obj = ctx.meta[key]
            return ctx.invoke(f, obj, *args, **kwargs)

        return update_wrapper(new_func, f)

    if doc_description is None:
        doc_description = f"the {key!r} key from :attr:`click.Context.meta`"

    decorator.__doc__ = (
        f"Decorator that passes {doc_description} as the first argument"
        " to the decorated function."
    )
    return decorator


CmdType = t.TypeVar("CmdType", bound=Command)


# variant: no call, directly as decorator for a function.
@t.overload
def command(name: _AnyCallable) -> Command: ...


# variant: with positional name and with positional or keyword cls argument:
# @command(namearg, CommandCls, ...) or @command(namearg, cls=CommandCls, ...)
@t.overload
def command(
    name: str | None,
    cls: type[CmdType],
    **attrs: t.Any,
) -> t.Callable[[_AnyCallable], CmdType]: ...


# variant: name omitted, cls _must_ be a keyword argument, @command(cls=CommandCls, ...)
@t.overload
def command(
    name: None = None,
    *,
    cls: type[CmdType],
    **attrs: t.Any,
) -> t.Callable[[_AnyCallable], CmdType]: ...


# variant: with optional string name, no cls argument provided.
@t.overload
def command(
    name: str | None = ..., cls: None = None, **attrs: t.Any
) -> t.Callable[[_AnyCallable], Command]: ...


def command(
    name: str | _AnyCallable | None = None,
    cls: type[CmdType] | None = None,
    **attrs: t.Any,
) -> Command | t.Callable[[_AnyCallable], Command | CmdType]:
    r"""Creates a new :class:`Command` and uses the decorated function as
    callback.  This will also automatically attach all decorated
    :func:`option`\s and :func:`argument`\s as parameters to the command.

    The name of the command defaults to the name of the function, converted to
    lowercase, with underscores ``_`` replaced by dashes ``-``, and the suffixes
    ``_command``, ``_cmd``, ``_group``, and ``_grp`` are removed. For example,
    ``init_data_command`` becomes ``init-data``.

    All keyword arguments are forwarded to the underlying command class.
    For the ``params`` argument, any decorated params are appended to
    the end of the list.

    Once decorated the function turns into a :class:`Command` instance
    that can be invoked as a command line utility or be attached to a
    command :class:`Group`.

    :param name: The name of the command. Defaults to modifying the function's
        name as described above.
    :param cls: The command class to create. Defaults to :class:`Command`.

    .. versionchanged:: 8.2
        The suffixes ``_command``, ``_cmd``, ``_group``, and ``_grp`` are
        removed when generating the name.

    .. versionchanged:: 8.1
        This decorator can be applied without parentheses.

    .. versionchanged:: 8.1
        The ``params`` argument can be used. Decorated params are
        appended to the end of the list.
    """

    func: t.Callable[[_AnyCallable], t.Any] | None = None

    if callable(name):
        func = name
        name = None
        assert cls is None, "Use 'command(cls=cls)(callable)' to specify a class."
        assert not attrs, "Use 'command(**kwargs)(callable)' to provide arguments."

    if cls is None:
        cls = t.cast("type[CmdType]", Command)

    def decorator(f: _AnyCallable) -> CmdType:
        if isinstance(f, Command):
            raise TypeError("Attempted to convert a callback into a command twice.")

        attr_params = attrs.pop("params", None)
        params = attr_params if attr_params is not None else []

        try:
            decorator_params = f.__click_params__  # type: ignore
        except AttributeError:
            pass
        else:
            del f.__click_params__  # type: ignore
            params.extend(reversed(decorator_params))

        if attrs.get("help") is None:
            attrs["help"] = f.__doc__

        if t.TYPE_CHECKING:
            assert cls is not None
            assert not callable(name)

        if name is not None:
            cmd_name = name
        else:
            cmd_name = f.__name__.lower().replace("_", "-")
            cmd_left, sep, suffix = cmd_name.rpartition("-")

            if sep and suffix in {"command", "cmd", "group", "grp"}:
                cmd_name = cmd_left

        cmd = cls(name=cmd_name, callback=f, params=params, **attrs)
        cmd.__doc__ = f.__doc__
        return cmd

    if func is not None:
        return decorator(func)

    return decorator


GrpType = t.TypeVar("GrpType", bound=Group)


# variant: no call, directly as decorator for a function.
@t.overload
def group(name: _AnyCallable) -> Group: ...


# variant: with positional name and with positional or keyword cls argument:
# @group(namearg, GroupCls, ...) or @group(namearg, cls=GroupCls, ...)
@t.overload
def group(
    name: str | None,
    cls: type[GrpType],
    **attrs: t.Any,
) -> t.Callable[[_AnyCallable], GrpType]: ...


# variant: name omitted, cls _must_ be a keyword argument, @group(cmd=GroupCls, ...)
@t.overload
def group(
    name: None = None,
    *,
    cls: type[GrpType],
    **attrs: t.Any,
) -> t.Callable[[_AnyCallable], GrpType]: ...


# variant: with optional string name, no cls argument provided.
@t.overload
def group(
    name: str | None = ..., cls: None = None, **attrs: t.Any
) -> t.Callable[[_AnyCallable], Group]: ...


def group(
    name: str | _AnyCallable | None = None,
    cls: type[GrpType] | None = None,
    **attrs: t.Any,
) -> Group | t.Callable[[_AnyCallable], Group | GrpType]:
    """Creates a new :class:`Group` with a function as callback.  This
    works otherwise the same as :func:`command` just that the `cls`
    parameter is set to :class:`Group`.

    .. versionchanged:: 8.1
        This decorator can be applied without parentheses.
    """
    if cls is None:
        cls = t.cast("type[GrpType]", Group)

    if callable(name):
        return command(cls=cls, **attrs)(name)

    return command(name, cls, **attrs)


def _param_memo(f: t.Callable[..., t.Any], param: Parameter) -> None:
    if isinstance(f, Command):
        f.params.append(param)
    else:
        if not hasattr(f, "__click_params__"):
            f.__click_params__ = []  # type: ignore

        f.__click_params__.append(param)  # type: ignore


def argument(
    *param_decls: str, cls: type[Argument] | None = None, **attrs: t.Any
) -> t.Callable[[FC], FC]:
    """Attaches an argument to the command.  All positional arguments are
    passed as parameter declarations to :class:`Argument`; all keyword
    arguments are forwarded unchanged (except ``cls``).
    This is equivalent to creating an :class:`Argument` instance manually
    and attaching it to the :attr:`Command.params` list.

    For the default argument class, refer to :class:`Argument` and
    :class:`Parameter` for descriptions of parameters.

    :param cls: the argument class to instantiate.  This defaults to
                :class:`Argument`.
    :param param_decls: Passed as positional arguments to the constructor of
        ``cls``.
    :param attrs: Passed as keyword arguments to the constructor of ``cls``.
    """
    if cls is None:
        cls = Argument

    def decorator(f: FC) -> FC:
        _param_memo(f, cls(param_decls, **attrs))
        return f

    return decorator


def option(
    *param_decls: str, cls: type[Option] | None = None, **attrs: t.Any
) -> t.Callable[[FC], FC]:
    """Attaches an option to the command.  All positional arguments are
    passed as parameter declarations to :class:`Option`; all keyword
    arguments are forwarded unchanged (except ``cls``).
    This is equivalent to creating an :class:`Option` instance manually
    and attaching it to the :attr:`Command.params` list.

    For the default option class, refer to :class:`Option` and
    :class:`Parameter` for descriptions of parameters.

    :param cls: the option class to instantiate.  This defaults to
                :class:`Option`.
    :param param_decls: Passed as positional arguments to the constructor of
        ``cls``.
    :param attrs: Passed as keyword arguments to the constructor of ``cls``.
    """
    if cls is None:
        cls = Option

    def decorator(f: FC) -> FC:
        _param_memo(f, cls(param_decls, **attrs))
        return f

    return decorator


def confirmation_option(*param_decls: str, **kwargs: t.Any) -> t.Callable[[FC], FC]:
    """Add a ``--yes`` option which shows a prompt before continuing if
    not passed. If the prompt is declined, the program will exit.

    :param param_decls: One or more option names. Defaults to the single
        value ``"--yes"``.
    :param kwargs: Extra arguments are passed to :func:`option`.
    """

    def callback(ctx: Context, param: Parameter, value: bool) -> None:
        if not value:
            ctx.abort()

    if not param_decls:
        param_decls = ("--yes",)

    kwargs.setdefault("is_flag", True)
    kwargs.setdefault("callback", callback)
    kwargs.setdefault("expose_value", False)
    kwargs.setdefault("prompt", "Do you want to continue?")
    kwargs.setdefault("help", "Confirm the action without prompting.")
    return option(*param_decls, **kwargs)


def password_option(*param_decls: str, **kwargs: t.Any) -> t.Callable[[FC], FC]:
    """Add a ``--password`` option which prompts for a password, hiding
    input and asking to enter the value again for confirmation.

    :param param_decls: One or more option names. Defaults to the single
        value ``"--password"``.
    :param kwargs: Extra arguments are passed to :func:`option`.
    """
    if not param_decls:
        param_decls = ("--password",)

    kwargs.setdefault("prompt", True)
    kwargs.setdefault("confirmation_prompt", True)
    kwargs.setdefault("hide_input", True)
    return option(*param_decls, **kwargs)


def version_option(
    version: str | None = None,
    *param_decls: str,
    package_name: str | None = None,
    prog_name: str | None = None,
    message: str | None = None,
    **kwargs: t.Any,
) -> t.Callable[[FC], FC]:
    """Add a ``--version`` option which immediately prints the version
    number and exits the program.

    If ``version`` is not provided, Click will try to detect it using
    :func:`importlib.metadata.version` to get the version for the
    ``package_name``.

    If ``package_name`` is not provided, Click will try to detect it by
    inspecting the stack frames. This will be used to detect the
    version, so it must match the name of the installed package.

    :param version: The version number to show. If not provided, Click
        will try to detect it.
    :param param_decls: One or more option names. Defaults to the single
        value ``"--version"``.
    :param package_name: The package name to detect the version from. If
        not provided, Click will try to detect it.
    :param prog_name: The name of the CLI to show in the message. If not
        provided, it will be detected from the command.
    :param message: The message to show. The values ``%(prog)s``,
        ``%(package)s``, and ``%(version)s`` are available. Defaults to
        ``"%(prog)s, version %(version)s"``.
    :param kwargs: Extra arguments are passed to :func:`option`.
    :raise RuntimeError: ``version`` could not be detected.

    .. versionchanged:: 8.0
        Add the ``package_name`` parameter, and the ``%(package)s``
        value for messages.

    .. versionchanged:: 8.0
        Use :mod:`importlib.metadata` instead of ``pkg_resources``. The
        version is detected based on the package name, not the entry
        point name. The Python package name must match the installed
        package name, or be passed with ``package_name=``.
    """
    if message is None:
        message = _("%(prog)s, version %(version)s")

    if version is None and package_name is None:
        frame = inspect.currentframe()
        f_back = frame.f_back if frame is not None else None
        f_globals = f_back.f_globals if f_back is not None else None
        # break reference cycle
        # https://docs.python.org/3/library/inspect.html#the-interpreter-stack
        del frame

        if f_globals is not None:
            package_name = f_globals.get("__name__")

            if package_name == "__main__":
                package_name = f_globals.get("__package__")

            if package_name:
                package_name = package_name.partition(".")[0]

    def callback(ctx: Context, param: Parameter, value: bool) -> None:
        if not value or ctx.resilient_parsing:
            return

        nonlocal prog_name
        nonlocal version

        if prog_name is None:
            prog_name = ctx.find_root().info_name

        if version is None and package_name is not None:
            import importlib.metadata

            try:
                version = importlib.metadata.version(package_name)
            except importlib.metadata.PackageNotFoundError:
                raise RuntimeError(
                    f"{package_name!r} is not installed. Try passing"
                    " 'package_name' instead."
                ) from None

        if version is None:
            raise RuntimeError(
                f"Could not determine the version for {package_name!r} automatically."
            )

        echo(
            message % {"prog": prog_name, "package": package_name, "version": version},
            color=ctx.color,
        )
        ctx.exit()

    if not param_decls:
        param_decls = ("--version",)

    kwargs.setdefault("is_flag", True)
    kwargs.setdefault("expose_value", False)
    kwargs.setdefault("is_eager", True)
    kwargs.setdefault("help", _("Show the version and exit."))
    kwargs["callback"] = callback
    return option(*param_decls, **kwargs)


def help_option(*param_decls: str, **kwargs: t.Any) -> t.Callable[[FC], FC]:
    """Pre-configured ``--help`` option which immediately prints the help page
    and exits the program.

    :param param_decls: One or more option names. Defaults to the single
        value ``"--help"``.
    :param kwargs: Extra arguments are passed to :func:`option`.
    """

    def show_help(ctx: Context, param: Parameter, value: bool) -> None:
        """Callback that print the help page on ``<stdout>`` and exits."""
        if value and not ctx.resilient_parsing:
            echo(ctx.get_help(), color=ctx.color)
            ctx.exit()

    if not param_decls:
        param_decls = ("--help",)

    kwargs.setdefault("is_flag", True)
    kwargs.setdefault("expose_value", False)
    kwargs.setdefault("is_eager", True)
    kwargs.setdefault("help", _("Show this message and exit."))
    kwargs.setdefault("callback", show_help)

    return option(*param_decls, **kwargs)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\ctx_iv.py ===
import operator

from . import libmp

from .libmp.backend import basestring

from .libmp import (
    int_types, MPZ_ONE,
    prec_to_dps, dps_to_prec, repr_dps,
    round_floor, round_ceiling,
    fzero, finf, fninf, fnan,
    mpf_le, mpf_neg,
    from_int, from_float, from_str, from_rational,
    mpi_mid, mpi_delta, mpi_str,
    mpi_abs, mpi_pos, mpi_neg, mpi_add, mpi_sub,
    mpi_mul, mpi_div, mpi_pow_int, mpi_pow,
    mpi_from_str,
    mpci_pos, mpci_neg, mpci_add, mpci_sub, mpci_mul, mpci_div, mpci_pow,
    mpci_abs, mpci_pow, mpci_exp, mpci_log,
    ComplexResult,
    mpf_hash, mpc_hash)
from .matrices.matrices import _matrix

mpi_zero = (fzero, fzero)

from .ctx_base import StandardBaseContext

new = object.__new__

def convert_mpf_(x, prec, rounding):
    if hasattr(x, "_mpf_"): return x._mpf_
    if isinstance(x, int_types): return from_int(x, prec, rounding)
    if isinstance(x, float): return from_float(x, prec, rounding)
    if isinstance(x, basestring): return from_str(x, prec, rounding)
    raise NotImplementedError


class ivmpf(object):
    """
    Interval arithmetic class. Precision is controlled by iv.prec.
    """

    def __new__(cls, x=0):
        return cls.ctx.convert(x)

    def cast(self, cls, f_convert):
        a, b = self._mpi_
        if a == b:
            return cls(f_convert(a))
        raise ValueError

    def __int__(self):
        return self.cast(int, libmp.to_int)

    def __float__(self):
        return self.cast(float, libmp.to_float)

    def __complex__(self):
        return self.cast(complex, libmp.to_float)

    def __hash__(self):
        a, b = self._mpi_
        if a == b:
            return mpf_hash(a)
        else:
            return hash(self._mpi_)

    @property
    def real(self): return self

    @property
    def imag(self): return self.ctx.zero

    def conjugate(self): return self

    @property
    def a(self):
        a, b = self._mpi_
        return self.ctx.make_mpf((a, a))

    @property
    def b(self):
        a, b = self._mpi_
        return self.ctx.make_mpf((b, b))

    @property
    def mid(self):
        ctx = self.ctx
        v = mpi_mid(self._mpi_, ctx.prec)
        return ctx.make_mpf((v, v))

    @property
    def delta(self):
        ctx = self.ctx
        v = mpi_delta(self._mpi_, ctx.prec)
        return ctx.make_mpf((v,v))

    @property
    def _mpci_(self):
        return self._mpi_, mpi_zero

    def _compare(*args):
        raise TypeError("no ordering relation is defined for intervals")

    __gt__ = _compare
    __le__ = _compare
    __gt__ = _compare
    __ge__ = _compare

    def __contains__(self, t):
        t = self.ctx.mpf(t)
        return (self.a <= t.a) and (t.b <= self.b)

    def __str__(self):
        return mpi_str(self._mpi_, self.ctx.prec)

    def __repr__(self):
        if self.ctx.pretty:
            return str(self)
        a, b = self._mpi_
        n = repr_dps(self.ctx.prec)
        a = libmp.to_str(a, n)
        b = libmp.to_str(b, n)
        return "mpi(%r, %r)" % (a, b)

    def _compare(s, t, cmpfun):
        if not hasattr(t, "_mpi_"):
            try:
                t = s.ctx.convert(t)
            except:
                return NotImplemented
        return cmpfun(s._mpi_, t._mpi_)

    def __eq__(s, t): return s._compare(t, libmp.mpi_eq)
    def __ne__(s, t): return s._compare(t, libmp.mpi_ne)
    def __lt__(s, t): return s._compare(t, libmp.mpi_lt)
    def __le__(s, t): return s._compare(t, libmp.mpi_le)
    def __gt__(s, t): return s._compare(t, libmp.mpi_gt)
    def __ge__(s, t): return s._compare(t, libmp.mpi_ge)

    def __abs__(self):
        return self.ctx.make_mpf(mpi_abs(self._mpi_, self.ctx.prec))
    def __pos__(self):
        return self.ctx.make_mpf(mpi_pos(self._mpi_, self.ctx.prec))
    def __neg__(self):
        return self.ctx.make_mpf(mpi_neg(self._mpi_, self.ctx.prec))

    def ae(s, t, rel_eps=None, abs_eps=None):
        return s.ctx.almosteq(s, t, rel_eps, abs_eps)

class ivmpc(object):

    def __new__(cls, re=0, im=0):
        re = cls.ctx.convert(re)
        im = cls.ctx.convert(im)
        y = new(cls)
        y._mpci_ = re._mpi_, im._mpi_
        return y

    def __hash__(self):
        (a, b), (c,d) = self._mpci_
        if a == b and c == d:
            return mpc_hash((a, c))
        else:
            return hash(self._mpci_)

    def __repr__(s):
        if s.ctx.pretty:
            return str(s)
        return "iv.mpc(%s, %s)" % (repr(s.real), repr(s.imag))

    def __str__(s):
        return "(%s + %s*j)" % (str(s.real), str(s.imag))

    @property
    def a(self):
        (a, b), (c,d) = self._mpci_
        return self.ctx.make_mpf((a, a))

    @property
    def b(self):
        (a, b), (c,d) = self._mpci_
        return self.ctx.make_mpf((b, b))

    @property
    def c(self):
        (a, b), (c,d) = self._mpci_
        return self.ctx.make_mpf((c, c))

    @property
    def d(self):
        (a, b), (c,d) = self._mpci_
        return self.ctx.make_mpf((d, d))

    @property
    def real(s):
        return s.ctx.make_mpf(s._mpci_[0])

    @property
    def imag(s):
        return s.ctx.make_mpf(s._mpci_[1])

    def conjugate(s):
        a, b = s._mpci_
        return s.ctx.make_mpc((a, mpf_neg(b)))

    def overlap(s, t):
        t = s.ctx.convert(t)
        real_overlap = (s.a <= t.a <= s.b) or (s.a <= t.b <= s.b) or (t.a <= s.a <= t.b) or (t.a <= s.b <= t.b)
        imag_overlap = (s.c <= t.c <= s.d) or (s.c <= t.d <= s.d) or (t.c <= s.c <= t.d) or (t.c <= s.d <= t.d)
        return real_overlap and imag_overlap

    def __contains__(s, t):
        t = s.ctx.convert(t)
        return t.real in s.real and t.imag in s.imag

    def _compare(s, t, ne=False):
        if not isinstance(t, s.ctx._types):
            try:
                t = s.ctx.convert(t)
            except:
                return NotImplemented
        if hasattr(t, '_mpi_'):
            tval = t._mpi_, mpi_zero
        elif hasattr(t, '_mpci_'):
            tval = t._mpci_
        if ne:
            return s._mpci_ != tval
        return s._mpci_ == tval

    def __eq__(s, t): return s._compare(t)
    def __ne__(s, t): return s._compare(t, True)

    def __lt__(s, t): raise TypeError("complex intervals cannot be ordered")
    __le__ = __gt__ = __ge__ = __lt__

    def __neg__(s): return s.ctx.make_mpc(mpci_neg(s._mpci_, s.ctx.prec))
    def __pos__(s): return s.ctx.make_mpc(mpci_pos(s._mpci_, s.ctx.prec))
    def __abs__(s): return s.ctx.make_mpf(mpci_abs(s._mpci_, s.ctx.prec))

    def ae(s, t, rel_eps=None, abs_eps=None):
        return s.ctx.almosteq(s, t, rel_eps, abs_eps)

def _binary_op(f_real, f_complex):
    def g_complex(ctx, sval, tval):
        return ctx.make_mpc(f_complex(sval, tval, ctx.prec))
    def g_real(ctx, sval, tval):
        try:
            return ctx.make_mpf(f_real(sval, tval, ctx.prec))
        except ComplexResult:
            sval = (sval, mpi_zero)
            tval = (tval, mpi_zero)
            return g_complex(ctx, sval, tval)
    def lop_real(s, t):
        if isinstance(t, _matrix): return NotImplemented
        ctx = s.ctx
        if not isinstance(t, ctx._types): t = ctx.convert(t)
        if hasattr(t, "_mpi_"): return g_real(ctx, s._mpi_, t._mpi_)
        if hasattr(t, "_mpci_"): return g_complex(ctx, (s._mpi_, mpi_zero), t._mpci_)
        return NotImplemented
    def rop_real(s, t):
        ctx = s.ctx
        if not isinstance(t, ctx._types): t = ctx.convert(t)
        if hasattr(t, "_mpi_"): return g_real(ctx, t._mpi_, s._mpi_)
        if hasattr(t, "_mpci_"): return g_complex(ctx, t._mpci_, (s._mpi_, mpi_zero))
        return NotImplemented
    def lop_complex(s, t):
        if isinstance(t, _matrix): return NotImplemented
        ctx = s.ctx
        if not isinstance(t, s.ctx._types):
            try:
                t = s.ctx.convert(t)
            except (ValueError, TypeError):
                return NotImplemented
        return g_complex(ctx, s._mpci_, t._mpci_)
    def rop_complex(s, t):
        ctx = s.ctx
        if not isinstance(t, s.ctx._types):
            t = s.ctx.convert(t)
        return g_complex(ctx, t._mpci_, s._mpci_)
    return lop_real, rop_real, lop_complex, rop_complex

ivmpf.__add__, ivmpf.__radd__, ivmpc.__add__, ivmpc.__radd__ = _binary_op(mpi_add, mpci_add)
ivmpf.__sub__, ivmpf.__rsub__, ivmpc.__sub__, ivmpc.__rsub__ = _binary_op(mpi_sub, mpci_sub)
ivmpf.__mul__, ivmpf.__rmul__, ivmpc.__mul__, ivmpc.__rmul__ = _binary_op(mpi_mul, mpci_mul)
ivmpf.__div__, ivmpf.__rdiv__, ivmpc.__div__, ivmpc.__rdiv__ = _binary_op(mpi_div, mpci_div)
ivmpf.__pow__, ivmpf.__rpow__, ivmpc.__pow__, ivmpc.__rpow__ = _binary_op(mpi_pow, mpci_pow)

ivmpf.__truediv__ = ivmpf.__div__; ivmpf.__rtruediv__ = ivmpf.__rdiv__
ivmpc.__truediv__ = ivmpc.__div__; ivmpc.__rtruediv__ = ivmpc.__rdiv__

class ivmpf_constant(ivmpf):
    def __new__(cls, f):
        self = new(cls)
        self._f = f
        return self
    def _get_mpi_(self):
        prec = self.ctx._prec[0]
        a = self._f(prec, round_floor)
        b = self._f(prec, round_ceiling)
        return a, b
    _mpi_ = property(_get_mpi_)

class MPIntervalContext(StandardBaseContext):

    def __init__(ctx):
        ctx.mpf = type('ivmpf', (ivmpf,), {})
        ctx.mpc = type('ivmpc', (ivmpc,), {})
        ctx._types = (ctx.mpf, ctx.mpc)
        ctx._constant = type('ivmpf_constant', (ivmpf_constant,), {})
        ctx._prec = [53]
        ctx._set_prec(53)
        ctx._constant._ctxdata = ctx.mpf._ctxdata = ctx.mpc._ctxdata = [ctx.mpf, new, ctx._prec]
        ctx._constant.ctx = ctx.mpf.ctx = ctx.mpc.ctx = ctx
        ctx.pretty = False
        StandardBaseContext.__init__(ctx)
        ctx._init_builtins()

    def _mpi(ctx, a, b=None):
        if b is None:
            return ctx.mpf(a)
        return ctx.mpf((a,b))

    def _init_builtins(ctx):
        ctx.one = ctx.mpf(1)
        ctx.zero = ctx.mpf(0)
        ctx.inf = ctx.mpf('inf')
        ctx.ninf = -ctx.inf
        ctx.nan = ctx.mpf('nan')
        ctx.j = ctx.mpc(0,1)
        ctx.exp = ctx._wrap_mpi_function(libmp.mpi_exp, libmp.mpci_exp)
        ctx.sqrt = ctx._wrap_mpi_function(libmp.mpi_sqrt)
        ctx.ln = ctx._wrap_mpi_function(libmp.mpi_log, libmp.mpci_log)
        ctx.cos = ctx._wrap_mpi_function(libmp.mpi_cos, libmp.mpci_cos)
        ctx.sin = ctx._wrap_mpi_function(libmp.mpi_sin, libmp.mpci_sin)
        ctx.tan = ctx._wrap_mpi_function(libmp.mpi_tan)
        ctx.gamma = ctx._wrap_mpi_function(libmp.mpi_gamma, libmp.mpci_gamma)
        ctx.loggamma = ctx._wrap_mpi_function(libmp.mpi_loggamma, libmp.mpci_loggamma)
        ctx.rgamma = ctx._wrap_mpi_function(libmp.mpi_rgamma, libmp.mpci_rgamma)
        ctx.factorial = ctx._wrap_mpi_function(libmp.mpi_factorial, libmp.mpci_factorial)
        ctx.fac = ctx.factorial

        ctx.eps = ctx._constant(lambda prec, rnd: (0, MPZ_ONE, 1-prec, 1))
        ctx.pi = ctx._constant(libmp.mpf_pi)
        ctx.e = ctx._constant(libmp.mpf_e)
        ctx.ln2 = ctx._constant(libmp.mpf_ln2)
        ctx.ln10 = ctx._constant(libmp.mpf_ln10)
        ctx.phi = ctx._constant(libmp.mpf_phi)
        ctx.euler = ctx._constant(libmp.mpf_euler)
        ctx.catalan = ctx._constant(libmp.mpf_catalan)
        ctx.glaisher = ctx._constant(libmp.mpf_glaisher)
        ctx.khinchin = ctx._constant(libmp.mpf_khinchin)
        ctx.twinprime = ctx._constant(libmp.mpf_twinprime)

    def _wrap_mpi_function(ctx, f_real, f_complex=None):
        def g(x, **kwargs):
            if kwargs:
                prec = kwargs.get('prec', ctx._prec[0])
            else:
                prec = ctx._prec[0]
            x = ctx.convert(x)
            if hasattr(x, "_mpi_"):
                return ctx.make_mpf(f_real(x._mpi_, prec))
            if hasattr(x, "_mpci_"):
                return ctx.make_mpc(f_complex(x._mpci_, prec))
            raise ValueError
        return g

    @classmethod
    def _wrap_specfun(cls, name, f, wrap):
        if wrap:
            def f_wrapped(ctx, *args, **kwargs):
                convert = ctx.convert
                args = [convert(a) for a in args]
                prec = ctx.prec
                try:
                    ctx.prec += 10
                    retval = f(ctx, *args, **kwargs)
                finally:
                    ctx.prec = prec
                return +retval
        else:
            f_wrapped = f
        setattr(cls, name, f_wrapped)

    def _set_prec(ctx, n):
        ctx._prec[0] = max(1, int(n))
        ctx._dps = prec_to_dps(n)

    def _set_dps(ctx, n):
        ctx._prec[0] = dps_to_prec(n)
        ctx._dps = max(1, int(n))

    prec = property(lambda ctx: ctx._prec[0], _set_prec)
    dps = property(lambda ctx: ctx._dps, _set_dps)

    def make_mpf(ctx, v):
        a = new(ctx.mpf)
        a._mpi_ = v
        return a

    def make_mpc(ctx, v):
        a = new(ctx.mpc)
        a._mpci_ = v
        return a

    def _mpq(ctx, pq):
        p, q = pq
        a = libmp.from_rational(p, q, ctx.prec, round_floor)
        b = libmp.from_rational(p, q, ctx.prec, round_ceiling)
        return ctx.make_mpf((a, b))

    def convert(ctx, x):
        if isinstance(x, (ctx.mpf, ctx.mpc)):
            return x
        if isinstance(x, ctx._constant):
            return +x
        if isinstance(x, complex) or hasattr(x, "_mpc_"):
            re = ctx.convert(x.real)
            im = ctx.convert(x.imag)
            return ctx.mpc(re,im)
        if isinstance(x, basestring):
            v = mpi_from_str(x, ctx.prec)
            return ctx.make_mpf(v)
        if hasattr(x, "_mpi_"):
            a, b = x._mpi_
        else:
            try:
                a, b = x
            except (TypeError, ValueError):
                a = b = x
            if hasattr(a, "_mpi_"):
                a = a._mpi_[0]
            else:
                a = convert_mpf_(a, ctx.prec, round_floor)
            if hasattr(b, "_mpi_"):
                b = b._mpi_[1]
            else:
                b = convert_mpf_(b, ctx.prec, round_ceiling)
        if a == fnan or b == fnan:
            a = fninf
            b = finf
        assert mpf_le(a, b), "endpoints must be properly ordered"
        return ctx.make_mpf((a, b))

    def nstr(ctx, x, n=5, **kwargs):
        x = ctx.convert(x)
        if hasattr(x, "_mpi_"):
            return libmp.mpi_to_str(x._mpi_, n, **kwargs)
        if hasattr(x, "_mpci_"):
            re = libmp.mpi_to_str(x._mpci_[0], n, **kwargs)
            im = libmp.mpi_to_str(x._mpci_[1], n, **kwargs)
            return "(%s + %s*j)" % (re, im)

    def mag(ctx, x):
        x = ctx.convert(x)
        if isinstance(x, ctx.mpc):
            return max(ctx.mag(x.real), ctx.mag(x.imag)) + 1
        a, b = libmp.mpi_abs(x._mpi_)
        sign, man, exp, bc = b
        if man:
            return exp+bc
        if b == fzero:
            return ctx.ninf
        if b == fnan:
            return ctx.nan
        return ctx.inf

    def isnan(ctx, x):
        return False

    def isinf(ctx, x):
        return x == ctx.inf

    def isint(ctx, x):
        x = ctx.convert(x)
        a, b = x._mpi_
        if a == b:
            sign, man, exp, bc = a
            if man:
                return exp >= 0
            return a == fzero
        return None

    def ldexp(ctx, x, n):
        a, b = ctx.convert(x)._mpi_
        a = libmp.mpf_shift(a, n)
        b = libmp.mpf_shift(b, n)
        return ctx.make_mpf((a,b))

    def absmin(ctx, x):
        return abs(ctx.convert(x)).a

    def absmax(ctx, x):
        return abs(ctx.convert(x)).b

    def atan2(ctx, y, x):
        y = ctx.convert(y)._mpi_
        x = ctx.convert(x)._mpi_
        return ctx.make_mpf(libmp.mpi_atan2(y,x,ctx.prec))

    def _convert_param(ctx, x):
        if isinstance(x, libmp.int_types):
            return x, 'Z'
        if isinstance(x, tuple):
            p, q = x
            return (ctx.mpf(p) / ctx.mpf(q), 'R')
        x = ctx.convert(x)
        if isinstance(x, ctx.mpf):
            return x, 'R'
        if isinstance(x, ctx.mpc):
            return x, 'C'
        raise ValueError

    def _is_real_type(ctx, z):
        return isinstance(z, ctx.mpf) or isinstance(z, int_types)

    def _is_complex_type(ctx, z):
        return isinstance(z, ctx.mpc)

    def hypsum(ctx, p, q, types, coeffs, z, maxterms=6000, **kwargs):
        coeffs = list(coeffs)
        num = range(p)
        den = range(p,p+q)
        #tol = ctx.eps
        s = t = ctx.one
        k = 0
        while 1:
            for i in num: t *= (coeffs[i]+k)
            for i in den: t /= (coeffs[i]+k)
            k += 1; t /= k; t *= z; s += t
            if t == 0:
                return s
            #if abs(t) < tol:
            #    return s
            if k > maxterms:
                raise ctx.NoConvergence


# Register with "numbers" ABC
#     We do not subclass, hence we do not use the @abstractmethod checks. While
#     this is less invasive it may turn out that we do not actually support
#     parts of the expected interfaces.  See
#     http://docs.python.org/2/library/numbers.html for list of abstract
#     methods.
try:
    import numbers
    numbers.Complex.register(ivmpc)
    numbers.Real.register(ivmpf)
except ImportError:
    pass

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\combinatorics\homomorphisms.py ===
import itertools
from sympy.combinatorics.fp_groups import FpGroup, FpSubgroup, simplify_presentation
from sympy.combinatorics.free_groups import FreeGroup
from sympy.combinatorics.perm_groups import PermutationGroup
from sympy.core.intfunc import igcd
from sympy.functions.combinatorial.numbers import totient
from sympy.core.singleton import S

class GroupHomomorphism:
    '''
    A class representing group homomorphisms. Instantiate using `homomorphism()`.

    References
    ==========

    .. [1] Holt, D., Eick, B. and O'Brien, E. (2005). Handbook of computational group theory.

    '''

    def __init__(self, domain, codomain, images):
        self.domain = domain
        self.codomain = codomain
        self.images = images
        self._inverses = None
        self._kernel = None
        self._image = None

    def _invs(self):
        '''
        Return a dictionary with `{gen: inverse}` where `gen` is a rewriting
        generator of `codomain` (e.g. strong generator for permutation groups)
        and `inverse` is an element of its preimage

        '''
        image = self.image()
        inverses = {}
        for k in list(self.images.keys()):
            v = self.images[k]
            if not (v in inverses
                    or v.is_identity):
                inverses[v] = k
        if isinstance(self.codomain, PermutationGroup):
            gens = image.strong_gens
        else:
            gens = image.generators
        for g in gens:
            if g in inverses or g.is_identity:
                continue
            w = self.domain.identity
            if isinstance(self.codomain, PermutationGroup):
                parts = image._strong_gens_slp[g][::-1]
            else:
                parts = g
            for s in parts:
                if s in inverses:
                    w = w*inverses[s]
                else:
                    w = w*inverses[s**-1]**-1
            inverses[g] = w

        return inverses

    def invert(self, g):
        '''
        Return an element of the preimage of ``g`` or of each element
        of ``g`` if ``g`` is a list.

        Explanation
        ===========

        If the codomain is an FpGroup, the inverse for equal
        elements might not always be the same unless the FpGroup's
        rewriting system is confluent. However, making a system
        confluent can be time-consuming. If it's important, try
        `self.codomain.make_confluent()` first.

        '''
        from sympy.combinatorics import Permutation
        from sympy.combinatorics.free_groups import FreeGroupElement
        if isinstance(g, (Permutation, FreeGroupElement)):
            if isinstance(self.codomain, FpGroup):
                g = self.codomain.reduce(g)
            if self._inverses is None:
                self._inverses = self._invs()
            image = self.image()
            w = self.domain.identity
            if isinstance(self.codomain, PermutationGroup):
                gens = image.generator_product(g)[::-1]
            else:
                gens = g
            # the following can't be "for s in gens:"
            # because that would be equivalent to
            # "for s in gens.array_form:" when g is
            # a FreeGroupElement. On the other hand,
            # when you call gens by index, the generator
            # (or inverse) at position i is returned.
            for i in range(len(gens)):
                s = gens[i]
                if s.is_identity:
                    continue
                if s in self._inverses:
                    w = w*self._inverses[s]
                else:
                    w = w*self._inverses[s**-1]**-1
            return w
        elif isinstance(g, list):
            return [self.invert(e) for e in g]

    def kernel(self):
        '''
        Compute the kernel of `self`.

        '''
        if self._kernel is None:
            self._kernel = self._compute_kernel()
        return self._kernel

    def _compute_kernel(self):
        G = self.domain
        G_order = G.order()
        if G_order is S.Infinity:
            raise NotImplementedError(
                "Kernel computation is not implemented for infinite groups")
        gens = []
        if isinstance(G, PermutationGroup):
            K = PermutationGroup(G.identity)
        else:
            K = FpSubgroup(G, gens, normal=True)
        i = self.image().order()
        while K.order()*i != G_order:
            r = G.random()
            k = r*self.invert(self(r))**-1
            if k not in K:
                gens.append(k)
                if isinstance(G, PermutationGroup):
                    K = PermutationGroup(gens)
                else:
                    K = FpSubgroup(G, gens, normal=True)
        return K

    def image(self):
        '''
        Compute the image of `self`.

        '''
        if self._image is None:
            values = list(set(self.images.values()))
            if isinstance(self.codomain, PermutationGroup):
                self._image = self.codomain.subgroup(values)
            else:
                self._image = FpSubgroup(self.codomain, values)
        return self._image

    def _apply(self, elem):
        '''
        Apply `self` to `elem`.

        '''
        if elem not in self.domain:
            if isinstance(elem, (list, tuple)):
                return [self._apply(e) for e in elem]
            raise ValueError("The supplied element does not belong to the domain")
        if elem.is_identity:
            return self.codomain.identity
        else:
            images = self.images
            value = self.codomain.identity
            if isinstance(self.domain, PermutationGroup):
                gens = self.domain.generator_product(elem, original=True)
                for g in gens:
                    if g in self.images:
                        value = images[g]*value
                    else:
                        value = images[g**-1]**-1*value
            else:
                i = 0
                for _, p in elem.array_form:
                    if p < 0:
                        g = elem[i]**-1
                    else:
                        g = elem[i]
                    value = value*images[g]**p
                    i += abs(p)
        return value

    def __call__(self, elem):
        return self._apply(elem)

    def is_injective(self):
        '''
        Check if the homomorphism is injective

        '''
        return self.kernel().order() == 1

    def is_surjective(self):
        '''
        Check if the homomorphism is surjective

        '''
        im = self.image().order()
        oth = self.codomain.order()
        if im is S.Infinity and oth is S.Infinity:
            return None
        else:
            return im == oth

    def is_isomorphism(self):
        '''
        Check if `self` is an isomorphism.

        '''
        return self.is_injective() and self.is_surjective()

    def is_trivial(self):
        '''
        Check is `self` is a trivial homomorphism, i.e. all elements
        are mapped to the identity.

        '''
        return self.image().order() == 1

    def compose(self, other):
        '''
        Return the composition of `self` and `other`, i.e.
        the homomorphism phi such that for all g in the domain
        of `other`, phi(g) = self(other(g))

        '''
        if not other.image().is_subgroup(self.domain):
            raise ValueError("The image of `other` must be a subgroup of "
                    "the domain of `self`")
        images = {g: self(other(g)) for g in other.images}
        return GroupHomomorphism(other.domain, self.codomain, images)

    def restrict_to(self, H):
        '''
        Return the restriction of the homomorphism to the subgroup `H`
        of the domain.

        '''
        if not isinstance(H, PermutationGroup) or not H.is_subgroup(self.domain):
            raise ValueError("Given H is not a subgroup of the domain")
        domain = H
        images = {g: self(g) for g in H.generators}
        return GroupHomomorphism(domain, self.codomain, images)

    def invert_subgroup(self, H):
        '''
        Return the subgroup of the domain that is the inverse image
        of the subgroup ``H`` of the homomorphism image

        '''
        if not H.is_subgroup(self.image()):
            raise ValueError("Given H is not a subgroup of the image")
        gens = []
        P = PermutationGroup(self.image().identity)
        for h in H.generators:
            h_i = self.invert(h)
            if h_i not in P:
                gens.append(h_i)
                P = PermutationGroup(gens)
            for k in self.kernel().generators:
                if k*h_i not in P:
                    gens.append(k*h_i)
                    P = PermutationGroup(gens)
        return P

def homomorphism(domain, codomain, gens, images=(), check=True):
    '''
    Create (if possible) a group homomorphism from the group ``domain``
    to the group ``codomain`` defined by the images of the domain's
    generators ``gens``. ``gens`` and ``images`` can be either lists or tuples
    of equal sizes. If ``gens`` is a proper subset of the group's generators,
    the unspecified generators will be mapped to the identity. If the
    images are not specified, a trivial homomorphism will be created.

    If the given images of the generators do not define a homomorphism,
    an exception is raised.

    If ``check`` is ``False``, do not check whether the given images actually
    define a homomorphism.

    '''
    if not isinstance(domain, (PermutationGroup, FpGroup, FreeGroup)):
        raise TypeError("The domain must be a group")
    if not isinstance(codomain, (PermutationGroup, FpGroup, FreeGroup)):
        raise TypeError("The codomain must be a group")

    generators = domain.generators
    if not all(g in generators for g in gens):
        raise ValueError("The supplied generators must be a subset of the domain's generators")
    if not all(g in codomain for g in images):
        raise ValueError("The images must be elements of the codomain")

    if images and len(images) != len(gens):
        raise ValueError("The number of images must be equal to the number of generators")

    gens = list(gens)
    images = list(images)

    images.extend([codomain.identity]*(len(generators)-len(images)))
    gens.extend([g for g in generators if g not in gens])
    images = dict(zip(gens,images))

    if check and not _check_homomorphism(domain, codomain, images):
        raise ValueError("The given images do not define a homomorphism")
    return GroupHomomorphism(domain, codomain, images)

def _check_homomorphism(domain, codomain, images):
    """
    Check that a given mapping of generators to images defines a homomorphism.

    Parameters
    ==========
    domain : PermutationGroup, FpGroup, FreeGroup
    codomain : PermutationGroup, FpGroup, FreeGroup
    images : dict
        The set of keys must be equal to domain.generators.
        The values must be elements of the codomain.

    """
    pres = domain if hasattr(domain, 'relators') else domain.presentation()
    rels = pres.relators
    gens = pres.generators
    symbols = [g.ext_rep[0] for g in gens]
    symbols_to_domain_generators = dict(zip(symbols, domain.generators))
    identity = codomain.identity

    def _image(r):
        w = identity
        for symbol, power in r.array_form:
            g = symbols_to_domain_generators[symbol]
            w *= images[g]**power
        return w

    for r in rels:
        if isinstance(codomain, FpGroup):
            s = codomain.equals(_image(r), identity)
            if s is None:
                # only try to make the rewriting system
                # confluent when it can't determine the
                # truth of equality otherwise
                success = codomain.make_confluent()
                s = codomain.equals(_image(r), identity)
                if s is None and not success:
                    raise RuntimeError("Can't determine if the images "
                        "define a homomorphism. Try increasing "
                        "the maximum number of rewriting rules "
                        "(group._rewriting_system.set_max(new_value); "
                        "the current value is stored in group._rewriting"
                        "_system.maxeqns)")
        else:
            s = _image(r).is_identity
        if not s:
            return False
    return True

def orbit_homomorphism(group, omega):
    '''
    Return the homomorphism induced by the action of the permutation
    group ``group`` on the set ``omega`` that is closed under the action.

    '''
    from sympy.combinatorics import Permutation
    from sympy.combinatorics.named_groups import SymmetricGroup
    codomain = SymmetricGroup(len(omega))
    identity = codomain.identity
    omega = list(omega)
    images = {g: identity*Permutation([omega.index(o^g) for o in omega]) for g in group.generators}
    group._schreier_sims(base=omega)
    H = GroupHomomorphism(group, codomain, images)
    if len(group.basic_stabilizers) > len(omega):
        H._kernel = group.basic_stabilizers[len(omega)]
    else:
        H._kernel = PermutationGroup([group.identity])
    return H

def block_homomorphism(group, blocks):
    '''
    Return the homomorphism induced by the action of the permutation
    group ``group`` on the block system ``blocks``. The latter should be
    of the same form as returned by the ``minimal_block`` method for
    permutation groups, namely a list of length ``group.degree`` where
    the i-th entry is a representative of the block i belongs to.

    '''
    from sympy.combinatorics import Permutation
    from sympy.combinatorics.named_groups import SymmetricGroup

    n = len(blocks)

    # number the blocks; m is the total number,
    # b is such that b[i] is the number of the block i belongs to,
    # p is the list of length m such that p[i] is the representative
    # of the i-th block
    m = 0
    p = []
    b = [None]*n
    for i in range(n):
        if blocks[i] == i:
            p.append(i)
            b[i] = m
            m += 1
    for i in range(n):
        b[i] = b[blocks[i]]

    codomain = SymmetricGroup(m)
    # the list corresponding to the identity permutation in codomain
    identity = range(m)
    images = {g: Permutation([b[p[i]^g] for i in identity]) for g in group.generators}
    H = GroupHomomorphism(group, codomain, images)
    return H

def group_isomorphism(G, H, isomorphism=True):
    '''
    Compute an isomorphism between 2 given groups.

    Parameters
    ==========

    G : A finite ``FpGroup`` or a ``PermutationGroup``.
        First group.

    H : A finite ``FpGroup`` or a ``PermutationGroup``
        Second group.

    isomorphism : bool
        This is used to avoid the computation of homomorphism
        when the user only wants to check if there exists
        an isomorphism between the groups.

    Returns
    =======

    If isomorphism = False -- Returns a boolean.
    If isomorphism = True  -- Returns a boolean and an isomorphism between `G` and `H`.

    Examples
    ========

    >>> from sympy.combinatorics import free_group, Permutation
    >>> from sympy.combinatorics.perm_groups import PermutationGroup
    >>> from sympy.combinatorics.fp_groups import FpGroup
    >>> from sympy.combinatorics.homomorphisms import group_isomorphism
    >>> from sympy.combinatorics.named_groups import DihedralGroup, AlternatingGroup

    >>> D = DihedralGroup(8)
    >>> p = Permutation(0, 1, 2, 3, 4, 5, 6, 7)
    >>> P = PermutationGroup(p)
    >>> group_isomorphism(D, P)
    (False, None)

    >>> F, a, b = free_group("a, b")
    >>> G = FpGroup(F, [a**3, b**3, (a*b)**2])
    >>> H = AlternatingGroup(4)
    >>> (check, T) = group_isomorphism(G, H)
    >>> check
    True
    >>> T(b*a*b**-1*a**-1*b**-1)
    (0 2 3)

    Notes
    =====

    Uses the approach suggested by Robert Tarjan to compute the isomorphism between two groups.
    First, the generators of ``G`` are mapped to the elements of ``H`` and
    we check if the mapping induces an isomorphism.

    '''
    if not isinstance(G, (PermutationGroup, FpGroup)):
        raise TypeError("The group must be a PermutationGroup or an FpGroup")
    if not isinstance(H, (PermutationGroup, FpGroup)):
        raise TypeError("The group must be a PermutationGroup or an FpGroup")

    if isinstance(G, FpGroup) and isinstance(H, FpGroup):
        G = simplify_presentation(G)
        H = simplify_presentation(H)
        # Two infinite FpGroups with the same generators are isomorphic
        # when the relators are same but are ordered differently.
        if G.generators == H.generators and (G.relators).sort() == (H.relators).sort():
            if not isomorphism:
                return True
            return (True, homomorphism(G, H, G.generators, H.generators))

    #  `_H` is the permutation group isomorphic to `H`.
    _H = H
    g_order = G.order()
    h_order = H.order()

    if g_order is S.Infinity:
        raise NotImplementedError("Isomorphism methods are not implemented for infinite groups.")

    if isinstance(H, FpGroup):
        if h_order is S.Infinity:
            raise NotImplementedError("Isomorphism methods are not implemented for infinite groups.")
        _H, h_isomorphism = H._to_perm_group()

    if (g_order != h_order) or (G.is_abelian != H.is_abelian):
        if not isomorphism:
            return False
        return (False, None)

    if not isomorphism:
        # Two groups of the same cyclic numbered order
        # are isomorphic to each other.
        n = g_order
        if (igcd(n, totient(n))) == 1:
            return True

    # Match the generators of `G` with subsets of `_H`
    gens = list(G.generators)
    for subset in itertools.permutations(_H, len(gens)):
        images = list(subset)
        images.extend([_H.identity]*(len(G.generators)-len(images)))
        _images = dict(zip(gens,images))
        if _check_homomorphism(G, _H, _images):
            if isinstance(H, FpGroup):
                images = h_isomorphism.invert(images)
            T =  homomorphism(G, H, G.generators, images, check=False)
            if T.is_isomorphism():
                # It is a valid isomorphism
                if not isomorphism:
                    return True
                return (True, T)

    if not isomorphism:
        return False
    return (False, None)

def is_isomorphic(G, H):
    '''
    Check if the groups are isomorphic to each other

    Parameters
    ==========

    G : A finite ``FpGroup`` or a ``PermutationGroup``
        First group.

    H : A finite ``FpGroup`` or a ``PermutationGroup``
        Second group.

    Returns
    =======

    boolean
    '''
    return group_isomorphism(G, H, isomorphism=False)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\glsl.py ===
from __future__ import annotations

from sympy.core import Basic, S
from sympy.core.function import Lambda
from sympy.core.numbers import equal_valued
from sympy.printing.codeprinter import CodePrinter
from sympy.printing.precedence import precedence
from functools import reduce

known_functions = {
    'Abs': 'abs',
    'sin': 'sin',
    'cos': 'cos',
    'tan': 'tan',
    'acos': 'acos',
    'asin': 'asin',
    'atan': 'atan',
    'atan2': 'atan',
    'ceiling': 'ceil',
    'floor': 'floor',
    'sign': 'sign',
    'exp': 'exp',
    'log': 'log',
    'add': 'add',
    'sub': 'sub',
    'mul': 'mul',
    'pow': 'pow'
}

class GLSLPrinter(CodePrinter):
    """
    Rudimentary, generic GLSL printing tools.

    Additional settings:
    'use_operators': Boolean (should the printer use operators for +,-,*, or functions?)
    """
    _not_supported: set[Basic] = set()
    printmethod = "_glsl"
    language = "GLSL"

    _default_settings = dict(CodePrinter._default_settings, **{
        'use_operators': True,
        'zero': 0,
        'mat_nested': False,
        'mat_separator': ',\n',
        'mat_transpose': False,
        'array_type': 'float',
        'glsl_types': True,

        'precision': 9,
        'user_functions': {},
        'contract': True,
    })

    def __init__(self, settings={}):
        CodePrinter.__init__(self, settings)
        self.known_functions = dict(known_functions)
        userfuncs = settings.get('user_functions', {})
        self.known_functions.update(userfuncs)

    def _rate_index_position(self, p):
        return p*5

    def _get_statement(self, codestring):
        return "%s;" % codestring

    def _get_comment(self, text):
        return "// {}".format(text)

    def _declare_number_const(self, name, value):
        return "float {} = {};".format(name, value)

    def _format_code(self, lines):
        return self.indent_code(lines)

    def indent_code(self, code):
        """Accepts a string of code or a list of code lines"""

        if isinstance(code, str):
            code_lines = self.indent_code(code.splitlines(True))
            return ''.join(code_lines)

        tab = "   "
        inc_token = ('{', '(', '{\n', '(\n')
        dec_token = ('}', ')')

        code = [line.lstrip(' \t') for line in code]

        increase = [int(any(map(line.endswith, inc_token))) for line in code]
        decrease = [int(any(map(line.startswith, dec_token))) for line in code]

        pretty = []
        level = 0
        for n, line in enumerate(code):
            if line in ('', '\n'):
                pretty.append(line)
                continue
            level -= decrease[n]
            pretty.append("%s%s" % (tab*level, line))
            level += increase[n]
        return pretty

    def _print_MatrixBase(self, mat):
        mat_separator = self._settings['mat_separator']
        mat_transpose = self._settings['mat_transpose']
        column_vector = (mat.rows == 1) if mat_transpose else (mat.cols == 1)
        A = mat.transpose() if mat_transpose != column_vector else mat

        glsl_types = self._settings['glsl_types']
        array_type = self._settings['array_type']
        array_size = A.cols*A.rows
        array_constructor = "{}[{}]".format(array_type, array_size)

        if A.cols == 1:
            return self._print(A[0])
        if A.rows <= 4 and A.cols <= 4 and glsl_types:
            if A.rows == 1:
                return "vec{}{}".format(
                    A.cols, A.table(self,rowstart='(',rowend=')')
                )
            elif A.rows == A.cols:
                return "mat{}({})".format(
                    A.rows, A.table(self,rowsep=', ',
                    rowstart='',rowend='')
                )
            else:
                return "mat{}x{}({})".format(
                    A.cols, A.rows,
                    A.table(self,rowsep=', ',
                    rowstart='',rowend='')
                )
        elif S.One in A.shape:
            return "{}({})".format(
                array_constructor,
                A.table(self,rowsep=mat_separator,rowstart='',rowend='')
            )
        elif not self._settings['mat_nested']:
            return "{}(\n{}\n) /* a {}x{} matrix */".format(
                array_constructor,
                A.table(self,rowsep=mat_separator,rowstart='',rowend=''),
                A.rows, A.cols
            )
        elif self._settings['mat_nested']:
            return "{}[{}][{}](\n{}\n)".format(
                array_type, A.rows, A.cols,
                A.table(self,rowsep=mat_separator,rowstart='float[](',rowend=')')
            )

    def _print_SparseRepMatrix(self, mat):
        # do not allow sparse matrices to be made dense
        return self._print_not_supported(mat)

    def _traverse_matrix_indices(self, mat):
        mat_transpose = self._settings['mat_transpose']
        if mat_transpose:
            rows,cols = mat.shape
        else:
            cols,rows = mat.shape
        return ((i, j) for i in range(cols) for j in range(rows))

    def _print_MatrixElement(self, expr):
        # print('begin _print_MatrixElement')
        nest = self._settings['mat_nested']
        glsl_types = self._settings['glsl_types']
        mat_transpose = self._settings['mat_transpose']
        if mat_transpose:
            cols,rows = expr.parent.shape
            i,j = expr.j,expr.i
        else:
            rows,cols = expr.parent.shape
            i,j = expr.i,expr.j
        pnt = self._print(expr.parent)
        if glsl_types and ((rows <= 4 and cols <=4) or nest):
            return "{}[{}][{}]".format(pnt, i, j)
        else:
            return "{}[{}]".format(pnt, i + j*rows)

    def _print_list(self, expr):
        l = ', '.join(self._print(item) for item in expr)
        glsl_types = self._settings['glsl_types']
        array_type = self._settings['array_type']
        array_size = len(expr)
        array_constructor = '{}[{}]'.format(array_type, array_size)

        if array_size <= 4 and glsl_types:
            return 'vec{}({})'.format(array_size, l)
        else:
            return '{}({})'.format(array_constructor, l)

    _print_tuple = _print_list
    _print_Tuple = _print_list

    def _get_loop_opening_ending(self, indices):
        open_lines = []
        close_lines = []
        loopstart = "for (int %(varble)s=%(start)s; %(varble)s<%(end)s; %(varble)s++){"
        for i in indices:
            # GLSL arrays start at 0 and end at dimension-1
            open_lines.append(loopstart % {
                'varble': self._print(i.label),
                'start': self._print(i.lower),
                'end': self._print(i.upper + 1)})
            close_lines.append("}")
        return open_lines, close_lines

    def _print_Function_with_args(self, func, func_args):
        if func in self.known_functions:
            cond_func = self.known_functions[func]
            func = None
            if isinstance(cond_func, str):
                func = cond_func
            else:
                for cond, func in cond_func:
                    if cond(func_args):
                        break
            if func is not None:
                try:
                    return func(*[self.parenthesize(item, 0) for item in func_args])
                except TypeError:
                    return '{}({})'.format(func, self.stringify(func_args, ", "))
        elif isinstance(func, Lambda):
            # inlined function
            return self._print(func(*func_args))
        else:
            return self._print_not_supported(func)

    def _print_Piecewise(self, expr):
        from sympy.codegen.ast import Assignment
        if expr.args[-1].cond != True:
            # We need the last conditional to be a True, otherwise the resulting
            # function may not return a result.
            raise ValueError("All Piecewise expressions must contain an "
                             "(expr, True) statement to be used as a default "
                             "condition. Without one, the generated "
                             "expression may not evaluate to anything under "
                             "some condition.")
        lines = []
        if expr.has(Assignment):
            for i, (e, c) in enumerate(expr.args):
                if i == 0:
                    lines.append("if (%s) {" % self._print(c))
                elif i == len(expr.args) - 1 and c == True:
                    lines.append("else {")
                else:
                    lines.append("else if (%s) {" % self._print(c))
                code0 = self._print(e)
                lines.append(code0)
                lines.append("}")
            return "\n".join(lines)
        else:
            # The piecewise was used in an expression, need to do inline
            # operators. This has the downside that inline operators will
            # not work for statements that span multiple lines (Matrix or
            # Indexed expressions).
            ecpairs = ["((%s) ? (\n%s\n)\n" % (self._print(c),
                                               self._print(e))
                    for e, c in expr.args[:-1]]
            last_line = ": (\n%s\n)" % self._print(expr.args[-1].expr)
            return ": ".join(ecpairs) + last_line + " ".join([")"*len(ecpairs)])

    def _print_Indexed(self, expr):
        # calculate index for 1d array
        dims = expr.shape
        elem = S.Zero
        offset = S.One
        for i in reversed(range(expr.rank)):
            elem += expr.indices[i]*offset
            offset *= dims[i]
        return "{}[{}]".format(
            self._print(expr.base.label),
            self._print(elem)
        )

    def _print_Pow(self, expr):
        PREC = precedence(expr)
        if equal_valued(expr.exp, -1):
            return '1.0/%s' % (self.parenthesize(expr.base, PREC))
        elif equal_valued(expr.exp, 0.5):
            return 'sqrt(%s)' % self._print(expr.base)
        else:
            try:
                e = self._print(float(expr.exp))
            except TypeError:
                e = self._print(expr.exp)
            return self._print_Function_with_args('pow', (
                self._print(expr.base),
                e
            ))

    def _print_int(self, expr):
        return str(float(expr))

    def _print_Rational(self, expr):
        return "{}.0/{}.0".format(expr.p, expr.q)

    def _print_Relational(self, expr):
        lhs_code = self._print(expr.lhs)
        rhs_code = self._print(expr.rhs)
        op = expr.rel_op
        return "{} {} {}".format(lhs_code, op, rhs_code)

    def _print_Add(self, expr, order=None):
        if self._settings['use_operators']:
            return CodePrinter._print_Add(self, expr, order=order)

        terms = expr.as_ordered_terms()

        def partition(p,l):
            return reduce(lambda x, y: (x[0]+[y], x[1]) if p(y) else (x[0], x[1]+[y]), l,  ([], []))
        def add(a,b):
            return self._print_Function_with_args('add', (a, b))
            # return self.known_functions['add']+'(%s, %s)' % (a,b)
        neg, pos = partition(lambda arg: arg.could_extract_minus_sign(), terms)
        if pos:
            s = pos = reduce(lambda a,b: add(a,b), (self._print(t) for t in pos))
        else:
            s = pos = self._print(self._settings['zero'])

        if neg:
            # sum the absolute values of the negative terms
            neg = reduce(lambda a,b: add(a,b), (self._print(-n) for n in neg))
            # then subtract them from the positive terms
            s = self._print_Function_with_args('sub', (pos,neg))
            # s = self.known_functions['sub']+'(%s, %s)' % (pos,neg)
        return s

    def _print_Mul(self, expr, **kwargs):
        if self._settings['use_operators']:
            return CodePrinter._print_Mul(self, expr, **kwargs)
        terms = expr.as_ordered_factors()
        def mul(a,b):
            # return self.known_functions['mul']+'(%s, %s)' % (a,b)
            return self._print_Function_with_args('mul', (a,b))

        s = reduce(lambda a,b: mul(a,b), (self._print(t) for t in terms))
        return s

def glsl_code(expr,assign_to=None,**settings):
    """Converts an expr to a string of GLSL code

    Parameters
    ==========

    expr : Expr
        A SymPy expression to be converted.
    assign_to : optional
        When given, the argument is used for naming the variable or variables
        to which the expression is assigned. Can be a string, ``Symbol``,
        ``MatrixSymbol`` or ``Indexed`` type object. In cases where ``expr``
        would be printed as an array, a list of string or ``Symbol`` objects
        can also be passed.

        This is helpful in case of line-wrapping, or for expressions that
        generate multi-line statements.  It can also be used to spread an array-like
        expression into multiple assignments.
    use_operators: bool, optional
        If set to False, then *,/,+,- operators will be replaced with functions
        mul, add, and sub, which must be implemented by the user, e.g. for
        implementing non-standard rings or emulated quad/octal precision.
        [default=True]
    glsl_types: bool, optional
        Set this argument to ``False`` in order to avoid using the ``vec`` and ``mat``
        types.  The printer will instead use arrays (or nested arrays).
        [default=True]
    mat_nested: bool, optional
        GLSL version 4.3 and above support nested arrays (arrays of arrays).  Set this to ``True``
        to render matrices as nested arrays.
        [default=False]
    mat_separator: str, optional
        By default, matrices are rendered with newlines using this separator,
        making them easier to read, but less compact.  By removing the newline
        this option can be used to make them more vertically compact.
        [default=',\n']
    mat_transpose: bool, optional
        GLSL's matrix multiplication implementation assumes column-major indexing.
        By default, this printer ignores that convention. Setting this option to
        ``True`` transposes all matrix output.
        [default=False]
    array_type: str, optional
        The GLSL array constructor type.
        [default='float']
    precision : integer, optional
        The precision for numbers such as pi [default=15].
    user_functions : dict, optional
        A dictionary where keys are ``FunctionClass`` instances and values are
        their string representations. Alternatively, the dictionary value can
        be a list of tuples i.e. [(argument_test, js_function_string)]. See
        below for examples.
    human : bool, optional
        If True, the result is a single string that may contain some constant
        declarations for the number symbols. If False, the same information is
        returned in a tuple of (symbols_to_declare, not_supported_functions,
        code_text). [default=True].
    contract: bool, optional
        If True, ``Indexed`` instances are assumed to obey tensor contraction
        rules and the corresponding nested loops over indices are generated.
        Setting contract=False will not generate loops, instead the user is
        responsible to provide values for the indices in the code.
        [default=True].

    Examples
    ========

    >>> from sympy import glsl_code, symbols, Rational, sin, ceiling, Abs
    >>> x, tau = symbols("x, tau")
    >>> glsl_code((2*tau)**Rational(7, 2))
    '8*sqrt(2)*pow(tau, 3.5)'
    >>> glsl_code(sin(x), assign_to="float y")
    'float y = sin(x);'

    Various GLSL types are supported:
    >>> from sympy import Matrix, glsl_code
    >>> glsl_code(Matrix([1,2,3]))
    'vec3(1, 2, 3)'

    >>> glsl_code(Matrix([[1, 2],[3, 4]]))
    'mat2(1, 2, 3, 4)'

    Pass ``mat_transpose = True`` to switch to column-major indexing:
    >>> glsl_code(Matrix([[1, 2],[3, 4]]), mat_transpose = True)
    'mat2(1, 3, 2, 4)'

    By default, larger matrices get collapsed into float arrays:
    >>> print(glsl_code( Matrix([[1,2,3,4,5],[6,7,8,9,10]]) ))
    float[10](
       1, 2, 3, 4,  5,
       6, 7, 8, 9, 10
    ) /* a 2x5 matrix */

    The type of array constructor used to print GLSL arrays can be controlled
    via the ``array_type`` parameter:
    >>> glsl_code(Matrix([1,2,3,4,5]), array_type='int')
    'int[5](1, 2, 3, 4, 5)'

    Passing a list of strings or ``symbols`` to the ``assign_to`` parameter will yield
    a multi-line assignment for each item in an array-like expression:
    >>> x_struct_members = symbols('x.a x.b x.c x.d')
    >>> print(glsl_code(Matrix([1,2,3,4]), assign_to=x_struct_members))
    x.a = 1;
    x.b = 2;
    x.c = 3;
    x.d = 4;

    This could be useful in cases where it's desirable to modify members of a
    GLSL ``Struct``.  It could also be used to spread items from an array-like
    expression into various miscellaneous assignments:
    >>> misc_assignments = ('x[0]', 'x[1]', 'float y', 'float z')
    >>> print(glsl_code(Matrix([1,2,3,4]), assign_to=misc_assignments))
    x[0] = 1;
    x[1] = 2;
    float y = 3;
    float z = 4;

    Passing ``mat_nested = True`` instead prints out nested float arrays, which are
    supported in GLSL 4.3 and above.
    >>> mat = Matrix([
    ... [ 0,  1,  2],
    ... [ 3,  4,  5],
    ... [ 6,  7,  8],
    ... [ 9, 10, 11],
    ... [12, 13, 14]])
    >>> print(glsl_code( mat, mat_nested = True ))
    float[5][3](
       float[]( 0,  1,  2),
       float[]( 3,  4,  5),
       float[]( 6,  7,  8),
       float[]( 9, 10, 11),
       float[](12, 13, 14)
    )



    Custom printing can be defined for certain types by passing a dictionary of
    "type" : "function" to the ``user_functions`` kwarg. Alternatively, the
    dictionary value can be a list of tuples i.e. [(argument_test,
    js_function_string)].

    >>> custom_functions = {
    ...   "ceiling": "CEIL",
    ...   "Abs": [(lambda x: not x.is_integer, "fabs"),
    ...           (lambda x: x.is_integer, "ABS")]
    ... }
    >>> glsl_code(Abs(x) + ceiling(x), user_functions=custom_functions)
    'fabs(x) + CEIL(x)'

    If further control is needed, addition, subtraction, multiplication and
    division operators can be replaced with ``add``, ``sub``, and ``mul``
    functions.  This is done by passing ``use_operators = False``:

    >>> x,y,z = symbols('x,y,z')
    >>> glsl_code(x*(y+z), use_operators = False)
    'mul(x, add(y, z))'
    >>> glsl_code(x*(y+z*(x-y)**z), use_operators = False)
    'mul(x, add(y, mul(z, pow(sub(x, y), z))))'

    ``Piecewise`` expressions are converted into conditionals. If an
    ``assign_to`` variable is provided an if statement is created, otherwise
    the ternary operator is used. Note that if the ``Piecewise`` lacks a
    default term, represented by ``(expr, True)`` then an error will be thrown.
    This is to prevent generating an expression that may not evaluate to
    anything.

    >>> from sympy import Piecewise
    >>> expr = Piecewise((x + 1, x > 0), (x, True))
    >>> print(glsl_code(expr, tau))
    if (x > 0) {
       tau = x + 1;
    }
    else {
       tau = x;
    }

    Support for loops is provided through ``Indexed`` types. With
    ``contract=True`` these expressions will be turned into loops, whereas
    ``contract=False`` will just print the assignment expression that should be
    looped over:

    >>> from sympy import Eq, IndexedBase, Idx
    >>> len_y = 5
    >>> y = IndexedBase('y', shape=(len_y,))
    >>> t = IndexedBase('t', shape=(len_y,))
    >>> Dy = IndexedBase('Dy', shape=(len_y-1,))
    >>> i = Idx('i', len_y-1)
    >>> e=Eq(Dy[i], (y[i+1]-y[i])/(t[i+1]-t[i]))
    >>> glsl_code(e.rhs, assign_to=e.lhs, contract=False)
    'Dy[i] = (y[i + 1] - y[i])/(t[i + 1] - t[i]);'

    >>> from sympy import Matrix, MatrixSymbol
    >>> mat = Matrix([x**2, Piecewise((x + 1, x > 0), (x, True)), sin(x)])
    >>> A = MatrixSymbol('A', 3, 1)
    >>> print(glsl_code(mat, A))
    A[0][0] = pow(x, 2.0);
    if (x > 0) {
       A[1][0] = x + 1;
    }
    else {
       A[1][0] = x;
    }
    A[2][0] = sin(x);
    """
    return GLSLPrinter(settings).doprint(expr,assign_to)

def print_glsl(expr, **settings):
    """Prints the GLSL representation of the given expression.

       See GLSLPrinter init function for settings.
    """
    print(glsl_code(expr, **settings))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\fusion_gpt_attention.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from logging import getLogger

import numpy as np
from fusion_base import Fusion
from fusion_utils import FusionUtils
from onnx import helper
from onnx_model import OnnxModel

logger = getLogger(__name__)


class FusionGptAttentionPastBase(Fusion):
    """Base class for GPT Attention Fusion with past state"""

    def __init__(self, model: OnnxModel, num_heads: int):
        super().__init__(model, "Attention", ["LayerNormalization", "SkipLayerNormalization"], "with past")
        self.num_heads = num_heads
        self.utils = FusionUtils(model)
        self.casted_attention_mask = {}  # map from name of attention mask to the name that casted to int32
        self.mask_filter_value = None

    def match_past_pattern_1(self, concat_k, concat_v, output_name_to_node):
        # Pattern 1:
        #                      {past}
        #                    /        \
        #                   /          \
        #    Gather(axes=0, indices=0)  Gather(indices=1)
        #      |                          |
        #    Transpose (perm=0,1,3,2)     |
        #      |                          |
        #  Concat_k                     Concat_v
        #      |                        /
        #  Transpose (perm=0,1,3,2)    /
        #      |                      /
        #  Unsqueeze        Unsqueeze
        #        \        /
        #         \      /
        #           Concat
        #             |
        #         {present}
        gather = self.model.get_parent(concat_v, 0, output_name_to_node)
        if gather is None or gather.op_type != "Gather":
            logger.debug("match_past_pattern_1: expect Gather for past")
            return None

        if self.model.find_constant_input(gather, 1) != 1:
            logger.debug("match_past_pattern_1: expect indices=1 for Gather of past")
            return None
        past = gather.input[0]

        parent = self.model.get_parent(concat_k, 0, output_name_to_node)
        if parent and parent.op_type == "Gather":
            gather_past_k = parent
        else:
            past_k_nodes = self.model.match_parent_path(concat_k, ["Transpose", "Gather"], [0, 0])
            if past_k_nodes is None:
                logger.debug("match_past_pattern_1: failed match Transpose and Gather")
                return None
            gather_past_k = past_k_nodes[-1]

        if self.model.find_constant_input(gather_past_k, 0) != 1:
            logger.debug("match_past_pattern_1: expect indices=0 for Gather k of past")
            return None
        past_k = gather_past_k.input[0]
        if past != past_k:
            logger.debug("match_past_pattern_1: expect past to be same")
            return None

        return past

    def match_past_pattern_2(self, concat_k, concat_v, output_name_to_node):
        # Pattern 2:
        #      Split (QKV)
        #      / |   |
        #     /  |   +----------------------+
        #        |                          |
        #        |         {past}           |
        #        |           |              |
        #      Reshape     Split         Reshape
        #        |         /    \           |
        # Transpose_k  Squeeze  Squeeze  Transpose_v
        #        |      |        \        /
        #        +------|---+     \      /
        #               |   |      \    /
        #              Concat_k   Concat_v
        #               |            |
        #          Unsqueeze    Unsqueeze
        #                \       /
        #                 Concat
        #                   |
        #               {present}
        #
        squeeze = self.model.get_parent(concat_v, 0, output_name_to_node)
        if squeeze is None or squeeze.op_type != "Squeeze":
            logger.debug("match_past_pattern_2: expect Squeeze as parent of concat_v")
            return None

        split = self.model.get_parent(squeeze, 0, output_name_to_node)
        if split is None or split.op_type != "Split":
            logger.debug("match_past_pattern_2: expect Split for past path")
            return None

        opset_version = self.model.get_opset_version()
        if opset_version < 13:
            if not FusionUtils.check_node_attribute(squeeze, "axes", [0]):
                logger.debug("match_past_pattern_2: axes != [0] for Squeeze in past path")
                return None

            if not FusionUtils.check_node_attribute(split, "split", [1, 1]):
                logger.debug("match_past_pattern_2: split != [1, 1] for Split in past path")
                return None
        else:
            if not self.utils.check_node_input_value(squeeze, 1, [0]):
                logger.debug("match_past_pattern_2: axes != [0] for Squeeze in past path")
                return None

            if not self.utils.check_node_input_value(split, 1, [1, 1]):
                logger.debug("match_past_pattern_2: split != [1, 1] for Split in past path")
                return None

        if not FusionUtils.check_node_attribute(split, "axis", 0, default_value=0):
            logger.debug("match_past_pattern_2: attribute axis of Split are not expected in past path")
            return None
        past = split.input[0]

        past_k_nodes = self.model.match_parent_path(concat_k, ["Squeeze", "Split"], [0, 0])
        if past_k_nodes is None:
            logger.debug("match_past_pattern_2: failed to match past_k_nodes path")
            return None
        past_k = past_k_nodes[-1].input[0]

        if past != past_k:
            logger.info("match_past_pattern_2: expect past to be same")
            return None

        return past

    def match_present(self, concat_v, input_name_to_nodes):
        unsqueeze_present_v = self.model.find_first_child_by_type(
            concat_v, "Unsqueeze", input_name_to_nodes, recursive=False
        )
        if not unsqueeze_present_v:
            logger.info("expect unsqueeze for present")
            return None
        concat_present = self.model.find_first_child_by_type(
            unsqueeze_present_v, "Concat", input_name_to_nodes, recursive=False
        )
        if not concat_present:
            logger.info("expect concat for present")
            return None

        present = concat_present.output[0]
        return present

    def cast_attention_mask(self, input_name):
        if input_name in self.casted_attention_mask:
            attention_mask_input_name = self.casted_attention_mask[input_name]
        elif self.model.find_graph_input(input_name):
            casted, attention_mask_input_name = self.utils.cast_graph_input_to_int32(input_name)
            self.casted_attention_mask[input_name] = attention_mask_input_name
        else:
            attention_mask_input_name, cast_node = self.utils.cast_input_to_int32(input_name)
            self.casted_attention_mask[input_name] = attention_mask_input_name
        return attention_mask_input_name


class FusionGptAttention(FusionGptAttentionPastBase):
    """
    Fuse GPT-2 Attention with past state subgraph into one Attention node.
    """

    def __init__(self, model: OnnxModel, num_heads: int):
        super().__init__(model, num_heads)

    def create_attention_node(
        self,
        fc_weight,
        fc_bias,
        gemm_qkv,
        past,
        present,
        input,
        output,
        mask,
        is_unidirectional,
    ):
        attention_node_name = self.model.create_node_name("GptAttention")
        attention_node = helper.make_node(
            "Attention",
            inputs=[input, fc_weight, fc_bias, mask, past],
            outputs=[attention_node_name + "_output", present],
            name=attention_node_name,
        )
        attention_node.domain = "com.microsoft"
        attention_node.attribute.extend(
            [
                helper.make_attribute("num_heads", self.num_heads),
                helper.make_attribute("unidirectional", 1 if is_unidirectional else 0),
            ]
        )

        if self.mask_filter_value is not None:
            attention_node.attribute.extend([helper.make_attribute("mask_filter_value", float(self.mask_filter_value))])

        matmul_node = helper.make_node(
            "MatMul",
            inputs=[attention_node_name + "_output", gemm_qkv.input[1]],
            outputs=[attention_node_name + "_matmul_output"],
            name=attention_node_name + "_matmul",
        )

        add_node = helper.make_node(
            "Add",
            inputs=[attention_node_name + "_matmul_output", gemm_qkv.input[2]],
            outputs=[output],
            name=attention_node_name + "_add",
        )
        self.nodes_to_add.extend([attention_node, matmul_node, add_node])
        self.node_name_to_graph_name[attention_node.name] = self.this_graph_name
        self.node_name_to_graph_name[matmul_node.name] = self.this_graph_name
        self.node_name_to_graph_name[add_node.name] = self.this_graph_name

    def fuse(self, normalize_node, input_name_to_nodes, output_name_to_node):
        past = None
        present = None
        return_indice = []

        is_normalize_node_skiplayernorm = normalize_node.op_type == "SkipLayerNormalization"
        qkv_nodes = None

        if not is_normalize_node_skiplayernorm:
            qkv_nodes = self.model.match_parent_path(
                normalize_node,
                ["Add", "Reshape", "Gemm", "Reshape", "Reshape", "Transpose", "MatMul"],
                [0, None, 0, 0, 0, 0, 0],
                output_name_to_node=output_name_to_node,
                return_indice=return_indice,
            )
        else:
            qkv_nodes = self.model.match_parent_path(
                normalize_node,
                ["Reshape", "Gemm", "Reshape", "Reshape", "Transpose", "MatMul"],
                [None, 0, 0, 0, 0, 0],
                output_name_to_node=output_name_to_node,
                return_indice=return_indice,
            )

        if qkv_nodes is None:
            return

        another_input = None
        if not is_normalize_node_skiplayernorm:
            (
                add_qkv,
                reshape_qkv,
                gemm_qkv,
                reshape_1,
                reshape_2,
                transpose_qkv,
                matmul_qkv,
            ) = qkv_nodes

            another_input = add_qkv.input[1 - return_indice[0]]
        else:
            (
                reshape_qkv,
                gemm_qkv,
                reshape_1,
                reshape_2,
                transpose_qkv,
                matmul_qkv,
            ) = qkv_nodes

        v_nodes = self.model.match_parent_path(matmul_qkv, ["Concat", "Transpose", "Reshape", "Split"], [1, 1, 0, 0])
        if v_nodes is None:
            logger.debug("fuse_attention: failed to match v path")
            return
        (concat_v, transpose_v, reshape_v, split_fc) = v_nodes

        # Try match pattern using Gemm + LayerNormalization
        fc_nodes = self.model.match_parent_path(
            split_fc,
            ["Reshape", "Gemm", "Reshape", "LayerNormalization"],
            [0, 0, 0, 0],
            output_name_to_node,
        )

        # Try match pattern using Gemm + SkipLayerNormalization
        if fc_nodes is None:
            fc_nodes = self.model.match_parent_path(
                split_fc,
                ["Reshape", "Gemm", "Reshape", "SkipLayerNormalization"],
                [0, 0, 0, 0],
                output_name_to_node,
            )

        # Try match pattern using MatMul
        if fc_nodes is None:
            # LayerNormalization
            fc_nodes = self.model.match_parent_path(
                split_fc,
                ["Add", "MatMul", "LayerNormalization"],
                [0, None, 0],
                output_name_to_node,
            )

            # SkipLayerNormalization
            if fc_nodes is None:
                fc_nodes = self.model.match_parent_path(
                    split_fc,
                    ["Add", "MatMul", "SkipLayerNormalization"],
                    [0, None, 0],
                    output_name_to_node,
                )

            if fc_nodes is None:
                logger.debug("fuse_attention: failed to match fc path")
                return

            fc_weight = fc_nodes[1].input[1]
            i, _ = self.model.get_constant_input(fc_nodes[0])
            fc_bias = fc_nodes[0].input[i]
        else:
            fc_weight = fc_nodes[1].input[1]
            fc_bias = fc_nodes[1].input[2]

        layernorm_before_attention = fc_nodes[-1]

        # `another_input` will be non-None only if
        # (1) SkipLayerNorm fusion wasn't turned ON
        # (2) SkipLayerNorm fusion was turned ON but upstream layer's LayerNorm + Add was not
        # fused into a SkipLayerNorm. This can happen if the shapes to the Add node are different.
        # So, keep the following check if SkipLayerNorm fusion is turned ON or OFF.
        if another_input is not None and another_input not in layernorm_before_attention.input:
            logger.debug("Upstream Add and (Skip)LayerNormalization shall have one same input")
            return

        is_unidirectional = True
        slice_mask = None
        input_mask_nodes = None
        concat_k_to_match = None
        qk_nodes = self.model.match_parent_path(matmul_qkv, ["Softmax", "Sub", "Mul", "Div", "MatMul"], [0, 0, 0, 0, 0])
        if qk_nodes is not None:
            (softmax_qk, sub_qk, mul_qk, div_qk, matmul_qk) = qk_nodes
            mask_nodes = self.model.match_parent_path(
                sub_qk,
                [
                    "Mul",
                    "Sub",
                    "Slice",
                    "Slice",
                    "Unsqueeze",
                    "Sub",
                    "Squeeze",
                    "Slice",
                    "Shape",
                    "Div",
                ],
                [1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
            )
            if mask_nodes is None:
                logger.debug("fuse_attention: failed to match unidirectional mask path")
                return
            div_mask = mask_nodes[-1]
            slice_mask = mask_nodes[3]

            if div_qk != div_mask:
                logger.debug("fuse_attention: skip since div_qk != div_mask")
                return

            if len(mask_nodes) > 1 and mask_nodes[0].op_type == "Mul":
                _, mul_val = self.model.get_constant_input(mask_nodes[0])
                if mul_val != -10000:
                    self.mask_filter_value = -mul_val

        else:
            # New pattern for gpt2 from PyTorch 1.5.0 and Transformers 2.9.0.
            i, qk_nodes, _ = self.model.match_parent_paths(
                matmul_qkv,
                [
                    (["Softmax", "Where", "Div", "MatMul"], [0, 0, 1, 0]),
                    (["Softmax", "Add", "Where", "Div", "MatMul"], [0, 0, None, 1, 0]),
                ],
                output_name_to_node,
            )
            if qk_nodes is None:
                logger.debug("fuse_attention: failed to match qk nodes")
                return

            where_qk = qk_nodes[-3]
            div_qk = qk_nodes[-2]
            matmul_qk = qk_nodes[-1]

            if i == 1:
                add_qk = qk_nodes[1]
                _, input_mask_nodes, _ = self.model.match_parent_paths(
                    add_qk,
                    [
                        (
                            ["Mul", "Sub", "Cast", "Unsqueeze", "Unsqueeze", "Reshape"],
                            [None, 0, 1, 0, 0, 0],
                        ),
                        (
                            ["Mul", "Sub", "Unsqueeze", "Unsqueeze", "Reshape"],
                            [None, 0, 1, 0, 0],
                        ),
                        (
                            ["Mul", "Sub", "Unsqueeze", "Unsqueeze"],
                            [None, 0, 1, 0],
                        ),  # useless cast and reshape are removed.
                    ],
                    output_name_to_node,
                )
                if input_mask_nodes is None:
                    logger.debug("fuse_attention: failed to match input attention mask path")
                    return
                if len(input_mask_nodes) > 1 and input_mask_nodes[0].op_type == "Mul":
                    _, mul_val = self.model.get_constant_input(input_mask_nodes[0])
                    if mul_val != -10000:
                        self.mask_filter_value = mul_val

            i, mask_nodes, _ = self.model.match_parent_paths(
                where_qk,
                [
                    (
                        ["Cast", "Slice", "Slice", "Unsqueeze", "Sub", "Squeeze", "Slice", "Shape"],
                        [0, 0, 0, 1, 0, 0, 0, 0],
                    ),
                    # For Transformers >= 4.27, causal mask uses torch.bool instead of torch.uint8, so no Cast to bool.
                    (
                        ["Slice", "Slice", "Unsqueeze", "Sub", "Squeeze", "Slice", "Shape"],
                        [0, 0, 1, 0, 0, 0, 0],
                    ),
                ],
                output_name_to_node,
            )
            if mask_nodes is None:
                # TODO: match mask path for GPT2LMHeadModel_BeamSearchStep.
                logger.debug("fuse_attention: failed to match mask path")
                return

            slice_mask = mask_nodes[2 if i == 0 else 1]

            div_or_concat = self.model.get_parent(mask_nodes[-1], 0, output_name_to_node)
            if div_or_concat.op_type == "Div":
                div_mask = div_or_concat
                if div_qk != div_mask:
                    logger.debug("fuse_attention: skip since div_qk != div_mask")
                    return
            elif div_or_concat.op_type == "Concat":
                concat_k_to_match = div_or_concat
            else:
                logger.debug("fuse_attention: failed to match mask path")

        # Validate that the mask data is either lower triangular (unidirectional) or all ones
        mask_data = self.model.get_constant_value(slice_mask.input[0])
        if not (
            isinstance(mask_data, np.ndarray)
            and len(mask_data.shape) == 4
            and mask_data.shape[:2] == (1, 1)
            and mask_data.shape[2] == mask_data.shape[3]
        ):
            logger.debug("fuse_attention: skip since mask shape is not 1x1xWxW")
            return

        if np.allclose(mask_data, np.ones_like(mask_data)):
            is_unidirectional = False
        elif not np.allclose(mask_data, np.tril(np.ones_like(mask_data))):
            logger.debug("fuse_attention: skip since mask is neither lower triangular nor ones")
            return

        q_nodes = self.model.match_parent_path(matmul_qk, ["Transpose", "Reshape", "Split"], [0, 0, 0])
        if q_nodes is None:
            logger.debug("fuse_attention: failed to match q path")
            return
        (transpose_q, reshape_q, split_q) = q_nodes
        if split_fc != split_q:
            logger.debug("fuse_attention: skip since split_fc != split_q")
            return

        k_nodes = self.model.match_parent_path(matmul_qk, ["Concat", "Transpose", "Reshape", "Split"], [1, 1, 0, 0])
        if k_nodes is None:
            # This pattern is from pytorch 1.7.1 and transformers 4.6.1
            k_nodes = self.model.match_parent_path(
                matmul_qk,
                ["Transpose", "Concat", "Transpose", "Reshape", "Split"],
                [1, 0, 1, 0, 0],
            )
            if k_nodes is None:
                logger.debug("fuse_attention: failed to match k path")
                return
            else:
                (_, concat_k, transpose_k, reshape_k, split_k) = k_nodes
        else:
            (concat_k, transpose_k, reshape_k, split_k) = k_nodes
        if split_fc != split_k:
            logger.debug("fuse_attention: skip since split_fc != split_k")
            return

        if concat_k_to_match and concat_k != concat_k_to_match:
            logger.debug("fuse_attention: skip since concat_k != concat_k_to_match")
            return

        attention_mask_input_name = ""
        if input_mask_nodes is not None:
            input_name = input_mask_nodes[-1].input[0]
            attention_mask_input_name = self.cast_attention_mask(input_name)

        # Match past and present paths
        past = self.match_past_pattern_1(concat_k, concat_v, output_name_to_node) or self.match_past_pattern_2(
            concat_k, concat_v, output_name_to_node
        )
        if past is None:
            logger.info("fuse_attention: failed to match past path")
            return
        if not self.model.find_graph_input(past):
            logger.debug("past is not graph input.")
            # For GPT2LMHeadModel_BeamSearchStep, there is an extra Gather node to select beam index so it is not graph input.

        present = self.match_present(concat_v, input_name_to_nodes)
        if present is None:
            logger.info("fuse_attention: failed to match present path")
            return
        if not self.model.find_graph_output(present):
            logger.info("expect present to be graph output")
            return

        self.create_attention_node(
            fc_weight,
            fc_bias,
            gemm_qkv,
            past,
            present,
            layernorm_before_attention.output[0],
            reshape_qkv.output[0],
            attention_mask_input_name,
            is_unidirectional,
        )

        # we rely on prune_graph() to clean old subgraph nodes:
        # qk_nodes + q_nodes + k_nodes + v_nodes + mask_nodes + [reshape_qkv, transpose_qkv, matmul_qkv]
        self.prune_graph = True

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\function_base.py ===
import functools
import operator
import types
import warnings

import numpy as np
from numpy._core import overrides
from numpy._core._multiarray_umath import _array_converter
from numpy._core.multiarray import add_docstring

from . import numeric as _nx
from .numeric import asanyarray, nan, ndim, result_type

__all__ = ['logspace', 'linspace', 'geomspace']


array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy')


def _linspace_dispatcher(start, stop, num=None, endpoint=None, retstep=None,
                         dtype=None, axis=None, *, device=None):
    return (start, stop)


@array_function_dispatch(_linspace_dispatcher)
def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None,
             axis=0, *, device=None):
    """
    Return evenly spaced numbers over a specified interval.

    Returns `num` evenly spaced samples, calculated over the
    interval [`start`, `stop`].

    The endpoint of the interval can optionally be excluded.

    .. versionchanged:: 1.20.0
        Values are rounded towards ``-inf`` instead of ``0`` when an
        integer ``dtype`` is specified. The old behavior can
        still be obtained with ``np.linspace(start, stop, num).astype(int)``

    Parameters
    ----------
    start : array_like
        The starting value of the sequence.
    stop : array_like
        The end value of the sequence, unless `endpoint` is set to False.
        In that case, the sequence consists of all but the last of ``num + 1``
        evenly spaced samples, so that `stop` is excluded.  Note that the step
        size changes when `endpoint` is False.
    num : int, optional
        Number of samples to generate. Default is 50. Must be non-negative.
    endpoint : bool, optional
        If True, `stop` is the last sample. Otherwise, it is not included.
        Default is True.
    retstep : bool, optional
        If True, return (`samples`, `step`), where `step` is the spacing
        between samples.
    dtype : dtype, optional
        The type of the output array.  If `dtype` is not given, the data type
        is inferred from `start` and `stop`. The inferred dtype will never be
        an integer; `float` is chosen even if the arguments would produce an
        array of integers.
    axis : int, optional
        The axis in the result to store the samples.  Relevant only if start
        or stop are array-like.  By default (0), the samples will be along a
        new axis inserted at the beginning. Use -1 to get an axis at the end.
    device : str, optional
        The device on which to place the created array. Default: None.
        For Array-API interoperability only, so must be ``"cpu"`` if passed.

        .. versionadded:: 2.0.0

    Returns
    -------
    samples : ndarray
        There are `num` equally spaced samples in the closed interval
        ``[start, stop]`` or the half-open interval ``[start, stop)``
        (depending on whether `endpoint` is True or False).
    step : float, optional
        Only returned if `retstep` is True

        Size of spacing between samples.


    See Also
    --------
    arange : Similar to `linspace`, but uses a step size (instead of the
             number of samples).
    geomspace : Similar to `linspace`, but with numbers spaced evenly on a log
                scale (a geometric progression).
    logspace : Similar to `geomspace`, but with the end points specified as
               logarithms.
    :ref:`how-to-partition`

    Examples
    --------
    >>> import numpy as np
    >>> np.linspace(2.0, 3.0, num=5)
    array([2.  , 2.25, 2.5 , 2.75, 3.  ])
    >>> np.linspace(2.0, 3.0, num=5, endpoint=False)
    array([2. ,  2.2,  2.4,  2.6,  2.8])
    >>> np.linspace(2.0, 3.0, num=5, retstep=True)
    (array([2.  ,  2.25,  2.5 ,  2.75,  3.  ]), 0.25)

    Graphical illustration:

    >>> import matplotlib.pyplot as plt
    >>> N = 8
    >>> y = np.zeros(N)
    >>> x1 = np.linspace(0, 10, N, endpoint=True)
    >>> x2 = np.linspace(0, 10, N, endpoint=False)
    >>> plt.plot(x1, y, 'o')
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.plot(x2, y + 0.5, 'o')
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.ylim([-0.5, 1])
    (-0.5, 1)
    >>> plt.show()

    """
    num = operator.index(num)
    if num < 0:
        raise ValueError(
            f"Number of samples, {num}, must be non-negative."
        )
    div = (num - 1) if endpoint else num

    conv = _array_converter(start, stop)
    start, stop = conv.as_arrays()
    dt = conv.result_type(ensure_inexact=True)

    if dtype is None:
        dtype = dt
        integer_dtype = False
    else:
        integer_dtype = _nx.issubdtype(dtype, _nx.integer)

    # Use `dtype=type(dt)` to enforce a floating point evaluation:
    delta = np.subtract(stop, start, dtype=type(dt))
    y = _nx.arange(
        0, num, dtype=dt, device=device
    ).reshape((-1,) + (1,) * ndim(delta))

    # In-place multiplication y *= delta/div is faster, but prevents
    # the multiplicant from overriding what class is produced, and thus
    # prevents, e.g. use of Quantities, see gh-7142. Hence, we multiply
    # in place only for standard scalar types.
    if div > 0:
        _mult_inplace = _nx.isscalar(delta)
        step = delta / div
        any_step_zero = (
            step == 0 if _mult_inplace else _nx.asanyarray(step == 0).any())
        if any_step_zero:
            # Special handling for denormal numbers, gh-5437
            y /= div
            if _mult_inplace:
                y *= delta
            else:
                y = y * delta
        elif _mult_inplace:
            y *= step
        else:
            y = y * step
    else:
        # sequences with 0 items or 1 item with endpoint=True (i.e. div <= 0)
        # have an undefined step
        step = nan
        # Multiply with delta to allow possible override of output class.
        y = y * delta

    y += start

    if endpoint and num > 1:
        y[-1, ...] = stop

    if axis != 0:
        y = _nx.moveaxis(y, 0, axis)

    if integer_dtype:
        _nx.floor(y, out=y)

    y = conv.wrap(y.astype(dtype, copy=False))
    if retstep:
        return y, step
    else:
        return y


def _logspace_dispatcher(start, stop, num=None, endpoint=None, base=None,
                         dtype=None, axis=None):
    return (start, stop, base)


@array_function_dispatch(_logspace_dispatcher)
def logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None,
             axis=0):
    """
    Return numbers spaced evenly on a log scale.

    In linear space, the sequence starts at ``base ** start``
    (`base` to the power of `start`) and ends with ``base ** stop``
    (see `endpoint` below).

    .. versionchanged:: 1.25.0
        Non-scalar 'base` is now supported

    Parameters
    ----------
    start : array_like
        ``base ** start`` is the starting value of the sequence.
    stop : array_like
        ``base ** stop`` is the final value of the sequence, unless `endpoint`
        is False.  In that case, ``num + 1`` values are spaced over the
        interval in log-space, of which all but the last (a sequence of
        length `num`) are returned.
    num : integer, optional
        Number of samples to generate.  Default is 50.
    endpoint : boolean, optional
        If true, `stop` is the last sample. Otherwise, it is not included.
        Default is True.
    base : array_like, optional
        The base of the log space. The step size between the elements in
        ``ln(samples) / ln(base)`` (or ``log_base(samples)``) is uniform.
        Default is 10.0.
    dtype : dtype
        The type of the output array.  If `dtype` is not given, the data type
        is inferred from `start` and `stop`. The inferred type will never be
        an integer; `float` is chosen even if the arguments would produce an
        array of integers.
    axis : int, optional
        The axis in the result to store the samples.  Relevant only if start,
        stop, or base are array-like.  By default (0), the samples will be
        along a new axis inserted at the beginning. Use -1 to get an axis at
        the end.

    Returns
    -------
    samples : ndarray
        `num` samples, equally spaced on a log scale.

    See Also
    --------
    arange : Similar to linspace, with the step size specified instead of the
             number of samples. Note that, when used with a float endpoint, the
             endpoint may or may not be included.
    linspace : Similar to logspace, but with the samples uniformly distributed
               in linear space, instead of log space.
    geomspace : Similar to logspace, but with endpoints specified directly.
    :ref:`how-to-partition`

    Notes
    -----
    If base is a scalar, logspace is equivalent to the code

    >>> y = np.linspace(start, stop, num=num, endpoint=endpoint)
    ... # doctest: +SKIP
    >>> power(base, y).astype(dtype)
    ... # doctest: +SKIP

    Examples
    --------
    >>> import numpy as np
    >>> np.logspace(2.0, 3.0, num=4)
    array([ 100.        ,  215.443469  ,  464.15888336, 1000.        ])
    >>> np.logspace(2.0, 3.0, num=4, endpoint=False)
    array([100.        ,  177.827941  ,  316.22776602,  562.34132519])
    >>> np.logspace(2.0, 3.0, num=4, base=2.0)
    array([4.        ,  5.0396842 ,  6.34960421,  8.        ])
    >>> np.logspace(2.0, 3.0, num=4, base=[2.0, 3.0], axis=-1)
    array([[ 4.        ,  5.0396842 ,  6.34960421,  8.        ],
           [ 9.        , 12.98024613, 18.72075441, 27.        ]])

    Graphical illustration:

    >>> import matplotlib.pyplot as plt
    >>> N = 10
    >>> x1 = np.logspace(0.1, 1, N, endpoint=True)
    >>> x2 = np.logspace(0.1, 1, N, endpoint=False)
    >>> y = np.zeros(N)
    >>> plt.plot(x1, y, 'o')
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.plot(x2, y + 0.5, 'o')
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.ylim([-0.5, 1])
    (-0.5, 1)
    >>> plt.show()

    """
    if not isinstance(base, (float, int)) and np.ndim(base):
        # If base is non-scalar, broadcast it with the others, since it
        # may influence how axis is interpreted.
        ndmax = np.broadcast(start, stop, base).ndim
        start, stop, base = (
            np.array(a, copy=None, subok=True, ndmin=ndmax)
            for a in (start, stop, base)
        )
        base = np.expand_dims(base, axis=axis)
    y = linspace(start, stop, num=num, endpoint=endpoint, axis=axis)
    if dtype is None:
        return _nx.power(base, y)
    return _nx.power(base, y).astype(dtype, copy=False)


def _geomspace_dispatcher(start, stop, num=None, endpoint=None, dtype=None,
                          axis=None):
    return (start, stop)


@array_function_dispatch(_geomspace_dispatcher)
def geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0):
    """
    Return numbers spaced evenly on a log scale (a geometric progression).

    This is similar to `logspace`, but with endpoints specified directly.
    Each output sample is a constant multiple of the previous.

    Parameters
    ----------
    start : array_like
        The starting value of the sequence.
    stop : array_like
        The final value of the sequence, unless `endpoint` is False.
        In that case, ``num + 1`` values are spaced over the
        interval in log-space, of which all but the last (a sequence of
        length `num`) are returned.
    num : integer, optional
        Number of samples to generate.  Default is 50.
    endpoint : boolean, optional
        If true, `stop` is the last sample. Otherwise, it is not included.
        Default is True.
    dtype : dtype
        The type of the output array.  If `dtype` is not given, the data type
        is inferred from `start` and `stop`. The inferred dtype will never be
        an integer; `float` is chosen even if the arguments would produce an
        array of integers.
    axis : int, optional
        The axis in the result to store the samples.  Relevant only if start
        or stop are array-like.  By default (0), the samples will be along a
        new axis inserted at the beginning. Use -1 to get an axis at the end.

    Returns
    -------
    samples : ndarray
        `num` samples, equally spaced on a log scale.

    See Also
    --------
    logspace : Similar to geomspace, but with endpoints specified using log
               and base.
    linspace : Similar to geomspace, but with arithmetic instead of geometric
               progression.
    arange : Similar to linspace, with the step size specified instead of the
             number of samples.
    :ref:`how-to-partition`

    Notes
    -----
    If the inputs or dtype are complex, the output will follow a logarithmic
    spiral in the complex plane.  (There are an infinite number of spirals
    passing through two points; the output will follow the shortest such path.)

    Examples
    --------
    >>> import numpy as np
    >>> np.geomspace(1, 1000, num=4)
    array([    1.,    10.,   100.,  1000.])
    >>> np.geomspace(1, 1000, num=3, endpoint=False)
    array([   1.,   10.,  100.])
    >>> np.geomspace(1, 1000, num=4, endpoint=False)
    array([   1.        ,    5.62341325,   31.6227766 ,  177.827941  ])
    >>> np.geomspace(1, 256, num=9)
    array([   1.,    2.,    4.,    8.,   16.,   32.,   64.,  128.,  256.])

    Note that the above may not produce exact integers:

    >>> np.geomspace(1, 256, num=9, dtype=int)
    array([  1,   2,   4,   7,  16,  32,  63, 127, 256])
    >>> np.around(np.geomspace(1, 256, num=9)).astype(int)
    array([  1,   2,   4,   8,  16,  32,  64, 128, 256])

    Negative, decreasing, and complex inputs are allowed:

    >>> np.geomspace(1000, 1, num=4)
    array([1000.,  100.,   10.,    1.])
    >>> np.geomspace(-1000, -1, num=4)
    array([-1000.,  -100.,   -10.,    -1.])
    >>> np.geomspace(1j, 1000j, num=4)  # Straight line
    array([0.   +1.j, 0.  +10.j, 0. +100.j, 0.+1000.j])
    >>> np.geomspace(-1+0j, 1+0j, num=5)  # Circle
    array([-1.00000000e+00+1.22464680e-16j, -7.07106781e-01+7.07106781e-01j,
            6.12323400e-17+1.00000000e+00j,  7.07106781e-01+7.07106781e-01j,
            1.00000000e+00+0.00000000e+00j])

    Graphical illustration of `endpoint` parameter:

    >>> import matplotlib.pyplot as plt
    >>> N = 10
    >>> y = np.zeros(N)
    >>> plt.semilogx(np.geomspace(1, 1000, N, endpoint=True), y + 1, 'o')
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.semilogx(np.geomspace(1, 1000, N, endpoint=False), y + 2, 'o')
    [<matplotlib.lines.Line2D object at 0x...>]
    >>> plt.axis([0.5, 2000, 0, 3])
    [0.5, 2000, 0, 3]
    >>> plt.grid(True, color='0.7', linestyle='-', which='both', axis='both')
    >>> plt.show()

    """
    start = asanyarray(start)
    stop = asanyarray(stop)
    if _nx.any(start == 0) or _nx.any(stop == 0):
        raise ValueError('Geometric sequence cannot include zero')

    dt = result_type(start, stop, float(num), _nx.zeros((), dtype))
    if dtype is None:
        dtype = dt
    else:
        # complex to dtype('complex128'), for instance
        dtype = _nx.dtype(dtype)

    # Promote both arguments to the same dtype in case, for instance, one is
    # complex and another is negative and log would produce NaN otherwise.
    # Copy since we may change things in-place further down.
    start = start.astype(dt, copy=True)
    stop = stop.astype(dt, copy=True)

    # Allow negative real values and ensure a consistent result for complex
    # (including avoiding negligible real or imaginary parts in output) by
    # rotating start to positive real, calculating, then undoing rotation.
    out_sign = _nx.sign(start)
    start /= out_sign
    stop = stop / out_sign

    log_start = _nx.log10(start)
    log_stop = _nx.log10(stop)
    result = logspace(log_start, log_stop, num=num,
                      endpoint=endpoint, base=10.0, dtype=dt)

    # Make sure the endpoints match the start and stop arguments. This is
    # necessary because np.exp(np.log(x)) is not necessarily equal to x.
    if num > 0:
        result[0] = start
        if num > 1 and endpoint:
            result[-1] = stop

    result *= out_sign

    if axis != 0:
        result = _nx.moveaxis(result, 0, axis)

    return result.astype(dtype, copy=False)


def _needs_add_docstring(obj):
    """
    Returns true if the only way to set the docstring of `obj` from python is
    via add_docstring.

    This function errs on the side of being overly conservative.
    """
    Py_TPFLAGS_HEAPTYPE = 1 << 9

    if isinstance(obj, (types.FunctionType, types.MethodType, property)):
        return False

    if isinstance(obj, type) and obj.__flags__ & Py_TPFLAGS_HEAPTYPE:
        return False

    return True


def _add_docstring(obj, doc, warn_on_python):
    if warn_on_python and not _needs_add_docstring(obj):
        warnings.warn(
            f"add_newdoc was used on a pure-python object {obj}. "
            "Prefer to attach it directly to the source.",
            UserWarning,
            stacklevel=3)
    try:
        add_docstring(obj, doc)
    except Exception:
        pass


def add_newdoc(place, obj, doc, warn_on_python=True):
    """
    Add documentation to an existing object, typically one defined in C

    The purpose is to allow easier editing of the docstrings without requiring
    a re-compile. This exists primarily for internal use within numpy itself.

    Parameters
    ----------
    place : str
        The absolute name of the module to import from
    obj : str or None
        The name of the object to add documentation to, typically a class or
        function name.
    doc : {str, Tuple[str, str], List[Tuple[str, str]]}
        If a string, the documentation to apply to `obj`

        If a tuple, then the first element is interpreted as an attribute
        of `obj` and the second as the docstring to apply -
        ``(method, docstring)``

        If a list, then each element of the list should be a tuple of length
        two - ``[(method1, docstring1), (method2, docstring2), ...]``
    warn_on_python : bool
        If True, the default, emit `UserWarning` if this is used to attach
        documentation to a pure-python object.

    Notes
    -----
    This routine never raises an error if the docstring can't be written, but
    will raise an error if the object being documented does not exist.

    This routine cannot modify read-only docstrings, as appear
    in new-style classes or built-in functions. Because this
    routine never raises an error the caller must check manually
    that the docstrings were changed.

    Since this function grabs the ``char *`` from a c-level str object and puts
    it into the ``tp_doc`` slot of the type of `obj`, it violates a number of
    C-API best-practices, by:

    - modifying a `PyTypeObject` after calling `PyType_Ready`
    - calling `Py_INCREF` on the str and losing the reference, so the str
      will never be released

    If possible it should be avoided.
    """
    new = getattr(__import__(place, globals(), {}, [obj]), obj)
    if isinstance(doc, str):
        if "${ARRAY_FUNCTION_LIKE}" in doc:
            doc = overrides.get_array_function_like_doc(new, doc)
        _add_docstring(new, doc.strip(), warn_on_python)
    elif isinstance(doc, tuple):
        attr, docstring = doc
        _add_docstring(getattr(new, attr), docstring.strip(), warn_on_python)
    elif isinstance(doc, list):
        for attr, docstring in doc:
            _add_docstring(
                getattr(new, attr), docstring.strip(), warn_on_python
            )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\__init__.py ===
"""
SymPy is a Python library for symbolic mathematics. It aims to become a
full-featured computer algebra system (CAS) while keeping the code as simple
as possible in order to be comprehensible and easily extensible.  SymPy is
written entirely in Python. It depends on mpmath, and other external libraries
may be optionally for things like plotting support.

See the webpage for more information and documentation:

    https://sympy.org

"""


# Keep this in sync with setup.py/pyproject.toml
import sys
if sys.version_info < (3, 9):
    raise ImportError("Python version 3.9 or above is required for SymPy.")
del sys


try:
    import mpmath
except ImportError:
    raise ImportError("SymPy now depends on mpmath as an external library. "
    "See https://docs.sympy.org/latest/install.html#mpmath for more information.")

del mpmath

from sympy.release import __version__
from sympy.core.cache import lazy_function

if 'dev' in __version__:
    def enable_warnings():
        import warnings
        warnings.filterwarnings('default',   '.*',   DeprecationWarning, module='sympy.*')
        del warnings
    enable_warnings()
    del enable_warnings


def __sympy_debug():
    # helper function so we don't import os globally
    import os
    debug_str = os.getenv('SYMPY_DEBUG', 'False')
    if debug_str in ('True', 'False'):
        return eval(debug_str)
    else:
        raise RuntimeError("unrecognized value for SYMPY_DEBUG: %s" %
                           debug_str)
# Fails py2 test if using type hinting
SYMPY_DEBUG = __sympy_debug()  # type: bool


from .core import (sympify, SympifyError, cacheit, Basic, Atom,
        preorder_traversal, S, Expr, AtomicExpr, UnevaluatedExpr, Symbol,
        Wild, Dummy, symbols, var, Number, Float, Rational, Integer,
        NumberSymbol, RealNumber, igcd, ilcm, seterr, E, I, nan, oo, pi, zoo,
        AlgebraicNumber, comp, mod_inverse, Pow, integer_nthroot, integer_log,
        trailing, Mul, prod, Add, Mod, Rel, Eq, Ne, Lt, Le, Gt, Ge, Equality,
        GreaterThan, LessThan, Unequality, StrictGreaterThan, StrictLessThan,
        vectorize, Lambda, WildFunction, Derivative, diff, FunctionClass,
        Function, Subs, expand, PoleError, count_ops, expand_mul, expand_log,
        expand_func, expand_trig, expand_complex, expand_multinomial, nfloat,
        expand_power_base, expand_power_exp, arity, PrecisionExhausted, N,
        evalf, Tuple, Dict, gcd_terms, factor_terms, factor_nc, evaluate,
        Catalan, EulerGamma, GoldenRatio, TribonacciConstant, bottom_up, use,
        postorder_traversal, default_sort_key, ordered, num_digits)

from .logic import (to_cnf, to_dnf, to_nnf, And, Or, Not, Xor, Nand, Nor,
        Implies, Equivalent, ITE, POSform, SOPform, simplify_logic, bool_map,
        true, false, satisfiable)

from .assumptions import (AppliedPredicate, Predicate, AssumptionsContext,
        assuming, Q, ask, register_handler, remove_handler, refine)

from .polys import (Poly, PurePoly, poly_from_expr, parallel_poly_from_expr,
        degree, total_degree, degree_list, LC, LM, LT, pdiv, prem, pquo,
        pexquo, div, rem, quo, exquo, half_gcdex, gcdex, invert,
        subresultants, resultant, discriminant, cofactors, gcd_list, gcd,
        lcm_list, lcm, terms_gcd, trunc, monic, content, primitive, compose,
        decompose, sturm, gff_list, gff, sqf_norm, sqf_part, sqf_list, sqf,
        factor_list, factor, intervals, refine_root, count_roots, all_roots,
        real_roots, nroots, ground_roots, nth_power_roots_poly, cancel,
        reduced, groebner, is_zero_dimensional, GroebnerBasis, poly,
        symmetrize, horner, interpolate, rational_interpolate, viete, together,
        BasePolynomialError, ExactQuotientFailed, PolynomialDivisionFailed,
        OperationNotSupported, HeuristicGCDFailed, HomomorphismFailed,
        IsomorphismFailed, ExtraneousFactors, EvaluationFailed,
        RefinementFailed, CoercionFailed, NotInvertible, NotReversible,
        NotAlgebraic, DomainError, PolynomialError, UnificationFailed,
        GeneratorsError, GeneratorsNeeded, ComputationFailed,
        UnivariatePolynomialError, MultivariatePolynomialError,
        PolificationFailed, OptionError, FlagError, minpoly,
        minimal_polynomial, primitive_element, field_isomorphism,
        to_number_field, isolate, round_two, prime_decomp, prime_valuation,
        galois_group, itermonomials, Monomial, lex, grlex,
        grevlex, ilex, igrlex, igrevlex, CRootOf, rootof, RootOf,
        ComplexRootOf, RootSum, roots, Domain, FiniteField, IntegerRing,
        RationalField, RealField, ComplexField, PythonFiniteField,
        GMPYFiniteField, PythonIntegerRing, GMPYIntegerRing, PythonRational,
        GMPYRationalField, AlgebraicField, PolynomialRing, FractionField,
        ExpressionDomain, FF_python, FF_gmpy, ZZ_python, ZZ_gmpy, QQ_python,
        QQ_gmpy, GF, FF, ZZ, QQ, ZZ_I, QQ_I, RR, CC, EX, EXRAW,
        construct_domain, swinnerton_dyer_poly, cyclotomic_poly,
        symmetric_poly, random_poly, interpolating_poly, jacobi_poly,
        chebyshevt_poly, chebyshevu_poly, hermite_poly, hermite_prob_poly,
        legendre_poly, laguerre_poly, apart, apart_list, assemble_partfrac_list,
        Options, ring, xring, vring, sring, field, xfield, vfield, sfield)

from .series import (Order, O, limit, Limit, gruntz, series, approximants,
        residue, EmptySequence, SeqPer, SeqFormula, sequence, SeqAdd, SeqMul,
        fourier_series, fps, difference_delta, limit_seq)

from .functions import (factorial, factorial2, rf, ff, binomial,
        RisingFactorial, FallingFactorial, subfactorial, carmichael,
        fibonacci, lucas, motzkin, tribonacci, harmonic, bernoulli, bell, euler,
        catalan, genocchi, andre, partition, divisor_sigma, legendre_symbol,
        jacobi_symbol, kronecker_symbol, mobius, primenu, primeomega,
        totient, reduced_totient, primepi, sqrt, root, Min, Max, Id,
        real_root, Rem, cbrt, re, im, sign, Abs, conjugate, arg, polar_lift,
        periodic_argument, unbranched_argument, principal_branch, transpose,
        adjoint, polarify, unpolarify, sin, cos, tan, sec, csc, cot, sinc,
        asin, acos, atan, asec, acsc, acot, atan2, exp_polar, exp, ln, log,
        LambertW, sinh, cosh, tanh, coth, sech, csch, asinh, acosh, atanh,
        acoth, asech, acsch, floor, ceiling, frac, Piecewise, piecewise_fold,
        piecewise_exclusive, erf, erfc, erfi, erf2, erfinv, erfcinv, erf2inv,
        Ei, expint, E1, li, Li, Si, Ci, Shi, Chi, fresnels, fresnelc, gamma,
        lowergamma, uppergamma, polygamma, loggamma, digamma, trigamma,
        multigamma, dirichlet_eta, zeta, lerchphi, polylog, stieltjes, Eijk,
        LeviCivita, KroneckerDelta, SingularityFunction, DiracDelta, Heaviside,
        bspline_basis, bspline_basis_set, interpolating_spline, besselj,
        bessely, besseli, besselk, hankel1, hankel2, jn, yn, jn_zeros, hn1,
        hn2, airyai, airybi, airyaiprime, airybiprime, marcumq, hyper,
        meijerg, appellf1, legendre, assoc_legendre, hermite, hermite_prob,
        chebyshevt, chebyshevu, chebyshevu_root, chebyshevt_root, laguerre,
        assoc_laguerre, gegenbauer, jacobi, jacobi_normalized, Ynm, Ynm_c,
        Znm, elliptic_k, elliptic_f, elliptic_e, elliptic_pi, beta, mathieus,
        mathieuc, mathieusprime, mathieucprime, riemann_xi, betainc, betainc_regularized)

from .ntheory import (nextprime, prevprime, prime, primerange,
        randprime, Sieve, sieve, primorial, cycle_length, composite,
        compositepi, isprime, divisors, proper_divisors, factorint,
        multiplicity, perfect_power, factor_cache, pollard_pm1, pollard_rho, primefactors,
        divisor_count, proper_divisor_count,
        factorrat,
        mersenne_prime_exponent, is_perfect, is_mersenne_prime, is_abundant,
        is_deficient, is_amicable, is_carmichael, abundance, npartitions, is_primitive_root,
        is_quad_residue, n_order, sqrt_mod,
        quadratic_residues, primitive_root, nthroot_mod, is_nthpow_residue,
        sqrt_mod_iter, discrete_log, quadratic_congruence,
        binomial_coefficients, binomial_coefficients_list,
        multinomial_coefficients, continued_fraction_periodic,
        continued_fraction_iterator, continued_fraction_reduce,
        continued_fraction_convergents, continued_fraction, egyptian_fraction)

from .concrete import product, Product, summation, Sum

from .discrete import (fft, ifft, ntt, intt, fwht, ifwht, mobius_transform,
        inverse_mobius_transform, convolution, covering_product,
        intersecting_product)

from .simplify import (simplify, hypersimp, hypersimilar, logcombine,
        separatevars, posify, besselsimp, kroneckersimp, signsimp,
        nsimplify, FU, fu, sqrtdenest, cse, epath, EPath, hyperexpand,
        collect, rcollect, radsimp, collect_const, fraction, numer, denom,
        trigsimp, exptrigsimp, powsimp, powdenest, combsimp, gammasimp,
        ratsimp, ratsimpmodprime)

from .sets import (Set, Interval, Union, EmptySet, FiniteSet, ProductSet,
        Intersection, DisjointUnion, imageset, Complement, SymmetricDifference, ImageSet,
        Range, ComplexRegion, Complexes, Reals, Contains, ConditionSet, Ordinal,
        OmegaPower, ord0, PowerSet, Naturals, Naturals0, UniversalSet,
        Integers, Rationals)

from .solvers import (solve, solve_linear_system, solve_linear_system_LU,
        solve_undetermined_coeffs, nsolve, solve_linear, checksol, det_quick,
        inv_quick, check_assumptions, failing_assumptions, diophantine,
        rsolve, rsolve_poly, rsolve_ratio, rsolve_hyper, checkodesol,
        classify_ode, dsolve, homogeneous_order, solve_poly_system, factor_system,
        solve_triangulated, pde_separate, pde_separate_add, pde_separate_mul,
        pdsolve, classify_pde, checkpdesol, ode_order, reduce_inequalities,
        reduce_abs_inequality, reduce_abs_inequalities, solve_poly_inequality,
        solve_rational_inequalities, solve_univariate_inequality, decompogen,
        solveset, linsolve, linear_eq_to_matrix, nonlinsolve, substitution)

from .matrices import (ShapeError, NonSquareMatrixError, GramSchmidt,
        casoratian, diag, eye, hessian, jordan_cell, list2numpy, matrix2numpy,
        matrix_multiply_elementwise, ones, randMatrix, rot_axis1, rot_axis2,
        rot_axis3, symarray, wronskian, zeros, MutableDenseMatrix,
        DeferredVector, MatrixBase, Matrix, MutableMatrix,
        MutableSparseMatrix, banded, ImmutableDenseMatrix,
        ImmutableSparseMatrix, ImmutableMatrix, SparseMatrix, MatrixSlice,
        BlockDiagMatrix, BlockMatrix, FunctionMatrix, Identity, Inverse,
        MatAdd, MatMul, MatPow, MatrixExpr, MatrixSymbol, Trace, Transpose,
        ZeroMatrix, OneMatrix, blockcut, block_collapse, matrix_symbols,
        Adjoint, hadamard_product, HadamardProduct, HadamardPower,
        Determinant, det, diagonalize_vector, DiagMatrix, DiagonalMatrix,
        DiagonalOf, trace, DotProduct, kronecker_product, KroneckerProduct,
        PermutationMatrix, MatrixPermute, Permanent, per, rot_ccw_axis1,
        rot_ccw_axis2, rot_ccw_axis3, rot_givens)

from .geometry import (Point, Point2D, Point3D, Line, Ray, Segment, Line2D,
        Segment2D, Ray2D, Line3D, Segment3D, Ray3D, Plane, Ellipse, Circle,
        Polygon, RegularPolygon, Triangle, rad, deg, are_similar, centroid,
        convex_hull, idiff, intersection, closest_points, farthest_points,
        GeometryError, Curve, Parabola)

from .utilities import (flatten, group, take, subsets, variations,
        numbered_symbols, cartes, capture, dict_merge, prefixes, postfixes,
        sift, topological_sort, unflatten, has_dups, has_variety, reshape,
        rotations, filldedent, lambdify,
        threaded, xthreaded, public, memoize_property, timed)

from .integrals import (integrate, Integral, line_integrate, mellin_transform,
        inverse_mellin_transform, MellinTransform, InverseMellinTransform,
        laplace_transform, laplace_correspondence, laplace_initial_conds,
        inverse_laplace_transform, LaplaceTransform,
        InverseLaplaceTransform, fourier_transform, inverse_fourier_transform,
        FourierTransform, InverseFourierTransform, sine_transform,
        inverse_sine_transform, SineTransform, InverseSineTransform,
        cosine_transform, inverse_cosine_transform, CosineTransform,
        InverseCosineTransform, hankel_transform, inverse_hankel_transform,
        HankelTransform, InverseHankelTransform, singularityintegrate)

from .tensor import (IndexedBase, Idx, Indexed, get_contraction_structure,
        get_indices, shape, MutableDenseNDimArray, ImmutableDenseNDimArray,
        MutableSparseNDimArray, ImmutableSparseNDimArray, NDimArray,
        tensorproduct, tensorcontraction, tensordiagonal, derive_by_array,
        permutedims, Array, DenseNDimArray, SparseNDimArray)

from .parsing import parse_expr

from .calculus import (euler_equations, singularities, is_increasing,
        is_strictly_increasing, is_decreasing, is_strictly_decreasing,
        is_monotonic, finite_diff_weights, apply_finite_diff,
        differentiate_finite, periodicity, not_empty_in, AccumBounds,
        is_convex, stationary_points, minimum, maximum)

from .algebras import Quaternion

from .printing import (pager_print, pretty, pretty_print, pprint,
        pprint_use_unicode, pprint_try_use_unicode, latex, print_latex,
        multiline_latex, mathml, print_mathml, python, print_python, pycode,
        ccode, print_ccode, smtlib_code, glsl_code, print_glsl, cxxcode, fcode,
        print_fcode, rcode, print_rcode, jscode, print_jscode, julia_code,
        mathematica_code, octave_code, rust_code, print_gtk, preview, srepr,
        print_tree, StrPrinter, sstr, sstrrepr, TableForm, dotprint,
        maple_code, print_maple_code)

test = lazy_function('sympy.testing.runtests_pytest', 'test')
doctest = lazy_function('sympy.testing.runtests', 'doctest')

# This module causes conflicts with other modules:
# from .stats import *
# Adds about .04-.05 seconds of import time
# from combinatorics import *
# This module is slow to import:
#from physics import units
from .plotting import plot, textplot, plot_backends, plot_implicit, plot_parametric
from .interactive import init_session, init_printing, interactive_traversal

evalf._create_evalf_table()

__all__ = [
    '__version__',

    # sympy.core
    'sympify', 'SympifyError', 'cacheit', 'Basic', 'Atom',
    'preorder_traversal', 'S', 'Expr', 'AtomicExpr', 'UnevaluatedExpr',
    'Symbol', 'Wild', 'Dummy', 'symbols', 'var', 'Number', 'Float',
    'Rational', 'Integer', 'NumberSymbol', 'RealNumber', 'igcd', 'ilcm',
    'seterr', 'E', 'I', 'nan', 'oo', 'pi', 'zoo', 'AlgebraicNumber', 'comp',
    'mod_inverse', 'Pow', 'integer_nthroot', 'integer_log', 'trailing', 'Mul', 'prod',
    'Add', 'Mod', 'Rel', 'Eq', 'Ne', 'Lt', 'Le', 'Gt', 'Ge', 'Equality',
    'GreaterThan', 'LessThan', 'Unequality', 'StrictGreaterThan',
    'StrictLessThan', 'vectorize', 'Lambda', 'WildFunction', 'Derivative',
    'diff', 'FunctionClass', 'Function', 'Subs', 'expand', 'PoleError',
    'count_ops', 'expand_mul', 'expand_log', 'expand_func', 'expand_trig',
    'expand_complex', 'expand_multinomial', 'nfloat', 'expand_power_base',
    'expand_power_exp', 'arity', 'PrecisionExhausted', 'N', 'evalf', 'Tuple',
    'Dict', 'gcd_terms', 'factor_terms', 'factor_nc', 'evaluate', 'Catalan',
    'EulerGamma', 'GoldenRatio', 'TribonacciConstant', 'bottom_up', 'use',
    'postorder_traversal', 'default_sort_key', 'ordered', 'num_digits',

    # sympy.logic
    'to_cnf', 'to_dnf', 'to_nnf', 'And', 'Or', 'Not', 'Xor', 'Nand', 'Nor',
    'Implies', 'Equivalent', 'ITE', 'POSform', 'SOPform', 'simplify_logic',
    'bool_map', 'true', 'false', 'satisfiable',

    # sympy.assumptions
    'AppliedPredicate', 'Predicate', 'AssumptionsContext', 'assuming', 'Q',
    'ask', 'register_handler', 'remove_handler', 'refine',

    # sympy.polys
    'Poly', 'PurePoly', 'poly_from_expr', 'parallel_poly_from_expr', 'degree',
    'total_degree', 'degree_list', 'LC', 'LM', 'LT', 'pdiv', 'prem', 'pquo',
    'pexquo', 'div', 'rem', 'quo', 'exquo', 'half_gcdex', 'gcdex', 'invert',
    'subresultants', 'resultant', 'discriminant', 'cofactors', 'gcd_list',
    'gcd', 'lcm_list', 'lcm', 'terms_gcd', 'trunc', 'monic', 'content',
    'primitive', 'compose', 'decompose', 'sturm', 'gff_list', 'gff',
    'sqf_norm', 'sqf_part', 'sqf_list', 'sqf', 'factor_list', 'factor',
    'intervals', 'refine_root', 'count_roots', 'all_roots', 'real_roots',
    'nroots', 'ground_roots', 'nth_power_roots_poly', 'cancel', 'reduced',
    'groebner', 'is_zero_dimensional', 'GroebnerBasis', 'poly', 'symmetrize',
    'horner', 'interpolate', 'rational_interpolate', 'viete', 'together',
    'BasePolynomialError', 'ExactQuotientFailed', 'PolynomialDivisionFailed',
    'OperationNotSupported', 'HeuristicGCDFailed', 'HomomorphismFailed',
    'IsomorphismFailed', 'ExtraneousFactors', 'EvaluationFailed',
    'RefinementFailed', 'CoercionFailed', 'NotInvertible', 'NotReversible',
    'NotAlgebraic', 'DomainError', 'PolynomialError', 'UnificationFailed',
    'GeneratorsError', 'GeneratorsNeeded', 'ComputationFailed',
    'UnivariatePolynomialError', 'MultivariatePolynomialError',
    'PolificationFailed', 'OptionError', 'FlagError', 'minpoly',
    'minimal_polynomial', 'primitive_element', 'field_isomorphism',
    'to_number_field', 'isolate', 'round_two', 'prime_decomp',
    'prime_valuation', 'galois_group', 'itermonomials', 'Monomial', 'lex', 'grlex',
    'grevlex', 'ilex', 'igrlex', 'igrevlex', 'CRootOf', 'rootof', 'RootOf',
    'ComplexRootOf', 'RootSum', 'roots', 'Domain', 'FiniteField',
    'IntegerRing', 'RationalField', 'RealField', 'ComplexField',
    'PythonFiniteField', 'GMPYFiniteField', 'PythonIntegerRing',
    'GMPYIntegerRing', 'PythonRational', 'GMPYRationalField',
    'AlgebraicField', 'PolynomialRing', 'FractionField', 'ExpressionDomain',
    'FF_python', 'FF_gmpy', 'ZZ_python', 'ZZ_gmpy', 'QQ_python', 'QQ_gmpy',
    'GF', 'FF', 'ZZ', 'QQ', 'ZZ_I', 'QQ_I', 'RR', 'CC', 'EX', 'EXRAW',
    'construct_domain', 'swinnerton_dyer_poly', 'cyclotomic_poly',
    'symmetric_poly', 'random_poly', 'interpolating_poly', 'jacobi_poly',
    'chebyshevt_poly', 'chebyshevu_poly', 'hermite_poly', 'hermite_prob_poly',
    'legendre_poly', 'laguerre_poly', 'apart', 'apart_list', 'assemble_partfrac_list',
    'Options', 'ring', 'xring', 'vring', 'sring', 'field', 'xfield', 'vfield',
    'sfield',

    # sympy.series
    'Order', 'O', 'limit', 'Limit', 'gruntz', 'series', 'approximants',
    'residue', 'EmptySequence', 'SeqPer', 'SeqFormula', 'sequence', 'SeqAdd',
    'SeqMul', 'fourier_series', 'fps', 'difference_delta', 'limit_seq',

    # sympy.functions
    'factorial', 'factorial2', 'rf', 'ff', 'binomial', 'RisingFactorial',
    'FallingFactorial', 'subfactorial', 'carmichael', 'fibonacci', 'lucas',
    'motzkin', 'tribonacci', 'harmonic', 'bernoulli', 'bell', 'euler', 'catalan',
    'genocchi', 'andre', 'partition',  'divisor_sigma', 'legendre_symbol', 'jacobi_symbol',
    'kronecker_symbol', 'mobius', 'primenu', 'primeomega', 'totient', 'primepi',
    'reduced_totient', 'sqrt', 'root', 'Min', 'Max', 'Id', 'real_root',
    'Rem', 'cbrt', 're', 'im', 'sign', 'Abs', 'conjugate', 'arg', 'polar_lift',
    'periodic_argument', 'unbranched_argument', 'principal_branch',
    'transpose', 'adjoint', 'polarify', 'unpolarify', 'sin', 'cos', 'tan',
    'sec', 'csc', 'cot', 'sinc', 'asin', 'acos', 'atan', 'asec', 'acsc',
    'acot', 'atan2', 'exp_polar', 'exp', 'ln', 'log', 'LambertW', 'sinh',
    'cosh', 'tanh', 'coth', 'sech', 'csch', 'asinh', 'acosh', 'atanh',
    'acoth', 'asech', 'acsch', 'floor', 'ceiling', 'frac', 'Piecewise',
    'piecewise_fold', 'piecewise_exclusive', 'erf', 'erfc', 'erfi', 'erf2',
    'erfinv', 'erfcinv', 'erf2inv', 'Ei', 'expint', 'E1', 'li', 'Li', 'Si',
    'Ci', 'Shi', 'Chi', 'fresnels', 'fresnelc', 'gamma', 'lowergamma',
    'uppergamma', 'polygamma', 'loggamma', 'digamma', 'trigamma', 'multigamma',
    'dirichlet_eta', 'zeta', 'lerchphi', 'polylog', 'stieltjes', 'Eijk', 'LeviCivita',
    'KroneckerDelta', 'SingularityFunction', 'DiracDelta', 'Heaviside',
    'bspline_basis', 'bspline_basis_set', 'interpolating_spline', 'besselj',
    'bessely', 'besseli', 'besselk', 'hankel1', 'hankel2', 'jn', 'yn',
    'jn_zeros', 'hn1', 'hn2', 'airyai', 'airybi', 'airyaiprime',
    'airybiprime', 'marcumq', 'hyper', 'meijerg', 'appellf1', 'legendre',
    'assoc_legendre', 'hermite', 'hermite_prob', 'chebyshevt', 'chebyshevu',
    'chebyshevu_root', 'chebyshevt_root', 'laguerre', 'assoc_laguerre',
    'gegenbauer', 'jacobi', 'jacobi_normalized', 'Ynm', 'Ynm_c', 'Znm',
    'elliptic_k', 'elliptic_f', 'elliptic_e', 'elliptic_pi', 'beta',
    'mathieus', 'mathieuc', 'mathieusprime', 'mathieucprime', 'riemann_xi','betainc',
    'betainc_regularized',

    # sympy.ntheory
    'nextprime', 'prevprime', 'prime', 'primerange', 'randprime',
    'Sieve', 'sieve', 'primorial', 'cycle_length', 'composite', 'compositepi',
    'isprime', 'divisors', 'proper_divisors', 'factorint', 'multiplicity',
    'perfect_power', 'pollard_pm1', 'factor_cache', 'pollard_rho', 'primefactors',
    'divisor_count', 'proper_divisor_count',
    'factorrat',
    'mersenne_prime_exponent', 'is_perfect', 'is_mersenne_prime',
    'is_abundant', 'is_deficient', 'is_amicable', 'is_carmichael', 'abundance',
    'npartitions',
    'is_primitive_root', 'is_quad_residue',
    'n_order', 'sqrt_mod', 'quadratic_residues',
    'primitive_root', 'nthroot_mod', 'is_nthpow_residue', 'sqrt_mod_iter',
    'discrete_log', 'quadratic_congruence', 'binomial_coefficients',
    'binomial_coefficients_list', 'multinomial_coefficients',
    'continued_fraction_periodic', 'continued_fraction_iterator',
    'continued_fraction_reduce', 'continued_fraction_convergents',
    'continued_fraction', 'egyptian_fraction',

    # sympy.concrete
    'product', 'Product', 'summation', 'Sum',

    # sympy.discrete
    'fft', 'ifft', 'ntt', 'intt', 'fwht', 'ifwht', 'mobius_transform',
    'inverse_mobius_transform', 'convolution', 'covering_product',
    'intersecting_product',

    # sympy.simplify
    'simplify', 'hypersimp', 'hypersimilar', 'logcombine', 'separatevars',
    'posify', 'besselsimp', 'kroneckersimp', 'signsimp',
    'nsimplify', 'FU', 'fu', 'sqrtdenest', 'cse', 'epath', 'EPath',
    'hyperexpand', 'collect', 'rcollect', 'radsimp', 'collect_const',
    'fraction', 'numer', 'denom', 'trigsimp', 'exptrigsimp', 'powsimp',
    'powdenest', 'combsimp', 'gammasimp', 'ratsimp', 'ratsimpmodprime',

    # sympy.sets
    'Set', 'Interval', 'Union', 'EmptySet', 'FiniteSet', 'ProductSet',
    'Intersection', 'imageset', 'DisjointUnion', 'Complement', 'SymmetricDifference',
    'ImageSet', 'Range', 'ComplexRegion', 'Reals', 'Contains', 'ConditionSet',
    'Ordinal', 'OmegaPower', 'ord0', 'PowerSet', 'Naturals',
    'Naturals0', 'UniversalSet', 'Integers', 'Rationals', 'Complexes',

    # sympy.solvers
    'solve', 'solve_linear_system', 'solve_linear_system_LU',
    'solve_undetermined_coeffs', 'nsolve', 'solve_linear', 'checksol',
    'det_quick', 'inv_quick', 'check_assumptions', 'failing_assumptions',
    'diophantine', 'rsolve', 'rsolve_poly', 'rsolve_ratio', 'rsolve_hyper',
    'checkodesol', 'classify_ode', 'dsolve', 'homogeneous_order',
    'solve_poly_system', 'factor_system', 'solve_triangulated', 'pde_separate',
    'pde_separate_add', 'pde_separate_mul', 'pdsolve', 'classify_pde',
    'checkpdesol', 'ode_order', 'reduce_inequalities',
    'reduce_abs_inequality', 'reduce_abs_inequalities',
    'solve_poly_inequality', 'solve_rational_inequalities',
    'solve_univariate_inequality', 'decompogen', 'solveset', 'linsolve',
    'linear_eq_to_matrix', 'nonlinsolve', 'substitution',

    # sympy.matrices
    'ShapeError', 'NonSquareMatrixError', 'GramSchmidt', 'casoratian', 'diag',
    'eye', 'hessian', 'jordan_cell', 'list2numpy', 'matrix2numpy',
    'matrix_multiply_elementwise', 'ones', 'randMatrix', 'rot_axis1',
    'rot_axis2', 'rot_axis3', 'symarray', 'wronskian', 'zeros',
    'MutableDenseMatrix', 'DeferredVector', 'MatrixBase', 'Matrix',
    'MutableMatrix', 'MutableSparseMatrix', 'banded', 'ImmutableDenseMatrix',
    'ImmutableSparseMatrix', 'ImmutableMatrix', 'SparseMatrix', 'MatrixSlice',
    'BlockDiagMatrix', 'BlockMatrix', 'FunctionMatrix', 'Identity', 'Inverse',
    'MatAdd', 'MatMul', 'MatPow', 'MatrixExpr', 'MatrixSymbol', 'Trace',
    'Transpose', 'ZeroMatrix', 'OneMatrix', 'blockcut', 'block_collapse',
    'matrix_symbols', 'Adjoint', 'hadamard_product', 'HadamardProduct',
    'HadamardPower', 'Determinant', 'det', 'diagonalize_vector', 'DiagMatrix',
    'DiagonalMatrix', 'DiagonalOf', 'trace', 'DotProduct',
    'kronecker_product', 'KroneckerProduct', 'PermutationMatrix',
    'MatrixPermute', 'Permanent', 'per', 'rot_ccw_axis1', 'rot_ccw_axis2',
    'rot_ccw_axis3', 'rot_givens',

    # sympy.geometry
    'Point', 'Point2D', 'Point3D', 'Line', 'Ray', 'Segment', 'Line2D',
    'Segment2D', 'Ray2D', 'Line3D', 'Segment3D', 'Ray3D', 'Plane', 'Ellipse',
    'Circle', 'Polygon', 'RegularPolygon', 'Triangle', 'rad', 'deg',
    'are_similar', 'centroid', 'convex_hull', 'idiff', 'intersection',
    'closest_points', 'farthest_points', 'GeometryError', 'Curve', 'Parabola',

    # sympy.utilities
    'flatten', 'group', 'take', 'subsets', 'variations', 'numbered_symbols',
    'cartes', 'capture', 'dict_merge', 'prefixes', 'postfixes', 'sift',
    'topological_sort', 'unflatten', 'has_dups', 'has_variety', 'reshape',
    'rotations', 'filldedent', 'lambdify', 'threaded', 'xthreaded',
    'public', 'memoize_property', 'timed',

    # sympy.integrals
    'integrate', 'Integral', 'line_integrate', 'mellin_transform',
    'inverse_mellin_transform', 'MellinTransform', 'InverseMellinTransform',
    'laplace_transform', 'inverse_laplace_transform', 'LaplaceTransform',
    'laplace_correspondence', 'laplace_initial_conds',
    'InverseLaplaceTransform', 'fourier_transform',
    'inverse_fourier_transform', 'FourierTransform',
    'InverseFourierTransform', 'sine_transform', 'inverse_sine_transform',
    'SineTransform', 'InverseSineTransform', 'cosine_transform',
    'inverse_cosine_transform', 'CosineTransform', 'InverseCosineTransform',
    'hankel_transform', 'inverse_hankel_transform', 'HankelTransform',
    'InverseHankelTransform', 'singularityintegrate',

    # sympy.tensor
    'IndexedBase', 'Idx', 'Indexed', 'get_contraction_structure',
    'get_indices', 'shape', 'MutableDenseNDimArray', 'ImmutableDenseNDimArray',
    'MutableSparseNDimArray', 'ImmutableSparseNDimArray', 'NDimArray',
    'tensorproduct', 'tensorcontraction', 'tensordiagonal', 'derive_by_array',
    'permutedims', 'Array', 'DenseNDimArray', 'SparseNDimArray',

    # sympy.parsing
    'parse_expr',

    # sympy.calculus
    'euler_equations', 'singularities', 'is_increasing',
    'is_strictly_increasing', 'is_decreasing', 'is_strictly_decreasing',
    'is_monotonic', 'finite_diff_weights', 'apply_finite_diff',
    'differentiate_finite', 'periodicity', 'not_empty_in',
    'AccumBounds', 'is_convex', 'stationary_points', 'minimum', 'maximum',

    # sympy.algebras
    'Quaternion',

    # sympy.printing
    'pager_print', 'pretty', 'pretty_print', 'pprint', 'pprint_use_unicode',
    'pprint_try_use_unicode', 'latex', 'print_latex', 'multiline_latex',
    'mathml', 'print_mathml', 'python', 'print_python', 'pycode', 'ccode',
    'print_ccode', 'smtlib_code', 'glsl_code', 'print_glsl', 'cxxcode', 'fcode',
    'print_fcode', 'rcode', 'print_rcode', 'jscode', 'print_jscode',
    'julia_code', 'mathematica_code', 'octave_code', 'rust_code', 'print_gtk',
    'preview', 'srepr', 'print_tree', 'StrPrinter', 'sstr', 'sstrrepr',
    'TableForm', 'dotprint', 'maple_code', 'print_maple_code',

    # sympy.plotting
    'plot', 'textplot', 'plot_backends', 'plot_implicit', 'plot_parametric',

    # sympy.interactive
    'init_session', 'init_printing', 'interactive_traversal',

    # sympy.testing
    'test', 'doctest',
]


#===========================================================================#
#                                                                           #
# XXX: The names below were importable before SymPy 1.6 using               #
#                                                                           #
#          from sympy import *                                              #
#                                                                           #
# This happened implicitly because there was no __all__ defined in this     #
# __init__.py file. Not every package is imported. The list matches what    #
# would have been imported before. It is possible that these packages will  #
# not be imported by a star-import from sympy in future.                    #
#                                                                           #
#===========================================================================#


__all__.extend((
    'algebras',
    'assumptions',
    'calculus',
    'concrete',
    'discrete',
    'external',
    'functions',
    'geometry',
    'interactive',
    'multipledispatch',
    'ntheory',
    'parsing',
    'plotting',
    'polys',
    'printing',
    'release',
    'strategies',
    'tensor',
    'utilities',
))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\vector\dyadic.py ===
from sympy import sympify, Add, ImmutableMatrix as Matrix
from sympy.core.evalf import EvalfMixin
from sympy.printing.defaults import Printable

from mpmath.libmp.libmpf import prec_to_dps


__all__ = ['Dyadic']


class Dyadic(Printable, EvalfMixin):
    """A Dyadic object.

    See:
    https://en.wikipedia.org/wiki/Dyadic_tensor
    Kane, T., Levinson, D. Dynamics Theory and Applications. 1985 McGraw-Hill

    A more powerful way to represent a rigid body's inertia. While it is more
    complex, by choosing Dyadic components to be in body fixed basis vectors,
    the resulting matrix is equivalent to the inertia tensor.

    """

    is_number = False

    def __init__(self, inlist):
        """
        Just like Vector's init, you should not call this unless creating a
        zero dyadic.

        zd = Dyadic(0)

        Stores a Dyadic as a list of lists; the inner list has the measure
        number and the two unit vectors; the outerlist holds each unique
        unit vector pair.

        """

        self.args = []
        if inlist == 0:
            inlist = []
        while len(inlist) != 0:
            added = 0
            for i, v in enumerate(self.args):
                if ((str(inlist[0][1]) == str(self.args[i][1])) and
                        (str(inlist[0][2]) == str(self.args[i][2]))):
                    self.args[i] = (self.args[i][0] + inlist[0][0],
                                    inlist[0][1], inlist[0][2])
                    inlist.remove(inlist[0])
                    added = 1
                    break
            if added != 1:
                self.args.append(inlist[0])
                inlist.remove(inlist[0])
        i = 0
        # This code is to remove empty parts from the list
        while i < len(self.args):
            if ((self.args[i][0] == 0) | (self.args[i][1] == 0) |
                    (self.args[i][2] == 0)):
                self.args.remove(self.args[i])
                i -= 1
            i += 1

    @property
    def func(self):
        """Returns the class Dyadic. """
        return Dyadic

    def __add__(self, other):
        """The add operator for Dyadic. """
        other = _check_dyadic(other)
        return Dyadic(self.args + other.args)

    __radd__ = __add__

    def __mul__(self, other):
        """Multiplies the Dyadic by a sympifyable expression.

        Parameters
        ==========

        other : Sympafiable
            The scalar to multiply this Dyadic with

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame, outer
        >>> N = ReferenceFrame('N')
        >>> d = outer(N.x, N.x)
        >>> 5 * d
        5*(N.x|N.x)

        """
        newlist = list(self.args)
        other = sympify(other)
        for i in range(len(newlist)):
            newlist[i] = (other * newlist[i][0], newlist[i][1],
                          newlist[i][2])
        return Dyadic(newlist)

    __rmul__ = __mul__

    def dot(self, other):
        """The inner product operator for a Dyadic and a Dyadic or Vector.

        Parameters
        ==========

        other : Dyadic or Vector
            The other Dyadic or Vector to take the inner product with

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame, outer
        >>> N = ReferenceFrame('N')
        >>> D1 = outer(N.x, N.y)
        >>> D2 = outer(N.y, N.y)
        >>> D1.dot(D2)
        (N.x|N.y)
        >>> D1.dot(N.y)
        N.x

        """
        from sympy.physics.vector.vector import Vector, _check_vector
        if isinstance(other, Dyadic):
            other = _check_dyadic(other)
            ol = Dyadic(0)
            for v in self.args:
                for v2 in other.args:
                    ol += v[0] * v2[0] * (v[2].dot(v2[1])) * (v[1].outer(v2[2]))
        else:
            other = _check_vector(other)
            ol = Vector(0)
            for v in self.args:
                ol += v[0] * v[1] * (v[2].dot(other))
        return ol

    # NOTE : supports non-advertised Dyadic & Dyadic, Dyadic & Vector notation
    __and__ = dot

    def __truediv__(self, other):
        """Divides the Dyadic by a sympifyable expression. """
        return self.__mul__(1 / other)

    def __eq__(self, other):
        """Tests for equality.

        Is currently weak; needs stronger comparison testing

        """

        if other == 0:
            other = Dyadic(0)
        other = _check_dyadic(other)
        if (self.args == []) and (other.args == []):
            return True
        elif (self.args == []) or (other.args == []):
            return False
        return set(self.args) == set(other.args)

    def __ne__(self, other):
        return not self == other

    def __neg__(self):
        return self * -1

    def _latex(self, printer):
        ar = self.args  # just to shorten things
        if len(ar) == 0:
            return str(0)
        ol = []  # output list, to be concatenated to a string
        for v in ar:
            # if the coef of the dyadic is 1, we skip the 1
            if v[0] == 1:
                ol.append(' + ' + printer._print(v[1]) + r"\otimes " +
                          printer._print(v[2]))
            # if the coef of the dyadic is -1, we skip the 1
            elif v[0] == -1:
                ol.append(' - ' +
                          printer._print(v[1]) +
                          r"\otimes " +
                          printer._print(v[2]))
            # If the coefficient of the dyadic is not 1 or -1,
            # we might wrap it in parentheses, for readability.
            elif v[0] != 0:
                arg_str = printer._print(v[0])
                if isinstance(v[0], Add):
                    arg_str = '(%s)' % arg_str
                if arg_str.startswith('-'):
                    arg_str = arg_str[1:]
                    str_start = ' - '
                else:
                    str_start = ' + '
                ol.append(str_start + arg_str + printer._print(v[1]) +
                          r"\otimes " + printer._print(v[2]))
        outstr = ''.join(ol)
        if outstr.startswith(' + '):
            outstr = outstr[3:]
        elif outstr.startswith(' '):
            outstr = outstr[1:]
        return outstr

    def _pretty(self, printer):
        e = self

        class Fake:
            baseline = 0

            def render(self, *args, **kwargs):
                ar = e.args  # just to shorten things
                mpp = printer
                if len(ar) == 0:
                    return str(0)
                bar = "\N{CIRCLED TIMES}" if printer._use_unicode else "|"
                ol = []  # output list, to be concatenated to a string
                for v in ar:
                    # if the coef of the dyadic is 1, we skip the 1
                    if v[0] == 1:
                        ol.extend([" + ",
                                  mpp.doprint(v[1]),
                                  bar,
                                  mpp.doprint(v[2])])

                    # if the coef of the dyadic is -1, we skip the 1
                    elif v[0] == -1:
                        ol.extend([" - ",
                                  mpp.doprint(v[1]),
                                  bar,
                                  mpp.doprint(v[2])])

                    # If the coefficient of the dyadic is not 1 or -1,
                    # we might wrap it in parentheses, for readability.
                    elif v[0] != 0:
                        if isinstance(v[0], Add):
                            arg_str = mpp._print(
                                v[0]).parens()[0]
                        else:
                            arg_str = mpp.doprint(v[0])
                        if arg_str.startswith("-"):
                            arg_str = arg_str[1:]
                            str_start = " - "
                        else:
                            str_start = " + "
                        ol.extend([str_start, arg_str, " ",
                                  mpp.doprint(v[1]),
                                  bar,
                                  mpp.doprint(v[2])])

                outstr = "".join(ol)
                if outstr.startswith(" + "):
                    outstr = outstr[3:]
                elif outstr.startswith(" "):
                    outstr = outstr[1:]
                return outstr
        return Fake()

    def __rsub__(self, other):
        return (-1 * self) + other

    def _sympystr(self, printer):
        """Printing method. """
        ar = self.args  # just to shorten things
        if len(ar) == 0:
            return printer._print(0)
        ol = []  # output list, to be concatenated to a string
        for v in ar:
            # if the coef of the dyadic is 1, we skip the 1
            if v[0] == 1:
                ol.append(' + (' + printer._print(v[1]) + '|' +
                          printer._print(v[2]) + ')')
            # if the coef of the dyadic is -1, we skip the 1
            elif v[0] == -1:
                ol.append(' - (' + printer._print(v[1]) + '|' +
                          printer._print(v[2]) + ')')
            # If the coefficient of the dyadic is not 1 or -1,
            # we might wrap it in parentheses, for readability.
            elif v[0] != 0:
                arg_str = printer._print(v[0])
                if isinstance(v[0], Add):
                    arg_str = "(%s)" % arg_str
                if arg_str[0] == '-':
                    arg_str = arg_str[1:]
                    str_start = ' - '
                else:
                    str_start = ' + '
                ol.append(str_start + arg_str + '*(' +
                          printer._print(v[1]) +
                          '|' + printer._print(v[2]) + ')')
        outstr = ''.join(ol)
        if outstr.startswith(' + '):
            outstr = outstr[3:]
        elif outstr.startswith(' '):
            outstr = outstr[1:]
        return outstr

    def __sub__(self, other):
        """The subtraction operator. """
        return self.__add__(other * -1)

    def cross(self, other):
        """Returns the dyadic resulting from the dyadic vector cross product:
        Dyadic x Vector.

        Parameters
        ==========
        other : Vector
            Vector to cross with.

        Examples
        ========
        >>> from sympy.physics.vector import ReferenceFrame, outer, cross
        >>> N = ReferenceFrame('N')
        >>> d = outer(N.x, N.x)
        >>> cross(d, N.y)
        (N.x|N.z)

        """
        from sympy.physics.vector.vector import _check_vector
        other = _check_vector(other)
        ol = Dyadic(0)
        for v in self.args:
            ol += v[0] * (v[1].outer((v[2].cross(other))))
        return ol

    # NOTE : supports non-advertised Dyadic ^ Vector notation
    __xor__ = cross

    def express(self, frame1, frame2=None):
        """Expresses this Dyadic in alternate frame(s)

        The first frame is the list side expression, the second frame is the
        right side; if Dyadic is in form A.x|B.y, you can express it in two
        different frames. If no second frame is given, the Dyadic is
        expressed in only one frame.

        Calls the global express function

        Parameters
        ==========

        frame1 : ReferenceFrame
            The frame to express the left side of the Dyadic in
        frame2 : ReferenceFrame
            If provided, the frame to express the right side of the Dyadic in

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame, outer, dynamicsymbols
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> N = ReferenceFrame('N')
        >>> q = dynamicsymbols('q')
        >>> B = N.orientnew('B', 'Axis', [q, N.z])
        >>> d = outer(N.x, N.x)
        >>> d.express(B, N)
        cos(q)*(B.x|N.x) - sin(q)*(B.y|N.x)

        """
        from sympy.physics.vector.functions import express
        return express(self, frame1, frame2)

    def to_matrix(self, reference_frame, second_reference_frame=None):
        """Returns the matrix form of the dyadic with respect to one or two
        reference frames.

        Parameters
        ----------
        reference_frame : ReferenceFrame
            The reference frame that the rows and columns of the matrix
            correspond to. If a second reference frame is provided, this
            only corresponds to the rows of the matrix.
        second_reference_frame : ReferenceFrame, optional, default=None
            The reference frame that the columns of the matrix correspond
            to.

        Returns
        -------
        matrix : ImmutableMatrix, shape(3,3)
            The matrix that gives the 2D tensor form.

        Examples
        ========

        >>> from sympy import symbols, trigsimp
        >>> from sympy.physics.vector import ReferenceFrame
        >>> from sympy.physics.mechanics import inertia
        >>> Ixx, Iyy, Izz, Ixy, Iyz, Ixz = symbols('Ixx, Iyy, Izz, Ixy, Iyz, Ixz')
        >>> N = ReferenceFrame('N')
        >>> inertia_dyadic = inertia(N, Ixx, Iyy, Izz, Ixy, Iyz, Ixz)
        >>> inertia_dyadic.to_matrix(N)
        Matrix([
        [Ixx, Ixy, Ixz],
        [Ixy, Iyy, Iyz],
        [Ixz, Iyz, Izz]])
        >>> beta = symbols('beta')
        >>> A = N.orientnew('A', 'Axis', (beta, N.x))
        >>> trigsimp(inertia_dyadic.to_matrix(A))
        Matrix([
        [                           Ixx,                                           Ixy*cos(beta) + Ixz*sin(beta),                                           -Ixy*sin(beta) + Ixz*cos(beta)],
        [ Ixy*cos(beta) + Ixz*sin(beta), Iyy*cos(2*beta)/2 + Iyy/2 + Iyz*sin(2*beta) - Izz*cos(2*beta)/2 + Izz/2,                 -Iyy*sin(2*beta)/2 + Iyz*cos(2*beta) + Izz*sin(2*beta)/2],
        [-Ixy*sin(beta) + Ixz*cos(beta),                -Iyy*sin(2*beta)/2 + Iyz*cos(2*beta) + Izz*sin(2*beta)/2, -Iyy*cos(2*beta)/2 + Iyy/2 - Iyz*sin(2*beta) + Izz*cos(2*beta)/2 + Izz/2]])

        """

        if second_reference_frame is None:
            second_reference_frame = reference_frame

        return Matrix([i.dot(self).dot(j) for i in reference_frame for j in
                      second_reference_frame]).reshape(3, 3)

    def doit(self, **hints):
        """Calls .doit() on each term in the Dyadic"""
        return sum([Dyadic([(v[0].doit(**hints), v[1], v[2])])
                    for v in self.args], Dyadic(0))

    def dt(self, frame):
        """Take the time derivative of this Dyadic in a frame.

        This function calls the global time_derivative method

        Parameters
        ==========

        frame : ReferenceFrame
            The frame to take the time derivative in

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame, outer, dynamicsymbols
        >>> from sympy.physics.vector import init_vprinting
        >>> init_vprinting(pretty_print=False)
        >>> N = ReferenceFrame('N')
        >>> q = dynamicsymbols('q')
        >>> B = N.orientnew('B', 'Axis', [q, N.z])
        >>> d = outer(N.x, N.x)
        >>> d.dt(B)
        - q'*(N.y|N.x) - q'*(N.x|N.y)

        """
        from sympy.physics.vector.functions import time_derivative
        return time_derivative(self, frame)

    def simplify(self):
        """Returns a simplified Dyadic."""
        out = Dyadic(0)
        for v in self.args:
            out += Dyadic([(v[0].simplify(), v[1], v[2])])
        return out

    def subs(self, *args, **kwargs):
        """Substitution on the Dyadic.

        Examples
        ========

        >>> from sympy.physics.vector import ReferenceFrame
        >>> from sympy import Symbol
        >>> N = ReferenceFrame('N')
        >>> s = Symbol('s')
        >>> a = s*(N.x|N.x)
        >>> a.subs({s: 2})
        2*(N.x|N.x)

        """

        return sum([Dyadic([(v[0].subs(*args, **kwargs), v[1], v[2])])
                    for v in self.args], Dyadic(0))

    def applyfunc(self, f):
        """Apply a function to each component of a Dyadic."""
        if not callable(f):
            raise TypeError("`f` must be callable.")

        out = Dyadic(0)
        for a, b, c in self.args:
            out += f(a) * (b.outer(c))
        return out

    def _eval_evalf(self, prec):
        if not self.args:
            return self
        new_args = []
        dps = prec_to_dps(prec)
        for inlist in self.args:
            new_inlist = list(inlist)
            new_inlist[0] = inlist[0].evalf(n=dps)
            new_args.append(tuple(new_inlist))
        return Dyadic(new_args)

    def xreplace(self, rule):
        """
        Replace occurrences of objects within the measure numbers of the
        Dyadic.

        Parameters
        ==========

        rule : dict-like
            Expresses a replacement rule.

        Returns
        =======

        Dyadic
            Result of the replacement.

        Examples
        ========

        >>> from sympy import symbols, pi
        >>> from sympy.physics.vector import ReferenceFrame, outer
        >>> N = ReferenceFrame('N')
        >>> D = outer(N.x, N.x)
        >>> x, y, z = symbols('x y z')
        >>> ((1 + x*y) * D).xreplace({x: pi})
        (pi*y + 1)*(N.x|N.x)
        >>> ((1 + x*y) * D).xreplace({x: pi, y: 2})
        (1 + 2*pi)*(N.x|N.x)

        Replacements occur only if an entire node in the expression tree is
        matched:

        >>> ((x*y + z) * D).xreplace({x*y: pi})
        (z + pi)*(N.x|N.x)
        >>> ((x*y*z) * D).xreplace({x*y: pi})
        x*y*z*(N.x|N.x)

        """

        new_args = []
        for inlist in self.args:
            new_inlist = list(inlist)
            new_inlist[0] = new_inlist[0].xreplace(rule)
            new_args.append(tuple(new_inlist))
        return Dyadic(new_args)


def _check_dyadic(other):
    if not isinstance(other, Dyadic):
        raise TypeError('A Dyadic must be supplied')
    return other