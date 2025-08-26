
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\metrics.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import datetime
import json

import pandas as pd


class BaseObject:
    def __init__(self):
        self.customized = {}

    def to_dict(self):
        default_values = self.__dict__.copy()
        default_values.pop("customized", None)
        default_values.update(self.customized)

        for k, v in default_values.items():
            if isinstance(v, BaseObject):
                default_values[k] = v.to_dict()

        return {k: v for k, v in default_values.items() if v}


class ModelInfo(BaseObject):
    def __init__(
        self,
        full_name: str | None = None,
        is_huggingface: bool | None = False,
        is_text_generation: bool | None = False,
        short_name: str | None = None,
    ):
        super().__init__()
        self.full_name = full_name
        self.is_huggingface = is_huggingface
        self.is_text_generation = is_text_generation
        self.short_name = short_name
        self.input_shape = []


class BackendOptions(BaseObject):
    def __init__(
        self,
        enable_profiling: bool | None = False,
        execution_provider: str | None = None,
        use_io_binding: bool | None = False,
    ):
        super().__init__()
        self.enable_profiling = enable_profiling
        self.execution_provider = execution_provider
        self.use_io_binding = use_io_binding


class Config(BaseObject):
    def __init__(
        self,
        backend: str | None = "onnxruntime",
        batch_size: int | None = 1,
        seq_length: int | None = 0,
        precision: str | None = "fp32",
        warmup_runs: int | None = 1,
        measured_runs: int | None = 10,
    ):
        super().__init__()
        self.backend = backend
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.precision = precision
        self.warmup_runs = warmup_runs
        self.measured_runs = measured_runs
        self.model_info = ModelInfo()
        self.backend_options = BackendOptions()


class Metadata(BaseObject):
    def __init__(
        self,
        device: str | None = None,
        package_name: str | None = None,
        package_version: str | None = None,
        platform: str | None = None,
        python_version: str | None = None,
    ):
        super().__init__()
        self.device = device
        self.package_name = package_name
        self.package_version = package_version
        self.platform = platform
        self.python_version = python_version


class Metrics(BaseObject):
    def __init__(
        self,
        latency_ms_mean: float | None = 0.0,
        throughput_qps: float | None = 0.0,
        max_memory_usage_GB: float | None = 0.0,
    ):
        super().__init__()
        self.latency_ms_mean = latency_ms_mean
        self.throughput_qps = throughput_qps
        self.max_memory_usage_GB = max_memory_usage_GB


class BenchmarkRecord:
    def __init__(
        self,
        model_name: str,
        precision: str,
        backend: str,
        device: str,
        package_name: str,
        package_version: str,
        batch_size: int | None = 1,
        warmup_runs: int | None = 1,
        measured_runs: int | None = 10,
        trigger_date: str | None = None,
    ):
        self.config = Config()
        self.metrics = Metrics()
        self.metadata = Metadata()
        self.trigger_date = trigger_date or datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        self.config.model_info.full_name = model_name
        self.config.precision = precision
        self.config.backend = backend
        self.config.batch_size = batch_size
        self.config.warmup_runs = warmup_runs
        self.config.measured_runs = measured_runs
        self.metadata.device = device
        self.metadata.package_name = package_name
        self.metadata.package_version = package_version

    def to_dict(self) -> dict:
        return {
            "config": self.config.to_dict(),
            "metadata": self.metadata.to_dict(),
            "metrics": self.metrics.to_dict(),
            "trigger_date": self.trigger_date,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def save_as_csv(cls, file_name: str, records: list) -> None:
        if records is None or len(records) == 0:
            return
        rds = [record.to_dict() for record in records]
        df = pd.json_normalize(rds)
        df.to_csv(file_name, index=False)

    @classmethod
    def save_as_json(cls, file_name: str, records: list) -> None:
        if records is None or len(records) == 0:
            return
        rds = [record.to_dict() for record in records]
        with open(file_name, "w") as f:
            json.dump(rds, f, indent=4, default=str)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\workbook\protection.py ===
# Copyright (c) 2010-2024 openpyxl

from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import (
    Alias,
    Typed,
    String,
    Float,
    Integer,
    Bool,
    NoneSet,
    Set,
)
from openpyxl.descriptors.excel import (
    ExtensionList,
    HexBinary,
    Guid,
    Relation,
    Base64Binary,
)
from openpyxl.utils.protection import hash_password


class WorkbookProtection(Serialisable):

    _workbook_password, _revisions_password = None, None

    tagname = "workbookPr"

    workbook_password = Alias("workbookPassword")
    workbookPasswordCharacterSet = String(allow_none=True)
    revision_password = Alias("revisionsPassword")
    revisionsPasswordCharacterSet = String(allow_none=True)
    lockStructure = Bool(allow_none=True)
    lock_structure = Alias("lockStructure")
    lockWindows = Bool(allow_none=True)
    lock_windows = Alias("lockWindows")
    lockRevision = Bool(allow_none=True)
    lock_revision = Alias("lockRevision")
    revisionsAlgorithmName = String(allow_none=True)
    revisionsHashValue = Base64Binary(allow_none=True)
    revisionsSaltValue = Base64Binary(allow_none=True)
    revisionsSpinCount = Integer(allow_none=True)
    workbookAlgorithmName = String(allow_none=True)
    workbookHashValue = Base64Binary(allow_none=True)
    workbookSaltValue = Base64Binary(allow_none=True)
    workbookSpinCount = Integer(allow_none=True)

    __attrs__ = ('workbookPassword', 'workbookPasswordCharacterSet', 'revisionsPassword',
                 'revisionsPasswordCharacterSet', 'lockStructure', 'lockWindows', 'lockRevision',
                 'revisionsAlgorithmName', 'revisionsHashValue', 'revisionsSaltValue',
                 'revisionsSpinCount', 'workbookAlgorithmName', 'workbookHashValue',
                 'workbookSaltValue', 'workbookSpinCount')

    def __init__(self,
                 workbookPassword=None,
                 workbookPasswordCharacterSet=None,
                 revisionsPassword=None,
                 revisionsPasswordCharacterSet=None,
                 lockStructure=None,
                 lockWindows=None,
                 lockRevision=None,
                 revisionsAlgorithmName=None,
                 revisionsHashValue=None,
                 revisionsSaltValue=None,
                 revisionsSpinCount=None,
                 workbookAlgorithmName=None,
                 workbookHashValue=None,
                 workbookSaltValue=None,
                 workbookSpinCount=None,
                ):
        if workbookPassword is not None:
            self.workbookPassword = workbookPassword
        self.workbookPasswordCharacterSet = workbookPasswordCharacterSet
        if revisionsPassword is not None:
            self.revisionsPassword = revisionsPassword
        self.revisionsPasswordCharacterSet = revisionsPasswordCharacterSet
        self.lockStructure = lockStructure
        self.lockWindows = lockWindows
        self.lockRevision = lockRevision
        self.revisionsAlgorithmName = revisionsAlgorithmName
        self.revisionsHashValue = revisionsHashValue
        self.revisionsSaltValue = revisionsSaltValue
        self.revisionsSpinCount = revisionsSpinCount
        self.workbookAlgorithmName = workbookAlgorithmName
        self.workbookHashValue = workbookHashValue
        self.workbookSaltValue = workbookSaltValue
        self.workbookSpinCount = workbookSpinCount

    def set_workbook_password(self, value='', already_hashed=False):
        """Set a password on this workbook."""
        if not already_hashed:
            value = hash_password(value)
        self._workbook_password = value

    @property
    def workbookPassword(self):
        """Return the workbook password value, regardless of hash."""
        return self._workbook_password

    @workbookPassword.setter
    def workbookPassword(self, value):
        """Set a workbook password directly, forcing a hash step."""
        self.set_workbook_password(value)

    def set_revisions_password(self, value='', already_hashed=False):
        """Set a revision password on this workbook."""
        if not already_hashed:
            value = hash_password(value)
        self._revisions_password = value

    @property
    def revisionsPassword(self):
        """Return the revisions password value, regardless of hash."""
        return self._revisions_password

    @revisionsPassword.setter
    def revisionsPassword(self, value):
        """Set a revisions password directly, forcing a hash step."""
        self.set_revisions_password(value)

    @classmethod
    def from_tree(cls, node):
        """Don't hash passwords when deserialising from XML"""
        self = super().from_tree(node)
        if self.workbookPassword:
            self.set_workbook_password(node.get('workbookPassword'), already_hashed=True)
        if self.revisionsPassword:
            self.set_revisions_password(node.get('revisionsPassword'), already_hashed=True)
        return self

# Backwards compatibility
DocumentSecurity = WorkbookProtection


class FileSharing(Serialisable):

    tagname = "fileSharing"

    readOnlyRecommended = Bool(allow_none=True)
    userName = String(allow_none=True)
    reservationPassword = HexBinary(allow_none=True)
    algorithmName = String(allow_none=True)
    hashValue = Base64Binary(allow_none=True)
    saltValue = Base64Binary(allow_none=True)
    spinCount = Integer(allow_none=True)

    def __init__(self,
                 readOnlyRecommended=None,
                 userName=None,
                 reservationPassword=None,
                 algorithmName=None,
                 hashValue=None,
                 saltValue=None,
                 spinCount=None,
                ):
        self.readOnlyRecommended = readOnlyRecommended
        self.userName = userName
        self.reservationPassword = reservationPassword
        self.algorithmName = algorithmName
        self.hashValue = hashValue
        self.saltValue = saltValue
        self.spinCount = spinCount

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\packaging\utils.py ===
# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.

from __future__ import annotations

import functools
import re
from typing import NewType, Tuple, Union, cast

from .tags import Tag, parse_tag
from .version import InvalidVersion, Version, _TrimmedRelease

BuildTag = Union[Tuple[()], Tuple[int, str]]
NormalizedName = NewType("NormalizedName", str)


class InvalidName(ValueError):
    """
    An invalid distribution name; users should refer to the packaging user guide.
    """


class InvalidWheelFilename(ValueError):
    """
    An invalid wheel filename was found, users should refer to PEP 427.
    """


class InvalidSdistFilename(ValueError):
    """
    An invalid sdist filename was found, users should refer to the packaging user guide.
    """


# Core metadata spec for `Name`
_validate_regex = re.compile(
    r"^([A-Z0-9]|[A-Z0-9][A-Z0-9._-]*[A-Z0-9])$", re.IGNORECASE
)
_canonicalize_regex = re.compile(r"[-_.]+")
_normalized_regex = re.compile(r"^([a-z0-9]|[a-z0-9]([a-z0-9-](?!--))*[a-z0-9])$")
# PEP 427: The build number must start with a digit.
_build_tag_regex = re.compile(r"(\d+)(.*)")


def canonicalize_name(name: str, *, validate: bool = False) -> NormalizedName:
    if validate and not _validate_regex.match(name):
        raise InvalidName(f"name is invalid: {name!r}")
    # This is taken from PEP 503.
    value = _canonicalize_regex.sub("-", name).lower()
    return cast(NormalizedName, value)


def is_normalized_name(name: str) -> bool:
    return _normalized_regex.match(name) is not None


@functools.singledispatch
def canonicalize_version(
    version: Version | str, *, strip_trailing_zero: bool = True
) -> str:
    """
    Return a canonical form of a version as a string.

    >>> canonicalize_version('1.0.1')
    '1.0.1'

    Per PEP 625, versions may have multiple canonical forms, differing
    only by trailing zeros.

    >>> canonicalize_version('1.0.0')
    '1'
    >>> canonicalize_version('1.0.0', strip_trailing_zero=False)
    '1.0.0'

    Invalid versions are returned unaltered.

    >>> canonicalize_version('foo bar baz')
    'foo bar baz'
    """
    return str(_TrimmedRelease(str(version)) if strip_trailing_zero else version)


@canonicalize_version.register
def _(version: str, *, strip_trailing_zero: bool = True) -> str:
    try:
        parsed = Version(version)
    except InvalidVersion:
        # Legacy versions cannot be normalized
        return version
    return canonicalize_version(parsed, strip_trailing_zero=strip_trailing_zero)


def parse_wheel_filename(
    filename: str,
) -> tuple[NormalizedName, Version, BuildTag, frozenset[Tag]]:
    if not filename.endswith(".whl"):
        raise InvalidWheelFilename(
            f"Invalid wheel filename (extension must be '.whl'): {filename!r}"
        )

    filename = filename[:-4]
    dashes = filename.count("-")
    if dashes not in (4, 5):
        raise InvalidWheelFilename(
            f"Invalid wheel filename (wrong number of parts): {filename!r}"
        )

    parts = filename.split("-", dashes - 2)
    name_part = parts[0]
    # See PEP 427 for the rules on escaping the project name.
    if "__" in name_part or re.match(r"^[\w\d._]*$", name_part, re.UNICODE) is None:
        raise InvalidWheelFilename(f"Invalid project name: {filename!r}")
    name = canonicalize_name(name_part)

    try:
        version = Version(parts[1])
    except InvalidVersion as e:
        raise InvalidWheelFilename(
            f"Invalid wheel filename (invalid version): {filename!r}"
        ) from e

    if dashes == 5:
        build_part = parts[2]
        build_match = _build_tag_regex.match(build_part)
        if build_match is None:
            raise InvalidWheelFilename(
                f"Invalid build number: {build_part} in {filename!r}"
            )
        build = cast(BuildTag, (int(build_match.group(1)), build_match.group(2)))
    else:
        build = ()
    tags = parse_tag(parts[-1])
    return (name, version, build, tags)


def parse_sdist_filename(filename: str) -> tuple[NormalizedName, Version]:
    if filename.endswith(".tar.gz"):
        file_stem = filename[: -len(".tar.gz")]
    elif filename.endswith(".zip"):
        file_stem = filename[: -len(".zip")]
    else:
        raise InvalidSdistFilename(
            f"Invalid sdist filename (extension must be '.tar.gz' or '.zip'):"
            f" {filename!r}"
        )

    # We are requiring a PEP 440 version, which cannot contain dashes,
    # so we split on the last dash.
    name_part, sep, version_part = file_stem.rpartition("-")
    if not sep:
        raise InvalidSdistFilename(f"Invalid sdist filename: {filename!r}")

    name = canonicalize_name(name_part)

    try:
        version = Version(version_part)
    except InvalidVersion as e:
        raise InvalidSdistFilename(
            f"Invalid sdist filename (invalid version): {filename!r}"
        ) from e

    return (name, version)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\vcs\mercurial.py ===
import configparser
import logging
import os
from typing import List, Optional, Tuple

from pip._internal.exceptions import BadCommand, InstallationError
from pip._internal.utils.misc import HiddenText, display_path
from pip._internal.utils.subprocess import make_command
from pip._internal.utils.urls import path_to_url
from pip._internal.vcs.versioncontrol import (
    RevOptions,
    VersionControl,
    find_path_to_project_root_from_repo_root,
    vcs,
)

logger = logging.getLogger(__name__)


class Mercurial(VersionControl):
    name = "hg"
    dirname = ".hg"
    repo_name = "clone"
    schemes = (
        "hg+file",
        "hg+http",
        "hg+https",
        "hg+ssh",
        "hg+static-http",
    )

    @staticmethod
    def get_base_rev_args(rev: str) -> List[str]:
        return [f"--rev={rev}"]

    def fetch_new(
        self, dest: str, url: HiddenText, rev_options: RevOptions, verbosity: int
    ) -> None:
        rev_display = rev_options.to_display()
        logger.info(
            "Cloning hg %s%s to %s",
            url,
            rev_display,
            display_path(dest),
        )
        if verbosity <= 0:
            flags: Tuple[str, ...] = ("--quiet",)
        elif verbosity == 1:
            flags = ()
        elif verbosity == 2:
            flags = ("--verbose",)
        else:
            flags = ("--verbose", "--debug")
        self.run_command(make_command("clone", "--noupdate", *flags, url, dest))
        self.run_command(
            make_command("update", *flags, rev_options.to_args()),
            cwd=dest,
        )

    def switch(self, dest: str, url: HiddenText, rev_options: RevOptions) -> None:
        repo_config = os.path.join(dest, self.dirname, "hgrc")
        config = configparser.RawConfigParser()
        try:
            config.read(repo_config)
            config.set("paths", "default", url.secret)
            with open(repo_config, "w") as config_file:
                config.write(config_file)
        except (OSError, configparser.NoSectionError) as exc:
            logger.warning("Could not switch Mercurial repository to %s: %s", url, exc)
        else:
            cmd_args = make_command("update", "-q", rev_options.to_args())
            self.run_command(cmd_args, cwd=dest)

    def update(self, dest: str, url: HiddenText, rev_options: RevOptions) -> None:
        self.run_command(["pull", "-q"], cwd=dest)
        cmd_args = make_command("update", "-q", rev_options.to_args())
        self.run_command(cmd_args, cwd=dest)

    @classmethod
    def get_remote_url(cls, location: str) -> str:
        url = cls.run_command(
            ["showconfig", "paths.default"],
            show_stdout=False,
            stdout_only=True,
            cwd=location,
        ).strip()
        if cls._is_local_repository(url):
            url = path_to_url(url)
        return url.strip()

    @classmethod
    def get_revision(cls, location: str) -> str:
        """
        Return the repository-local changeset revision number, as an integer.
        """
        current_revision = cls.run_command(
            ["parents", "--template={rev}"],
            show_stdout=False,
            stdout_only=True,
            cwd=location,
        ).strip()
        return current_revision

    @classmethod
    def get_requirement_revision(cls, location: str) -> str:
        """
        Return the changeset identification hash, as a 40-character
        hexadecimal string
        """
        current_rev_hash = cls.run_command(
            ["parents", "--template={node}"],
            show_stdout=False,
            stdout_only=True,
            cwd=location,
        ).strip()
        return current_rev_hash

    @classmethod
    def is_commit_id_equal(cls, dest: str, name: Optional[str]) -> bool:
        """Always assume the versions don't match"""
        return False

    @classmethod
    def get_subdirectory(cls, location: str) -> Optional[str]:
        """
        Return the path to Python project root, relative to the repo root.
        Return None if the project root is in the repo root.
        """
        # find the repo root
        repo_root = cls.run_command(
            ["root"], show_stdout=False, stdout_only=True, cwd=location
        ).strip()
        if not os.path.isabs(repo_root):
            repo_root = os.path.abspath(os.path.join(location, repo_root))
        return find_path_to_project_root_from_repo_root(location, repo_root)

    @classmethod
    def get_repository_root(cls, location: str) -> Optional[str]:
        loc = super().get_repository_root(location)
        if loc:
            return loc
        try:
            r = cls.run_command(
                ["root"],
                cwd=location,
                show_stdout=False,
                stdout_only=True,
                on_returncode="raise",
                log_failed_cmd=False,
            )
        except BadCommand:
            logger.debug(
                "could not determine if %s is under hg control "
                "because hg is not available",
                location,
            )
            return None
        except InstallationError:
            return None
        return os.path.normpath(r.rstrip("\r\n"))


vcs.register(Mercurial)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\packaging\utils.py ===
# This file is dual licensed under the terms of the Apache License, Version
# 2.0, and the BSD License. See the LICENSE file in the root of this repository
# for complete details.

from __future__ import annotations

import functools
import re
from typing import NewType, Tuple, Union, cast

from .tags import Tag, parse_tag
from .version import InvalidVersion, Version, _TrimmedRelease

BuildTag = Union[Tuple[()], Tuple[int, str]]
NormalizedName = NewType("NormalizedName", str)


class InvalidName(ValueError):
    """
    An invalid distribution name; users should refer to the packaging user guide.
    """


class InvalidWheelFilename(ValueError):
    """
    An invalid wheel filename was found, users should refer to PEP 427.
    """


class InvalidSdistFilename(ValueError):
    """
    An invalid sdist filename was found, users should refer to the packaging user guide.
    """


# Core metadata spec for `Name`
_validate_regex = re.compile(
    r"^([A-Z0-9]|[A-Z0-9][A-Z0-9._-]*[A-Z0-9])$", re.IGNORECASE
)
_canonicalize_regex = re.compile(r"[-_.]+")
_normalized_regex = re.compile(r"^([a-z0-9]|[a-z0-9]([a-z0-9-](?!--))*[a-z0-9])$")
# PEP 427: The build number must start with a digit.
_build_tag_regex = re.compile(r"(\d+)(.*)")


def canonicalize_name(name: str, *, validate: bool = False) -> NormalizedName:
    if validate and not _validate_regex.match(name):
        raise InvalidName(f"name is invalid: {name!r}")
    # This is taken from PEP 503.
    value = _canonicalize_regex.sub("-", name).lower()
    return cast(NormalizedName, value)


def is_normalized_name(name: str) -> bool:
    return _normalized_regex.match(name) is not None


@functools.singledispatch
def canonicalize_version(
    version: Version | str, *, strip_trailing_zero: bool = True
) -> str:
    """
    Return a canonical form of a version as a string.

    >>> canonicalize_version('1.0.1')
    '1.0.1'

    Per PEP 625, versions may have multiple canonical forms, differing
    only by trailing zeros.

    >>> canonicalize_version('1.0.0')
    '1'
    >>> canonicalize_version('1.0.0', strip_trailing_zero=False)
    '1.0.0'

    Invalid versions are returned unaltered.

    >>> canonicalize_version('foo bar baz')
    'foo bar baz'
    """
    return str(_TrimmedRelease(str(version)) if strip_trailing_zero else version)


@canonicalize_version.register
def _(version: str, *, strip_trailing_zero: bool = True) -> str:
    try:
        parsed = Version(version)
    except InvalidVersion:
        # Legacy versions cannot be normalized
        return version
    return canonicalize_version(parsed, strip_trailing_zero=strip_trailing_zero)


def parse_wheel_filename(
    filename: str,
) -> tuple[NormalizedName, Version, BuildTag, frozenset[Tag]]:
    if not filename.endswith(".whl"):
        raise InvalidWheelFilename(
            f"Invalid wheel filename (extension must be '.whl'): {filename!r}"
        )

    filename = filename[:-4]
    dashes = filename.count("-")
    if dashes not in (4, 5):
        raise InvalidWheelFilename(
            f"Invalid wheel filename (wrong number of parts): {filename!r}"
        )

    parts = filename.split("-", dashes - 2)
    name_part = parts[0]
    # See PEP 427 for the rules on escaping the project name.
    if "__" in name_part or re.match(r"^[\w\d._]*$", name_part, re.UNICODE) is None:
        raise InvalidWheelFilename(f"Invalid project name: {filename!r}")
    name = canonicalize_name(name_part)

    try:
        version = Version(parts[1])
    except InvalidVersion as e:
        raise InvalidWheelFilename(
            f"Invalid wheel filename (invalid version): {filename!r}"
        ) from e

    if dashes == 5:
        build_part = parts[2]
        build_match = _build_tag_regex.match(build_part)
        if build_match is None:
            raise InvalidWheelFilename(
                f"Invalid build number: {build_part} in {filename!r}"
            )
        build = cast(BuildTag, (int(build_match.group(1)), build_match.group(2)))
    else:
        build = ()
    tags = parse_tag(parts[-1])
    return (name, version, build, tags)


def parse_sdist_filename(filename: str) -> tuple[NormalizedName, Version]:
    if filename.endswith(".tar.gz"):
        file_stem = filename[: -len(".tar.gz")]
    elif filename.endswith(".zip"):
        file_stem = filename[: -len(".zip")]
    else:
        raise InvalidSdistFilename(
            f"Invalid sdist filename (extension must be '.tar.gz' or '.zip'):"
            f" {filename!r}"
        )

    # We are requiring a PEP 440 version, which cannot contain dashes,
    # so we split on the last dash.
    name_part, sep, version_part = file_stem.rpartition("-")
    if not sep:
        raise InvalidSdistFilename(f"Invalid sdist filename: {filename!r}")

    name = canonicalize_name(name_part)

    try:
        version = Version(version_part)
    except InvalidVersion as e:
        raise InvalidSdistFilename(
            f"Invalid sdist filename (invalid version): {filename!r}"
        ) from e

    return (name, version)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v135\extensions.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Extensions (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

class StorageArea(enum.Enum):
    '''
    Storage areas.
    '''
    SESSION = "session"
    LOCAL = "local"
    SYNC = "sync"
    MANAGED = "managed"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


def load_unpacked(
        path: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,str]:
    '''
    Installs an unpacked extension from the filesystem similar to
    --load-extension CLI flags. Returns extension ID once the extension
    has been installed. Available if the client is connected using the
    --remote-debugging-pipe flag and the --enable-unsafe-extension-debugging
    flag is set.

    :param path: Absolute file path.
    :returns: Extension id.
    '''
    params: T_JSON_DICT = dict()
    params['path'] = path
    cmd_dict: T_JSON_DICT = {
        'method': 'Extensions.loadUnpacked',
        'params': params,
    }
    json = yield cmd_dict
    return str(json['id'])


def uninstall(
        id_: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Uninstalls an unpacked extension (others not supported) from the profile.
    Available if the client is connected using the --remote-debugging-pipe flag
    and the --enable-unsafe-extension-debugging.

    :param id_: Extension id.
    '''
    params: T_JSON_DICT = dict()
    params['id'] = id_
    cmd_dict: T_JSON_DICT = {
        'method': 'Extensions.uninstall',
        'params': params,
    }
    json = yield cmd_dict


def get_storage_items(
        id_: str,
        storage_area: StorageArea,
        keys: typing.Optional[typing.List[str]] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,dict]:
    '''
    Gets data from extension storage in the given ``storageArea``. If ``keys`` is
    specified, these are used to filter the result.

    :param id_: ID of extension.
    :param storage_area: StorageArea to retrieve data from.
    :param keys: *(Optional)* Keys to retrieve.
    :returns:
    '''
    params: T_JSON_DICT = dict()
    params['id'] = id_
    params['storageArea'] = storage_area.to_json()
    if keys is not None:
        params['keys'] = [i for i in keys]
    cmd_dict: T_JSON_DICT = {
        'method': 'Extensions.getStorageItems',
        'params': params,
    }
    json = yield cmd_dict
    return dict(json['data'])


def remove_storage_items(
        id_: str,
        storage_area: StorageArea,
        keys: typing.List[str]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Removes ``keys`` from extension storage in the given ``storageArea``.

    :param id_: ID of extension.
    :param storage_area: StorageArea to remove data from.
    :param keys: Keys to remove.
    '''
    params: T_JSON_DICT = dict()
    params['id'] = id_
    params['storageArea'] = storage_area.to_json()
    params['keys'] = [i for i in keys]
    cmd_dict: T_JSON_DICT = {
        'method': 'Extensions.removeStorageItems',
        'params': params,
    }
    json = yield cmd_dict


def clear_storage_items(
        id_: str,
        storage_area: StorageArea
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Clears extension storage in the given ``storageArea``.

    :param id_: ID of extension.
    :param storage_area: StorageArea to remove data from.
    '''
    params: T_JSON_DICT = dict()
    params['id'] = id_
    params['storageArea'] = storage_area.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Extensions.clearStorageItems',
        'params': params,
    }
    json = yield cmd_dict


def set_storage_items(
        id_: str,
        storage_area: StorageArea,
        values: dict
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets ``values`` in extension storage in the given ``storageArea``. The provided ``values``
    will be merged with existing values in the storage area.

    :param id_: ID of extension.
    :param storage_area: StorageArea to set data in.
    :param values: Values to set.
    '''
    params: T_JSON_DICT = dict()
    params['id'] = id_
    params['storageArea'] = storage_area.to_json()
    params['values'] = values
    cmd_dict: T_JSON_DICT = {
        'method': 'Extensions.setStorageItems',
        'params': params,
    }
    json = yield cmd_dict

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v136\extensions.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Extensions (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

class StorageArea(enum.Enum):
    '''
    Storage areas.
    '''
    SESSION = "session"
    LOCAL = "local"
    SYNC = "sync"
    MANAGED = "managed"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


def load_unpacked(
        path: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,str]:
    '''
    Installs an unpacked extension from the filesystem similar to
    --load-extension CLI flags. Returns extension ID once the extension
    has been installed. Available if the client is connected using the
    --remote-debugging-pipe flag and the --enable-unsafe-extension-debugging
    flag is set.

    :param path: Absolute file path.
    :returns: Extension id.
    '''
    params: T_JSON_DICT = dict()
    params['path'] = path
    cmd_dict: T_JSON_DICT = {
        'method': 'Extensions.loadUnpacked',
        'params': params,
    }
    json = yield cmd_dict
    return str(json['id'])


def uninstall(
        id_: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Uninstalls an unpacked extension (others not supported) from the profile.
    Available if the client is connected using the --remote-debugging-pipe flag
    and the --enable-unsafe-extension-debugging.

    :param id_: Extension id.
    '''
    params: T_JSON_DICT = dict()
    params['id'] = id_
    cmd_dict: T_JSON_DICT = {
        'method': 'Extensions.uninstall',
        'params': params,
    }
    json = yield cmd_dict


def get_storage_items(
        id_: str,
        storage_area: StorageArea,
        keys: typing.Optional[typing.List[str]] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,dict]:
    '''
    Gets data from extension storage in the given ``storageArea``. If ``keys`` is
    specified, these are used to filter the result.

    :param id_: ID of extension.
    :param storage_area: StorageArea to retrieve data from.
    :param keys: *(Optional)* Keys to retrieve.
    :returns:
    '''
    params: T_JSON_DICT = dict()
    params['id'] = id_
    params['storageArea'] = storage_area.to_json()
    if keys is not None:
        params['keys'] = [i for i in keys]
    cmd_dict: T_JSON_DICT = {
        'method': 'Extensions.getStorageItems',
        'params': params,
    }
    json = yield cmd_dict
    return dict(json['data'])


def remove_storage_items(
        id_: str,
        storage_area: StorageArea,
        keys: typing.List[str]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Removes ``keys`` from extension storage in the given ``storageArea``.

    :param id_: ID of extension.
    :param storage_area: StorageArea to remove data from.
    :param keys: Keys to remove.
    '''
    params: T_JSON_DICT = dict()
    params['id'] = id_
    params['storageArea'] = storage_area.to_json()
    params['keys'] = [i for i in keys]
    cmd_dict: T_JSON_DICT = {
        'method': 'Extensions.removeStorageItems',
        'params': params,
    }
    json = yield cmd_dict


def clear_storage_items(
        id_: str,
        storage_area: StorageArea
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Clears extension storage in the given ``storageArea``.

    :param id_: ID of extension.
    :param storage_area: StorageArea to remove data from.
    '''
    params: T_JSON_DICT = dict()
    params['id'] = id_
    params['storageArea'] = storage_area.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Extensions.clearStorageItems',
        'params': params,
    }
    json = yield cmd_dict


def set_storage_items(
        id_: str,
        storage_area: StorageArea,
        values: dict
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets ``values`` in extension storage in the given ``storageArea``. The provided ``values``
    will be merged with existing values in the storage area.

    :param id_: ID of extension.
    :param storage_area: StorageArea to set data in.
    :param values: Values to set.
    '''
    params: T_JSON_DICT = dict()
    params['id'] = id_
    params['storageArea'] = storage_area.to_json()
    params['values'] = values
    cmd_dict: T_JSON_DICT = {
        'method': 'Extensions.setStorageItems',
        'params': params,
    }
    json = yield cmd_dict

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v137\extensions.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Extensions (experimental)
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing

class StorageArea(enum.Enum):
    '''
    Storage areas.
    '''
    SESSION = "session"
    LOCAL = "local"
    SYNC = "sync"
    MANAGED = "managed"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


def load_unpacked(
        path: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,str]:
    '''
    Installs an unpacked extension from the filesystem similar to
    --load-extension CLI flags. Returns extension ID once the extension
    has been installed. Available if the client is connected using the
    --remote-debugging-pipe flag and the --enable-unsafe-extension-debugging
    flag is set.

    :param path: Absolute file path.
    :returns: Extension id.
    '''
    params: T_JSON_DICT = dict()
    params['path'] = path
    cmd_dict: T_JSON_DICT = {
        'method': 'Extensions.loadUnpacked',
        'params': params,
    }
    json = yield cmd_dict
    return str(json['id'])


def uninstall(
        id_: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Uninstalls an unpacked extension (others not supported) from the profile.
    Available if the client is connected using the --remote-debugging-pipe flag
    and the --enable-unsafe-extension-debugging.

    :param id_: Extension id.
    '''
    params: T_JSON_DICT = dict()
    params['id'] = id_
    cmd_dict: T_JSON_DICT = {
        'method': 'Extensions.uninstall',
        'params': params,
    }
    json = yield cmd_dict


def get_storage_items(
        id_: str,
        storage_area: StorageArea,
        keys: typing.Optional[typing.List[str]] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,dict]:
    '''
    Gets data from extension storage in the given ``storageArea``. If ``keys`` is
    specified, these are used to filter the result.

    :param id_: ID of extension.
    :param storage_area: StorageArea to retrieve data from.
    :param keys: *(Optional)* Keys to retrieve.
    :returns:
    '''
    params: T_JSON_DICT = dict()
    params['id'] = id_
    params['storageArea'] = storage_area.to_json()
    if keys is not None:
        params['keys'] = [i for i in keys]
    cmd_dict: T_JSON_DICT = {
        'method': 'Extensions.getStorageItems',
        'params': params,
    }
    json = yield cmd_dict
    return dict(json['data'])


def remove_storage_items(
        id_: str,
        storage_area: StorageArea,
        keys: typing.List[str]
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Removes ``keys`` from extension storage in the given ``storageArea``.

    :param id_: ID of extension.
    :param storage_area: StorageArea to remove data from.
    :param keys: Keys to remove.
    '''
    params: T_JSON_DICT = dict()
    params['id'] = id_
    params['storageArea'] = storage_area.to_json()
    params['keys'] = [i for i in keys]
    cmd_dict: T_JSON_DICT = {
        'method': 'Extensions.removeStorageItems',
        'params': params,
    }
    json = yield cmd_dict


def clear_storage_items(
        id_: str,
        storage_area: StorageArea
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Clears extension storage in the given ``storageArea``.

    :param id_: ID of extension.
    :param storage_area: StorageArea to remove data from.
    '''
    params: T_JSON_DICT = dict()
    params['id'] = id_
    params['storageArea'] = storage_area.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Extensions.clearStorageItems',
        'params': params,
    }
    json = yield cmd_dict


def set_storage_items(
        id_: str,
        storage_area: StorageArea,
        values: dict
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Sets ``values`` in extension storage in the given ``storageArea``. The provided ``values``
    will be merged with existing values in the storage area.

    :param id_: ID of extension.
    :param storage_area: StorageArea to set data in.
    :param values: Values to set.
    '''
    params: T_JSON_DICT = dict()
    params['id'] = id_
    params['storageArea'] = storage_area.to_json()
    params['values'] = values
    cmd_dict: T_JSON_DICT = {
        'method': 'Extensions.setStorageItems',
        'params': params,
    }
    json = yield cmd_dict

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_core\_unbounded_queue.py ===
from __future__ import annotations

from typing import TYPE_CHECKING, Generic, TypeVar

import attrs

from .. import _core
from .._deprecate import deprecated
from .._util import final

T = TypeVar("T")

if TYPE_CHECKING:
    from typing_extensions import Self


@attrs.frozen
class UnboundedQueueStatistics:
    """An object containing debugging information.

    Currently, the following fields are defined:

    * ``qsize``: The number of items currently in the queue.
    * ``tasks_waiting``: The number of tasks blocked on this queue's
      :meth:`get_batch` method.

    """

    qsize: int
    tasks_waiting: int


@final
class UnboundedQueue(Generic[T]):
    """An unbounded queue suitable for certain unusual forms of inter-task
    communication.

    This class is designed for use as a queue in cases where the producer for
    some reason cannot be subjected to back-pressure, i.e., :meth:`put_nowait`
    has to always succeed. In order to prevent the queue backlog from actually
    growing without bound, the consumer API is modified to dequeue items in
    "batches". If a consumer task processes each batch without yielding, then
    this helps achieve (but does not guarantee) an effective bound on the
    queue's memory use, at the cost of potentially increasing system latencies
    in general. You should generally prefer to use a memory channel
    instead if you can.

    Currently each batch completely empties the queue, but `this may change in
    the future <https://github.com/python-trio/trio/issues/51>`__.

    A :class:`UnboundedQueue` object can be used as an asynchronous iterator,
    where each iteration returns a new batch of items. I.e., these two loops
    are equivalent::

       async for batch in queue:
           ...

       while True:
           obj = await queue.get_batch()
           ...

    """

    @deprecated(
        "0.9.0",
        issue=497,
        thing="trio.lowlevel.UnboundedQueue",
        instead="trio.open_memory_channel(math.inf)",
        use_triodeprecationwarning=True,
    )
    def __init__(self) -> None:
        self._lot = _core.ParkingLot()
        self._data: list[T] = []
        # used to allow handoff from put to the first task in the lot
        self._can_get = False

    def __repr__(self) -> str:
        return f"<UnboundedQueue holding {len(self._data)} items>"

    def qsize(self) -> int:
        """Returns the number of items currently in the queue."""
        return len(self._data)

    def empty(self) -> bool:
        """Returns True if the queue is empty, False otherwise.

        There is some subtlety to interpreting this method's return value: see
        `issue #63 <https://github.com/python-trio/trio/issues/63>`__.

        """
        return not self._data

    @_core.enable_ki_protection
    def put_nowait(self, obj: T) -> None:
        """Put an object into the queue, without blocking.

        This always succeeds, because the queue is unbounded. We don't provide
        a blocking ``put`` method, because it would never need to block.

        Args:
          obj (object): The object to enqueue.

        """
        if not self._data:
            assert not self._can_get
            if self._lot:
                self._lot.unpark(count=1)
            else:
                self._can_get = True
        self._data.append(obj)

    def _get_batch_protected(self) -> list[T]:
        data = self._data.copy()
        self._data.clear()
        self._can_get = False
        return data

    def get_batch_nowait(self) -> list[T]:
        """Attempt to get the next batch from the queue, without blocking.

        Returns:
          list: A list of dequeued items, in order. On a successful call this
              list is always non-empty; if it would be empty we raise
              :exc:`~trio.WouldBlock` instead.

        Raises:
          ~trio.WouldBlock: if the queue is empty.

        """
        if not self._can_get:
            raise _core.WouldBlock
        return self._get_batch_protected()

    async def get_batch(self) -> list[T]:
        """Get the next batch from the queue, blocking as necessary.

        Returns:
          list: A list of dequeued items, in order. This list is always
              non-empty.

        """
        await _core.checkpoint_if_cancelled()
        if not self._can_get:
            await self._lot.park()
            return self._get_batch_protected()
        else:
            try:
                return self._get_batch_protected()
            finally:
                await _core.cancel_shielded_checkpoint()

    def statistics(self) -> UnboundedQueueStatistics:
        """Return an :class:`UnboundedQueueStatistics` object containing debugging information."""
        return UnboundedQueueStatistics(
            qsize=len(self._data),
            tasks_waiting=self._lot.statistics().tasks_waiting,
        )

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> list[T]:
        return await self.get_batch()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\webdriver_manager\core\os_manager.py ===
import platform
import sys

from webdriver_manager.core.utils import linux_browser_apps_to_cmd, windows_browser_apps_to_cmd, \
    read_version_from_cmd


class ChromeType(object):
    GOOGLE = "google-chrome"
    CHROMIUM = "chromium"
    BRAVE = "brave-browser"
    MSEDGE = "edge"


class OSType(object):
    LINUX = "linux"
    MAC = "mac"
    WIN = "win"


PATTERN = {
    ChromeType.CHROMIUM: r"\d+\.\d+\.\d+",
    ChromeType.GOOGLE: r"\d+\.\d+\.\d+",
    ChromeType.MSEDGE: r"\d+\.\d+\.\d+",
    "brave-browser": r"\d+\.\d+\.\d+(\.\d+)?",
    "firefox": r"(\d+.\d+)",
}


class OperationSystemManager(object):

    def __init__(self, os_type=None):
        self._os_type = os_type

    @staticmethod
    def get_os_name():
        pl = sys.platform
        if pl == "linux" or pl == "linux2":
            return OSType.LINUX
        elif pl == "darwin":
            return OSType.MAC
        elif pl == "win32" or pl == "cygwin":
            return OSType.WIN

    @staticmethod
    def get_os_architecture():
        if platform.machine().endswith("64"):
            return 64
        else:
            return 32

    def get_os_type(self):
        if self._os_type:
            return self._os_type
        return f"{self.get_os_name()}{self.get_os_architecture()}"

    @staticmethod
    def is_arch(os_sys_type):
        if '_m1' in os_sys_type:
            return True
        return platform.processor() != 'i386'

    @staticmethod
    def is_mac_os(os_sys_type):
        return OSType.MAC in os_sys_type

    def get_browser_version_from_os(self, browser_type=None):
        """Return installed browser version."""
        cmd_mapping = {
            ChromeType.GOOGLE: {
                OSType.LINUX: linux_browser_apps_to_cmd(
                    "google-chrome",
                    "google-chrome-stable",
                    "google-chrome-beta",
                    "google-chrome-dev",
                ),
                OSType.MAC: r"/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --version",
                OSType.WIN: windows_browser_apps_to_cmd(
                    r'(Get-Item -Path "$env:PROGRAMFILES\Google\Chrome\Application\chrome.exe").VersionInfo.FileVersion',
                    r'(Get-Item -Path "$env:PROGRAMFILES (x86)\Google\Chrome\Application\chrome.exe").VersionInfo.FileVersion',
                    r'(Get-Item -Path "$env:LOCALAPPDATA\Google\Chrome\Application\chrome.exe").VersionInfo.FileVersion',
                    r'(Get-ItemProperty -Path Registry::"HKCU\SOFTWARE\Google\Chrome\BLBeacon").version',
                    r'(Get-ItemProperty -Path Registry::"HKLM\SOFTWARE\Wow6432Node\Microsoft\Windows\CurrentVersion\Uninstall\Google Chrome").version',
                ),
            },
            ChromeType.CHROMIUM: {
                OSType.LINUX: linux_browser_apps_to_cmd("chromium", "chromium-browser"),
                OSType.MAC: r"/Applications/Chromium.app/Contents/MacOS/Chromium --version",
                OSType.WIN: windows_browser_apps_to_cmd(
                    r'(Get-Item -Path "$env:PROGRAMFILES\Chromium\Application\chrome.exe").VersionInfo.FileVersion',
                    r'(Get-Item -Path "$env:PROGRAMFILES (x86)\Chromium\Application\chrome.exe").VersionInfo.FileVersion',
                    r'(Get-Item -Path "$env:LOCALAPPDATA\Chromium\Application\chrome.exe").VersionInfo.FileVersion',
                    r'(Get-ItemProperty -Path Registry::"HKCU\SOFTWARE\Chromium\BLBeacon").version',
                    r'(Get-ItemProperty -Path Registry::"HKLM\SOFTWARE\Wow6432Node\Microsoft\Windows\CurrentVersion\Uninstall\Chromium").version',
                ),
            },
            ChromeType.BRAVE: {
                OSType.LINUX: linux_browser_apps_to_cmd(
                    "brave-browser", "brave-browser-beta", "brave-browser-nightly"
                ),
                OSType.MAC: r"/Applications/Brave\ Browser.app/Contents/MacOS/Brave\ Browser --version",
                OSType.WIN: windows_browser_apps_to_cmd(
                    r'(Get-Item -Path "$env:PROGRAMFILES\BraveSoftware\Brave-Browser\Application\brave.exe").VersionInfo.FileVersion',
                    r'(Get-Item -Path "$env:PROGRAMFILES (x86)\BraveSoftware\Brave-Browser\Application\brave.exe").VersionInfo.FileVersion',
                    r'(Get-Item -Path "$env:LOCALAPPDATA\BraveSoftware\Brave-Browser\Application\brave.exe").VersionInfo.FileVersion',
                    r'(Get-ItemProperty -Path Registry::"HKCU\SOFTWARE\BraveSoftware\Brave-Browser\BLBeacon").version',
                    r'(Get-ItemProperty -Path Registry::"HKLM\SOFTWARE\Wow6432Node\Microsoft\Windows\CurrentVersion\Uninstall\BraveSoftware Brave-Browser").version',
                ),
            },
            ChromeType.MSEDGE: {
                OSType.LINUX: linux_browser_apps_to_cmd(
                    "microsoft-edge",
                    "microsoft-edge-stable",
                    "microsoft-edge-beta",
                    "microsoft-edge-dev",
                ),
                OSType.MAC: r"/Applications/Microsoft\ Edge.app/Contents/MacOS/Microsoft\ Edge --version",
                OSType.WIN: windows_browser_apps_to_cmd(
                    # stable edge
                    r'(Get-Item -Path "$env:PROGRAMFILES\Microsoft\Edge\Application\msedge.exe").VersionInfo.FileVersion',
                    r'(Get-Item -Path "$env:PROGRAMFILES (x86)\Microsoft\Edge\Application\msedge.exe").VersionInfo.FileVersion',
                    r'(Get-ItemProperty -Path Registry::"HKCU\SOFTWARE\Microsoft\Edge\BLBeacon").version',
                    r'(Get-ItemProperty -Path Registry::"HKLM\SOFTWARE\Microsoft\EdgeUpdate\Clients\{56EB18F8-8008-4CBD-B6D2-8C97FE7E9062}").pv',
                    # beta edge
                    r'(Get-Item -Path "$env:LOCALAPPDATA\Microsoft\Edge Beta\Application\msedge.exe").VersionInfo.FileVersion',
                    r'(Get-Item -Path "$env:PROGRAMFILES\Microsoft\Edge Beta\Application\msedge.exe").VersionInfo.FileVersion',
                    r'(Get-Item -Path "$env:PROGRAMFILES (x86)\Microsoft\Edge Beta\Application\msedge.exe").VersionInfo.FileVersion',
                    r'(Get-ItemProperty -Path Registry::"HKCU\SOFTWARE\Microsoft\Edge Beta\BLBeacon").version',
                    # dev edge
                    r'(Get-Item -Path "$env:LOCALAPPDATA\Microsoft\Edge Dev\Application\msedge.exe").VersionInfo.FileVersion',
                    r'(Get-Item -Path "$env:PROGRAMFILES\Microsoft\Edge Dev\Application\msedge.exe").VersionInfo.FileVersion',
                    r'(Get-Item -Path "$env:PROGRAMFILES (x86)\Microsoft\Edge Dev\Application\msedge.exe").VersionInfo.FileVersion',
                    r'(Get-ItemProperty -Path Registry::"HKCU\SOFTWARE\Microsoft\Edge Dev\BLBeacon").version',
                    # canary edge
                    r'(Get-Item -Path "$env:LOCALAPPDATA\Microsoft\Edge SxS\Application\msedge.exe").VersionInfo.FileVersion',
                    r'(Get-ItemProperty -Path Registry::"HKCU\SOFTWARE\Microsoft\Edge SxS\BLBeacon").version',
                    # highest edge
                    r"(Get-Item (Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\msedge.exe').'(Default)').VersionInfo.ProductVersion",
                    r"[System.Diagnostics.FileVersionInfo]::GetVersionInfo((Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\msedge.exe').'(Default)').ProductVersion",
                    r"Get-AppxPackage -Name *MicrosoftEdge.* | Foreach Version",
                    r'(Get-ItemProperty -Path Registry::"HKLM\SOFTWARE\Wow6432Node\Microsoft\Windows\CurrentVersion\Uninstall\Microsoft Edge").version',
                ),
            },
            "firefox": {
                OSType.LINUX: linux_browser_apps_to_cmd("firefox"),
                OSType.MAC: r"/Applications/Firefox.app/Contents/MacOS/firefox --version",
                OSType.WIN: windows_browser_apps_to_cmd(
                    r'(Get-Item -Path "$env:PROGRAMFILES\Mozilla Firefox\firefox.exe").VersionInfo.FileVersion',
                    r'(Get-Item -Path "$env:PROGRAMFILES (x86)\Mozilla Firefox\firefox.exe").VersionInfo.FileVersion',
                    r"(Get-Item (Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\firefox.exe').'(Default)').VersionInfo.ProductVersion",
                    r'(Get-ItemProperty -Path Registry::"HKLM\SOFTWARE\Mozilla\Mozilla Firefox").CurrentVersion',
                ),
            },
        }

        try:
            cmd_mapping = cmd_mapping[browser_type][OperationSystemManager.get_os_name()]
            pattern = PATTERN[browser_type]
            version = read_version_from_cmd(cmd_mapping, pattern)
            return version
        except Exception:
            return None
            # raise Exception("Can not get browser version from OS")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\attr\converters.py ===
# SPDX-License-Identifier: MIT

"""
Commonly useful converters.
"""

import typing

from ._compat import _AnnotationExtractor
from ._make import NOTHING, Converter, Factory, pipe


__all__ = [
    "default_if_none",
    "optional",
    "pipe",
    "to_bool",
]


def optional(converter):
    """
    A converter that allows an attribute to be optional. An optional attribute
    is one which can be set to `None`.

    Type annotations will be inferred from the wrapped converter's, if it has
    any.

    Args:
        converter (typing.Callable):
            the converter that is used for non-`None` values.

    .. versionadded:: 17.1.0
    """

    if isinstance(converter, Converter):

        def optional_converter(val, inst, field):
            if val is None:
                return None
            return converter(val, inst, field)

    else:

        def optional_converter(val):
            if val is None:
                return None
            return converter(val)

    xtr = _AnnotationExtractor(converter)

    t = xtr.get_first_param_type()
    if t:
        optional_converter.__annotations__["val"] = typing.Optional[t]

    rt = xtr.get_return_type()
    if rt:
        optional_converter.__annotations__["return"] = typing.Optional[rt]

    if isinstance(converter, Converter):
        return Converter(optional_converter, takes_self=True, takes_field=True)

    return optional_converter


def default_if_none(default=NOTHING, factory=None):
    """
    A converter that allows to replace `None` values by *default* or the result
    of *factory*.

    Args:
        default:
            Value to be used if `None` is passed. Passing an instance of
            `attrs.Factory` is supported, however the ``takes_self`` option is
            *not*.

        factory (typing.Callable):
            A callable that takes no parameters whose result is used if `None`
            is passed.

    Raises:
        TypeError: If **neither** *default* or *factory* is passed.

        TypeError: If **both** *default* and *factory* are passed.

        ValueError:
            If an instance of `attrs.Factory` is passed with
            ``takes_self=True``.

    .. versionadded:: 18.2.0
    """
    if default is NOTHING and factory is None:
        msg = "Must pass either `default` or `factory`."
        raise TypeError(msg)

    if default is not NOTHING and factory is not None:
        msg = "Must pass either `default` or `factory` but not both."
        raise TypeError(msg)

    if factory is not None:
        default = Factory(factory)

    if isinstance(default, Factory):
        if default.takes_self:
            msg = "`takes_self` is not supported by default_if_none."
            raise ValueError(msg)

        def default_if_none_converter(val):
            if val is not None:
                return val

            return default.factory()

    else:

        def default_if_none_converter(val):
            if val is not None:
                return val

            return default

    return default_if_none_converter


def to_bool(val):
    """
    Convert "boolean" strings (for example, from environment variables) to real
    booleans.

    Values mapping to `True`:

    - ``True``
    - ``"true"`` / ``"t"``
    - ``"yes"`` / ``"y"``
    - ``"on"``
    - ``"1"``
    - ``1``

    Values mapping to `False`:

    - ``False``
    - ``"false"`` / ``"f"``
    - ``"no"`` / ``"n"``
    - ``"off"``
    - ``"0"``
    - ``0``

    Raises:
        ValueError: For any other value.

    .. versionadded:: 21.3.0
    """
    if isinstance(val, str):
        val = val.lower()

    if val in (True, "true", "t", "yes", "y", "on", "1", 1):
        return True
    if val in (False, "false", "f", "no", "n", "off", "0", 0):
        return False

    msg = f"Cannot convert value to bool: {val!r}"
    raise ValueError(msg)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_core\_exceptions.py ===
"""
Various richly-typed exceptions, that also help us deal with string formatting
in python where it's easier.

By putting the formatting in `__str__`, we also avoid paying the cost for
users who silence the exceptions.
"""

def _unpack_tuple(tup):
    if len(tup) == 1:
        return tup[0]
    else:
        return tup


def _display_as_base(cls):
    """
    A decorator that makes an exception class look like its base.

    We use this to hide subclasses that are implementation details - the user
    should catch the base type, which is what the traceback will show them.

    Classes decorated with this decorator are subject to removal without a
    deprecation warning.
    """
    assert issubclass(cls, Exception)
    cls.__name__ = cls.__base__.__name__
    return cls


class UFuncTypeError(TypeError):
    """ Base class for all ufunc exceptions """
    def __init__(self, ufunc):
        self.ufunc = ufunc


@_display_as_base
class _UFuncNoLoopError(UFuncTypeError):
    """ Thrown when a ufunc loop cannot be found """
    def __init__(self, ufunc, dtypes):
        super().__init__(ufunc)
        self.dtypes = tuple(dtypes)

    def __str__(self):
        return (
            f"ufunc {self.ufunc.__name__!r} did not contain a loop with signature "
            f"matching types {_unpack_tuple(self.dtypes[:self.ufunc.nin])!r} "
            f"-> {_unpack_tuple(self.dtypes[self.ufunc.nin:])!r}"
        )


@_display_as_base
class _UFuncBinaryResolutionError(_UFuncNoLoopError):
    """ Thrown when a binary resolution fails """
    def __init__(self, ufunc, dtypes):
        super().__init__(ufunc, dtypes)
        assert len(self.dtypes) == 2

    def __str__(self):
        return (
            "ufunc {!r} cannot use operands with types {!r} and {!r}"
        ).format(
            self.ufunc.__name__, *self.dtypes
        )


@_display_as_base
class _UFuncCastingError(UFuncTypeError):
    def __init__(self, ufunc, casting, from_, to):
        super().__init__(ufunc)
        self.casting = casting
        self.from_ = from_
        self.to = to


@_display_as_base
class _UFuncInputCastingError(_UFuncCastingError):
    """ Thrown when a ufunc input cannot be casted """
    def __init__(self, ufunc, casting, from_, to, i):
        super().__init__(ufunc, casting, from_, to)
        self.in_i = i

    def __str__(self):
        # only show the number if more than one input exists
        i_str = f"{self.in_i} " if self.ufunc.nin != 1 else ""
        return (
            f"Cannot cast ufunc {self.ufunc.__name__!r} input {i_str}from "
            f"{self.from_!r} to {self.to!r} with casting rule {self.casting!r}"
        )


@_display_as_base
class _UFuncOutputCastingError(_UFuncCastingError):
    """ Thrown when a ufunc output cannot be casted """
    def __init__(self, ufunc, casting, from_, to, i):
        super().__init__(ufunc, casting, from_, to)
        self.out_i = i

    def __str__(self):
        # only show the number if more than one output exists
        i_str = f"{self.out_i} " if self.ufunc.nout != 1 else ""
        return (
            f"Cannot cast ufunc {self.ufunc.__name__!r} output {i_str}from "
            f"{self.from_!r} to {self.to!r} with casting rule {self.casting!r}"
        )


@_display_as_base
class _ArrayMemoryError(MemoryError):
    """ Thrown when an array cannot be allocated"""
    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    @property
    def _total_size(self):
        num_bytes = self.dtype.itemsize
        for dim in self.shape:
            num_bytes *= dim
        return num_bytes

    @staticmethod
    def _size_to_string(num_bytes):
        """ Convert a number of bytes into a binary size string """

        # https://en.wikipedia.org/wiki/Binary_prefix
        LOG2_STEP = 10
        STEP = 1024
        units = ['bytes', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB']

        unit_i = max(num_bytes.bit_length() - 1, 1) // LOG2_STEP
        unit_val = 1 << (unit_i * LOG2_STEP)
        n_units = num_bytes / unit_val
        del unit_val

        # ensure we pick a unit that is correct after rounding
        if round(n_units) == STEP:
            unit_i += 1
            n_units /= STEP

        # deal with sizes so large that we don't have units for them
        if unit_i >= len(units):
            new_unit_i = len(units) - 1
            n_units *= 1 << ((unit_i - new_unit_i) * LOG2_STEP)
            unit_i = new_unit_i

        unit_name = units[unit_i]
        # format with a sensible number of digits
        if unit_i == 0:
            # no decimal point on bytes
            return f'{n_units:.0f} {unit_name}'
        elif round(n_units) < 1000:
            # 3 significant figures, if none are dropped to the left of the .
            return f'{n_units:#.3g} {unit_name}'
        else:
            # just give all the digits otherwise
            return f'{n_units:#.0f} {unit_name}'

    def __str__(self):
        size_str = self._size_to_string(self._total_size)
        return (f"Unable to allocate {size_str} for an array with shape "
                f"{self.shape} and data type {self.dtype}")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\chart\plotarea.py ===
# Copyright (c) 2010-2024 openpyxl

from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import (
    Typed,
    Alias,
)
from openpyxl.descriptors.excel import (
    ExtensionList,
)
from openpyxl.descriptors.sequence import (
    MultiSequence,
    MultiSequencePart,
)
from openpyxl.descriptors.nested import (
    NestedBool,
)

from ._3d import _3DBase
from .area_chart import AreaChart, AreaChart3D
from .bar_chart import BarChart, BarChart3D
from .bubble_chart import BubbleChart
from .line_chart import LineChart, LineChart3D
from .pie_chart import PieChart, PieChart3D, ProjectedPieChart, DoughnutChart
from .radar_chart import RadarChart
from .scatter_chart import ScatterChart
from .stock_chart import StockChart
from .surface_chart import SurfaceChart, SurfaceChart3D
from .layout import Layout
from .shapes import GraphicalProperties
from .text import RichText

from .axis import (
    NumericAxis,
    TextAxis,
    SeriesAxis,
    DateAxis,
)


class DataTable(Serialisable):

    tagname = "dTable"

    showHorzBorder = NestedBool(allow_none=True)
    showVertBorder = NestedBool(allow_none=True)
    showOutline = NestedBool(allow_none=True)
    showKeys = NestedBool(allow_none=True)
    spPr = Typed(expected_type=GraphicalProperties, allow_none=True)
    graphicalProperties = Alias('spPr')
    txPr = Typed(expected_type=RichText, allow_none=True)
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = ('showHorzBorder', 'showVertBorder', 'showOutline',
                    'showKeys', 'spPr', 'txPr')

    def __init__(self,
                 showHorzBorder=None,
                 showVertBorder=None,
                 showOutline=None,
                 showKeys=None,
                 spPr=None,
                 txPr=None,
                 extLst=None,
                ):
        self.showHorzBorder = showHorzBorder
        self.showVertBorder = showVertBorder
        self.showOutline = showOutline
        self.showKeys = showKeys
        self.spPr = spPr
        self.txPr = txPr


class PlotArea(Serialisable):

    tagname = "plotArea"

    layout = Typed(expected_type=Layout, allow_none=True)
    dTable = Typed(expected_type=DataTable, allow_none=True)
    spPr = Typed(expected_type=GraphicalProperties, allow_none=True)
    graphicalProperties = Alias("spPr")
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    # at least one chart
    _charts = MultiSequence()
    areaChart = MultiSequencePart(expected_type=AreaChart, store="_charts")
    area3DChart = MultiSequencePart(expected_type=AreaChart3D, store="_charts")
    lineChart = MultiSequencePart(expected_type=LineChart, store="_charts")
    line3DChart = MultiSequencePart(expected_type=LineChart3D, store="_charts")
    stockChart = MultiSequencePart(expected_type=StockChart, store="_charts")
    radarChart = MultiSequencePart(expected_type=RadarChart, store="_charts")
    scatterChart = MultiSequencePart(expected_type=ScatterChart, store="_charts")
    pieChart = MultiSequencePart(expected_type=PieChart, store="_charts")
    pie3DChart = MultiSequencePart(expected_type=PieChart3D, store="_charts")
    doughnutChart = MultiSequencePart(expected_type=DoughnutChart, store="_charts")
    barChart = MultiSequencePart(expected_type=BarChart, store="_charts")
    bar3DChart = MultiSequencePart(expected_type=BarChart3D, store="_charts")
    ofPieChart = MultiSequencePart(expected_type=ProjectedPieChart, store="_charts")
    surfaceChart = MultiSequencePart(expected_type=SurfaceChart, store="_charts")
    surface3DChart = MultiSequencePart(expected_type=SurfaceChart3D, store="_charts")
    bubbleChart = MultiSequencePart(expected_type=BubbleChart, store="_charts")

    # axes
    _axes = MultiSequence()
    valAx = MultiSequencePart(expected_type=NumericAxis, store="_axes")
    catAx = MultiSequencePart(expected_type=TextAxis, store="_axes")
    dateAx = MultiSequencePart(expected_type=DateAxis, store="_axes")
    serAx = MultiSequencePart(expected_type=SeriesAxis, store="_axes")

    __elements__ = ('layout', '_charts', '_axes', 'dTable', 'spPr')

    def __init__(self,
                 layout=None,
                 dTable=None,
                 spPr=None,
                 _charts=(),
                 _axes=(),
                 extLst=None,
                ):
        self.layout = layout
        self.dTable = dTable
        self.spPr = spPr
        self._charts = _charts
        self._axes = _axes


    def to_tree(self, tagname=None, idx=None, namespace=None):
        axIds = {ax.axId for ax in self._axes}
        for chart in self._charts:
            for id, axis in chart._axes.items():
                if id not in axIds:
                    setattr(self, axis.tagname, axis)
                    axIds.add(id)

        return super().to_tree(tagname)


    @classmethod
    def from_tree(cls, node):
        self = super().from_tree(node)
        axes = dict((axis.axId, axis) for axis in self._axes)
        for chart in self._charts:
            if isinstance(chart, (ScatterChart, BubbleChart)):
                x, y = (axes[axId] for axId in chart.axId)
                chart.x_axis = x
                chart.y_axis = y
                continue

            for axId in chart.axId:
                axis = axes.get(axId)
                if axis is None and isinstance(chart, _3DBase):
                    # Series Axis can be optional
                    chart.z_axis = None
                    continue
                if axis.tagname in ("catAx", "dateAx"):
                    chart.x_axis = axis
                elif axis.tagname == "valAx":
                    chart.y_axis = axis
                elif axis.tagname == "serAx":
                    chart.z_axis = axis

        return self

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\metadata\__init__.py ===
import contextlib
import functools
import os
import sys
from typing import List, Literal, Optional, Protocol, Type, cast

from pip._internal.utils.deprecation import deprecated
from pip._internal.utils.misc import strtobool

from .base import BaseDistribution, BaseEnvironment, FilesystemWheel, MemoryWheel, Wheel

__all__ = [
    "BaseDistribution",
    "BaseEnvironment",
    "FilesystemWheel",
    "MemoryWheel",
    "Wheel",
    "get_default_environment",
    "get_environment",
    "get_wheel_distribution",
    "select_backend",
]


def _should_use_importlib_metadata() -> bool:
    """Whether to use the ``importlib.metadata`` or ``pkg_resources`` backend.

    By default, pip uses ``importlib.metadata`` on Python 3.11+, and
    ``pkg_resources`` otherwise. Up to Python 3.13, This can be
    overridden by a couple of ways:

    * If environment variable ``_PIP_USE_IMPORTLIB_METADATA`` is set, it
      dictates whether ``importlib.metadata`` is used, for Python <3.14.
    * On Python 3.11, 3.12 and 3.13, Python distributors can patch
      ``importlib.metadata`` to add a global constant
      ``_PIP_USE_IMPORTLIB_METADATA = False``. This makes pip use
      ``pkg_resources`` (unless the user set the aforementioned environment
      variable to *True*).

    On Python 3.14+, the ``pkg_resources`` backend cannot be used.
    """
    if sys.version_info >= (3, 14):
        # On Python >=3.14 we only support importlib.metadata.
        return True
    with contextlib.suppress(KeyError, ValueError):
        # On Python <3.14, if the environment variable is set, we obey what it says.
        return bool(strtobool(os.environ["_PIP_USE_IMPORTLIB_METADATA"]))
    if sys.version_info < (3, 11):
        # On Python <3.11, we always use pkg_resources, unless the environment
        # variable was set.
        return False
    # On Python 3.11, 3.12 and 3.13, we check if the global constant is set.
    import importlib.metadata

    return bool(getattr(importlib.metadata, "_PIP_USE_IMPORTLIB_METADATA", True))


def _emit_pkg_resources_deprecation_if_needed() -> None:
    if sys.version_info < (3, 11):
        # All pip versions supporting Python<=3.11 will support pkg_resources,
        # and pkg_resources is the default for these, so let's not bother users.
        return

    import importlib.metadata

    if hasattr(importlib.metadata, "_PIP_USE_IMPORTLIB_METADATA"):
        # The Python distributor has set the global constant, so we don't
        # warn, since it is not a user decision.
        return

    # The user has decided to use pkg_resources, so we warn.
    deprecated(
        reason="Using the pkg_resources metadata backend is deprecated.",
        replacement=(
            "to use the default importlib.metadata backend, "
            "by unsetting the _PIP_USE_IMPORTLIB_METADATA environment variable"
        ),
        gone_in="26.3",
        issue=13317,
    )


class Backend(Protocol):
    NAME: 'Literal["importlib", "pkg_resources"]'
    Distribution: Type[BaseDistribution]
    Environment: Type[BaseEnvironment]


@functools.lru_cache(maxsize=None)
def select_backend() -> Backend:
    if _should_use_importlib_metadata():
        from . import importlib

        return cast(Backend, importlib)

    _emit_pkg_resources_deprecation_if_needed()

    from . import pkg_resources

    return cast(Backend, pkg_resources)


def get_default_environment() -> BaseEnvironment:
    """Get the default representation for the current environment.

    This returns an Environment instance from the chosen backend. The default
    Environment instance should be built from ``sys.path`` and may use caching
    to share instance state across calls.
    """
    return select_backend().Environment.default()


def get_environment(paths: Optional[List[str]]) -> BaseEnvironment:
    """Get a representation of the environment specified by ``paths``.

    This returns an Environment instance from the chosen backend based on the
    given import paths. The backend must build a fresh instance representing
    the state of installed distributions when this function is called.
    """
    return select_backend().Environment.from_paths(paths)


def get_directory_distribution(directory: str) -> BaseDistribution:
    """Get the distribution metadata representation in the specified directory.

    This returns a Distribution instance from the chosen backend based on
    the given on-disk ``.dist-info`` directory.
    """
    return select_backend().Distribution.from_directory(directory)


def get_wheel_distribution(wheel: Wheel, canonical_name: str) -> BaseDistribution:
    """Get the representation of the specified wheel's distribution metadata.

    This returns a Distribution instance from the chosen backend based on
    the given wheel's ``.dist-info`` directory.

    :param canonical_name: Normalized project name of the given wheel.
    """
    return select_backend().Distribution.from_wheel(wheel, canonical_name)


def get_metadata_distribution(
    metadata_contents: bytes,
    filename: str,
    canonical_name: str,
) -> BaseDistribution:
    """Get the dist representation of the specified METADATA file contents.

    This returns a Distribution instance from the chosen backend sourced from the data
    in `metadata_contents`.

    :param metadata_contents: Contents of a METADATA file within a dist, or one served
                              via PEP 658.
    :param filename: Filename for the dist this metadata represents.
    :param canonical_name: Normalized project name of the given dist.
    """
    return select_backend().Distribution.from_metadata_file_contents(
        metadata_contents,
        filename,
        canonical_name,
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\distlib\markers.py ===
# -*- coding: utf-8 -*-
#
# Copyright (C) 2012-2023 Vinay Sajip.
# Licensed to the Python Software Foundation under a contributor agreement.
# See LICENSE.txt and CONTRIBUTORS.txt.
#
"""
Parser for the environment markers micro-language defined in PEP 508.
"""

# Note: In PEP 345, the micro-language was Python compatible, so the ast
# module could be used to parse it. However, PEP 508 introduced operators such
# as ~= and === which aren't in Python, necessitating a different approach.

import os
import re
import sys
import platform

from .compat import string_types
from .util import in_venv, parse_marker
from .version import LegacyVersion as LV

__all__ = ['interpret']

_VERSION_PATTERN = re.compile(r'((\d+(\.\d+)*\w*)|\'(\d+(\.\d+)*\w*)\'|\"(\d+(\.\d+)*\w*)\")')
_VERSION_MARKERS = {'python_version', 'python_full_version'}


def _is_version_marker(s):
    return isinstance(s, string_types) and s in _VERSION_MARKERS


def _is_literal(o):
    if not isinstance(o, string_types) or not o:
        return False
    return o[0] in '\'"'


def _get_versions(s):
    return {LV(m.groups()[0]) for m in _VERSION_PATTERN.finditer(s)}


class Evaluator(object):
    """
    This class is used to evaluate marker expressions.
    """

    operations = {
        '==': lambda x, y: x == y,
        '===': lambda x, y: x == y,
        '~=': lambda x, y: x == y or x > y,
        '!=': lambda x, y: x != y,
        '<': lambda x, y: x < y,
        '<=': lambda x, y: x == y or x < y,
        '>': lambda x, y: x > y,
        '>=': lambda x, y: x == y or x > y,
        'and': lambda x, y: x and y,
        'or': lambda x, y: x or y,
        'in': lambda x, y: x in y,
        'not in': lambda x, y: x not in y,
    }

    def evaluate(self, expr, context):
        """
        Evaluate a marker expression returned by the :func:`parse_requirement`
        function in the specified context.
        """
        if isinstance(expr, string_types):
            if expr[0] in '\'"':
                result = expr[1:-1]
            else:
                if expr not in context:
                    raise SyntaxError('unknown variable: %s' % expr)
                result = context[expr]
        else:
            assert isinstance(expr, dict)
            op = expr['op']
            if op not in self.operations:
                raise NotImplementedError('op not implemented: %s' % op)
            elhs = expr['lhs']
            erhs = expr['rhs']
            if _is_literal(expr['lhs']) and _is_literal(expr['rhs']):
                raise SyntaxError('invalid comparison: %s %s %s' % (elhs, op, erhs))

            lhs = self.evaluate(elhs, context)
            rhs = self.evaluate(erhs, context)
            if ((_is_version_marker(elhs) or _is_version_marker(erhs)) and
                    op in ('<', '<=', '>', '>=', '===', '==', '!=', '~=')):
                lhs = LV(lhs)
                rhs = LV(rhs)
            elif _is_version_marker(elhs) and op in ('in', 'not in'):
                lhs = LV(lhs)
                rhs = _get_versions(rhs)
            result = self.operations[op](lhs, rhs)
        return result


_DIGITS = re.compile(r'\d+\.\d+')


def default_context():

    def format_full_version(info):
        version = '%s.%s.%s' % (info.major, info.minor, info.micro)
        kind = info.releaselevel
        if kind != 'final':
            version += kind[0] + str(info.serial)
        return version

    if hasattr(sys, 'implementation'):
        implementation_version = format_full_version(sys.implementation.version)
        implementation_name = sys.implementation.name
    else:
        implementation_version = '0'
        implementation_name = ''

    ppv = platform.python_version()
    m = _DIGITS.match(ppv)
    pv = m.group(0)
    result = {
        'implementation_name': implementation_name,
        'implementation_version': implementation_version,
        'os_name': os.name,
        'platform_machine': platform.machine(),
        'platform_python_implementation': platform.python_implementation(),
        'platform_release': platform.release(),
        'platform_system': platform.system(),
        'platform_version': platform.version(),
        'platform_in_venv': str(in_venv()),
        'python_full_version': ppv,
        'python_version': pv,
        'sys_platform': sys.platform,
    }
    return result


DEFAULT_CONTEXT = default_context()
del default_context

evaluator = Evaluator()


def interpret(marker, execution_context=None):
    """
    Interpret a marker and return a result depending on environment.

    :param marker: The marker to interpret.
    :type marker: str
    :param execution_context: The context used for name lookup.
    :type execution_context: mapping
    """
    try:
        expr, rest = parse_marker(marker)
    except Exception as e:
        raise SyntaxError('Unable to interpret marker syntax: %s: %s' % (marker, e))
    if rest and rest[0] != '#':
        raise SyntaxError('unexpected trailing data in marker: %s: %s' % (marker, rest))
    context = dict(DEFAULT_CONTEXT)
    if execution_context:
        context.update(execution_context)
    return evaluator.evaluate(expr, context)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\mssql\provision.py ===
# dialects/mssql/provision.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors

from sqlalchemy import inspect
from sqlalchemy import Integer
from ... import create_engine
from ... import exc
from ...schema import Column
from ...schema import DropConstraint
from ...schema import ForeignKeyConstraint
from ...schema import MetaData
from ...schema import Table
from ...testing.provision import create_db
from ...testing.provision import drop_all_schema_objects_pre_tables
from ...testing.provision import drop_db
from ...testing.provision import generate_driver_url
from ...testing.provision import get_temp_table_name
from ...testing.provision import log
from ...testing.provision import normalize_sequence
from ...testing.provision import post_configure_engine
from ...testing.provision import run_reap_dbs
from ...testing.provision import temp_table_keyword_args


@post_configure_engine.for_db("mssql")
def post_configure_engine(url, engine, follower_ident):
    if engine.driver == "pyodbc":
        engine.dialect.dbapi.pooling = False


@generate_driver_url.for_db("mssql")
def generate_driver_url(url, driver, query_str):
    backend = url.get_backend_name()

    new_url = url.set(drivername="%s+%s" % (backend, driver))

    if driver not in ("pyodbc", "aioodbc"):
        new_url = new_url.set(query="")

    if driver == "aioodbc":
        new_url = new_url.update_query_dict({"MARS_Connection": "Yes"})

    if query_str:
        new_url = new_url.update_query_string(query_str)

    try:
        new_url.get_dialect()
    except exc.NoSuchModuleError:
        return None
    else:
        return new_url


@create_db.for_db("mssql")
def _mssql_create_db(cfg, eng, ident):
    with eng.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.exec_driver_sql("create database %s" % ident)
        conn.exec_driver_sql(
            "ALTER DATABASE %s SET ALLOW_SNAPSHOT_ISOLATION ON" % ident
        )
        conn.exec_driver_sql(
            "ALTER DATABASE %s SET READ_COMMITTED_SNAPSHOT ON" % ident
        )
        conn.exec_driver_sql("use %s" % ident)
        conn.exec_driver_sql("create schema test_schema")
        conn.exec_driver_sql("create schema test_schema_2")


@drop_db.for_db("mssql")
def _mssql_drop_db(cfg, eng, ident):
    with eng.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        _mssql_drop_ignore(conn, ident)


def _mssql_drop_ignore(conn, ident):
    try:
        # typically when this happens, we can't KILL the session anyway,
        # so let the cleanup process drop the DBs
        # for row in conn.exec_driver_sql(
        #     "select session_id from sys.dm_exec_sessions "
        #        "where database_id=db_id('%s')" % ident):
        #    log.info("killing SQL server session %s", row['session_id'])
        #    conn.exec_driver_sql("kill %s" % row['session_id'])
        conn.exec_driver_sql("drop database %s" % ident)
        log.info("Reaped db: %s", ident)
        return True
    except exc.DatabaseError as err:
        log.warning("couldn't drop db: %s", err)
        return False


@run_reap_dbs.for_db("mssql")
def _reap_mssql_dbs(url, idents):
    log.info("db reaper connecting to %r", url)
    eng = create_engine(url)
    with eng.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        log.info("identifiers in file: %s", ", ".join(idents))

        to_reap = conn.exec_driver_sql(
            "select d.name from sys.databases as d where name "
            "like 'TEST_%' and not exists (select session_id "
            "from sys.dm_exec_sessions "
            "where database_id=d.database_id)"
        )
        all_names = {dbname.lower() for (dbname,) in to_reap}
        to_drop = set()
        for name in all_names:
            if name in idents:
                to_drop.add(name)

        dropped = total = 0
        for total, dbname in enumerate(to_drop, 1):
            if _mssql_drop_ignore(conn, dbname):
                dropped += 1
        log.info(
            "Dropped %d out of %d stale databases detected", dropped, total
        )


@temp_table_keyword_args.for_db("mssql")
def _mssql_temp_table_keyword_args(cfg, eng):
    return {}


@get_temp_table_name.for_db("mssql")
def _mssql_get_temp_table_name(cfg, eng, base_name):
    return "##" + base_name


@drop_all_schema_objects_pre_tables.for_db("mssql")
def drop_all_schema_objects_pre_tables(cfg, eng):
    with eng.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        inspector = inspect(conn)
        for schema in (None, "dbo", cfg.test_schema, cfg.test_schema_2):
            for tname in inspector.get_table_names(schema=schema):
                tb = Table(
                    tname,
                    MetaData(),
                    Column("x", Integer),
                    Column("y", Integer),
                    schema=schema,
                )
                for fk in inspect(conn).get_foreign_keys(tname, schema=schema):
                    conn.execute(
                        DropConstraint(
                            ForeignKeyConstraint(
                                [tb.c.x], [tb.c.y], name=fk["name"]
                            )
                        )
                    )


@normalize_sequence.for_db("mssql")
def normalize_sequence(cfg, sequence):
    if sequence.start is None:
        sequence.start = 1
    return sequence

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\sql\expression.py ===
# sql/expression.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php

"""Defines the public namespace for SQL expression constructs.


"""


from __future__ import annotations

from ._dml_constructors import delete as delete
from ._dml_constructors import insert as insert
from ._dml_constructors import update as update
from ._elements_constructors import all_ as all_
from ._elements_constructors import and_ as and_
from ._elements_constructors import any_ as any_
from ._elements_constructors import asc as asc
from ._elements_constructors import between as between
from ._elements_constructors import bindparam as bindparam
from ._elements_constructors import bitwise_not as bitwise_not
from ._elements_constructors import case as case
from ._elements_constructors import cast as cast
from ._elements_constructors import collate as collate
from ._elements_constructors import column as column
from ._elements_constructors import desc as desc
from ._elements_constructors import distinct as distinct
from ._elements_constructors import extract as extract
from ._elements_constructors import false as false
from ._elements_constructors import funcfilter as funcfilter
from ._elements_constructors import label as label
from ._elements_constructors import not_ as not_
from ._elements_constructors import null as null
from ._elements_constructors import nulls_first as nulls_first
from ._elements_constructors import nulls_last as nulls_last
from ._elements_constructors import or_ as or_
from ._elements_constructors import outparam as outparam
from ._elements_constructors import over as over
from ._elements_constructors import text as text
from ._elements_constructors import true as true
from ._elements_constructors import try_cast as try_cast
from ._elements_constructors import tuple_ as tuple_
from ._elements_constructors import type_coerce as type_coerce
from ._elements_constructors import within_group as within_group
from ._selectable_constructors import alias as alias
from ._selectable_constructors import cte as cte
from ._selectable_constructors import except_ as except_
from ._selectable_constructors import except_all as except_all
from ._selectable_constructors import exists as exists
from ._selectable_constructors import intersect as intersect
from ._selectable_constructors import intersect_all as intersect_all
from ._selectable_constructors import join as join
from ._selectable_constructors import lateral as lateral
from ._selectable_constructors import outerjoin as outerjoin
from ._selectable_constructors import select as select
from ._selectable_constructors import table as table
from ._selectable_constructors import tablesample as tablesample
from ._selectable_constructors import union as union
from ._selectable_constructors import union_all as union_all
from ._selectable_constructors import values as values
from ._typing import ColumnExpressionArgument as ColumnExpressionArgument
from .base import _from_objects as _from_objects
from .base import _select_iterables as _select_iterables
from .base import ColumnCollection as ColumnCollection
from .base import Executable as Executable
from .cache_key import CacheKey as CacheKey
from .dml import Delete as Delete
from .dml import Insert as Insert
from .dml import Update as Update
from .dml import UpdateBase as UpdateBase
from .dml import ValuesBase as ValuesBase
from .elements import _truncated_label as _truncated_label
from .elements import BinaryExpression as BinaryExpression
from .elements import BindParameter as BindParameter
from .elements import BooleanClauseList as BooleanClauseList
from .elements import Case as Case
from .elements import Cast as Cast
from .elements import ClauseElement as ClauseElement
from .elements import ClauseList as ClauseList
from .elements import CollectionAggregate as CollectionAggregate
from .elements import ColumnClause as ColumnClause
from .elements import ColumnElement as ColumnElement
from .elements import ExpressionClauseList as ExpressionClauseList
from .elements import Extract as Extract
from .elements import False_ as False_
from .elements import FunctionFilter as FunctionFilter
from .elements import Grouping as Grouping
from .elements import Label as Label
from .elements import literal as literal
from .elements import literal_column as literal_column
from .elements import Null as Null
from .elements import Over as Over
from .elements import quoted_name as quoted_name
from .elements import ReleaseSavepointClause as ReleaseSavepointClause
from .elements import RollbackToSavepointClause as RollbackToSavepointClause
from .elements import SavepointClause as SavepointClause
from .elements import SQLColumnExpression as SQLColumnExpression
from .elements import TextClause as TextClause
from .elements import True_ as True_
from .elements import TryCast as TryCast
from .elements import Tuple as Tuple
from .elements import TypeClause as TypeClause
from .elements import TypeCoerce as TypeCoerce
from .elements import UnaryExpression as UnaryExpression
from .elements import WithinGroup as WithinGroup
from .functions import func as func
from .functions import Function as Function
from .functions import FunctionElement as FunctionElement
from .functions import modifier as modifier
from .lambdas import lambda_stmt as lambda_stmt
from .lambdas import LambdaElement as LambdaElement
from .lambdas import StatementLambdaElement as StatementLambdaElement
from .operators import ColumnOperators as ColumnOperators
from .operators import custom_op as custom_op
from .operators import Operators as Operators
from .selectable import Alias as Alias
from .selectable import AliasedReturnsRows as AliasedReturnsRows
from .selectable import CompoundSelect as CompoundSelect
from .selectable import CTE as CTE
from .selectable import Exists as Exists
from .selectable import FromClause as FromClause
from .selectable import FromGrouping as FromGrouping
from .selectable import GenerativeSelect as GenerativeSelect
from .selectable import HasCTE as HasCTE
from .selectable import HasPrefixes as HasPrefixes
from .selectable import HasSuffixes as HasSuffixes
from .selectable import Join as Join
from .selectable import LABEL_STYLE_DEFAULT as LABEL_STYLE_DEFAULT
from .selectable import (
    LABEL_STYLE_DISAMBIGUATE_ONLY as LABEL_STYLE_DISAMBIGUATE_ONLY,
)
from .selectable import LABEL_STYLE_NONE as LABEL_STYLE_NONE
from .selectable import (
    LABEL_STYLE_TABLENAME_PLUS_COL as LABEL_STYLE_TABLENAME_PLUS_COL,
)
from .selectable import Lateral as Lateral
from .selectable import ReturnsRows as ReturnsRows
from .selectable import ScalarSelect as ScalarSelect
from .selectable import ScalarValues as ScalarValues
from .selectable import Select as Select
from .selectable import Selectable as Selectable
from .selectable import SelectBase as SelectBase
from .selectable import SelectLabelStyle as SelectLabelStyle
from .selectable import Subquery as Subquery
from .selectable import TableClause as TableClause
from .selectable import TableSample as TableSample
from .selectable import TableValuedAlias as TableValuedAlias
from .selectable import TextAsFrom as TextAsFrom
from .selectable import TextualSelect as TextualSelect
from .selectable import Values as Values
from .visitors import Visitable as Visitable

nullsfirst = nulls_first
"""Synonym for the :func:`.nulls_first` function."""


nullslast = nulls_last
"""Synonym for the :func:`.nulls_last` function."""

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\liealgebras\type_f.py ===
from .cartan_type import Standard_Cartan
from sympy.core.backend import Matrix, Rational


class TypeF(Standard_Cartan):

    def __new__(cls, n):
        if n != 4:
            raise ValueError("n should be 4")
        return Standard_Cartan.__new__(cls, "F", 4)

    def dimension(self):
        """Dimension of the vector space V underlying the Lie algebra

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType("F4")
        >>> c.dimension()
        4
        """

        return 4


    def basic_root(self, i, j):
        """Generate roots with 1 in ith position and -1 in jth position

        """

        n = self.n
        root = [0]*n
        root[i] = 1
        root[j] = -1
        return root

    def simple_root(self, i):
        """The ith simple root of F_4

        Every lie algebra has a unique root system.
        Given a root system Q, there is a subset of the
        roots such that an element of Q is called a
        simple root if it cannot be written as the sum
        of two elements in Q.  If we let D denote the
        set of simple roots, then it is clear that every
        element of Q can be written as a linear combination
        of elements of D with all coefficients non-negative.

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType("F4")
        >>> c.simple_root(3)
        [0, 0, 0, 1]

        """

        if i < 3:
            return self.basic_root(i-1, i)
        if i == 3:
            root = [0]*4
            root[3] = 1
            return root
        if i == 4:
            root = [Rational(-1, 2)]*4
            return root

    def positive_roots(self):
        """Generate all the positive roots of A_n

        This is half of all of the roots of F_4; by multiplying all the
        positive roots by -1 we get the negative roots.

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType("A3")
        >>> c.positive_roots()
        {1: [1, -1, 0, 0], 2: [1, 0, -1, 0], 3: [1, 0, 0, -1], 4: [0, 1, -1, 0],
                5: [0, 1, 0, -1], 6: [0, 0, 1, -1]}

        """

        n = self.n
        posroots = {}
        k = 0
        for i in range(0, n-1):
            for j in range(i+1, n):
                k += 1
                posroots[k] = self.basic_root(i, j)
                k += 1
                root = self.basic_root(i, j)
                root[j] = 1
                posroots[k] = root

        for i in range(0, n):
            k += 1
            root = [0]*n
            root[i] = 1
            posroots[k] = root

        k += 1
        root = [Rational(1, 2)]*n
        posroots[k] = root
        for i in range(1, 4):
            k += 1
            root = [Rational(1, 2)]*n
            root[i] = Rational(-1, 2)
            posroots[k] = root

        posroots[k+1] = [Rational(1, 2), Rational(1, 2), Rational(-1, 2), Rational(-1, 2)]
        posroots[k+2] = [Rational(1, 2), Rational(-1, 2), Rational(1, 2), Rational(-1, 2)]
        posroots[k+3] = [Rational(1, 2), Rational(-1, 2), Rational(-1, 2), Rational(1, 2)]
        posroots[k+4] = [Rational(1, 2), Rational(-1, 2), Rational(-1, 2), Rational(-1, 2)]

        return posroots


    def roots(self):
        """
        Returns the total number of roots for F_4
        """
        return 48

    def cartan_matrix(self):
        """The Cartan matrix for F_4

        The Cartan matrix matrix for a Lie algebra is
        generated by assigning an ordering to the simple
        roots, (alpha[1], ...., alpha[l]).  Then the ijth
        entry of the Cartan matrix is (<alpha[i],alpha[j]>).

        Examples
        ========

        >>> from sympy.liealgebras.cartan_type import CartanType
        >>> c = CartanType('A4')
        >>> c.cartan_matrix()
        Matrix([
        [ 2, -1,  0,  0],
        [-1,  2, -1,  0],
        [ 0, -1,  2, -1],
        [ 0,  0, -1,  2]])
        """

        m = Matrix( 4, 4, [2, -1, 0, 0, -1, 2, -2, 0, 0,
            -1, 2, -1, 0, 0, -1, 2])
        return m

    def basis(self):
        """
        Returns the number of independent generators of F_4
        """
        return 52

    def dynkin_diagram(self):
        diag = "0---0=>=0---0\n"
        diag += "   ".join(str(i) for i in range(1, 5))
        return diag

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\core\parameters.py ===
"""Thread-safe global parameters"""

from .cache import clear_cache
from contextlib import contextmanager
from threading import local

class _global_parameters(local):
    """
    Thread-local global parameters.

    Explanation
    ===========

    This class generates thread-local container for SymPy's global parameters.
    Every global parameters must be passed as keyword argument when generating
    its instance.
    A variable, `global_parameters` is provided as default instance for this class.

    WARNING! Although the global parameters are thread-local, SymPy's cache is not
    by now.
    This may lead to undesired result in multi-threading operations.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.core.cache import clear_cache
    >>> from sympy.core.parameters import global_parameters as gp

    >>> gp.evaluate
    True
    >>> x+x
    2*x

    >>> log = []
    >>> def f():
    ...     clear_cache()
    ...     gp.evaluate = False
    ...     log.append(x+x)
    ...     clear_cache()
    >>> import threading
    >>> thread = threading.Thread(target=f)
    >>> thread.start()
    >>> thread.join()

    >>> print(log)
    [x + x]

    >>> gp.evaluate
    True
    >>> x+x
    2*x

    References
    ==========

    .. [1] https://docs.python.org/3/library/threading.html

    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __setattr__(self, name, value):
        if getattr(self, name) != value:
            clear_cache()
        return super().__setattr__(name, value)

global_parameters = _global_parameters(evaluate=True, distribute=True, exp_is_pow=False)

class evaluate:
    """ Control automatic evaluation

    Explanation
    ===========

    This context manager controls whether or not all SymPy functions evaluate
    by default.

    Note that much of SymPy expects evaluated expressions.  This functionality
    is experimental and is unlikely to function as intended on large
    expressions.

    Examples
    ========

    >>> from sympy import evaluate
    >>> from sympy.abc import x
    >>> print(x + x)
    2*x
    >>> with evaluate(False):
    ...     print(x + x)
    x + x
    """
    def __init__(self, x):
        self.x = x
        self.old = []

    def __enter__(self):
        self.old.append(global_parameters.evaluate)
        global_parameters.evaluate = self.x

    def __exit__(self, exc_type, exc_val, exc_tb):
        global_parameters.evaluate = self.old.pop()

@contextmanager
def distribute(x):
    """ Control automatic distribution of Number over Add

    Explanation
    ===========

    This context manager controls whether or not Mul distribute Number over
    Add. Plan is to avoid distributing Number over Add in all of sympy. Once
    that is done, this contextmanager will be removed.

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.core.parameters import distribute
    >>> print(2*(x + 1))
    2*x + 2
    >>> with distribute(False):
    ...     print(2*(x + 1))
    2*(x + 1)
    """

    old = global_parameters.distribute

    try:
        global_parameters.distribute = x
        yield
    finally:
        global_parameters.distribute = old


@contextmanager
def _exp_is_pow(x):
    """
    Control whether `e^x` should be represented as ``exp(x)`` or a ``Pow(E, x)``.

    Examples
    ========

    >>> from sympy import exp
    >>> from sympy.abc import x
    >>> from sympy.core.parameters import _exp_is_pow
    >>> with _exp_is_pow(True): print(type(exp(x)))
    <class 'sympy.core.power.Pow'>
    >>> with _exp_is_pow(False): print(type(exp(x)))
    exp
    """
    old = global_parameters.exp_is_pow

    clear_cache()
    try:
        global_parameters.exp_is_pow = x
        yield
    finally:
        clear_cache()
        global_parameters.exp_is_pow = old

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\attr\_cmp.py ===
# SPDX-License-Identifier: MIT


import functools
import types

from ._make import __ne__


_operation_names = {"eq": "==", "lt": "<", "le": "<=", "gt": ">", "ge": ">="}


def cmp_using(
    eq=None,
    lt=None,
    le=None,
    gt=None,
    ge=None,
    require_same_type=True,
    class_name="Comparable",
):
    """
    Create a class that can be passed into `attrs.field`'s ``eq``, ``order``,
    and ``cmp`` arguments to customize field comparison.

    The resulting class will have a full set of ordering methods if at least
    one of ``{lt, le, gt, ge}`` and ``eq``  are provided.

    Args:
        eq (typing.Callable | None):
            Callable used to evaluate equality of two objects.

        lt (typing.Callable | None):
            Callable used to evaluate whether one object is less than another
            object.

        le (typing.Callable | None):
            Callable used to evaluate whether one object is less than or equal
            to another object.

        gt (typing.Callable | None):
            Callable used to evaluate whether one object is greater than
            another object.

        ge (typing.Callable | None):
            Callable used to evaluate whether one object is greater than or
            equal to another object.

        require_same_type (bool):
            When `True`, equality and ordering methods will return
            `NotImplemented` if objects are not of the same type.

        class_name (str | None): Name of class. Defaults to "Comparable".

    See `comparison` for more details.

    .. versionadded:: 21.1.0
    """

    body = {
        "__slots__": ["value"],
        "__init__": _make_init(),
        "_requirements": [],
        "_is_comparable_to": _is_comparable_to,
    }

    # Add operations.
    num_order_functions = 0
    has_eq_function = False

    if eq is not None:
        has_eq_function = True
        body["__eq__"] = _make_operator("eq", eq)
        body["__ne__"] = __ne__

    if lt is not None:
        num_order_functions += 1
        body["__lt__"] = _make_operator("lt", lt)

    if le is not None:
        num_order_functions += 1
        body["__le__"] = _make_operator("le", le)

    if gt is not None:
        num_order_functions += 1
        body["__gt__"] = _make_operator("gt", gt)

    if ge is not None:
        num_order_functions += 1
        body["__ge__"] = _make_operator("ge", ge)

    type_ = types.new_class(
        class_name, (object,), {}, lambda ns: ns.update(body)
    )

    # Add same type requirement.
    if require_same_type:
        type_._requirements.append(_check_same_type)

    # Add total ordering if at least one operation was defined.
    if 0 < num_order_functions < 4:
        if not has_eq_function:
            # functools.total_ordering requires __eq__ to be defined,
            # so raise early error here to keep a nice stack.
            msg = "eq must be define is order to complete ordering from lt, le, gt, ge."
            raise ValueError(msg)
        type_ = functools.total_ordering(type_)

    return type_


def _make_init():
    """
    Create __init__ method.
    """

    def __init__(self, value):
        """
        Initialize object with *value*.
        """
        self.value = value

    return __init__


def _make_operator(name, func):
    """
    Create operator method.
    """

    def method(self, other):
        if not self._is_comparable_to(other):
            return NotImplemented

        result = func(self.value, other.value)
        if result is NotImplemented:
            return NotImplemented

        return result

    method.__name__ = f"__{name}__"
    method.__doc__ = (
        f"Return a {_operation_names[name]} b.  Computed by attrs."
    )

    return method


def _is_comparable_to(self, other):
    """
    Check whether `other` is comparable to `self`.
    """
    return all(func(self, other) for func in self._requirements)


def _check_same_type(self, other):
    """
    Return True if *self* and *other* are of the same type, False otherwise.
    """
    return other.value.__class__ is self.value.__class__

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\tools\ort_format_model\ort_flatbuffers_py\fbs\NodesToOptimizeIndices.py ===
# automatically generated by the FlatBuffers compiler, do not modify

# namespace: fbs

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

# nodes to consider for a runtime optimization
# see corresponding type in onnxruntime/core/graph/runtime_optimization_record.h
class NodesToOptimizeIndices(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = NodesToOptimizeIndices()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsNodesToOptimizeIndices(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def NodesToOptimizeIndicesBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x4F\x52\x54\x4D", size_prefixed=size_prefixed)

    # NodesToOptimizeIndices
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # NodesToOptimizeIndices
    def NodeIndices(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # NodesToOptimizeIndices
    def NodeIndicesAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Uint32Flags, o)
        return 0

    # NodesToOptimizeIndices
    def NodeIndicesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # NodesToOptimizeIndices
    def NodeIndicesIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        return o == 0

    # NodesToOptimizeIndices
    def NumInputs(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 0

    # NodesToOptimizeIndices
    def NumOutputs(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 0

    # NodesToOptimizeIndices
    def HasVariadicInput(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

    # NodesToOptimizeIndices
    def HasVariadicOutput(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

    # NodesToOptimizeIndices
    def NumVariadicInputs(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 0

    # NodesToOptimizeIndices
    def NumVariadicOutputs(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 0

def NodesToOptimizeIndicesStart(builder):
    builder.StartObject(7)

def Start(builder):
    NodesToOptimizeIndicesStart(builder)

def NodesToOptimizeIndicesAddNodeIndices(builder, nodeIndices):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(nodeIndices), 0)

def AddNodeIndices(builder, nodeIndices):
    NodesToOptimizeIndicesAddNodeIndices(builder, nodeIndices)

def NodesToOptimizeIndicesStartNodeIndicesVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartNodeIndicesVector(builder, numElems: int) -> int:
    return NodesToOptimizeIndicesStartNodeIndicesVector(builder, numElems)

def NodesToOptimizeIndicesAddNumInputs(builder, numInputs):
    builder.PrependUint32Slot(1, numInputs, 0)

def AddNumInputs(builder, numInputs):
    NodesToOptimizeIndicesAddNumInputs(builder, numInputs)

def NodesToOptimizeIndicesAddNumOutputs(builder, numOutputs):
    builder.PrependUint32Slot(2, numOutputs, 0)

def AddNumOutputs(builder, numOutputs):
    NodesToOptimizeIndicesAddNumOutputs(builder, numOutputs)

def NodesToOptimizeIndicesAddHasVariadicInput(builder, hasVariadicInput):
    builder.PrependBoolSlot(3, hasVariadicInput, 0)

def AddHasVariadicInput(builder, hasVariadicInput):
    NodesToOptimizeIndicesAddHasVariadicInput(builder, hasVariadicInput)

def NodesToOptimizeIndicesAddHasVariadicOutput(builder, hasVariadicOutput):
    builder.PrependBoolSlot(4, hasVariadicOutput, 0)

def AddHasVariadicOutput(builder, hasVariadicOutput):
    NodesToOptimizeIndicesAddHasVariadicOutput(builder, hasVariadicOutput)

def NodesToOptimizeIndicesAddNumVariadicInputs(builder, numVariadicInputs):
    builder.PrependUint32Slot(5, numVariadicInputs, 0)

def AddNumVariadicInputs(builder, numVariadicInputs):
    NodesToOptimizeIndicesAddNumVariadicInputs(builder, numVariadicInputs)

def NodesToOptimizeIndicesAddNumVariadicOutputs(builder, numVariadicOutputs):
    builder.PrependUint32Slot(6, numVariadicOutputs, 0)

def AddNumVariadicOutputs(builder, numVariadicOutputs):
    NodesToOptimizeIndicesAddNumVariadicOutputs(builder, numVariadicOutputs)

def NodesToOptimizeIndicesEnd(builder):
    return builder.EndObject()

def End(builder):
    return NodesToOptimizeIndicesEnd(builder)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\worksheet\_write_only.py ===
# Copyright (c) 2010-2024 openpyxl


"""Write worksheets to xml representations in an optimized way"""

from inspect import isgenerator

from openpyxl.cell import Cell, WriteOnlyCell
from openpyxl.workbook.child import _WorkbookChild
from .worksheet import Worksheet
from openpyxl.utils.exceptions import WorkbookAlreadySaved

from ._writer import WorksheetWriter


class WriteOnlyWorksheet(_WorkbookChild):
    """
    Streaming worksheet. Optimised to reduce memory by writing rows just in
    time.
    Cells can be styled and have comments Styles for rows and columns
    must be applied before writing cells
    """

    __saved = False
    _writer = None
    _rows = None
    _rel_type = Worksheet._rel_type
    _path = Worksheet._path
    mime_type = Worksheet.mime_type

    # copy methods from Standard worksheet
    _add_row = Worksheet._add_row
    _add_column = Worksheet._add_column
    add_chart = Worksheet.add_chart
    add_image = Worksheet.add_image
    add_table = Worksheet.add_table
    tables = Worksheet.tables
    print_titles = Worksheet.print_titles
    print_title_cols = Worksheet.print_title_cols
    print_title_rows = Worksheet.print_title_rows
    freeze_panes = Worksheet.freeze_panes
    print_area = Worksheet.print_area
    sheet_view = Worksheet.sheet_view
    _setup = Worksheet._setup

    def __init__(self, parent, title):
        super().__init__(parent, title)
        self._max_col = 0
        self._max_row = 0
        self._setup()

    @property
    def closed(self):
        return self.__saved


    def _write_rows(self):
        """
        Send rows to the writer's stream
        """
        try:
            xf = self._writer.xf.send(True)
        except StopIteration:
            self._already_saved()

        with xf.element("sheetData"):
            row_idx = 1
            try:
                while True:
                    row = (yield)
                    row = self._values_to_row(row, row_idx)
                    self._writer.write_row(xf, row, row_idx)
                    row_idx += 1
            except GeneratorExit:
                pass

        self._writer.xf.send(None)


    def _get_writer(self):
        if self._writer is None:
            self._writer = WorksheetWriter(self)
            self._writer.write_top()


    def close(self):
        if self.__saved:
            self._already_saved()

        self._get_writer()

        if self._rows is None:
            self._writer.write_rows()
        else:
            self._rows.close()

        self._writer.write_tail()

        self._writer.close()
        self.__saved = True


    def append(self, row):
        """
        :param row: iterable containing values to append
        :type row: iterable
        """

        if (not isgenerator(row) and
            not isinstance(row, (list, tuple, range))
            ):
            self._invalid_row(row)

        self._get_writer()

        if self._rows is None:
            self._rows = self._write_rows()
            next(self._rows)

        self._rows.send(row)


    def _values_to_row(self, values, row_idx):
        """
        Convert whatever has been appended into a form suitable for work_rows
        """
        cell = WriteOnlyCell(self)

        for col_idx, value in enumerate(values, 1):
            if value is None:
                continue
            try:
                cell.value = value
            except ValueError:
                if isinstance(value, Cell):
                    cell = value
                else:
                    raise ValueError

            cell.column = col_idx
            cell.row = row_idx

            if cell.hyperlink is not None:
                cell.hyperlink.ref = cell.coordinate

            yield cell

            # reset cell if style applied
            if cell.has_style or cell.hyperlink:
                cell = WriteOnlyCell(self)


    def _already_saved(self):
        raise WorkbookAlreadySaved('Workbook has already been saved and cannot be modified or saved anymore.')


    def _invalid_row(self, iterable):
        raise TypeError('Value must be a list, tuple, range or a generator Supplied value is {0}'.format(
            type(iterable))
                        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\util\__init__.py ===
# util/__init__.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php


from collections import defaultdict as defaultdict
from functools import partial as partial
from functools import update_wrapper as update_wrapper

from . import preloaded as preloaded
from ._collections import coerce_generator_arg as coerce_generator_arg
from ._collections import coerce_to_immutabledict as coerce_to_immutabledict
from ._collections import column_dict as column_dict
from ._collections import column_set as column_set
from ._collections import EMPTY_DICT as EMPTY_DICT
from ._collections import EMPTY_SET as EMPTY_SET
from ._collections import FacadeDict as FacadeDict
from ._collections import flatten_iterator as flatten_iterator
from ._collections import has_dupes as has_dupes
from ._collections import has_intersection as has_intersection
from ._collections import IdentitySet as IdentitySet
from ._collections import immutabledict as immutabledict
from ._collections import LRUCache as LRUCache
from ._collections import merge_lists_w_ordering as merge_lists_w_ordering
from ._collections import NONE_SET as NONE_SET
from ._collections import ordered_column_set as ordered_column_set
from ._collections import OrderedDict as OrderedDict
from ._collections import OrderedIdentitySet as OrderedIdentitySet
from ._collections import OrderedProperties as OrderedProperties
from ._collections import OrderedSet as OrderedSet
from ._collections import PopulateDict as PopulateDict
from ._collections import Properties as Properties
from ._collections import ReadOnlyContainer as ReadOnlyContainer
from ._collections import ReadOnlyProperties as ReadOnlyProperties
from ._collections import ScopedRegistry as ScopedRegistry
from ._collections import sort_dictionary as sort_dictionary
from ._collections import ThreadLocalRegistry as ThreadLocalRegistry
from ._collections import to_column_set as to_column_set
from ._collections import to_list as to_list
from ._collections import to_set as to_set
from ._collections import unique_list as unique_list
from ._collections import UniqueAppender as UniqueAppender
from ._collections import update_copy as update_copy
from ._collections import WeakPopulateDict as WeakPopulateDict
from ._collections import WeakSequence as WeakSequence
from .compat import anext_ as anext_
from .compat import arm as arm
from .compat import b as b
from .compat import b64decode as b64decode
from .compat import b64encode as b64encode
from .compat import cmp as cmp
from .compat import cpython as cpython
from .compat import dataclass_fields as dataclass_fields
from .compat import decode_backslashreplace as decode_backslashreplace
from .compat import dottedgetter as dottedgetter
from .compat import has_refcount_gc as has_refcount_gc
from .compat import inspect_getfullargspec as inspect_getfullargspec
from .compat import is64bit as is64bit
from .compat import local_dataclass_fields as local_dataclass_fields
from .compat import osx as osx
from .compat import py310 as py310
from .compat import py311 as py311
from .compat import py312 as py312
from .compat import py313 as py313
from .compat import py314 as py314
from .compat import py38 as py38
from .compat import py39 as py39
from .compat import pypy as pypy
from .compat import win32 as win32
from .concurrency import await_fallback as await_fallback
from .concurrency import await_only as await_only
from .concurrency import greenlet_spawn as greenlet_spawn
from .concurrency import is_exit_exception as is_exit_exception
from .deprecations import became_legacy_20 as became_legacy_20
from .deprecations import deprecated as deprecated
from .deprecations import deprecated_cls as deprecated_cls
from .deprecations import deprecated_params as deprecated_params
from .deprecations import moved_20 as moved_20
from .deprecations import warn_deprecated as warn_deprecated
from .langhelpers import add_parameter_text as add_parameter_text
from .langhelpers import as_interface as as_interface
from .langhelpers import asbool as asbool
from .langhelpers import asint as asint
from .langhelpers import assert_arg_type as assert_arg_type
from .langhelpers import attrsetter as attrsetter
from .langhelpers import bool_or_str as bool_or_str
from .langhelpers import chop_traceback as chop_traceback
from .langhelpers import class_hierarchy as class_hierarchy
from .langhelpers import classproperty as classproperty
from .langhelpers import clsname_as_plain_name as clsname_as_plain_name
from .langhelpers import coerce_kw_type as coerce_kw_type
from .langhelpers import constructor_copy as constructor_copy
from .langhelpers import constructor_key as constructor_key
from .langhelpers import counter as counter
from .langhelpers import create_proxy_methods as create_proxy_methods
from .langhelpers import decode_slice as decode_slice
from .langhelpers import decorator as decorator
from .langhelpers import dictlike_iteritems as dictlike_iteritems
from .langhelpers import duck_type_collection as duck_type_collection
from .langhelpers import ellipses_string as ellipses_string
from .langhelpers import EnsureKWArg as EnsureKWArg
from .langhelpers import FastIntFlag as FastIntFlag
from .langhelpers import format_argspec_init as format_argspec_init
from .langhelpers import format_argspec_plus as format_argspec_plus
from .langhelpers import generic_fn_descriptor as generic_fn_descriptor
from .langhelpers import generic_repr as generic_repr
from .langhelpers import get_annotations as get_annotations
from .langhelpers import get_callable_argspec as get_callable_argspec
from .langhelpers import get_cls_kwargs as get_cls_kwargs
from .langhelpers import get_func_kwargs as get_func_kwargs
from .langhelpers import getargspec_init as getargspec_init
from .langhelpers import has_compiled_ext as has_compiled_ext
from .langhelpers import HasMemoized as HasMemoized
from .langhelpers import (
    HasMemoized_ro_memoized_attribute as HasMemoized_ro_memoized_attribute,
)
from .langhelpers import hybridmethod as hybridmethod
from .langhelpers import hybridproperty as hybridproperty
from .langhelpers import inject_docstring_text as inject_docstring_text
from .langhelpers import iterate_attributes as iterate_attributes
from .langhelpers import map_bits as map_bits
from .langhelpers import md5_hex as md5_hex
from .langhelpers import memoized_instancemethod as memoized_instancemethod
from .langhelpers import memoized_property as memoized_property
from .langhelpers import MemoizedSlots as MemoizedSlots
from .langhelpers import method_is_overridden as method_is_overridden
from .langhelpers import methods_equivalent as methods_equivalent
from .langhelpers import (
    monkeypatch_proxied_specials as monkeypatch_proxied_specials,
)
from .langhelpers import non_memoized_property as non_memoized_property
from .langhelpers import NoneType as NoneType
from .langhelpers import only_once as only_once
from .langhelpers import (
    parse_user_argument_for_enum as parse_user_argument_for_enum,
)
from .langhelpers import PluginLoader as PluginLoader
from .langhelpers import portable_instancemethod as portable_instancemethod
from .langhelpers import quoted_token_parser as quoted_token_parser
from .langhelpers import ro_memoized_property as ro_memoized_property
from .langhelpers import ro_non_memoized_property as ro_non_memoized_property
from .langhelpers import rw_hybridproperty as rw_hybridproperty
from .langhelpers import safe_reraise as safe_reraise
from .langhelpers import set_creation_order as set_creation_order
from .langhelpers import string_or_unprintable as string_or_unprintable
from .langhelpers import symbol as symbol
from .langhelpers import TypingOnly as TypingOnly
from .langhelpers import (
    unbound_method_to_callable as unbound_method_to_callable,
)
from .langhelpers import walk_subclasses as walk_subclasses
from .langhelpers import warn as warn
from .langhelpers import warn_exception as warn_exception
from .langhelpers import warn_limited as warn_limited
from .langhelpers import wrap_callable as wrap_callable
from .preloaded import preload_module as preload_module
from .typing import is_non_string_iterable as is_non_string_iterable

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\combinatorics\schur_number.py ===
"""
The Schur number S(k) is the largest integer n for which the interval [1,n]
can be partitioned into k sum-free sets.(https://mathworld.wolfram.com/SchurNumber.html)
"""
import math
from sympy.core import S
from sympy.core.basic import Basic
from sympy.core.function import Function
from sympy.core.numbers import Integer


class SchurNumber(Function):
    r"""
    This function creates a SchurNumber object
    which is evaluated for `k \le 5` otherwise only
    the lower bound information can be retrieved.

    Examples
    ========

    >>> from sympy.combinatorics.schur_number import SchurNumber

    Since S(3) = 13, hence the output is a number
    >>> SchurNumber(3)
    13

    We do not know the Schur number for values greater than 5, hence
    only the object is returned
    >>> SchurNumber(6)
    SchurNumber(6)

    Now, the lower bound information can be retrieved using lower_bound()
    method
    >>> SchurNumber(6).lower_bound()
    536

    """

    @classmethod
    def eval(cls, k):
        if k.is_Number:
            if k is S.Infinity:
                return S.Infinity
            if k.is_zero:
                return S.Zero
            if not k.is_integer or k.is_negative:
                raise ValueError("k should be a positive integer")
            first_known_schur_numbers = {1: 1, 2: 4, 3: 13, 4: 44, 5: 160}
            if k <= 5:
                return Integer(first_known_schur_numbers[k])

    def lower_bound(self):
        f_ = self.args[0]
        # Improved lower bounds known for S(6) and S(7)
        if f_ == 6:
            return Integer(536)
        if f_ == 7:
            return Integer(1680)
        # For other cases, use general expression
        if f_.is_Integer:
            return 3*self.func(f_ - 1).lower_bound() - 1
        return (3**f_ - 1)/2


def _schur_subsets_number(n):

    if n is S.Infinity:
        raise ValueError("Input must be finite")
    if n <= 0:
        raise ValueError("n must be a non-zero positive integer.")
    elif n <= 3:
        min_k = 1
    else:
        min_k = math.ceil(math.log(2*n + 1, 3))

    return Integer(min_k)


def schur_partition(n):
    """

    This function returns the partition in the minimum number of sum-free subsets
    according to the lower bound given by the Schur Number.

    Parameters
    ==========

    n: a number
        n is the upper limit of the range [1, n] for which we need to find and
        return the minimum number of free subsets according to the lower bound
        of schur number

    Returns
    =======

    List of lists
        List of the minimum number of sum-free subsets

    Notes
    =====

    It is possible for some n to make the partition into less
    subsets since the only known Schur numbers are:
    S(1) = 1, S(2) = 4, S(3) = 13, S(4) = 44.
    e.g for n = 44 the lower bound from the function above is 5 subsets but it has been proven
    that can be done with 4 subsets.

    Examples
    ========

    For n = 1, 2, 3 the answer is the set itself

    >>> from sympy.combinatorics.schur_number import schur_partition
    >>> schur_partition(2)
    [[1, 2]]

    For n > 3, the answer is the minimum number of sum-free subsets:

    >>> schur_partition(5)
    [[3, 2], [5], [1, 4]]

    >>> schur_partition(8)
    [[3, 2], [6, 5, 8], [1, 4, 7]]
    """

    if isinstance(n, Basic) and not n.is_Number:
        raise ValueError("Input value must be a number")

    number_of_subsets = _schur_subsets_number(n)
    if n == 1:
        sum_free_subsets = [[1]]
    elif n == 2:
        sum_free_subsets = [[1, 2]]
    elif n == 3:
        sum_free_subsets = [[1, 2, 3]]
    else:
        sum_free_subsets = [[1, 4], [2, 3]]

    while len(sum_free_subsets) < number_of_subsets:
        sum_free_subsets = _generate_next_list(sum_free_subsets, n)
        missed_elements = [3*k + 1 for k in range(len(sum_free_subsets), (n-1)//3 + 1)]
        sum_free_subsets[-1] += missed_elements

    return sum_free_subsets


def _generate_next_list(current_list, n):
    new_list = []

    for item in current_list:
        temp_1 = [number*3 for number in item if number*3 <= n]
        temp_2 = [number*3 - 1 for number in item if number*3 - 1 <= n]
        new_item = temp_1 + temp_2
        new_list.append(new_item)

    last_list = [3*k + 1 for k in range(len(current_list)+1) if 3*k + 1 <= n]
    new_list.append(last_list)
    current_list = new_list

    return current_list

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\cli\spinners.py ===
import contextlib
import itertools
import logging
import sys
import time
from typing import IO, Generator, Optional

from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.logging import get_indentation

logger = logging.getLogger(__name__)


class SpinnerInterface:
    def spin(self) -> None:
        raise NotImplementedError()

    def finish(self, final_status: str) -> None:
        raise NotImplementedError()


class InteractiveSpinner(SpinnerInterface):
    def __init__(
        self,
        message: str,
        file: Optional[IO[str]] = None,
        spin_chars: str = "-\\|/",
        # Empirically, 8 updates/second looks nice
        min_update_interval_seconds: float = 0.125,
    ):
        self._message = message
        if file is None:
            file = sys.stdout
        self._file = file
        self._rate_limiter = RateLimiter(min_update_interval_seconds)
        self._finished = False

        self._spin_cycle = itertools.cycle(spin_chars)

        self._file.write(" " * get_indentation() + self._message + " ... ")
        self._width = 0

    def _write(self, status: str) -> None:
        assert not self._finished
        # Erase what we wrote before by backspacing to the beginning, writing
        # spaces to overwrite the old text, and then backspacing again
        backup = "\b" * self._width
        self._file.write(backup + " " * self._width + backup)
        # Now we have a blank slate to add our status
        self._file.write(status)
        self._width = len(status)
        self._file.flush()
        self._rate_limiter.reset()

    def spin(self) -> None:
        if self._finished:
            return
        if not self._rate_limiter.ready():
            return
        self._write(next(self._spin_cycle))

    def finish(self, final_status: str) -> None:
        if self._finished:
            return
        self._write(final_status)
        self._file.write("\n")
        self._file.flush()
        self._finished = True


# Used for dumb terminals, non-interactive installs (no tty), etc.
# We still print updates occasionally (once every 60 seconds by default) to
# act as a keep-alive for systems like Travis-CI that take lack-of-output as
# an indication that a task has frozen.
class NonInteractiveSpinner(SpinnerInterface):
    def __init__(self, message: str, min_update_interval_seconds: float = 60.0) -> None:
        self._message = message
        self._finished = False
        self._rate_limiter = RateLimiter(min_update_interval_seconds)
        self._update("started")

    def _update(self, status: str) -> None:
        assert not self._finished
        self._rate_limiter.reset()
        logger.info("%s: %s", self._message, status)

    def spin(self) -> None:
        if self._finished:
            return
        if not self._rate_limiter.ready():
            return
        self._update("still running...")

    def finish(self, final_status: str) -> None:
        if self._finished:
            return
        self._update(f"finished with status '{final_status}'")
        self._finished = True


class RateLimiter:
    def __init__(self, min_update_interval_seconds: float) -> None:
        self._min_update_interval_seconds = min_update_interval_seconds
        self._last_update: float = 0

    def ready(self) -> bool:
        now = time.time()
        delta = now - self._last_update
        return delta >= self._min_update_interval_seconds

    def reset(self) -> None:
        self._last_update = time.time()


@contextlib.contextmanager
def open_spinner(message: str) -> Generator[SpinnerInterface, None, None]:
    # Interactive spinner goes directly to sys.stdout rather than being routed
    # through the logging system, but it acts like it has level INFO,
    # i.e. it's only displayed if we're at level INFO or better.
    # Non-interactive spinner goes through the logging system, so it is always
    # in sync with logging configuration.
    if sys.stdout.isatty() and logger.getEffectiveLevel() <= logging.INFO:
        spinner: SpinnerInterface = InteractiveSpinner(message)
    else:
        spinner = NonInteractiveSpinner(message)
    try:
        with hidden_cursor(sys.stdout):
            yield spinner
    except KeyboardInterrupt:
        spinner.finish("canceled")
        raise
    except Exception:
        spinner.finish("error")
        raise
    else:
        spinner.finish("done")


HIDE_CURSOR = "\x1b[?25l"
SHOW_CURSOR = "\x1b[?25h"


@contextlib.contextmanager
def hidden_cursor(file: IO[str]) -> Generator[None, None, None]:
    # The Windows terminal does not support the hide/show cursor ANSI codes,
    # even via colorama. So don't even try.
    if WINDOWS:
        yield
    # We don't want to clutter the output with control characters if we're
    # writing to a file, or if the user is running with --quiet.
    # See https://github.com/pypa/pip/issues/3418
    elif not file.isatty() or logger.getEffectiveLevel() > logging.INFO:
        yield
    else:
        file.write(HIDE_CURSOR)
        try:
            yield
        finally:
            file.write(SHOW_CURSOR)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\rich\_ratio.py ===
import sys
from fractions import Fraction
from math import ceil
from typing import cast, List, Optional, Sequence

if sys.version_info >= (3, 8):
    from typing import Protocol
else:
    from pip._vendor.typing_extensions import Protocol  # pragma: no cover


class Edge(Protocol):
    """Any object that defines an edge (such as Layout)."""

    size: Optional[int] = None
    ratio: int = 1
    minimum_size: int = 1


def ratio_resolve(total: int, edges: Sequence[Edge]) -> List[int]:
    """Divide total space to satisfy size, ratio, and minimum_size, constraints.

    The returned list of integers should add up to total in most cases, unless it is
    impossible to satisfy all the constraints. For instance, if there are two edges
    with a minimum size of 20 each and `total` is 30 then the returned list will be
    greater than total. In practice, this would mean that a Layout object would
    clip the rows that would overflow the screen height.

    Args:
        total (int): Total number of characters.
        edges (List[Edge]): Edges within total space.

    Returns:
        List[int]: Number of characters for each edge.
    """
    # Size of edge or None for yet to be determined
    sizes = [(edge.size or None) for edge in edges]

    _Fraction = Fraction

    # While any edges haven't been calculated
    while None in sizes:
        # Get flexible edges and index to map these back on to sizes list
        flexible_edges = [
            (index, edge)
            for index, (size, edge) in enumerate(zip(sizes, edges))
            if size is None
        ]
        # Remaining space in total
        remaining = total - sum(size or 0 for size in sizes)
        if remaining <= 0:
            # No room for flexible edges
            return [
                ((edge.minimum_size or 1) if size is None else size)
                for size, edge in zip(sizes, edges)
            ]
        # Calculate number of characters in a ratio portion
        portion = _Fraction(
            remaining, sum((edge.ratio or 1) for _, edge in flexible_edges)
        )

        # If any edges will be less than their minimum, replace size with the minimum
        for index, edge in flexible_edges:
            if portion * edge.ratio <= edge.minimum_size:
                sizes[index] = edge.minimum_size
                # New fixed size will invalidate calculations, so we need to repeat the process
                break
        else:
            # Distribute flexible space and compensate for rounding error
            # Since edge sizes can only be integers we need to add the remainder
            # to the following line
            remainder = _Fraction(0)
            for index, edge in flexible_edges:
                size, remainder = divmod(portion * edge.ratio + remainder, 1)
                sizes[index] = size
            break
    # Sizes now contains integers only
    return cast(List[int], sizes)


def ratio_reduce(
    total: int, ratios: List[int], maximums: List[int], values: List[int]
) -> List[int]:
    """Divide an integer total in to parts based on ratios.

    Args:
        total (int): The total to divide.
        ratios (List[int]): A list of integer ratios.
        maximums (List[int]): List of maximums values for each slot.
        values (List[int]): List of values

    Returns:
        List[int]: A list of integers guaranteed to sum to total.
    """
    ratios = [ratio if _max else 0 for ratio, _max in zip(ratios, maximums)]
    total_ratio = sum(ratios)
    if not total_ratio:
        return values[:]
    total_remaining = total
    result: List[int] = []
    append = result.append
    for ratio, maximum, value in zip(ratios, maximums, values):
        if ratio and total_ratio > 0:
            distributed = min(maximum, round(ratio * total_remaining / total_ratio))
            append(value - distributed)
            total_remaining -= distributed
            total_ratio -= ratio
        else:
            append(value)
    return result


def ratio_distribute(
    total: int, ratios: List[int], minimums: Optional[List[int]] = None
) -> List[int]:
    """Distribute an integer total in to parts based on ratios.

    Args:
        total (int): The total to divide.
        ratios (List[int]): A list of integer ratios.
        minimums (List[int]): List of minimum values for each slot.

    Returns:
        List[int]: A list of integers guaranteed to sum to total.
    """
    if minimums:
        ratios = [ratio if _min else 0 for ratio, _min in zip(ratios, minimums)]
    total_ratio = sum(ratios)
    assert total_ratio > 0, "Sum of ratios must be > 0"

    total_remaining = total
    distributed_total: List[int] = []
    append = distributed_total.append
    if minimums is None:
        _minimums = [0] * len(ratios)
    else:
        _minimums = minimums
    for ratio, minimum in zip(ratios, _minimums):
        if total_ratio > 0:
            distributed = max(minimum, ceil(ratio * total_remaining / total_ratio))
        else:
            distributed = total_remaining
        append(distributed)
        total_ratio -= ratio
        total_remaining -= distributed
    return distributed_total


if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class E:
        size: Optional[int] = None
        ratio: int = 1
        minimum_size: int = 1

    resolved = ratio_resolve(110, [E(None, 1, 1), E(None, 1, 1), E(None, 1, 1)])
    print(sum(resolved))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\urllib3\util\ssl_match_hostname.py ===
"""The match_hostname() function from Python 3.3.3, essential when using SSL."""

# Note: This file is under the PSF license as the code comes from the python
# stdlib.   http://docs.python.org/3/license.html

import re
import sys

# ipaddress has been backported to 2.6+ in pypi.  If it is installed on the
# system, use it to handle IPAddress ServerAltnames (this was added in
# python-3.5) otherwise only do DNS matching.  This allows
# util.ssl_match_hostname to continue to be used in Python 2.7.
try:
    import ipaddress
except ImportError:
    ipaddress = None

__version__ = "3.5.0.1"


class CertificateError(ValueError):
    pass


def _dnsname_match(dn, hostname, max_wildcards=1):
    """Matching according to RFC 6125, section 6.4.3

    http://tools.ietf.org/html/rfc6125#section-6.4.3
    """
    pats = []
    if not dn:
        return False

    # Ported from python3-syntax:
    # leftmost, *remainder = dn.split(r'.')
    parts = dn.split(r".")
    leftmost = parts[0]
    remainder = parts[1:]

    wildcards = leftmost.count("*")
    if wildcards > max_wildcards:
        # Issue #17980: avoid denials of service by refusing more
        # than one wildcard per fragment.  A survey of established
        # policy among SSL implementations showed it to be a
        # reasonable choice.
        raise CertificateError(
            "too many wildcards in certificate DNS name: " + repr(dn)
        )

    # speed up common case w/o wildcards
    if not wildcards:
        return dn.lower() == hostname.lower()

    # RFC 6125, section 6.4.3, subitem 1.
    # The client SHOULD NOT attempt to match a presented identifier in which
    # the wildcard character comprises a label other than the left-most label.
    if leftmost == "*":
        # When '*' is a fragment by itself, it matches a non-empty dotless
        # fragment.
        pats.append("[^.]+")
    elif leftmost.startswith("xn--") or hostname.startswith("xn--"):
        # RFC 6125, section 6.4.3, subitem 3.
        # The client SHOULD NOT attempt to match a presented identifier
        # where the wildcard character is embedded within an A-label or
        # U-label of an internationalized domain name.
        pats.append(re.escape(leftmost))
    else:
        # Otherwise, '*' matches any dotless string, e.g. www*
        pats.append(re.escape(leftmost).replace(r"\*", "[^.]*"))

    # add the remaining fragments, ignore any wildcards
    for frag in remainder:
        pats.append(re.escape(frag))

    pat = re.compile(r"\A" + r"\.".join(pats) + r"\Z", re.IGNORECASE)
    return pat.match(hostname)


def _to_unicode(obj):
    if isinstance(obj, str) and sys.version_info < (3,):
        # ignored flake8 # F821 to support python 2.7 function
        obj = unicode(obj, encoding="ascii", errors="strict")  # noqa: F821
    return obj


def _ipaddress_match(ipname, host_ip):
    """Exact matching of IP addresses.

    RFC 6125 explicitly doesn't define an algorithm for this
    (section 1.7.2 - "Out of Scope").
    """
    # OpenSSL may add a trailing newline to a subjectAltName's IP address
    # Divergence from upstream: ipaddress can't handle byte str
    ip = ipaddress.ip_address(_to_unicode(ipname).rstrip())
    return ip == host_ip


def match_hostname(cert, hostname):
    """Verify that *cert* (in decoded format as returned by
    SSLSocket.getpeercert()) matches the *hostname*.  RFC 2818 and RFC 6125
    rules are followed, but IP addresses are not accepted for *hostname*.

    CertificateError is raised on failure. On success, the function
    returns nothing.
    """
    if not cert:
        raise ValueError(
            "empty or no certificate, match_hostname needs a "
            "SSL socket or SSL context with either "
            "CERT_OPTIONAL or CERT_REQUIRED"
        )
    try:
        # Divergence from upstream: ipaddress can't handle byte str
        host_ip = ipaddress.ip_address(_to_unicode(hostname))
    except (UnicodeError, ValueError):
        # ValueError: Not an IP address (common case)
        # UnicodeError: Divergence from upstream: Have to deal with ipaddress not taking
        # byte strings.  addresses should be all ascii, so we consider it not
        # an ipaddress in this case
        host_ip = None
    except AttributeError:
        # Divergence from upstream: Make ipaddress library optional
        if ipaddress is None:
            host_ip = None
        else:  # Defensive
            raise
    dnsnames = []
    san = cert.get("subjectAltName", ())
    for key, value in san:
        if key == "DNS":
            if host_ip is None and _dnsname_match(value, hostname):
                return
            dnsnames.append(value)
        elif key == "IP Address":
            if host_ip is not None and _ipaddress_match(value, host_ip):
                return
            dnsnames.append(value)
    if not dnsnames:
        # The subject is only checked when there is no dNSName entry
        # in subjectAltName
        for sub in cert.get("subject", ()):
            for key, value in sub:
                # XXX according to RFC 2818, the most specific Common Name
                # must be used.
                if key == "commonName":
                    if _dnsname_match(value, hostname):
                        return
                    dnsnames.append(value)
    if len(dnsnames) > 1:
        raise CertificateError(
            "hostname %r "
            "doesn't match either of %s" % (hostname, ", ".join(map(repr, dnsnames)))
        )
    elif len(dnsnames) == 1:
        raise CertificateError("hostname %r doesn't match %r" % (hostname, dnsnames[0]))
    else:
        raise CertificateError(
            "no appropriate commonName or subjectAltName fields were found"
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\urllib3\util\ssl_match_hostname.py ===
"""The match_hostname() function from Python 3.5, essential when using SSL."""

# Note: This file is under the PSF license as the code comes from the python
# stdlib.   http://docs.python.org/3/license.html
# It is modified to remove commonName support.

from __future__ import annotations

import ipaddress
import re
import typing
from ipaddress import IPv4Address, IPv6Address

if typing.TYPE_CHECKING:
    from .ssl_ import _TYPE_PEER_CERT_RET_DICT

__version__ = "3.5.0.1"


class CertificateError(ValueError):
    pass


def _dnsname_match(
    dn: typing.Any, hostname: str, max_wildcards: int = 1
) -> typing.Match[str] | None | bool:
    """Matching according to RFC 6125, section 6.4.3

    http://tools.ietf.org/html/rfc6125#section-6.4.3
    """
    pats = []
    if not dn:
        return False

    # Ported from python3-syntax:
    # leftmost, *remainder = dn.split(r'.')
    parts = dn.split(r".")
    leftmost = parts[0]
    remainder = parts[1:]

    wildcards = leftmost.count("*")
    if wildcards > max_wildcards:
        # Issue #17980: avoid denials of service by refusing more
        # than one wildcard per fragment.  A survey of established
        # policy among SSL implementations showed it to be a
        # reasonable choice.
        raise CertificateError(
            "too many wildcards in certificate DNS name: " + repr(dn)
        )

    # speed up common case w/o wildcards
    if not wildcards:
        return bool(dn.lower() == hostname.lower())

    # RFC 6125, section 6.4.3, subitem 1.
    # The client SHOULD NOT attempt to match a presented identifier in which
    # the wildcard character comprises a label other than the left-most label.
    if leftmost == "*":
        # When '*' is a fragment by itself, it matches a non-empty dotless
        # fragment.
        pats.append("[^.]+")
    elif leftmost.startswith("xn--") or hostname.startswith("xn--"):
        # RFC 6125, section 6.4.3, subitem 3.
        # The client SHOULD NOT attempt to match a presented identifier
        # where the wildcard character is embedded within an A-label or
        # U-label of an internationalized domain name.
        pats.append(re.escape(leftmost))
    else:
        # Otherwise, '*' matches any dotless string, e.g. www*
        pats.append(re.escape(leftmost).replace(r"\*", "[^.]*"))

    # add the remaining fragments, ignore any wildcards
    for frag in remainder:
        pats.append(re.escape(frag))

    pat = re.compile(r"\A" + r"\.".join(pats) + r"\Z", re.IGNORECASE)
    return pat.match(hostname)


def _ipaddress_match(ipname: str, host_ip: IPv4Address | IPv6Address) -> bool:
    """Exact matching of IP addresses.

    RFC 9110 section 4.3.5: "A reference identity of IP-ID contains the decoded
    bytes of the IP address. An IP version 4 address is 4 octets, and an IP
    version 6 address is 16 octets. [...] A reference identity of type IP-ID
    matches if the address is identical to an iPAddress value of the
    subjectAltName extension of the certificate."
    """
    # OpenSSL may add a trailing newline to a subjectAltName's IP address
    # Divergence from upstream: ipaddress can't handle byte str
    ip = ipaddress.ip_address(ipname.rstrip())
    return bool(ip.packed == host_ip.packed)


def match_hostname(
    cert: _TYPE_PEER_CERT_RET_DICT | None,
    hostname: str,
    hostname_checks_common_name: bool = False,
) -> None:
    """Verify that *cert* (in decoded format as returned by
    SSLSocket.getpeercert()) matches the *hostname*.  RFC 2818 and RFC 6125
    rules are followed, but IP addresses are not accepted for *hostname*.

    CertificateError is raised on failure. On success, the function
    returns nothing.
    """
    if not cert:
        raise ValueError(
            "empty or no certificate, match_hostname needs a "
            "SSL socket or SSL context with either "
            "CERT_OPTIONAL or CERT_REQUIRED"
        )
    try:
        # Divergence from upstream: ipaddress can't handle byte str
        #
        # The ipaddress module shipped with Python < 3.9 does not support
        # scoped IPv6 addresses so we unconditionally strip the Zone IDs for
        # now. Once we drop support for Python 3.9 we can remove this branch.
        if "%" in hostname:
            host_ip = ipaddress.ip_address(hostname[: hostname.rfind("%")])
        else:
            host_ip = ipaddress.ip_address(hostname)

    except ValueError:
        # Not an IP address (common case)
        host_ip = None
    dnsnames = []
    san: tuple[tuple[str, str], ...] = cert.get("subjectAltName", ())
    key: str
    value: str
    for key, value in san:
        if key == "DNS":
            if host_ip is None and _dnsname_match(value, hostname):
                return
            dnsnames.append(value)
        elif key == "IP Address":
            if host_ip is not None and _ipaddress_match(value, host_ip):
                return
            dnsnames.append(value)

    # We only check 'commonName' if it's enabled and we're not verifying
    # an IP address. IP addresses aren't valid within 'commonName'.
    if hostname_checks_common_name and host_ip is None and not dnsnames:
        for sub in cert.get("subject", ()):
            for key, value in sub:
                if key == "commonName":
                    if _dnsname_match(value, hostname):
                        return
                    dnsnames.append(value)  # Defensive: for Python < 3.9.3

    if len(dnsnames) > 1:
        raise CertificateError(
            "hostname %r "
            "doesn't match either of %s" % (hostname, ", ".join(map(repr, dnsnames)))
        )
    elif len(dnsnames) == 1:
        raise CertificateError(f"hostname {hostname!r} doesn't match {dnsnames[0]!r}")
    else:
        raise CertificateError("no appropriate subjectAltName fields were found")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\et_xmlfile\xmlfile.py ===
from __future__ import absolute_import
# Copyright (c) 2010-2015 openpyxl

"""Implements the lxml.etree.xmlfile API using the standard library xml.etree"""


from contextlib import contextmanager

from xml.etree.ElementTree import (
    Element,
    _escape_cdata,
)

from . import incremental_tree


class LxmlSyntaxError(Exception):
    pass


class _IncrementalFileWriter(object):
    """Replacement for _IncrementalFileWriter of lxml"""
    def __init__(self, output_file):
        self._element_stack = []
        self._file = output_file
        self._have_root = False
        self.global_nsmap = incremental_tree.current_global_nsmap()
        self.is_html = False

    @contextmanager
    def element(self, tag, attrib=None, nsmap=None, **_extra):
        """Create a new xml element using a context manager."""
        if nsmap and None in nsmap:
            # Normalise None prefix (lxml's default namespace prefix) -> "", as
            # required for incremental_tree
            if "" in nsmap and nsmap[""] != nsmap[None]:
                raise ValueError(
                    'Found None and "" as default nsmap prefixes with different URIs'
                )
            nsmap = nsmap.copy()
            nsmap[""] = nsmap.pop(None)

        # __enter__ part
        self._have_root = True
        if attrib is None:
            attrib = {}
        elem = Element(tag, attrib=attrib, **_extra)
        elem.text = ''
        elem.tail = ''
        if self._element_stack:
            is_root = False
            (
                nsmap_scope,
                default_ns_attr_prefix,
                uri_to_prefix,
            ) = self._element_stack[-1]
        else:
            is_root = True
            nsmap_scope = {}
            default_ns_attr_prefix = None
            uri_to_prefix = {}
        (
            tag,
            nsmap_scope,
            default_ns_attr_prefix,
            uri_to_prefix,
            next_remains_root,
        ) = incremental_tree.write_elem_start(
            self._file,
            elem,
            nsmap_scope=nsmap_scope,
            global_nsmap=self.global_nsmap,
            short_empty_elements=False,
            is_html=self.is_html,
            is_root=is_root,
            uri_to_prefix=uri_to_prefix,
            default_ns_attr_prefix=default_ns_attr_prefix,
            new_nsmap=nsmap,
        )
        self._element_stack.append(
            (
                nsmap_scope,
                default_ns_attr_prefix,
                uri_to_prefix,
            )
        )
        yield

        # __exit__ part
        self._element_stack.pop()
        self._file(f"</{tag}>")
        if elem.tail:
            self._file(_escape_cdata(elem.tail))

    def write(self, arg):
        """Write a string or subelement."""

        if isinstance(arg, str):
            # it is not allowed to write a string outside of an element
            if not self._element_stack:
                raise LxmlSyntaxError()
            self._file(_escape_cdata(arg))

        else:
            if not self._element_stack and self._have_root:
                raise LxmlSyntaxError()

            if self._element_stack:
                is_root = False
                (
                    nsmap_scope,
                    default_ns_attr_prefix,
                    uri_to_prefix,
                ) = self._element_stack[-1]
            else:
                is_root = True
                nsmap_scope = {}
                default_ns_attr_prefix = None
                uri_to_prefix = {}
            incremental_tree._serialize_ns_xml(
                self._file,
                arg,
                nsmap_scope=nsmap_scope,
                global_nsmap=self.global_nsmap,
                short_empty_elements=True,
                is_html=self.is_html,
                is_root=is_root,
                uri_to_prefix=uri_to_prefix,
                default_ns_attr_prefix=default_ns_attr_prefix,
            )

    def __enter__(self):
        pass

    def __exit__(self, type, value, traceback):
        # without root the xml document is incomplete
        if not self._have_root:
            raise LxmlSyntaxError()


class xmlfile(object):
    """Context manager that can replace lxml.etree.xmlfile."""
    def __init__(self, output_file, buffered=False, encoding="utf-8", close=False):
        self._file = output_file
        self._close = close
        self.encoding = encoding
        self.writer_cm = None

    def __enter__(self):
        self.writer_cm = incremental_tree._get_writer(self._file, encoding=self.encoding)
        writer, declared_encoding = self.writer_cm.__enter__()
        return _IncrementalFileWriter(writer)

    def __exit__(self, type, value, traceback):
        if self.writer_cm:
            self.writer_cm.__exit__(type, value, traceback)
        if self._close:
            self._file.close()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\packaging\relationship.py ===
# Copyright (c) 2010-2024 openpyxl

import posixpath
from warnings import warn

from openpyxl.descriptors import (
    String,
    Alias,
    Sequence,
)
from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors.container import ElementList

from openpyxl.xml.constants import REL_NS, PKG_REL_NS
from openpyxl.xml.functions import (
    Element,
    fromstring,
)


class Relationship(Serialisable):
    """Represents many kinds of relationships."""

    tagname = "Relationship"

    Type = String()
    Target = String()
    target = Alias("Target")
    TargetMode = String(allow_none=True)
    Id = String(allow_none=True)
    id = Alias("Id")


    def __init__(self,
                 Id=None,
                 Type=None,
                 type=None,
                 Target=None,
                 TargetMode=None
                 ):
        """
        `type` can be used as a shorthand with the default relationships namespace
        otherwise the `Type` must be a fully qualified URL
        """
        if type is not None:
            Type = "{0}/{1}".format(REL_NS, type)
        self.Type = Type
        self.Target = Target
        self.TargetMode = TargetMode
        self.Id = Id


class RelationshipList(ElementList):

    tagname = "Relationships"
    expected_type = Relationship


    def append(self, value):
        super().append(value)
        if not value.Id:
            value.Id = f"rId{len(self)}"


    def find(self, content_type):
        """
        Find relationships by content-type
        NB. these content-types namespaced objects and different to the MIME-types
        in the package manifest :-(
        """
        for r in self:
            if r.Type == content_type:
                yield r


    def get(self, key):
        for r in self:
            if r.Id == key:
                return r
        raise KeyError("Unknown relationship: {0}".format(key))


    def to_dict(self):
        """Return a dictionary of relations keyed by id"""
        return {r.id:r for r in self}


    def to_tree(self):
        tree = super().to_tree()
        tree.set("xmlns", PKG_REL_NS)
        return tree


def get_rels_path(path):
    """
    Convert relative path to absolutes that can be loaded from a zip
    archive.
    The path to be passed in is that of containing object (workbook,
    worksheet, etc.)
    """
    folder, obj = posixpath.split(path)
    filename = posixpath.join(folder, '_rels', '{0}.rels'.format(obj))
    return filename


def get_dependents(archive, filename):
    """
    Normalise dependency file paths to absolute ones

    Relative paths are relative to parent object
    """
    src = archive.read(filename)
    node = fromstring(src)
    try:
        rels = RelationshipList.from_tree(node)
    except TypeError:
        msg = "{0} contains invalid dependency definitions".format(filename)
        warn(msg)
        rels = RelationshipList()
    folder = posixpath.dirname(filename)
    parent = posixpath.split(folder)[0]
    for r in rels:
        if r.TargetMode == "External":
            continue
        elif r.target.startswith("/"):
            r.target = r.target[1:]
        else:
            pth = posixpath.join(parent, r.target)
            r.target = posixpath.normpath(pth)
    return rels


def get_rel(archive, deps, id=None, cls=None):
    """
    Get related object based on id or rel_type
    """
    if not any([id, cls]):
        raise ValueError("Either the id or the content type are required")
    if id is not None:
        rel = deps.get(id)
    else:
        try:
            rel = next(deps.find(cls.rel_type))
        except StopIteration: # no known dependency
            return

    path = rel.target
    src = archive.read(path)
    tree = fromstring(src)
    obj = cls.from_tree(tree)

    rels_path = get_rels_path(path)
    try:
        obj.deps = get_dependents(archive, rels_path)
    except KeyError:
        obj.deps = []

    return obj

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\util\_print_versions.py ===
from __future__ import annotations

import codecs
import json
import locale
import os
import platform
import struct
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pandas._typing import JSONSerializable

from pandas.compat._optional import (
    VERSIONS,
    get_version,
    import_optional_dependency,
)


def _get_commit_hash() -> str | None:
    """
    Use vendored versioneer code to get git hash, which handles
    git worktree correctly.
    """
    try:
        from pandas._version_meson import (  # pyright: ignore [reportMissingImports]
            __git_version__,
        )

        return __git_version__
    except ImportError:
        from pandas._version import get_versions

        versions = get_versions()
        return versions["full-revisionid"]


def _get_sys_info() -> dict[str, JSONSerializable]:
    """
    Returns system information as a JSON serializable dictionary.
    """
    uname_result = platform.uname()
    language_code, encoding = locale.getlocale()
    return {
        "commit": _get_commit_hash(),
        "python": platform.python_version(),
        "python-bits": struct.calcsize("P") * 8,
        "OS": uname_result.system,
        "OS-release": uname_result.release,
        "Version": uname_result.version,
        "machine": uname_result.machine,
        "processor": uname_result.processor,
        "byteorder": sys.byteorder,
        "LC_ALL": os.environ.get("LC_ALL"),
        "LANG": os.environ.get("LANG"),
        "LOCALE": {"language-code": language_code, "encoding": encoding},
    }


def _get_dependency_info() -> dict[str, JSONSerializable]:
    """
    Returns dependency information as a JSON serializable dictionary.
    """
    deps = [
        "pandas",
        # required
        "numpy",
        "pytz",
        "dateutil",
        # install / build,
        "pip",
        "Cython",
        # docs
        "sphinx",
        # Other, not imported.
        "IPython",
    ]
    # Optional dependencies
    deps.extend(list(VERSIONS))

    result: dict[str, JSONSerializable] = {}
    for modname in deps:
        try:
            mod = import_optional_dependency(modname, errors="ignore")
        except Exception:
            # Dependency conflicts may cause a non ImportError
            result[modname] = "N/A"
        else:
            result[modname] = get_version(mod) if mod else None
    return result


def show_versions(as_json: str | bool = False) -> None:
    """
    Provide useful information, important for bug reports.

    It comprises info about hosting operation system, pandas version,
    and versions of other installed relative packages.

    Parameters
    ----------
    as_json : str or bool, default False
        * If False, outputs info in a human readable form to the console.
        * If str, it will be considered as a path to a file.
          Info will be written to that file in JSON format.
        * If True, outputs info in JSON format to the console.

    Examples
    --------
    >>> pd.show_versions()  # doctest: +SKIP
    Your output may look something like this:
    INSTALLED VERSIONS
    ------------------
    commit           : 37ea63d540fd27274cad6585082c91b1283f963d
    python           : 3.10.6.final.0
    python-bits      : 64
    OS               : Linux
    OS-release       : 5.10.102.1-microsoft-standard-WSL2
    Version          : #1 SMP Wed Mar 2 00:30:59 UTC 2022
    machine          : x86_64
    processor        : x86_64
    byteorder        : little
    LC_ALL           : None
    LANG             : en_GB.UTF-8
    LOCALE           : en_GB.UTF-8
    pandas           : 2.0.1
    numpy            : 1.24.3
    ...
    """
    sys_info = _get_sys_info()
    deps = _get_dependency_info()

    if as_json:
        j = {"system": sys_info, "dependencies": deps}

        if as_json is True:
            sys.stdout.writelines(json.dumps(j, indent=2))
        else:
            assert isinstance(as_json, str)  # needed for mypy
            with codecs.open(as_json, "wb", encoding="utf8") as f:
                json.dump(j, f, indent=2)

    else:
        assert isinstance(sys_info["LOCALE"], dict)  # needed for mypy
        language_code = sys_info["LOCALE"]["language-code"]
        encoding = sys_info["LOCALE"]["encoding"]
        sys_info["LOCALE"] = f"{language_code}.{encoding}"

        maxlen = max(len(x) for x in deps)
        print("\nINSTALLED VERSIONS")
        print("------------------")
        for k, v in sys_info.items():
            print(f"{k:<{maxlen}}: {v}")
        print("")
        for k, v in deps.items():
            print(f"{k:<{maxlen}}: {v}")

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\distributions\sdist.py ===
import logging
from typing import TYPE_CHECKING, Iterable, Optional, Set, Tuple

from pip._internal.build_env import BuildEnvironment
from pip._internal.distributions.base import AbstractDistribution
from pip._internal.exceptions import InstallationError
from pip._internal.metadata import BaseDistribution
from pip._internal.utils.subprocess import runner_with_spinner_message

if TYPE_CHECKING:
    from pip._internal.index.package_finder import PackageFinder

logger = logging.getLogger(__name__)


class SourceDistribution(AbstractDistribution):
    """Represents a source distribution.

    The preparation step for these needs metadata for the packages to be
    generated, either using PEP 517 or using the legacy `setup.py egg_info`.
    """

    @property
    def build_tracker_id(self) -> Optional[str]:
        """Identify this requirement uniquely by its link."""
        assert self.req.link
        return self.req.link.url_without_fragment

    def get_metadata_distribution(self) -> BaseDistribution:
        return self.req.get_dist()

    def prepare_distribution_metadata(
        self,
        finder: "PackageFinder",
        build_isolation: bool,
        check_build_deps: bool,
    ) -> None:
        # Load pyproject.toml, to determine whether PEP 517 is to be used
        self.req.load_pyproject_toml()

        # Set up the build isolation, if this requirement should be isolated
        should_isolate = self.req.use_pep517 and build_isolation
        if should_isolate:
            # Setup an isolated environment and install the build backend static
            # requirements in it.
            self._prepare_build_backend(finder)
            # Check that if the requirement is editable, it either supports PEP 660 or
            # has a setup.py or a setup.cfg. This cannot be done earlier because we need
            # to setup the build backend to verify it supports build_editable, nor can
            # it be done later, because we want to avoid installing build requirements
            # needlessly. Doing it here also works around setuptools generating
            # UNKNOWN.egg-info when running get_requires_for_build_wheel on a directory
            # without setup.py nor setup.cfg.
            self.req.isolated_editable_sanity_check()
            # Install the dynamic build requirements.
            self._install_build_reqs(finder)
        # Check if the current environment provides build dependencies
        should_check_deps = self.req.use_pep517 and check_build_deps
        if should_check_deps:
            pyproject_requires = self.req.pyproject_requires
            assert pyproject_requires is not None
            conflicting, missing = self.req.build_env.check_requirements(
                pyproject_requires
            )
            if conflicting:
                self._raise_conflicts("the backend dependencies", conflicting)
            if missing:
                self._raise_missing_reqs(missing)
        self.req.prepare_metadata()

    def _prepare_build_backend(self, finder: "PackageFinder") -> None:
        # Isolate in a BuildEnvironment and install the build-time
        # requirements.
        pyproject_requires = self.req.pyproject_requires
        assert pyproject_requires is not None

        self.req.build_env = BuildEnvironment()
        self.req.build_env.install_requirements(
            finder, pyproject_requires, "overlay", kind="build dependencies"
        )
        conflicting, missing = self.req.build_env.check_requirements(
            self.req.requirements_to_check
        )
        if conflicting:
            self._raise_conflicts("PEP 517/518 supported requirements", conflicting)
        if missing:
            logger.warning(
                "Missing build requirements in pyproject.toml for %s.",
                self.req,
            )
            logger.warning(
                "The project does not specify a build backend, and "
                "pip cannot fall back to setuptools without %s.",
                " and ".join(map(repr, sorted(missing))),
            )

    def _get_build_requires_wheel(self) -> Iterable[str]:
        with self.req.build_env:
            runner = runner_with_spinner_message("Getting requirements to build wheel")
            backend = self.req.pep517_backend
            assert backend is not None
            with backend.subprocess_runner(runner):
                return backend.get_requires_for_build_wheel()

    def _get_build_requires_editable(self) -> Iterable[str]:
        with self.req.build_env:
            runner = runner_with_spinner_message(
                "Getting requirements to build editable"
            )
            backend = self.req.pep517_backend
            assert backend is not None
            with backend.subprocess_runner(runner):
                return backend.get_requires_for_build_editable()

    def _install_build_reqs(self, finder: "PackageFinder") -> None:
        # Install any extra build dependencies that the backend requests.
        # This must be done in a second pass, as the pyproject.toml
        # dependencies must be installed before we can call the backend.
        if (
            self.req.editable
            and self.req.permit_editable_wheels
            and self.req.supports_pyproject_editable
        ):
            build_reqs = self._get_build_requires_editable()
        else:
            build_reqs = self._get_build_requires_wheel()
        conflicting, missing = self.req.build_env.check_requirements(build_reqs)
        if conflicting:
            self._raise_conflicts("the backend dependencies", conflicting)
        self.req.build_env.install_requirements(
            finder, missing, "normal", kind="backend dependencies"
        )

    def _raise_conflicts(
        self, conflicting_with: str, conflicting_reqs: Set[Tuple[str, str]]
    ) -> None:
        format_string = (
            "Some build dependencies for {requirement} "
            "conflict with {conflicting_with}: {description}."
        )
        error_message = format_string.format(
            requirement=self.req,
            conflicting_with=conflicting_with,
            description=", ".join(
                f"{installed} is incompatible with {wanted}"
                for installed, wanted in sorted(conflicting_reqs)
            ),
        )
        raise InstallationError(error_message)

    def _raise_missing_reqs(self, missing: Set[str]) -> None:
        format_string = (
            "Some build dependencies for {requirement} are missing: {missing}."
        )
        error_message = format_string.format(
            requirement=self.req, missing=", ".join(map(repr, sorted(missing)))
        )
        raise InstallationError(error_message)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\humanfriendly\case.py ===
# Human friendly input/output in Python.
#
# Author: Peter Odding <peter@peterodding.com>
# Last Change: April 19, 2020
# URL: https://humanfriendly.readthedocs.io

"""
Simple case insensitive dictionaries.

The :class:`CaseInsensitiveDict` class is a dictionary whose string keys
are case insensitive. It works by automatically coercing string keys to
:class:`CaseInsensitiveKey` objects. Keys that are not strings are
supported as well, just without case insensitivity.

At its core this module works by normalizing strings to lowercase before
comparing or hashing them. It doesn't support proper case folding nor
does it support Unicode normalization, hence the word "simple".
"""

# Standard library modules.
import collections

try:
    # Python >= 3.3.
    from collections.abc import Iterable, Mapping
except ImportError:
    # Python 2.7.
    from collections import Iterable, Mapping

# Modules included in our package.
from humanfriendly.compat import basestring, unicode

# Public identifiers that require documentation.
__all__ = ("CaseInsensitiveDict", "CaseInsensitiveKey")


class CaseInsensitiveDict(collections.OrderedDict):

    """
    Simple case insensitive dictionary implementation (that remembers insertion order).

    This class works by overriding methods that deal with dictionary keys to
    coerce string keys to :class:`CaseInsensitiveKey` objects before calling
    down to the regular dictionary handling methods. While intended to be
    complete this class has not been extensively tested yet.
    """

    def __init__(self, other=None, **kw):
        """Initialize a :class:`CaseInsensitiveDict` object."""
        # Initialize our superclass.
        super(CaseInsensitiveDict, self).__init__()
        # Handle the initializer arguments.
        self.update(other, **kw)

    def coerce_key(self, key):
        """
        Coerce string keys to :class:`CaseInsensitiveKey` objects.

        :param key: The value to coerce (any type).
        :returns: If `key` is a string then a :class:`CaseInsensitiveKey`
                  object is returned, otherwise the value of `key` is
                  returned unmodified.
        """
        if isinstance(key, basestring):
            key = CaseInsensitiveKey(key)
        return key

    @classmethod
    def fromkeys(cls, iterable, value=None):
        """Create a case insensitive dictionary with keys from `iterable` and values set to `value`."""
        return cls((k, value) for k in iterable)

    def get(self, key, default=None):
        """Get the value of an existing item."""
        return super(CaseInsensitiveDict, self).get(self.coerce_key(key), default)

    def pop(self, key, default=None):
        """Remove an item from a case insensitive dictionary."""
        return super(CaseInsensitiveDict, self).pop(self.coerce_key(key), default)

    def setdefault(self, key, default=None):
        """Get the value of an existing item or add a new item."""
        return super(CaseInsensitiveDict, self).setdefault(self.coerce_key(key), default)

    def update(self, other=None, **kw):
        """Update a case insensitive dictionary with new items."""
        if isinstance(other, Mapping):
            # Copy the items from the given mapping.
            for key, value in other.items():
                self[key] = value
        elif isinstance(other, Iterable):
            # Copy the items from the given iterable.
            for key, value in other:
                self[key] = value
        elif other is not None:
            # Complain about unsupported values.
            msg = "'%s' object is not iterable"
            type_name = type(value).__name__
            raise TypeError(msg % type_name)
        # Copy the keyword arguments (if any).
        for key, value in kw.items():
            self[key] = value

    def __contains__(self, key):
        """Check if a case insensitive dictionary contains the given key."""
        return super(CaseInsensitiveDict, self).__contains__(self.coerce_key(key))

    def __delitem__(self, key):
        """Delete an item in a case insensitive dictionary."""
        return super(CaseInsensitiveDict, self).__delitem__(self.coerce_key(key))

    def __getitem__(self, key):
        """Get the value of an item in a case insensitive dictionary."""
        return super(CaseInsensitiveDict, self).__getitem__(self.coerce_key(key))

    def __setitem__(self, key, value):
        """Set the value of an item in a case insensitive dictionary."""
        return super(CaseInsensitiveDict, self).__setitem__(self.coerce_key(key), value)


class CaseInsensitiveKey(unicode):

    """
    Simple case insensitive dictionary key implementation.

    The :class:`CaseInsensitiveKey` class provides an intentionally simple
    implementation of case insensitive strings to be used as dictionary keys.

    If you need features like Unicode normalization or proper case folding
    please consider using a more advanced implementation like the :pypi:`istr`
    package instead.
    """

    def __new__(cls, value):
        """Create a :class:`CaseInsensitiveKey` object."""
        # Delegate string object creation to our superclass.
        obj = unicode.__new__(cls, value)
        # Store the lowercased string and its hash value.
        normalized = obj.lower()
        obj._normalized = normalized
        obj._hash_value = hash(normalized)
        return obj

    def __hash__(self):
        """Get the hash value of the lowercased string."""
        return self._hash_value

    def __eq__(self, other):
        """Compare two strings as lowercase."""
        if isinstance(other, CaseInsensitiveKey):
            # Fast path (and the most common case): Comparison with same type.
            return self._normalized == other._normalized
        elif isinstance(other, unicode):
            # Slow path: Comparison with strings that need lowercasing.
            return self._normalized == other.lower()
        else:
            return NotImplemented

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\io\formats\_color_data.py ===
# GH37967: Enable the use of CSS named colors, as defined in
# matplotlib.colors.CSS4_COLORS, when exporting to Excel.
# This data has been copied here, instead of being imported from matplotlib,
# not to have ``to_excel`` methods require matplotlib.
# source: matplotlib._color_data (3.3.3)
from __future__ import annotations

CSS4_COLORS = {
    "aliceblue": "F0F8FF",
    "antiquewhite": "FAEBD7",
    "aqua": "00FFFF",
    "aquamarine": "7FFFD4",
    "azure": "F0FFFF",
    "beige": "F5F5DC",
    "bisque": "FFE4C4",
    "black": "000000",
    "blanchedalmond": "FFEBCD",
    "blue": "0000FF",
    "blueviolet": "8A2BE2",
    "brown": "A52A2A",
    "burlywood": "DEB887",
    "cadetblue": "5F9EA0",
    "chartreuse": "7FFF00",
    "chocolate": "D2691E",
    "coral": "FF7F50",
    "cornflowerblue": "6495ED",
    "cornsilk": "FFF8DC",
    "crimson": "DC143C",
    "cyan": "00FFFF",
    "darkblue": "00008B",
    "darkcyan": "008B8B",
    "darkgoldenrod": "B8860B",
    "darkgray": "A9A9A9",
    "darkgreen": "006400",
    "darkgrey": "A9A9A9",
    "darkkhaki": "BDB76B",
    "darkmagenta": "8B008B",
    "darkolivegreen": "556B2F",
    "darkorange": "FF8C00",
    "darkorchid": "9932CC",
    "darkred": "8B0000",
    "darksalmon": "E9967A",
    "darkseagreen": "8FBC8F",
    "darkslateblue": "483D8B",
    "darkslategray": "2F4F4F",
    "darkslategrey": "2F4F4F",
    "darkturquoise": "00CED1",
    "darkviolet": "9400D3",
    "deeppink": "FF1493",
    "deepskyblue": "00BFFF",
    "dimgray": "696969",
    "dimgrey": "696969",
    "dodgerblue": "1E90FF",
    "firebrick": "B22222",
    "floralwhite": "FFFAF0",
    "forestgreen": "228B22",
    "fuchsia": "FF00FF",
    "gainsboro": "DCDCDC",
    "ghostwhite": "F8F8FF",
    "gold": "FFD700",
    "goldenrod": "DAA520",
    "gray": "808080",
    "green": "008000",
    "greenyellow": "ADFF2F",
    "grey": "808080",
    "honeydew": "F0FFF0",
    "hotpink": "FF69B4",
    "indianred": "CD5C5C",
    "indigo": "4B0082",
    "ivory": "FFFFF0",
    "khaki": "F0E68C",
    "lavender": "E6E6FA",
    "lavenderblush": "FFF0F5",
    "lawngreen": "7CFC00",
    "lemonchiffon": "FFFACD",
    "lightblue": "ADD8E6",
    "lightcoral": "F08080",
    "lightcyan": "E0FFFF",
    "lightgoldenrodyellow": "FAFAD2",
    "lightgray": "D3D3D3",
    "lightgreen": "90EE90",
    "lightgrey": "D3D3D3",
    "lightpink": "FFB6C1",
    "lightsalmon": "FFA07A",
    "lightseagreen": "20B2AA",
    "lightskyblue": "87CEFA",
    "lightslategray": "778899",
    "lightslategrey": "778899",
    "lightsteelblue": "B0C4DE",
    "lightyellow": "FFFFE0",
    "lime": "00FF00",
    "limegreen": "32CD32",
    "linen": "FAF0E6",
    "magenta": "FF00FF",
    "maroon": "800000",
    "mediumaquamarine": "66CDAA",
    "mediumblue": "0000CD",
    "mediumorchid": "BA55D3",
    "mediumpurple": "9370DB",
    "mediumseagreen": "3CB371",
    "mediumslateblue": "7B68EE",
    "mediumspringgreen": "00FA9A",
    "mediumturquoise": "48D1CC",
    "mediumvioletred": "C71585",
    "midnightblue": "191970",
    "mintcream": "F5FFFA",
    "mistyrose": "FFE4E1",
    "moccasin": "FFE4B5",
    "navajowhite": "FFDEAD",
    "navy": "000080",
    "oldlace": "FDF5E6",
    "olive": "808000",
    "olivedrab": "6B8E23",
    "orange": "FFA500",
    "orangered": "FF4500",
    "orchid": "DA70D6",
    "palegoldenrod": "EEE8AA",
    "palegreen": "98FB98",
    "paleturquoise": "AFEEEE",
    "palevioletred": "DB7093",
    "papayawhip": "FFEFD5",
    "peachpuff": "FFDAB9",
    "peru": "CD853F",
    "pink": "FFC0CB",
    "plum": "DDA0DD",
    "powderblue": "B0E0E6",
    "purple": "800080",
    "rebeccapurple": "663399",
    "red": "FF0000",
    "rosybrown": "BC8F8F",
    "royalblue": "4169E1",
    "saddlebrown": "8B4513",
    "salmon": "FA8072",
    "sandybrown": "F4A460",
    "seagreen": "2E8B57",
    "seashell": "FFF5EE",
    "sienna": "A0522D",
    "silver": "C0C0C0",
    "skyblue": "87CEEB",
    "slateblue": "6A5ACD",
    "slategray": "708090",
    "slategrey": "708090",
    "snow": "FFFAFA",
    "springgreen": "00FF7F",
    "steelblue": "4682B4",
    "tan": "D2B48C",
    "teal": "008080",
    "thistle": "D8BFD8",
    "tomato": "FF6347",
    "turquoise": "40E0D0",
    "violet": "EE82EE",
    "wheat": "F5DEB3",
    "white": "FFFFFF",
    "whitesmoke": "F5F5F5",
    "yellow": "FFFF00",
    "yellowgreen": "9ACD32",
}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\cachecontrol\heuristics.py ===
# SPDX-FileCopyrightText: 2015 Eric Larson
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import calendar
import time
from datetime import datetime, timedelta, timezone
from email.utils import formatdate, parsedate, parsedate_tz
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from pip._vendor.urllib3 import HTTPResponse

TIME_FMT = "%a, %d %b %Y %H:%M:%S GMT"


def expire_after(delta: timedelta, date: datetime | None = None) -> datetime:
    date = date or datetime.now(timezone.utc)
    return date + delta


def datetime_to_header(dt: datetime) -> str:
    return formatdate(calendar.timegm(dt.timetuple()))


class BaseHeuristic:
    def warning(self, response: HTTPResponse) -> str | None:
        """
        Return a valid 1xx warning header value describing the cache
        adjustments.

        The response is provided too allow warnings like 113
        http://tools.ietf.org/html/rfc7234#section-5.5.4 where we need
        to explicitly say response is over 24 hours old.
        """
        return '110 - "Response is Stale"'

    def update_headers(self, response: HTTPResponse) -> dict[str, str]:
        """Update the response headers with any new headers.

        NOTE: This SHOULD always include some Warning header to
              signify that the response was cached by the client, not
              by way of the provided headers.
        """
        return {}

    def apply(self, response: HTTPResponse) -> HTTPResponse:
        updated_headers = self.update_headers(response)

        if updated_headers:
            response.headers.update(updated_headers)
            warning_header_value = self.warning(response)
            if warning_header_value is not None:
                response.headers.update({"Warning": warning_header_value})

        return response


class OneDayCache(BaseHeuristic):
    """
    Cache the response by providing an expires 1 day in the
    future.
    """

    def update_headers(self, response: HTTPResponse) -> dict[str, str]:
        headers = {}

        if "expires" not in response.headers:
            date = parsedate(response.headers["date"])
            expires = expire_after(
                timedelta(days=1),
                date=datetime(*date[:6], tzinfo=timezone.utc),  # type: ignore[index,misc]
            )
            headers["expires"] = datetime_to_header(expires)
            headers["cache-control"] = "public"
        return headers


class ExpiresAfter(BaseHeuristic):
    """
    Cache **all** requests for a defined time period.
    """

    def __init__(self, **kw: Any) -> None:
        self.delta = timedelta(**kw)

    def update_headers(self, response: HTTPResponse) -> dict[str, str]:
        expires = expire_after(self.delta)
        return {"expires": datetime_to_header(expires), "cache-control": "public"}

    def warning(self, response: HTTPResponse) -> str | None:
        tmpl = "110 - Automatically cached for %s. Response might be stale"
        return tmpl % self.delta


class LastModified(BaseHeuristic):
    """
    If there is no Expires header already, fall back on Last-Modified
    using the heuristic from
    http://tools.ietf.org/html/rfc7234#section-4.2.2
    to calculate a reasonable value.

    Firefox also does something like this per
    https://developer.mozilla.org/en-US/docs/Web/HTTP/Caching_FAQ
    http://lxr.mozilla.org/mozilla-release/source/netwerk/protocol/http/nsHttpResponseHead.cpp#397
    Unlike mozilla we limit this to 24-hr.
    """

    cacheable_by_default_statuses = {
        200,
        203,
        204,
        206,
        300,
        301,
        404,
        405,
        410,
        414,
        501,
    }

    def update_headers(self, resp: HTTPResponse) -> dict[str, str]:
        headers: Mapping[str, str] = resp.headers

        if "expires" in headers:
            return {}

        if "cache-control" in headers and headers["cache-control"] != "public":
            return {}

        if resp.status not in self.cacheable_by_default_statuses:
            return {}

        if "date" not in headers or "last-modified" not in headers:
            return {}

        time_tuple = parsedate_tz(headers["date"])
        assert time_tuple is not None
        date = calendar.timegm(time_tuple[:6])
        last_modified = parsedate(headers["last-modified"])
        if last_modified is None:
            return {}

        now = time.time()
        current_age = max(0, now - date)
        delta = date - calendar.timegm(last_modified)
        freshness_lifetime = max(0, min(delta / 10, 24 * 3600))
        if freshness_lifetime <= current_age:
            return {}

        expires = date + freshness_lifetime
        return {"expires": time.strftime(TIME_FMT, time.gmtime(expires))}

    def warning(self, resp: HTTPResponse) -> str | None:
        return None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\pygments\formatters\__init__.py ===
"""
    pygments.formatters
    ~~~~~~~~~~~~~~~~~~~

    Pygments formatters.

    :copyright: Copyright 2006-2025 by the Pygments team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""

import re
import sys
import types
import fnmatch
from os.path import basename

from pip._vendor.pygments.formatters._mapping import FORMATTERS
from pip._vendor.pygments.plugin import find_plugin_formatters
from pip._vendor.pygments.util import ClassNotFound

__all__ = ['get_formatter_by_name', 'get_formatter_for_filename',
           'get_all_formatters', 'load_formatter_from_file'] + list(FORMATTERS)

_formatter_cache = {}  # classes by name
_pattern_cache = {}


def _fn_matches(fn, glob):
    """Return whether the supplied file name fn matches pattern filename."""
    if glob not in _pattern_cache:
        pattern = _pattern_cache[glob] = re.compile(fnmatch.translate(glob))
        return pattern.match(fn)
    return _pattern_cache[glob].match(fn)


def _load_formatters(module_name):
    """Load a formatter (and all others in the module too)."""
    mod = __import__(module_name, None, None, ['__all__'])
    for formatter_name in mod.__all__:
        cls = getattr(mod, formatter_name)
        _formatter_cache[cls.name] = cls


def get_all_formatters():
    """Return a generator for all formatter classes."""
    # NB: this returns formatter classes, not info like get_all_lexers().
    for info in FORMATTERS.values():
        if info[1] not in _formatter_cache:
            _load_formatters(info[0])
        yield _formatter_cache[info[1]]
    for _, formatter in find_plugin_formatters():
        yield formatter


def find_formatter_class(alias):
    """Lookup a formatter by alias.

    Returns None if not found.
    """
    for module_name, name, aliases, _, _ in FORMATTERS.values():
        if alias in aliases:
            if name not in _formatter_cache:
                _load_formatters(module_name)
            return _formatter_cache[name]
    for _, cls in find_plugin_formatters():
        if alias in cls.aliases:
            return cls


def get_formatter_by_name(_alias, **options):
    """
    Return an instance of a :class:`.Formatter` subclass that has `alias` in its
    aliases list. The formatter is given the `options` at its instantiation.

    Will raise :exc:`pygments.util.ClassNotFound` if no formatter with that
    alias is found.
    """
    cls = find_formatter_class(_alias)
    if cls is None:
        raise ClassNotFound(f"no formatter found for name {_alias!r}")
    return cls(**options)


def load_formatter_from_file(filename, formattername="CustomFormatter", **options):
    """
    Return a `Formatter` subclass instance loaded from the provided file, relative
    to the current directory.

    The file is expected to contain a Formatter class named ``formattername``
    (by default, CustomFormatter). Users should be very careful with the input, because
    this method is equivalent to running ``eval()`` on the input file. The formatter is
    given the `options` at its instantiation.

    :exc:`pygments.util.ClassNotFound` is raised if there are any errors loading
    the formatter.

    .. versionadded:: 2.2
    """
    try:
        # This empty dict will contain the namespace for the exec'd file
        custom_namespace = {}
        with open(filename, 'rb') as f:
            exec(f.read(), custom_namespace)
        # Retrieve the class `formattername` from that namespace
        if formattername not in custom_namespace:
            raise ClassNotFound(f'no valid {formattername} class found in {filename}')
        formatter_class = custom_namespace[formattername]
        # And finally instantiate it with the options
        return formatter_class(**options)
    except OSError as err:
        raise ClassNotFound(f'cannot read {filename}: {err}')
    except ClassNotFound:
        raise
    except Exception as err:
        raise ClassNotFound(f'error when loading custom formatter: {err}')


def get_formatter_for_filename(fn, **options):
    """
    Return a :class:`.Formatter` subclass instance that has a filename pattern
    matching `fn`. The formatter is given the `options` at its instantiation.

    Will raise :exc:`pygments.util.ClassNotFound` if no formatter for that filename
    is found.
    """
    fn = basename(fn)
    for modname, name, _, filenames, _ in FORMATTERS.values():
        for filename in filenames:
            if _fn_matches(fn, filename):
                if name not in _formatter_cache:
                    _load_formatters(modname)
                return _formatter_cache[name](**options)
    for _name, cls in find_plugin_formatters():
        for filename in cls.filenames:
            if _fn_matches(fn, filename):
                return cls(**options)
    raise ClassNotFound(f"no formatter found for file name {fn!r}")


class _automodule(types.ModuleType):
    """Automatically import formatters."""

    def __getattr__(self, name):
        info = FORMATTERS.get(name)
        if info:
            _load_formatters(info[0])
            cls = _formatter_cache[info[1]]
            setattr(self, name, cls)
            return cls
        raise AttributeError(name)


oldmod = sys.modules[__name__]
newmod = _automodule(__name__)
newmod.__dict__.update(oldmod.__dict__)
sys.modules[__name__] = newmod
del newmod.newmod, newmod.oldmod, newmod.sys, newmod.types

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\requests\api.py ===
"""
requests.api
~~~~~~~~~~~~

This module implements the Requests API.

:copyright: (c) 2012 by Kenneth Reitz.
:license: Apache2, see LICENSE for more details.
"""

from . import sessions


def request(method, url, **kwargs):
    """Constructs and sends a :class:`Request <Request>`.

    :param method: method for the new :class:`Request` object: ``GET``, ``OPTIONS``, ``HEAD``, ``POST``, ``PUT``, ``PATCH``, or ``DELETE``.
    :param url: URL for the new :class:`Request` object.
    :param params: (optional) Dictionary, list of tuples or bytes to send
        in the query string for the :class:`Request`.
    :param data: (optional) Dictionary, list of tuples, bytes, or file-like
        object to send in the body of the :class:`Request`.
    :param json: (optional) A JSON serializable Python object to send in the body of the :class:`Request`.
    :param headers: (optional) Dictionary of HTTP Headers to send with the :class:`Request`.
    :param cookies: (optional) Dict or CookieJar object to send with the :class:`Request`.
    :param files: (optional) Dictionary of ``'name': file-like-objects`` (or ``{'name': file-tuple}``) for multipart encoding upload.
        ``file-tuple`` can be a 2-tuple ``('filename', fileobj)``, 3-tuple ``('filename', fileobj, 'content_type')``
        or a 4-tuple ``('filename', fileobj, 'content_type', custom_headers)``, where ``'content_type'`` is a string
        defining the content type of the given file and ``custom_headers`` a dict-like object containing additional headers
        to add for the file.
    :param auth: (optional) Auth tuple to enable Basic/Digest/Custom HTTP Auth.
    :param timeout: (optional) How many seconds to wait for the server to send data
        before giving up, as a float, or a :ref:`(connect timeout, read
        timeout) <timeouts>` tuple.
    :type timeout: float or tuple
    :param allow_redirects: (optional) Boolean. Enable/disable GET/OPTIONS/POST/PUT/PATCH/DELETE/HEAD redirection. Defaults to ``True``.
    :type allow_redirects: bool
    :param proxies: (optional) Dictionary mapping protocol to the URL of the proxy.
    :param verify: (optional) Either a boolean, in which case it controls whether we verify
            the server's TLS certificate, or a string, in which case it must be a path
            to a CA bundle to use. Defaults to ``True``.
    :param stream: (optional) if ``False``, the response content will be immediately downloaded.
    :param cert: (optional) if String, path to ssl client cert file (.pem). If Tuple, ('cert', 'key') pair.
    :return: :class:`Response <Response>` object
    :rtype: requests.Response

    Usage::

      >>> import requests
      >>> req = requests.request('GET', 'https://httpbin.org/get')
      >>> req
      <Response [200]>
    """

    # By using the 'with' statement we are sure the session is closed, thus we
    # avoid leaving sockets open which can trigger a ResourceWarning in some
    # cases, and look like a memory leak in others.
    with sessions.Session() as session:
        return session.request(method=method, url=url, **kwargs)


def get(url, params=None, **kwargs):
    r"""Sends a GET request.

    :param url: URL for the new :class:`Request` object.
    :param params: (optional) Dictionary, list of tuples or bytes to send
        in the query string for the :class:`Request`.
    :param \*\*kwargs: Optional arguments that ``request`` takes.
    :return: :class:`Response <Response>` object
    :rtype: requests.Response
    """

    return request("get", url, params=params, **kwargs)


def options(url, **kwargs):
    r"""Sends an OPTIONS request.

    :param url: URL for the new :class:`Request` object.
    :param \*\*kwargs: Optional arguments that ``request`` takes.
    :return: :class:`Response <Response>` object
    :rtype: requests.Response
    """

    return request("options", url, **kwargs)


def head(url, **kwargs):
    r"""Sends a HEAD request.

    :param url: URL for the new :class:`Request` object.
    :param \*\*kwargs: Optional arguments that ``request`` takes. If
        `allow_redirects` is not provided, it will be set to `False` (as
        opposed to the default :meth:`request` behavior).
    :return: :class:`Response <Response>` object
    :rtype: requests.Response
    """

    kwargs.setdefault("allow_redirects", False)
    return request("head", url, **kwargs)


def post(url, data=None, json=None, **kwargs):
    r"""Sends a POST request.

    :param url: URL for the new :class:`Request` object.
    :param data: (optional) Dictionary, list of tuples, bytes, or file-like
        object to send in the body of the :class:`Request`.
    :param json: (optional) A JSON serializable Python object to send in the body of the :class:`Request`.
    :param \*\*kwargs: Optional arguments that ``request`` takes.
    :return: :class:`Response <Response>` object
    :rtype: requests.Response
    """

    return request("post", url, data=data, json=json, **kwargs)


def put(url, data=None, **kwargs):
    r"""Sends a PUT request.

    :param url: URL for the new :class:`Request` object.
    :param data: (optional) Dictionary, list of tuples, bytes, or file-like
        object to send in the body of the :class:`Request`.
    :param json: (optional) A JSON serializable Python object to send in the body of the :class:`Request`.
    :param \*\*kwargs: Optional arguments that ``request`` takes.
    :return: :class:`Response <Response>` object
    :rtype: requests.Response
    """

    return request("put", url, data=data, **kwargs)


def patch(url, data=None, **kwargs):
    r"""Sends a PATCH request.

    :param url: URL for the new :class:`Request` object.
    :param data: (optional) Dictionary, list of tuples, bytes, or file-like
        object to send in the body of the :class:`Request`.
    :param json: (optional) A JSON serializable Python object to send in the body of the :class:`Request`.
    :param \*\*kwargs: Optional arguments that ``request`` takes.
    :return: :class:`Response <Response>` object
    :rtype: requests.Response
    """

    return request("patch", url, data=data, **kwargs)


def delete(url, **kwargs):
    r"""Sends a DELETE request.

    :param url: URL for the new :class:`Request` object.
    :param \*\*kwargs: Optional arguments that ``request`` takes.
    :return: :class:`Response <Response>` object
    :rtype: requests.Response
    """

    return request("delete", url, **kwargs)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pyreadline3\clipboard\win32_clipboard.py ===
# -*- coding: utf-8 -*-
# *****************************************************************************
#     Copyright (C) 2003-2006 Jack Trainor.
#     Copyright (C) 2006-2020 Jorgen Stenarson. <jorgen.stenarson@bostream.nu>
#     Copyright (C) 2020 Bassem Girgis. <brgirgis@gmail.com>
#
#  Distributed under the terms of the BSD License.  The full license is in
#  the file COPYING, distributed as part of this software.
# *****************************************************************************
###################################
#
# Based on recipe posted to ctypes-users
# see archive
# http://aspn.activestate.com/ASPN/Mail/Message/ctypes-users/1771866
#
#

##########################################################################
#
# The Python win32clipboard lib functions work well enough ... except that they
# can only cut and paste items from within one application, not across
# applications or processes.
#
# I've written a number of Python text filters I like to run on the contents of
# the clipboard so I need to call the Windows clipboard API with global memory
# for my filters to work properly.
#
# Here's some sample code solving this problem using ctypes.
#
# This is my first work with ctypes.  It's powerful stuff, but passing
# arguments in and out of functions is tricky.  More sample code would have
# been helpful, hence this contribution.
#
##########################################################################

import ctypes
import ctypes.wintypes as wintypes
from ctypes import (
    addressof,
    c_buffer,
    c_char_p,
    c_int,
    c_size_t,
    c_void_p,
    c_wchar_p,
    cast,
    create_unicode_buffer,
    sizeof,
    windll,
    wstring_at,
)
from typing import Union

from pyreadline3.keysyms.winconstants import CF_UNICODETEXT, GHND
from pyreadline3.unicode_helper import ensure_unicode

OpenClipboard = windll.user32.OpenClipboard
OpenClipboard.argtypes = [wintypes.HWND]
OpenClipboard.restype = wintypes.BOOL

EmptyClipboard = windll.user32.EmptyClipboard

GetClipboardData = windll.user32.GetClipboardData
GetClipboardData.argtypes = [wintypes.UINT]
GetClipboardData.restype = wintypes.HANDLE

GetClipboardFormatName = windll.user32.GetClipboardFormatNameA
GetClipboardFormatName.argtypes = [wintypes.UINT, c_char_p, c_int]

SetClipboardData = windll.user32.SetClipboardData
SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
SetClipboardData.restype = wintypes.HANDLE

EnumClipboardFormats = windll.user32.EnumClipboardFormats
EnumClipboardFormats.argtypes = [c_int]

CloseClipboard = windll.user32.CloseClipboard
CloseClipboard.argtypes = []


GlobalAlloc = windll.kernel32.GlobalAlloc
GlobalAlloc.argtypes = [wintypes.UINT, c_size_t]
GlobalAlloc.restype = wintypes.HGLOBAL
GlobalLock = windll.kernel32.GlobalLock
GlobalLock.argtypes = [wintypes.HGLOBAL]
GlobalLock.restype = c_void_p
GlobalUnlock = windll.kernel32.GlobalUnlock
GlobalUnlock.argtypes = [c_int]

_strncpy = ctypes.windll.kernel32.lstrcpynW
_strncpy.restype = c_wchar_p
_strncpy.argtypes = [c_wchar_p, c_wchar_p, c_size_t]


def _enum() -> None:
    OpenClipboard(0)

    q = EnumClipboardFormats(0)
    while q:
        q = EnumClipboardFormats(q)

    CloseClipboard()


def _get_format_name(format_str: str) -> bytes:
    buffer = c_buffer(100)
    bufferSize = sizeof(buffer)

    OpenClipboard(0)

    GetClipboardFormatName(format_str, buffer, bufferSize)

    CloseClipboard()

    return buffer.value


def get_clipboard_text() -> str:
    text = ""

    if OpenClipboard(0):
        h_clip_mem = GetClipboardData(CF_UNICODETEXT)

        if h_clip_mem:
            text = wstring_at(GlobalLock(h_clip_mem))
            GlobalUnlock(h_clip_mem)

        CloseClipboard()

    return text


def set_clipboard_text(text: Union[str, bytes]) -> None:
    buffer = create_unicode_buffer(ensure_unicode(text))
    buffer_size = sizeof(buffer)

    h_global_mem = GlobalAlloc(GHND, c_size_t(buffer_size))
    GlobalLock.restype = c_void_p
    lp_global_mem = GlobalLock(h_global_mem)

    _strncpy(
        cast(lp_global_mem, c_wchar_p),
        cast(addressof(buffer), c_wchar_p),
        c_size_t(buffer_size),
    )

    GlobalUnlock(c_int(h_global_mem))

    if OpenClipboard(0):
        EmptyClipboard()
        SetClipboardData(CF_UNICODETEXT, h_global_mem)
        CloseClipboard()


if __name__ == "__main__":
    txt = get_clipboard_text()  # display last text clipped
    print(txt)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\requests\api.py ===
"""
requests.api
~~~~~~~~~~~~

This module implements the Requests API.

:copyright: (c) 2012 by Kenneth Reitz.
:license: Apache2, see LICENSE for more details.
"""

from . import sessions


def request(method, url, **kwargs):
    """Constructs and sends a :class:`Request <Request>`.

    :param method: method for the new :class:`Request` object: ``GET``, ``OPTIONS``, ``HEAD``, ``POST``, ``PUT``, ``PATCH``, or ``DELETE``.
    :param url: URL for the new :class:`Request` object.
    :param params: (optional) Dictionary, list of tuples or bytes to send
        in the query string for the :class:`Request`.
    :param data: (optional) Dictionary, list of tuples, bytes, or file-like
        object to send in the body of the :class:`Request`.
    :param json: (optional) A JSON serializable Python object to send in the body of the :class:`Request`.
    :param headers: (optional) Dictionary of HTTP Headers to send with the :class:`Request`.
    :param cookies: (optional) Dict or CookieJar object to send with the :class:`Request`.
    :param files: (optional) Dictionary of ``'name': file-like-objects`` (or ``{'name': file-tuple}``) for multipart encoding upload.
        ``file-tuple`` can be a 2-tuple ``('filename', fileobj)``, 3-tuple ``('filename', fileobj, 'content_type')``
        or a 4-tuple ``('filename', fileobj, 'content_type', custom_headers)``, where ``'content_type'`` is a string
        defining the content type of the given file and ``custom_headers`` a dict-like object containing additional headers
        to add for the file.
    :param auth: (optional) Auth tuple to enable Basic/Digest/Custom HTTP Auth.
    :param timeout: (optional) How many seconds to wait for the server to send data
        before giving up, as a float, or a :ref:`(connect timeout, read
        timeout) <timeouts>` tuple.
    :type timeout: float or tuple
    :param allow_redirects: (optional) Boolean. Enable/disable GET/OPTIONS/POST/PUT/PATCH/DELETE/HEAD redirection. Defaults to ``True``.
    :type allow_redirects: bool
    :param proxies: (optional) Dictionary mapping protocol to the URL of the proxy.
    :param verify: (optional) Either a boolean, in which case it controls whether we verify
            the server's TLS certificate, or a string, in which case it must be a path
            to a CA bundle to use. Defaults to ``True``.
    :param stream: (optional) if ``False``, the response content will be immediately downloaded.
    :param cert: (optional) if String, path to ssl client cert file (.pem). If Tuple, ('cert', 'key') pair.
    :return: :class:`Response <Response>` object
    :rtype: requests.Response

    Usage::

      >>> import requests
      >>> req = requests.request('GET', 'https://httpbin.org/get')
      >>> req
      <Response [200]>
    """

    # By using the 'with' statement we are sure the session is closed, thus we
    # avoid leaving sockets open which can trigger a ResourceWarning in some
    # cases, and look like a memory leak in others.
    with sessions.Session() as session:
        return session.request(method=method, url=url, **kwargs)


def get(url, params=None, **kwargs):
    r"""Sends a GET request.

    :param url: URL for the new :class:`Request` object.
    :param params: (optional) Dictionary, list of tuples or bytes to send
        in the query string for the :class:`Request`.
    :param \*\*kwargs: Optional arguments that ``request`` takes.
    :return: :class:`Response <Response>` object
    :rtype: requests.Response
    """

    return request("get", url, params=params, **kwargs)


def options(url, **kwargs):
    r"""Sends an OPTIONS request.

    :param url: URL for the new :class:`Request` object.
    :param \*\*kwargs: Optional arguments that ``request`` takes.
    :return: :class:`Response <Response>` object
    :rtype: requests.Response
    """

    return request("options", url, **kwargs)


def head(url, **kwargs):
    r"""Sends a HEAD request.

    :param url: URL for the new :class:`Request` object.
    :param \*\*kwargs: Optional arguments that ``request`` takes. If
        `allow_redirects` is not provided, it will be set to `False` (as
        opposed to the default :meth:`request` behavior).
    :return: :class:`Response <Response>` object
    :rtype: requests.Response
    """

    kwargs.setdefault("allow_redirects", False)
    return request("head", url, **kwargs)


def post(url, data=None, json=None, **kwargs):
    r"""Sends a POST request.

    :param url: URL for the new :class:`Request` object.
    :param data: (optional) Dictionary, list of tuples, bytes, or file-like
        object to send in the body of the :class:`Request`.
    :param json: (optional) A JSON serializable Python object to send in the body of the :class:`Request`.
    :param \*\*kwargs: Optional arguments that ``request`` takes.
    :return: :class:`Response <Response>` object
    :rtype: requests.Response
    """

    return request("post", url, data=data, json=json, **kwargs)


def put(url, data=None, **kwargs):
    r"""Sends a PUT request.

    :param url: URL for the new :class:`Request` object.
    :param data: (optional) Dictionary, list of tuples, bytes, or file-like
        object to send in the body of the :class:`Request`.
    :param json: (optional) A JSON serializable Python object to send in the body of the :class:`Request`.
    :param \*\*kwargs: Optional arguments that ``request`` takes.
    :return: :class:`Response <Response>` object
    :rtype: requests.Response
    """

    return request("put", url, data=data, **kwargs)


def patch(url, data=None, **kwargs):
    r"""Sends a PATCH request.

    :param url: URL for the new :class:`Request` object.
    :param data: (optional) Dictionary, list of tuples, bytes, or file-like
        object to send in the body of the :class:`Request`.
    :param json: (optional) A JSON serializable Python object to send in the body of the :class:`Request`.
    :param \*\*kwargs: Optional arguments that ``request`` takes.
    :return: :class:`Response <Response>` object
    :rtype: requests.Response
    """

    return request("patch", url, data=data, **kwargs)


def delete(url, **kwargs):
    r"""Sends a DELETE request.

    :param url: URL for the new :class:`Request` object.
    :param \*\*kwargs: Optional arguments that ``request`` takes.
    :return: :class:`Response <Response>` object
    :rtype: requests.Response
    """

    return request("delete", url, **kwargs)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\dialects\sqlite\pysqlcipher.py ===
# dialects/sqlite/pysqlcipher.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors


"""
.. dialect:: sqlite+pysqlcipher
    :name: pysqlcipher
    :dbapi: sqlcipher 3 or pysqlcipher
    :connectstring: sqlite+pysqlcipher://:passphrase@/file_path[?kdf_iter=<iter>]

    Dialect for support of DBAPIs that make use of the
    `SQLCipher <https://www.zetetic.net/sqlcipher>`_ backend.


Driver
------

Current dialect selection logic is:

* If the :paramref:`_sa.create_engine.module` parameter supplies a DBAPI module,
  that module is used.
* Otherwise for Python 3, choose https://pypi.org/project/sqlcipher3/
* If not available, fall back to https://pypi.org/project/pysqlcipher3/
* For Python 2, https://pypi.org/project/pysqlcipher/ is used.

.. warning:: The ``pysqlcipher3`` and ``pysqlcipher`` DBAPI drivers are no
   longer maintained; the ``sqlcipher3`` driver as of this writing appears
   to be current.  For future compatibility, any pysqlcipher-compatible DBAPI
   may be used as follows::

        import sqlcipher_compatible_driver

        from sqlalchemy import create_engine

        e = create_engine(
            "sqlite+pysqlcipher://:password@/dbname.db",
            module=sqlcipher_compatible_driver,
        )

These drivers make use of the SQLCipher engine. This system essentially
introduces new PRAGMA commands to SQLite which allows the setting of a
passphrase and other encryption parameters, allowing the database file to be
encrypted.


Connect Strings
---------------

The format of the connect string is in every way the same as that
of the :mod:`~sqlalchemy.dialects.sqlite.pysqlite` driver, except that the
"password" field is now accepted, which should contain a passphrase::

    e = create_engine("sqlite+pysqlcipher://:testing@/foo.db")

For an absolute file path, two leading slashes should be used for the
database name::

    e = create_engine("sqlite+pysqlcipher://:testing@//path/to/foo.db")

A selection of additional encryption-related pragmas supported by SQLCipher
as documented at https://www.zetetic.net/sqlcipher/sqlcipher-api/ can be passed
in the query string, and will result in that PRAGMA being called for each
new connection.  Currently, ``cipher``, ``kdf_iter``
``cipher_page_size`` and ``cipher_use_hmac`` are supported::

    e = create_engine(
        "sqlite+pysqlcipher://:testing@/foo.db?cipher=aes-256-cfb&kdf_iter=64000"
    )

.. warning:: Previous versions of sqlalchemy did not take into consideration
   the encryption-related pragmas passed in the url string, that were silently
   ignored. This may cause errors when opening files saved by a
   previous sqlalchemy version if the encryption options do not match.


Pooling Behavior
----------------

The driver makes a change to the default pool behavior of pysqlite
as described in :ref:`pysqlite_threading_pooling`.   The pysqlcipher driver
has been observed to be significantly slower on connection than the
pysqlite driver, most likely due to the encryption overhead, so the
dialect here defaults to using the :class:`.SingletonThreadPool`
implementation,
instead of the :class:`.NullPool` pool used by pysqlite.  As always, the pool
implementation is entirely configurable using the
:paramref:`_sa.create_engine.poolclass` parameter; the :class:`.
StaticPool` may
be more feasible for single-threaded use, or :class:`.NullPool` may be used
to prevent unencrypted connections from being held open for long periods of
time, at the expense of slower startup time for new connections.


"""  # noqa

from .pysqlite import SQLiteDialect_pysqlite
from ... import pool


class SQLiteDialect_pysqlcipher(SQLiteDialect_pysqlite):
    driver = "pysqlcipher"
    supports_statement_cache = True

    pragmas = ("kdf_iter", "cipher", "cipher_page_size", "cipher_use_hmac")

    @classmethod
    def import_dbapi(cls):
        try:
            import sqlcipher3 as sqlcipher
        except ImportError:
            pass
        else:
            return sqlcipher

        from pysqlcipher3 import dbapi2 as sqlcipher

        return sqlcipher

    @classmethod
    def get_pool_class(cls, url):
        return pool.SingletonThreadPool

    def on_connect_url(self, url):
        super_on_connect = super().on_connect_url(url)

        # pull the info we need from the URL early.  Even though URL
        # is immutable, we don't want any in-place changes to the URL
        # to affect things
        passphrase = url.password or ""
        url_query = dict(url.query)

        def on_connect(conn):
            cursor = conn.cursor()
            cursor.execute('pragma key="%s"' % passphrase)
            for prag in self.pragmas:
                value = url_query.get(prag, None)
                if value is not None:
                    cursor.execute('pragma %s="%s"' % (prag, value))
            cursor.close()

            if super_on_connect:
                super_on_connect(conn)

        return on_connect

    def create_connect_args(self, url):
        plain_url = url._replace(password=None)
        plain_url = plain_url.difference_update_query(self.pragmas)
        return super().create_connect_args(plain_url)


dialect = SQLiteDialect_pysqlcipher

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\internals\api.py ===
"""
This is a pseudo-public API for downstream libraries.  We ask that downstream
authors

1) Try to avoid using internals directly altogether, and failing that,
2) Use only functions exposed here (or in core.internals)

"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pandas._libs.internals import BlockPlacement

from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.dtypes import (
    DatetimeTZDtype,
    PeriodDtype,
)

from pandas.core.arrays import DatetimeArray
from pandas.core.construction import extract_array
from pandas.core.internals.blocks import (
    check_ndim,
    ensure_block_shape,
    extract_pandas_array,
    get_block_type,
    maybe_coerce_values,
)

if TYPE_CHECKING:
    from pandas._typing import Dtype

    from pandas.core.internals.blocks import Block


def make_block(
    values, placement, klass=None, ndim=None, dtype: Dtype | None = None
) -> Block:
    """
    This is a pseudo-public analogue to blocks.new_block.

    We ask that downstream libraries use this rather than any fully-internal
    APIs, including but not limited to:

    - core.internals.blocks.make_block
    - Block.make_block
    - Block.make_block_same_class
    - Block.__init__
    """
    if dtype is not None:
        dtype = pandas_dtype(dtype)

    values, dtype = extract_pandas_array(values, dtype, ndim)

    from pandas.core.internals.blocks import (
        DatetimeTZBlock,
        ExtensionBlock,
    )

    if klass is ExtensionBlock and isinstance(values.dtype, PeriodDtype):
        # GH-44681 changed PeriodArray to be stored in the 2D
        # NDArrayBackedExtensionBlock instead of ExtensionBlock
        # -> still allow ExtensionBlock to be passed in this case for back compat
        klass = None

    if klass is None:
        dtype = dtype or values.dtype
        klass = get_block_type(dtype)

    elif klass is DatetimeTZBlock and not isinstance(values.dtype, DatetimeTZDtype):
        # pyarrow calls get here
        values = DatetimeArray._simple_new(
            # error: Argument "dtype" to "_simple_new" of "DatetimeArray" has
            # incompatible type "Union[ExtensionDtype, dtype[Any], None]";
            # expected "Union[dtype[datetime64], DatetimeTZDtype]"
            values,
            dtype=dtype,  # type: ignore[arg-type]
        )

    if not isinstance(placement, BlockPlacement):
        placement = BlockPlacement(placement)

    ndim = maybe_infer_ndim(values, placement, ndim)
    if isinstance(values.dtype, (PeriodDtype, DatetimeTZDtype)):
        # GH#41168 ensure we can pass 1D dt64tz values
        # More generally, any EA dtype that isn't is_1d_only_ea_dtype
        values = extract_array(values, extract_numpy=True)
        values = ensure_block_shape(values, ndim)

    check_ndim(values, placement, ndim)
    values = maybe_coerce_values(values)
    return klass(values, ndim=ndim, placement=placement)


def maybe_infer_ndim(values, placement: BlockPlacement, ndim: int | None) -> int:
    """
    If `ndim` is not provided, infer it from placement and values.
    """
    if ndim is None:
        # GH#38134 Block constructor now assumes ndim is not None
        if not isinstance(values.dtype, np.dtype):
            if len(placement) != 1:
                ndim = 1
            else:
                ndim = 2
        else:
            ndim = values.ndim
    return ndim


def __getattr__(name: str):
    # GH#55139
    import warnings

    if name in [
        "Block",
        "ExtensionBlock",
        "DatetimeTZBlock",
        "create_block_manager_from_blocks",
    ]:
        # GH#33892
        warnings.warn(
            f"{name} is deprecated and will be removed in a future version. "
            "Use public APIs instead.",
            DeprecationWarning,
            # https://github.com/pandas-dev/pandas/pull/55139#pullrequestreview-1720690758
            # on hard-coding stacklevel
            stacklevel=2,
        )

        if name == "create_block_manager_from_blocks":
            from pandas.core.internals.managers import create_block_manager_from_blocks

            return create_block_manager_from_blocks

        elif name == "Block":
            from pandas.core.internals.blocks import Block

            return Block

        elif name == "DatetimeTZBlock":
            from pandas.core.internals.blocks import DatetimeTZBlock

            return DatetimeTZBlock

        elif name == "ExtensionBlock":
            from pandas.core.internals.blocks import ExtensionBlock

            return ExtensionBlock

    raise AttributeError(
        f"module 'pandas.core.internals.api' has no attribute '{name}'"
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\codegen\cnodes.py ===
"""
AST nodes specific to the C family of languages
"""

from sympy.codegen.ast import (
    Attribute, Declaration, Node, String, Token, Type, none,
    FunctionCall, CodeBlock
    )
from sympy.core.basic import Basic
from sympy.core.containers import Tuple
from sympy.core.sympify import sympify

void = Type('void')

restrict = Attribute('restrict')  # guarantees no pointer aliasing
volatile = Attribute('volatile')
static = Attribute('static')


def alignof(arg):
    """ Generate of FunctionCall instance for calling 'alignof' """
    return FunctionCall('alignof', [String(arg) if isinstance(arg, str) else arg])


def sizeof(arg):
    """ Generate of FunctionCall instance for calling 'sizeof'

    Examples
    ========

    >>> from sympy.codegen.ast import real
    >>> from sympy.codegen.cnodes import sizeof
    >>> from sympy import ccode
    >>> ccode(sizeof(real))
    'sizeof(double)'
    """
    return FunctionCall('sizeof', [String(arg) if isinstance(arg, str) else arg])


class CommaOperator(Basic):
    """ Represents the comma operator in C """
    def __new__(cls, *args):
        return Basic.__new__(cls, *[sympify(arg) for arg in args])


class Label(Node):
    """ Label for use with e.g. goto statement.

    Examples
    ========

    >>> from sympy import ccode, Symbol
    >>> from sympy.codegen.cnodes import Label, PreIncrement
    >>> print(ccode(Label('foo')))
    foo:
    >>> print(ccode(Label('bar', [PreIncrement(Symbol('a'))])))
    bar:
    ++(a);

    """
    __slots__ = _fields = ('name', 'body')
    defaults = {'body': none}
    _construct_name = String

    @classmethod
    def _construct_body(cls, itr):
        if isinstance(itr, CodeBlock):
            return itr
        else:
            return CodeBlock(*itr)


class goto(Token):
    """ Represents goto in C """
    __slots__ = _fields = ('label',)
    _construct_label = Label


class PreDecrement(Basic):
    """ Represents the pre-decrement operator

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.codegen.cnodes import PreDecrement
    >>> from sympy import ccode
    >>> ccode(PreDecrement(x))
    '--(x)'

    """
    nargs = 1


class PostDecrement(Basic):
    """ Represents the post-decrement operator

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.codegen.cnodes import PostDecrement
    >>> from sympy import ccode
    >>> ccode(PostDecrement(x))
    '(x)--'

    """
    nargs = 1


class PreIncrement(Basic):
    """ Represents the pre-increment operator

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.codegen.cnodes import PreIncrement
    >>> from sympy import ccode
    >>> ccode(PreIncrement(x))
    '++(x)'

    """
    nargs = 1


class PostIncrement(Basic):
    """ Represents the post-increment operator

    Examples
    ========

    >>> from sympy.abc import x
    >>> from sympy.codegen.cnodes import PostIncrement
    >>> from sympy import ccode
    >>> ccode(PostIncrement(x))
    '(x)++'

    """
    nargs = 1


class struct(Node):
    """ Represents a struct in C """
    __slots__ = _fields = ('name', 'declarations')
    defaults = {'name': none}
    _construct_name = String

    @classmethod
    def _construct_declarations(cls, args):
        return Tuple(*[Declaration(arg) for arg in args])


class union(struct):
    """ Represents a union in C """
    __slots__ = ()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\normalforms.py ===
'''Functions returning normal forms of matrices'''

from sympy.polys.domains.integerring import ZZ
from sympy.polys.polytools import Poly
from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.normalforms import (
        smith_normal_form as _snf,
        is_smith_normal_form as _is_snf,
        smith_normal_decomp as _snd,
        invariant_factors as _invf,
        hermite_normal_form as _hnf,
    )


def _to_domain(m, domain=None):
    """Convert Matrix to DomainMatrix"""
    # XXX: deprecated support for RawMatrix:
    ring = getattr(m, "ring", None)
    m = m.applyfunc(lambda e: e.as_expr() if isinstance(e, Poly) else e)

    dM = DomainMatrix.from_Matrix(m)

    domain = domain or ring
    if domain is not None:
        dM = dM.convert_to(domain)
    return dM


def smith_normal_form(m, domain=None):
    '''
    Return the Smith Normal Form of a matrix `m` over the ring `domain`.
    This will only work if the ring is a principal ideal domain.

    Examples
    ========

    >>> from sympy import Matrix, ZZ
    >>> from sympy.matrices.normalforms import smith_normal_form
    >>> m = Matrix([[12, 6, 4], [3, 9, 6], [2, 16, 14]])
    >>> print(smith_normal_form(m, domain=ZZ))
    Matrix([[1, 0, 0], [0, 10, 0], [0, 0, 30]])

    '''
    dM = _to_domain(m, domain)
    return _snf(dM).to_Matrix()


def is_smith_normal_form(m, domain=None):
    '''
    Checks that the matrix is in Smith Normal Form
    '''
    dM = _to_domain(m, domain)
    return _is_snf(dM)


def smith_normal_decomp(m, domain=None):
    '''
    Return the Smith Normal Decomposition of a matrix `m` over the ring
    `domain`. This will only work if the ring is a principal ideal domain.

    Examples
    ========

    >>> from sympy import Matrix, ZZ
    >>> from sympy.matrices.normalforms import smith_normal_decomp
    >>> m = Matrix([[12, 6, 4], [3, 9, 6], [2, 16, 14]])
    >>> a, s, t = smith_normal_decomp(m, domain=ZZ)
    >>> assert a == s * m * t
    '''
    dM = _to_domain(m, domain)
    a, s, t = _snd(dM)
    return a.to_Matrix(), s.to_Matrix(), t.to_Matrix()


def invariant_factors(m, domain=None):
    '''
    Return the tuple of abelian invariants for a matrix `m`
    (as in the Smith-Normal form)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Smith_normal_form#Algorithm
    .. [2] https://web.archive.org/web/20200331143852/https://sierra.nmsu.edu/morandi/notes/SmithNormalForm.pdf

    '''
    dM = _to_domain(m, domain)
    factors = _invf(dM)
    factors = tuple(dM.domain.to_sympy(f) for f in factors)
    # XXX: deprecated.
    if hasattr(m, "ring"):
        if m.ring.is_PolynomialRing:
            K = m.ring
            to_poly = lambda f: Poly(f, K.symbols, domain=K.domain)
            factors = tuple(to_poly(f) for f in factors)
    return factors


def hermite_normal_form(A, *, D=None, check_rank=False):
    r"""
    Compute the Hermite Normal Form of a Matrix *A* of integers.

    Examples
    ========

    >>> from sympy import Matrix
    >>> from sympy.matrices.normalforms import hermite_normal_form
    >>> m = Matrix([[12, 6, 4], [3, 9, 6], [2, 16, 14]])
    >>> print(hermite_normal_form(m))
    Matrix([[10, 0, 2], [0, 15, 3], [0, 0, 2]])

    Parameters
    ==========

    A : $m \times n$ ``Matrix`` of integers.

    D : int, optional
        Let $W$ be the HNF of *A*. If known in advance, a positive integer *D*
        being any multiple of $\det(W)$ may be provided. In this case, if *A*
        also has rank $m$, then we may use an alternative algorithm that works
        mod *D* in order to prevent coefficient explosion.

    check_rank : boolean, optional (default=False)
        The basic assumption is that, if you pass a value for *D*, then
        you already believe that *A* has rank $m$, so we do not waste time
        checking it for you. If you do want this to be checked (and the
        ordinary, non-modulo *D* algorithm to be used if the check fails), then
        set *check_rank* to ``True``.

    Returns
    =======

    ``Matrix``
        The HNF of matrix *A*.

    Raises
    ======

    DMDomainError
        If the domain of the matrix is not :ref:`ZZ`.

    DMShapeError
        If the mod *D* algorithm is used but the matrix has more rows than
        columns.

    References
    ==========

    .. [1] Cohen, H. *A Course in Computational Algebraic Number Theory.*
       (See Algorithms 2.4.5 and 2.4.8.)

    """
    # Accept any of Python int, SymPy Integer, and ZZ itself:
    if D is not None and not ZZ.of_type(D):
        D = ZZ(int(D))
    return _hnf(A._rep, D=D, check_rank=check_rank).to_Matrix()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\units\systems\length_weight_time.py ===
from sympy.core.singleton import S

from sympy.core.numbers import pi

from sympy.physics.units import DimensionSystem, hertz, kilogram
from sympy.physics.units.definitions import (
    G, Hz, J, N, Pa, W, c, g, kg, m, s, meter, gram, second, newton,
    joule, watt, pascal)
from sympy.physics.units.definitions.dimension_definitions import (
    acceleration, action, energy, force, frequency, momentum,
    power, pressure, velocity, length, mass, time)
from sympy.physics.units.prefixes import PREFIXES, prefix_unit
from sympy.physics.units.prefixes import (
    kibi, mebi, gibi, tebi, pebi, exbi
)
from sympy.physics.units.definitions import (
    cd, K, coulomb, volt, ohm, siemens, farad, henry, tesla, weber, dioptre,
    lux, katal, gray, becquerel, inch, hectare, liter, julian_year,
    gravitational_constant, speed_of_light, elementary_charge, planck, hbar,
    electronvolt, avogadro_number, avogadro_constant, boltzmann_constant,
    stefan_boltzmann_constant, atomic_mass_constant, molar_gas_constant,
    faraday_constant, josephson_constant, von_klitzing_constant,
    acceleration_due_to_gravity, magnetic_constant, vacuum_permittivity,
    vacuum_impedance, coulomb_constant, atmosphere, bar, pound, psi, mmHg,
    milli_mass_unit, quart, lightyear, astronomical_unit, planck_mass,
    planck_time, planck_temperature, planck_length, planck_charge,
    planck_area, planck_volume, planck_momentum, planck_energy, planck_force,
    planck_power, planck_density, planck_energy_density, planck_intensity,
    planck_angular_frequency, planck_pressure, planck_current, planck_voltage,
    planck_impedance, planck_acceleration, bit, byte, kibibyte, mebibyte,
    gibibyte, tebibyte, pebibyte, exbibyte, curie, rutherford, radian, degree,
    steradian, angular_mil, atomic_mass_unit, gee, kPa, ampere, u0, kelvin,
    mol, mole, candela, electric_constant, boltzmann, angstrom
)


dimsys_length_weight_time = DimensionSystem([
    # Dimensional dependencies for MKS base dimensions
    length,
    mass,
    time,
], dimensional_dependencies={
    # Dimensional dependencies for derived dimensions
    "velocity": {"length": 1, "time": -1},
    "acceleration": {"length": 1, "time": -2},
    "momentum": {"mass": 1, "length": 1, "time": -1},
    "force": {"mass": 1, "length": 1, "time": -2},
    "energy": {"mass": 1, "length": 2, "time": -2},
    "power": {"length": 2, "mass": 1, "time": -3},
    "pressure": {"mass": 1, "length": -1, "time": -2},
    "frequency": {"time": -1},
    "action": {"length": 2, "mass": 1, "time": -1},
    "area": {"length": 2},
    "volume": {"length": 3},
})


One = S.One


# Base units:
dimsys_length_weight_time.set_quantity_dimension(meter, length)
dimsys_length_weight_time.set_quantity_scale_factor(meter, One)

# gram; used to define its prefixed units
dimsys_length_weight_time.set_quantity_dimension(gram, mass)
dimsys_length_weight_time.set_quantity_scale_factor(gram, One)

dimsys_length_weight_time.set_quantity_dimension(second, time)
dimsys_length_weight_time.set_quantity_scale_factor(second, One)

# derived units

dimsys_length_weight_time.set_quantity_dimension(newton, force)
dimsys_length_weight_time.set_quantity_scale_factor(newton, kilogram*meter/second**2)

dimsys_length_weight_time.set_quantity_dimension(joule, energy)
dimsys_length_weight_time.set_quantity_scale_factor(joule, newton*meter)

dimsys_length_weight_time.set_quantity_dimension(watt, power)
dimsys_length_weight_time.set_quantity_scale_factor(watt, joule/second)

dimsys_length_weight_time.set_quantity_dimension(pascal, pressure)
dimsys_length_weight_time.set_quantity_scale_factor(pascal, newton/meter**2)

dimsys_length_weight_time.set_quantity_dimension(hertz, frequency)
dimsys_length_weight_time.set_quantity_scale_factor(hertz, One)

# Other derived units:

dimsys_length_weight_time.set_quantity_dimension(dioptre, 1 / length)
dimsys_length_weight_time.set_quantity_scale_factor(dioptre, 1/meter)

# Common volume and area units

dimsys_length_weight_time.set_quantity_dimension(hectare, length**2)
dimsys_length_weight_time.set_quantity_scale_factor(hectare, (meter**2)*(10000))

dimsys_length_weight_time.set_quantity_dimension(liter, length**3)
dimsys_length_weight_time.set_quantity_scale_factor(liter, meter**3/1000)


# Newton constant
# REF: NIST SP 959 (June 2019)

dimsys_length_weight_time.set_quantity_dimension(gravitational_constant, length ** 3 * mass ** -1 * time ** -2)
dimsys_length_weight_time.set_quantity_scale_factor(gravitational_constant, 6.67430e-11*m**3/(kg*s**2))

# speed of light

dimsys_length_weight_time.set_quantity_dimension(speed_of_light, velocity)
dimsys_length_weight_time.set_quantity_scale_factor(speed_of_light, 299792458*meter/second)


# Planck constant
# REF: NIST SP 959 (June 2019)

dimsys_length_weight_time.set_quantity_dimension(planck, action)
dimsys_length_weight_time.set_quantity_scale_factor(planck, 6.62607015e-34*joule*second)

# Reduced Planck constant
# REF: NIST SP 959 (June 2019)

dimsys_length_weight_time.set_quantity_dimension(hbar, action)
dimsys_length_weight_time.set_quantity_scale_factor(hbar, planck / (2 * pi))


__all__ = [
    'mmHg', 'atmosphere', 'newton', 'meter', 'vacuum_permittivity', 'pascal',
    'magnetic_constant', 'angular_mil', 'julian_year', 'weber', 'exbibyte',
    'liter', 'molar_gas_constant', 'faraday_constant', 'avogadro_constant',
    'planck_momentum', 'planck_density', 'gee', 'mol', 'bit', 'gray', 'kibi',
    'bar', 'curie', 'prefix_unit', 'PREFIXES', 'planck_time', 'gram',
    'candela', 'force', 'planck_intensity', 'energy', 'becquerel',
    'planck_acceleration', 'speed_of_light', 'dioptre', 'second', 'frequency',
    'Hz', 'power', 'lux', 'planck_current', 'momentum', 'tebibyte',
    'planck_power', 'degree', 'mebi', 'K', 'planck_volume',
    'quart', 'pressure', 'W', 'joule', 'boltzmann_constant', 'c', 'g',
    'planck_force', 'exbi', 's', 'watt', 'action', 'hbar', 'gibibyte',
    'DimensionSystem', 'cd', 'volt', 'planck_charge', 'angstrom',
    'dimsys_length_weight_time', 'pebi', 'vacuum_impedance', 'planck',
    'farad', 'gravitational_constant', 'u0', 'hertz', 'tesla', 'steradian',
    'josephson_constant', 'planck_area', 'stefan_boltzmann_constant',
    'astronomical_unit', 'J', 'N', 'planck_voltage', 'planck_energy',
    'atomic_mass_constant', 'rutherford', 'elementary_charge', 'Pa',
    'planck_mass', 'henry', 'planck_angular_frequency', 'ohm', 'pound',
    'planck_pressure', 'G', 'avogadro_number', 'psi', 'von_klitzing_constant',
    'planck_length', 'radian', 'mole', 'acceleration',
    'planck_energy_density', 'mebibyte', 'length',
    'acceleration_due_to_gravity', 'planck_temperature', 'tebi', 'inch',
    'electronvolt', 'coulomb_constant', 'kelvin', 'kPa', 'boltzmann',
    'milli_mass_unit', 'gibi', 'planck_impedance', 'electric_constant', 'kg',
    'coulomb', 'siemens', 'byte', 'atomic_mass_unit', 'm', 'kibibyte',
    'kilogram', 'lightyear', 'mass', 'time', 'pebibyte', 'velocity',
    'ampere', 'katal',
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\workbook\views.py ===
# Copyright (c) 2010-2024 openpyxl

from openpyxl.descriptors.serialisable import Serialisable
from openpyxl.descriptors import (
    Typed,
    Sequence,
    String,
    Float,
    Integer,
    Bool,
    NoneSet,
    Set,
)
from openpyxl.descriptors.excel import (
    ExtensionList,
    Guid,
)


class BookView(Serialisable):

    tagname = "workbookView"

    visibility = NoneSet(values=(['visible', 'hidden', 'veryHidden']))
    minimized = Bool(allow_none=True)
    showHorizontalScroll = Bool(allow_none=True)
    showVerticalScroll = Bool(allow_none=True)
    showSheetTabs = Bool(allow_none=True)
    xWindow = Integer(allow_none=True)
    yWindow = Integer(allow_none=True)
    windowWidth = Integer(allow_none=True)
    windowHeight = Integer(allow_none=True)
    tabRatio = Integer(allow_none=True)
    firstSheet = Integer(allow_none=True)
    activeTab = Integer(allow_none=True)
    autoFilterDateGrouping = Bool(allow_none=True)
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = ()

    def __init__(self,
                 visibility="visible",
                 minimized=False,
                 showHorizontalScroll=True,
                 showVerticalScroll=True,
                 showSheetTabs=True,
                 xWindow=None,
                 yWindow=None,
                 windowWidth=None,
                 windowHeight=None,
                 tabRatio=600,
                 firstSheet=0,
                 activeTab=0,
                 autoFilterDateGrouping=True,
                 extLst=None,
                ):
        self.visibility = visibility
        self.minimized = minimized
        self.showHorizontalScroll = showHorizontalScroll
        self.showVerticalScroll = showVerticalScroll
        self.showSheetTabs = showSheetTabs
        self.xWindow = xWindow
        self.yWindow = yWindow
        self.windowWidth = windowWidth
        self.windowHeight = windowHeight
        self.tabRatio = tabRatio
        self.firstSheet = firstSheet
        self.activeTab = activeTab
        self.autoFilterDateGrouping = autoFilterDateGrouping


class CustomWorkbookView(Serialisable):

    tagname = "customWorkbookView"

    name = String()
    guid = Guid()
    autoUpdate = Bool(allow_none=True)
    mergeInterval = Integer(allow_none=True)
    changesSavedWin = Bool(allow_none=True)
    onlySync = Bool(allow_none=True)
    personalView = Bool(allow_none=True)
    includePrintSettings = Bool(allow_none=True)
    includeHiddenRowCol = Bool(allow_none=True)
    maximized = Bool(allow_none=True)
    minimized = Bool(allow_none=True)
    showHorizontalScroll = Bool(allow_none=True)
    showVerticalScroll = Bool(allow_none=True)
    showSheetTabs = Bool(allow_none=True)
    xWindow = Integer(allow_none=True)
    yWindow = Integer(allow_none=True)
    windowWidth = Integer()
    windowHeight = Integer()
    tabRatio = Integer(allow_none=True)
    activeSheetId = Integer()
    showFormulaBar = Bool(allow_none=True)
    showStatusbar = Bool(allow_none=True)
    showComments = NoneSet(values=(['commNone', 'commIndicator',
                                'commIndAndComment']))
    showObjects = NoneSet(values=(['all', 'placeholders']))
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = ()

    def __init__(self,
                 name=None,
                 guid=None,
                 autoUpdate=None,
                 mergeInterval=None,
                 changesSavedWin=None,
                 onlySync=None,
                 personalView=None,
                 includePrintSettings=None,
                 includeHiddenRowCol=None,
                 maximized=None,
                 minimized=None,
                 showHorizontalScroll=None,
                 showVerticalScroll=None,
                 showSheetTabs=None,
                 xWindow=None,
                 yWindow=None,
                 windowWidth=None,
                 windowHeight=None,
                 tabRatio=None,
                 activeSheetId=None,
                 showFormulaBar=None,
                 showStatusbar=None,
                 showComments="commIndicator",
                 showObjects="all",
                 extLst=None,
                ):
        self.name = name
        self.guid = guid
        self.autoUpdate = autoUpdate
        self.mergeInterval = mergeInterval
        self.changesSavedWin = changesSavedWin
        self.onlySync = onlySync
        self.personalView = personalView
        self.includePrintSettings = includePrintSettings
        self.includeHiddenRowCol = includeHiddenRowCol
        self.maximized = maximized
        self.minimized = minimized
        self.showHorizontalScroll = showHorizontalScroll
        self.showVerticalScroll = showVerticalScroll
        self.showSheetTabs = showSheetTabs
        self.xWindow = xWindow
        self.yWindow = yWindow
        self.windowWidth = windowWidth
        self.windowHeight = windowHeight
        self.tabRatio = tabRatio
        self.activeSheetId = activeSheetId
        self.showFormulaBar = showFormulaBar
        self.showStatusbar = showStatusbar
        self.showComments = showComments
        self.showObjects = showObjects

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\openpyxl\worksheet\views.py ===
# Copyright (c) 2010-2024 openpyxl

from openpyxl.descriptors import (
    Bool,
    Integer,
    String,
    Set,
    Float,
    Typed,
    NoneSet,
    Sequence,
)
from openpyxl.descriptors.excel import ExtensionList
from openpyxl.descriptors.serialisable import Serialisable


class Pane(Serialisable):
    xSplit = Float(allow_none=True)
    ySplit = Float(allow_none=True)
    topLeftCell = String(allow_none=True)
    activePane = Set(values=("bottomRight", "topRight", "bottomLeft", "topLeft"))
    state = Set(values=("split", "frozen", "frozenSplit"))

    def __init__(self,
                 xSplit=None,
                 ySplit=None,
                 topLeftCell=None,
                 activePane="topLeft",
                 state="split"):
        self.xSplit = xSplit
        self.ySplit = ySplit
        self.topLeftCell = topLeftCell
        self.activePane = activePane
        self.state = state


class Selection(Serialisable):
    pane = NoneSet(values=("bottomRight", "topRight", "bottomLeft", "topLeft"))
    activeCell = String(allow_none=True)
    activeCellId = Integer(allow_none=True)
    sqref = String(allow_none=True)

    def __init__(self,
                 pane=None,
                 activeCell="A1",
                 activeCellId=None,
                 sqref="A1"):
        self.pane = pane
        self.activeCell = activeCell
        self.activeCellId = activeCellId
        self.sqref = sqref


class SheetView(Serialisable):

    """Information about the visible portions of this sheet."""

    tagname = "sheetView"

    windowProtection = Bool(allow_none=True)
    showFormulas = Bool(allow_none=True)
    showGridLines = Bool(allow_none=True)
    showRowColHeaders = Bool(allow_none=True)
    showZeros = Bool(allow_none=True)
    rightToLeft = Bool(allow_none=True)
    tabSelected = Bool(allow_none=True)
    showRuler = Bool(allow_none=True)
    showOutlineSymbols = Bool(allow_none=True)
    defaultGridColor = Bool(allow_none=True)
    showWhiteSpace = Bool(allow_none=True)
    view = NoneSet(values=("normal", "pageBreakPreview", "pageLayout"))
    topLeftCell = String(allow_none=True)
    colorId = Integer(allow_none=True)
    zoomScale = Integer(allow_none=True)
    zoomScaleNormal = Integer(allow_none=True)
    zoomScaleSheetLayoutView = Integer(allow_none=True)
    zoomScalePageLayoutView = Integer(allow_none=True)
    zoomToFit = Bool(allow_none=True) # Chart sheets only
    workbookViewId = Integer()
    selection = Sequence(expected_type=Selection)
    pane = Typed(expected_type=Pane, allow_none=True)

    def __init__(self,
                 windowProtection=None,
                 showFormulas=None,
                 showGridLines=None,
                 showRowColHeaders=None,
                 showZeros=None,
                 rightToLeft=None,
                 tabSelected=None,
                 showRuler=None,
                 showOutlineSymbols=None,
                 defaultGridColor=None,
                 showWhiteSpace=None,
                 view=None,
                 topLeftCell=None,
                 colorId=None,
                 zoomScale=None,
                 zoomScaleNormal=None,
                 zoomScaleSheetLayoutView=None,
                 zoomScalePageLayoutView=None,
                 zoomToFit=None,
                 workbookViewId=0,
                 selection=None,
                 pane=None,):
        self.windowProtection = windowProtection
        self.showFormulas = showFormulas
        self.showGridLines = showGridLines
        self.showRowColHeaders = showRowColHeaders
        self.showZeros = showZeros
        self.rightToLeft = rightToLeft
        self.tabSelected = tabSelected
        self.showRuler = showRuler
        self.showOutlineSymbols = showOutlineSymbols
        self.defaultGridColor = defaultGridColor
        self.showWhiteSpace = showWhiteSpace
        self.view = view
        self.topLeftCell = topLeftCell
        self.colorId = colorId
        self.zoomScale = zoomScale
        self.zoomScaleNormal = zoomScaleNormal
        self.zoomScaleSheetLayoutView = zoomScaleSheetLayoutView
        self.zoomScalePageLayoutView = zoomScalePageLayoutView
        self.zoomToFit = zoomToFit
        self.workbookViewId = workbookViewId
        self.pane = pane
        if selection is None:
            selection = (Selection(), )
        self.selection = selection


class SheetViewList(Serialisable):

    tagname = "sheetViews"

    sheetView = Sequence(expected_type=SheetView, )
    extLst = Typed(expected_type=ExtensionList, allow_none=True)

    __elements__ = ('sheetView',)

    def __init__(self,
                 sheetView=None,
                 extLst=None,
                ):
        if sheetView is None:
            sheetView = [SheetView()]
        self.sheetView = sheetView


    @property
    def active(self):
        """
        Returns the first sheet view which is assumed to be active
        """
        return self.sheetView[0]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\urllib3\packages\backports\weakref_finalize.py ===
# -*- coding: utf-8 -*-
"""
backports.weakref_finalize
~~~~~~~~~~~~~~~~~~

Backports the Python 3 ``weakref.finalize`` method.
"""
from __future__ import absolute_import

import itertools
import sys
from weakref import ref

__all__ = ["weakref_finalize"]


class weakref_finalize(object):
    """Class for finalization of weakrefable objects
    finalize(obj, func, *args, **kwargs) returns a callable finalizer
    object which will be called when obj is garbage collected. The
    first time the finalizer is called it evaluates func(*arg, **kwargs)
    and returns the result. After this the finalizer is dead, and
    calling it just returns None.
    When the program exits any remaining finalizers for which the
    atexit attribute is true will be run in reverse order of creation.
    By default atexit is true.
    """

    # Finalizer objects don't have any state of their own.  They are
    # just used as keys to lookup _Info objects in the registry.  This
    # ensures that they cannot be part of a ref-cycle.

    __slots__ = ()
    _registry = {}
    _shutdown = False
    _index_iter = itertools.count()
    _dirty = False
    _registered_with_atexit = False

    class _Info(object):
        __slots__ = ("weakref", "func", "args", "kwargs", "atexit", "index")

    def __init__(self, obj, func, *args, **kwargs):
        if not self._registered_with_atexit:
            # We may register the exit function more than once because
            # of a thread race, but that is harmless
            import atexit

            atexit.register(self._exitfunc)
            weakref_finalize._registered_with_atexit = True
        info = self._Info()
        info.weakref = ref(obj, self)
        info.func = func
        info.args = args
        info.kwargs = kwargs or None
        info.atexit = True
        info.index = next(self._index_iter)
        self._registry[self] = info
        weakref_finalize._dirty = True

    def __call__(self, _=None):
        """If alive then mark as dead and return func(*args, **kwargs);
        otherwise return None"""
        info = self._registry.pop(self, None)
        if info and not self._shutdown:
            return info.func(*info.args, **(info.kwargs or {}))

    def detach(self):
        """If alive then mark as dead and return (obj, func, args, kwargs);
        otherwise return None"""
        info = self._registry.get(self)
        obj = info and info.weakref()
        if obj is not None and self._registry.pop(self, None):
            return (obj, info.func, info.args, info.kwargs or {})

    def peek(self):
        """If alive then return (obj, func, args, kwargs);
        otherwise return None"""
        info = self._registry.get(self)
        obj = info and info.weakref()
        if obj is not None:
            return (obj, info.func, info.args, info.kwargs or {})

    @property
    def alive(self):
        """Whether finalizer is alive"""
        return self in self._registry

    @property
    def atexit(self):
        """Whether finalizer should be called at exit"""
        info = self._registry.get(self)
        return bool(info) and info.atexit

    @atexit.setter
    def atexit(self, value):
        info = self._registry.get(self)
        if info:
            info.atexit = bool(value)

    def __repr__(self):
        info = self._registry.get(self)
        obj = info and info.weakref()
        if obj is None:
            return "<%s object at %#x; dead>" % (type(self).__name__, id(self))
        else:
            return "<%s object at %#x; for %r at %#x>" % (
                type(self).__name__,
                id(self),
                type(obj).__name__,
                id(obj),
            )

    @classmethod
    def _select_for_exit(cls):
        # Return live finalizers marked for exit, oldest first
        L = [(f, i) for (f, i) in cls._registry.items() if i.atexit]
        L.sort(key=lambda item: item[1].index)
        return [f for (f, i) in L]

    @classmethod
    def _exitfunc(cls):
        # At shutdown invoke finalizers for which atexit is true.
        # This is called once all other non-daemonic threads have been
        # joined.
        reenable_gc = False
        try:
            if cls._registry:
                import gc

                if gc.isenabled():
                    reenable_gc = True
                    gc.disable()
                pending = None
                while True:
                    if pending is None or weakref_finalize._dirty:
                        pending = cls._select_for_exit()
                        weakref_finalize._dirty = False
                    if not pending:
                        break
                    f = pending.pop()
                    try:
                        # gc is disabled, so (assuming no daemonic
                        # threads) the following is the only line in
                        # this function which might trigger creation
                        # of a new finalizer
                        f()
                    except Exception:
                        sys.excepthook(*sys.exc_info())
                    assert f not in cls._registry
        finally:
            # prevent any more finalizers from executing during shutdown
            weakref_finalize._shutdown = True
            if reenable_gc:
                gc.enable()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\engine\characteristics.py ===
# engine/characteristics.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
from __future__ import annotations

import abc
import typing
from typing import Any
from typing import ClassVar

if typing.TYPE_CHECKING:
    from .base import Connection
    from .interfaces import DBAPIConnection
    from .interfaces import Dialect


class ConnectionCharacteristic(abc.ABC):
    """An abstract base for an object that can set, get and reset a
    per-connection characteristic, typically one that gets reset when the
    connection is returned to the connection pool.

    transaction isolation is the canonical example, and the
    ``IsolationLevelCharacteristic`` implementation provides this for the
    ``DefaultDialect``.

    The ``ConnectionCharacteristic`` class should call upon the ``Dialect`` for
    the implementation of each method.   The object exists strictly to serve as
    a dialect visitor that can be placed into the
    ``DefaultDialect.connection_characteristics`` dictionary where it will take
    effect for calls to :meth:`_engine.Connection.execution_options` and
    related APIs.

    .. versionadded:: 1.4

    """

    __slots__ = ()

    transactional: ClassVar[bool] = False

    @abc.abstractmethod
    def reset_characteristic(
        self, dialect: Dialect, dbapi_conn: DBAPIConnection
    ) -> None:
        """Reset the characteristic on the DBAPI connection to its default
        value."""

    @abc.abstractmethod
    def set_characteristic(
        self, dialect: Dialect, dbapi_conn: DBAPIConnection, value: Any
    ) -> None:
        """set characteristic on the DBAPI connection to a given value."""

    def set_connection_characteristic(
        self,
        dialect: Dialect,
        conn: Connection,
        dbapi_conn: DBAPIConnection,
        value: Any,
    ) -> None:
        """set characteristic on the :class:`_engine.Connection` to a given
        value.

        .. versionadded:: 2.0.30 - added to support elements that are local
           to the :class:`_engine.Connection` itself.

        """
        self.set_characteristic(dialect, dbapi_conn, value)

    @abc.abstractmethod
    def get_characteristic(
        self, dialect: Dialect, dbapi_conn: DBAPIConnection
    ) -> Any:
        """Given a DBAPI connection, get the current value of the
        characteristic.

        """

    def get_connection_characteristic(
        self, dialect: Dialect, conn: Connection, dbapi_conn: DBAPIConnection
    ) -> Any:
        """Given a :class:`_engine.Connection`, get the current value of the
        characteristic.

        .. versionadded:: 2.0.30 - added to support elements that are local
           to the :class:`_engine.Connection` itself.

        """
        return self.get_characteristic(dialect, dbapi_conn)


class IsolationLevelCharacteristic(ConnectionCharacteristic):
    """Manage the isolation level on a DBAPI connection"""

    transactional: ClassVar[bool] = True

    def reset_characteristic(
        self, dialect: Dialect, dbapi_conn: DBAPIConnection
    ) -> None:
        dialect.reset_isolation_level(dbapi_conn)

    def set_characteristic(
        self, dialect: Dialect, dbapi_conn: DBAPIConnection, value: Any
    ) -> None:
        dialect._assert_and_set_isolation_level(dbapi_conn, value)

    def get_characteristic(
        self, dialect: Dialect, dbapi_conn: DBAPIConnection
    ) -> Any:
        return dialect.get_isolation_level(dbapi_conn)


class LoggingTokenCharacteristic(ConnectionCharacteristic):
    """Manage the 'logging_token' option of a :class:`_engine.Connection`.

    .. versionadded:: 2.0.30

    """

    transactional: ClassVar[bool] = False

    def reset_characteristic(
        self, dialect: Dialect, dbapi_conn: DBAPIConnection
    ) -> None:
        pass

    def set_characteristic(
        self, dialect: Dialect, dbapi_conn: DBAPIConnection, value: Any
    ) -> None:
        raise NotImplementedError()

    def set_connection_characteristic(
        self,
        dialect: Dialect,
        conn: Connection,
        dbapi_conn: DBAPIConnection,
        value: Any,
    ) -> None:
        if value:
            conn._message_formatter = lambda msg: "[%s] %s" % (value, msg)
        else:
            del conn._message_formatter

    def get_characteristic(
        self, dialect: Dialect, dbapi_conn: DBAPIConnection
    ) -> Any:
        raise NotImplementedError()

    def get_connection_characteristic(
        self, dialect: Dialect, conn: Connection, dbapi_conn: DBAPIConnection
    ) -> Any:
        return conn._execution_options.get("logging_token", None)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\testing\pickleable.py ===
# testing/pickleable.py
# Copyright (C) 2005-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: ignore-errors


"""Classes used in pickling tests, need to be at the module level for
unpickling.
"""

from __future__ import annotations

from .entities import ComparableEntity
from ..schema import Column
from ..types import String


class User(ComparableEntity):
    pass


class Order(ComparableEntity):
    pass


class Dingaling(ComparableEntity):
    pass


class EmailUser(User):
    pass


class Address(ComparableEntity):
    pass


# TODO: these are kind of arbitrary....
class Child1(ComparableEntity):
    pass


class Child2(ComparableEntity):
    pass


class Parent(ComparableEntity):
    pass


class Screen:
    def __init__(self, obj, parent=None):
        self.obj = obj
        self.parent = parent


class Mixin:
    email_address = Column(String)


class AddressWMixin(Mixin, ComparableEntity):
    pass


class Foo:
    def __init__(self, moredata, stuff="im stuff"):
        self.data = "im data"
        self.stuff = stuff
        self.moredata = moredata

    __hash__ = object.__hash__

    def __eq__(self, other):
        return (
            other.data == self.data
            and other.stuff == self.stuff
            and other.moredata == self.moredata
        )


class Bar:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    __hash__ = object.__hash__

    def __eq__(self, other):
        return (
            other.__class__ is self.__class__
            and other.x == self.x
            and other.y == self.y
        )

    def __str__(self):
        return "Bar(%d, %d)" % (self.x, self.y)


class OldSchool:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return (
            other.__class__ is self.__class__
            and other.x == self.x
            and other.y == self.y
        )


class OldSchoolWithoutCompare:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class BarWithoutCompare:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return "Bar(%d, %d)" % (self.x, self.y)


class NotComparable:
    def __init__(self, data):
        self.data = data

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return NotImplemented

    def __ne__(self, other):
        return NotImplemented


class BrokenComparable:
    def __init__(self, data):
        self.data = data

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        raise NotImplementedError

    def __ne__(self, other):
        raise NotImplementedError

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\matrices\expressions\matadd.py ===
from functools import reduce
import operator

from sympy.core import Basic, sympify
from sympy.core.add import add, Add, _could_extract_minus_sign
from sympy.core.sorting import default_sort_key
from sympy.functions import adjoint
from sympy.matrices.matrixbase import MatrixBase
from sympy.matrices.expressions.transpose import transpose
from sympy.strategies import (rm_id, unpack, flatten, sort, condition,
    exhaust, do_one, glom)
from sympy.matrices.expressions.matexpr import MatrixExpr
from sympy.matrices.expressions.special import ZeroMatrix, GenericZeroMatrix
from sympy.matrices.expressions._shape import validate_matadd_integer as validate
from sympy.utilities.iterables import sift
from sympy.utilities.exceptions import sympy_deprecation_warning

# XXX: MatAdd should perhaps not subclass directly from Add
class MatAdd(MatrixExpr, Add):
    """A Sum of Matrix Expressions

    MatAdd inherits from and operates like SymPy Add

    Examples
    ========

    >>> from sympy import MatAdd, MatrixSymbol
    >>> A = MatrixSymbol('A', 5, 5)
    >>> B = MatrixSymbol('B', 5, 5)
    >>> C = MatrixSymbol('C', 5, 5)
    >>> MatAdd(A, B, C)
    A + B + C
    """
    is_MatAdd = True

    identity = GenericZeroMatrix()

    def __new__(cls, *args, evaluate=False, check=None, _sympify=True):
        if not args:
            return cls.identity

        # This must be removed aggressively in the constructor to avoid
        # TypeErrors from GenericZeroMatrix().shape
        args = list(filter(lambda i: cls.identity != i, args))
        if _sympify:
            args = list(map(sympify, args))

        if not all(isinstance(arg, MatrixExpr) for arg in args):
            raise TypeError("Mix of Matrix and Scalar symbols")

        obj = Basic.__new__(cls, *args)

        if check is not None:
            sympy_deprecation_warning(
                "Passing check to MatAdd is deprecated and the check argument will be removed in a future version.",
                deprecated_since_version="1.11",
                active_deprecations_target='remove-check-argument-from-matrix-operations')

        if check is not False:
            validate(*args)

        if evaluate:
            obj = cls._evaluate(obj)

        return obj

    @classmethod
    def _evaluate(cls, expr):
        return canonicalize(expr)

    @property
    def shape(self):
        return self.args[0].shape

    def could_extract_minus_sign(self):
        return _could_extract_minus_sign(self)

    def expand(self, **kwargs):
        expanded = super(MatAdd, self).expand(**kwargs)
        return self._evaluate(expanded)

    def _entry(self, i, j, **kwargs):
        return Add(*[arg._entry(i, j, **kwargs) for arg in self.args])

    def _eval_transpose(self):
        return MatAdd(*[transpose(arg) for arg in self.args]).doit()

    def _eval_adjoint(self):
        return MatAdd(*[adjoint(arg) for arg in self.args]).doit()

    def _eval_trace(self):
        from .trace import trace
        return Add(*[trace(arg) for arg in self.args]).doit()

    def doit(self, **hints):
        deep = hints.get('deep', True)
        if deep:
            args = [arg.doit(**hints) for arg in self.args]
        else:
            args = self.args
        return canonicalize(MatAdd(*args))

    def _eval_derivative_matrix_lines(self, x):
        add_lines = [arg._eval_derivative_matrix_lines(x) for arg in self.args]
        return [j for i in add_lines for j in i]

add.register_handlerclass((Add, MatAdd), MatAdd)


factor_of = lambda arg: arg.as_coeff_mmul()[0]
matrix_of = lambda arg: unpack(arg.as_coeff_mmul()[1])
def combine(cnt, mat):
    if cnt == 1:
        return mat
    else:
        return cnt * mat


def merge_explicit(matadd):
    """ Merge explicit MatrixBase arguments

    Examples
    ========

    >>> from sympy import MatrixSymbol, eye, Matrix, MatAdd, pprint
    >>> from sympy.matrices.expressions.matadd import merge_explicit
    >>> A = MatrixSymbol('A', 2, 2)
    >>> B = eye(2)
    >>> C = Matrix([[1, 2], [3, 4]])
    >>> X = MatAdd(A, B, C)
    >>> pprint(X)
        [1  0]   [1  2]
    A + [    ] + [    ]
        [0  1]   [3  4]
    >>> pprint(merge_explicit(X))
        [2  2]
    A + [    ]
        [3  5]
    """
    groups = sift(matadd.args, lambda arg: isinstance(arg, MatrixBase))
    if len(groups[True]) > 1:
        return MatAdd(*(groups[False] + [reduce(operator.add, groups[True])]))
    else:
        return matadd


rules = (rm_id(lambda x: x == 0 or isinstance(x, ZeroMatrix)),
         unpack,
         flatten,
         glom(matrix_of, factor_of, combine),
         merge_explicit,
         sort(default_sort_key))

canonicalize = exhaust(condition(lambda x: isinstance(x, MatAdd),
                                 do_one(*rules)))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\werkzeug\middleware\profiler.py ===
"""
Application Profiler
====================

This module provides a middleware that profiles each request with the
:mod:`cProfile` module. This can help identify bottlenecks in your code
that may be slowing down your application.

.. autoclass:: ProfilerMiddleware

:copyright: 2007 Pallets
:license: BSD-3-Clause
"""

from __future__ import annotations

import os.path
import sys
import time
import typing as t
from pstats import Stats

try:
    from cProfile import Profile
except ImportError:
    from profile import Profile  # type: ignore

if t.TYPE_CHECKING:
    from _typeshed.wsgi import StartResponse
    from _typeshed.wsgi import WSGIApplication
    from _typeshed.wsgi import WSGIEnvironment


class ProfilerMiddleware:
    """Wrap a WSGI application and profile the execution of each
    request. Responses are buffered so that timings are more exact.

    If ``stream`` is given, :class:`pstats.Stats` are written to it
    after each request. If ``profile_dir`` is given, :mod:`cProfile`
    data files are saved to that directory, one file per request.

    The filename can be customized by passing ``filename_format``. If
    it is a string, it will be formatted using :meth:`str.format` with
    the following fields available:

    -   ``{method}`` - The request method; GET, POST, etc.
    -   ``{path}`` - The request path or 'root' should one not exist.
    -   ``{elapsed}`` - The elapsed time of the request in milliseconds.
    -   ``{time}`` - The time of the request.

    If it is a callable, it will be called with the WSGI ``environ`` and
    be expected to return a filename string. The ``environ`` dictionary
    will also have the ``"werkzeug.profiler"`` key populated with a
    dictionary containing the following fields (more may be added in the
    future):
    -   ``{elapsed}`` - The elapsed time of the request in milliseconds.
    -   ``{time}`` - The time of the request.

    :param app: The WSGI application to wrap.
    :param stream: Write stats to this stream. Disable with ``None``.
    :param sort_by: A tuple of columns to sort stats by. See
        :meth:`pstats.Stats.sort_stats`.
    :param restrictions: A tuple of restrictions to filter stats by. See
        :meth:`pstats.Stats.print_stats`.
    :param profile_dir: Save profile data files to this directory.
    :param filename_format: Format string for profile data file names,
        or a callable returning a name. See explanation above.

    .. code-block:: python

        from werkzeug.middleware.profiler import ProfilerMiddleware
        app = ProfilerMiddleware(app)

    .. versionchanged:: 3.0
        Added the ``"werkzeug.profiler"`` key to the ``filename_format(environ)``
        parameter with the  ``elapsed`` and ``time`` fields.

    .. versionchanged:: 0.15
        Stats are written even if ``profile_dir`` is given, and can be
        disable by passing ``stream=None``.

    .. versionadded:: 0.15
        Added ``filename_format``.

    .. versionadded:: 0.9
        Added ``restrictions`` and ``profile_dir``.
    """

    def __init__(
        self,
        app: WSGIApplication,
        stream: t.IO[str] | None = sys.stdout,
        sort_by: t.Iterable[str] = ("time", "calls"),
        restrictions: t.Iterable[str | int | float] = (),
        profile_dir: str | None = None,
        filename_format: str = "{method}.{path}.{elapsed:.0f}ms.{time:.0f}.prof",
    ) -> None:
        self._app = app
        self._stream = stream
        self._sort_by = sort_by
        self._restrictions = restrictions
        self._profile_dir = profile_dir
        self._filename_format = filename_format

    def __call__(
        self, environ: WSGIEnvironment, start_response: StartResponse
    ) -> t.Iterable[bytes]:
        response_body: list[bytes] = []

        def catching_start_response(status, headers, exc_info=None):  # type: ignore
            start_response(status, headers, exc_info)
            return response_body.append

        def runapp() -> None:
            app_iter = self._app(
                environ, t.cast("StartResponse", catching_start_response)
            )
            response_body.extend(app_iter)

            if hasattr(app_iter, "close"):
                app_iter.close()

        profile = Profile()
        start = time.time()
        profile.runcall(runapp)
        body = b"".join(response_body)
        elapsed = time.time() - start

        if self._profile_dir is not None:
            if callable(self._filename_format):
                environ["werkzeug.profiler"] = {
                    "elapsed": elapsed * 1000.0,
                    "time": time.time(),
                }
                filename = self._filename_format(environ)
            else:
                filename = self._filename_format.format(
                    method=environ["REQUEST_METHOD"],
                    path=environ["PATH_INFO"].strip("/").replace("/", ".") or "root",
                    elapsed=elapsed * 1000.0,
                    time=time.time(),
                )
            filename = os.path.join(self._profile_dir, filename)
            profile.dump_stats(filename)

        if self._stream is not None:
            stats = Stats(profile, stream=self._stream)
            stats.sort_stats(*self._sort_by)
            print("-" * 80, file=self._stream)
            path_info = environ.get("PATH_INFO", "")
            print(f"PATH: {path_info!r}", file=self._stream)
            stats.print_stats(*self._restrictions)
            print(f"{'-' * 80}\n", file=self._stream)

        return [body]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\capi\onnxruntime_validation.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""
Check OS requirements for ONNX Runtime Python Bindings.
"""

import linecache
import platform
import warnings


def check_distro_info():
    __my_distro__ = ""
    __my_distro_ver__ = ""
    __my_system__ = platform.system().lower()

    __OS_RELEASE_FILE__ = "/etc/os-release"  # noqa: N806
    __LSB_RELEASE_FILE__ = "/etc/lsb-release"  # noqa: N806

    if __my_system__ == "windows":
        __my_distro__ = __my_system__
        __my_distro_ver__ = platform.release().lower()

        if __my_distro_ver__ not in ["10", "11"]:
            warnings.warn(
                f"Unsupported Windows version ({__my_distro_ver__}). ONNX Runtime supports Windows 10 and above, only."
            )
    elif __my_system__ == "linux":
        """Although the 'platform' python module for getting Distro information works well on standard OS images
        running on real hardware, it is not accurate when running on Azure VMs, Git Bash, Cygwin, etc.
        The returned values for release and version are unpredictable for virtualized or emulated environments.
        /etc/os-release and /etc/lsb_release files, on the other hand, are guaranteed to exist and have standard values
        in all OSes supported by onnxruntime. The former is the current standard file to check OS info and the latter
        is its predecessor.
        """
        # Newer systems have /etc/os-release with relevant distro info
        __my_distro__ = linecache.getline(__OS_RELEASE_FILE__, 3)[3:-1]
        __my_distro_ver__ = linecache.getline(__OS_RELEASE_FILE__, 6)[12:-2]

        # Older systems may have /etc/os-release instead
        if not __my_distro__:
            __my_distro__ = linecache.getline(__LSB_RELEASE_FILE__, 1)[11:-1]
            __my_distro_ver__ = linecache.getline(__LSB_RELEASE_FILE__, 2)[16:-1]

        # Instead of trying to parse distro specific files,
        # warn the user ONNX Runtime may not work out of the box
        __my_distro__ = __my_distro__.lower()
        __my_distro_ver__ = __my_distro_ver__.lower()
    elif __my_system__ == "darwin":
        __my_distro__ = __my_system__
        __my_distro_ver__ = platform.release().lower()

        if int(__my_distro_ver__.split(".")[0]) < 11:
            warnings.warn(
                f"Unsupported macOS version ({__my_distro_ver__}). ONNX Runtime supports macOS 11.0 or later."
            )
    elif __my_system__ == "aix":
        import subprocess

        returned_output = subprocess.check_output("oslevel")
        __my_distro_ver__str = returned_output.decode("utf-8")
        __my_distro_ver = __my_distro_ver__str[:3]
    else:
        warnings.warn(
            f"Unsupported platform ({__my_system__}). ONNX Runtime supports Linux, macOS, AIX and Windows platforms, only."
        )


def get_package_name_and_version_info():
    package_name = ""
    version = ""
    cuda_version = ""

    try:
        from .build_and_package_info import __version__ as version
        from .build_and_package_info import package_name

        try:  # noqa: SIM105
            from .build_and_package_info import cuda_version
        except ImportError:
            # cuda_version is optional. For example, cpu only package does not have the attribute.
            pass
    except Exception as e:
        warnings.warn("WARNING: failed to collect package name and version info")
        print(e)

    return package_name, version, cuda_version


def check_training_module():
    import_ortmodule_exception = None

    has_ortmodule = False
    try:
        from onnxruntime.training.ortmodule import ORTModule  # noqa: F401

        has_ortmodule = True
    except ImportError:
        # ORTModule not present
        has_ortmodule = False
    except Exception as e:
        # this may happen if Cuda is not installed, we want to raise it after
        # for any exception other than not having ortmodule, we want to continue
        # device version validation and raise the exception after.
        try:
            from onnxruntime.training.ortmodule._fallback import ORTModuleInitException

            if isinstance(e, ORTModuleInitException):
                # ORTModule is present but not ready to run yet
                has_ortmodule = True
        except Exception:
            # ORTModule not present
            has_ortmodule = False

        if not has_ortmodule:
            import_ortmodule_exception = e

    # collect onnxruntime package name, version, and cuda version
    package_name, version, cuda_version = get_package_name_and_version_info()

    if has_ortmodule and cuda_version:
        try:
            # collect cuda library build info. the library info may not be available
            # when the build environment has none or multiple libraries installed
            try:
                from .build_and_package_info import cudart_version
            except ImportError:
                warnings.warn("WARNING: failed to get cudart_version from onnxruntime build info.")
                cudart_version = None

            def print_build_package_info():
                warnings.warn(f"onnxruntime training package info: package_name: {package_name}")
                warnings.warn(f"onnxruntime training package info: __version__: {version}")
                warnings.warn(f"onnxruntime training package info: cuda_version: {cuda_version}")
                warnings.warn(f"onnxruntime build info: cudart_version: {cudart_version}")

            # collection cuda library info from current environment.
            from onnxruntime.capi.onnxruntime_collect_build_info import find_cudart_versions

            local_cudart_versions = find_cudart_versions(build_env=False, build_cuda_version=cuda_version)
            if cudart_version and local_cudart_versions and cudart_version not in local_cudart_versions:
                print_build_package_info()
                warnings.warn("WARNING: failed to find cudart version that matches onnxruntime build info")
                warnings.warn(f"WARNING: found cudart versions: {local_cudart_versions}")
        except Exception as e:
            warnings.warn("WARNING: failed to collect onnxruntime version and build info")
            print(e)

    if import_ortmodule_exception:
        raise import_ortmodule_exception

    return has_ortmodule, package_name, version, cuda_version

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\sample.py ===
"""
Module containing utilities for NDFrame.sample() and .GroupBy.sample()
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pandas._libs import lib

from pandas.core.dtypes.generic import (
    ABCDataFrame,
    ABCSeries,
)

if TYPE_CHECKING:
    from pandas._typing import AxisInt

    from pandas.core.generic import NDFrame


def preprocess_weights(obj: NDFrame, weights, axis: AxisInt) -> np.ndarray:
    """
    Process and validate the `weights` argument to `NDFrame.sample` and
    `.GroupBy.sample`.

    Returns `weights` as an ndarray[np.float64], validated except for normalizing
    weights (because that must be done groupwise in groupby sampling).
    """
    # If a series, align with frame
    if isinstance(weights, ABCSeries):
        weights = weights.reindex(obj.axes[axis])

    # Strings acceptable if a dataframe and axis = 0
    if isinstance(weights, str):
        if isinstance(obj, ABCDataFrame):
            if axis == 0:
                try:
                    weights = obj[weights]
                except KeyError as err:
                    raise KeyError(
                        "String passed to weights not a valid column"
                    ) from err
            else:
                raise ValueError(
                    "Strings can only be passed to "
                    "weights when sampling from rows on "
                    "a DataFrame"
                )
        else:
            raise ValueError(
                "Strings cannot be passed as weights when sampling from a Series."
            )

    if isinstance(obj, ABCSeries):
        func = obj._constructor
    else:
        func = obj._constructor_sliced

    weights = func(weights, dtype="float64")._values

    if len(weights) != obj.shape[axis]:
        raise ValueError("Weights and axis to be sampled must be of same length")

    if lib.has_infs(weights):
        raise ValueError("weight vector may not include `inf` values")

    if (weights < 0).any():
        raise ValueError("weight vector many not include negative values")

    missing = np.isnan(weights)
    if missing.any():
        # Don't modify weights in place
        weights = weights.copy()
        weights[missing] = 0
    return weights


def process_sampling_size(
    n: int | None, frac: float | None, replace: bool
) -> int | None:
    """
    Process and validate the `n` and `frac` arguments to `NDFrame.sample` and
    `.GroupBy.sample`.

    Returns None if `frac` should be used (variable sampling sizes), otherwise returns
    the constant sampling size.
    """
    # If no frac or n, default to n=1.
    if n is None and frac is None:
        n = 1
    elif n is not None and frac is not None:
        raise ValueError("Please enter a value for `frac` OR `n`, not both")
    elif n is not None:
        if n < 0:
            raise ValueError(
                "A negative number of rows requested. Please provide `n` >= 0."
            )
        if n % 1 != 0:
            raise ValueError("Only integers accepted as `n` values")
    else:
        assert frac is not None  # for mypy
        if frac > 1 and not replace:
            raise ValueError(
                "Replace has to be set to `True` when "
                "upsampling the population `frac` > 1."
            )
        if frac < 0:
            raise ValueError(
                "A negative number of rows requested. Please provide `frac` >= 0."
            )

    return n


def sample(
    obj_len: int,
    size: int,
    replace: bool,
    weights: np.ndarray | None,
    random_state: np.random.RandomState | np.random.Generator,
) -> np.ndarray:
    """
    Randomly sample `size` indices in `np.arange(obj_len)`

    Parameters
    ----------
    obj_len : int
        The length of the indices being considered
    size : int
        The number of values to choose
    replace : bool
        Allow or disallow sampling of the same row more than once.
    weights : np.ndarray[np.float64] or None
        If None, equal probability weighting, otherwise weights according
        to the vector normalized
    random_state: np.random.RandomState or np.random.Generator
        State used for the random sampling

    Returns
    -------
    np.ndarray[np.intp]
    """
    if weights is not None:
        weight_sum = weights.sum()
        if weight_sum != 0:
            weights = weights / weight_sum
        else:
            raise ValueError("Invalid weights: weights sum to zero")

    return random_state.choice(obj_len, size=size, replace=replace, p=weights).astype(
        np.intp, copy=False
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\array_algos\replace.py ===
"""
Methods used by Block.replace and related methods.
"""
from __future__ import annotations

import operator
import re
from re import Pattern
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np

from pandas.core.dtypes.common import (
    is_bool,
    is_re,
    is_re_compilable,
)
from pandas.core.dtypes.missing import isna

if TYPE_CHECKING:
    from pandas._typing import (
        ArrayLike,
        Scalar,
        npt,
    )


def should_use_regex(regex: bool, to_replace: Any) -> bool:
    """
    Decide whether to treat `to_replace` as a regular expression.
    """
    if is_re(to_replace):
        regex = True

    regex = regex and is_re_compilable(to_replace)

    # Don't use regex if the pattern is empty.
    regex = regex and re.compile(to_replace).pattern != ""
    return regex


def compare_or_regex_search(
    a: ArrayLike, b: Scalar | Pattern, regex: bool, mask: npt.NDArray[np.bool_]
) -> ArrayLike:
    """
    Compare two array-like inputs of the same shape or two scalar values

    Calls operator.eq or re.search, depending on regex argument. If regex is
    True, perform an element-wise regex matching.

    Parameters
    ----------
    a : array-like
    b : scalar or regex pattern
    regex : bool
    mask : np.ndarray[bool]

    Returns
    -------
    mask : array-like of bool
    """
    if isna(b):
        return ~mask

    def _check_comparison_types(
        result: ArrayLike | bool, a: ArrayLike, b: Scalar | Pattern
    ):
        """
        Raises an error if the two arrays (a,b) cannot be compared.
        Otherwise, returns the comparison result as expected.
        """
        if is_bool(result) and isinstance(a, np.ndarray):
            type_names = [type(a).__name__, type(b).__name__]

            type_names[0] = f"ndarray(dtype={a.dtype})"

            raise TypeError(
                f"Cannot compare types {repr(type_names[0])} and {repr(type_names[1])}"
            )

    if not regex or not should_use_regex(regex, b):
        # TODO: should use missing.mask_missing?
        op = lambda x: operator.eq(x, b)
    else:
        op = np.vectorize(
            lambda x: bool(re.search(b, x))
            if isinstance(x, str) and isinstance(b, (str, Pattern))
            else False
        )

    # GH#32621 use mask to avoid comparing to NAs
    if isinstance(a, np.ndarray):
        a = a[mask]

    result = op(a)

    if isinstance(result, np.ndarray) and mask is not None:
        # The shape of the mask can differ to that of the result
        # since we may compare only a subset of a's or b's elements
        tmp = np.zeros(mask.shape, dtype=np.bool_)
        np.place(tmp, mask, result)
        result = tmp

    _check_comparison_types(result, a, b)
    return result


def replace_regex(
    values: ArrayLike, rx: re.Pattern, value, mask: npt.NDArray[np.bool_] | None
) -> None:
    """
    Parameters
    ----------
    values : ArrayLike
        Object dtype.
    rx : re.Pattern
    value : Any
    mask : np.ndarray[bool], optional

    Notes
    -----
    Alters values in-place.
    """

    # deal with replacing values with objects (strings) that match but
    # whose replacement is not a string (numeric, nan, object)
    if isna(value) or not isinstance(value, str):

        def re_replacer(s):
            if is_re(rx) and isinstance(s, str):
                return value if rx.search(s) is not None else s
            else:
                return s

    else:
        # value is guaranteed to be a string here, s can be either a string
        # or null if it's null it gets returned
        def re_replacer(s):
            if is_re(rx) and isinstance(s, str):
                return rx.sub(value, s)
            else:
                return s

    f = np.vectorize(re_replacer, otypes=[np.object_])

    if mask is None:
        values[:] = f(values)
    else:
        if values.ndim != mask.ndim:
            mask = np.broadcast_to(mask, values.shape)
        values[mask] = f(values[mask])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\internals\ops.py ===
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    NamedTuple,
)

from pandas.core.dtypes.common import is_1d_only_ea_dtype

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pandas._libs.internals import BlockPlacement
    from pandas._typing import ArrayLike

    from pandas.core.internals.blocks import Block
    from pandas.core.internals.managers import BlockManager


class BlockPairInfo(NamedTuple):
    lvals: ArrayLike
    rvals: ArrayLike
    locs: BlockPlacement
    left_ea: bool
    right_ea: bool
    rblk: Block


def _iter_block_pairs(
    left: BlockManager, right: BlockManager
) -> Iterator[BlockPairInfo]:
    # At this point we have already checked the parent DataFrames for
    #  assert rframe._indexed_same(lframe)

    for blk in left.blocks:
        locs = blk.mgr_locs
        blk_vals = blk.values

        left_ea = blk_vals.ndim == 1

        rblks = right._slice_take_blocks_ax0(locs.indexer, only_slice=True)

        # Assertions are disabled for performance, but should hold:
        # if left_ea:
        #    assert len(locs) == 1, locs
        #    assert len(rblks) == 1, rblks
        #    assert rblks[0].shape[0] == 1, rblks[0].shape

        for rblk in rblks:
            right_ea = rblk.values.ndim == 1

            lvals, rvals = _get_same_shape_values(blk, rblk, left_ea, right_ea)
            info = BlockPairInfo(lvals, rvals, locs, left_ea, right_ea, rblk)
            yield info


def operate_blockwise(
    left: BlockManager, right: BlockManager, array_op
) -> BlockManager:
    # At this point we have already checked the parent DataFrames for
    #  assert rframe._indexed_same(lframe)

    res_blks: list[Block] = []
    for lvals, rvals, locs, left_ea, right_ea, rblk in _iter_block_pairs(left, right):
        res_values = array_op(lvals, rvals)
        if (
            left_ea
            and not right_ea
            and hasattr(res_values, "reshape")
            and not is_1d_only_ea_dtype(res_values.dtype)
        ):
            res_values = res_values.reshape(1, -1)
        nbs = rblk._split_op_result(res_values)

        # Assertions are disabled for performance, but should hold:
        # if right_ea or left_ea:
        #    assert len(nbs) == 1
        # else:
        #    assert res_values.shape == lvals.shape, (res_values.shape, lvals.shape)

        _reset_block_mgr_locs(nbs, locs)

        res_blks.extend(nbs)

    # Assertions are disabled for performance, but should hold:
    #  slocs = {y for nb in res_blks for y in nb.mgr_locs.as_array}
    #  nlocs = sum(len(nb.mgr_locs.as_array) for nb in res_blks)
    #  assert nlocs == len(left.items), (nlocs, len(left.items))
    #  assert len(slocs) == nlocs, (len(slocs), nlocs)
    #  assert slocs == set(range(nlocs)), slocs

    new_mgr = type(right)(tuple(res_blks), axes=right.axes, verify_integrity=False)
    return new_mgr


def _reset_block_mgr_locs(nbs: list[Block], locs) -> None:
    """
    Reset mgr_locs to correspond to our original DataFrame.
    """
    for nb in nbs:
        nblocs = locs[nb.mgr_locs.indexer]
        nb.mgr_locs = nblocs
        # Assertions are disabled for performance, but should hold:
        #  assert len(nblocs) == nb.shape[0], (len(nblocs), nb.shape)
        #  assert all(x in locs.as_array for x in nb.mgr_locs.as_array)


def _get_same_shape_values(
    lblk: Block, rblk: Block, left_ea: bool, right_ea: bool
) -> tuple[ArrayLike, ArrayLike]:
    """
    Slice lblk.values to align with rblk.  Squeeze if we have EAs.
    """
    lvals = lblk.values
    rvals = rblk.values

    # Require that the indexing into lvals be slice-like
    assert rblk.mgr_locs.is_slice_like, rblk.mgr_locs

    # TODO(EA2D): with 2D EAs only this first clause would be needed
    if not (left_ea or right_ea):
        # error: No overload variant of "__getitem__" of "ExtensionArray" matches
        # argument type "Tuple[Union[ndarray, slice], slice]"
        lvals = lvals[rblk.mgr_locs.indexer, :]  # type: ignore[call-overload]
        assert lvals.shape == rvals.shape, (lvals.shape, rvals.shape)
    elif left_ea and right_ea:
        assert lvals.shape == rvals.shape, (lvals.shape, rvals.shape)
    elif right_ea:
        # lvals are 2D, rvals are 1D

        # error: No overload variant of "__getitem__" of "ExtensionArray" matches
        # argument type "Tuple[Union[ndarray, slice], slice]"
        lvals = lvals[rblk.mgr_locs.indexer, :]  # type: ignore[call-overload]
        assert lvals.shape[0] == 1, lvals.shape
        lvals = lvals[0, :]
    else:
        # lvals are 1D, rvals are 2D
        assert rvals.shape[0] == 1, rvals.shape
        # error: No overload variant of "__getitem__" of "ExtensionArray" matches
        # argument type "Tuple[int, slice]"
        rvals = rvals[0, :]  # type: ignore[call-overload]

    return lvals, rvals


def blockwise_all(left: BlockManager, right: BlockManager, op) -> bool:
    """
    Blockwise `all` reduction.
    """
    for info in _iter_block_pairs(left, right):
        res = op(info.lvals, info.rvals)
        if not res:
            return False
    return True

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\remote\websocket_connection.py ===
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
import json
import logging
from ssl import CERT_NONE
from threading import Thread
from time import sleep

from websocket import WebSocketApp  # type: ignore

from selenium.common import WebDriverException

logger = logging.getLogger(__name__)


class WebSocketConnection:
    _response_wait_timeout = 30
    _response_wait_interval = 0.1

    _max_log_message_size = 9999

    def __init__(self, url):
        self.callbacks = {}
        self.session_id = None
        self.url = url

        self._id = 0
        self._messages = {}
        self._started = False

        self._start_ws()
        self._wait_until(lambda: self._started)

    def close(self):
        self._ws_thread.join(timeout=self._response_wait_timeout)
        self._ws.close()
        self._started = False
        self._ws = None

    def execute(self, command):
        self._id += 1
        payload = self._serialize_command(command)
        payload["id"] = self._id
        if self.session_id:
            payload["sessionId"] = self.session_id

        data = json.dumps(payload)
        logger.debug(f"-> {data}"[: self._max_log_message_size])
        self._ws.send(data)

        current_id = self._id
        self._wait_until(lambda: current_id in self._messages)
        response = self._messages.pop(current_id)

        if "error" in response:
            error = response["error"]
            if "message" in response:
                error_msg = f"{error}: {response['message']}"
                raise WebDriverException(error_msg)
            else:
                raise WebDriverException(error)
        else:
            result = response["result"]
            return self._deserialize_result(result, command)

    def add_callback(self, event, callback):
        event_name = event.event_class
        if event_name not in self.callbacks:
            self.callbacks[event_name] = []

        def _callback(params):
            callback(event.from_json(params))

        self.callbacks[event_name].append(_callback)
        return id(_callback)

    on = add_callback

    def remove_callback(self, event, callback_id):
        event_name = event.event_class
        if event_name in self.callbacks:
            for callback in self.callbacks[event_name]:
                if id(callback) == callback_id:
                    self.callbacks[event_name].remove(callback)
                    return

    def _serialize_command(self, command):
        return next(command)

    def _deserialize_result(self, result, command):
        try:
            _ = command.send(result)
            raise WebDriverException("The command's generator function did not exit when expected!")
        except StopIteration as exit:
            return exit.value

    def _start_ws(self):
        def on_open(ws):
            self._started = True

        def on_message(ws, message):
            self._process_message(message)

        def on_error(ws, error):
            logger.debug(f"error: {error}")
            ws.close()

        def run_socket():
            if self.url.startswith("wss://"):
                self._ws.run_forever(sslopt={"cert_reqs": CERT_NONE}, suppress_origin=True)
            else:
                self._ws.run_forever(suppress_origin=True)

        self._ws = WebSocketApp(self.url, on_open=on_open, on_message=on_message, on_error=on_error)
        self._ws_thread = Thread(target=run_socket)
        self._ws_thread.start()

    def _process_message(self, message):
        message = json.loads(message)
        logger.debug(f"<- {message}"[: self._max_log_message_size])

        if "id" in message:
            self._messages[message["id"]] = message

        if "method" in message:
            params = message["params"]
            for callback in self.callbacks.get(message["method"], []):
                Thread(target=callback, args=(params,)).start()

    def _wait_until(self, condition):
        timeout = self._response_wait_timeout
        interval = self._response_wait_interval

        while timeout > 0:
            result = condition()
            if result:
                return result
            else:
                timeout -= interval
                sleep(interval)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\tensor\functions.py ===
from collections.abc import Iterable
from functools import singledispatch

from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.core.sympify import sympify
from sympy.core.parameters import global_parameters


class TensorProduct(Expr):
    """
    Generic class for tensor products.
    """
    is_number = False

    def __new__(cls, *args, **kwargs):
        from sympy.tensor.array import NDimArray, tensorproduct, Array
        from sympy.matrices.expressions.matexpr import MatrixExpr
        from sympy.matrices.matrixbase import MatrixBase
        from sympy.strategies import flatten

        args = [sympify(arg) for arg in args]
        evaluate = kwargs.get("evaluate", global_parameters.evaluate)

        if not evaluate:
            obj = Expr.__new__(cls, *args)
            return obj

        arrays = []
        other = []
        scalar = S.One
        for arg in args:
            if isinstance(arg, (Iterable, MatrixBase, NDimArray)):
                arrays.append(Array(arg))
            elif isinstance(arg, (MatrixExpr,)):
                other.append(arg)
            else:
                scalar *= arg

        coeff = scalar*tensorproduct(*arrays)
        if len(other) == 0:
            return coeff
        if coeff != 1:
            newargs = [coeff] + other
        else:
            newargs = other
        obj = Expr.__new__(cls, *newargs, **kwargs)
        return flatten(obj)

    def rank(self):
        return len(self.shape)

    def _get_args_shapes(self):
        from sympy.tensor.array import Array
        return [i.shape if hasattr(i, "shape") else Array(i).shape for i in self.args]

    @property
    def shape(self):
        shape_list = self._get_args_shapes()
        return sum(shape_list, ())

    def __getitem__(self, index):
        index = iter(index)
        return Mul.fromiter(
            arg.__getitem__(tuple(next(index) for i in shp))
            for arg, shp in zip(self.args, self._get_args_shapes())
        )


@singledispatch
def shape(expr):
    """
    Return the shape of the *expr* as a tuple. *expr* should represent
    suitable object such as matrix or array.

    Parameters
    ==========

    expr : SymPy object having ``MatrixKind`` or ``ArrayKind``.

    Raises
    ======

    NoShapeError : Raised when object with wrong kind is passed.

    Examples
    ========

    This function returns the shape of any object representing matrix or array.

    >>> from sympy import shape, Array, ImmutableDenseMatrix, Integral
    >>> from sympy.abc import x
    >>> A = Array([1, 2])
    >>> shape(A)
    (2,)
    >>> shape(Integral(A, x))
    (2,)
    >>> M = ImmutableDenseMatrix([1, 2])
    >>> shape(M)
    (2, 1)
    >>> shape(Integral(M, x))
    (2, 1)

    You can support new type by dispatching.

    >>> from sympy import Expr
    >>> class NewExpr(Expr):
    ...     pass
    >>> @shape.register(NewExpr)
    ... def _(expr):
    ...     return shape(expr.args[0])
    >>> shape(NewExpr(M))
    (2, 1)

    If unsuitable expression is passed, ``NoShapeError()`` will be raised.

    >>> shape(Integral(x, x))
    Traceback (most recent call last):
      ...
    sympy.tensor.functions.NoShapeError: shape() called on non-array object: Integral(x, x)

    Notes
    =====

    Array-like classes (such as ``Matrix`` or ``NDimArray``) has ``shape``
    property which returns its shape, but it cannot be used for non-array
    classes containing array. This function returns the shape of any
    registered object representing array.

    """
    if hasattr(expr, "shape"):
        return expr.shape
    raise NoShapeError(
        "%s does not have shape, or its type is not registered to shape()." % expr)


class NoShapeError(Exception):
    """
    Raised when ``shape()`` is called on non-array object.

    This error can be imported from ``sympy.tensor.functions``.

    Examples
    ========

    >>> from sympy import shape
    >>> from sympy.abc import x
    >>> shape(x)
    Traceback (most recent call last):
      ...
    sympy.tensor.functions.NoShapeError: shape() called on non-array object: x
    """
    pass

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\trio\_core\_tests\test_unbounded_queue.py ===
from __future__ import annotations

import itertools

import pytest

from ... import _core
from ...testing import assert_checkpoints, wait_all_tasks_blocked

pytestmark = pytest.mark.filterwarnings(
    "ignore:.*UnboundedQueue:trio.TrioDeprecationWarning",
)


async def test_UnboundedQueue_basic() -> None:
    q: _core.UnboundedQueue[str | int | None] = _core.UnboundedQueue()
    q.put_nowait("hi")
    assert await q.get_batch() == ["hi"]
    with pytest.raises(_core.WouldBlock):
        q.get_batch_nowait()
    q.put_nowait(1)
    q.put_nowait(2)
    q.put_nowait(3)
    assert q.get_batch_nowait() == [1, 2, 3]

    assert q.empty()
    assert q.qsize() == 0
    q.put_nowait(None)
    assert not q.empty()
    assert q.qsize() == 1

    stats = q.statistics()
    assert stats.qsize == 1
    assert stats.tasks_waiting == 0

    # smoke test
    repr(q)


async def test_UnboundedQueue_blocking() -> None:
    record = []
    q = _core.UnboundedQueue[int]()

    async def get_batch_consumer() -> None:
        while True:
            batch = await q.get_batch()
            assert batch
            record.append(batch)

    async def aiter_consumer() -> None:
        async for batch in q:
            assert batch
            record.append(batch)

    for consumer in (get_batch_consumer, aiter_consumer):
        record.clear()
        async with _core.open_nursery() as nursery:
            nursery.start_soon(consumer)
            await _core.wait_all_tasks_blocked()
            stats = q.statistics()
            assert stats.qsize == 0
            assert stats.tasks_waiting == 1
            q.put_nowait(10)
            q.put_nowait(11)
            await _core.wait_all_tasks_blocked()
            q.put_nowait(12)
            await _core.wait_all_tasks_blocked()
            assert record == [[10, 11], [12]]
            nursery.cancel_scope.cancel()


async def test_UnboundedQueue_fairness() -> None:
    q = _core.UnboundedQueue[int]()

    # If there's no-one else around, we can put stuff in and take it out
    # again, no problem
    q.put_nowait(1)
    q.put_nowait(2)
    assert q.get_batch_nowait() == [1, 2]

    result = None

    async def get_batch(q: _core.UnboundedQueue[int]) -> None:
        nonlocal result
        result = await q.get_batch()

    # But if someone else is waiting to read, then they get dibs
    async with _core.open_nursery() as nursery:
        nursery.start_soon(get_batch, q)
        await _core.wait_all_tasks_blocked()
        q.put_nowait(3)
        q.put_nowait(4)
        with pytest.raises(_core.WouldBlock):
            q.get_batch_nowait()
    assert result == [3, 4]

    # If two tasks are trying to read, they alternate
    record = []

    async def reader(name: str) -> None:
        while True:
            record.append((name, await q.get_batch()))

    async with _core.open_nursery() as nursery:
        nursery.start_soon(reader, "a")
        await _core.wait_all_tasks_blocked()
        nursery.start_soon(reader, "b")
        await _core.wait_all_tasks_blocked()

        for i in range(20):
            q.put_nowait(i)
            await _core.wait_all_tasks_blocked()

        nursery.cancel_scope.cancel()

    assert record == list(zip(itertools.cycle("ab"), [[i] for i in range(20)]))


async def test_UnboundedQueue_trivial_yields() -> None:
    q = _core.UnboundedQueue[None]()

    q.put_nowait(None)
    with assert_checkpoints():
        await q.get_batch()

    q.put_nowait(None)
    with assert_checkpoints():
        async for _ in q:  # pragma: no branch
            break


async def test_UnboundedQueue_no_spurious_wakeups() -> None:
    # If we have two tasks waiting, and put two items into the queue... then
    # only one task wakes up
    record = []

    async def getter(q: _core.UnboundedQueue[int], i: int) -> None:
        got = await q.get_batch()
        record.append((i, got))

    async with _core.open_nursery() as nursery:
        q = _core.UnboundedQueue[int]()
        nursery.start_soon(getter, q, 1)
        await wait_all_tasks_blocked()
        nursery.start_soon(getter, q, 2)
        await wait_all_tasks_blocked()

        for i in range(10):
            q.put_nowait(i)
        await wait_all_tasks_blocked()

        assert record == [(1, list(range(10)))]

        nursery.cancel_scope.cancel()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\google\protobuf\proto.py ===
# Protocol Buffers - Google's data interchange format
# Copyright 2008 Google Inc.  All rights reserved.
#
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file or at
# https://developers.google.com/open-source/licenses/bsd

"""Contains the Nextgen Pythonic protobuf APIs."""

import io
from typing import Text, Type, TypeVar

from google.protobuf.internal import decoder
from google.protobuf.internal import encoder
from google.protobuf.message import Message

_MESSAGE = TypeVar('_MESSAGE', bound='Message')


def serialize(message: _MESSAGE, deterministic: bool = None) -> bytes:
  """Return the serialized proto.

  Args:
    message: The proto message to be serialized.
    deterministic: If true, requests deterministic serialization
        of the protobuf, with predictable ordering of map keys.

  Returns:
    A binary bytes representation of the message.
  """
  return message.SerializeToString(deterministic=deterministic)


def parse(message_class: Type[_MESSAGE], payload: bytes) -> _MESSAGE:
  """Given a serialized data in binary form, deserialize it into a Message.

  Args:
    message_class: The message meta class.
    payload: A serialized bytes in binary form.

  Returns:
    A new message deserialized from payload.
  """
  new_message = message_class()
  new_message.ParseFromString(payload)
  return new_message


def serialize_length_prefixed(message: _MESSAGE, output: io.BytesIO) -> None:
  """Writes the size of the message as a varint and the serialized message.

  Writes the size of the message as a varint and then the serialized message.
  This allows more data to be written to the output after the message. Use
  parse_length_prefixed to parse messages written by this method.

  The output stream must be buffered, e.g. using
  https://docs.python.org/3/library/io.html#buffered-streams.

  Example usage:
    out = io.BytesIO()
    for msg in message_list:
      proto.serialize_length_prefixed(msg, out)

  Args:
    message: The protocol buffer message that should be serialized.
    output: BytesIO or custom buffered IO that data should be written to.
  """
  size = message.ByteSize()
  encoder._VarintEncoder()(output.write, size)
  out_size = output.write(serialize(message))

  if out_size != size:
    raise TypeError(
        'Failed to write complete message (wrote: %d, expected: %d)'
        '. Ensure output is using buffered IO.' % (out_size, size)
    )


def parse_length_prefixed(
    message_class: Type[_MESSAGE], input_bytes: io.BytesIO
) -> _MESSAGE:
  """Parse a message from input_bytes.

  Args:
    message_class: The protocol buffer message class that parser should parse.
    input_bytes: A buffered input.

  Example usage:
    while True:
      msg = proto.parse_length_prefixed(message_class, input_bytes)
      if msg is None:
        break
      ...

  Returns:
    A parsed message if successful. None if input_bytes is at EOF.
  """
  size = decoder._DecodeVarint(input_bytes)
  if size is None:
    # It is the end of buffered input. See example usage in the
    # API description.
    return None

  message = message_class()

  if size == 0:
    return message

  parsed_size = message.ParseFromString(input_bytes.read(size))
  if parsed_size != size:
    raise ValueError(
        'Truncated message or non-buffered input_bytes: '
        'Expected {0} bytes but only {1} bytes parsed for '
        '{2}.'.format(size, parsed_size, message.DESCRIPTOR.name)
    )
  return message


def byte_size(message: Message) -> int:
  """Returns the serialized size of this message.

  Args:
    message: A proto message.

  Returns:
    int: The number of bytes required to serialize this message.
  """
  return message.ByteSize()


def clear_message(message: Message) -> None:
  """Clears all data that was set in the message.

  Args:
    message: The proto message to be cleared.
  """
  message.Clear()


def clear_field(message: Message, field_name: Text) -> None:
  """Clears the contents of a given field.

  Inside a oneof group, clears the field set. If the name neither refers to a
  defined field or oneof group, :exc:`ValueError` is raised.

  Args:
    message: The proto message.
    field_name (str): The name of the field to be cleared.

  Raises:
    ValueError: if the `field_name` is not a member of this message.
  """
  message.ClearField(field_name)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\h11\_receivebuffer.py ===
import re
import sys
from typing import List, Optional, Union

__all__ = ["ReceiveBuffer"]


# Operations we want to support:
# - find next \r\n or \r\n\r\n (\n or \n\n are also acceptable),
#   or wait until there is one
# - read at-most-N bytes
# Goals:
# - on average, do this fast
# - worst case, do this in O(n) where n is the number of bytes processed
# Plan:
# - store bytearray, offset, how far we've searched for a separator token
# - use the how-far-we've-searched data to avoid rescanning
# - while doing a stream of uninterrupted processing, advance offset instead
#   of constantly copying
# WARNING:
# - I haven't benchmarked or profiled any of this yet.
#
# Note that starting in Python 3.4, deleting the initial n bytes from a
# bytearray is amortized O(n), thanks to some excellent work by Antoine
# Martin:
#
#     https://bugs.python.org/issue19087
#
# This means that if we only supported 3.4+, we could get rid of the code here
# involving self._start and self.compress, because it's doing exactly the same
# thing that bytearray now does internally.
#
# BUT unfortunately, we still support 2.7, and reading short segments out of a
# long buffer MUST be O(bytes read) to avoid DoS issues, so we can't actually
# delete this code. Yet:
#
#     https://pythonclock.org/
#
# (Two things to double-check first though: make sure PyPy also has the
# optimization, and benchmark to make sure it's a win, since we do have a
# slightly clever thing where we delay calling compress() until we've
# processed a whole event, which could in theory be slightly more efficient
# than the internal bytearray support.)
blank_line_regex = re.compile(b"\n\r?\n", re.MULTILINE)


class ReceiveBuffer:
    def __init__(self) -> None:
        self._data = bytearray()
        self._next_line_search = 0
        self._multiple_lines_search = 0

    def __iadd__(self, byteslike: Union[bytes, bytearray]) -> "ReceiveBuffer":
        self._data += byteslike
        return self

    def __bool__(self) -> bool:
        return bool(len(self))

    def __len__(self) -> int:
        return len(self._data)

    # for @property unprocessed_data
    def __bytes__(self) -> bytes:
        return bytes(self._data)

    def _extract(self, count: int) -> bytearray:
        # extracting an initial slice of the data buffer and return it
        out = self._data[:count]
        del self._data[:count]

        self._next_line_search = 0
        self._multiple_lines_search = 0

        return out

    def maybe_extract_at_most(self, count: int) -> Optional[bytearray]:
        """
        Extract a fixed number of bytes from the buffer.
        """
        out = self._data[:count]
        if not out:
            return None

        return self._extract(count)

    def maybe_extract_next_line(self) -> Optional[bytearray]:
        """
        Extract the first line, if it is completed in the buffer.
        """
        # Only search in buffer space that we've not already looked at.
        search_start_index = max(0, self._next_line_search - 1)
        partial_idx = self._data.find(b"\r\n", search_start_index)

        if partial_idx == -1:
            self._next_line_search = len(self._data)
            return None

        # + 2 is to compensate len(b"\r\n")
        idx = partial_idx + 2

        return self._extract(idx)

    def maybe_extract_lines(self) -> Optional[List[bytearray]]:
        """
        Extract everything up to the first blank line, and return a list of lines.
        """
        # Handle the case where we have an immediate empty line.
        if self._data[:1] == b"\n":
            self._extract(1)
            return []

        if self._data[:2] == b"\r\n":
            self._extract(2)
            return []

        # Only search in buffer space that we've not already looked at.
        match = blank_line_regex.search(self._data, self._multiple_lines_search)
        if match is None:
            self._multiple_lines_search = max(0, len(self._data) - 2)
            return None

        # Truncate the buffer and return it.
        idx = match.span(0)[-1]
        out = self._extract(idx)
        lines = out.split(b"\n")

        for line in lines:
            if line.endswith(b"\r"):
                del line[-1]

        assert lines[-2] == lines[-1] == b""

        del lines[-2:]

        return lines

    # In theory we should wait until `\r\n` before starting to validate
    # incoming data. However it's interesting to detect (very) invalid data
    # early given they might not even contain `\r\n` at all (hence only
    # timeout will get rid of them).
    # This is not a 100% effective detection but more of a cheap sanity check
    # allowing for early abort in some useful cases.
    # This is especially interesting when peer is messing up with HTTPS and
    # sent us a TLS stream where we were expecting plain HTTP given all
    # versions of TLS so far start handshake with a 0x16 message type code.
    def is_next_line_obviously_invalid_request_line(self) -> bool:
        try:
            # HTTP header line must not contain non-printable characters
            # and should not start with a space
            return self._data[0] < 0x21
        except IndexError:
            return False

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\numpy\_typing\_add_docstring.py ===
"""A module for creating docstrings for sphinx ``data`` domains."""

import re
import textwrap

from ._array_like import NDArray

_docstrings_list = []


def add_newdoc(name: str, value: str, doc: str) -> None:
    """Append ``_docstrings_list`` with a docstring for `name`.

    Parameters
    ----------
    name : str
        The name of the object.
    value : str
        A string-representation of the object.
    doc : str
        The docstring of the object.

    """
    _docstrings_list.append((name, value, doc))


def _parse_docstrings() -> str:
    """Convert all docstrings in ``_docstrings_list`` into a single
    sphinx-legible text block.

    """
    type_list_ret = []
    for name, value, doc in _docstrings_list:
        s = textwrap.dedent(doc).replace("\n", "\n    ")

        # Replace sections by rubrics
        lines = s.split("\n")
        new_lines = []
        indent = ""
        for line in lines:
            m = re.match(r'^(\s+)[-=]+\s*$', line)
            if m and new_lines:
                prev = textwrap.dedent(new_lines.pop())
                if prev == "Examples":
                    indent = ""
                    new_lines.append(f'{m.group(1)}.. rubric:: {prev}')
                else:
                    indent = 4 * " "
                    new_lines.append(f'{m.group(1)}.. admonition:: {prev}')
                new_lines.append("")
            else:
                new_lines.append(f"{indent}{line}")

        s = "\n".join(new_lines)
        s_block = f""".. data:: {name}\n    :value: {value}\n    {s}"""
        type_list_ret.append(s_block)
    return "\n".join(type_list_ret)


add_newdoc('ArrayLike', 'typing.Union[...]',
    """
    A `~typing.Union` representing objects that can be coerced
    into an `~numpy.ndarray`.

    Among others this includes the likes of:

    * Scalars.
    * (Nested) sequences.
    * Objects implementing the `~class.__array__` protocol.

    .. versionadded:: 1.20

    See Also
    --------
    :term:`array_like`:
        Any scalar or sequence that can be interpreted as an ndarray.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import numpy.typing as npt

        >>> def as_array(a: npt.ArrayLike) -> np.ndarray:
        ...     return np.array(a)

    """)

add_newdoc('DTypeLike', 'typing.Union[...]',
    """
    A `~typing.Union` representing objects that can be coerced
    into a `~numpy.dtype`.

    Among others this includes the likes of:

    * :class:`type` objects.
    * Character codes or the names of :class:`type` objects.
    * Objects with the ``.dtype`` attribute.

    .. versionadded:: 1.20

    See Also
    --------
    :ref:`Specifying and constructing data types <arrays.dtypes.constructing>`
        A comprehensive overview of all objects that can be coerced
        into data types.

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import numpy.typing as npt

        >>> def as_dtype(d: npt.DTypeLike) -> np.dtype:
        ...     return np.dtype(d)

    """)

add_newdoc('NDArray', repr(NDArray),
    """
    A `np.ndarray[tuple[Any, ...], np.dtype[ScalarT]] <numpy.ndarray>`
    type alias :term:`generic <generic type>` w.r.t. its
    `dtype.type <numpy.dtype.type>`.

    Can be used during runtime for typing arrays with a given dtype
    and unspecified shape.

    .. versionadded:: 1.21

    Examples
    --------
    .. code-block:: python

        >>> import numpy as np
        >>> import numpy.typing as npt

        >>> print(npt.NDArray)
        numpy.ndarray[tuple[typing.Any, ...], numpy.dtype[~_ScalarT]]

        >>> print(npt.NDArray[np.float64])
        numpy.ndarray[tuple[typing.Any, ...], numpy.dtype[numpy.float64]]

        >>> NDArrayInt = npt.NDArray[np.int_]
        >>> a: NDArrayInt = np.arange(10)

        >>> def func(a: npt.ArrayLike) -> npt.NDArray[Any]:
        ...     return np.array(a)

    """)

_docstrings = _parse_docstrings()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\commands\index.py ===
import json
import logging
from optparse import Values
from typing import Any, Iterable, List, Optional

from pip._vendor.packaging.version import Version

from pip._internal.cli import cmdoptions
from pip._internal.cli.req_command import IndexGroupCommand
from pip._internal.cli.status_codes import ERROR, SUCCESS
from pip._internal.commands.search import (
    get_installed_distribution,
    print_dist_installation_info,
)
from pip._internal.exceptions import CommandError, DistributionNotFound, PipError
from pip._internal.index.collector import LinkCollector
from pip._internal.index.package_finder import PackageFinder
from pip._internal.models.selection_prefs import SelectionPreferences
from pip._internal.models.target_python import TargetPython
from pip._internal.network.session import PipSession
from pip._internal.utils.misc import write_output

logger = logging.getLogger(__name__)


class IndexCommand(IndexGroupCommand):
    """
    Inspect information available from package indexes.
    """

    ignore_require_venv = True
    usage = """
        %prog versions <package>
    """

    def add_options(self) -> None:
        cmdoptions.add_target_python_options(self.cmd_opts)

        self.cmd_opts.add_option(cmdoptions.ignore_requires_python())
        self.cmd_opts.add_option(cmdoptions.pre())
        self.cmd_opts.add_option(cmdoptions.json())
        self.cmd_opts.add_option(cmdoptions.no_binary())
        self.cmd_opts.add_option(cmdoptions.only_binary())

        index_opts = cmdoptions.make_option_group(
            cmdoptions.index_group,
            self.parser,
        )

        self.parser.insert_option_group(0, index_opts)
        self.parser.insert_option_group(0, self.cmd_opts)

    def run(self, options: Values, args: List[str]) -> int:
        handlers = {
            "versions": self.get_available_package_versions,
        }

        # Determine action
        if not args or args[0] not in handlers:
            logger.error(
                "Need an action (%s) to perform.",
                ", ".join(sorted(handlers)),
            )
            return ERROR

        action = args[0]

        # Error handling happens here, not in the action-handlers.
        try:
            handlers[action](options, args[1:])
        except PipError as e:
            logger.error(e.args[0])
            return ERROR

        return SUCCESS

    def _build_package_finder(
        self,
        options: Values,
        session: PipSession,
        target_python: Optional[TargetPython] = None,
        ignore_requires_python: Optional[bool] = None,
    ) -> PackageFinder:
        """
        Create a package finder appropriate to the index command.
        """
        link_collector = LinkCollector.create(session, options=options)

        # Pass allow_yanked=False to ignore yanked versions.
        selection_prefs = SelectionPreferences(
            allow_yanked=False,
            allow_all_prereleases=options.pre,
            ignore_requires_python=ignore_requires_python,
        )

        return PackageFinder.create(
            link_collector=link_collector,
            selection_prefs=selection_prefs,
            target_python=target_python,
        )

    def get_available_package_versions(self, options: Values, args: List[Any]) -> None:
        if len(args) != 1:
            raise CommandError("You need to specify exactly one argument")

        target_python = cmdoptions.make_target_python(options)
        query = args[0]

        with self._build_session(options) as session:
            finder = self._build_package_finder(
                options=options,
                session=session,
                target_python=target_python,
                ignore_requires_python=options.ignore_requires_python,
            )

            versions: Iterable[Version] = (
                candidate.version for candidate in finder.find_all_candidates(query)
            )

            if not options.pre:
                # Remove prereleases
                versions = (
                    version for version in versions if not version.is_prerelease
                )
            versions = set(versions)

            if not versions:
                raise DistributionNotFound(
                    f"No matching distribution found for {query}"
                )

            formatted_versions = [str(ver) for ver in sorted(versions, reverse=True)]
            latest = formatted_versions[0]

        dist = get_installed_distribution(query)

        if options.json:
            structured_output = {
                "name": query,
                "versions": formatted_versions,
                "latest": latest,
            }

            if dist is not None:
                structured_output["installed_version"] = str(dist.version)

            write_output(json.dumps(structured_output))

        else:
            write_output(f"{query} ({latest})")
            write_output("Available versions: {}".format(", ".join(formatted_versions)))
            print_dist_installation_info(latest, dist)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\pygments\unistring.py ===
"""
    pygments.unistring
    ~~~~~~~~~~~~~~~~~~

    Strings of all Unicode characters of a certain category.
    Used for matching in Unicode-aware languages. Run to regenerate.

    Inspired by chartypes_create.py from the MoinMoin project.

    :copyright: Copyright 2006-2025 by the Pygments team, see AUTHORS.
    :license: BSD, see LICENSE for details.
"""

Cc = '\x00-\x1f\x7f-\x9f'

Cf = '\xad\u0600-\u0605\u061c\u06dd\u070f\u08e2\u180e\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u206f\ufeff\ufff9-\ufffb\U000110bd\U000110cd\U0001bca0-\U0001bca3\U0001d173-\U0001d17a\U000e0001\U000e0020-\U000e007f'

Cn = '\u0378-\u0379\u0380-\u0383\u038b\u038d\u03a2\u0530\u0557-\u0558\u058b-\u058c\u0590\u05c8-\u05cf\u05eb-\u05ee\u05f5-\u05ff\u061d\u070e\u074b-\u074c\u07b2-\u07bf\u07fb-\u07fc\u082e-\u082f\u083f\u085c-\u085d\u085f\u086b-\u089f\u08b5\u08be-\u08d2\u0984\u098d-\u098e\u0991-\u0992\u09a9\u09b1\u09b3-\u09b5\u09ba-\u09bb\u09c5-\u09c6\u09c9-\u09ca\u09cf-\u09d6\u09d8-\u09db\u09de\u09e4-\u09e5\u09ff-\u0a00\u0a04\u0a0b-\u0a0e\u0a11-\u0a12\u0a29\u0a31\u0a34\u0a37\u0a3a-\u0a3b\u0a3d\u0a43-\u0a46\u0a49-\u0a4a\u0a4e-\u0a50\u0a52-\u0a58\u0a5d\u0a5f-\u0a65\u0a77-\u0a80\u0a84\u0a8e\u0a92\u0aa9\u0ab1\u0ab4\u0aba-\u0abb\u0ac6\u0aca\u0ace-\u0acf\u0ad1-\u0adf\u0ae4-\u0ae5\u0af2-\u0af8\u0b00\u0b04\u0b0d-\u0b0e\u0b11-\u0b12\u0b29\u0b31\u0b34\u0b3a-\u0b3b\u0b45-\u0b46\u0b49-\u0b4a\u0b4e-\u0b55\u0b58-\u0b5b\u0b5e\u0b64-\u0b65\u0b78-\u0b81\u0b84\u0b8b-\u0b8d\u0b91\u0b96-\u0b98\u0b9b\u0b9d\u0ba0-\u0ba2\u0ba5-\u0ba7\u0bab-\u0bad\u0bba-\u0bbd\u0bc3-\u0bc5\u0bc9\u0bce-\u0bcf\u0bd1-\u0bd6\u0bd8-\u0be5\u0bfb-\u0bff\u0c0d\u0c11\u0c29\u0c3a-\u0c3c\u0c45\u0c49\u0c4e-\u0c54\u0c57\u0c5b-\u0c5f\u0c64-\u0c65\u0c70-\u0c77\u0c8d\u0c91\u0ca9\u0cb4\u0cba-\u0cbb\u0cc5\u0cc9\u0cce-\u0cd4\u0cd7-\u0cdd\u0cdf\u0ce4-\u0ce5\u0cf0\u0cf3-\u0cff\u0d04\u0d0d\u0d11\u0d45\u0d49\u0d50-\u0d53\u0d64-\u0d65\u0d80-\u0d81\u0d84\u0d97-\u0d99\u0db2\u0dbc\u0dbe-\u0dbf\u0dc7-\u0dc9\u0dcb-\u0dce\u0dd5\u0dd7\u0de0-\u0de5\u0df0-\u0df1\u0df5-\u0e00\u0e3b-\u0e3e\u0e5c-\u0e80\u0e83\u0e85-\u0e86\u0e89\u0e8b-\u0e8c\u0e8e-\u0e93\u0e98\u0ea0\u0ea4\u0ea6\u0ea8-\u0ea9\u0eac\u0eba\u0ebe-\u0ebf\u0ec5\u0ec7\u0ece-\u0ecf\u0eda-\u0edb\u0ee0-\u0eff\u0f48\u0f6d-\u0f70\u0f98\u0fbd\u0fcd\u0fdb-\u0fff\u10c6\u10c8-\u10cc\u10ce-\u10cf\u1249\u124e-\u124f\u1257\u1259\u125e-\u125f\u1289\u128e-\u128f\u12b1\u12b6-\u12b7\u12bf\u12c1\u12c6-\u12c7\u12d7\u1311\u1316-\u1317\u135b-\u135c\u137d-\u137f\u139a-\u139f\u13f6-\u13f7\u13fe-\u13ff\u169d-\u169f\u16f9-\u16ff\u170d\u1715-\u171f\u1737-\u173f\u1754-\u175f\u176d\u1771\u1774-\u177f\u17de-\u17df\u17ea-\u17ef\u17fa-\u17ff\u180f\u181a-\u181f\u1879-\u187f\u18ab-\u18af\u18f6-\u18ff\u191f\u192c-\u192f\u193c-\u193f\u1941-\u1943\u196e-\u196f\u1975-\u197f\u19ac-\u19af\u19ca-\u19cf\u19db-\u19dd\u1a1c-\u1a1d\u1a5f\u1a7d-\u1a7e\u1a8a-\u1a8f\u1a9a-\u1a9f\u1aae-\u1aaf\u1abf-\u1aff\u1b4c-\u1b4f\u1b7d-\u1b7f\u1bf4-\u1bfb\u1c38-\u1c3a\u1c4a-\u1c4c\u1c89-\u1c8f\u1cbb-\u1cbc\u1cc8-\u1ccf\u1cfa-\u1cff\u1dfa\u1f16-\u1f17\u1f1e-\u1f1f\u1f46-\u1f47\u1f4e-\u1f4f\u1f58\u1f5a\u1f5c\u1f5e\u1f7e-\u1f7f\u1fb5\u1fc5\u1fd4-\u1fd5\u1fdc\u1ff0-\u1ff1\u1ff5\u1fff\u2065\u2072-\u2073\u208f\u209d-\u209f\u20c0-\u20cf\u20f1-\u20ff\u218c-\u218f\u2427-\u243f\u244b-\u245f\u2b74-\u2b75\u2b96-\u2b97\u2bc9\u2bff\u2c2f\u2c5f\u2cf4-\u2cf8\u2d26\u2d28-\u2d2c\u2d2e-\u2d2f\u2d68-\u2d6e\u2d71-\u2d7e\u2d97-\u2d9f\u2da7\u2daf\u2db7\u2dbf\u2dc7\u2dcf\u2dd7\u2ddf\u2e4f-\u2e7f\u2e9a\u2ef4-\u2eff\u2fd6-\u2fef\u2ffc-\u2fff\u3040\u3097-\u3098\u3100-\u3104\u3130\u318f\u31bb-\u31bf\u31e4-\u31ef\u321f\u32ff\u4db6-\u4dbf\u9ff0-\u9fff\ua48d-\ua48f\ua4c7-\ua4cf\ua62c-\ua63f\ua6f8-\ua6ff\ua7ba-\ua7f6\ua82c-\ua82f\ua83a-\ua83f\ua878-\ua87f\ua8c6-\ua8cd\ua8da-\ua8df\ua954-\ua95e\ua97d-\ua97f\ua9ce\ua9da-\ua9dd\ua9ff\uaa37-\uaa3f\uaa4e-\uaa4f\uaa5a-\uaa5b\uaac3-\uaada\uaaf7-\uab00\uab07-\uab08\uab0f-\uab10\uab17-\uab1f\uab27\uab2f\uab66-\uab6f\uabee-\uabef\uabfa-\uabff\ud7a4-\ud7af\ud7c7-\ud7ca\ud7fc-\ud7ff\ufa6e-\ufa6f\ufada-\ufaff\ufb07-\ufb12\ufb18-\ufb1c\ufb37\ufb3d\ufb3f\ufb42\ufb45\ufbc2-\ufbd2\ufd40-\ufd4f\ufd90-\ufd91\ufdc8-\ufdef\ufdfe-\ufdff\ufe1a-\ufe1f\ufe53\ufe67\ufe6c-\ufe6f\ufe75\ufefd-\ufefe\uff00\uffbf-\uffc1\uffc8-\uffc9\uffd0-\uffd1\uffd8-\uffd9\uffdd-\uffdf\uffe7\uffef-\ufff8\ufffe-\uffff\U0001000c\U00010027\U0001003b\U0001003e\U0001004e-\U0001004f\U0001005e-\U0001007f\U000100fb-\U000100ff\U00010103-\U00010106\U00010134-\U00010136\U0001018f\U0001019c-\U0001019f\U000101a1-\U000101cf\U000101fe-\U0001027f\U0001029d-\U0001029f\U000102d1-\U000102df\U000102fc-\U000102ff\U00010324-\U0001032c\U0001034b-\U0001034f\U0001037b-\U0001037f\U0001039e\U000103c4-\U000103c7\U000103d6-\U000103ff\U0001049e-\U0001049f\U000104aa-\U000104af\U000104d4-\U000104d7\U000104fc-\U000104ff\U00010528-\U0001052f\U00010564-\U0001056e\U00010570-\U000105ff\U00010737-\U0001073f\U00010756-\U0001075f\U00010768-\U000107ff\U00010806-\U00010807\U00010809\U00010836\U00010839-\U0001083b\U0001083d-\U0001083e\U00010856\U0001089f-\U000108a6\U000108b0-\U000108df\U000108f3\U000108f6-\U000108fa\U0001091c-\U0001091e\U0001093a-\U0001093e\U00010940-\U0001097f\U000109b8-\U000109bb\U000109d0-\U000109d1\U00010a04\U00010a07-\U00010a0b\U00010a14\U00010a18\U00010a36-\U00010a37\U00010a3b-\U00010a3e\U00010a49-\U00010a4f\U00010a59-\U00010a5f\U00010aa0-\U00010abf\U00010ae7-\U00010aea\U00010af7-\U00010aff\U00010b36-\U00010b38\U00010b56-\U00010b57\U00010b73-\U00010b77\U00010b92-\U00010b98\U00010b9d-\U00010ba8\U00010bb0-\U00010bff\U00010c49-\U00010c7f\U00010cb3-\U00010cbf\U00010cf3-\U00010cf9\U00010d28-\U00010d2f\U00010d3a-\U00010e5f\U00010e7f-\U00010eff\U00010f28-\U00010f2f\U00010f5a-\U00010fff\U0001104e-\U00011051\U00011070-\U0001107e\U000110c2-\U000110cc\U000110ce-\U000110cf\U000110e9-\U000110ef\U000110fa-\U000110ff\U00011135\U00011147-\U0001114f\U00011177-\U0001117f\U000111ce-\U000111cf\U000111e0\U000111f5-\U000111ff\U00011212\U0001123f-\U0001127f\U00011287\U00011289\U0001128e\U0001129e\U000112aa-\U000112af\U000112eb-\U000112ef\U000112fa-\U000112ff\U00011304\U0001130d-\U0001130e\U00011311-\U00011312\U00011329\U00011331\U00011334\U0001133a\U00011345-\U00011346\U00011349-\U0001134a\U0001134e-\U0001134f\U00011351-\U00011356\U00011358-\U0001135c\U00011364-\U00011365\U0001136d-\U0001136f\U00011375-\U000113ff\U0001145a\U0001145c\U0001145f-\U0001147f\U000114c8-\U000114cf\U000114da-\U0001157f\U000115b6-\U000115b7\U000115de-\U000115ff\U00011645-\U0001164f\U0001165a-\U0001165f\U0001166d-\U0001167f\U000116b8-\U000116bf\U000116ca-\U000116ff\U0001171b-\U0001171c\U0001172c-\U0001172f\U00011740-\U000117ff\U0001183c-\U0001189f\U000118f3-\U000118fe\U00011900-\U000119ff\U00011a48-\U00011a4f\U00011a84-\U00011a85\U00011aa3-\U00011abf\U00011af9-\U00011bff\U00011c09\U00011c37\U00011c46-\U00011c4f\U00011c6d-\U00011c6f\U00011c90-\U00011c91\U00011ca8\U00011cb7-\U00011cff\U00011d07\U00011d0a\U00011d37-\U00011d39\U00011d3b\U00011d3e\U00011d48-\U00011d4f\U00011d5a-\U00011d5f\U00011d66\U00011d69\U00011d8f\U00011d92\U00011d99-\U00011d9f\U00011daa-\U00011edf\U00011ef9-\U00011fff\U0001239a-\U000123ff\U0001246f\U00012475-\U0001247f\U00012544-\U00012fff\U0001342f-\U000143ff\U00014647-\U000167ff\U00016a39-\U00016a3f\U00016a5f\U00016a6a-\U00016a6d\U00016a70-\U00016acf\U00016aee-\U00016aef\U00016af6-\U00016aff\U00016b46-\U00016b4f\U00016b5a\U00016b62\U00016b78-\U00016b7c\U00016b90-\U00016e3f\U00016e9b-\U00016eff\U00016f45-\U00016f4f\U00016f7f-\U00016f8e\U00016fa0-\U00016fdf\U00016fe2-\U00016fff\U000187f2-\U000187ff\U00018af3-\U0001afff\U0001b11f-\U0001b16f\U0001b2fc-\U0001bbff\U0001bc6b-\U0001bc6f\U0001bc7d-\U0001bc7f\U0001bc89-\U0001bc8f\U0001bc9a-\U0001bc9b\U0001bca4-\U0001cfff\U0001d0f6-\U0001d0ff\U0001d127-\U0001d128\U0001d1e9-\U0001d1ff\U0001d246-\U0001d2df\U0001d2f4-\U0001d2ff\U0001d357-\U0001d35f\U0001d379-\U0001d3ff\U0001d455\U0001d49d\U0001d4a0-\U0001d4a1\U0001d4a3-\U0001d4a4\U0001d4a7-\U0001d4a8\U0001d4ad\U0001d4ba\U0001d4bc\U0001d4c4\U0001d506\U0001d50b-\U0001d50c\U0001d515\U0001d51d\U0001d53a\U0001d53f\U0001d545\U0001d547-\U0001d549\U0001d551\U0001d6a6-\U0001d6a7\U0001d7cc-\U0001d7cd\U0001da8c-\U0001da9a\U0001daa0\U0001dab0-\U0001dfff\U0001e007\U0001e019-\U0001e01a\U0001e022\U0001e025\U0001e02b-\U0001e7ff\U0001e8c5-\U0001e8c6\U0001e8d7-\U0001e8ff\U0001e94b-\U0001e94f\U0001e95a-\U0001e95d\U0001e960-\U0001ec70\U0001ecb5-\U0001edff\U0001ee04\U0001ee20\U0001ee23\U0001ee25-\U0001ee26\U0001ee28\U0001ee33\U0001ee38\U0001ee3a\U0001ee3c-\U0001ee41\U0001ee43-\U0001ee46\U0001ee48\U0001ee4a\U0001ee4c\U0001ee50\U0001ee53\U0001ee55-\U0001ee56\U0001ee58\U0001ee5a\U0001ee5c\U0001ee5e\U0001ee60\U0001ee63\U0001ee65-\U0001ee66\U0001ee6b\U0001ee73\U0001ee78\U0001ee7d\U0001ee7f\U0001ee8a\U0001ee9c-\U0001eea0\U0001eea4\U0001eeaa\U0001eebc-\U0001eeef\U0001eef2-\U0001efff\U0001f02c-\U0001f02f\U0001f094-\U0001f09f\U0001f0af-\U0001f0b0\U0001f0c0\U0001f0d0\U0001f0f6-\U0001f0ff\U0001f10d-\U0001f10f\U0001f16c-\U0001f16f\U0001f1ad-\U0001f1e5\U0001f203-\U0001f20f\U0001f23c-\U0001f23f\U0001f249-\U0001f24f\U0001f252-\U0001f25f\U0001f266-\U0001f2ff\U0001f6d5-\U0001f6df\U0001f6ed-\U0001f6ef\U0001f6fa-\U0001f6ff\U0001f774-\U0001f77f\U0001f7d9-\U0001f7ff\U0001f80c-\U0001f80f\U0001f848-\U0001f84f\U0001f85a-\U0001f85f\U0001f888-\U0001f88f\U0001f8ae-\U0001f8ff\U0001f90c-\U0001f90f\U0001f93f\U0001f971-\U0001f972\U0001f977-\U0001f979\U0001f97b\U0001f9a3-\U0001f9af\U0001f9ba-\U0001f9bf\U0001f9c3-\U0001f9cf\U0001fa00-\U0001fa5f\U0001fa6e-\U0001ffff\U0002a6d7-\U0002a6ff\U0002b735-\U0002b73f\U0002b81e-\U0002b81f\U0002cea2-\U0002ceaf\U0002ebe1-\U0002f7ff\U0002fa1e-\U000e0000\U000e0002-\U000e001f\U000e0080-\U000e00ff\U000e01f0-\U000effff\U000ffffe-\U000fffff\U0010fffe-\U0010ffff'

Co = '\ue000-\uf8ff\U000f0000-\U000ffffd\U00100000-\U0010fffd'

Cs = '\ud800-\udbff\\\udc00\udc01-\udfff'

Ll = 'a-z\xb5\xdf-\xf6\xf8-\xff\u0101\u0103\u0105\u0107\u0109\u010b\u010d\u010f\u0111\u0113\u0115\u0117\u0119\u011b\u011d\u011f\u0121\u0123\u0125\u0127\u0129\u012b\u012d\u012f\u0131\u0133\u0135\u0137-\u0138\u013a\u013c\u013e\u0140\u0142\u0144\u0146\u0148-\u0149\u014b\u014d\u014f\u0151\u0153\u0155\u0157\u0159\u015b\u015d\u015f\u0161\u0163\u0165\u0167\u0169\u016b\u016d\u016f\u0171\u0173\u0175\u0177\u017a\u017c\u017e-\u0180\u0183\u0185\u0188\u018c-\u018d\u0192\u0195\u0199-\u019b\u019e\u01a1\u01a3\u01a5\u01a8\u01aa-\u01ab\u01ad\u01b0\u01b4\u01b6\u01b9-\u01ba\u01bd-\u01bf\u01c6\u01c9\u01cc\u01ce\u01d0\u01d2\u01d4\u01d6\u01d8\u01da\u01dc-\u01dd\u01df\u01e1\u01e3\u01e5\u01e7\u01e9\u01eb\u01ed\u01ef-\u01f0\u01f3\u01f5\u01f9\u01fb\u01fd\u01ff\u0201\u0203\u0205\u0207\u0209\u020b\u020d\u020f\u0211\u0213\u0215\u0217\u0219\u021b\u021d\u021f\u0221\u0223\u0225\u0227\u0229\u022b\u022d\u022f\u0231\u0233-\u0239\u023c\u023f-\u0240\u0242\u0247\u0249\u024b\u024d\u024f-\u0293\u0295-\u02af\u0371\u0373\u0377\u037b-\u037d\u0390\u03ac-\u03ce\u03d0-\u03d1\u03d5-\u03d7\u03d9\u03db\u03dd\u03df\u03e1\u03e3\u03e5\u03e7\u03e9\u03eb\u03ed\u03ef-\u03f3\u03f5\u03f8\u03fb-\u03fc\u0430-\u045f\u0461\u0463\u0465\u0467\u0469\u046b\u046d\u046f\u0471\u0473\u0475\u0477\u0479\u047b\u047d\u047f\u0481\u048b\u048d\u048f\u0491\u0493\u0495\u0497\u0499\u049b\u049d\u049f\u04a1\u04a3\u04a5\u04a7\u04a9\u04ab\u04ad\u04af\u04b1\u04b3\u04b5\u04b7\u04b9\u04bb\u04bd\u04bf\u04c2\u04c4\u04c6\u04c8\u04ca\u04cc\u04ce-\u04cf\u04d1\u04d3\u04d5\u04d7\u04d9\u04db\u04dd\u04df\u04e1\u04e3\u04e5\u04e7\u04e9\u04eb\u04ed\u04ef\u04f1\u04f3\u04f5\u04f7\u04f9\u04fb\u04fd\u04ff\u0501\u0503\u0505\u0507\u0509\u050b\u050d\u050f\u0511\u0513\u0515\u0517\u0519\u051b\u051d\u051f\u0521\u0523\u0525\u0527\u0529\u052b\u052d\u052f\u0560-\u0588\u10d0-\u10fa\u10fd-\u10ff\u13f8-\u13fd\u1c80-\u1c88\u1d00-\u1d2b\u1d6b-\u1d77\u1d79-\u1d9a\u1e01\u1e03\u1e05\u1e07\u1e09\u1e0b\u1e0d\u1e0f\u1e11\u1e13\u1e15\u1e17\u1e19\u1e1b\u1e1d\u1e1f\u1e21\u1e23\u1e25\u1e27\u1e29\u1e2b\u1e2d\u1e2f\u1e31\u1e33\u1e35\u1e37\u1e39\u1e3b\u1e3d\u1e3f\u1e41\u1e43\u1e45\u1e47\u1e49\u1e4b\u1e4d\u1e4f\u1e51\u1e53\u1e55\u1e57\u1e59\u1e5b\u1e5d\u1e5f\u1e61\u1e63\u1e65\u1e67\u1e69\u1e6b\u1e6d\u1e6f\u1e71\u1e73\u1e75\u1e77\u1e79\u1e7b\u1e7d\u1e7f\u1e81\u1e83\u1e85\u1e87\u1e89\u1e8b\u1e8d\u1e8f\u1e91\u1e93\u1e95-\u1e9d\u1e9f\u1ea1\u1ea3\u1ea5\u1ea7\u1ea9\u1eab\u1ead\u1eaf\u1eb1\u1eb3\u1eb5\u1eb7\u1eb9\u1ebb\u1ebd\u1ebf\u1ec1\u1ec3\u1ec5\u1ec7\u1ec9\u1ecb\u1ecd\u1ecf\u1ed1\u1ed3\u1ed5\u1ed7\u1ed9\u1edb\u1edd\u1edf\u1ee1\u1ee3\u1ee5\u1ee7\u1ee9\u1eeb\u1eed\u1eef\u1ef1\u1ef3\u1ef5\u1ef7\u1ef9\u1efb\u1efd\u1eff-\u1f07\u1f10-\u1f15\u1f20-\u1f27\u1f30-\u1f37\u1f40-\u1f45\u1f50-\u1f57\u1f60-\u1f67\u1f70-\u1f7d\u1f80-\u1f87\u1f90-\u1f97\u1fa0-\u1fa7\u1fb0-\u1fb4\u1fb6-\u1fb7\u1fbe\u1fc2-\u1fc4\u1fc6-\u1fc7\u1fd0-\u1fd3\u1fd6-\u1fd7\u1fe0-\u1fe7\u1ff2-\u1ff4\u1ff6-\u1ff7\u210a\u210e-\u210f\u2113\u212f\u2134\u2139\u213c-\u213d\u2146-\u2149\u214e\u2184\u2c30-\u2c5e\u2c61\u2c65-\u2c66\u2c68\u2c6a\u2c6c\u2c71\u2c73-\u2c74\u2c76-\u2c7b\u2c81\u2c83\u2c85\u2c87\u2c89\u2c8b\u2c8d\u2c8f\u2c91\u2c93\u2c95\u2c97\u2c99\u2c9b\u2c9d\u2c9f\u2ca1\u2ca3\u2ca5\u2ca7\u2ca9\u2cab\u2cad\u2caf\u2cb1\u2cb3\u2cb5\u2cb7\u2cb9\u2cbb\u2cbd\u2cbf\u2cc1\u2cc3\u2cc5\u2cc7\u2cc9\u2ccb\u2ccd\u2ccf\u2cd1\u2cd3\u2cd5\u2cd7\u2cd9\u2cdb\u2cdd\u2cdf\u2ce1\u2ce3-\u2ce4\u2cec\u2cee\u2cf3\u2d00-\u2d25\u2d27\u2d2d\ua641\ua643\ua645\ua647\ua649\ua64b\ua64d\ua64f\ua651\ua653\ua655\ua657\ua659\ua65b\ua65d\ua65f\ua661\ua663\ua665\ua667\ua669\ua66b\ua66d\ua681\ua683\ua685\ua687\ua689\ua68b\ua68d\ua68f\ua691\ua693\ua695\ua697\ua699\ua69b\ua723\ua725\ua727\ua729\ua72b\ua72d\ua72f-\ua731\ua733\ua735\ua737\ua739\ua73b\ua73d\ua73f\ua741\ua743\ua745\ua747\ua749\ua74b\ua74d\ua74f\ua751\ua753\ua755\ua757\ua759\ua75b\ua75d\ua75f\ua761\ua763\ua765\ua767\ua769\ua76b\ua76d\ua76f\ua771-\ua778\ua77a\ua77c\ua77f\ua781\ua783\ua785\ua787\ua78c\ua78e\ua791\ua793-\ua795\ua797\ua799\ua79b\ua79d\ua79f\ua7a1\ua7a3\ua7a5\ua7a7\ua7a9\ua7af\ua7b5\ua7b7\ua7b9\ua7fa\uab30-\uab5a\uab60-\uab65\uab70-\uabbf\ufb00-\ufb06\ufb13-\ufb17\uff41-\uff5a\U00010428-\U0001044f\U000104d8-\U000104fb\U00010cc0-\U00010cf2\U000118c0-\U000118df\U00016e60-\U00016e7f\U0001d41a-\U0001d433\U0001d44e-\U0001d454\U0001d456-\U0001d467\U0001d482-\U0001d49b\U0001d4b6-\U0001d4b9\U0001d4bb\U0001d4bd-\U0001d4c3\U0001d4c5-\U0001d4cf\U0001d4ea-\U0001d503\U0001d51e-\U0001d537\U0001d552-\U0001d56b\U0001d586-\U0001d59f\U0001d5ba-\U0001d5d3\U0001d5ee-\U0001d607\U0001d622-\U0001d63b\U0001d656-\U0001d66f\U0001d68a-\U0001d6a5\U0001d6c2-\U0001d6da\U0001d6dc-\U0001d6e1\U0001d6fc-\U0001d714\U0001d716-\U0001d71b\U0001d736-\U0001d74e\U0001d750-\U0001d755\U0001d770-\U0001d788\U0001d78a-\U0001d78f\U0001d7aa-\U0001d7c2\U0001d7c4-\U0001d7c9\U0001d7cb\U0001e922-\U0001e943'

Lm = '\u02b0-\u02c1\u02c6-\u02d1\u02e0-\u02e4\u02ec\u02ee\u0374\u037a\u0559\u0640\u06e5-\u06e6\u07f4-\u07f5\u07fa\u081a\u0824\u0828\u0971\u0e46\u0ec6\u10fc\u17d7\u1843\u1aa7\u1c78-\u1c7d\u1d2c-\u1d6a\u1d78\u1d9b-\u1dbf\u2071\u207f\u2090-\u209c\u2c7c-\u2c7d\u2d6f\u2e2f\u3005\u3031-\u3035\u303b\u309d-\u309e\u30fc-\u30fe\ua015\ua4f8-\ua4fd\ua60c\ua67f\ua69c-\ua69d\ua717-\ua71f\ua770\ua788\ua7f8-\ua7f9\ua9cf\ua9e6\uaa70\uaadd\uaaf3-\uaaf4\uab5c-\uab5f\uff70\uff9e-\uff9f\U00016b40-\U00016b43\U00016f93-\U00016f9f\U00016fe0-\U00016fe1'

Lo = '\xaa\xba\u01bb\u01c0-\u01c3\u0294\u05d0-\u05ea\u05ef-\u05f2\u0620-\u063f\u0641-\u064a\u066e-\u066f\u0671-\u06d3\u06d5\u06ee-\u06ef\u06fa-\u06fc\u06ff\u0710\u0712-\u072f\u074d-\u07a5\u07b1\u07ca-\u07ea\u0800-\u0815\u0840-\u0858\u0860-\u086a\u08a0-\u08b4\u08b6-\u08bd\u0904-\u0939\u093d\u0950\u0958-\u0961\u0972-\u0980\u0985-\u098c\u098f-\u0990\u0993-\u09a8\u09aa-\u09b0\u09b2\u09b6-\u09b9\u09bd\u09ce\u09dc-\u09dd\u09df-\u09e1\u09f0-\u09f1\u09fc\u0a05-\u0a0a\u0a0f-\u0a10\u0a13-\u0a28\u0a2a-\u0a30\u0a32-\u0a33\u0a35-\u0a36\u0a38-\u0a39\u0a59-\u0a5c\u0a5e\u0a72-\u0a74\u0a85-\u0a8d\u0a8f-\u0a91\u0a93-\u0aa8\u0aaa-\u0ab0\u0ab2-\u0ab3\u0ab5-\u0ab9\u0abd\u0ad0\u0ae0-\u0ae1\u0af9\u0b05-\u0b0c\u0b0f-\u0b10\u0b13-\u0b28\u0b2a-\u0b30\u0b32-\u0b33\u0b35-\u0b39\u0b3d\u0b5c-\u0b5d\u0b5f-\u0b61\u0b71\u0b83\u0b85-\u0b8a\u0b8e-\u0b90\u0b92-\u0b95\u0b99-\u0b9a\u0b9c\u0b9e-\u0b9f\u0ba3-\u0ba4\u0ba8-\u0baa\u0bae-\u0bb9\u0bd0\u0c05-\u0c0c\u0c0e-\u0c10\u0c12-\u0c28\u0c2a-\u0c39\u0c3d\u0c58-\u0c5a\u0c60-\u0c61\u0c80\u0c85-\u0c8c\u0c8e-\u0c90\u0c92-\u0ca8\u0caa-\u0cb3\u0cb5-\u0cb9\u0cbd\u0cde\u0ce0-\u0ce1\u0cf1-\u0cf2\u0d05-\u0d0c\u0d0e-\u0d10\u0d12-\u0d3a\u0d3d\u0d4e\u0d54-\u0d56\u0d5f-\u0d61\u0d7a-\u0d7f\u0d85-\u0d96\u0d9a-\u0db1\u0db3-\u0dbb\u0dbd\u0dc0-\u0dc6\u0e01-\u0e30\u0e32-\u0e33\u0e40-\u0e45\u0e81-\u0e82\u0e84\u0e87-\u0e88\u0e8a\u0e8d\u0e94-\u0e97\u0e99-\u0e9f\u0ea1-\u0ea3\u0ea5\u0ea7\u0eaa-\u0eab\u0ead-\u0eb0\u0eb2-\u0eb3\u0ebd\u0ec0-\u0ec4\u0edc-\u0edf\u0f00\u0f40-\u0f47\u0f49-\u0f6c\u0f88-\u0f8c\u1000-\u102a\u103f\u1050-\u1055\u105a-\u105d\u1061\u1065-\u1066\u106e-\u1070\u1075-\u1081\u108e\u1100-\u1248\u124a-\u124d\u1250-\u1256\u1258\u125a-\u125d\u1260-\u1288\u128a-\u128d\u1290-\u12b0\u12b2-\u12b5\u12b8-\u12be\u12c0\u12c2-\u12c5\u12c8-\u12d6\u12d8-\u1310\u1312-\u1315\u1318-\u135a\u1380-\u138f\u1401-\u166c\u166f-\u167f\u1681-\u169a\u16a0-\u16ea\u16f1-\u16f8\u1700-\u170c\u170e-\u1711\u1720-\u1731\u1740-\u1751\u1760-\u176c\u176e-\u1770\u1780-\u17b3\u17dc\u1820-\u1842\u1844-\u1878\u1880-\u1884\u1887-\u18a8\u18aa\u18b0-\u18f5\u1900-\u191e\u1950-\u196d\u1970-\u1974\u1980-\u19ab\u19b0-\u19c9\u1a00-\u1a16\u1a20-\u1a54\u1b05-\u1b33\u1b45-\u1b4b\u1b83-\u1ba0\u1bae-\u1baf\u1bba-\u1be5\u1c00-\u1c23\u1c4d-\u1c4f\u1c5a-\u1c77\u1ce9-\u1cec\u1cee-\u1cf1\u1cf5-\u1cf6\u2135-\u2138\u2d30-\u2d67\u2d80-\u2d96\u2da0-\u2da6\u2da8-\u2dae\u2db0-\u2db6\u2db8-\u2dbe\u2dc0-\u2dc6\u2dc8-\u2dce\u2dd0-\u2dd6\u2dd8-\u2dde\u3006\u303c\u3041-\u3096\u309f\u30a1-\u30fa\u30ff\u3105-\u312f\u3131-\u318e\u31a0-\u31ba\u31f0-\u31ff\u3400-\u4db5\u4e00-\u9fef\ua000-\ua014\ua016-\ua48c\ua4d0-\ua4f7\ua500-\ua60b\ua610-\ua61f\ua62a-\ua62b\ua66e\ua6a0-\ua6e5\ua78f\ua7f7\ua7fb-\ua801\ua803-\ua805\ua807-\ua80a\ua80c-\ua822\ua840-\ua873\ua882-\ua8b3\ua8f2-\ua8f7\ua8fb\ua8fd-\ua8fe\ua90a-\ua925\ua930-\ua946\ua960-\ua97c\ua984-\ua9b2\ua9e0-\ua9e4\ua9e7-\ua9ef\ua9fa-\ua9fe\uaa00-\uaa28\uaa40-\uaa42\uaa44-\uaa4b\uaa60-\uaa6f\uaa71-\uaa76\uaa7a\uaa7e-\uaaaf\uaab1\uaab5-\uaab6\uaab9-\uaabd\uaac0\uaac2\uaadb-\uaadc\uaae0-\uaaea\uaaf2\uab01-\uab06\uab09-\uab0e\uab11-\uab16\uab20-\uab26\uab28-\uab2e\uabc0-\uabe2\uac00-\ud7a3\ud7b0-\ud7c6\ud7cb-\ud7fb\uf900-\ufa6d\ufa70-\ufad9\ufb1d\ufb1f-\ufb28\ufb2a-\ufb36\ufb38-\ufb3c\ufb3e\ufb40-\ufb41\ufb43-\ufb44\ufb46-\ufbb1\ufbd3-\ufd3d\ufd50-\ufd8f\ufd92-\ufdc7\ufdf0-\ufdfb\ufe70-\ufe74\ufe76-\ufefc\uff66-\uff6f\uff71-\uff9d\uffa0-\uffbe\uffc2-\uffc7\uffca-\uffcf\uffd2-\uffd7\uffda-\uffdc\U00010000-\U0001000b\U0001000d-\U00010026\U00010028-\U0001003a\U0001003c-\U0001003d\U0001003f-\U0001004d\U00010050-\U0001005d\U00010080-\U000100fa\U00010280-\U0001029c\U000102a0-\U000102d0\U00010300-\U0001031f\U0001032d-\U00010340\U00010342-\U00010349\U00010350-\U00010375\U00010380-\U0001039d\U000103a0-\U000103c3\U000103c8-\U000103cf\U00010450-\U0001049d\U00010500-\U00010527\U00010530-\U00010563\U00010600-\U00010736\U00010740-\U00010755\U00010760-\U00010767\U00010800-\U00010805\U00010808\U0001080a-\U00010835\U00010837-\U00010838\U0001083c\U0001083f-\U00010855\U00010860-\U00010876\U00010880-\U0001089e\U000108e0-\U000108f2\U000108f4-\U000108f5\U00010900-\U00010915\U00010920-\U00010939\U00010980-\U000109b7\U000109be-\U000109bf\U00010a00\U00010a10-\U00010a13\U00010a15-\U00010a17\U00010a19-\U00010a35\U00010a60-\U00010a7c\U00010a80-\U00010a9c\U00010ac0-\U00010ac7\U00010ac9-\U00010ae4\U00010b00-\U00010b35\U00010b40-\U00010b55\U00010b60-\U00010b72\U00010b80-\U00010b91\U00010c00-\U00010c48\U00010d00-\U00010d23\U00010f00-\U00010f1c\U00010f27\U00010f30-\U00010f45\U00011003-\U00011037\U00011083-\U000110af\U000110d0-\U000110e8\U00011103-\U00011126\U00011144\U00011150-\U00011172\U00011176\U00011183-\U000111b2\U000111c1-\U000111c4\U000111da\U000111dc\U00011200-\U00011211\U00011213-\U0001122b\U00011280-\U00011286\U00011288\U0001128a-\U0001128d\U0001128f-\U0001129d\U0001129f-\U000112a8\U000112b0-\U000112de\U00011305-\U0001130c\U0001130f-\U00011310\U00011313-\U00011328\U0001132a-\U00011330\U00011332-\U00011333\U00011335-\U00011339\U0001133d\U00011350\U0001135d-\U00011361\U00011400-\U00011434\U00011447-\U0001144a\U00011480-\U000114af\U000114c4-\U000114c5\U000114c7\U00011580-\U000115ae\U000115d8-\U000115db\U00011600-\U0001162f\U00011644\U00011680-\U000116aa\U00011700-\U0001171a\U00011800-\U0001182b\U000118ff\U00011a00\U00011a0b-\U00011a32\U00011a3a\U00011a50\U00011a5c-\U00011a83\U00011a86-\U00011a89\U00011a9d\U00011ac0-\U00011af8\U00011c00-\U00011c08\U00011c0a-\U00011c2e\U00011c40\U00011c72-\U00011c8f\U00011d00-\U00011d06\U00011d08-\U00011d09\U00011d0b-\U00011d30\U00011d46\U00011d60-\U00011d65\U00011d67-\U00011d68\U00011d6a-\U00011d89\U00011d98\U00011ee0-\U00011ef2\U00012000-\U00012399\U00012480-\U00012543\U00013000-\U0001342e\U00014400-\U00014646\U00016800-\U00016a38\U00016a40-\U00016a5e\U00016ad0-\U00016aed\U00016b00-\U00016b2f\U00016b63-\U00016b77\U00016b7d-\U00016b8f\U00016f00-\U00016f44\U00016f50\U00017000-\U000187f1\U00018800-\U00018af2\U0001b000-\U0001b11e\U0001b170-\U0001b2fb\U0001bc00-\U0001bc6a\U0001bc70-\U0001bc7c\U0001bc80-\U0001bc88\U0001bc90-\U0001bc99\U0001e800-\U0001e8c4\U0001ee00-\U0001ee03\U0001ee05-\U0001ee1f\U0001ee21-\U0001ee22\U0001ee24\U0001ee27\U0001ee29-\U0001ee32\U0001ee34-\U0001ee37\U0001ee39\U0001ee3b\U0001ee42\U0001ee47\U0001ee49\U0001ee4b\U0001ee4d-\U0001ee4f\U0001ee51-\U0001ee52\U0001ee54\U0001ee57\U0001ee59\U0001ee5b\U0001ee5d\U0001ee5f\U0001ee61-\U0001ee62\U0001ee64\U0001ee67-\U0001ee6a\U0001ee6c-\U0001ee72\U0001ee74-\U0001ee77\U0001ee79-\U0001ee7c\U0001ee7e\U0001ee80-\U0001ee89\U0001ee8b-\U0001ee9b\U0001eea1-\U0001eea3\U0001eea5-\U0001eea9\U0001eeab-\U0001eebb\U00020000-\U0002a6d6\U0002a700-\U0002b734\U0002b740-\U0002b81d\U0002b820-\U0002cea1\U0002ceb0-\U0002ebe0\U0002f800-\U0002fa1d'

Lt = '\u01c5\u01c8\u01cb\u01f2\u1f88-\u1f8f\u1f98-\u1f9f\u1fa8-\u1faf\u1fbc\u1fcc\u1ffc'

Lu = 'A-Z\xc0-\xd6\xd8-\xde\u0100\u0102\u0104\u0106\u0108\u010a\u010c\u010e\u0110\u0112\u0114\u0116\u0118\u011a\u011c\u011e\u0120\u0122\u0124\u0126\u0128\u012a\u012c\u012e\u0130\u0132\u0134\u0136\u0139\u013b\u013d\u013f\u0141\u0143\u0145\u0147\u014a\u014c\u014e\u0150\u0152\u0154\u0156\u0158\u015a\u015c\u015e\u0160\u0162\u0164\u0166\u0168\u016a\u016c\u016e\u0170\u0172\u0174\u0176\u0178-\u0179\u017b\u017d\u0181-\u0182\u0184\u0186-\u0187\u0189-\u018b\u018e-\u0191\u0193-\u0194\u0196-\u0198\u019c-\u019d\u019f-\u01a0\u01a2\u01a4\u01a6-\u01a7\u01a9\u01ac\u01ae-\u01af\u01b1-\u01b3\u01b5\u01b7-\u01b8\u01bc\u01c4\u01c7\u01ca\u01cd\u01cf\u01d1\u01d3\u01d5\u01d7\u01d9\u01db\u01de\u01e0\u01e2\u01e4\u01e6\u01e8\u01ea\u01ec\u01ee\u01f1\u01f4\u01f6-\u01f8\u01fa\u01fc\u01fe\u0200\u0202\u0204\u0206\u0208\u020a\u020c\u020e\u0210\u0212\u0214\u0216\u0218\u021a\u021c\u021e\u0220\u0222\u0224\u0226\u0228\u022a\u022c\u022e\u0230\u0232\u023a-\u023b\u023d-\u023e\u0241\u0243-\u0246\u0248\u024a\u024c\u024e\u0370\u0372\u0376\u037f\u0386\u0388-\u038a\u038c\u038e-\u038f\u0391-\u03a1\u03a3-\u03ab\u03cf\u03d2-\u03d4\u03d8\u03da\u03dc\u03de\u03e0\u03e2\u03e4\u03e6\u03e8\u03ea\u03ec\u03ee\u03f4\u03f7\u03f9-\u03fa\u03fd-\u042f\u0460\u0462\u0464\u0466\u0468\u046a\u046c\u046e\u0470\u0472\u0474\u0476\u0478\u047a\u047c\u047e\u0480\u048a\u048c\u048e\u0490\u0492\u0494\u0496\u0498\u049a\u049c\u049e\u04a0\u04a2\u04a4\u04a6\u04a8\u04aa\u04ac\u04ae\u04b0\u04b2\u04b4\u04b6\u04b8\u04ba\u04bc\u04be\u04c0-\u04c1\u04c3\u04c5\u04c7\u04c9\u04cb\u04cd\u04d0\u04d2\u04d4\u04d6\u04d8\u04da\u04dc\u04de\u04e0\u04e2\u04e4\u04e6\u04e8\u04ea\u04ec\u04ee\u04f0\u04f2\u04f4\u04f6\u04f8\u04fa\u04fc\u04fe\u0500\u0502\u0504\u0506\u0508\u050a\u050c\u050e\u0510\u0512\u0514\u0516\u0518\u051a\u051c\u051e\u0520\u0522\u0524\u0526\u0528\u052a\u052c\u052e\u0531-\u0556\u10a0-\u10c5\u10c7\u10cd\u13a0-\u13f5\u1c90-\u1cba\u1cbd-\u1cbf\u1e00\u1e02\u1e04\u1e06\u1e08\u1e0a\u1e0c\u1e0e\u1e10\u1e12\u1e14\u1e16\u1e18\u1e1a\u1e1c\u1e1e\u1e20\u1e22\u1e24\u1e26\u1e28\u1e2a\u1e2c\u1e2e\u1e30\u1e32\u1e34\u1e36\u1e38\u1e3a\u1e3c\u1e3e\u1e40\u1e42\u1e44\u1e46\u1e48\u1e4a\u1e4c\u1e4e\u1e50\u1e52\u1e54\u1e56\u1e58\u1e5a\u1e5c\u1e5e\u1e60\u1e62\u1e64\u1e66\u1e68\u1e6a\u1e6c\u1e6e\u1e70\u1e72\u1e74\u1e76\u1e78\u1e7a\u1e7c\u1e7e\u1e80\u1e82\u1e84\u1e86\u1e88\u1e8a\u1e8c\u1e8e\u1e90\u1e92\u1e94\u1e9e\u1ea0\u1ea2\u1ea4\u1ea6\u1ea8\u1eaa\u1eac\u1eae\u1eb0\u1eb2\u1eb4\u1eb6\u1eb8\u1eba\u1ebc\u1ebe\u1ec0\u1ec2\u1ec4\u1ec6\u1ec8\u1eca\u1ecc\u1ece\u1ed0\u1ed2\u1ed4\u1ed6\u1ed8\u1eda\u1edc\u1ede\u1ee0\u1ee2\u1ee4\u1ee6\u1ee8\u1eea\u1eec\u1eee\u1ef0\u1ef2\u1ef4\u1ef6\u1ef8\u1efa\u1efc\u1efe\u1f08-\u1f0f\u1f18-\u1f1d\u1f28-\u1f2f\u1f38-\u1f3f\u1f48-\u1f4d\u1f59\u1f5b\u1f5d\u1f5f\u1f68-\u1f6f\u1fb8-\u1fbb\u1fc8-\u1fcb\u1fd8-\u1fdb\u1fe8-\u1fec\u1ff8-\u1ffb\u2102\u2107\u210b-\u210d\u2110-\u2112\u2115\u2119-\u211d\u2124\u2126\u2128\u212a-\u212d\u2130-\u2133\u213e-\u213f\u2145\u2183\u2c00-\u2c2e\u2c60\u2c62-\u2c64\u2c67\u2c69\u2c6b\u2c6d-\u2c70\u2c72\u2c75\u2c7e-\u2c80\u2c82\u2c84\u2c86\u2c88\u2c8a\u2c8c\u2c8e\u2c90\u2c92\u2c94\u2c96\u2c98\u2c9a\u2c9c\u2c9e\u2ca0\u2ca2\u2ca4\u2ca6\u2ca8\u2caa\u2cac\u2cae\u2cb0\u2cb2\u2cb4\u2cb6\u2cb8\u2cba\u2cbc\u2cbe\u2cc0\u2cc2\u2cc4\u2cc6\u2cc8\u2cca\u2ccc\u2cce\u2cd0\u2cd2\u2cd4\u2cd6\u2cd8\u2cda\u2cdc\u2cde\u2ce0\u2ce2\u2ceb\u2ced\u2cf2\ua640\ua642\ua644\ua646\ua648\ua64a\ua64c\ua64e\ua650\ua652\ua654\ua656\ua658\ua65a\ua65c\ua65e\ua660\ua662\ua664\ua666\ua668\ua66a\ua66c\ua680\ua682\ua684\ua686\ua688\ua68a\ua68c\ua68e\ua690\ua692\ua694\ua696\ua698\ua69a\ua722\ua724\ua726\ua728\ua72a\ua72c\ua72e\ua732\ua734\ua736\ua738\ua73a\ua73c\ua73e\ua740\ua742\ua744\ua746\ua748\ua74a\ua74c\ua74e\ua750\ua752\ua754\ua756\ua758\ua75a\ua75c\ua75e\ua760\ua762\ua764\ua766\ua768\ua76a\ua76c\ua76e\ua779\ua77b\ua77d-\ua77e\ua780\ua782\ua784\ua786\ua78b\ua78d\ua790\ua792\ua796\ua798\ua79a\ua79c\ua79e\ua7a0\ua7a2\ua7a4\ua7a6\ua7a8\ua7aa-\ua7ae\ua7b0-\ua7b4\ua7b6\ua7b8\uff21-\uff3a\U00010400-\U00010427\U000104b0-\U000104d3\U00010c80-\U00010cb2\U000118a0-\U000118bf\U00016e40-\U00016e5f\U0001d400-\U0001d419\U0001d434-\U0001d44d\U0001d468-\U0001d481\U0001d49c\U0001d49e-\U0001d49f\U0001d4a2\U0001d4a5-\U0001d4a6\U0001d4a9-\U0001d4ac\U0001d4ae-\U0001d4b5\U0001d4d0-\U0001d4e9\U0001d504-\U0001d505\U0001d507-\U0001d50a\U0001d50d-\U0001d514\U0001d516-\U0001d51c\U0001d538-\U0001d539\U0001d53b-\U0001d53e\U0001d540-\U0001d544\U0001d546\U0001d54a-\U0001d550\U0001d56c-\U0001d585\U0001d5a0-\U0001d5b9\U0001d5d4-\U0001d5ed\U0001d608-\U0001d621\U0001d63c-\U0001d655\U0001d670-\U0001d689\U0001d6a8-\U0001d6c0\U0001d6e2-\U0001d6fa\U0001d71c-\U0001d734\U0001d756-\U0001d76e\U0001d790-\U0001d7a8\U0001d7ca\U0001e900-\U0001e921'

Mc = '\u0903\u093b\u093e-\u0940\u0949-\u094c\u094e-\u094f\u0982-\u0983\u09be-\u09c0\u09c7-\u09c8\u09cb-\u09cc\u09d7\u0a03\u0a3e-\u0a40\u0a83\u0abe-\u0ac0\u0ac9\u0acb-\u0acc\u0b02-\u0b03\u0b3e\u0b40\u0b47-\u0b48\u0b4b-\u0b4c\u0b57\u0bbe-\u0bbf\u0bc1-\u0bc2\u0bc6-\u0bc8\u0bca-\u0bcc\u0bd7\u0c01-\u0c03\u0c41-\u0c44\u0c82-\u0c83\u0cbe\u0cc0-\u0cc4\u0cc7-\u0cc8\u0cca-\u0ccb\u0cd5-\u0cd6\u0d02-\u0d03\u0d3e-\u0d40\u0d46-\u0d48\u0d4a-\u0d4c\u0d57\u0d82-\u0d83\u0dcf-\u0dd1\u0dd8-\u0ddf\u0df2-\u0df3\u0f3e-\u0f3f\u0f7f\u102b-\u102c\u1031\u1038\u103b-\u103c\u1056-\u1057\u1062-\u1064\u1067-\u106d\u1083-\u1084\u1087-\u108c\u108f\u109a-\u109c\u17b6\u17be-\u17c5\u17c7-\u17c8\u1923-\u1926\u1929-\u192b\u1930-\u1931\u1933-\u1938\u1a19-\u1a1a\u1a55\u1a57\u1a61\u1a63-\u1a64\u1a6d-\u1a72\u1b04\u1b35\u1b3b\u1b3d-\u1b41\u1b43-\u1b44\u1b82\u1ba1\u1ba6-\u1ba7\u1baa\u1be7\u1bea-\u1bec\u1bee\u1bf2-\u1bf3\u1c24-\u1c2b\u1c34-\u1c35\u1ce1\u1cf2-\u1cf3\u1cf7\u302e-\u302f\ua823-\ua824\ua827\ua880-\ua881\ua8b4-\ua8c3\ua952-\ua953\ua983\ua9b4-\ua9b5\ua9ba-\ua9bb\ua9bd-\ua9c0\uaa2f-\uaa30\uaa33-\uaa34\uaa4d\uaa7b\uaa7d\uaaeb\uaaee-\uaaef\uaaf5\uabe3-\uabe4\uabe6-\uabe7\uabe9-\uabea\uabec\U00011000\U00011002\U00011082\U000110b0-\U000110b2\U000110b7-\U000110b8\U0001112c\U00011145-\U00011146\U00011182\U000111b3-\U000111b5\U000111bf-\U000111c0\U0001122c-\U0001122e\U00011232-\U00011233\U00011235\U000112e0-\U000112e2\U00011302-\U00011303\U0001133e-\U0001133f\U00011341-\U00011344\U00011347-\U00011348\U0001134b-\U0001134d\U00011357\U00011362-\U00011363\U00011435-\U00011437\U00011440-\U00011441\U00011445\U000114b0-\U000114b2\U000114b9\U000114bb-\U000114be\U000114c1\U000115af-\U000115b1\U000115b8-\U000115bb\U000115be\U00011630-\U00011632\U0001163b-\U0001163c\U0001163e\U000116ac\U000116ae-\U000116af\U000116b6\U00011720-\U00011721\U00011726\U0001182c-\U0001182e\U00011838\U00011a39\U00011a57-\U00011a58\U00011a97\U00011c2f\U00011c3e\U00011ca9\U00011cb1\U00011cb4\U00011d8a-\U00011d8e\U00011d93-\U00011d94\U00011d96\U00011ef5-\U00011ef6\U00016f51-\U00016f7e\U0001d165-\U0001d166\U0001d16d-\U0001d172'

Me = '\u0488-\u0489\u1abe\u20dd-\u20e0\u20e2-\u20e4\ua670-\ua672'

Mn = '\u0300-\u036f\u0483-\u0487\u0591-\u05bd\u05bf\u05c1-\u05c2\u05c4-\u05c5\u05c7\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06dc\u06df-\u06e4\u06e7-\u06e8\u06ea-\u06ed\u0711\u0730-\u074a\u07a6-\u07b0\u07eb-\u07f3\u07fd\u0816-\u0819\u081b-\u0823\u0825-\u0827\u0829-\u082d\u0859-\u085b\u08d3-\u08e1\u08e3-\u0902\u093a\u093c\u0941-\u0948\u094d\u0951-\u0957\u0962-\u0963\u0981\u09bc\u09c1-\u09c4\u09cd\u09e2-\u09e3\u09fe\u0a01-\u0a02\u0a3c\u0a41-\u0a42\u0a47-\u0a48\u0a4b-\u0a4d\u0a51\u0a70-\u0a71\u0a75\u0a81-\u0a82\u0abc\u0ac1-\u0ac5\u0ac7-\u0ac8\u0acd\u0ae2-\u0ae3\u0afa-\u0aff\u0b01\u0b3c\u0b3f\u0b41-\u0b44\u0b4d\u0b56\u0b62-\u0b63\u0b82\u0bc0\u0bcd\u0c00\u0c04\u0c3e-\u0c40\u0c46-\u0c48\u0c4a-\u0c4d\u0c55-\u0c56\u0c62-\u0c63\u0c81\u0cbc\u0cbf\u0cc6\u0ccc-\u0ccd\u0ce2-\u0ce3\u0d00-\u0d01\u0d3b-\u0d3c\u0d41-\u0d44\u0d4d\u0d62-\u0d63\u0dca\u0dd2-\u0dd4\u0dd6\u0e31\u0e34-\u0e3a\u0e47-\u0e4e\u0eb1\u0eb4-\u0eb9\u0ebb-\u0ebc\u0ec8-\u0ecd\u0f18-\u0f19\u0f35\u0f37\u0f39\u0f71-\u0f7e\u0f80-\u0f84\u0f86-\u0f87\u0f8d-\u0f97\u0f99-\u0fbc\u0fc6\u102d-\u1030\u1032-\u1037\u1039-\u103a\u103d-\u103e\u1058-\u1059\u105e-\u1060\u1071-\u1074\u1082\u1085-\u1086\u108d\u109d\u135d-\u135f\u1712-\u1714\u1732-\u1734\u1752-\u1753\u1772-\u1773\u17b4-\u17b5\u17b7-\u17bd\u17c6\u17c9-\u17d3\u17dd\u180b-\u180d\u1885-\u1886\u18a9\u1920-\u1922\u1927-\u1928\u1932\u1939-\u193b\u1a17-\u1a18\u1a1b\u1a56\u1a58-\u1a5e\u1a60\u1a62\u1a65-\u1a6c\u1a73-\u1a7c\u1a7f\u1ab0-\u1abd\u1b00-\u1b03\u1b34\u1b36-\u1b3a\u1b3c\u1b42\u1b6b-\u1b73\u1b80-\u1b81\u1ba2-\u1ba5\u1ba8-\u1ba9\u1bab-\u1bad\u1be6\u1be8-\u1be9\u1bed\u1bef-\u1bf1\u1c2c-\u1c33\u1c36-\u1c37\u1cd0-\u1cd2\u1cd4-\u1ce0\u1ce2-\u1ce8\u1ced\u1cf4\u1cf8-\u1cf9\u1dc0-\u1df9\u1dfb-\u1dff\u20d0-\u20dc\u20e1\u20e5-\u20f0\u2cef-\u2cf1\u2d7f\u2de0-\u2dff\u302a-\u302d\u3099-\u309a\ua66f\ua674-\ua67d\ua69e-\ua69f\ua6f0-\ua6f1\ua802\ua806\ua80b\ua825-\ua826\ua8c4-\ua8c5\ua8e0-\ua8f1\ua8ff\ua926-\ua92d\ua947-\ua951\ua980-\ua982\ua9b3\ua9b6-\ua9b9\ua9bc\ua9e5\uaa29-\uaa2e\uaa31-\uaa32\uaa35-\uaa36\uaa43\uaa4c\uaa7c\uaab0\uaab2-\uaab4\uaab7-\uaab8\uaabe-\uaabf\uaac1\uaaec-\uaaed\uaaf6\uabe5\uabe8\uabed\ufb1e\ufe00-\ufe0f\ufe20-\ufe2f\U000101fd\U000102e0\U00010376-\U0001037a\U00010a01-\U00010a03\U00010a05-\U00010a06\U00010a0c-\U00010a0f\U00010a38-\U00010a3a\U00010a3f\U00010ae5-\U00010ae6\U00010d24-\U00010d27\U00010f46-\U00010f50\U00011001\U00011038-\U00011046\U0001107f-\U00011081\U000110b3-\U000110b6\U000110b9-\U000110ba\U00011100-\U00011102\U00011127-\U0001112b\U0001112d-\U00011134\U00011173\U00011180-\U00011181\U000111b6-\U000111be\U000111c9-\U000111cc\U0001122f-\U00011231\U00011234\U00011236-\U00011237\U0001123e\U000112df\U000112e3-\U000112ea\U00011300-\U00011301\U0001133b-\U0001133c\U00011340\U00011366-\U0001136c\U00011370-\U00011374\U00011438-\U0001143f\U00011442-\U00011444\U00011446\U0001145e\U000114b3-\U000114b8\U000114ba\U000114bf-\U000114c0\U000114c2-\U000114c3\U000115b2-\U000115b5\U000115bc-\U000115bd\U000115bf-\U000115c0\U000115dc-\U000115dd\U00011633-\U0001163a\U0001163d\U0001163f-\U00011640\U000116ab\U000116ad\U000116b0-\U000116b5\U000116b7\U0001171d-\U0001171f\U00011722-\U00011725\U00011727-\U0001172b\U0001182f-\U00011837\U00011839-\U0001183a\U00011a01-\U00011a0a\U00011a33-\U00011a38\U00011a3b-\U00011a3e\U00011a47\U00011a51-\U00011a56\U00011a59-\U00011a5b\U00011a8a-\U00011a96\U00011a98-\U00011a99\U00011c30-\U00011c36\U00011c38-\U00011c3d\U00011c3f\U00011c92-\U00011ca7\U00011caa-\U00011cb0\U00011cb2-\U00011cb3\U00011cb5-\U00011cb6\U00011d31-\U00011d36\U00011d3a\U00011d3c-\U00011d3d\U00011d3f-\U00011d45\U00011d47\U00011d90-\U00011d91\U00011d95\U00011d97\U00011ef3-\U00011ef4\U00016af0-\U00016af4\U00016b30-\U00016b36\U00016f8f-\U00016f92\U0001bc9d-\U0001bc9e\U0001d167-\U0001d169\U0001d17b-\U0001d182\U0001d185-\U0001d18b\U0001d1aa-\U0001d1ad\U0001d242-\U0001d244\U0001da00-\U0001da36\U0001da3b-\U0001da6c\U0001da75\U0001da84\U0001da9b-\U0001da9f\U0001daa1-\U0001daaf\U0001e000-\U0001e006\U0001e008-\U0001e018\U0001e01b-\U0001e021\U0001e023-\U0001e024\U0001e026-\U0001e02a\U0001e8d0-\U0001e8d6\U0001e944-\U0001e94a\U000e0100-\U000e01ef'

Nd = '0-9\u0660-\u0669\u06f0-\u06f9\u07c0-\u07c9\u0966-\u096f\u09e6-\u09ef\u0a66-\u0a6f\u0ae6-\u0aef\u0b66-\u0b6f\u0be6-\u0bef\u0c66-\u0c6f\u0ce6-\u0cef\u0d66-\u0d6f\u0de6-\u0def\u0e50-\u0e59\u0ed0-\u0ed9\u0f20-\u0f29\u1040-\u1049\u1090-\u1099\u17e0-\u17e9\u1810-\u1819\u1946-\u194f\u19d0-\u19d9\u1a80-\u1a89\u1a90-\u1a99\u1b50-\u1b59\u1bb0-\u1bb9\u1c40-\u1c49\u1c50-\u1c59\ua620-\ua629\ua8d0-\ua8d9\ua900-\ua909\ua9d0-\ua9d9\ua9f0-\ua9f9\uaa50-\uaa59\uabf0-\uabf9\uff10-\uff19\U000104a0-\U000104a9\U00010d30-\U00010d39\U00011066-\U0001106f\U000110f0-\U000110f9\U00011136-\U0001113f\U000111d0-\U000111d9\U000112f0-\U000112f9\U00011450-\U00011459\U000114d0-\U000114d9\U00011650-\U00011659\U000116c0-\U000116c9\U00011730-\U00011739\U000118e0-\U000118e9\U00011c50-\U00011c59\U00011d50-\U00011d59\U00011da0-\U00011da9\U00016a60-\U00016a69\U00016b50-\U00016b59\U0001d7ce-\U0001d7ff\U0001e950-\U0001e959'

Nl = '\u16ee-\u16f0\u2160-\u2182\u2185-\u2188\u3007\u3021-\u3029\u3038-\u303a\ua6e6-\ua6ef\U00010140-\U00010174\U00010341\U0001034a\U000103d1-\U000103d5\U00012400-\U0001246e'

No = '\xb2-\xb3\xb9\xbc-\xbe\u09f4-\u09f9\u0b72-\u0b77\u0bf0-\u0bf2\u0c78-\u0c7e\u0d58-\u0d5e\u0d70-\u0d78\u0f2a-\u0f33\u1369-\u137c\u17f0-\u17f9\u19da\u2070\u2074-\u2079\u2080-\u2089\u2150-\u215f\u2189\u2460-\u249b\u24ea-\u24ff\u2776-\u2793\u2cfd\u3192-\u3195\u3220-\u3229\u3248-\u324f\u3251-\u325f\u3280-\u3289\u32b1-\u32bf\ua830-\ua835\U00010107-\U00010133\U00010175-\U00010178\U0001018a-\U0001018b\U000102e1-\U000102fb\U00010320-\U00010323\U00010858-\U0001085f\U00010879-\U0001087f\U000108a7-\U000108af\U000108fb-\U000108ff\U00010916-\U0001091b\U000109bc-\U000109bd\U000109c0-\U000109cf\U000109d2-\U000109ff\U00010a40-\U00010a48\U00010a7d-\U00010a7e\U00010a9d-\U00010a9f\U00010aeb-\U00010aef\U00010b58-\U00010b5f\U00010b78-\U00010b7f\U00010ba9-\U00010baf\U00010cfa-\U00010cff\U00010e60-\U00010e7e\U00010f1d-\U00010f26\U00010f51-\U00010f54\U00011052-\U00011065\U000111e1-\U000111f4\U0001173a-\U0001173b\U000118ea-\U000118f2\U00011c5a-\U00011c6c\U00016b5b-\U00016b61\U00016e80-\U00016e96\U0001d2e0-\U0001d2f3\U0001d360-\U0001d378\U0001e8c7-\U0001e8cf\U0001ec71-\U0001ecab\U0001ecad-\U0001ecaf\U0001ecb1-\U0001ecb4\U0001f100-\U0001f10c'

Pc = '_\u203f-\u2040\u2054\ufe33-\ufe34\ufe4d-\ufe4f\uff3f'

Pd = '\\-\u058a\u05be\u1400\u1806\u2010-\u2015\u2e17\u2e1a\u2e3a-\u2e3b\u2e40\u301c\u3030\u30a0\ufe31-\ufe32\ufe58\ufe63\uff0d'

Pe = ')\\]}\u0f3b\u0f3d\u169c\u2046\u207e\u208e\u2309\u230b\u232a\u2769\u276b\u276d\u276f\u2771\u2773\u2775\u27c6\u27e7\u27e9\u27eb\u27ed\u27ef\u2984\u2986\u2988\u298a\u298c\u298e\u2990\u2992\u2994\u2996\u2998\u29d9\u29db\u29fd\u2e23\u2e25\u2e27\u2e29\u3009\u300b\u300d\u300f\u3011\u3015\u3017\u3019\u301b\u301e-\u301f\ufd3e\ufe18\ufe36\ufe38\ufe3a\ufe3c\ufe3e\ufe40\ufe42\ufe44\ufe48\ufe5a\ufe5c\ufe5e\uff09\uff3d\uff5d\uff60\uff63'

Pf = '\xbb\u2019\u201d\u203a\u2e03\u2e05\u2e0a\u2e0d\u2e1d\u2e21'

Pi = '\xab\u2018\u201b-\u201c\u201f\u2039\u2e02\u2e04\u2e09\u2e0c\u2e1c\u2e20'

Po = "!-#%-'*,.-/:-;?-@\\\\\xa1\xa7\xb6-\xb7\xbf\u037e\u0387\u055a-\u055f\u0589\u05c0\u05c3\u05c6\u05f3-\u05f4\u0609-\u060a\u060c-\u060d\u061b\u061e-\u061f\u066a-\u066d\u06d4\u0700-\u070d\u07f7-\u07f9\u0830-\u083e\u085e\u0964-\u0965\u0970\u09fd\u0a76\u0af0\u0c84\u0df4\u0e4f\u0e5a-\u0e5b\u0f04-\u0f12\u0f14\u0f85\u0fd0-\u0fd4\u0fd9-\u0fda\u104a-\u104f\u10fb\u1360-\u1368\u166d-\u166e\u16eb-\u16ed\u1735-\u1736\u17d4-\u17d6\u17d8-\u17da\u1800-\u1805\u1807-\u180a\u1944-\u1945\u1a1e-\u1a1f\u1aa0-\u1aa6\u1aa8-\u1aad\u1b5a-\u1b60\u1bfc-\u1bff\u1c3b-\u1c3f\u1c7e-\u1c7f\u1cc0-\u1cc7\u1cd3\u2016-\u2017\u2020-\u2027\u2030-\u2038\u203b-\u203e\u2041-\u2043\u2047-\u2051\u2053\u2055-\u205e\u2cf9-\u2cfc\u2cfe-\u2cff\u2d70\u2e00-\u2e01\u2e06-\u2e08\u2e0b\u2e0e-\u2e16\u2e18-\u2e19\u2e1b\u2e1e-\u2e1f\u2e2a-\u2e2e\u2e30-\u2e39\u2e3c-\u2e3f\u2e41\u2e43-\u2e4e\u3001-\u3003\u303d\u30fb\ua4fe-\ua4ff\ua60d-\ua60f\ua673\ua67e\ua6f2-\ua6f7\ua874-\ua877\ua8ce-\ua8cf\ua8f8-\ua8fa\ua8fc\ua92e-\ua92f\ua95f\ua9c1-\ua9cd\ua9de-\ua9df\uaa5c-\uaa5f\uaade-\uaadf\uaaf0-\uaaf1\uabeb\ufe10-\ufe16\ufe19\ufe30\ufe45-\ufe46\ufe49-\ufe4c\ufe50-\ufe52\ufe54-\ufe57\ufe5f-\ufe61\ufe68\ufe6a-\ufe6b\uff01-\uff03\uff05-\uff07\uff0a\uff0c\uff0e-\uff0f\uff1a-\uff1b\uff1f-\uff20\uff3c\uff61\uff64-\uff65\U00010100-\U00010102\U0001039f\U000103d0\U0001056f\U00010857\U0001091f\U0001093f\U00010a50-\U00010a58\U00010a7f\U00010af0-\U00010af6\U00010b39-\U00010b3f\U00010b99-\U00010b9c\U00010f55-\U00010f59\U00011047-\U0001104d\U000110bb-\U000110bc\U000110be-\U000110c1\U00011140-\U00011143\U00011174-\U00011175\U000111c5-\U000111c8\U000111cd\U000111db\U000111dd-\U000111df\U00011238-\U0001123d\U000112a9\U0001144b-\U0001144f\U0001145b\U0001145d\U000114c6\U000115c1-\U000115d7\U00011641-\U00011643\U00011660-\U0001166c\U0001173c-\U0001173e\U0001183b\U00011a3f-\U00011a46\U00011a9a-\U00011a9c\U00011a9e-\U00011aa2\U00011c41-\U00011c45\U00011c70-\U00011c71\U00011ef7-\U00011ef8\U00012470-\U00012474\U00016a6e-\U00016a6f\U00016af5\U00016b37-\U00016b3b\U00016b44\U00016e97-\U00016e9a\U0001bc9f\U0001da87-\U0001da8b\U0001e95e-\U0001e95f"

Ps = '(\\[{\u0f3a\u0f3c\u169b\u201a\u201e\u2045\u207d\u208d\u2308\u230a\u2329\u2768\u276a\u276c\u276e\u2770\u2772\u2774\u27c5\u27e6\u27e8\u27ea\u27ec\u27ee\u2983\u2985\u2987\u2989\u298b\u298d\u298f\u2991\u2993\u2995\u2997\u29d8\u29da\u29fc\u2e22\u2e24\u2e26\u2e28\u2e42\u3008\u300a\u300c\u300e\u3010\u3014\u3016\u3018\u301a\u301d\ufd3f\ufe17\ufe35\ufe37\ufe39\ufe3b\ufe3d\ufe3f\ufe41\ufe43\ufe47\ufe59\ufe5b\ufe5d\uff08\uff3b\uff5b\uff5f\uff62'

Sc = '$\xa2-\xa5\u058f\u060b\u07fe-\u07ff\u09f2-\u09f3\u09fb\u0af1\u0bf9\u0e3f\u17db\u20a0-\u20bf\ua838\ufdfc\ufe69\uff04\uffe0-\uffe1\uffe5-\uffe6\U0001ecb0'

Sk = '\\^`\xa8\xaf\xb4\xb8\u02c2-\u02c5\u02d2-\u02df\u02e5-\u02eb\u02ed\u02ef-\u02ff\u0375\u0384-\u0385\u1fbd\u1fbf-\u1fc1\u1fcd-\u1fcf\u1fdd-\u1fdf\u1fed-\u1fef\u1ffd-\u1ffe\u309b-\u309c\ua700-\ua716\ua720-\ua721\ua789-\ua78a\uab5b\ufbb2-\ufbc1\uff3e\uff40\uffe3\U0001f3fb-\U0001f3ff'

Sm = '+<->|~\xac\xb1\xd7\xf7\u03f6\u0606-\u0608\u2044\u2052\u207a-\u207c\u208a-\u208c\u2118\u2140-\u2144\u214b\u2190-\u2194\u219a-\u219b\u21a0\u21a3\u21a6\u21ae\u21ce-\u21cf\u21d2\u21d4\u21f4-\u22ff\u2320-\u2321\u237c\u239b-\u23b3\u23dc-\u23e1\u25b7\u25c1\u25f8-\u25ff\u266f\u27c0-\u27c4\u27c7-\u27e5\u27f0-\u27ff\u2900-\u2982\u2999-\u29d7\u29dc-\u29fb\u29fe-\u2aff\u2b30-\u2b44\u2b47-\u2b4c\ufb29\ufe62\ufe64-\ufe66\uff0b\uff1c-\uff1e\uff5c\uff5e\uffe2\uffe9-\uffec\U0001d6c1\U0001d6db\U0001d6fb\U0001d715\U0001d735\U0001d74f\U0001d76f\U0001d789\U0001d7a9\U0001d7c3\U0001eef0-\U0001eef1'

So = '\xa6\xa9\xae\xb0\u0482\u058d-\u058e\u060e-\u060f\u06de\u06e9\u06fd-\u06fe\u07f6\u09fa\u0b70\u0bf3-\u0bf8\u0bfa\u0c7f\u0d4f\u0d79\u0f01-\u0f03\u0f13\u0f15-\u0f17\u0f1a-\u0f1f\u0f34\u0f36\u0f38\u0fbe-\u0fc5\u0fc7-\u0fcc\u0fce-\u0fcf\u0fd5-\u0fd8\u109e-\u109f\u1390-\u1399\u1940\u19de-\u19ff\u1b61-\u1b6a\u1b74-\u1b7c\u2100-\u2101\u2103-\u2106\u2108-\u2109\u2114\u2116-\u2117\u211e-\u2123\u2125\u2127\u2129\u212e\u213a-\u213b\u214a\u214c-\u214d\u214f\u218a-\u218b\u2195-\u2199\u219c-\u219f\u21a1-\u21a2\u21a4-\u21a5\u21a7-\u21ad\u21af-\u21cd\u21d0-\u21d1\u21d3\u21d5-\u21f3\u2300-\u2307\u230c-\u231f\u2322-\u2328\u232b-\u237b\u237d-\u239a\u23b4-\u23db\u23e2-\u2426\u2440-\u244a\u249c-\u24e9\u2500-\u25b6\u25b8-\u25c0\u25c2-\u25f7\u2600-\u266e\u2670-\u2767\u2794-\u27bf\u2800-\u28ff\u2b00-\u2b2f\u2b45-\u2b46\u2b4d-\u2b73\u2b76-\u2b95\u2b98-\u2bc8\u2bca-\u2bfe\u2ce5-\u2cea\u2e80-\u2e99\u2e9b-\u2ef3\u2f00-\u2fd5\u2ff0-\u2ffb\u3004\u3012-\u3013\u3020\u3036-\u3037\u303e-\u303f\u3190-\u3191\u3196-\u319f\u31c0-\u31e3\u3200-\u321e\u322a-\u3247\u3250\u3260-\u327f\u328a-\u32b0\u32c0-\u32fe\u3300-\u33ff\u4dc0-\u4dff\ua490-\ua4c6\ua828-\ua82b\ua836-\ua837\ua839\uaa77-\uaa79\ufdfd\uffe4\uffe8\uffed-\uffee\ufffc-\ufffd\U00010137-\U0001013f\U00010179-\U00010189\U0001018c-\U0001018e\U00010190-\U0001019b\U000101a0\U000101d0-\U000101fc\U00010877-\U00010878\U00010ac8\U0001173f\U00016b3c-\U00016b3f\U00016b45\U0001bc9c\U0001d000-\U0001d0f5\U0001d100-\U0001d126\U0001d129-\U0001d164\U0001d16a-\U0001d16c\U0001d183-\U0001d184\U0001d18c-\U0001d1a9\U0001d1ae-\U0001d1e8\U0001d200-\U0001d241\U0001d245\U0001d300-\U0001d356\U0001d800-\U0001d9ff\U0001da37-\U0001da3a\U0001da6d-\U0001da74\U0001da76-\U0001da83\U0001da85-\U0001da86\U0001ecac\U0001f000-\U0001f02b\U0001f030-\U0001f093\U0001f0a0-\U0001f0ae\U0001f0b1-\U0001f0bf\U0001f0c1-\U0001f0cf\U0001f0d1-\U0001f0f5\U0001f110-\U0001f16b\U0001f170-\U0001f1ac\U0001f1e6-\U0001f202\U0001f210-\U0001f23b\U0001f240-\U0001f248\U0001f250-\U0001f251\U0001f260-\U0001f265\U0001f300-\U0001f3fa\U0001f400-\U0001f6d4\U0001f6e0-\U0001f6ec\U0001f6f0-\U0001f6f9\U0001f700-\U0001f773\U0001f780-\U0001f7d8\U0001f800-\U0001f80b\U0001f810-\U0001f847\U0001f850-\U0001f859\U0001f860-\U0001f887\U0001f890-\U0001f8ad\U0001f900-\U0001f90b\U0001f910-\U0001f93e\U0001f940-\U0001f970\U0001f973-\U0001f976\U0001f97a\U0001f97c-\U0001f9a2\U0001f9b0-\U0001f9b9\U0001f9c0-\U0001f9c2\U0001f9d0-\U0001f9ff\U0001fa60-\U0001fa6d'

Zl = '\u2028'

Zp = '\u2029'

Zs = ' \xa0\u1680\u2000-\u200a\u202f\u205f\u3000'

xid_continue = '0-9A-Z_a-z\xaa\xb5\xb7\xba\xc0-\xd6\xd8-\xf6\xf8-\u02c1\u02c6-\u02d1\u02e0-\u02e4\u02ec\u02ee\u0300-\u0374\u0376-\u0377\u037b-\u037d\u037f\u0386-\u038a\u038c\u038e-\u03a1\u03a3-\u03f5\u03f7-\u0481\u0483-\u0487\u048a-\u052f\u0531-\u0556\u0559\u0560-\u0588\u0591-\u05bd\u05bf\u05c1-\u05c2\u05c4-\u05c5\u05c7\u05d0-\u05ea\u05ef-\u05f2\u0610-\u061a\u0620-\u0669\u066e-\u06d3\u06d5-\u06dc\u06df-\u06e8\u06ea-\u06fc\u06ff\u0710-\u074a\u074d-\u07b1\u07c0-\u07f5\u07fa\u07fd\u0800-\u082d\u0840-\u085b\u0860-\u086a\u08a0-\u08b4\u08b6-\u08bd\u08d3-\u08e1\u08e3-\u0963\u0966-\u096f\u0971-\u0983\u0985-\u098c\u098f-\u0990\u0993-\u09a8\u09aa-\u09b0\u09b2\u09b6-\u09b9\u09bc-\u09c4\u09c7-\u09c8\u09cb-\u09ce\u09d7\u09dc-\u09dd\u09df-\u09e3\u09e6-\u09f1\u09fc\u09fe\u0a01-\u0a03\u0a05-\u0a0a\u0a0f-\u0a10\u0a13-\u0a28\u0a2a-\u0a30\u0a32-\u0a33\u0a35-\u0a36\u0a38-\u0a39\u0a3c\u0a3e-\u0a42\u0a47-\u0a48\u0a4b-\u0a4d\u0a51\u0a59-\u0a5c\u0a5e\u0a66-\u0a75\u0a81-\u0a83\u0a85-\u0a8d\u0a8f-\u0a91\u0a93-\u0aa8\u0aaa-\u0ab0\u0ab2-\u0ab3\u0ab5-\u0ab9\u0abc-\u0ac5\u0ac7-\u0ac9\u0acb-\u0acd\u0ad0\u0ae0-\u0ae3\u0ae6-\u0aef\u0af9-\u0aff\u0b01-\u0b03\u0b05-\u0b0c\u0b0f-\u0b10\u0b13-\u0b28\u0b2a-\u0b30\u0b32-\u0b33\u0b35-\u0b39\u0b3c-\u0b44\u0b47-\u0b48\u0b4b-\u0b4d\u0b56-\u0b57\u0b5c-\u0b5d\u0b5f-\u0b63\u0b66-\u0b6f\u0b71\u0b82-\u0b83\u0b85-\u0b8a\u0b8e-\u0b90\u0b92-\u0b95\u0b99-\u0b9a\u0b9c\u0b9e-\u0b9f\u0ba3-\u0ba4\u0ba8-\u0baa\u0bae-\u0bb9\u0bbe-\u0bc2\u0bc6-\u0bc8\u0bca-\u0bcd\u0bd0\u0bd7\u0be6-\u0bef\u0c00-\u0c0c\u0c0e-\u0c10\u0c12-\u0c28\u0c2a-\u0c39\u0c3d-\u0c44\u0c46-\u0c48\u0c4a-\u0c4d\u0c55-\u0c56\u0c58-\u0c5a\u0c60-\u0c63\u0c66-\u0c6f\u0c80-\u0c83\u0c85-\u0c8c\u0c8e-\u0c90\u0c92-\u0ca8\u0caa-\u0cb3\u0cb5-\u0cb9\u0cbc-\u0cc4\u0cc6-\u0cc8\u0cca-\u0ccd\u0cd5-\u0cd6\u0cde\u0ce0-\u0ce3\u0ce6-\u0cef\u0cf1-\u0cf2\u0d00-\u0d03\u0d05-\u0d0c\u0d0e-\u0d10\u0d12-\u0d44\u0d46-\u0d48\u0d4a-\u0d4e\u0d54-\u0d57\u0d5f-\u0d63\u0d66-\u0d6f\u0d7a-\u0d7f\u0d82-\u0d83\u0d85-\u0d96\u0d9a-\u0db1\u0db3-\u0dbb\u0dbd\u0dc0-\u0dc6\u0dca\u0dcf-\u0dd4\u0dd6\u0dd8-\u0ddf\u0de6-\u0def\u0df2-\u0df3\u0e01-\u0e3a\u0e40-\u0e4e\u0e50-\u0e59\u0e81-\u0e82\u0e84\u0e87-\u0e88\u0e8a\u0e8d\u0e94-\u0e97\u0e99-\u0e9f\u0ea1-\u0ea3\u0ea5\u0ea7\u0eaa-\u0eab\u0ead-\u0eb9\u0ebb-\u0ebd\u0ec0-\u0ec4\u0ec6\u0ec8-\u0ecd\u0ed0-\u0ed9\u0edc-\u0edf\u0f00\u0f18-\u0f19\u0f20-\u0f29\u0f35\u0f37\u0f39\u0f3e-\u0f47\u0f49-\u0f6c\u0f71-\u0f84\u0f86-\u0f97\u0f99-\u0fbc\u0fc6\u1000-\u1049\u1050-\u109d\u10a0-\u10c5\u10c7\u10cd\u10d0-\u10fa\u10fc-\u1248\u124a-\u124d\u1250-\u1256\u1258\u125a-\u125d\u1260-\u1288\u128a-\u128d\u1290-\u12b0\u12b2-\u12b5\u12b8-\u12be\u12c0\u12c2-\u12c5\u12c8-\u12d6\u12d8-\u1310\u1312-\u1315\u1318-\u135a\u135d-\u135f\u1369-\u1371\u1380-\u138f\u13a0-\u13f5\u13f8-\u13fd\u1401-\u166c\u166f-\u167f\u1681-\u169a\u16a0-\u16ea\u16ee-\u16f8\u1700-\u170c\u170e-\u1714\u1720-\u1734\u1740-\u1753\u1760-\u176c\u176e-\u1770\u1772-\u1773\u1780-\u17d3\u17d7\u17dc-\u17dd\u17e0-\u17e9\u180b-\u180d\u1810-\u1819\u1820-\u1878\u1880-\u18aa\u18b0-\u18f5\u1900-\u191e\u1920-\u192b\u1930-\u193b\u1946-\u196d\u1970-\u1974\u1980-\u19ab\u19b0-\u19c9\u19d0-\u19da\u1a00-\u1a1b\u1a20-\u1a5e\u1a60-\u1a7c\u1a7f-\u1a89\u1a90-\u1a99\u1aa7\u1ab0-\u1abd\u1b00-\u1b4b\u1b50-\u1b59\u1b6b-\u1b73\u1b80-\u1bf3\u1c00-\u1c37\u1c40-\u1c49\u1c4d-\u1c7d\u1c80-\u1c88\u1c90-\u1cba\u1cbd-\u1cbf\u1cd0-\u1cd2\u1cd4-\u1cf9\u1d00-\u1df9\u1dfb-\u1f15\u1f18-\u1f1d\u1f20-\u1f45\u1f48-\u1f4d\u1f50-\u1f57\u1f59\u1f5b\u1f5d\u1f5f-\u1f7d\u1f80-\u1fb4\u1fb6-\u1fbc\u1fbe\u1fc2-\u1fc4\u1fc6-\u1fcc\u1fd0-\u1fd3\u1fd6-\u1fdb\u1fe0-\u1fec\u1ff2-\u1ff4\u1ff6-\u1ffc\u203f-\u2040\u2054\u2071\u207f\u2090-\u209c\u20d0-\u20dc\u20e1\u20e5-\u20f0\u2102\u2107\u210a-\u2113\u2115\u2118-\u211d\u2124\u2126\u2128\u212a-\u2139\u213c-\u213f\u2145-\u2149\u214e\u2160-\u2188\u2c00-\u2c2e\u2c30-\u2c5e\u2c60-\u2ce4\u2ceb-\u2cf3\u2d00-\u2d25\u2d27\u2d2d\u2d30-\u2d67\u2d6f\u2d7f-\u2d96\u2da0-\u2da6\u2da8-\u2dae\u2db0-\u2db6\u2db8-\u2dbe\u2dc0-\u2dc6\u2dc8-\u2dce\u2dd0-\u2dd6\u2dd8-\u2dde\u2de0-\u2dff\u3005-\u3007\u3021-\u302f\u3031-\u3035\u3038-\u303c\u3041-\u3096\u3099-\u309a\u309d-\u309f\u30a1-\u30fa\u30fc-\u30ff\u3105-\u312f\u3131-\u318e\u31a0-\u31ba\u31f0-\u31ff\u3400-\u4db5\u4e00-\u9fef\ua000-\ua48c\ua4d0-\ua4fd\ua500-\ua60c\ua610-\ua62b\ua640-\ua66f\ua674-\ua67d\ua67f-\ua6f1\ua717-\ua71f\ua722-\ua788\ua78b-\ua7b9\ua7f7-\ua827\ua840-\ua873\ua880-\ua8c5\ua8d0-\ua8d9\ua8e0-\ua8f7\ua8fb\ua8fd-\ua92d\ua930-\ua953\ua960-\ua97c\ua980-\ua9c0\ua9cf-\ua9d9\ua9e0-\ua9fe\uaa00-\uaa36\uaa40-\uaa4d\uaa50-\uaa59\uaa60-\uaa76\uaa7a-\uaac2\uaadb-\uaadd\uaae0-\uaaef\uaaf2-\uaaf6\uab01-\uab06\uab09-\uab0e\uab11-\uab16\uab20-\uab26\uab28-\uab2e\uab30-\uab5a\uab5c-\uab65\uab70-\uabea\uabec-\uabed\uabf0-\uabf9\uac00-\ud7a3\ud7b0-\ud7c6\ud7cb-\ud7fb\uf900-\ufa6d\ufa70-\ufad9\ufb00-\ufb06\ufb13-\ufb17\ufb1d-\ufb28\ufb2a-\ufb36\ufb38-\ufb3c\ufb3e\ufb40-\ufb41\ufb43-\ufb44\ufb46-\ufbb1\ufbd3-\ufc5d\ufc64-\ufd3d\ufd50-\ufd8f\ufd92-\ufdc7\ufdf0-\ufdf9\ufe00-\ufe0f\ufe20-\ufe2f\ufe33-\ufe34\ufe4d-\ufe4f\ufe71\ufe73\ufe77\ufe79\ufe7b\ufe7d\ufe7f-\ufefc\uff10-\uff19\uff21-\uff3a\uff3f\uff41-\uff5a\uff66-\uffbe\uffc2-\uffc7\uffca-\uffcf\uffd2-\uffd7\uffda-\uffdc\U00010000-\U0001000b\U0001000d-\U00010026\U00010028-\U0001003a\U0001003c-\U0001003d\U0001003f-\U0001004d\U00010050-\U0001005d\U00010080-\U000100fa\U00010140-\U00010174\U000101fd\U00010280-\U0001029c\U000102a0-\U000102d0\U000102e0\U00010300-\U0001031f\U0001032d-\U0001034a\U00010350-\U0001037a\U00010380-\U0001039d\U000103a0-\U000103c3\U000103c8-\U000103cf\U000103d1-\U000103d5\U00010400-\U0001049d\U000104a0-\U000104a9\U000104b0-\U000104d3\U000104d8-\U000104fb\U00010500-\U00010527\U00010530-\U00010563\U00010600-\U00010736\U00010740-\U00010755\U00010760-\U00010767\U00010800-\U00010805\U00010808\U0001080a-\U00010835\U00010837-\U00010838\U0001083c\U0001083f-\U00010855\U00010860-\U00010876\U00010880-\U0001089e\U000108e0-\U000108f2\U000108f4-\U000108f5\U00010900-\U00010915\U00010920-\U00010939\U00010980-\U000109b7\U000109be-\U000109bf\U00010a00-\U00010a03\U00010a05-\U00010a06\U00010a0c-\U00010a13\U00010a15-\U00010a17\U00010a19-\U00010a35\U00010a38-\U00010a3a\U00010a3f\U00010a60-\U00010a7c\U00010a80-\U00010a9c\U00010ac0-\U00010ac7\U00010ac9-\U00010ae6\U00010b00-\U00010b35\U00010b40-\U00010b55\U00010b60-\U00010b72\U00010b80-\U00010b91\U00010c00-\U00010c48\U00010c80-\U00010cb2\U00010cc0-\U00010cf2\U00010d00-\U00010d27\U00010d30-\U00010d39\U00010f00-\U00010f1c\U00010f27\U00010f30-\U00010f50\U00011000-\U00011046\U00011066-\U0001106f\U0001107f-\U000110ba\U000110d0-\U000110e8\U000110f0-\U000110f9\U00011100-\U00011134\U00011136-\U0001113f\U00011144-\U00011146\U00011150-\U00011173\U00011176\U00011180-\U000111c4\U000111c9-\U000111cc\U000111d0-\U000111da\U000111dc\U00011200-\U00011211\U00011213-\U00011237\U0001123e\U00011280-\U00011286\U00011288\U0001128a-\U0001128d\U0001128f-\U0001129d\U0001129f-\U000112a8\U000112b0-\U000112ea\U000112f0-\U000112f9\U00011300-\U00011303\U00011305-\U0001130c\U0001130f-\U00011310\U00011313-\U00011328\U0001132a-\U00011330\U00011332-\U00011333\U00011335-\U00011339\U0001133b-\U00011344\U00011347-\U00011348\U0001134b-\U0001134d\U00011350\U00011357\U0001135d-\U00011363\U00011366-\U0001136c\U00011370-\U00011374\U00011400-\U0001144a\U00011450-\U00011459\U0001145e\U00011480-\U000114c5\U000114c7\U000114d0-\U000114d9\U00011580-\U000115b5\U000115b8-\U000115c0\U000115d8-\U000115dd\U00011600-\U00011640\U00011644\U00011650-\U00011659\U00011680-\U000116b7\U000116c0-\U000116c9\U00011700-\U0001171a\U0001171d-\U0001172b\U00011730-\U00011739\U00011800-\U0001183a\U000118a0-\U000118e9\U000118ff\U00011a00-\U00011a3e\U00011a47\U00011a50-\U00011a83\U00011a86-\U00011a99\U00011a9d\U00011ac0-\U00011af8\U00011c00-\U00011c08\U00011c0a-\U00011c36\U00011c38-\U00011c40\U00011c50-\U00011c59\U00011c72-\U00011c8f\U00011c92-\U00011ca7\U00011ca9-\U00011cb6\U00011d00-\U00011d06\U00011d08-\U00011d09\U00011d0b-\U00011d36\U00011d3a\U00011d3c-\U00011d3d\U00011d3f-\U00011d47\U00011d50-\U00011d59\U00011d60-\U00011d65\U00011d67-\U00011d68\U00011d6a-\U00011d8e\U00011d90-\U00011d91\U00011d93-\U00011d98\U00011da0-\U00011da9\U00011ee0-\U00011ef6\U00012000-\U00012399\U00012400-\U0001246e\U00012480-\U00012543\U00013000-\U0001342e\U00014400-\U00014646\U00016800-\U00016a38\U00016a40-\U00016a5e\U00016a60-\U00016a69\U00016ad0-\U00016aed\U00016af0-\U00016af4\U00016b00-\U00016b36\U00016b40-\U00016b43\U00016b50-\U00016b59\U00016b63-\U00016b77\U00016b7d-\U00016b8f\U00016e40-\U00016e7f\U00016f00-\U00016f44\U00016f50-\U00016f7e\U00016f8f-\U00016f9f\U00016fe0-\U00016fe1\U00017000-\U000187f1\U00018800-\U00018af2\U0001b000-\U0001b11e\U0001b170-\U0001b2fb\U0001bc00-\U0001bc6a\U0001bc70-\U0001bc7c\U0001bc80-\U0001bc88\U0001bc90-\U0001bc99\U0001bc9d-\U0001bc9e\U0001d165-\U0001d169\U0001d16d-\U0001d172\U0001d17b-\U0001d182\U0001d185-\U0001d18b\U0001d1aa-\U0001d1ad\U0001d242-\U0001d244\U0001d400-\U0001d454\U0001d456-\U0001d49c\U0001d49e-\U0001d49f\U0001d4a2\U0001d4a5-\U0001d4a6\U0001d4a9-\U0001d4ac\U0001d4ae-\U0001d4b9\U0001d4bb\U0001d4bd-\U0001d4c3\U0001d4c5-\U0001d505\U0001d507-\U0001d50a\U0001d50d-\U0001d514\U0001d516-\U0001d51c\U0001d51e-\U0001d539\U0001d53b-\U0001d53e\U0001d540-\U0001d544\U0001d546\U0001d54a-\U0001d550\U0001d552-\U0001d6a5\U0001d6a8-\U0001d6c0\U0001d6c2-\U0001d6da\U0001d6dc-\U0001d6fa\U0001d6fc-\U0001d714\U0001d716-\U0001d734\U0001d736-\U0001d74e\U0001d750-\U0001d76e\U0001d770-\U0001d788\U0001d78a-\U0001d7a8\U0001d7aa-\U0001d7c2\U0001d7c4-\U0001d7cb\U0001d7ce-\U0001d7ff\U0001da00-\U0001da36\U0001da3b-\U0001da6c\U0001da75\U0001da84\U0001da9b-\U0001da9f\U0001daa1-\U0001daaf\U0001e000-\U0001e006\U0001e008-\U0001e018\U0001e01b-\U0001e021\U0001e023-\U0001e024\U0001e026-\U0001e02a\U0001e800-\U0001e8c4\U0001e8d0-\U0001e8d6\U0001e900-\U0001e94a\U0001e950-\U0001e959\U0001ee00-\U0001ee03\U0001ee05-\U0001ee1f\U0001ee21-\U0001ee22\U0001ee24\U0001ee27\U0001ee29-\U0001ee32\U0001ee34-\U0001ee37\U0001ee39\U0001ee3b\U0001ee42\U0001ee47\U0001ee49\U0001ee4b\U0001ee4d-\U0001ee4f\U0001ee51-\U0001ee52\U0001ee54\U0001ee57\U0001ee59\U0001ee5b\U0001ee5d\U0001ee5f\U0001ee61-\U0001ee62\U0001ee64\U0001ee67-\U0001ee6a\U0001ee6c-\U0001ee72\U0001ee74-\U0001ee77\U0001ee79-\U0001ee7c\U0001ee7e\U0001ee80-\U0001ee89\U0001ee8b-\U0001ee9b\U0001eea1-\U0001eea3\U0001eea5-\U0001eea9\U0001eeab-\U0001eebb\U00020000-\U0002a6d6\U0002a700-\U0002b734\U0002b740-\U0002b81d\U0002b820-\U0002cea1\U0002ceb0-\U0002ebe0\U0002f800-\U0002fa1d\U000e0100-\U000e01ef'

xid_start = 'A-Z_a-z\xaa\xb5\xba\xc0-\xd6\xd8-\xf6\xf8-\u02c1\u02c6-\u02d1\u02e0-\u02e4\u02ec\u02ee\u0370-\u0374\u0376-\u0377\u037b-\u037d\u037f\u0386\u0388-\u038a\u038c\u038e-\u03a1\u03a3-\u03f5\u03f7-\u0481\u048a-\u052f\u0531-\u0556\u0559\u0560-\u0588\u05d0-\u05ea\u05ef-\u05f2\u0620-\u064a\u066e-\u066f\u0671-\u06d3\u06d5\u06e5-\u06e6\u06ee-\u06ef\u06fa-\u06fc\u06ff\u0710\u0712-\u072f\u074d-\u07a5\u07b1\u07ca-\u07ea\u07f4-\u07f5\u07fa\u0800-\u0815\u081a\u0824\u0828\u0840-\u0858\u0860-\u086a\u08a0-\u08b4\u08b6-\u08bd\u0904-\u0939\u093d\u0950\u0958-\u0961\u0971-\u0980\u0985-\u098c\u098f-\u0990\u0993-\u09a8\u09aa-\u09b0\u09b2\u09b6-\u09b9\u09bd\u09ce\u09dc-\u09dd\u09df-\u09e1\u09f0-\u09f1\u09fc\u0a05-\u0a0a\u0a0f-\u0a10\u0a13-\u0a28\u0a2a-\u0a30\u0a32-\u0a33\u0a35-\u0a36\u0a38-\u0a39\u0a59-\u0a5c\u0a5e\u0a72-\u0a74\u0a85-\u0a8d\u0a8f-\u0a91\u0a93-\u0aa8\u0aaa-\u0ab0\u0ab2-\u0ab3\u0ab5-\u0ab9\u0abd\u0ad0\u0ae0-\u0ae1\u0af9\u0b05-\u0b0c\u0b0f-\u0b10\u0b13-\u0b28\u0b2a-\u0b30\u0b32-\u0b33\u0b35-\u0b39\u0b3d\u0b5c-\u0b5d\u0b5f-\u0b61\u0b71\u0b83\u0b85-\u0b8a\u0b8e-\u0b90\u0b92-\u0b95\u0b99-\u0b9a\u0b9c\u0b9e-\u0b9f\u0ba3-\u0ba4\u0ba8-\u0baa\u0bae-\u0bb9\u0bd0\u0c05-\u0c0c\u0c0e-\u0c10\u0c12-\u0c28\u0c2a-\u0c39\u0c3d\u0c58-\u0c5a\u0c60-\u0c61\u0c80\u0c85-\u0c8c\u0c8e-\u0c90\u0c92-\u0ca8\u0caa-\u0cb3\u0cb5-\u0cb9\u0cbd\u0cde\u0ce0-\u0ce1\u0cf1-\u0cf2\u0d05-\u0d0c\u0d0e-\u0d10\u0d12-\u0d3a\u0d3d\u0d4e\u0d54-\u0d56\u0d5f-\u0d61\u0d7a-\u0d7f\u0d85-\u0d96\u0d9a-\u0db1\u0db3-\u0dbb\u0dbd\u0dc0-\u0dc6\u0e01-\u0e30\u0e32\u0e40-\u0e46\u0e81-\u0e82\u0e84\u0e87-\u0e88\u0e8a\u0e8d\u0e94-\u0e97\u0e99-\u0e9f\u0ea1-\u0ea3\u0ea5\u0ea7\u0eaa-\u0eab\u0ead-\u0eb0\u0eb2\u0ebd\u0ec0-\u0ec4\u0ec6\u0edc-\u0edf\u0f00\u0f40-\u0f47\u0f49-\u0f6c\u0f88-\u0f8c\u1000-\u102a\u103f\u1050-\u1055\u105a-\u105d\u1061\u1065-\u1066\u106e-\u1070\u1075-\u1081\u108e\u10a0-\u10c5\u10c7\u10cd\u10d0-\u10fa\u10fc-\u1248\u124a-\u124d\u1250-\u1256\u1258\u125a-\u125d\u1260-\u1288\u128a-\u128d\u1290-\u12b0\u12b2-\u12b5\u12b8-\u12be\u12c0\u12c2-\u12c5\u12c8-\u12d6\u12d8-\u1310\u1312-\u1315\u1318-\u135a\u1380-\u138f\u13a0-\u13f5\u13f8-\u13fd\u1401-\u166c\u166f-\u167f\u1681-\u169a\u16a0-\u16ea\u16ee-\u16f8\u1700-\u170c\u170e-\u1711\u1720-\u1731\u1740-\u1751\u1760-\u176c\u176e-\u1770\u1780-\u17b3\u17d7\u17dc\u1820-\u1878\u1880-\u18a8\u18aa\u18b0-\u18f5\u1900-\u191e\u1950-\u196d\u1970-\u1974\u1980-\u19ab\u19b0-\u19c9\u1a00-\u1a16\u1a20-\u1a54\u1aa7\u1b05-\u1b33\u1b45-\u1b4b\u1b83-\u1ba0\u1bae-\u1baf\u1bba-\u1be5\u1c00-\u1c23\u1c4d-\u1c4f\u1c5a-\u1c7d\u1c80-\u1c88\u1c90-\u1cba\u1cbd-\u1cbf\u1ce9-\u1cec\u1cee-\u1cf1\u1cf5-\u1cf6\u1d00-\u1dbf\u1e00-\u1f15\u1f18-\u1f1d\u1f20-\u1f45\u1f48-\u1f4d\u1f50-\u1f57\u1f59\u1f5b\u1f5d\u1f5f-\u1f7d\u1f80-\u1fb4\u1fb6-\u1fbc\u1fbe\u1fc2-\u1fc4\u1fc6-\u1fcc\u1fd0-\u1fd3\u1fd6-\u1fdb\u1fe0-\u1fec\u1ff2-\u1ff4\u1ff6-\u1ffc\u2071\u207f\u2090-\u209c\u2102\u2107\u210a-\u2113\u2115\u2118-\u211d\u2124\u2126\u2128\u212a-\u2139\u213c-\u213f\u2145-\u2149\u214e\u2160-\u2188\u2c00-\u2c2e\u2c30-\u2c5e\u2c60-\u2ce4\u2ceb-\u2cee\u2cf2-\u2cf3\u2d00-\u2d25\u2d27\u2d2d\u2d30-\u2d67\u2d6f\u2d80-\u2d96\u2da0-\u2da6\u2da8-\u2dae\u2db0-\u2db6\u2db8-\u2dbe\u2dc0-\u2dc6\u2dc8-\u2dce\u2dd0-\u2dd6\u2dd8-\u2dde\u3005-\u3007\u3021-\u3029\u3031-\u3035\u3038-\u303c\u3041-\u3096\u309d-\u309f\u30a1-\u30fa\u30fc-\u30ff\u3105-\u312f\u3131-\u318e\u31a0-\u31ba\u31f0-\u31ff\u3400-\u4db5\u4e00-\u9fef\ua000-\ua48c\ua4d0-\ua4fd\ua500-\ua60c\ua610-\ua61f\ua62a-\ua62b\ua640-\ua66e\ua67f-\ua69d\ua6a0-\ua6ef\ua717-\ua71f\ua722-\ua788\ua78b-\ua7b9\ua7f7-\ua801\ua803-\ua805\ua807-\ua80a\ua80c-\ua822\ua840-\ua873\ua882-\ua8b3\ua8f2-\ua8f7\ua8fb\ua8fd-\ua8fe\ua90a-\ua925\ua930-\ua946\ua960-\ua97c\ua984-\ua9b2\ua9cf\ua9e0-\ua9e4\ua9e6-\ua9ef\ua9fa-\ua9fe\uaa00-\uaa28\uaa40-\uaa42\uaa44-\uaa4b\uaa60-\uaa76\uaa7a\uaa7e-\uaaaf\uaab1\uaab5-\uaab6\uaab9-\uaabd\uaac0\uaac2\uaadb-\uaadd\uaae0-\uaaea\uaaf2-\uaaf4\uab01-\uab06\uab09-\uab0e\uab11-\uab16\uab20-\uab26\uab28-\uab2e\uab30-\uab5a\uab5c-\uab65\uab70-\uabe2\uac00-\ud7a3\ud7b0-\ud7c6\ud7cb-\ud7fb\uf900-\ufa6d\ufa70-\ufad9\ufb00-\ufb06\ufb13-\ufb17\ufb1d\ufb1f-\ufb28\ufb2a-\ufb36\ufb38-\ufb3c\ufb3e\ufb40-\ufb41\ufb43-\ufb44\ufb46-\ufbb1\ufbd3-\ufc5d\ufc64-\ufd3d\ufd50-\ufd8f\ufd92-\ufdc7\ufdf0-\ufdf9\ufe71\ufe73\ufe77\ufe79\ufe7b\ufe7d\ufe7f-\ufefc\uff21-\uff3a\uff41-\uff5a\uff66-\uff9d\uffa0-\uffbe\uffc2-\uffc7\uffca-\uffcf\uffd2-\uffd7\uffda-\uffdc\U00010000-\U0001000b\U0001000d-\U00010026\U00010028-\U0001003a\U0001003c-\U0001003d\U0001003f-\U0001004d\U00010050-\U0001005d\U00010080-\U000100fa\U00010140-\U00010174\U00010280-\U0001029c\U000102a0-\U000102d0\U00010300-\U0001031f\U0001032d-\U0001034a\U00010350-\U00010375\U00010380-\U0001039d\U000103a0-\U000103c3\U000103c8-\U000103cf\U000103d1-\U000103d5\U00010400-\U0001049d\U000104b0-\U000104d3\U000104d8-\U000104fb\U00010500-\U00010527\U00010530-\U00010563\U00010600-\U00010736\U00010740-\U00010755\U00010760-\U00010767\U00010800-\U00010805\U00010808\U0001080a-\U00010835\U00010837-\U00010838\U0001083c\U0001083f-\U00010855\U00010860-\U00010876\U00010880-\U0001089e\U000108e0-\U000108f2\U000108f4-\U000108f5\U00010900-\U00010915\U00010920-\U00010939\U00010980-\U000109b7\U000109be-\U000109bf\U00010a00\U00010a10-\U00010a13\U00010a15-\U00010a17\U00010a19-\U00010a35\U00010a60-\U00010a7c\U00010a80-\U00010a9c\U00010ac0-\U00010ac7\U00010ac9-\U00010ae4\U00010b00-\U00010b35\U00010b40-\U00010b55\U00010b60-\U00010b72\U00010b80-\U00010b91\U00010c00-\U00010c48\U00010c80-\U00010cb2\U00010cc0-\U00010cf2\U00010d00-\U00010d23\U00010f00-\U00010f1c\U00010f27\U00010f30-\U00010f45\U00011003-\U00011037\U00011083-\U000110af\U000110d0-\U000110e8\U00011103-\U00011126\U00011144\U00011150-\U00011172\U00011176\U00011183-\U000111b2\U000111c1-\U000111c4\U000111da\U000111dc\U00011200-\U00011211\U00011213-\U0001122b\U00011280-\U00011286\U00011288\U0001128a-\U0001128d\U0001128f-\U0001129d\U0001129f-\U000112a8\U000112b0-\U000112de\U00011305-\U0001130c\U0001130f-\U00011310\U00011313-\U00011328\U0001132a-\U00011330\U00011332-\U00011333\U00011335-\U00011339\U0001133d\U00011350\U0001135d-\U00011361\U00011400-\U00011434\U00011447-\U0001144a\U00011480-\U000114af\U000114c4-\U000114c5\U000114c7\U00011580-\U000115ae\U000115d8-\U000115db\U00011600-\U0001162f\U00011644\U00011680-\U000116aa\U00011700-\U0001171a\U00011800-\U0001182b\U000118a0-\U000118df\U000118ff\U00011a00\U00011a0b-\U00011a32\U00011a3a\U00011a50\U00011a5c-\U00011a83\U00011a86-\U00011a89\U00011a9d\U00011ac0-\U00011af8\U00011c00-\U00011c08\U00011c0a-\U00011c2e\U00011c40\U00011c72-\U00011c8f\U00011d00-\U00011d06\U00011d08-\U00011d09\U00011d0b-\U00011d30\U00011d46\U00011d60-\U00011d65\U00011d67-\U00011d68\U00011d6a-\U00011d89\U00011d98\U00011ee0-\U00011ef2\U00012000-\U00012399\U00012400-\U0001246e\U00012480-\U00012543\U00013000-\U0001342e\U00014400-\U00014646\U00016800-\U00016a38\U00016a40-\U00016a5e\U00016ad0-\U00016aed\U00016b00-\U00016b2f\U00016b40-\U00016b43\U00016b63-\U00016b77\U00016b7d-\U00016b8f\U00016e40-\U00016e7f\U00016f00-\U00016f44\U00016f50\U00016f93-\U00016f9f\U00016fe0-\U00016fe1\U00017000-\U000187f1\U00018800-\U00018af2\U0001b000-\U0001b11e\U0001b170-\U0001b2fb\U0001bc00-\U0001bc6a\U0001bc70-\U0001bc7c\U0001bc80-\U0001bc88\U0001bc90-\U0001bc99\U0001d400-\U0001d454\U0001d456-\U0001d49c\U0001d49e-\U0001d49f\U0001d4a2\U0001d4a5-\U0001d4a6\U0001d4a9-\U0001d4ac\U0001d4ae-\U0001d4b9\U0001d4bb\U0001d4bd-\U0001d4c3\U0001d4c5-\U0001d505\U0001d507-\U0001d50a\U0001d50d-\U0001d514\U0001d516-\U0001d51c\U0001d51e-\U0001d539\U0001d53b-\U0001d53e\U0001d540-\U0001d544\U0001d546\U0001d54a-\U0001d550\U0001d552-\U0001d6a5\U0001d6a8-\U0001d6c0\U0001d6c2-\U0001d6da\U0001d6dc-\U0001d6fa\U0001d6fc-\U0001d714\U0001d716-\U0001d734\U0001d736-\U0001d74e\U0001d750-\U0001d76e\U0001d770-\U0001d788\U0001d78a-\U0001d7a8\U0001d7aa-\U0001d7c2\U0001d7c4-\U0001d7cb\U0001e800-\U0001e8c4\U0001e900-\U0001e943\U0001ee00-\U0001ee03\U0001ee05-\U0001ee1f\U0001ee21-\U0001ee22\U0001ee24\U0001ee27\U0001ee29-\U0001ee32\U0001ee34-\U0001ee37\U0001ee39\U0001ee3b\U0001ee42\U0001ee47\U0001ee49\U0001ee4b\U0001ee4d-\U0001ee4f\U0001ee51-\U0001ee52\U0001ee54\U0001ee57\U0001ee59\U0001ee5b\U0001ee5d\U0001ee5f\U0001ee61-\U0001ee62\U0001ee64\U0001ee67-\U0001ee6a\U0001ee6c-\U0001ee72\U0001ee74-\U0001ee77\U0001ee79-\U0001ee7c\U0001ee7e\U0001ee80-\U0001ee89\U0001ee8b-\U0001ee9b\U0001eea1-\U0001eea3\U0001eea5-\U0001eea9\U0001eeab-\U0001eebb\U00020000-\U0002a6d6\U0002a700-\U0002b734\U0002b740-\U0002b81d\U0002b820-\U0002cea1\U0002ceb0-\U0002ebe0\U0002f800-\U0002fa1d'

cats = ['Cc', 'Cf', 'Cn', 'Co', 'Cs', 'Ll', 'Lm', 'Lo', 'Lt', 'Lu', 'Mc', 'Me', 'Mn', 'Nd', 'Nl', 'No', 'Pc', 'Pd', 'Pe', 'Pf', 'Pi', 'Po', 'Ps', 'Sc', 'Sk', 'Sm', 'So', 'Zl', 'Zp', 'Zs']

# Generated from unidata 11.0.0

def combine(*args):
    return ''.join(globals()[cat] for cat in args)


def allexcept(*args):
    newcats = cats[:]
    for arg in args:
        newcats.remove(arg)
    return ''.join(globals()[cat] for cat in newcats)


def _handle_runs(char_list):  # pragma: no cover
    buf = []
    for c in char_list:
        if len(c) == 1:
            if buf and buf[-1][1] == chr(ord(c)-1):
                buf[-1] = (buf[-1][0], c)
            else:
                buf.append((c, c))
        else:
            buf.append((c, c))
    for a, b in buf:
        if a == b:
            yield a
        else:
            yield f'{a}-{b}'


if __name__ == '__main__':  # pragma: no cover
    import unicodedata

    categories = {'xid_start': [], 'xid_continue': []}

    with open(__file__, encoding='utf-8') as fp:
        content = fp.read()

    header = content[:content.find('Cc =')]
    footer = content[content.find("def combine("):]

    for code in range(0x110000):
        c = chr(code)
        cat = unicodedata.category(c)
        if ord(c) == 0xdc00:
            # Hack to avoid combining this combining with the preceding high
            # surrogate, 0xdbff, when doing a repr.
            c = '\\' + c
        elif ord(c) in (0x2d, 0x5b, 0x5c, 0x5d, 0x5e):
            # Escape regex metachars.
            c = '\\' + c
        categories.setdefault(cat, []).append(c)
        # XID_START and XID_CONTINUE are special categories used for matching
        # identifiers in Python 3.
        if c.isidentifier():
            categories['xid_start'].append(c)
        if ('a' + c).isidentifier():
            categories['xid_continue'].append(c)

    with open(__file__, 'w', encoding='utf-8') as fp:
        fp.write(header)

        for cat in sorted(categories):
            val = ''.join(_handle_runs(categories[cat]))
            fp.write(f'{cat} = {val!a}\n\n')

        cats = sorted(categories)
        cats.remove('xid_start')
        cats.remove('xid_continue')
        fp.write(f'cats = {cats!r}\n\n')

        fp.write(f'# Generated from unidata {unicodedata.unidata_version}\n\n')

        fp.write(footer)