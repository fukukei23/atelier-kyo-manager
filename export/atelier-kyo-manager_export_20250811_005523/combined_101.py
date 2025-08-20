
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\exceptions.py ===
"""Exceptions used throughout package.

This module MUST NOT try to import from anything within `pip._internal` to
operate. This is expected to be importable from any/all files within the
subpackage and, thus, should not depend on them.
"""

import configparser
import contextlib
import locale
import logging
import pathlib
import re
import sys
from itertools import chain, groupby, repeat
from typing import TYPE_CHECKING, Dict, Iterator, List, Literal, Optional, Union

from pip._vendor.packaging.requirements import InvalidRequirement
from pip._vendor.packaging.version import InvalidVersion
from pip._vendor.rich.console import Console, ConsoleOptions, RenderResult
from pip._vendor.rich.markup import escape
from pip._vendor.rich.text import Text

if TYPE_CHECKING:
    from hashlib import _Hash

    from pip._vendor.requests.models import Request, Response

    from pip._internal.metadata import BaseDistribution
    from pip._internal.models.link import Link
    from pip._internal.req.req_install import InstallRequirement

logger = logging.getLogger(__name__)


#
# Scaffolding
#
def _is_kebab_case(s: str) -> bool:
    return re.match(r"^[a-z]+(-[a-z]+)*$", s) is not None


def _prefix_with_indent(
    s: Union[Text, str],
    console: Console,
    *,
    prefix: str,
    indent: str,
) -> Text:
    if isinstance(s, Text):
        text = s
    else:
        text = console.render_str(s)

    return console.render_str(prefix, overflow="ignore") + console.render_str(
        f"\n{indent}", overflow="ignore"
    ).join(text.split(allow_blank=True))


class PipError(Exception):
    """The base pip error."""


class DiagnosticPipError(PipError):
    """An error, that presents diagnostic information to the user.

    This contains a bunch of logic, to enable pretty presentation of our error
    messages. Each error gets a unique reference. Each error can also include
    additional context, a hint and/or a note -- which are presented with the
    main error message in a consistent style.

    This is adapted from the error output styling in `sphinx-theme-builder`.
    """

    reference: str

    def __init__(
        self,
        *,
        kind: 'Literal["error", "warning"]' = "error",
        reference: Optional[str] = None,
        message: Union[str, Text],
        context: Optional[Union[str, Text]],
        hint_stmt: Optional[Union[str, Text]],
        note_stmt: Optional[Union[str, Text]] = None,
        link: Optional[str] = None,
    ) -> None:
        # Ensure a proper reference is provided.
        if reference is None:
            assert hasattr(self, "reference"), "error reference not provided!"
            reference = self.reference
        assert _is_kebab_case(reference), "error reference must be kebab-case!"

        self.kind = kind
        self.reference = reference

        self.message = message
        self.context = context

        self.note_stmt = note_stmt
        self.hint_stmt = hint_stmt

        self.link = link

        super().__init__(f"<{self.__class__.__name__}: {self.reference}>")

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__}("
            f"reference={self.reference!r}, "
            f"message={self.message!r}, "
            f"context={self.context!r}, "
            f"note_stmt={self.note_stmt!r}, "
            f"hint_stmt={self.hint_stmt!r}"
            ")>"
        )

    def __rich_console__(
        self,
        console: Console,
        options: ConsoleOptions,
    ) -> RenderResult:
        colour = "red" if self.kind == "error" else "yellow"

        yield f"[{colour} bold]{self.kind}[/]: [bold]{self.reference}[/]"
        yield ""

        if not options.ascii_only:
            # Present the main message, with relevant context indented.
            if self.context is not None:
                yield _prefix_with_indent(
                    self.message,
                    console,
                    prefix=f"[{colour}]×[/] ",
                    indent=f"[{colour}]│[/] ",
                )
                yield _prefix_with_indent(
                    self.context,
                    console,
                    prefix=f"[{colour}]╰─>[/] ",
                    indent=f"[{colour}]   [/] ",
                )
            else:
                yield _prefix_with_indent(
                    self.message,
                    console,
                    prefix="[red]×[/] ",
                    indent="  ",
                )
        else:
            yield self.message
            if self.context is not None:
                yield ""
                yield self.context

        if self.note_stmt is not None or self.hint_stmt is not None:
            yield ""

        if self.note_stmt is not None:
            yield _prefix_with_indent(
                self.note_stmt,
                console,
                prefix="[magenta bold]note[/]: ",
                indent="      ",
            )
        if self.hint_stmt is not None:
            yield _prefix_with_indent(
                self.hint_stmt,
                console,
                prefix="[cyan bold]hint[/]: ",
                indent="      ",
            )

        if self.link is not None:
            yield ""
            yield f"Link: {self.link}"


#
# Actual Errors
#
class ConfigurationError(PipError):
    """General exception in configuration"""


class InstallationError(PipError):
    """General exception during installation"""


class MissingPyProjectBuildRequires(DiagnosticPipError):
    """Raised when pyproject.toml has `build-system`, but no `build-system.requires`."""

    reference = "missing-pyproject-build-system-requires"

    def __init__(self, *, package: str) -> None:
        super().__init__(
            message=f"Can not process {escape(package)}",
            context=Text(
                "This package has an invalid pyproject.toml file.\n"
                "The [build-system] table is missing the mandatory `requires` key."
            ),
            note_stmt="This is an issue with the package mentioned above, not pip.",
            hint_stmt=Text("See PEP 518 for the detailed specification."),
        )


class InvalidPyProjectBuildRequires(DiagnosticPipError):
    """Raised when pyproject.toml an invalid `build-system.requires`."""

    reference = "invalid-pyproject-build-system-requires"

    def __init__(self, *, package: str, reason: str) -> None:
        super().__init__(
            message=f"Can not process {escape(package)}",
            context=Text(
                "This package has an invalid `build-system.requires` key in "
                f"pyproject.toml.\n{reason}"
            ),
            note_stmt="This is an issue with the package mentioned above, not pip.",
            hint_stmt=Text("See PEP 518 for the detailed specification."),
        )


class NoneMetadataError(PipError):
    """Raised when accessing a Distribution's "METADATA" or "PKG-INFO".

    This signifies an inconsistency, when the Distribution claims to have
    the metadata file (if not, raise ``FileNotFoundError`` instead), but is
    not actually able to produce its content. This may be due to permission
    errors.
    """

    def __init__(
        self,
        dist: "BaseDistribution",
        metadata_name: str,
    ) -> None:
        """
        :param dist: A Distribution object.
        :param metadata_name: The name of the metadata being accessed
            (can be "METADATA" or "PKG-INFO").
        """
        self.dist = dist
        self.metadata_name = metadata_name

    def __str__(self) -> str:
        # Use `dist` in the error message because its stringification
        # includes more information, like the version and location.
        return f"None {self.metadata_name} metadata found for distribution: {self.dist}"


class UserInstallationInvalid(InstallationError):
    """A --user install is requested on an environment without user site."""

    def __str__(self) -> str:
        return "User base directory is not specified"


class InvalidSchemeCombination(InstallationError):
    def __str__(self) -> str:
        before = ", ".join(str(a) for a in self.args[:-1])
        return f"Cannot set {before} and {self.args[-1]} together"


class DistributionNotFound(InstallationError):
    """Raised when a distribution cannot be found to satisfy a requirement"""


class RequirementsFileParseError(InstallationError):
    """Raised when a general error occurs parsing a requirements file line."""


class BestVersionAlreadyInstalled(PipError):
    """Raised when the most up-to-date version of a package is already
    installed."""


class BadCommand(PipError):
    """Raised when virtualenv or a command is not found"""


class CommandError(PipError):
    """Raised when there is an error in command-line arguments"""


class PreviousBuildDirError(PipError):
    """Raised when there's a previous conflicting build directory"""


class NetworkConnectionError(PipError):
    """HTTP connection error"""

    def __init__(
        self,
        error_msg: str,
        response: Optional["Response"] = None,
        request: Optional["Request"] = None,
    ) -> None:
        """
        Initialize NetworkConnectionError with  `request` and `response`
        objects.
        """
        self.response = response
        self.request = request
        self.error_msg = error_msg
        if (
            self.response is not None
            and not self.request
            and hasattr(response, "request")
        ):
            self.request = self.response.request
        super().__init__(error_msg, response, request)

    def __str__(self) -> str:
        return str(self.error_msg)


class InvalidWheelFilename(InstallationError):
    """Invalid wheel filename."""


class UnsupportedWheel(InstallationError):
    """Unsupported wheel."""


class InvalidWheel(InstallationError):
    """Invalid (e.g. corrupt) wheel."""

    def __init__(self, location: str, name: str):
        self.location = location
        self.name = name

    def __str__(self) -> str:
        return f"Wheel '{self.name}' located at {self.location} is invalid."


class MetadataInconsistent(InstallationError):
    """Built metadata contains inconsistent information.

    This is raised when the metadata contains values (e.g. name and version)
    that do not match the information previously obtained from sdist filename,
    user-supplied ``#egg=`` value, or an install requirement name.
    """

    def __init__(
        self, ireq: "InstallRequirement", field: str, f_val: str, m_val: str
    ) -> None:
        self.ireq = ireq
        self.field = field
        self.f_val = f_val
        self.m_val = m_val

    def __str__(self) -> str:
        return (
            f"Requested {self.ireq} has inconsistent {self.field}: "
            f"expected {self.f_val!r}, but metadata has {self.m_val!r}"
        )


class MetadataInvalid(InstallationError):
    """Metadata is invalid."""

    def __init__(self, ireq: "InstallRequirement", error: str) -> None:
        self.ireq = ireq
        self.error = error

    def __str__(self) -> str:
        return f"Requested {self.ireq} has invalid metadata: {self.error}"


class InstallationSubprocessError(DiagnosticPipError, InstallationError):
    """A subprocess call failed."""

    reference = "subprocess-exited-with-error"

    def __init__(
        self,
        *,
        command_description: str,
        exit_code: int,
        output_lines: Optional[List[str]],
    ) -> None:
        if output_lines is None:
            output_prompt = Text("See above for output.")
        else:
            output_prompt = (
                Text.from_markup(f"[red][{len(output_lines)} lines of output][/]\n")
                + Text("".join(output_lines))
                + Text.from_markup(R"[red]\[end of output][/]")
            )

        super().__init__(
            message=(
                f"[green]{escape(command_description)}[/] did not run successfully.\n"
                f"exit code: {exit_code}"
            ),
            context=output_prompt,
            hint_stmt=None,
            note_stmt=(
                "This error originates from a subprocess, and is likely not a "
                "problem with pip."
            ),
        )

        self.command_description = command_description
        self.exit_code = exit_code

    def __str__(self) -> str:
        return f"{self.command_description} exited with {self.exit_code}"


class MetadataGenerationFailed(InstallationSubprocessError, InstallationError):
    reference = "metadata-generation-failed"

    def __init__(
        self,
        *,
        package_details: str,
    ) -> None:
        super(InstallationSubprocessError, self).__init__(
            message="Encountered error while generating package metadata.",
            context=escape(package_details),
            hint_stmt="See above for details.",
            note_stmt="This is an issue with the package mentioned above, not pip.",
        )

    def __str__(self) -> str:
        return "metadata generation failed"


class HashErrors(InstallationError):
    """Multiple HashError instances rolled into one for reporting"""

    def __init__(self) -> None:
        self.errors: List[HashError] = []

    def append(self, error: "HashError") -> None:
        self.errors.append(error)

    def __str__(self) -> str:
        lines = []
        self.errors.sort(key=lambda e: e.order)
        for cls, errors_of_cls in groupby(self.errors, lambda e: e.__class__):
            lines.append(cls.head)
            lines.extend(e.body() for e in errors_of_cls)
        if lines:
            return "\n".join(lines)
        return ""

    def __bool__(self) -> bool:
        return bool(self.errors)


class HashError(InstallationError):
    """
    A failure to verify a package against known-good hashes

    :cvar order: An int sorting hash exception classes by difficulty of
        recovery (lower being harder), so the user doesn't bother fretting
        about unpinned packages when he has deeper issues, like VCS
        dependencies, to deal with. Also keeps error reports in a
        deterministic order.
    :cvar head: A section heading for display above potentially many
        exceptions of this kind
    :ivar req: The InstallRequirement that triggered this error. This is
        pasted on after the exception is instantiated, because it's not
        typically available earlier.

    """

    req: Optional["InstallRequirement"] = None
    head = ""
    order: int = -1

    def body(self) -> str:
        """Return a summary of me for display under the heading.

        This default implementation simply prints a description of the
        triggering requirement.

        :param req: The InstallRequirement that provoked this error, with
            its link already populated by the resolver's _populate_link().

        """
        return f"    {self._requirement_name()}"

    def __str__(self) -> str:
        return f"{self.head}\n{self.body()}"

    def _requirement_name(self) -> str:
        """Return a description of the requirement that triggered me.

        This default implementation returns long description of the req, with
        line numbers

        """
        return str(self.req) if self.req else "unknown package"


class VcsHashUnsupported(HashError):
    """A hash was provided for a version-control-system-based requirement, but
    we don't have a method for hashing those."""

    order = 0
    head = (
        "Can't verify hashes for these requirements because we don't "
        "have a way to hash version control repositories:"
    )


class DirectoryUrlHashUnsupported(HashError):
    """A hash was provided for a version-control-system-based requirement, but
    we don't have a method for hashing those."""

    order = 1
    head = (
        "Can't verify hashes for these file:// requirements because they "
        "point to directories:"
    )


class HashMissing(HashError):
    """A hash was needed for a requirement but is absent."""

    order = 2
    head = (
        "Hashes are required in --require-hashes mode, but they are "
        "missing from some requirements. Here is a list of those "
        "requirements along with the hashes their downloaded archives "
        "actually had. Add lines like these to your requirements files to "
        "prevent tampering. (If you did not enable --require-hashes "
        "manually, note that it turns on automatically when any package "
        "has a hash.)"
    )

    def __init__(self, gotten_hash: str) -> None:
        """
        :param gotten_hash: The hash of the (possibly malicious) archive we
            just downloaded
        """
        self.gotten_hash = gotten_hash

    def body(self) -> str:
        # Dodge circular import.
        from pip._internal.utils.hashes import FAVORITE_HASH

        package = None
        if self.req:
            # In the case of URL-based requirements, display the original URL
            # seen in the requirements file rather than the package name,
            # so the output can be directly copied into the requirements file.
            package = (
                self.req.original_link
                if self.req.is_direct
                # In case someone feeds something downright stupid
                # to InstallRequirement's constructor.
                else getattr(self.req, "req", None)
            )
        return "    {} --hash={}:{}".format(
            package or "unknown package", FAVORITE_HASH, self.gotten_hash
        )


class HashUnpinned(HashError):
    """A requirement had a hash specified but was not pinned to a specific
    version."""

    order = 3
    head = (
        "In --require-hashes mode, all requirements must have their "
        "versions pinned with ==. These do not:"
    )


class HashMismatch(HashError):
    """
    Distribution file hash values don't match.

    :ivar package_name: The name of the package that triggered the hash
        mismatch. Feel free to write to this after the exception is raise to
        improve its error message.

    """

    order = 4
    head = (
        "THESE PACKAGES DO NOT MATCH THE HASHES FROM THE REQUIREMENTS "
        "FILE. If you have updated the package versions, please update "
        "the hashes. Otherwise, examine the package contents carefully; "
        "someone may have tampered with them."
    )

    def __init__(self, allowed: Dict[str, List[str]], gots: Dict[str, "_Hash"]) -> None:
        """
        :param allowed: A dict of algorithm names pointing to lists of allowed
            hex digests
        :param gots: A dict of algorithm names pointing to hashes we
            actually got from the files under suspicion
        """
        self.allowed = allowed
        self.gots = gots

    def body(self) -> str:
        return f"    {self._requirement_name()}:\n{self._hash_comparison()}"

    def _hash_comparison(self) -> str:
        """
        Return a comparison of actual and expected hash values.

        Example::

               Expected sha256 abcdeabcdeabcdeabcdeabcdeabcdeabcdeabcdeabcde
                            or 123451234512345123451234512345123451234512345
                    Got        bcdefbcdefbcdefbcdefbcdefbcdefbcdefbcdefbcdef

        """

        def hash_then_or(hash_name: str) -> "chain[str]":
            # For now, all the decent hashes have 6-char names, so we can get
            # away with hard-coding space literals.
            return chain([hash_name], repeat("    or"))

        lines: List[str] = []
        for hash_name, expecteds in self.allowed.items():
            prefix = hash_then_or(hash_name)
            lines.extend((f"        Expected {next(prefix)} {e}") for e in expecteds)
            lines.append(
                f"             Got        {self.gots[hash_name].hexdigest()}\n"
            )
        return "\n".join(lines)


class UnsupportedPythonVersion(InstallationError):
    """Unsupported python version according to Requires-Python package
    metadata."""


class ConfigurationFileCouldNotBeLoaded(ConfigurationError):
    """When there are errors while loading a configuration file"""

    def __init__(
        self,
        reason: str = "could not be loaded",
        fname: Optional[str] = None,
        error: Optional[configparser.Error] = None,
    ) -> None:
        super().__init__(error)
        self.reason = reason
        self.fname = fname
        self.error = error

    def __str__(self) -> str:
        if self.fname is not None:
            message_part = f" in {self.fname}."
        else:
            assert self.error is not None
            message_part = f".\n{self.error}\n"
        return f"Configuration file {self.reason}{message_part}"


_DEFAULT_EXTERNALLY_MANAGED_ERROR = f"""\
The Python environment under {sys.prefix} is managed externally, and may not be
manipulated by the user. Please use specific tooling from the distributor of
the Python installation to interact with this environment instead.
"""


class ExternallyManagedEnvironment(DiagnosticPipError):
    """The current environment is externally managed.

    This is raised when the current environment is externally managed, as
    defined by `PEP 668`_. The ``EXTERNALLY-MANAGED`` configuration is checked
    and displayed when the error is bubbled up to the user.

    :param error: The error message read from ``EXTERNALLY-MANAGED``.
    """

    reference = "externally-managed-environment"

    def __init__(self, error: Optional[str]) -> None:
        if error is None:
            context = Text(_DEFAULT_EXTERNALLY_MANAGED_ERROR)
        else:
            context = Text(error)
        super().__init__(
            message="This environment is externally managed",
            context=context,
            note_stmt=(
                "If you believe this is a mistake, please contact your "
                "Python installation or OS distribution provider. "
                "You can override this, at the risk of breaking your Python "
                "installation or OS, by passing --break-system-packages."
            ),
            hint_stmt=Text("See PEP 668 for the detailed specification."),
        )

    @staticmethod
    def _iter_externally_managed_error_keys() -> Iterator[str]:
        # LC_MESSAGES is in POSIX, but not the C standard. The most common
        # platform that does not implement this category is Windows, where
        # using other categories for console message localization is equally
        # unreliable, so we fall back to the locale-less vendor message. This
        # can always be re-evaluated when a vendor proposes a new alternative.
        try:
            category = locale.LC_MESSAGES
        except AttributeError:
            lang: Optional[str] = None
        else:
            lang, _ = locale.getlocale(category)
        if lang is not None:
            yield f"Error-{lang}"
            for sep in ("-", "_"):
                before, found, _ = lang.partition(sep)
                if not found:
                    continue
                yield f"Error-{before}"
        yield "Error"

    @classmethod
    def from_config(
        cls,
        config: Union[pathlib.Path, str],
    ) -> "ExternallyManagedEnvironment":
        parser = configparser.ConfigParser(interpolation=None)
        try:
            parser.read(config, encoding="utf-8")
            section = parser["externally-managed"]
            for key in cls._iter_externally_managed_error_keys():
                with contextlib.suppress(KeyError):
                    return cls(section[key])
        except KeyError:
            pass
        except (OSError, UnicodeDecodeError, configparser.ParsingError):
            from pip._internal.utils._log import VERBOSE

            exc_info = logger.isEnabledFor(VERBOSE)
            logger.warning("Failed to read %s", config, exc_info=exc_info)
        return cls(None)


class UninstallMissingRecord(DiagnosticPipError):
    reference = "uninstall-no-record-file"

    def __init__(self, *, distribution: "BaseDistribution") -> None:
        installer = distribution.installer
        if not installer or installer == "pip":
            dep = f"{distribution.raw_name}=={distribution.version}"
            hint = Text.assemble(
                "You might be able to recover from this via: ",
                (f"pip install --force-reinstall --no-deps {dep}", "green"),
            )
        else:
            hint = Text(
                f"The package was installed by {installer}. "
                "You should check if it can uninstall the package."
            )

        super().__init__(
            message=Text(f"Cannot uninstall {distribution}"),
            context=(
                "The package's contents are unknown: "
                f"no RECORD file was found for {distribution.raw_name}."
            ),
            hint_stmt=hint,
        )


class LegacyDistutilsInstall(DiagnosticPipError):
    reference = "uninstall-distutils-installed-package"

    def __init__(self, *, distribution: "BaseDistribution") -> None:
        super().__init__(
            message=Text(f"Cannot uninstall {distribution}"),
            context=(
                "It is a distutils installed project and thus we cannot accurately "
                "determine which files belong to it which would lead to only a partial "
                "uninstall."
            ),
            hint_stmt=None,
        )


class InvalidInstalledPackage(DiagnosticPipError):
    reference = "invalid-installed-package"

    def __init__(
        self,
        *,
        dist: "BaseDistribution",
        invalid_exc: Union[InvalidRequirement, InvalidVersion],
    ) -> None:
        installed_location = dist.installed_location

        if isinstance(invalid_exc, InvalidRequirement):
            invalid_type = "requirement"
        else:
            invalid_type = "version"

        super().__init__(
            message=Text(
                f"Cannot process installed package {dist} "
                + (f"in {installed_location!r} " if installed_location else "")
                + f"because it has an invalid {invalid_type}:\n{invalid_exc.args[0]}"
            ),
            context=(
                "Starting with pip 24.1, packages with invalid "
                f"{invalid_type}s can not be processed."
            ),
            hint_stmt="To proceed this package must be uninstalled.",
        )


class IncompleteDownloadError(DiagnosticPipError):
    """Raised when the downloader receives fewer bytes than advertised
    in the Content-Length header."""

    reference = "incomplete-download"

    def __init__(
        self, link: "Link", received: int, expected: int, *, retries: int
    ) -> None:
        # Dodge circular import.
        from pip._internal.utils.misc import format_size

        download_status = f"{format_size(received)}/{format_size(expected)}"
        if retries:
            retry_status = f"after {retries} attempts "
            hint = "Use --resume-retries to configure resume attempt limit."
        else:
            retry_status = ""
            hint = "Consider using --resume-retries to enable download resumption."
        message = Text(
            f"Download failed {retry_status}because not enough bytes "
            f"were received ({download_status})"
        )

        super().__init__(
            message=message,
            context=f"URL: {link.redacted_url}",
            hint_stmt=hint,
            note_stmt="This is an issue with network connectivity, not pip.",
        )


class ResolutionTooDeepError(DiagnosticPipError):
    """Raised when the dependency resolver exceeds the maximum recursion depth."""

    reference = "resolution-too-deep"

    def __init__(self) -> None:
        super().__init__(
            message="Dependency resolution exceeded maximum depth",
            context=(
                "Pip cannot resolve the current dependencies as the dependency graph "
                "is too complex for pip to solve efficiently."
            ),
            hint_stmt=(
                "Try adding lower bounds to constrain your dependencies, "
                "for example: 'package>=2.0.0' instead of just 'package'. "
            ),
            link="https://pip.pypa.io/en/stable/topics/dependency-resolution/#handling-resolution-too-deep-errors",
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\packaging\metadata.py ===
from __future__ import annotations

import email.feedparser
import email.header
import email.message
import email.parser
import email.policy
import pathlib
import sys
import typing
from typing import (
    Any,
    Callable,
    Generic,
    Literal,
    TypedDict,
    cast,
)

from . import licenses, requirements, specifiers, utils
from . import version as version_module
from .licenses import NormalizedLicenseExpression

T = typing.TypeVar("T")


if sys.version_info >= (3, 11):  # pragma: no cover
    ExceptionGroup = ExceptionGroup
else:  # pragma: no cover

    class ExceptionGroup(Exception):
        """A minimal implementation of :external:exc:`ExceptionGroup` from Python 3.11.

        If :external:exc:`ExceptionGroup` is already defined by Python itself,
        that version is used instead.
        """

        message: str
        exceptions: list[Exception]

        def __init__(self, message: str, exceptions: list[Exception]) -> None:
            self.message = message
            self.exceptions = exceptions

        def __repr__(self) -> str:
            return f"{self.__class__.__name__}({self.message!r}, {self.exceptions!r})"


class InvalidMetadata(ValueError):
    """A metadata field contains invalid data."""

    field: str
    """The name of the field that contains invalid data."""

    def __init__(self, field: str, message: str) -> None:
        self.field = field
        super().__init__(message)


# The RawMetadata class attempts to make as few assumptions about the underlying
# serialization formats as possible. The idea is that as long as a serialization
# formats offer some very basic primitives in *some* way then we can support
# serializing to and from that format.
class RawMetadata(TypedDict, total=False):
    """A dictionary of raw core metadata.

    Each field in core metadata maps to a key of this dictionary (when data is
    provided). The key is lower-case and underscores are used instead of dashes
    compared to the equivalent core metadata field. Any core metadata field that
    can be specified multiple times or can hold multiple values in a single
    field have a key with a plural name. See :class:`Metadata` whose attributes
    match the keys of this dictionary.

    Core metadata fields that can be specified multiple times are stored as a
    list or dict depending on which is appropriate for the field. Any fields
    which hold multiple values in a single field are stored as a list.

    """

    # Metadata 1.0 - PEP 241
    metadata_version: str
    name: str
    version: str
    platforms: list[str]
    summary: str
    description: str
    keywords: list[str]
    home_page: str
    author: str
    author_email: str
    license: str

    # Metadata 1.1 - PEP 314
    supported_platforms: list[str]
    download_url: str
    classifiers: list[str]
    requires: list[str]
    provides: list[str]
    obsoletes: list[str]

    # Metadata 1.2 - PEP 345
    maintainer: str
    maintainer_email: str
    requires_dist: list[str]
    provides_dist: list[str]
    obsoletes_dist: list[str]
    requires_python: str
    requires_external: list[str]
    project_urls: dict[str, str]

    # Metadata 2.0
    # PEP 426 attempted to completely revamp the metadata format
    # but got stuck without ever being able to build consensus on
    # it and ultimately ended up withdrawn.
    #
    # However, a number of tools had started emitting METADATA with
    # `2.0` Metadata-Version, so for historical reasons, this version
    # was skipped.

    # Metadata 2.1 - PEP 566
    description_content_type: str
    provides_extra: list[str]

    # Metadata 2.2 - PEP 643
    dynamic: list[str]

    # Metadata 2.3 - PEP 685
    # No new fields were added in PEP 685, just some edge case were
    # tightened up to provide better interoptability.

    # Metadata 2.4 - PEP 639
    license_expression: str
    license_files: list[str]


_STRING_FIELDS = {
    "author",
    "author_email",
    "description",
    "description_content_type",
    "download_url",
    "home_page",
    "license",
    "license_expression",
    "maintainer",
    "maintainer_email",
    "metadata_version",
    "name",
    "requires_python",
    "summary",
    "version",
}

_LIST_FIELDS = {
    "classifiers",
    "dynamic",
    "license_files",
    "obsoletes",
    "obsoletes_dist",
    "platforms",
    "provides",
    "provides_dist",
    "provides_extra",
    "requires",
    "requires_dist",
    "requires_external",
    "supported_platforms",
}

_DICT_FIELDS = {
    "project_urls",
}


def _parse_keywords(data: str) -> list[str]:
    """Split a string of comma-separated keywords into a list of keywords."""
    return [k.strip() for k in data.split(",")]


def _parse_project_urls(data: list[str]) -> dict[str, str]:
    """Parse a list of label/URL string pairings separated by a comma."""
    urls = {}
    for pair in data:
        # Our logic is slightly tricky here as we want to try and do
        # *something* reasonable with malformed data.
        #
        # The main thing that we have to worry about, is data that does
        # not have a ',' at all to split the label from the Value. There
        # isn't a singular right answer here, and we will fail validation
        # later on (if the caller is validating) so it doesn't *really*
        # matter, but since the missing value has to be an empty str
        # and our return value is dict[str, str], if we let the key
        # be the missing value, then they'd have multiple '' values that
        # overwrite each other in a accumulating dict.
        #
        # The other potentional issue is that it's possible to have the
        # same label multiple times in the metadata, with no solid "right"
        # answer with what to do in that case. As such, we'll do the only
        # thing we can, which is treat the field as unparseable and add it
        # to our list of unparsed fields.
        parts = [p.strip() for p in pair.split(",", 1)]
        parts.extend([""] * (max(0, 2 - len(parts))))  # Ensure 2 items

        # TODO: The spec doesn't say anything about if the keys should be
        #       considered case sensitive or not... logically they should
        #       be case-preserving and case-insensitive, but doing that
        #       would open up more cases where we might have duplicate
        #       entries.
        label, url = parts
        if label in urls:
            # The label already exists in our set of urls, so this field
            # is unparseable, and we can just add the whole thing to our
            # unparseable data and stop processing it.
            raise KeyError("duplicate labels in project urls")
        urls[label] = url

    return urls


def _get_payload(msg: email.message.Message, source: bytes | str) -> str:
    """Get the body of the message."""
    # If our source is a str, then our caller has managed encodings for us,
    # and we don't need to deal with it.
    if isinstance(source, str):
        payload = msg.get_payload()
        assert isinstance(payload, str)
        return payload
    # If our source is a bytes, then we're managing the encoding and we need
    # to deal with it.
    else:
        bpayload = msg.get_payload(decode=True)
        assert isinstance(bpayload, bytes)
        try:
            return bpayload.decode("utf8", "strict")
        except UnicodeDecodeError as exc:
            raise ValueError("payload in an invalid encoding") from exc


# The various parse_FORMAT functions here are intended to be as lenient as
# possible in their parsing, while still returning a correctly typed
# RawMetadata.
#
# To aid in this, we also generally want to do as little touching of the
# data as possible, except where there are possibly some historic holdovers
# that make valid data awkward to work with.
#
# While this is a lower level, intermediate format than our ``Metadata``
# class, some light touch ups can make a massive difference in usability.

# Map METADATA fields to RawMetadata.
_EMAIL_TO_RAW_MAPPING = {
    "author": "author",
    "author-email": "author_email",
    "classifier": "classifiers",
    "description": "description",
    "description-content-type": "description_content_type",
    "download-url": "download_url",
    "dynamic": "dynamic",
    "home-page": "home_page",
    "keywords": "keywords",
    "license": "license",
    "license-expression": "license_expression",
    "license-file": "license_files",
    "maintainer": "maintainer",
    "maintainer-email": "maintainer_email",
    "metadata-version": "metadata_version",
    "name": "name",
    "obsoletes": "obsoletes",
    "obsoletes-dist": "obsoletes_dist",
    "platform": "platforms",
    "project-url": "project_urls",
    "provides": "provides",
    "provides-dist": "provides_dist",
    "provides-extra": "provides_extra",
    "requires": "requires",
    "requires-dist": "requires_dist",
    "requires-external": "requires_external",
    "requires-python": "requires_python",
    "summary": "summary",
    "supported-platform": "supported_platforms",
    "version": "version",
}
_RAW_TO_EMAIL_MAPPING = {raw: email for email, raw in _EMAIL_TO_RAW_MAPPING.items()}


def parse_email(data: bytes | str) -> tuple[RawMetadata, dict[str, list[str]]]:
    """Parse a distribution's metadata stored as email headers (e.g. from ``METADATA``).

    This function returns a two-item tuple of dicts. The first dict is of
    recognized fields from the core metadata specification. Fields that can be
    parsed and translated into Python's built-in types are converted
    appropriately. All other fields are left as-is. Fields that are allowed to
    appear multiple times are stored as lists.

    The second dict contains all other fields from the metadata. This includes
    any unrecognized fields. It also includes any fields which are expected to
    be parsed into a built-in type but were not formatted appropriately. Finally,
    any fields that are expected to appear only once but are repeated are
    included in this dict.

    """
    raw: dict[str, str | list[str] | dict[str, str]] = {}
    unparsed: dict[str, list[str]] = {}

    if isinstance(data, str):
        parsed = email.parser.Parser(policy=email.policy.compat32).parsestr(data)
    else:
        parsed = email.parser.BytesParser(policy=email.policy.compat32).parsebytes(data)

    # We have to wrap parsed.keys() in a set, because in the case of multiple
    # values for a key (a list), the key will appear multiple times in the
    # list of keys, but we're avoiding that by using get_all().
    for name in frozenset(parsed.keys()):
        # Header names in RFC are case insensitive, so we'll normalize to all
        # lower case to make comparisons easier.
        name = name.lower()

        # We use get_all() here, even for fields that aren't multiple use,
        # because otherwise someone could have e.g. two Name fields, and we
        # would just silently ignore it rather than doing something about it.
        headers = parsed.get_all(name) or []

        # The way the email module works when parsing bytes is that it
        # unconditionally decodes the bytes as ascii using the surrogateescape
        # handler. When you pull that data back out (such as with get_all() ),
        # it looks to see if the str has any surrogate escapes, and if it does
        # it wraps it in a Header object instead of returning the string.
        #
        # As such, we'll look for those Header objects, and fix up the encoding.
        value = []
        # Flag if we have run into any issues processing the headers, thus
        # signalling that the data belongs in 'unparsed'.
        valid_encoding = True
        for h in headers:
            # It's unclear if this can return more types than just a Header or
            # a str, so we'll just assert here to make sure.
            assert isinstance(h, (email.header.Header, str))

            # If it's a header object, we need to do our little dance to get
            # the real data out of it. In cases where there is invalid data
            # we're going to end up with mojibake, but there's no obvious, good
            # way around that without reimplementing parts of the Header object
            # ourselves.
            #
            # That should be fine since, if mojibacked happens, this key is
            # going into the unparsed dict anyways.
            if isinstance(h, email.header.Header):
                # The Header object stores it's data as chunks, and each chunk
                # can be independently encoded, so we'll need to check each
                # of them.
                chunks: list[tuple[bytes, str | None]] = []
                for bin, encoding in email.header.decode_header(h):
                    try:
                        bin.decode("utf8", "strict")
                    except UnicodeDecodeError:
                        # Enable mojibake.
                        encoding = "latin1"
                        valid_encoding = False
                    else:
                        encoding = "utf8"
                    chunks.append((bin, encoding))

                # Turn our chunks back into a Header object, then let that
                # Header object do the right thing to turn them into a
                # string for us.
                value.append(str(email.header.make_header(chunks)))
            # This is already a string, so just add it.
            else:
                value.append(h)

        # We've processed all of our values to get them into a list of str,
        # but we may have mojibake data, in which case this is an unparsed
        # field.
        if not valid_encoding:
            unparsed[name] = value
            continue

        raw_name = _EMAIL_TO_RAW_MAPPING.get(name)
        if raw_name is None:
            # This is a bit of a weird situation, we've encountered a key that
            # we don't know what it means, so we don't know whether it's meant
            # to be a list or not.
            #
            # Since we can't really tell one way or another, we'll just leave it
            # as a list, even though it may be a single item list, because that's
            # what makes the most sense for email headers.
            unparsed[name] = value
            continue

        # If this is one of our string fields, then we'll check to see if our
        # value is a list of a single item. If it is then we'll assume that
        # it was emitted as a single string, and unwrap the str from inside
        # the list.
        #
        # If it's any other kind of data, then we haven't the faintest clue
        # what we should parse it as, and we have to just add it to our list
        # of unparsed stuff.
        if raw_name in _STRING_FIELDS and len(value) == 1:
            raw[raw_name] = value[0]
        # If this is one of our list of string fields, then we can just assign
        # the value, since email *only* has strings, and our get_all() call
        # above ensures that this is a list.
        elif raw_name in _LIST_FIELDS:
            raw[raw_name] = value
        # Special Case: Keywords
        # The keywords field is implemented in the metadata spec as a str,
        # but it conceptually is a list of strings, and is serialized using
        # ", ".join(keywords), so we'll do some light data massaging to turn
        # this into what it logically is.
        elif raw_name == "keywords" and len(value) == 1:
            raw[raw_name] = _parse_keywords(value[0])
        # Special Case: Project-URL
        # The project urls is implemented in the metadata spec as a list of
        # specially-formatted strings that represent a key and a value, which
        # is fundamentally a mapping, however the email format doesn't support
        # mappings in a sane way, so it was crammed into a list of strings
        # instead.
        #
        # We will do a little light data massaging to turn this into a map as
        # it logically should be.
        elif raw_name == "project_urls":
            try:
                raw[raw_name] = _parse_project_urls(value)
            except KeyError:
                unparsed[name] = value
        # Nothing that we've done has managed to parse this, so it'll just
        # throw it in our unparseable data and move on.
        else:
            unparsed[name] = value

    # We need to support getting the Description from the message payload in
    # addition to getting it from the the headers. This does mean, though, there
    # is the possibility of it being set both ways, in which case we put both
    # in 'unparsed' since we don't know which is right.
    try:
        payload = _get_payload(parsed, data)
    except ValueError:
        unparsed.setdefault("description", []).append(
            parsed.get_payload(decode=isinstance(data, bytes))  # type: ignore[call-overload]
        )
    else:
        if payload:
            # Check to see if we've already got a description, if so then both
            # it, and this body move to unparseable.
            if "description" in raw:
                description_header = cast(str, raw.pop("description"))
                unparsed.setdefault("description", []).extend(
                    [description_header, payload]
                )
            elif "description" in unparsed:
                unparsed["description"].append(payload)
            else:
                raw["description"] = payload

    # We need to cast our `raw` to a metadata, because a TypedDict only support
    # literal key names, but we're computing our key names on purpose, but the
    # way this function is implemented, our `TypedDict` can only have valid key
    # names.
    return cast(RawMetadata, raw), unparsed


_NOT_FOUND = object()


# Keep the two values in sync.
_VALID_METADATA_VERSIONS = ["1.0", "1.1", "1.2", "2.1", "2.2", "2.3", "2.4"]
_MetadataVersion = Literal["1.0", "1.1", "1.2", "2.1", "2.2", "2.3", "2.4"]

_REQUIRED_ATTRS = frozenset(["metadata_version", "name", "version"])


class _Validator(Generic[T]):
    """Validate a metadata field.

    All _process_*() methods correspond to a core metadata field. The method is
    called with the field's raw value. If the raw value is valid it is returned
    in its "enriched" form (e.g. ``version.Version`` for the ``Version`` field).
    If the raw value is invalid, :exc:`InvalidMetadata` is raised (with a cause
    as appropriate).
    """

    name: str
    raw_name: str
    added: _MetadataVersion

    def __init__(
        self,
        *,
        added: _MetadataVersion = "1.0",
    ) -> None:
        self.added = added

    def __set_name__(self, _owner: Metadata, name: str) -> None:
        self.name = name
        self.raw_name = _RAW_TO_EMAIL_MAPPING[name]

    def __get__(self, instance: Metadata, _owner: type[Metadata]) -> T:
        # With Python 3.8, the caching can be replaced with functools.cached_property().
        # No need to check the cache as attribute lookup will resolve into the
        # instance's __dict__ before __get__ is called.
        cache = instance.__dict__
        value = instance._raw.get(self.name)

        # To make the _process_* methods easier, we'll check if the value is None
        # and if this field is NOT a required attribute, and if both of those
        # things are true, we'll skip the the converter. This will mean that the
        # converters never have to deal with the None union.
        if self.name in _REQUIRED_ATTRS or value is not None:
            try:
                converter: Callable[[Any], T] = getattr(self, f"_process_{self.name}")
            except AttributeError:
                pass
            else:
                value = converter(value)

        cache[self.name] = value
        try:
            del instance._raw[self.name]  # type: ignore[misc]
        except KeyError:
            pass

        return cast(T, value)

    def _invalid_metadata(
        self, msg: str, cause: Exception | None = None
    ) -> InvalidMetadata:
        exc = InvalidMetadata(
            self.raw_name, msg.format_map({"field": repr(self.raw_name)})
        )
        exc.__cause__ = cause
        return exc

    def _process_metadata_version(self, value: str) -> _MetadataVersion:
        # Implicitly makes Metadata-Version required.
        if value not in _VALID_METADATA_VERSIONS:
            raise self._invalid_metadata(f"{value!r} is not a valid metadata version")
        return cast(_MetadataVersion, value)

    def _process_name(self, value: str) -> str:
        if not value:
            raise self._invalid_metadata("{field} is a required field")
        # Validate the name as a side-effect.
        try:
            utils.canonicalize_name(value, validate=True)
        except utils.InvalidName as exc:
            raise self._invalid_metadata(
                f"{value!r} is invalid for {{field}}", cause=exc
            ) from exc
        else:
            return value

    def _process_version(self, value: str) -> version_module.Version:
        if not value:
            raise self._invalid_metadata("{field} is a required field")
        try:
            return version_module.parse(value)
        except version_module.InvalidVersion as exc:
            raise self._invalid_metadata(
                f"{value!r} is invalid for {{field}}", cause=exc
            ) from exc

    def _process_summary(self, value: str) -> str:
        """Check the field contains no newlines."""
        if "\n" in value:
            raise self._invalid_metadata("{field} must be a single line")
        return value

    def _process_description_content_type(self, value: str) -> str:
        content_types = {"text/plain", "text/x-rst", "text/markdown"}
        message = email.message.EmailMessage()
        message["content-type"] = value

        content_type, parameters = (
            # Defaults to `text/plain` if parsing failed.
            message.get_content_type().lower(),
            message["content-type"].params,
        )
        # Check if content-type is valid or defaulted to `text/plain` and thus was
        # not parseable.
        if content_type not in content_types or content_type not in value.lower():
            raise self._invalid_metadata(
                f"{{field}} must be one of {list(content_types)}, not {value!r}"
            )

        charset = parameters.get("charset", "UTF-8")
        if charset != "UTF-8":
            raise self._invalid_metadata(
                f"{{field}} can only specify the UTF-8 charset, not {list(charset)}"
            )

        markdown_variants = {"GFM", "CommonMark"}
        variant = parameters.get("variant", "GFM")  # Use an acceptable default.
        if content_type == "text/markdown" and variant not in markdown_variants:
            raise self._invalid_metadata(
                f"valid Markdown variants for {{field}} are {list(markdown_variants)}, "
                f"not {variant!r}",
            )
        return value

    def _process_dynamic(self, value: list[str]) -> list[str]:
        for dynamic_field in map(str.lower, value):
            if dynamic_field in {"name", "version", "metadata-version"}:
                raise self._invalid_metadata(
                    f"{dynamic_field!r} is not allowed as a dynamic field"
                )
            elif dynamic_field not in _EMAIL_TO_RAW_MAPPING:
                raise self._invalid_metadata(
                    f"{dynamic_field!r} is not a valid dynamic field"
                )
        return list(map(str.lower, value))

    def _process_provides_extra(
        self,
        value: list[str],
    ) -> list[utils.NormalizedName]:
        normalized_names = []
        try:
            for name in value:
                normalized_names.append(utils.canonicalize_name(name, validate=True))
        except utils.InvalidName as exc:
            raise self._invalid_metadata(
                f"{name!r} is invalid for {{field}}", cause=exc
            ) from exc
        else:
            return normalized_names

    def _process_requires_python(self, value: str) -> specifiers.SpecifierSet:
        try:
            return specifiers.SpecifierSet(value)
        except specifiers.InvalidSpecifier as exc:
            raise self._invalid_metadata(
                f"{value!r} is invalid for {{field}}", cause=exc
            ) from exc

    def _process_requires_dist(
        self,
        value: list[str],
    ) -> list[requirements.Requirement]:
        reqs = []
        try:
            for req in value:
                reqs.append(requirements.Requirement(req))
        except requirements.InvalidRequirement as exc:
            raise self._invalid_metadata(
                f"{req!r} is invalid for {{field}}", cause=exc
            ) from exc
        else:
            return reqs

    def _process_license_expression(
        self, value: str
    ) -> NormalizedLicenseExpression | None:
        try:
            return licenses.canonicalize_license_expression(value)
        except ValueError as exc:
            raise self._invalid_metadata(
                f"{value!r} is invalid for {{field}}", cause=exc
            ) from exc

    def _process_license_files(self, value: list[str]) -> list[str]:
        paths = []
        for path in value:
            if ".." in path:
                raise self._invalid_metadata(
                    f"{path!r} is invalid for {{field}}, "
                    "parent directory indicators are not allowed"
                )
            if "*" in path:
                raise self._invalid_metadata(
                    f"{path!r} is invalid for {{field}}, paths must be resolved"
                )
            if (
                pathlib.PurePosixPath(path).is_absolute()
                or pathlib.PureWindowsPath(path).is_absolute()
            ):
                raise self._invalid_metadata(
                    f"{path!r} is invalid for {{field}}, paths must be relative"
                )
            if pathlib.PureWindowsPath(path).as_posix() != path:
                raise self._invalid_metadata(
                    f"{path!r} is invalid for {{field}}, paths must use '/' delimiter"
                )
            paths.append(path)
        return paths


class Metadata:
    """Representation of distribution metadata.

    Compared to :class:`RawMetadata`, this class provides objects representing
    metadata fields instead of only using built-in types. Any invalid metadata
    will cause :exc:`InvalidMetadata` to be raised (with a
    :py:attr:`~BaseException.__cause__` attribute as appropriate).
    """

    _raw: RawMetadata

    @classmethod
    def from_raw(cls, data: RawMetadata, *, validate: bool = True) -> Metadata:
        """Create an instance from :class:`RawMetadata`.

        If *validate* is true, all metadata will be validated. All exceptions
        related to validation will be gathered and raised as an :class:`ExceptionGroup`.
        """
        ins = cls()
        ins._raw = data.copy()  # Mutations occur due to caching enriched values.

        if validate:
            exceptions: list[Exception] = []
            try:
                metadata_version = ins.metadata_version
                metadata_age = _VALID_METADATA_VERSIONS.index(metadata_version)
            except InvalidMetadata as metadata_version_exc:
                exceptions.append(metadata_version_exc)
                metadata_version = None

            # Make sure to check for the fields that are present, the required
            # fields (so their absence can be reported).
            fields_to_check = frozenset(ins._raw) | _REQUIRED_ATTRS
            # Remove fields that have already been checked.
            fields_to_check -= {"metadata_version"}

            for key in fields_to_check:
                try:
                    if metadata_version:
                        # Can't use getattr() as that triggers descriptor protocol which
                        # will fail due to no value for the instance argument.
                        try:
                            field_metadata_version = cls.__dict__[key].added
                        except KeyError:
                            exc = InvalidMetadata(key, f"unrecognized field: {key!r}")
                            exceptions.append(exc)
                            continue
                        field_age = _VALID_METADATA_VERSIONS.index(
                            field_metadata_version
                        )
                        if field_age > metadata_age:
                            field = _RAW_TO_EMAIL_MAPPING[key]
                            exc = InvalidMetadata(
                                field,
                                f"{field} introduced in metadata version "
                                f"{field_metadata_version}, not {metadata_version}",
                            )
                            exceptions.append(exc)
                            continue
                    getattr(ins, key)
                except InvalidMetadata as exc:
                    exceptions.append(exc)

            if exceptions:
                raise ExceptionGroup("invalid metadata", exceptions)

        return ins

    @classmethod
    def from_email(cls, data: bytes | str, *, validate: bool = True) -> Metadata:
        """Parse metadata from email headers.

        If *validate* is true, the metadata will be validated. All exceptions
        related to validation will be gathered and raised as an :class:`ExceptionGroup`.
        """
        raw, unparsed = parse_email(data)

        if validate:
            exceptions: list[Exception] = []
            for unparsed_key in unparsed:
                if unparsed_key in _EMAIL_TO_RAW_MAPPING:
                    message = f"{unparsed_key!r} has invalid data"
                else:
                    message = f"unrecognized field: {unparsed_key!r}"
                exceptions.append(InvalidMetadata(unparsed_key, message))

            if exceptions:
                raise ExceptionGroup("unparsed", exceptions)

        try:
            return cls.from_raw(raw, validate=validate)
        except ExceptionGroup as exc_group:
            raise ExceptionGroup(
                "invalid or unparsed metadata", exc_group.exceptions
            ) from None

    metadata_version: _Validator[_MetadataVersion] = _Validator()
    """:external:ref:`core-metadata-metadata-version`
    (required; validated to be a valid metadata version)"""
    # `name` is not normalized/typed to NormalizedName so as to provide access to
    # the original/raw name.
    name: _Validator[str] = _Validator()
    """:external:ref:`core-metadata-name`
    (required; validated using :func:`~packaging.utils.canonicalize_name` and its
    *validate* parameter)"""
    version: _Validator[version_module.Version] = _Validator()
    """:external:ref:`core-metadata-version` (required)"""
    dynamic: _Validator[list[str] | None] = _Validator(
        added="2.2",
    )
    """:external:ref:`core-metadata-dynamic`
    (validated against core metadata field names and lowercased)"""
    platforms: _Validator[list[str] | None] = _Validator()
    """:external:ref:`core-metadata-platform`"""
    supported_platforms: _Validator[list[str] | None] = _Validator(added="1.1")
    """:external:ref:`core-metadata-supported-platform`"""
    summary: _Validator[str | None] = _Validator()
    """:external:ref:`core-metadata-summary` (validated to contain no newlines)"""
    description: _Validator[str | None] = _Validator()  # TODO 2.1: can be in body
    """:external:ref:`core-metadata-description`"""
    description_content_type: _Validator[str | None] = _Validator(added="2.1")
    """:external:ref:`core-metadata-description-content-type` (validated)"""
    keywords: _Validator[list[str] | None] = _Validator()
    """:external:ref:`core-metadata-keywords`"""
    home_page: _Validator[str | None] = _Validator()
    """:external:ref:`core-metadata-home-page`"""
    download_url: _Validator[str | None] = _Validator(added="1.1")
    """:external:ref:`core-metadata-download-url`"""
    author: _Validator[str | None] = _Validator()
    """:external:ref:`core-metadata-author`"""
    author_email: _Validator[str | None] = _Validator()
    """:external:ref:`core-metadata-author-email`"""
    maintainer: _Validator[str | None] = _Validator(added="1.2")
    """:external:ref:`core-metadata-maintainer`"""
    maintainer_email: _Validator[str | None] = _Validator(added="1.2")
    """:external:ref:`core-metadata-maintainer-email`"""
    license: _Validator[str | None] = _Validator()
    """:external:ref:`core-metadata-license`"""
    license_expression: _Validator[NormalizedLicenseExpression | None] = _Validator(
        added="2.4"
    )
    """:external:ref:`core-metadata-license-expression`"""
    license_files: _Validator[list[str] | None] = _Validator(added="2.4")
    """:external:ref:`core-metadata-license-file`"""
    classifiers: _Validator[list[str] | None] = _Validator(added="1.1")
    """:external:ref:`core-metadata-classifier`"""
    requires_dist: _Validator[list[requirements.Requirement] | None] = _Validator(
        added="1.2"
    )
    """:external:ref:`core-metadata-requires-dist`"""
    requires_python: _Validator[specifiers.SpecifierSet | None] = _Validator(
        added="1.2"
    )
    """:external:ref:`core-metadata-requires-python`"""
    # Because `Requires-External` allows for non-PEP 440 version specifiers, we
    # don't do any processing on the values.
    requires_external: _Validator[list[str] | None] = _Validator(added="1.2")
    """:external:ref:`core-metadata-requires-external`"""
    project_urls: _Validator[dict[str, str] | None] = _Validator(added="1.2")
    """:external:ref:`core-metadata-project-url`"""
    # PEP 685 lets us raise an error if an extra doesn't pass `Name` validation
    # regardless of metadata version.
    provides_extra: _Validator[list[utils.NormalizedName] | None] = _Validator(
        added="2.1",
    )
    """:external:ref:`core-metadata-provides-extra`"""
    provides_dist: _Validator[list[str] | None] = _Validator(added="1.2")
    """:external:ref:`core-metadata-provides-dist`"""
    obsoletes_dist: _Validator[list[str] | None] = _Validator(added="1.2")
    """:external:ref:`core-metadata-obsoletes-dist`"""
    requires: _Validator[list[str] | None] = _Validator(added="1.1")
    """``Requires`` (deprecated)"""
    provides: _Validator[list[str] | None] = _Validator(added="1.1")
    """``Provides`` (deprecated)"""
    obsoletes: _Validator[list[str] | None] = _Validator(added="1.1")
    """``Obsoletes`` (deprecated)"""

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\polys\groebnertools.py ===
"""Groebner bases algorithms. """


from sympy.core.symbol import Dummy
from sympy.polys.monomials import monomial_mul, monomial_lcm, monomial_divides, term_div
from sympy.polys.orderings import lex
from sympy.polys.polyerrors import DomainError
from sympy.polys.polyconfig import query

def groebner(seq, ring, method=None):
    """
    Computes Groebner basis for a set of polynomials in `K[X]`.

    Wrapper around the (default) improved Buchberger and the other algorithms
    for computing Groebner bases. The choice of algorithm can be changed via
    ``method`` argument or :func:`sympy.polys.polyconfig.setup`, where
    ``method`` can be either ``buchberger`` or ``f5b``.

    """
    if method is None:
        method = query('groebner')

    _groebner_methods = {
        'buchberger': _buchberger,
        'f5b': _f5b,
    }

    try:
        _groebner = _groebner_methods[method]
    except KeyError:
        raise ValueError("'%s' is not a valid Groebner bases algorithm (valid are 'buchberger' and 'f5b')" % method)

    domain, orig = ring.domain, None

    if not domain.is_Field or not domain.has_assoc_Field:
        try:
            orig, ring = ring, ring.clone(domain=domain.get_field())
        except DomainError:
            raise DomainError("Cannot compute a Groebner basis over %s" % domain)
        else:
            seq = [ s.set_ring(ring) for s in seq ]

    G = _groebner(seq, ring)

    if orig is not None:
        G = [ g.clear_denoms()[1].set_ring(orig) for g in G ]

    return G

def _buchberger(f, ring):
    """
    Computes Groebner basis for a set of polynomials in `K[X]`.

    Given a set of multivariate polynomials `F`, finds another
    set `G`, such that Ideal `F = Ideal G` and `G` is a reduced
    Groebner basis.

    The resulting basis is unique and has monic generators if the
    ground domains is a field. Otherwise the result is non-unique
    but Groebner bases over e.g. integers can be computed (if the
    input polynomials are monic).

    Groebner bases can be used to choose specific generators for a
    polynomial ideal. Because these bases are unique you can check
    for ideal equality by comparing the Groebner bases.  To see if
    one polynomial lies in an ideal, divide by the elements in the
    base and see if the remainder vanishes.

    They can also be used to solve systems of polynomial equations
    as,  by choosing lexicographic ordering,  you can eliminate one
    variable at a time, provided that the ideal is zero-dimensional
    (finite number of solutions).

    Notes
    =====

    Algorithm used: an improved version of Buchberger's algorithm
    as presented in T. Becker, V. Weispfenning, Groebner Bases: A
    Computational Approach to Commutative Algebra, Springer, 1993,
    page 232.

    References
    ==========

    .. [1] [Bose03]_
    .. [2] [Giovini91]_
    .. [3] [Ajwa95]_
    .. [4] [Cox97]_

    """
    order = ring.order

    monomial_mul = ring.monomial_mul
    monomial_div = ring.monomial_div
    monomial_lcm = ring.monomial_lcm

    def select(P):
        # normal selection strategy
        # select the pair with minimum LCM(LM(f), LM(g))
        pr = min(P, key=lambda pair: order(monomial_lcm(f[pair[0]].LM, f[pair[1]].LM)))
        return pr

    def normal(g, J):
        h = g.rem([ f[j] for j in J ])

        if not h:
            return None
        else:
            h = h.monic()

            if h not in I:
                I[h] = len(f)
                f.append(h)

            return h.LM, I[h]

    def update(G, B, ih):
        # update G using the set of critical pairs B and h
        # [BW] page 230
        h = f[ih]
        mh = h.LM

        # filter new pairs (h, g), g in G
        C = G.copy()
        D = set()

        while C:
            # select a pair (h, g) by popping an element from C
            ig = C.pop()
            g = f[ig]
            mg = g.LM
            LCMhg = monomial_lcm(mh, mg)

            def lcm_divides(ip):
                # LCM(LM(h), LM(p)) divides LCM(LM(h), LM(g))
                m = monomial_lcm(mh, f[ip].LM)
                return monomial_div(LCMhg, m)

            # HT(h) and HT(g) disjoint: mh*mg == LCMhg
            if monomial_mul(mh, mg) == LCMhg or (
                not any(lcm_divides(ipx) for ipx in C) and
                    not any(lcm_divides(pr[1]) for pr in D)):
                D.add((ih, ig))

        E = set()

        while D:
            # select h, g from D (h the same as above)
            ih, ig = D.pop()
            mg = f[ig].LM
            LCMhg = monomial_lcm(mh, mg)

            if not monomial_mul(mh, mg) == LCMhg:
                E.add((ih, ig))

        # filter old pairs
        B_new = set()

        while B:
            # select g1, g2 from B (-> CP)
            ig1, ig2 = B.pop()
            mg1 = f[ig1].LM
            mg2 = f[ig2].LM
            LCM12 = monomial_lcm(mg1, mg2)

            # if HT(h) does not divide lcm(HT(g1), HT(g2))
            if not monomial_div(LCM12, mh) or \
                monomial_lcm(mg1, mh) == LCM12 or \
                    monomial_lcm(mg2, mh) == LCM12:
                B_new.add((ig1, ig2))

        B_new |= E

        # filter polynomials
        G_new = set()

        while G:
            ig = G.pop()
            mg = f[ig].LM

            if not monomial_div(mg, mh):
                G_new.add(ig)

        G_new.add(ih)

        return G_new, B_new
        # end of update ################################

    if not f:
        return []

    # replace f with a reduced list of initial polynomials; see [BW] page 203
    f1 = f[:]

    while True:
        f = f1[:]
        f1 = []

        for i in range(len(f)):
            p = f[i]
            r = p.rem(f[:i])

            if r:
                f1.append(r.monic())

        if f == f1:
            break

    I = {}            # ip = I[p]; p = f[ip]
    F = set()         # set of indices of polynomials
    G = set()         # set of indices of intermediate would-be Groebner basis
    CP = set()        # set of pairs of indices of critical pairs

    for i, h in enumerate(f):
        I[h] = i
        F.add(i)

    #####################################
    # algorithm GROEBNERNEWS2 in [BW] page 232

    while F:
        # select p with minimum monomial according to the monomial ordering
        h = min([f[x] for x in F], key=lambda f: order(f.LM))
        ih = I[h]
        F.remove(ih)
        G, CP = update(G, CP, ih)

    # count the number of critical pairs which reduce to zero
    reductions_to_zero = 0

    while CP:
        ig1, ig2 = select(CP)
        CP.remove((ig1, ig2))

        h = spoly(f[ig1], f[ig2], ring)
        # ordering divisors is on average more efficient [Cox] page 111
        G1 = sorted(G, key=lambda g: order(f[g].LM))
        ht = normal(h, G1)

        if ht:
            G, CP = update(G, CP, ht[1])
        else:
            reductions_to_zero += 1

    ######################################
    # now G is a Groebner basis; reduce it
    Gr = set()

    for ig in G:
        ht = normal(f[ig], G - {ig})

        if ht:
            Gr.add(ht[1])

    Gr = [f[ig] for ig in Gr]

    # order according to the monomial ordering
    Gr = sorted(Gr, key=lambda f: order(f.LM), reverse=True)

    return Gr

def spoly(p1, p2, ring):
    """
    Compute LCM(LM(p1), LM(p2))/LM(p1)*p1 - LCM(LM(p1), LM(p2))/LM(p2)*p2
    This is the S-poly provided p1 and p2 are monic
    """
    LM1 = p1.LM
    LM2 = p2.LM
    LCM12 = ring.monomial_lcm(LM1, LM2)
    m1 = ring.monomial_div(LCM12, LM1)
    m2 = ring.monomial_div(LCM12, LM2)
    s1 = p1.mul_monom(m1)
    s2 = p2.mul_monom(m2)
    s = s1 - s2
    return s

# F5B

# convenience functions


def Sign(f):
    return f[0]


def Polyn(f):
    return f[1]


def Num(f):
    return f[2]


def sig(monomial, index):
    return (monomial, index)


def lbp(signature, polynomial, number):
    return (signature, polynomial, number)

# signature functions


def sig_cmp(u, v, order):
    """
    Compare two signatures by extending the term order to K[X]^n.

    u < v iff
        - the index of v is greater than the index of u
    or
        - the index of v is equal to the index of u and u[0] < v[0] w.r.t. order

    u > v otherwise
    """
    if u[1] > v[1]:
        return -1
    if u[1] == v[1]:
        #if u[0] == v[0]:
        #    return 0
        if order(u[0]) < order(v[0]):
            return -1
    return 1


def sig_key(s, order):
    """
    Key for comparing two signatures.

    s = (m, k), t = (n, l)

    s < t iff [k > l] or [k == l and m < n]
    s > t otherwise
    """
    return (-s[1], order(s[0]))


def sig_mult(s, m):
    """
    Multiply a signature by a monomial.

    The product of a signature (m, i) and a monomial n is defined as
    (m * t, i).
    """
    return sig(monomial_mul(s[0], m), s[1])

# labeled polynomial functions


def lbp_sub(f, g):
    """
    Subtract labeled polynomial g from f.

    The signature and number of the difference of f and g are signature
    and number of the maximum of f and g, w.r.t. lbp_cmp.
    """
    if sig_cmp(Sign(f), Sign(g), Polyn(f).ring.order) < 0:
        max_poly = g
    else:
        max_poly = f

    ret = Polyn(f) - Polyn(g)

    return lbp(Sign(max_poly), ret, Num(max_poly))


def lbp_mul_term(f, cx):
    """
    Multiply a labeled polynomial with a term.

    The product of a labeled polynomial (s, p, k) by a monomial is
    defined as (m * s, m * p, k).
    """
    return lbp(sig_mult(Sign(f), cx[0]), Polyn(f).mul_term(cx), Num(f))


def lbp_cmp(f, g):
    """
    Compare two labeled polynomials.

    f < g iff
        - Sign(f) < Sign(g)
    or
        - Sign(f) == Sign(g) and Num(f) > Num(g)

    f > g otherwise
    """
    if sig_cmp(Sign(f), Sign(g), Polyn(f).ring.order) == -1:
        return -1
    if Sign(f) == Sign(g):
        if Num(f) > Num(g):
            return -1
        #if Num(f) == Num(g):
        #    return 0
    return 1


def lbp_key(f):
    """
    Key for comparing two labeled polynomials.
    """
    return (sig_key(Sign(f), Polyn(f).ring.order), -Num(f))

# algorithm and helper functions


def critical_pair(f, g, ring):
    """
    Compute the critical pair corresponding to two labeled polynomials.

    A critical pair is a tuple (um, f, vm, g), where um and vm are
    terms such that um * f - vm * g is the S-polynomial of f and g (so,
    wlog assume um * f > vm * g).
    For performance sake, a critical pair is represented as a tuple
    (Sign(um * f), um, f, Sign(vm * g), vm, g), since um * f creates
    a new, relatively expensive object in memory, whereas Sign(um *
    f) and um are lightweight and f (in the tuple) is a reference to
    an already existing object in memory.
    """
    domain = ring.domain

    ltf = Polyn(f).LT
    ltg = Polyn(g).LT
    lt = (monomial_lcm(ltf[0], ltg[0]), domain.one)

    um = term_div(lt, ltf, domain)
    vm = term_div(lt, ltg, domain)

    # The full information is not needed (now), so only the product
    # with the leading term is considered:
    fr = lbp_mul_term(lbp(Sign(f), Polyn(f).leading_term(), Num(f)), um)
    gr = lbp_mul_term(lbp(Sign(g), Polyn(g).leading_term(), Num(g)), vm)

    # return in proper order, such that the S-polynomial is just
    # u_first * f_first - u_second * f_second:
    if lbp_cmp(fr, gr) == -1:
        return (Sign(gr), vm, g, Sign(fr), um, f)
    else:
        return (Sign(fr), um, f, Sign(gr), vm, g)


def cp_cmp(c, d):
    """
    Compare two critical pairs c and d.

    c < d iff
        - lbp(c[0], _, Num(c[2]) < lbp(d[0], _, Num(d[2])) (this
        corresponds to um_c * f_c and um_d * f_d)
    or
        - lbp(c[0], _, Num(c[2]) >< lbp(d[0], _, Num(d[2])) and
        lbp(c[3], _, Num(c[5])) < lbp(d[3], _, Num(d[5])) (this
        corresponds to vm_c * g_c and vm_d * g_d)

    c > d otherwise
    """
    zero = Polyn(c[2]).ring.zero

    c0 = lbp(c[0], zero, Num(c[2]))
    d0 = lbp(d[0], zero, Num(d[2]))

    r = lbp_cmp(c0, d0)

    if r == -1:
        return -1
    if r == 0:
        c1 = lbp(c[3], zero, Num(c[5]))
        d1 = lbp(d[3], zero, Num(d[5]))

        r = lbp_cmp(c1, d1)

        if r == -1:
            return -1
        #if r == 0:
        #    return 0
    return 1


def cp_key(c, ring):
    """
    Key for comparing critical pairs.
    """
    return (lbp_key(lbp(c[0], ring.zero, Num(c[2]))), lbp_key(lbp(c[3], ring.zero, Num(c[5]))))


def s_poly(cp):
    """
    Compute the S-polynomial of a critical pair.

    The S-polynomial of a critical pair cp is cp[1] * cp[2] - cp[4] * cp[5].
    """
    return lbp_sub(lbp_mul_term(cp[2], cp[1]), lbp_mul_term(cp[5], cp[4]))


def is_rewritable_or_comparable(sign, num, B):
    """
    Check if a labeled polynomial is redundant by checking if its
    signature and number imply rewritability or comparability.

    (sign, num) is comparable if there exists a labeled polynomial
    h in B, such that sign[1] (the index) is less than Sign(h)[1]
    and sign[0] is divisible by the leading monomial of h.

    (sign, num) is rewritable if there exists a labeled polynomial
    h in B, such thatsign[1] is equal to Sign(h)[1], num < Num(h)
    and sign[0] is divisible by Sign(h)[0].
    """
    for h in B:
        # comparable
        if sign[1] < Sign(h)[1]:
            if monomial_divides(Polyn(h).LM, sign[0]):
                return True

        # rewritable
        if sign[1] == Sign(h)[1]:
            if num < Num(h):
                if monomial_divides(Sign(h)[0], sign[0]):
                    return True
    return False


def f5_reduce(f, B):
    """
    F5-reduce a labeled polynomial f by B.

    Continuously searches for non-zero labeled polynomial h in B, such
    that the leading term lt_h of h divides the leading term lt_f of
    f and Sign(lt_h * h) < Sign(f). If such a labeled polynomial h is
    found, f gets replaced by f - lt_f / lt_h * h. If no such h can be
    found or f is 0, f is no further F5-reducible and f gets returned.

    A polynomial that is reducible in the usual sense need not be
    F5-reducible, e.g.:

    >>> from sympy.polys.groebnertools import lbp, sig, f5_reduce, Polyn
    >>> from sympy.polys import ring, QQ, lex

    >>> R, x,y,z = ring("x,y,z", QQ, lex)

    >>> f = lbp(sig((1, 1, 1), 4), x, 3)
    >>> g = lbp(sig((0, 0, 0), 2), x, 2)

    >>> Polyn(f).rem([Polyn(g)])
    0
    >>> f5_reduce(f, [g])
    (((1, 1, 1), 4), x, 3)

    """
    order = Polyn(f).ring.order
    domain = Polyn(f).ring.domain

    if not Polyn(f):
        return f

    while True:
        g = f

        for h in B:
            if Polyn(h):
                if monomial_divides(Polyn(h).LM, Polyn(f).LM):
                    t = term_div(Polyn(f).LT, Polyn(h).LT, domain)
                    if sig_cmp(sig_mult(Sign(h), t[0]), Sign(f), order) < 0:
                        # The following check need not be done and is in general slower than without.
                        #if not is_rewritable_or_comparable(Sign(gp), Num(gp), B):
                        hp = lbp_mul_term(h, t)
                        f = lbp_sub(f, hp)
                        break

        if g == f or not Polyn(f):
            return f


def _f5b(F, ring):
    """
    Computes a reduced Groebner basis for the ideal generated by F.

    f5b is an implementation of the F5B algorithm by Yao Sun and
    Dingkang Wang. Similarly to Buchberger's algorithm, the algorithm
    proceeds by computing critical pairs, computing the S-polynomial,
    reducing it and adjoining the reduced S-polynomial if it is not 0.

    Unlike Buchberger's algorithm, each polynomial contains additional
    information, namely a signature and a number. The signature
    specifies the path of computation (i.e. from which polynomial in
    the original basis was it derived and how), the number says when
    the polynomial was added to the basis.  With this information it
    is (often) possible to decide if an S-polynomial will reduce to
    0 and can be discarded.

    Optimizations include: Reducing the generators before computing
    a Groebner basis, removing redundant critical pairs when a new
    polynomial enters the basis and sorting the critical pairs and
    the current basis.

    Once a Groebner basis has been found, it gets reduced.

    References
    ==========

    .. [1] Yao Sun, Dingkang Wang: "A New Proof for the Correctness of F5
           (F5-Like) Algorithm", https://arxiv.org/abs/1004.0084 (specifically
           v4)

    .. [2] Thomas Becker, Volker Weispfenning, Groebner bases: A computational
           approach to commutative algebra, 1993, p. 203, 216
    """
    order = ring.order

    # reduce polynomials (like in Mario Pernici's implementation) (Becker, Weispfenning, p. 203)
    B = F
    while True:
        F = B
        B = []

        for i in range(len(F)):
            p = F[i]
            r = p.rem(F[:i])

            if r:
                B.append(r)

        if F == B:
            break

    # basis
    B = [lbp(sig(ring.zero_monom, i + 1), F[i], i + 1) for i in range(len(F))]
    B.sort(key=lambda f: order(Polyn(f).LM), reverse=True)

    # critical pairs
    CP = [critical_pair(B[i], B[j], ring) for i in range(len(B)) for j in range(i + 1, len(B))]
    CP.sort(key=lambda cp: cp_key(cp, ring), reverse=True)

    k = len(B)

    reductions_to_zero = 0

    while len(CP):
        cp = CP.pop()

        # discard redundant critical pairs:
        if is_rewritable_or_comparable(cp[0], Num(cp[2]), B):
            continue
        if is_rewritable_or_comparable(cp[3], Num(cp[5]), B):
            continue

        s = s_poly(cp)

        p = f5_reduce(s, B)

        p = lbp(Sign(p), Polyn(p).monic(), k + 1)

        if Polyn(p):
            # remove old critical pairs, that become redundant when adding p:
            indices = []
            for i, cp in enumerate(CP):
                if is_rewritable_or_comparable(cp[0], Num(cp[2]), [p]):
                    indices.append(i)
                elif is_rewritable_or_comparable(cp[3], Num(cp[5]), [p]):
                    indices.append(i)

            for i in reversed(indices):
                del CP[i]

            # only add new critical pairs that are not made redundant by p:
            for g in B:
                if Polyn(g):
                    cp = critical_pair(p, g, ring)
                    if is_rewritable_or_comparable(cp[0], Num(cp[2]), [p]):
                        continue
                    elif is_rewritable_or_comparable(cp[3], Num(cp[5]), [p]):
                        continue

                    CP.append(cp)

            # sort (other sorting methods/selection strategies were not as successful)
            CP.sort(key=lambda cp: cp_key(cp, ring), reverse=True)

            # insert p into B:
            m = Polyn(p).LM
            if order(m) <= order(Polyn(B[-1]).LM):
                B.append(p)
            else:
                for i, q in enumerate(B):
                    if order(m) > order(Polyn(q).LM):
                        B.insert(i, p)
                        break

            k += 1

            #print(len(B), len(CP), "%d critical pairs removed" % len(indices))
        else:
            reductions_to_zero += 1

    # reduce Groebner basis:
    H = [Polyn(g).monic() for g in B]
    H = red_groebner(H, ring)

    return sorted(H, key=lambda f: order(f.LM), reverse=True)


def red_groebner(G, ring):
    """
    Compute reduced Groebner basis, from BeckerWeispfenning93, p. 216

    Selects a subset of generators, that already generate the ideal
    and computes a reduced Groebner basis for them.
    """
    def reduction(P):
        """
        The actual reduction algorithm.
        """
        Q = []
        for i, p in enumerate(P):
            h = p.rem(P[:i] + P[i + 1:])
            if h:
                Q.append(h)

        return [p.monic() for p in Q]

    F = G
    H = []

    while F:
        f0 = F.pop()

        if not any(monomial_divides(f.LM, f0.LM) for f in F + H):
            H.append(f0)

    # Becker, Weispfenning, p. 217: H is Groebner basis of the ideal generated by G.
    return reduction(H)


def is_groebner(G, ring):
    """
    Check if G is a Groebner basis.
    """
    for i in range(len(G)):
        for j in range(i + 1, len(G)):
            s = spoly(G[i], G[j], ring)
            s = s.rem(G)
            if s:
                return False

    return True


def is_minimal(G, ring):
    """
    Checks if G is a minimal Groebner basis.
    """
    order = ring.order
    domain = ring.domain

    G.sort(key=lambda g: order(g.LM))

    for i, g in enumerate(G):
        if g.LC != domain.one:
            return False

        for h in G[:i] + G[i + 1:]:
            if monomial_divides(h.LM, g.LM):
                return False

    return True


def is_reduced(G, ring):
    """
    Checks if G is a reduced Groebner basis.
    """
    order = ring.order
    domain = ring.domain

    G.sort(key=lambda g: order(g.LM))

    for i, g in enumerate(G):
        if g.LC != domain.one:
            return False

        for term in g.terms():
            for h in G[:i] + G[i + 1:]:
                if monomial_divides(h.LM, term[0]):
                    return False

    return True

def groebner_lcm(f, g):
    """
    Computes LCM of two polynomials using Groebner bases.

    The LCM is computed as the unique generator of the intersection
    of the two ideals generated by `f` and `g`. The approach is to
    compute a Groebner basis with respect to lexicographic ordering
    of `t*f` and `(1 - t)*g`, where `t` is an unrelated variable and
    then filtering out the solution that does not contain `t`.

    References
    ==========

    .. [1] [Cox97]_

    """
    if f.ring != g.ring:
        raise ValueError("Values should be equal")

    ring = f.ring
    domain = ring.domain

    if not f or not g:
        return ring.zero

    if len(f) <= 1 and len(g) <= 1:
        monom = monomial_lcm(f.LM, g.LM)
        coeff = domain.lcm(f.LC, g.LC)
        return ring.term_new(monom, coeff)

    fc, f = f.primitive()
    gc, g = g.primitive()

    lcm = domain.lcm(fc, gc)

    f_terms = [ ((1,) + monom, coeff) for monom, coeff in f.terms() ]
    g_terms = [ ((0,) + monom, coeff) for monom, coeff in g.terms() ] \
            + [ ((1,) + monom,-coeff) for monom, coeff in g.terms() ]

    t = Dummy("t")
    t_ring = ring.clone(symbols=(t,) + ring.symbols, order=lex)

    F = t_ring.from_terms(f_terms)
    G = t_ring.from_terms(g_terms)

    basis = groebner([F, G], t_ring)

    def is_independent(h, j):
        return not any(monom[j] for monom in h.monoms())

    H = [ h for h in basis if is_independent(h, 0) ]

    h_terms = [ (monom[1:], coeff*lcm) for monom, coeff in H[0].terms() ]
    h = ring.from_terms(h_terms)

    return h

def groebner_gcd(f, g):
    """Computes GCD of two polynomials using Groebner bases. """
    if f.ring != g.ring:
        raise ValueError("Values should be equal")
    domain = f.ring.domain

    if not domain.is_Field:
        fc, f = f.primitive()
        gc, g = g.primitive()
        gcd = domain.gcd(fc, gc)

    H = (f*g).quo([groebner_lcm(f, g)])

    if len(H) != 1:
        raise ValueError("Length should be 1")
    h = H[0]

    if not domain.is_Field:
        return gcd*h
    else:
        return h.monic()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\mechanics\kane.py ===
from sympy import zeros, Matrix, diff, eye, linear_eq_to_matrix
from sympy.core.sorting import default_sort_key
from sympy.physics.vector import (ReferenceFrame, dynamicsymbols,
                                  partial_velocity)
from sympy.physics.mechanics.method import _Methods
from sympy.physics.mechanics.particle import Particle
from sympy.physics.mechanics.rigidbody import RigidBody
from sympy.physics.mechanics.functions import (msubs, find_dynamicsymbols,
                                               _f_list_parser,
                                               _validate_coordinates,
                                               _parse_linear_solver)
from sympy.physics.mechanics.linearize import Linearizer
from sympy.utilities.iterables import iterable


__all__ = ['KanesMethod']


class KanesMethod(_Methods):
    r"""Kane's method object.

    Explanation
    ===========

    This object is used to do the "book-keeping" as you go through and form
    equations of motion in the way Kane presents in:
    Kane, T., Levinson, D. Dynamics Theory and Applications. 1985 McGraw-Hill

    The attributes are for equations in the form [M] udot = forcing.

    Attributes
    ==========

    q, u : Matrix
        Matrices of the generalized coordinates and speeds
    bodies : iterable
        Iterable of Particle and RigidBody objects in the system.
    loads : iterable
        Iterable of (Point, vector) or (ReferenceFrame, vector) tuples
        describing the forces on the system.
    auxiliary_eqs : Matrix
        If applicable, the set of auxiliary Kane's
        equations used to solve for non-contributing
        forces.
    mass_matrix : Matrix
        The system's dynamics mass matrix: [k_d; k_dnh]
    forcing : Matrix
        The system's dynamics forcing vector: -[f_d; f_dnh]
    mass_matrix_kin : Matrix
        The "mass matrix" for kinematic differential equations: k_kqdot
    forcing_kin : Matrix
        The forcing vector for kinematic differential equations: -(k_ku*u + f_k)
    mass_matrix_full : Matrix
        The "mass matrix" for the u's and q's with dynamics and kinematics
    forcing_full : Matrix
        The "forcing vector" for the u's and q's with dynamics and kinematics

    Parameters
    ==========

    frame : ReferenceFrame
        The inertial reference frame for the system.
    q_ind : iterable of dynamicsymbols
        Independent generalized coordinates.
    u_ind : iterable of dynamicsymbols
        Independent generalized speeds.
    kd_eqs : iterable of Expr, optional
        Kinematic differential equations, which linearly relate the generalized
        speeds to the time-derivatives of the generalized coordinates.
    q_dependent : iterable of dynamicsymbols, optional
        Dependent generalized coordinates.
    configuration_constraints : iterable of Expr, optional
        Constraints on the system's configuration, i.e. holonomic constraints.
    u_dependent : iterable of dynamicsymbols, optional
        Dependent generalized speeds.
    velocity_constraints : iterable of Expr, optional
        Constraints on the system's velocity, i.e. the combination of the
        nonholonomic constraints and the time-derivative of the holonomic
        constraints.
    acceleration_constraints : iterable of Expr, optional
        Constraints on the system's acceleration, by default these are the
        time-derivative of the velocity constraints.
    u_auxiliary : iterable of dynamicsymbols, optional
        Auxiliary generalized speeds.
    bodies : iterable of Particle and/or RigidBody, optional
        The particles and rigid bodies in the system.
    forcelist : iterable of tuple[Point | ReferenceFrame, Vector], optional
        Forces and torques applied on the system.
    explicit_kinematics : bool
        Boolean whether the mass matrices and forcing vectors should use the
        explicit form (default) or implicit form for kinematics.
        See the notes for more details.
    kd_eqs_solver : str, callable
        Method used to solve the kinematic differential equations. If a string
        is supplied, it should be a valid method that can be used with the
        :meth:`sympy.matrices.matrixbase.MatrixBase.solve`. If a callable is
        supplied, it should have the format ``f(A, rhs)``, where it solves the
        equations and returns the solution. The default utilizes LU solve. See
        the notes for more information.
    constraint_solver : str, callable
        Method used to solve the velocity constraints. If a string is
        supplied, it should be a valid method that can be used with the
        :meth:`sympy.matrices.matrixbase.MatrixBase.solve`. If a callable is
        supplied, it should have the format ``f(A, rhs)``, where it solves the
        equations and returns the solution. The default utilizes LU solve. See
        the notes for more information.

    Notes
    =====

    The mass matrices and forcing vectors related to kinematic equations
    are given in the explicit form by default. In other words, the kinematic
    mass matrix is $\mathbf{k_{k\dot{q}}} = \mathbf{I}$.
    In order to get the implicit form of those matrices/vectors, you can set the
    ``explicit_kinematics`` attribute to ``False``. So $\mathbf{k_{k\dot{q}}}$
    is not necessarily an identity matrix. This can provide more compact
    equations for non-simple kinematics.

    Two linear solvers can be supplied to ``KanesMethod``: one for solving the
    kinematic differential equations and one to solve the velocity constraints.
    Both of these sets of equations can be expressed as a linear system ``Ax = rhs``,
    which have to be solved in order to obtain the equations of motion.

    The default solver ``'LU'``, which stands for LU solve, results relatively low
    number of operations. The weakness of this method is that it can result in zero
    division errors.

    If zero divisions are encountered, a possible solver which may solve the problem
    is ``"CRAMER"``. This method uses Cramer's rule to solve the system. This method
    is slower and results in more operations than the default solver. However it only
    uses a single division by default per entry of the solution.

    While a valid list of solvers can be found at
    :meth:`sympy.matrices.matrixbase.MatrixBase.solve`, it is also possible to supply a
    `callable`. This way it is possible to use a different solver routine. If the
    kinematic differential equations are not too complex it can be worth it to simplify
    the solution by using ``lambda A, b: simplify(Matrix.LUsolve(A, b))``. Another
    option solver one may use is :func:`sympy.solvers.solveset.linsolve`. This can be
    done using `lambda A, b: tuple(linsolve((A, b)))[0]`, where we select the first
    solution as our system should have only one unique solution.

    Examples
    ========

    This is a simple example for a one degree of freedom translational
    spring-mass-damper.

    In this example, we first need to do the kinematics.
    This involves creating generalized speeds and coordinates and their
    derivatives.
    Then we create a point and set its velocity in a frame.

        >>> from sympy import symbols
        >>> from sympy.physics.mechanics import dynamicsymbols, ReferenceFrame
        >>> from sympy.physics.mechanics import Point, Particle, KanesMethod
        >>> q, u = dynamicsymbols('q u')
        >>> qd, ud = dynamicsymbols('q u', 1)
        >>> m, c, k = symbols('m c k')
        >>> N = ReferenceFrame('N')
        >>> P = Point('P')
        >>> P.set_vel(N, u * N.x)

    Next we need to arrange/store information in the way that KanesMethod
    requires. The kinematic differential equations should be an iterable of
    expressions. A list of forces/torques must be constructed, where each entry
    in the list is a (Point, Vector) or (ReferenceFrame, Vector) tuple, where
    the Vectors represent the Force or Torque.
    Next a particle needs to be created, and it needs to have a point and mass
    assigned to it.
    Finally, a list of all bodies and particles needs to be created.

        >>> kd = [qd - u]
        >>> FL = [(P, (-k * q - c * u) * N.x)]
        >>> pa = Particle('pa', P, m)
        >>> BL = [pa]

    Finally we can generate the equations of motion.
    First we create the KanesMethod object and supply an inertial frame,
    coordinates, generalized speeds, and the kinematic differential equations.
    Additional quantities such as configuration and motion constraints,
    dependent coordinates and speeds, and auxiliary speeds are also supplied
    here (see the online documentation).
    Next we form FR* and FR to complete: Fr + Fr* = 0.
    We have the equations of motion at this point.
    It makes sense to rearrange them though, so we calculate the mass matrix and
    the forcing terms, for E.o.M. in the form: [MM] udot = forcing, where MM is
    the mass matrix, udot is a vector of the time derivatives of the
    generalized speeds, and forcing is a vector representing "forcing" terms.

        >>> KM = KanesMethod(N, q_ind=[q], u_ind=[u], kd_eqs=kd)
        >>> (fr, frstar) = KM.kanes_equations(BL, FL)
        >>> MM = KM.mass_matrix
        >>> forcing = KM.forcing
        >>> rhs = MM.inv() * forcing
        >>> rhs
        Matrix([[(-c*u(t) - k*q(t))/m]])
        >>> KM.linearize(A_and_B=True)[0]
        Matrix([
        [   0,    1],
        [-k/m, -c/m]])

    Please look at the documentation pages for more information on how to
    perform linearization and how to deal with dependent coordinates & speeds,
    and how do deal with bringing non-contributing forces into evidence.

    """

    def __init__(self, frame, q_ind, u_ind, kd_eqs=None, q_dependent=None,
                 configuration_constraints=None, u_dependent=None,
                 velocity_constraints=None, acceleration_constraints=None,
                 u_auxiliary=None, bodies=None, forcelist=None,
                 explicit_kinematics=True, kd_eqs_solver='LU',
                 constraint_solver='LU'):

        """Please read the online documentation. """
        if not q_ind:
            q_ind = [dynamicsymbols('dummy_q')]
            kd_eqs = [dynamicsymbols('dummy_kd')]

        if not isinstance(frame, ReferenceFrame):
            raise TypeError('An inertial ReferenceFrame must be supplied')
        self._inertial = frame

        self._fr = None
        self._frstar = None

        self._forcelist = forcelist
        self._bodylist = bodies

        self.explicit_kinematics = explicit_kinematics
        self._constraint_solver = constraint_solver
        self._initialize_vectors(q_ind, q_dependent, u_ind, u_dependent,
                u_auxiliary)
        _validate_coordinates(self.q, self.u)
        self._initialize_kindiffeq_matrices(kd_eqs, kd_eqs_solver)
        self._initialize_constraint_matrices(
            configuration_constraints, velocity_constraints,
            acceleration_constraints, constraint_solver)

    def _initialize_vectors(self, q_ind, q_dep, u_ind, u_dep, u_aux):
        """Initialize the coordinate and speed vectors."""

        none_handler = lambda x: Matrix(x) if x else Matrix()

        # Initialize generalized coordinates
        q_dep = none_handler(q_dep)
        if not iterable(q_ind):
            raise TypeError('Generalized coordinates must be an iterable.')
        if not iterable(q_dep):
            raise TypeError('Dependent coordinates must be an iterable.')
        q_ind = Matrix(q_ind)
        self._qdep = q_dep
        self._q = Matrix([q_ind, q_dep])
        self._qdot = self.q.diff(dynamicsymbols._t)

        # Initialize generalized speeds
        u_dep = none_handler(u_dep)
        if not iterable(u_ind):
            raise TypeError('Generalized speeds must be an iterable.')
        if not iterable(u_dep):
            raise TypeError('Dependent speeds must be an iterable.')
        u_ind = Matrix(u_ind)
        self._udep = u_dep
        self._u = Matrix([u_ind, u_dep])
        self._udot = self.u.diff(dynamicsymbols._t)
        self._uaux = none_handler(u_aux)

    def _initialize_constraint_matrices(self, config, vel, acc, linear_solver='LU'):
        """Initializes constraint matrices."""
        linear_solver = _parse_linear_solver(linear_solver)
        # Define vector dimensions
        o = len(self.u)
        m = len(self._udep)
        p = o - m
        none_handler = lambda x: Matrix(x) if x else Matrix()

        # Initialize configuration constraints
        config = none_handler(config)
        if len(self._qdep) != len(config):
            raise ValueError('There must be an equal number of dependent '
                             'coordinates and configuration constraints.')
        self._f_h = none_handler(config)

        # Initialize velocity and acceleration constraints
        vel = none_handler(vel)
        acc = none_handler(acc)
        if len(vel) != m:
            raise ValueError('There must be an equal number of dependent '
                             'speeds and velocity constraints.')
        if acc and (len(acc) != m):
            raise ValueError('There must be an equal number of dependent '
                             'speeds and acceleration constraints.')
        if vel:

            # When calling kanes_equations, another class instance will be
            # created if auxiliary u's are present. In this case, the
            # computation of kinetic differential equation matrices will be
            # skipped as this was computed during the original KanesMethod
            # object, and the qd_u_map will not be available.
            if self._qdot_u_map is not None:
                vel = msubs(vel, self._qdot_u_map)
            self._k_nh, f_nh_neg = linear_eq_to_matrix(vel, self.u[:])
            self._f_nh = -f_nh_neg

            # If no acceleration constraints given, calculate them.
            if not acc:
                _f_dnh = (self._k_nh.diff(dynamicsymbols._t) * self.u +
                    self._f_nh.diff(dynamicsymbols._t))
                if self._qdot_u_map is not None:
                    _f_dnh = msubs(_f_dnh, self._qdot_u_map)
                self._f_dnh = _f_dnh
                self._k_dnh = self._k_nh
            else:
                if self._qdot_u_map is not None:
                    acc = msubs(acc, self._qdot_u_map)

                self._k_dnh, f_dnh_neg = linear_eq_to_matrix(acc, self._udot[:])
                self._f_dnh = -f_dnh_neg
            # Form of non-holonomic constraints is B*u + C = 0.
            # We partition B into independent and dependent columns:
            # Ars is then -B_dep.inv() * B_ind, and it relates dependent speeds
            # to independent speeds as: udep = Ars*uind, neglecting the C term.
            B_ind = self._k_nh[:, :p]
            B_dep = self._k_nh[:, p:o]
            self._Ars = -linear_solver(B_dep, B_ind)
        else:
            self._f_nh = Matrix()
            self._k_nh = Matrix()
            self._f_dnh = Matrix()
            self._k_dnh = Matrix()
            self._Ars = Matrix()

    def _initialize_kindiffeq_matrices(self, kdeqs, linear_solver='LU'):
        """Initialize the kinematic differential equation matrices.

        Parameters
        ==========
        kdeqs : sequence of sympy expressions
            Kinematic differential equations in the form of f(u,q',q,t) where
            f() = 0. The equations have to be linear in the time-derivatives of
            the generalized coordinates and in the generalized speeds.

        """
        linear_solver = _parse_linear_solver(linear_solver)
        if kdeqs:
            if len(self.q) != len(kdeqs):
                raise ValueError('There must be an equal number of kinematic '
                                 'differential equations and coordinates.')

            u = self.u
            qdot = self._qdot

            kdeqs = Matrix(kdeqs)

            u_zero = dict.fromkeys(u, 0)
            uaux_zero = dict.fromkeys(self._uaux, 0)
            qdot_zero = dict.fromkeys(qdot, 0)

            # Extract the linear coefficient matrices as per the following
            # equation:
            #
            # k_ku(q,t)*u(t) + k_kqdot(q,t)*q'(t) + f_k(q,t) = 0
            #
            k_ku = kdeqs.jacobian(u)
            k_kqdot = kdeqs.jacobian(qdot)
            f_k = kdeqs.xreplace(u_zero).xreplace(qdot_zero)

            # The kinematic differential equations should be linear in both q'
            # and u so check for u and q' in the components.
            dy_syms = find_dynamicsymbols(k_ku.row_join(k_kqdot).row_join(f_k))
            nonlin_vars = [vari for vari in u[:] + qdot[:] if vari in dy_syms]
            if nonlin_vars:
                msg = ('The provided kinematic differential equations are '
                       'nonlinear in {}. They must be linear in the '
                       'generalized speeds and derivatives of the generalized '
                       'coordinates.')
                raise ValueError(msg.format(nonlin_vars))

            self._f_k_implicit = f_k.xreplace(uaux_zero)
            self._k_ku_implicit = k_ku.xreplace(uaux_zero)
            self._k_kqdot_implicit = k_kqdot

            # Solve for q'(t) such that the coefficient matrices are now in
            # this form:
            #
            # k_kqdot^-1*k_ku*u(t) + I*q'(t) + k_kqdot^-1*f_k = 0
            #
            # NOTE : Solving the kinematic differential equations here is not
            # necessary and prevents the equations from being provided in fully
            # implicit form.
            f_k_explicit = linear_solver(k_kqdot, f_k)
            k_ku_explicit = linear_solver(k_kqdot, k_ku)
            self._qdot_u_map = dict(zip(qdot, -(k_ku_explicit*u + f_k_explicit)))

            self._f_k = f_k_explicit.xreplace(uaux_zero)
            self._k_ku = k_ku_explicit.xreplace(uaux_zero)
            self._k_kqdot = eye(len(qdot))

        else:
            self._qdot_u_map = None
            self._f_k_implicit = self._f_k = Matrix()
            self._k_ku_implicit = self._k_ku = Matrix()
            self._k_kqdot_implicit = self._k_kqdot = Matrix()

    def _form_fr(self, fl):
        """Form the generalized active force."""
        if fl is not None and (len(fl) == 0 or not iterable(fl)):
            raise ValueError('Force pairs must be supplied in an '
                'non-empty iterable or None.')

        N = self._inertial
        # pull out relevant velocities for constructing partial velocities
        vel_list, f_list = _f_list_parser(fl, N)
        vel_list = [msubs(i, self._qdot_u_map) for i in vel_list]
        f_list = [msubs(i, self._qdot_u_map) for i in f_list]

        # Fill Fr with dot product of partial velocities and forces
        o = len(self.u)
        b = len(f_list)
        FR = zeros(o, 1)
        partials = partial_velocity(vel_list, self.u, N)
        for i in range(o):
            FR[i] = sum(partials[j][i].dot(f_list[j]) for j in range(b))

        # In case there are dependent speeds
        if self._udep:
            p = o - len(self._udep)
            FRtilde = FR[:p, 0]
            FRold = FR[p:o, 0]
            FRtilde += self._Ars.T * FRold
            FR = FRtilde

        self._forcelist = fl
        self._fr = FR
        return FR

    def _form_frstar(self, bl):
        """Form the generalized inertia force."""

        if not iterable(bl):
            raise TypeError('Bodies must be supplied in an iterable.')

        t = dynamicsymbols._t
        N = self._inertial
        # Dicts setting things to zero
        udot_zero = dict.fromkeys(self._udot, 0)
        uaux_zero = dict.fromkeys(self._uaux, 0)
        uauxdot = [diff(i, t) for i in self._uaux]
        uauxdot_zero = dict.fromkeys(uauxdot, 0)
        # Dictionary of q' and q'' to u and u'
        q_ddot_u_map = {k.diff(t): v.diff(t).xreplace(
            self._qdot_u_map) for (k, v) in self._qdot_u_map.items()}
        q_ddot_u_map.update(self._qdot_u_map)

        # Fill up the list of partials: format is a list with num elements
        # equal to number of entries in body list. Each of these elements is a
        # list - either of length 1 for the translational components of
        # particles or of length 2 for the translational and rotational
        # components of rigid bodies. The inner most list is the list of
        # partial velocities.
        def get_partial_velocity(body):
            if isinstance(body, RigidBody):
                vlist = [body.masscenter.vel(N), body.frame.ang_vel_in(N)]
            elif isinstance(body, Particle):
                vlist = [body.point.vel(N),]
            else:
                raise TypeError('The body list may only contain either '
                                'RigidBody or Particle as list elements.')
            v = [msubs(vel, self._qdot_u_map) for vel in vlist]
            return partial_velocity(v, self.u, N)
        partials = [get_partial_velocity(body) for body in bl]

        # Compute fr_star in two components:
        # fr_star = -(MM*u' + nonMM)
        o = len(self.u)
        MM = zeros(o, o)
        nonMM = zeros(o, 1)
        zero_uaux = lambda expr: msubs(expr, uaux_zero)
        zero_udot_uaux = lambda expr: msubs(msubs(expr, udot_zero), uaux_zero)
        for i, body in enumerate(bl):
            if isinstance(body, RigidBody):
                M = zero_uaux(body.mass)
                I = zero_uaux(body.central_inertia)
                vel = zero_uaux(body.masscenter.vel(N))
                omega = zero_uaux(body.frame.ang_vel_in(N))
                acc = zero_udot_uaux(body.masscenter.acc(N))
                inertial_force = (M.diff(t) * vel + M * acc)
                inertial_torque = zero_uaux((I.dt(body.frame).dot(omega)) +
                    msubs(I.dot(body.frame.ang_acc_in(N)), udot_zero) +
                    (omega.cross(I.dot(omega))))
                for j in range(o):
                    tmp_vel = zero_uaux(partials[i][0][j])
                    tmp_ang = zero_uaux(I.dot(partials[i][1][j]))
                    for k in range(o):
                        # translational
                        MM[j, k] += M*tmp_vel.dot(partials[i][0][k])
                        # rotational
                        MM[j, k] += tmp_ang.dot(partials[i][1][k])
                    nonMM[j] += inertial_force.dot(partials[i][0][j])
                    nonMM[j] += inertial_torque.dot(partials[i][1][j])
            else:
                M = zero_uaux(body.mass)
                vel = zero_uaux(body.point.vel(N))
                acc = zero_udot_uaux(body.point.acc(N))
                inertial_force = (M.diff(t) * vel + M * acc)
                for j in range(o):
                    temp = zero_uaux(partials[i][0][j])
                    for k in range(o):
                        MM[j, k] += M*temp.dot(partials[i][0][k])
                    nonMM[j] += inertial_force.dot(partials[i][0][j])
        # Compose fr_star out of MM and nonMM
        MM = zero_uaux(msubs(MM, q_ddot_u_map))
        nonMM = msubs(msubs(nonMM, q_ddot_u_map),
                udot_zero, uauxdot_zero, uaux_zero)
        fr_star = -(MM * msubs(Matrix(self._udot), uauxdot_zero) + nonMM)

        # If there are dependent speeds, we need to find fr_star_tilde
        if self._udep:
            p = o - len(self._udep)
            fr_star_ind = fr_star[:p, 0]
            fr_star_dep = fr_star[p:o, 0]
            fr_star = fr_star_ind + (self._Ars.T * fr_star_dep)
            # Apply the same to MM
            MMi = MM[:p, :]
            MMd = MM[p:o, :]
            MM = MMi + (self._Ars.T * MMd)
            # Apply the same to nonMM
            nonMM = nonMM[:p, :] + (self._Ars.T * nonMM[p:o, :])

        self._bodylist = bl
        self._frstar = fr_star
        self._k_d = MM
        self._f_d = -(self._fr - nonMM)
        return fr_star

    def to_linearizer(self, linear_solver='LU'):
        """Returns an instance of the Linearizer class, initiated from the
        data in the KanesMethod class. This may be more desirable than using
        the linearize class method, as the Linearizer object will allow more
        efficient recalculation (i.e. about varying operating points).

        Parameters
        ==========
        linear_solver : str, callable
            Method used to solve the several symbolic linear systems of the
            form ``A*x=b`` in the linearization process. If a string is
            supplied, it should be a valid method that can be used with the
            :meth:`sympy.matrices.matrixbase.MatrixBase.solve`. If a callable is
            supplied, it should have the format ``x = f(A, b)``, where it
            solves the equations and returns the solution. The default is
            ``'LU'`` which corresponds to SymPy's ``A.LUsolve(b)``.
            ``LUsolve()`` is fast to compute but will often result in
            divide-by-zero and thus ``nan`` results.

        Returns
        =======
        Linearizer
            An instantiated
            :class:`sympy.physics.mechanics.linearize.Linearizer`.

        """

        if (self._fr is None) or (self._frstar is None):
            raise ValueError('Need to compute Fr, Fr* first.')

        # Get required equation components. The Kane's method class breaks
        # these into pieces. Need to reassemble
        f_c = self._f_h
        if self._f_nh and self._k_nh:
            f_v = self._f_nh + self._k_nh*Matrix(self.u)
        else:
            f_v = Matrix()
        if self._f_dnh and self._k_dnh:
            f_a = self._f_dnh + self._k_dnh*Matrix(self._udot)
        else:
            f_a = Matrix()
        # Dicts to sub to zero, for splitting up expressions
        u_zero = dict.fromkeys(self.u, 0)
        ud_zero = dict.fromkeys(self._udot, 0)
        qd_zero = dict.fromkeys(self._qdot, 0)
        qd_u_zero = dict.fromkeys(Matrix([self._qdot, self.u]), 0)
        # Break the kinematic differential eqs apart into f_0 and f_1
        f_0 = msubs(self._f_k, u_zero) + self._k_kqdot*Matrix(self._qdot)
        f_1 = msubs(self._f_k, qd_zero) + self._k_ku*Matrix(self.u)
        # Break the dynamic differential eqs into f_2 and f_3
        f_2 = msubs(self._frstar, qd_u_zero)
        f_3 = msubs(self._frstar, ud_zero) + self._fr
        f_4 = zeros(len(f_2), 1)

        # Get the required vector components
        q = self.q
        u = self.u
        if self._qdep:
            q_i = q[:-len(self._qdep)]
        else:
            q_i = q
        q_d = self._qdep
        if self._udep:
            u_i = u[:-len(self._udep)]
        else:
            u_i = u
        u_d = self._udep

        # Form dictionary to set auxiliary speeds & their derivatives to 0.
        uaux = self._uaux
        uauxdot = uaux.diff(dynamicsymbols._t)
        uaux_zero = dict.fromkeys(Matrix([uaux, uauxdot]), 0)

        # Checking for dynamic symbols outside the dynamic differential
        # equations; throws error if there is.
        sym_list = set(Matrix([q, self._qdot, u, self._udot, uaux, uauxdot]))
        if any(find_dynamicsymbols(i, sym_list) for i in [self._k_kqdot,
                self._k_ku, self._f_k, self._k_dnh, self._f_dnh, self._k_d]):
            raise ValueError('Cannot have dynamicsymbols outside dynamic \
                             forcing vector.')

        # Find all other dynamic symbols, forming the forcing vector r.
        # Sort r to make it canonical.
        r = list(find_dynamicsymbols(msubs(self._f_d, uaux_zero), sym_list))
        r.sort(key=default_sort_key)

        # Check for any derivatives of variables in r that are also found in r.
        for i in r:
            if diff(i, dynamicsymbols._t) in r:
                raise ValueError('Cannot have derivatives of specified \
                                 quantities when linearizing forcing terms.')
        return Linearizer(f_0, f_1, f_2, f_3, f_4, f_c, f_v, f_a, q, u, q_i,
                q_d, u_i, u_d, r, linear_solver=linear_solver)

    # TODO : Remove `new_method` after 1.1 has been released.
    def linearize(self, *, new_method=None, linear_solver='LU', **kwargs):
        """ Linearize the equations of motion about a symbolic operating point.

        Parameters
        ==========
        new_method
            Deprecated, does nothing and will be removed.
        linear_solver : str, callable
            Method used to solve the several symbolic linear systems of the
            form ``A*x=b`` in the linearization process. If a string is
            supplied, it should be a valid method that can be used with the
            :meth:`sympy.matrices.matrixbase.MatrixBase.solve`. If a callable is
            supplied, it should have the format ``x = f(A, b)``, where it
            solves the equations and returns the solution. The default is
            ``'LU'`` which corresponds to SymPy's ``A.LUsolve(b)``.
            ``LUsolve()`` is fast to compute but will often result in
            divide-by-zero and thus ``nan`` results.
        **kwargs
            Extra keyword arguments are passed to
            :meth:`sympy.physics.mechanics.linearize.Linearizer.linearize`.

        Explanation
        ===========

        If kwarg A_and_B is False (default), returns M, A, B, r for the
        linearized form, M*[q', u']^T = A*[q_ind, u_ind]^T + B*r.

        If kwarg A_and_B is True, returns A, B, r for the linearized form
        dx = A*x + B*r, where x = [q_ind, u_ind]^T. Note that this is
        computationally intensive if there are many symbolic parameters. For
        this reason, it may be more desirable to use the default A_and_B=False,
        returning M, A, and B. Values may then be substituted in to these
        matrices, and the state space form found as
        A = P.T*M.inv()*A, B = P.T*M.inv()*B, where P = Linearizer.perm_mat.

        In both cases, r is found as all dynamicsymbols in the equations of
        motion that are not part of q, u, q', or u'. They are sorted in
        canonical form.

        The operating points may be also entered using the ``op_point`` kwarg.
        This takes a dictionary of {symbol: value}, or a an iterable of such
        dictionaries. The values may be numeric or symbolic. The more values
        you can specify beforehand, the faster this computation will run.

        For more documentation, please see the ``Linearizer`` class.

        """

        linearizer = self.to_linearizer(linear_solver=linear_solver)
        result = linearizer.linearize(**kwargs)
        return result + (linearizer.r,)

    def kanes_equations(self, bodies=None, loads=None):
        """ Method to form Kane's equations, Fr + Fr* = 0.

        Explanation
        ===========

        Returns (Fr, Fr*). In the case where auxiliary generalized speeds are
        present (say, s auxiliary speeds, o generalized speeds, and m motion
        constraints) the length of the returned vectors will be o - m + s in
        length. The first o - m equations will be the constrained Kane's
        equations, then the s auxiliary Kane's equations. These auxiliary
        equations can be accessed with the auxiliary_eqs property.

        Parameters
        ==========

        bodies : iterable
            An iterable of all RigidBody's and Particle's in the system.
            A system must have at least one body.
        loads : iterable
            Takes in an iterable of (Particle, Vector) or (ReferenceFrame, Vector)
            tuples which represent the force at a point or torque on a frame.
            Must be either a non-empty iterable of tuples or None which corresponds
            to a system with no constraints.
        """
        if bodies is None:
            bodies = self.bodies
        if  loads is None and self._forcelist is not None:
            loads = self._forcelist
        if loads == []:
            loads = None
        if not self._k_kqdot:
            raise AttributeError('Create an instance of KanesMethod with '
                    'kinematic differential equations to use this method.')
        fr = self._form_fr(loads)
        frstar = self._form_frstar(bodies)
        if self._uaux:
            if not self._udep:
                km = KanesMethod(self._inertial, self.q, self._uaux,
                             u_auxiliary=self._uaux, constraint_solver=self._constraint_solver)
            else:
                km = KanesMethod(self._inertial, self.q, self._uaux,
                        u_auxiliary=self._uaux, u_dependent=self._udep,
                        velocity_constraints=(self._k_nh * self.u +
                        self._f_nh),
                        acceleration_constraints=(self._k_dnh * self._udot +
                        self._f_dnh),
                        constraint_solver=self._constraint_solver
                        )
            km._qdot_u_map = self._qdot_u_map
            self._km = km
            fraux = km._form_fr(loads)
            frstaraux = km._form_frstar(bodies)
            self._aux_eq = fraux + frstaraux
            self._fr = fr.col_join(fraux)
            self._frstar = frstar.col_join(frstaraux)
        return (self._fr, self._frstar)

    def _form_eoms(self):
        fr, frstar = self.kanes_equations(self.bodylist, self.forcelist)
        return fr + frstar

    def rhs(self, inv_method=None):
        """Returns the system's equations of motion in first order form. The
        output is the right hand side of::

           x' = |q'| =: f(q, u, r, p, t)
                |u'|

        The right hand side is what is needed by most numerical ODE
        integrators.

        Parameters
        ==========

        inv_method : str
            The specific sympy inverse matrix calculation method to use. For a
            list of valid methods, see
            :meth:`~sympy.matrices.matrixbase.MatrixBase.inv`

        """
        rhs = zeros(len(self.q) + len(self.u), 1)
        kdes = self.kindiffdict()
        for i, q_i in enumerate(self.q):
            rhs[i] = kdes[q_i.diff()]

        if inv_method is None:
            rhs[len(self.q):, 0] = self.mass_matrix.LUsolve(self.forcing)
        else:
            rhs[len(self.q):, 0] = (self.mass_matrix.inv(inv_method,
                                                         try_block_diag=True) *
                                    self.forcing)

        return rhs

    def kindiffdict(self):
        """Returns a dictionary mapping q' to u."""
        if not self._qdot_u_map:
            raise AttributeError('Create an instance of KanesMethod with '
                    'kinematic differential equations to use this method.')
        return self._qdot_u_map

    @property
    def auxiliary_eqs(self):
        """A matrix containing the auxiliary equations."""
        if not self._fr or not self._frstar:
            raise ValueError('Need to compute Fr, Fr* first.')
        if not self._uaux:
            raise ValueError('No auxiliary speeds have been declared.')
        return self._aux_eq

    @property
    def mass_matrix_kin(self):
        r"""The kinematic "mass matrix" $\mathbf{k_{k\dot{q}}}$ of the system."""
        return self._k_kqdot if self.explicit_kinematics else self._k_kqdot_implicit

    @property
    def forcing_kin(self):
        """The kinematic "forcing vector" of the system."""
        if self.explicit_kinematics:
            return -(self._k_ku * Matrix(self.u) + self._f_k)
        else:
            return -(self._k_ku_implicit * Matrix(self.u) + self._f_k_implicit)

    @property
    def mass_matrix(self):
        """The mass matrix of the system."""
        if not self._fr or not self._frstar:
            raise ValueError('Need to compute Fr, Fr* first.')
        return Matrix([self._k_d, self._k_dnh])

    @property
    def forcing(self):
        """The forcing vector of the system."""
        if not self._fr or not self._frstar:
            raise ValueError('Need to compute Fr, Fr* first.')
        return -Matrix([self._f_d, self._f_dnh])

    @property
    def mass_matrix_full(self):
        """The mass matrix of the system, augmented by the kinematic
        differential equations in explicit or implicit form."""
        if not self._fr or not self._frstar:
            raise ValueError('Need to compute Fr, Fr* first.')
        o, n = len(self.u), len(self.q)
        return (self.mass_matrix_kin.row_join(zeros(n, o))).col_join(
            zeros(o, n).row_join(self.mass_matrix))

    @property
    def forcing_full(self):
        """The forcing vector of the system, augmented by the kinematic
        differential equations in explicit or implicit form."""
        return Matrix([self.forcing_kin, self.forcing])

    @property
    def q(self):
        return self._q

    @property
    def u(self):
        return self._u

    @property
    def bodylist(self):
        return self._bodylist

    @property
    def forcelist(self):
        return self._forcelist

    @property
    def bodies(self):
        return self._bodylist

    @property
    def loads(self):
        return self._forcelist

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\quantum\identitysearch.py ===
from collections import deque
from sympy.core.random import randint

from sympy.external import import_module
from sympy.core.basic import Basic
from sympy.core.mul import Mul
from sympy.core.numbers import Number, equal_valued
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.dagger import Dagger

__all__ = [
    # Public interfaces
    'generate_gate_rules',
    'generate_equivalent_ids',
    'GateIdentity',
    'bfs_identity_search',
    'random_identity_search',

    # "Private" functions
    'is_scalar_sparse_matrix',
    'is_scalar_nonsparse_matrix',
    'is_degenerate',
    'is_reducible',
]

np = import_module('numpy')
scipy = import_module('scipy', import_kwargs={'fromlist': ['sparse']})


def is_scalar_sparse_matrix(circuit, nqubits, identity_only, eps=1e-11):
    """Checks if a given scipy.sparse matrix is a scalar matrix.

    A scalar matrix is such that B = bI, where B is the scalar
    matrix, b is some scalar multiple, and I is the identity
    matrix.  A scalar matrix would have only the element b along
    it's main diagonal and zeroes elsewhere.

    Parameters
    ==========

    circuit : Gate tuple
        Sequence of quantum gates representing a quantum circuit
    nqubits : int
        Number of qubits in the circuit
    identity_only : bool
        Check for only identity matrices
    eps : number
        The tolerance value for zeroing out elements in the matrix.
        Values in the range [-eps, +eps] will be changed to a zero.
    """

    if not np or not scipy:
        pass

    matrix = represent(Mul(*circuit), nqubits=nqubits,
                       format='scipy.sparse')

    # In some cases, represent returns a 1D scalar value in place
    # of a multi-dimensional scalar matrix
    if (isinstance(matrix, int)):
        return matrix == 1 if identity_only else True

    # If represent returns a matrix, check if the matrix is diagonal
    # and if every item along the diagonal is the same
    else:
        # Due to floating pointing operations, must zero out
        # elements that are "very" small in the dense matrix
        # See parameter for default value.

        # Get the ndarray version of the dense matrix
        dense_matrix = matrix.todense().getA()
        # Since complex values can't be compared, must split
        # the matrix into real and imaginary components
        # Find the real values in between -eps and eps
        bool_real = np.logical_and(dense_matrix.real > -eps,
                                   dense_matrix.real < eps)
        # Find the imaginary values between -eps and eps
        bool_imag = np.logical_and(dense_matrix.imag > -eps,
                                   dense_matrix.imag < eps)
        # Replaces values between -eps and eps with 0
        corrected_real = np.where(bool_real, 0.0, dense_matrix.real)
        corrected_imag = np.where(bool_imag, 0.0, dense_matrix.imag)
        # Convert the matrix with real values into imaginary values
        corrected_imag = corrected_imag * complex(1j)
        # Recombine the real and imaginary components
        corrected_dense = corrected_real + corrected_imag

        # Check if it's diagonal
        row_indices = corrected_dense.nonzero()[0]
        col_indices = corrected_dense.nonzero()[1]
        # Check if the rows indices and columns indices are the same
        # If they match, then matrix only contains elements along diagonal
        bool_indices = row_indices == col_indices
        is_diagonal = bool_indices.all()

        first_element = corrected_dense[0][0]
        # If the first element is a zero, then can't rescale matrix
        # and definitely not diagonal
        if (first_element == 0.0 + 0.0j):
            return False

        # The dimensions of the dense matrix should still
        # be 2^nqubits if there are elements all along the
        # the main diagonal
        trace_of_corrected = (corrected_dense/first_element).trace()
        expected_trace = pow(2, nqubits)
        has_correct_trace = trace_of_corrected == expected_trace

        # If only looking for identity matrices
        # first element must be a 1
        real_is_one = abs(first_element.real - 1.0) < eps
        imag_is_zero = abs(first_element.imag) < eps
        is_one = real_is_one and imag_is_zero
        is_identity = is_one if identity_only else True
        return bool(is_diagonal and has_correct_trace and is_identity)


def is_scalar_nonsparse_matrix(circuit, nqubits, identity_only, eps=None):
    """Checks if a given circuit, in matrix form, is equivalent to
    a scalar value.

    Parameters
    ==========

    circuit : Gate tuple
        Sequence of quantum gates representing a quantum circuit
    nqubits : int
        Number of qubits in the circuit
    identity_only : bool
        Check for only identity matrices
    eps : number
        This argument is ignored. It is just for signature compatibility with
        is_scalar_sparse_matrix.

    Note: Used in situations when is_scalar_sparse_matrix has bugs
    """

    matrix = represent(Mul(*circuit), nqubits=nqubits)

    # In some cases, represent returns a 1D scalar value in place
    # of a multi-dimensional scalar matrix
    if (isinstance(matrix, Number)):
        return matrix == 1 if identity_only else True

    # If represent returns a matrix, check if the matrix is diagonal
    # and if every item along the diagonal is the same
    else:
        # Added up the diagonal elements
        matrix_trace = matrix.trace()
        # Divide the trace by the first element in the matrix
        # if matrix is not required to be the identity matrix
        adjusted_matrix_trace = (matrix_trace/matrix[0]
                                 if not identity_only
                                 else matrix_trace)

        is_identity = equal_valued(matrix[0], 1) if identity_only else True

        has_correct_trace = adjusted_matrix_trace == pow(2, nqubits)

        # The matrix is scalar if it's diagonal and the adjusted trace
        # value is equal to 2^nqubits
        return bool(
            matrix.is_diagonal() and has_correct_trace and is_identity)

if np and scipy:
    is_scalar_matrix = is_scalar_sparse_matrix
else:
    is_scalar_matrix = is_scalar_nonsparse_matrix


def _get_min_qubits(a_gate):
    if isinstance(a_gate, Pow):
        return a_gate.base.min_qubits
    else:
        return a_gate.min_qubits


def ll_op(left, right):
    """Perform a LL operation.

    A LL operation multiplies both left and right circuits
    with the dagger of the left circuit's leftmost gate, and
    the dagger is multiplied on the left side of both circuits.

    If a LL is possible, it returns the new gate rule as a
    2-tuple (LHS, RHS), where LHS is the left circuit and
    and RHS is the right circuit of the new rule.
    If a LL is not possible, None is returned.

    Parameters
    ==========

    left : Gate tuple
        The left circuit of a gate rule expression.
    right : Gate tuple
        The right circuit of a gate rule expression.

    Examples
    ========

    Generate a new gate rule using a LL operation:

    >>> from sympy.physics.quantum.identitysearch import ll_op
    >>> from sympy.physics.quantum.gate import X, Y, Z
    >>> x = X(0); y = Y(0); z = Z(0)
    >>> ll_op((x, y, z), ())
    ((Y(0), Z(0)), (X(0),))

    >>> ll_op((y, z), (x,))
    ((Z(0),), (Y(0), X(0)))
    """

    if (len(left) > 0):
        ll_gate = left[0]
        ll_gate_is_unitary = is_scalar_matrix(
            (Dagger(ll_gate), ll_gate), _get_min_qubits(ll_gate), True)

    if (len(left) > 0 and ll_gate_is_unitary):
        # Get the new left side w/o the leftmost gate
        new_left = left[1:len(left)]
        # Add the leftmost gate to the left position on the right side
        new_right = (Dagger(ll_gate),) + right
        # Return the new gate rule
        return (new_left, new_right)

    return None


def lr_op(left, right):
    """Perform a LR operation.

    A LR operation multiplies both left and right circuits
    with the dagger of the left circuit's rightmost gate, and
    the dagger is multiplied on the right side of both circuits.

    If a LR is possible, it returns the new gate rule as a
    2-tuple (LHS, RHS), where LHS is the left circuit and
    and RHS is the right circuit of the new rule.
    If a LR is not possible, None is returned.

    Parameters
    ==========

    left : Gate tuple
        The left circuit of a gate rule expression.
    right : Gate tuple
        The right circuit of a gate rule expression.

    Examples
    ========

    Generate a new gate rule using a LR operation:

    >>> from sympy.physics.quantum.identitysearch import lr_op
    >>> from sympy.physics.quantum.gate import X, Y, Z
    >>> x = X(0); y = Y(0); z = Z(0)
    >>> lr_op((x, y, z), ())
    ((X(0), Y(0)), (Z(0),))

    >>> lr_op((x, y), (z,))
    ((X(0),), (Z(0), Y(0)))
    """

    if (len(left) > 0):
        lr_gate = left[len(left) - 1]
        lr_gate_is_unitary = is_scalar_matrix(
            (Dagger(lr_gate), lr_gate), _get_min_qubits(lr_gate), True)

    if (len(left) > 0 and lr_gate_is_unitary):
        # Get the new left side w/o the rightmost gate
        new_left = left[0:len(left) - 1]
        # Add the rightmost gate to the right position on the right side
        new_right = right + (Dagger(lr_gate),)
        # Return the new gate rule
        return (new_left, new_right)

    return None


def rl_op(left, right):
    """Perform a RL operation.

    A RL operation multiplies both left and right circuits
    with the dagger of the right circuit's leftmost gate, and
    the dagger is multiplied on the left side of both circuits.

    If a RL is possible, it returns the new gate rule as a
    2-tuple (LHS, RHS), where LHS is the left circuit and
    and RHS is the right circuit of the new rule.
    If a RL is not possible, None is returned.

    Parameters
    ==========

    left : Gate tuple
        The left circuit of a gate rule expression.
    right : Gate tuple
        The right circuit of a gate rule expression.

    Examples
    ========

    Generate a new gate rule using a RL operation:

    >>> from sympy.physics.quantum.identitysearch import rl_op
    >>> from sympy.physics.quantum.gate import X, Y, Z
    >>> x = X(0); y = Y(0); z = Z(0)
    >>> rl_op((x,), (y, z))
    ((Y(0), X(0)), (Z(0),))

    >>> rl_op((x, y), (z,))
    ((Z(0), X(0), Y(0)), ())
    """

    if (len(right) > 0):
        rl_gate = right[0]
        rl_gate_is_unitary = is_scalar_matrix(
            (Dagger(rl_gate), rl_gate), _get_min_qubits(rl_gate), True)

    if (len(right) > 0 and rl_gate_is_unitary):
        # Get the new right side w/o the leftmost gate
        new_right = right[1:len(right)]
        # Add the leftmost gate to the left position on the left side
        new_left = (Dagger(rl_gate),) + left
        # Return the new gate rule
        return (new_left, new_right)

    return None


def rr_op(left, right):
    """Perform a RR operation.

    A RR operation multiplies both left and right circuits
    with the dagger of the right circuit's rightmost gate, and
    the dagger is multiplied on the right side of both circuits.

    If a RR is possible, it returns the new gate rule as a
    2-tuple (LHS, RHS), where LHS is the left circuit and
    and RHS is the right circuit of the new rule.
    If a RR is not possible, None is returned.

    Parameters
    ==========

    left : Gate tuple
        The left circuit of a gate rule expression.
    right : Gate tuple
        The right circuit of a gate rule expression.

    Examples
    ========

    Generate a new gate rule using a RR operation:

    >>> from sympy.physics.quantum.identitysearch import rr_op
    >>> from sympy.physics.quantum.gate import X, Y, Z
    >>> x = X(0); y = Y(0); z = Z(0)
    >>> rr_op((x, y), (z,))
    ((X(0), Y(0), Z(0)), ())

    >>> rr_op((x,), (y, z))
    ((X(0), Z(0)), (Y(0),))
    """

    if (len(right) > 0):
        rr_gate = right[len(right) - 1]
        rr_gate_is_unitary = is_scalar_matrix(
            (Dagger(rr_gate), rr_gate), _get_min_qubits(rr_gate), True)

    if (len(right) > 0 and rr_gate_is_unitary):
        # Get the new right side w/o the rightmost gate
        new_right = right[0:len(right) - 1]
        # Add the rightmost gate to the right position on the right side
        new_left = left + (Dagger(rr_gate),)
        # Return the new gate rule
        return (new_left, new_right)

    return None


def generate_gate_rules(gate_seq, return_as_muls=False):
    """Returns a set of gate rules.  Each gate rules is represented
    as a 2-tuple of tuples or Muls.  An empty tuple represents an arbitrary
    scalar value.

    This function uses the four operations (LL, LR, RL, RR)
    to generate the gate rules.

    A gate rule is an expression such as ABC = D or AB = CD, where
    A, B, C, and D are gates.  Each value on either side of the
    equal sign represents a circuit.  The four operations allow
    one to find a set of equivalent circuits from a gate identity.
    The letters denoting the operation tell the user what
    activities to perform on each expression.  The first letter
    indicates which side of the equal sign to focus on.  The
    second letter indicates which gate to focus on given the
    side.  Once this information is determined, the inverse
    of the gate is multiplied on both circuits to create a new
    gate rule.

    For example, given the identity, ABCD = 1, a LL operation
    means look at the left value and multiply both left sides by the
    inverse of the leftmost gate A.  If A is Hermitian, the inverse
    of A is still A.  The resulting new rule is BCD = A.

    The following is a summary of the four operations.  Assume
    that in the examples, all gates are Hermitian.

        LL : left circuit, left multiply
             ABCD = E -> AABCD = AE -> BCD = AE
        LR : left circuit, right multiply
             ABCD = E -> ABCDD = ED -> ABC = ED
        RL : right circuit, left multiply
             ABC = ED -> EABC = EED -> EABC = D
        RR : right circuit, right multiply
             AB = CD -> ABD = CDD -> ABD = C

    The number of gate rules generated is n*(n+1), where n
    is the number of gates in the sequence (unproven).

    Parameters
    ==========

    gate_seq : Gate tuple, Mul, or Number
        A variable length tuple or Mul of Gates whose product is equal to
        a scalar matrix
    return_as_muls : bool
        True to return a set of Muls; False to return a set of tuples

    Examples
    ========

    Find the gate rules of the current circuit using tuples:

    >>> from sympy.physics.quantum.identitysearch import generate_gate_rules
    >>> from sympy.physics.quantum.gate import X, Y, Z
    >>> x = X(0); y = Y(0); z = Z(0)
    >>> generate_gate_rules((x, x))
    {((X(0),), (X(0),)), ((X(0), X(0)), ())}

    >>> generate_gate_rules((x, y, z))
    {((), (X(0), Z(0), Y(0))), ((), (Y(0), X(0), Z(0))),
     ((), (Z(0), Y(0), X(0))), ((X(0),), (Z(0), Y(0))),
     ((Y(0),), (X(0), Z(0))), ((Z(0),), (Y(0), X(0))),
     ((X(0), Y(0)), (Z(0),)), ((Y(0), Z(0)), (X(0),)),
     ((Z(0), X(0)), (Y(0),)), ((X(0), Y(0), Z(0)), ()),
     ((Y(0), Z(0), X(0)), ()), ((Z(0), X(0), Y(0)), ())}

    Find the gate rules of the current circuit using Muls:

    >>> generate_gate_rules(x*x, return_as_muls=True)
    {(1, 1)}

    >>> generate_gate_rules(x*y*z, return_as_muls=True)
    {(1, X(0)*Z(0)*Y(0)), (1, Y(0)*X(0)*Z(0)),
     (1, Z(0)*Y(0)*X(0)), (X(0)*Y(0), Z(0)),
     (Y(0)*Z(0), X(0)), (Z(0)*X(0), Y(0)),
     (X(0)*Y(0)*Z(0), 1), (Y(0)*Z(0)*X(0), 1),
     (Z(0)*X(0)*Y(0), 1), (X(0), Z(0)*Y(0)),
     (Y(0), X(0)*Z(0)), (Z(0), Y(0)*X(0))}
    """

    if isinstance(gate_seq, Number):
        if return_as_muls:
            return {(S.One, S.One)}
        else:
            return {((), ())}

    elif isinstance(gate_seq, Mul):
        gate_seq = gate_seq.args

    # Each item in queue is a 3-tuple:
    #     i)   first item is the left side of an equality
    #    ii)   second item is the right side of an equality
    #   iii)   third item is the number of operations performed
    # The argument, gate_seq, will start on the left side, and
    # the right side will be empty, implying the presence of an
    # identity.
    queue = deque()
    # A set of gate rules
    rules = set()
    # Maximum number of operations to perform
    max_ops = len(gate_seq)

    def process_new_rule(new_rule, ops):
        if new_rule is not None:
            new_left, new_right = new_rule

            if new_rule not in rules and (new_right, new_left) not in rules:
                rules.add(new_rule)
            # If haven't reached the max limit on operations
            if ops + 1 < max_ops:
                queue.append(new_rule + (ops + 1,))

    queue.append((gate_seq, (), 0))
    rules.add((gate_seq, ()))

    while len(queue) > 0:
        left, right, ops = queue.popleft()

        # Do a LL
        new_rule = ll_op(left, right)
        process_new_rule(new_rule, ops)
        # Do a LR
        new_rule = lr_op(left, right)
        process_new_rule(new_rule, ops)
        # Do a RL
        new_rule = rl_op(left, right)
        process_new_rule(new_rule, ops)
        # Do a RR
        new_rule = rr_op(left, right)
        process_new_rule(new_rule, ops)

    if return_as_muls:
        # Convert each rule as tuples into a rule as muls
        mul_rules = set()
        for rule in rules:
            left, right = rule
            mul_rules.add((Mul(*left), Mul(*right)))

        rules = mul_rules

    return rules


def generate_equivalent_ids(gate_seq, return_as_muls=False):
    """Returns a set of equivalent gate identities.

    A gate identity is a quantum circuit such that the product
    of the gates in the circuit is equal to a scalar value.
    For example, XYZ = i, where X, Y, Z are the Pauli gates and
    i is the imaginary value, is considered a gate identity.

    This function uses the four operations (LL, LR, RL, RR)
    to generate the gate rules and, subsequently, to locate equivalent
    gate identities.

    Note that all equivalent identities are reachable in n operations
    from the starting gate identity, where n is the number of gates
    in the sequence.

    The max number of gate identities is 2n, where n is the number
    of gates in the sequence (unproven).

    Parameters
    ==========

    gate_seq : Gate tuple, Mul, or Number
        A variable length tuple or Mul of Gates whose product is equal to
        a scalar matrix.
    return_as_muls: bool
        True to return as Muls; False to return as tuples

    Examples
    ========

    Find equivalent gate identities from the current circuit with tuples:

    >>> from sympy.physics.quantum.identitysearch import generate_equivalent_ids
    >>> from sympy.physics.quantum.gate import X, Y, Z
    >>> x = X(0); y = Y(0); z = Z(0)
    >>> generate_equivalent_ids((x, x))
    {(X(0), X(0))}

    >>> generate_equivalent_ids((x, y, z))
    {(X(0), Y(0), Z(0)), (X(0), Z(0), Y(0)), (Y(0), X(0), Z(0)),
     (Y(0), Z(0), X(0)), (Z(0), X(0), Y(0)), (Z(0), Y(0), X(0))}

    Find equivalent gate identities from the current circuit with Muls:

    >>> generate_equivalent_ids(x*x, return_as_muls=True)
    {1}

    >>> generate_equivalent_ids(x*y*z, return_as_muls=True)
    {X(0)*Y(0)*Z(0), X(0)*Z(0)*Y(0), Y(0)*X(0)*Z(0),
     Y(0)*Z(0)*X(0), Z(0)*X(0)*Y(0), Z(0)*Y(0)*X(0)}
    """

    if isinstance(gate_seq, Number):
        return {S.One}
    elif isinstance(gate_seq, Mul):
        gate_seq = gate_seq.args

    # Filter through the gate rules and keep the rules
    # with an empty tuple either on the left or right side

    # A set of equivalent gate identities
    eq_ids = set()

    gate_rules = generate_gate_rules(gate_seq)
    for rule in gate_rules:
        l, r = rule
        if l == ():
            eq_ids.add(r)
        elif r == ():
            eq_ids.add(l)

    if return_as_muls:
        convert_to_mul = lambda id_seq: Mul(*id_seq)
        eq_ids = set(map(convert_to_mul, eq_ids))

    return eq_ids


class GateIdentity(Basic):
    """Wrapper class for circuits that reduce to a scalar value.

    A gate identity is a quantum circuit such that the product
    of the gates in the circuit is equal to a scalar value.
    For example, XYZ = i, where X, Y, Z are the Pauli gates and
    i is the imaginary value, is considered a gate identity.

    Parameters
    ==========

    args : Gate tuple
        A variable length tuple of Gates that form an identity.

    Examples
    ========

    Create a GateIdentity and look at its attributes:

    >>> from sympy.physics.quantum.identitysearch import GateIdentity
    >>> from sympy.physics.quantum.gate import X, Y, Z
    >>> x = X(0); y = Y(0); z = Z(0)
    >>> an_identity = GateIdentity(x, y, z)
    >>> an_identity.circuit
    X(0)*Y(0)*Z(0)

    >>> an_identity.equivalent_ids
    {(X(0), Y(0), Z(0)), (X(0), Z(0), Y(0)), (Y(0), X(0), Z(0)),
     (Y(0), Z(0), X(0)), (Z(0), X(0), Y(0)), (Z(0), Y(0), X(0))}
    """

    def __new__(cls, *args):
        # args should be a tuple - a variable length argument list
        obj = Basic.__new__(cls, *args)
        obj._circuit = Mul(*args)
        obj._rules = generate_gate_rules(args)
        obj._eq_ids = generate_equivalent_ids(args)

        return obj

    @property
    def circuit(self):
        return self._circuit

    @property
    def gate_rules(self):
        return self._rules

    @property
    def equivalent_ids(self):
        return self._eq_ids

    @property
    def sequence(self):
        return self.args

    def __str__(self):
        """Returns the string of gates in a tuple."""
        return str(self.circuit)


def is_degenerate(identity_set, gate_identity):
    """Checks if a gate identity is a permutation of another identity.

    Parameters
    ==========

    identity_set : set
        A Python set with GateIdentity objects.
    gate_identity : GateIdentity
        The GateIdentity to check for existence in the set.

    Examples
    ========

    Check if the identity is a permutation of another identity:

    >>> from sympy.physics.quantum.identitysearch import (
    ...     GateIdentity, is_degenerate)
    >>> from sympy.physics.quantum.gate import X, Y, Z
    >>> x = X(0); y = Y(0); z = Z(0)
    >>> an_identity = GateIdentity(x, y, z)
    >>> id_set = {an_identity}
    >>> another_id = (y, z, x)
    >>> is_degenerate(id_set, another_id)
    True

    >>> another_id = (x, x)
    >>> is_degenerate(id_set, another_id)
    False
    """

    # For now, just iteratively go through the set and check if the current
    # gate_identity is a permutation of an identity in the set
    for an_id in identity_set:
        if (gate_identity in an_id.equivalent_ids):
            return True
    return False


def is_reducible(circuit, nqubits, begin, end):
    """Determines if a circuit is reducible by checking
    if its subcircuits are scalar values.

    Parameters
    ==========

    circuit : Gate tuple
        A tuple of Gates representing a circuit.  The circuit to check
        if a gate identity is contained in a subcircuit.
    nqubits : int
        The number of qubits the circuit operates on.
    begin : int
        The leftmost gate in the circuit to include in a subcircuit.
    end : int
        The rightmost gate in the circuit to include in a subcircuit.

    Examples
    ========

    Check if the circuit can be reduced:

    >>> from sympy.physics.quantum.identitysearch import is_reducible
    >>> from sympy.physics.quantum.gate import X, Y, Z
    >>> x = X(0); y = Y(0); z = Z(0)
    >>> is_reducible((x, y, z), 1, 0, 3)
    True

    Check if an interval in the circuit can be reduced:

    >>> is_reducible((x, y, z), 1, 1, 3)
    False

    >>> is_reducible((x, y, y), 1, 1, 3)
    True
    """

    current_circuit = ()
    # Start from the gate at "end" and go down to almost the gate at "begin"
    for ndx in reversed(range(begin, end)):
        next_gate = circuit[ndx]
        current_circuit = (next_gate,) + current_circuit

        # If a circuit as a matrix is equivalent to a scalar value
        if (is_scalar_matrix(current_circuit, nqubits, False)):
            return True

    return False


def bfs_identity_search(gate_list, nqubits, max_depth=None,
       identity_only=False):
    """Constructs a set of gate identities from the list of possible gates.

    Performs a breadth first search over the space of gate identities.
    This allows the finding of the shortest gate identities first.

    Parameters
    ==========

    gate_list : list, Gate
        A list of Gates from which to search for gate identities.
    nqubits : int
        The number of qubits the quantum circuit operates on.
    max_depth : int
        The longest quantum circuit to construct from gate_list.
    identity_only : bool
        True to search for gate identities that reduce to identity;
        False to search for gate identities that reduce to a scalar.

    Examples
    ========

    Find a list of gate identities:

    >>> from sympy.physics.quantum.identitysearch import bfs_identity_search
    >>> from sympy.physics.quantum.gate import X, Y, Z
    >>> x = X(0); y = Y(0); z = Z(0)
    >>> bfs_identity_search([x], 1, max_depth=2)
    {GateIdentity(X(0), X(0))}

    >>> bfs_identity_search([x, y, z], 1)
    {GateIdentity(X(0), X(0)), GateIdentity(Y(0), Y(0)),
     GateIdentity(Z(0), Z(0)), GateIdentity(X(0), Y(0), Z(0))}

    Find a list of identities that only equal to 1:

    >>> bfs_identity_search([x, y, z], 1, identity_only=True)
    {GateIdentity(X(0), X(0)), GateIdentity(Y(0), Y(0)),
     GateIdentity(Z(0), Z(0))}
    """

    if max_depth is None or max_depth <= 0:
        max_depth = len(gate_list)

    id_only = identity_only

    # Start with an empty sequence (implicitly contains an IdentityGate)
    queue = deque([()])

    # Create an empty set of gate identities
    ids = set()

    # Begin searching for gate identities in given space.
    while (len(queue) > 0):
        current_circuit = queue.popleft()

        for next_gate in gate_list:
            new_circuit = current_circuit + (next_gate,)

            # Determines if a (strict) subcircuit is a scalar matrix
            circuit_reducible = is_reducible(new_circuit, nqubits,
                                             1, len(new_circuit))

            # In many cases when the matrix is a scalar value,
            # the evaluated matrix will actually be an integer
            if (is_scalar_matrix(new_circuit, nqubits, id_only) and
                not is_degenerate(ids, new_circuit) and
                    not circuit_reducible):
                ids.add(GateIdentity(*new_circuit))

            elif (len(new_circuit) < max_depth and
                  not circuit_reducible):
                queue.append(new_circuit)

    return ids


def random_identity_search(gate_list, numgates, nqubits):
    """Randomly selects numgates from gate_list and checks if it is
    a gate identity.

    If the circuit is a gate identity, the circuit is returned;
    Otherwise, None is returned.
    """

    gate_size = len(gate_list)
    circuit = ()

    for i in range(numgates):
        next_gate = gate_list[randint(0, gate_size - 1)]
        circuit = circuit + (next_gate,)

    is_scalar = is_scalar_matrix(circuit, nqubits, False)

    return circuit if is_scalar else None

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\pycode.py ===
"""
Python code printers

This module contains Python code printers for plain Python as well as NumPy & SciPy enabled code.
"""
from collections import defaultdict
from itertools import chain
from sympy.core import S
from sympy.core.mod import Mod
from .precedence import precedence
from .codeprinter import CodePrinter

_kw = {
    'and', 'as', 'assert', 'break', 'class', 'continue', 'def', 'del', 'elif',
    'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in',
    'is', 'lambda', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while',
    'with', 'yield', 'None', 'False', 'nonlocal', 'True'
}

_known_functions = {
    'Abs': 'abs',
    'Min': 'min',
    'Max': 'max',
}
_known_functions_math = {
    'acos': 'acos',
    'acosh': 'acosh',
    'asin': 'asin',
    'asinh': 'asinh',
    'atan': 'atan',
    'atan2': 'atan2',
    'atanh': 'atanh',
    'ceiling': 'ceil',
    'cos': 'cos',
    'cosh': 'cosh',
    'erf': 'erf',
    'erfc': 'erfc',
    'exp': 'exp',
    'expm1': 'expm1',
    'factorial': 'factorial',
    'floor': 'floor',
    'gamma': 'gamma',
    'hypot': 'hypot',
    'isinf': 'isinf',
    'isnan': 'isnan',
    'loggamma': 'lgamma',
    'log': 'log',
    'ln': 'log',
    'log10': 'log10',
    'log1p': 'log1p',
    'log2': 'log2',
    'sin': 'sin',
    'sinh': 'sinh',
    'Sqrt': 'sqrt',
    'tan': 'tan',
    'tanh': 'tanh'
}  # Not used from ``math``: [copysign isclose isfinite isinf ldexp frexp pow modf
# radians trunc fmod fsum gcd degrees fabs]
_known_constants_math = {
    'Exp1': 'e',
    'Pi': 'pi',
    'E': 'e',
    'Infinity': 'inf',
    'NaN': 'nan',
    'ComplexInfinity': 'nan'
}

def _print_known_func(self, expr):
    known = self.known_functions[expr.__class__.__name__]
    return '{name}({args})'.format(name=self._module_format(known),
                                   args=', '.join((self._print(arg) for arg in expr.args)))


def _print_known_const(self, expr):
    known = self.known_constants[expr.__class__.__name__]
    return self._module_format(known)


class AbstractPythonCodePrinter(CodePrinter):
    printmethod = "_pythoncode"
    language = "Python"
    reserved_words = _kw
    modules = None  # initialized to a set in __init__
    tab = '    '
    _kf = dict(chain(
        _known_functions.items(),
        [(k, 'math.' + v) for k, v in _known_functions_math.items()]
    ))
    _kc = {k: 'math.'+v for k, v in _known_constants_math.items()}
    _operators = {'and': 'and', 'or': 'or', 'not': 'not'}
    _default_settings = dict(
        CodePrinter._default_settings,
        user_functions={},
        precision=17,
        inline=True,
        fully_qualified_modules=True,
        contract=False,
        standard='python3',
    )

    def __init__(self, settings=None):
        super().__init__(settings)

        # Python standard handler
        std = self._settings['standard']
        if std is None:
            import sys
            std = 'python{}'.format(sys.version_info.major)
        if std != 'python3':
            raise ValueError('Only Python 3 is supported.')
        self.standard = std

        self.module_imports = defaultdict(set)

        # Known functions and constants handler
        self.known_functions = dict(self._kf, **(settings or {}).get(
            'user_functions', {}))
        self.known_constants = dict(self._kc, **(settings or {}).get(
            'user_constants', {}))

    def _declare_number_const(self, name, value):
        return "%s = %s" % (name, value)

    def _module_format(self, fqn, register=True):
        parts = fqn.split('.')
        if register and len(parts) > 1:
            self.module_imports['.'.join(parts[:-1])].add(parts[-1])

        if self._settings['fully_qualified_modules']:
            return fqn
        else:
            return fqn.split('(')[0].split('[')[0].split('.')[-1]

    def _format_code(self, lines):
        return lines

    def _get_statement(self, codestring):
        return "{}".format(codestring)

    def _get_comment(self, text):
        return "  # {}".format(text)

    def _expand_fold_binary_op(self, op, args):
        """
        This method expands a fold on binary operations.

        ``functools.reduce`` is an example of a folded operation.

        For example, the expression

        `A + B + C + D`

        is folded into

        `((A + B) + C) + D`
        """
        if len(args) == 1:
            return self._print(args[0])
        else:
            return "%s(%s, %s)" % (
                self._module_format(op),
                self._expand_fold_binary_op(op, args[:-1]),
                self._print(args[-1]),
            )

    def _expand_reduce_binary_op(self, op, args):
        """
        This method expands a reduction on binary operations.

        Notice: this is NOT the same as ``functools.reduce``.

        For example, the expression

        `A + B + C + D`

        is reduced into:

        `(A + B) + (C + D)`
        """
        if len(args) == 1:
            return self._print(args[0])
        else:
            N = len(args)
            Nhalf = N // 2
            return "%s(%s, %s)" % (
                self._module_format(op),
                self._expand_reduce_binary_op(args[:Nhalf]),
                self._expand_reduce_binary_op(args[Nhalf:]),
            )

    def _print_NaN(self, expr):
        return "float('nan')"

    def _print_Infinity(self, expr):
        return "float('inf')"

    def _print_NegativeInfinity(self, expr):
        return "float('-inf')"

    def _print_ComplexInfinity(self, expr):
        return self._print_NaN(expr)

    def _print_Mod(self, expr):
        PREC = precedence(expr)
        return ('{} % {}'.format(*(self.parenthesize(x, PREC) for x in expr.args)))

    def _print_Piecewise(self, expr):
        result = []
        i = 0
        for arg in expr.args:
            e = arg.expr
            c = arg.cond
            if i == 0:
                result.append('(')
            result.append('(')
            result.append(self._print(e))
            result.append(')')
            result.append(' if ')
            result.append(self._print(c))
            result.append(' else ')
            i += 1
        result = result[:-1]
        if result[-1] == 'True':
            result = result[:-2]
            result.append(')')
        else:
            result.append(' else None)')
        return ''.join(result)

    def _print_Relational(self, expr):
        "Relational printer for Equality and Unequality"
        op = {
            '==' :'equal',
            '!=' :'not_equal',
            '<'  :'less',
            '<=' :'less_equal',
            '>'  :'greater',
            '>=' :'greater_equal',
        }
        if expr.rel_op in op:
            lhs = self._print(expr.lhs)
            rhs = self._print(expr.rhs)
            return '({lhs} {op} {rhs})'.format(op=expr.rel_op, lhs=lhs, rhs=rhs)
        return super()._print_Relational(expr)

    def _print_ITE(self, expr):
        from sympy.functions.elementary.piecewise import Piecewise
        return self._print(expr.rewrite(Piecewise))

    def _print_Sum(self, expr):
        loops = (
            'for {i} in range({a}, {b}+1)'.format(
                i=self._print(i),
                a=self._print(a),
                b=self._print(b))
            for i, a, b in expr.limits[::-1])
        return '(builtins.sum({function} {loops}))'.format(
            function=self._print(expr.function),
            loops=' '.join(loops))

    def _print_ImaginaryUnit(self, expr):
        return '1j'

    def _print_KroneckerDelta(self, expr):
        a, b = expr.args

        return '(1 if {a} == {b} else 0)'.format(
            a = self._print(a),
            b = self._print(b)
        )

    def _print_MatrixBase(self, expr):
        name = expr.__class__.__name__
        func = self.known_functions.get(name, name)
        return "%s(%s)" % (func, self._print(expr.tolist()))

    _print_SparseRepMatrix = \
        _print_MutableSparseMatrix = \
        _print_ImmutableSparseMatrix = \
        _print_Matrix = \
        _print_DenseMatrix = \
        _print_MutableDenseMatrix = \
        _print_ImmutableMatrix = \
        _print_ImmutableDenseMatrix = \
        lambda self, expr: self._print_MatrixBase(expr)

    def _indent_codestring(self, codestring):
        return '\n'.join([self.tab + line for line in codestring.split('\n')])

    def _print_FunctionDefinition(self, fd):
        body = '\n'.join((self._print(arg) for arg in fd.body))
        return "def {name}({parameters}):\n{body}".format(
            name=self._print(fd.name),
            parameters=', '.join([self._print(var.symbol) for var in fd.parameters]),
            body=self._indent_codestring(body)
        )

    def _print_While(self, whl):
        body = '\n'.join((self._print(arg) for arg in whl.body))
        return "while {cond}:\n{body}".format(
            cond=self._print(whl.condition),
            body=self._indent_codestring(body)
        )

    def _print_Declaration(self, decl):
        return '%s = %s' % (
            self._print(decl.variable.symbol),
            self._print(decl.variable.value)
        )

    def _print_BreakToken(self, bt):
        return 'break'

    def _print_Return(self, ret):
        arg, = ret.args
        return 'return %s' % self._print(arg)

    def _print_Raise(self, rs):
        arg, = rs.args
        return 'raise %s' % self._print(arg)

    def _print_RuntimeError_(self, re):
        message, = re.args
        return "RuntimeError(%s)" % self._print(message)

    def _print_Print(self, prnt):
        print_args = ', '.join((self._print(arg) for arg in prnt.print_args))
        from sympy.codegen.ast import none
        if prnt.format_string != none:
            print_args = '{} % ({}), end=""'.format(
                self._print(prnt.format_string),
                print_args
            )
        if prnt.file != None: # Must be '!= None', cannot be 'is not None'
            print_args += ', file=%s' % self._print(prnt.file)
        return 'print(%s)' % print_args

    def _print_Stream(self, strm):
        if str(strm.name) == 'stdout':
            return self._module_format('sys.stdout')
        elif str(strm.name) == 'stderr':
            return self._module_format('sys.stderr')
        else:
            return self._print(strm.name)

    def _print_NoneToken(self, arg):
        return 'None'

    def _hprint_Pow(self, expr, rational=False, sqrt='math.sqrt'):
        """Printing helper function for ``Pow``

        Notes
        =====

        This preprocesses the ``sqrt`` as math formatter and prints division

        Examples
        ========

        >>> from sympy import sqrt
        >>> from sympy.printing.pycode import PythonCodePrinter
        >>> from sympy.abc import x

        Python code printer automatically looks up ``math.sqrt``.

        >>> printer = PythonCodePrinter()
        >>> printer._hprint_Pow(sqrt(x), rational=True)
        'x**(1/2)'
        >>> printer._hprint_Pow(sqrt(x), rational=False)
        'math.sqrt(x)'
        >>> printer._hprint_Pow(1/sqrt(x), rational=True)
        'x**(-1/2)'
        >>> printer._hprint_Pow(1/sqrt(x), rational=False)
        '1/math.sqrt(x)'
        >>> printer._hprint_Pow(1/x, rational=False)
        '1/x'
        >>> printer._hprint_Pow(1/x, rational=True)
        'x**(-1)'

        Using sqrt from numpy or mpmath

        >>> printer._hprint_Pow(sqrt(x), sqrt='numpy.sqrt')
        'numpy.sqrt(x)'
        >>> printer._hprint_Pow(sqrt(x), sqrt='mpmath.sqrt')
        'mpmath.sqrt(x)'

        See Also
        ========

        sympy.printing.str.StrPrinter._print_Pow
        """
        PREC = precedence(expr)

        if expr.exp == S.Half and not rational:
            func = self._module_format(sqrt)
            arg = self._print(expr.base)
            return '{func}({arg})'.format(func=func, arg=arg)

        if expr.is_commutative and not rational:
            if -expr.exp is S.Half:
                func = self._module_format(sqrt)
                num = self._print(S.One)
                arg = self._print(expr.base)
                return f"{num}/{func}({arg})"
            if expr.exp is S.NegativeOne:
                num = self._print(S.One)
                arg = self.parenthesize(expr.base, PREC, strict=False)
                return f"{num}/{arg}"


        base_str = self.parenthesize(expr.base, PREC, strict=False)
        exp_str = self.parenthesize(expr.exp, PREC, strict=False)
        return "{}**{}".format(base_str, exp_str)


class ArrayPrinter:

    def _arrayify(self, indexed):
        from sympy.tensor.array.expressions.from_indexed_to_array import convert_indexed_to_array
        try:
            return convert_indexed_to_array(indexed)
        except Exception:
            return indexed

    def _get_einsum_string(self, subranks, contraction_indices):
        letters = self._get_letter_generator_for_einsum()
        contraction_string = ""
        counter = 0
        d = {j: min(i) for i in contraction_indices for j in i}
        indices = []
        for rank_arg in subranks:
            lindices = []
            for i in range(rank_arg):
                if counter in d:
                    lindices.append(d[counter])
                else:
                    lindices.append(counter)
                counter += 1
            indices.append(lindices)
        mapping = {}
        letters_free = []
        letters_dum = []
        for i in indices:
            for j in i:
                if j not in mapping:
                    l = next(letters)
                    mapping[j] = l
                else:
                    l = mapping[j]
                contraction_string += l
                if j in d:
                    if l not in letters_dum:
                        letters_dum.append(l)
                else:
                    letters_free.append(l)
            contraction_string += ","
        contraction_string = contraction_string[:-1]
        return contraction_string, letters_free, letters_dum

    def _get_letter_generator_for_einsum(self):
        for i in range(97, 123):
            yield chr(i)
        for i in range(65, 91):
            yield chr(i)
        raise ValueError("out of letters")

    def _print_ArrayTensorProduct(self, expr):
        letters = self._get_letter_generator_for_einsum()
        contraction_string = ",".join(["".join([next(letters) for j in range(i)]) for i in expr.subranks])
        return '%s("%s", %s)' % (
                self._module_format(self._module + "." + self._einsum),
                contraction_string,
                ", ".join([self._print(arg) for arg in expr.args])
        )

    def _print_ArrayContraction(self, expr):
        from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct
        base = expr.expr
        contraction_indices = expr.contraction_indices

        if isinstance(base, ArrayTensorProduct):
            elems = ",".join(["%s" % (self._print(arg)) for arg in base.args])
            ranks = base.subranks
        else:
            elems = self._print(base)
            ranks = [len(base.shape)]

        contraction_string, letters_free, letters_dum = self._get_einsum_string(ranks, contraction_indices)

        if not contraction_indices:
            return self._print(base)
        if isinstance(base, ArrayTensorProduct):
            elems = ",".join(["%s" % (self._print(arg)) for arg in base.args])
        else:
            elems = self._print(base)
        return "%s(\"%s\", %s)" % (
            self._module_format(self._module + "." + self._einsum),
            "{}->{}".format(contraction_string, "".join(sorted(letters_free))),
            elems,
        )

    def _print_ArrayDiagonal(self, expr):
        from sympy.tensor.array.expressions.array_expressions import ArrayTensorProduct
        diagonal_indices = list(expr.diagonal_indices)
        if isinstance(expr.expr, ArrayTensorProduct):
            subranks = expr.expr.subranks
            elems = expr.expr.args
        else:
            subranks = expr.subranks
            elems = [expr.expr]
        diagonal_string, letters_free, letters_dum = self._get_einsum_string(subranks, diagonal_indices)
        elems = [self._print(i) for i in elems]
        return '%s("%s", %s)' % (
            self._module_format(self._module + "." + self._einsum),
            "{}->{}".format(diagonal_string, "".join(letters_free+letters_dum)),
            ", ".join(elems)
        )

    def _print_PermuteDims(self, expr):
        return "%s(%s, %s)" % (
            self._module_format(self._module + "." + self._transpose),
            self._print(expr.expr),
            self._print(expr.permutation.array_form),
        )

    def _print_ArrayAdd(self, expr):
        return self._expand_fold_binary_op(self._module + "." + self._add, expr.args)

    def _print_OneArray(self, expr):
        return "%s((%s,))" % (
            self._module_format(self._module+ "." + self._ones),
            ','.join(map(self._print,expr.args))
        )

    def _print_ZeroArray(self, expr):
        return "%s((%s,))" % (
            self._module_format(self._module+ "." + self._zeros),
            ','.join(map(self._print,expr.args))
        )

    def _print_Assignment(self, expr):
        #XXX: maybe this needs to happen at a higher level e.g. at _print or
        #doprint?
        lhs = self._print(self._arrayify(expr.lhs))
        rhs = self._print(self._arrayify(expr.rhs))
        return "%s = %s" % ( lhs, rhs )

    def _print_IndexedBase(self, expr):
        return self._print_ArraySymbol(expr)


class PythonCodePrinter(AbstractPythonCodePrinter):

    def _print_sign(self, e):
        return '(0.0 if {e} == 0 else {f}(1, {e}))'.format(
            f=self._module_format('math.copysign'), e=self._print(e.args[0]))

    def _print_Not(self, expr):
        PREC = precedence(expr)
        return self._operators['not'] + ' ' + self.parenthesize(expr.args[0], PREC)

    def _print_IndexedBase(self, expr):
        return expr.name

    def _print_Indexed(self, expr):
        base = expr.args[0]
        index = expr.args[1:]
        return "{}[{}]".format(str(base), ", ".join([self._print(ind) for ind in index]))

    def _print_Pow(self, expr, rational=False):
        return self._hprint_Pow(expr, rational=rational)

    def _print_Rational(self, expr):
        return '{}/{}'.format(expr.p, expr.q)

    def _print_Half(self, expr):
        return self._print_Rational(expr)

    def _print_frac(self, expr):
        return self._print_Mod(Mod(expr.args[0], 1))

    def _print_Symbol(self, expr):

        name = super()._print_Symbol(expr)

        if name in self.reserved_words:
            if self._settings['error_on_reserved']:
                msg = ('This expression includes the symbol "{}" which is a '
                       'reserved keyword in this language.')
                raise ValueError(msg.format(name))
            return name + self._settings['reserved_word_suffix']
        elif '{' in name:   # Remove curly braces from subscripted variables
            return name.replace('{', '').replace('}', '')
        else:
            return name

    _print_lowergamma = CodePrinter._print_not_supported
    _print_uppergamma = CodePrinter._print_not_supported
    _print_fresnelc = CodePrinter._print_not_supported
    _print_fresnels = CodePrinter._print_not_supported


for k in PythonCodePrinter._kf:
    setattr(PythonCodePrinter, '_print_%s' % k, _print_known_func)

for k in _known_constants_math:
    setattr(PythonCodePrinter, '_print_%s' % k, _print_known_const)


def pycode(expr, **settings):
    """ Converts an expr to a string of Python code

    Parameters
    ==========

    expr : Expr
        A SymPy expression.
    fully_qualified_modules : bool
        Whether or not to write out full module names of functions
        (``math.sin`` vs. ``sin``). default: ``True``.
    standard : str or None, optional
        Only 'python3' (default) is supported.
        This parameter may be removed in the future.

    Examples
    ========

    >>> from sympy import pycode, tan, Symbol
    >>> pycode(tan(Symbol('x')) + 1)
    'math.tan(x) + 1'

    """
    return PythonCodePrinter(settings).doprint(expr)


from itertools import chain
from sympy.printing.pycode import PythonCodePrinter

_known_functions_cmath = {
    'exp': 'exp',
    'sqrt': 'sqrt',
    'log': 'log',
    'cos': 'cos',
    'sin': 'sin',
    'tan': 'tan',
    'acos': 'acos',
    'asin': 'asin',
    'atan': 'atan',
    'cosh': 'cosh',
    'sinh': 'sinh',
    'tanh': 'tanh',
    'acosh': 'acosh',
    'asinh': 'asinh',
    'atanh': 'atanh',
}

_known_constants_cmath = {
    'Pi': 'pi',
    'E': 'e',
    'Infinity': 'inf',
    'NegativeInfinity': '-inf',
}

class CmathPrinter(PythonCodePrinter):
    """ Printer for Python's cmath module """
    printmethod = "_cmathcode"
    language = "Python with cmath"

    _kf = dict(chain(
        _known_functions_cmath.items()
    ))

    _kc = {k: 'cmath.' + v for k, v in _known_constants_cmath.items()}

    def _print_Pow(self, expr, rational=False):
        return self._hprint_Pow(expr, rational=rational, sqrt='cmath.sqrt')

    def _print_Float(self, e):
        return '{func}({val})'.format(func=self._module_format('cmath.mpf'), val=self._print(e))

    def _print_known_func(self, expr):
        func_name = expr.func.__name__
        if func_name in self._kf:
            return f"cmath.{self._kf[func_name]}({', '.join(map(self._print, expr.args))})"
        return super()._print_Function(expr)

    def _print_known_const(self, expr):
        return self._kc[expr.__class__.__name__]

    def _print_re(self, expr):
        """Prints `re(z)` as `z.real`"""
        return f"({self._print(expr.args[0])}).real"

    def _print_im(self, expr):
        """Prints `im(z)` as `z.imag`"""
        return f"({self._print(expr.args[0])}).imag"


for k in CmathPrinter._kf:
    setattr(CmathPrinter, '_print_%s' % k, CmathPrinter._print_known_func)

for k in _known_constants_cmath:
    setattr(CmathPrinter, '_print_%s' % k, CmathPrinter._print_known_const)


_not_in_mpmath = 'log1p log2'.split()
_in_mpmath = [(k, v) for k, v in _known_functions_math.items() if k not in _not_in_mpmath]
_known_functions_mpmath = dict(_in_mpmath, **{
    'beta': 'beta',
    'frac': 'frac',
    'fresnelc': 'fresnelc',
    'fresnels': 'fresnels',
    'sign': 'sign',
    'loggamma': 'loggamma',
    'hyper': 'hyper',
    'meijerg': 'meijerg',
    'besselj': 'besselj',
    'bessely': 'bessely',
    'besseli': 'besseli',
    'besselk': 'besselk',
})
_known_constants_mpmath = {
    'Exp1': 'e',
    'Pi': 'pi',
    'GoldenRatio': 'phi',
    'EulerGamma': 'euler',
    'Catalan': 'catalan',
    'NaN': 'nan',
    'Infinity': 'inf',
    'NegativeInfinity': 'ninf'
}


def _unpack_integral_limits(integral_expr):
    """ helper function for _print_Integral that
        - accepts an Integral expression
        - returns a tuple of
           - a list variables of integration
           - a list of tuples of the upper and lower limits of integration
    """
    integration_vars = []
    limits = []
    for integration_range in integral_expr.limits:
        if len(integration_range) == 3:
            integration_var, lower_limit, upper_limit = integration_range
        else:
            raise NotImplementedError("Only definite integrals are supported")
        integration_vars.append(integration_var)
        limits.append((lower_limit, upper_limit))
    return integration_vars, limits


class MpmathPrinter(PythonCodePrinter):
    """
    Lambda printer for mpmath which maintains precision for floats
    """
    printmethod = "_mpmathcode"

    language = "Python with mpmath"

    _kf = dict(chain(
        _known_functions.items(),
        [(k, 'mpmath.' + v) for k, v in _known_functions_mpmath.items()]
    ))
    _kc = {k: 'mpmath.'+v for k, v in _known_constants_mpmath.items()}

    def _print_Float(self, e):
        # XXX: This does not handle setting mpmath.mp.dps. It is assumed that
        # the caller of the lambdified function will have set it to sufficient
        # precision to match the Floats in the expression.

        # Remove 'mpz' if gmpy is installed.
        args = str(tuple(map(int, e._mpf_)))
        return '{func}({args})'.format(func=self._module_format('mpmath.mpf'), args=args)


    def _print_Rational(self, e):
        return "{func}({p})/{func}({q})".format(
            func=self._module_format('mpmath.mpf'),
            q=self._print(e.q),
            p=self._print(e.p)
        )

    def _print_Half(self, e):
        return self._print_Rational(e)

    def _print_uppergamma(self, e):
        return "{}({}, {}, {})".format(
            self._module_format('mpmath.gammainc'),
            self._print(e.args[0]),
            self._print(e.args[1]),
            self._module_format('mpmath.inf'))

    def _print_lowergamma(self, e):
        return "{}({}, 0, {})".format(
            self._module_format('mpmath.gammainc'),
            self._print(e.args[0]),
            self._print(e.args[1]))

    def _print_log2(self, e):
        return '{0}({1})/{0}(2)'.format(
            self._module_format('mpmath.log'), self._print(e.args[0]))

    def _print_log1p(self, e):
        return '{}({})'.format(
            self._module_format('mpmath.log1p'), self._print(e.args[0]))

    def _print_Pow(self, expr, rational=False):
        return self._hprint_Pow(expr, rational=rational, sqrt='mpmath.sqrt')

    def _print_Integral(self, e):
        integration_vars, limits = _unpack_integral_limits(e)

        return "{}(lambda {}: {}, {})".format(
                self._module_format("mpmath.quad"),
                ", ".join(map(self._print, integration_vars)),
                self._print(e.args[0]),
                ", ".join("(%s, %s)" % tuple(map(self._print, l)) for l in limits))


    def _print_Derivative_zeta(self, args, seq_orders):
        arg, = args
        deriv_order, = seq_orders
        return '{}({}, derivative={})'.format(
            self._module_format('mpmath.zeta'),
            self._print(arg), deriv_order
        )


for k in MpmathPrinter._kf:
    setattr(MpmathPrinter, '_print_%s' % k, _print_known_func)

for k in _known_constants_mpmath:
    setattr(MpmathPrinter, '_print_%s' % k, _print_known_const)


class SymPyPrinter(AbstractPythonCodePrinter):

    language = "Python with SymPy"

    _default_settings = dict(
        AbstractPythonCodePrinter._default_settings,
        strict=False   # any class name will per definition be what we target in SymPyPrinter.
    )

    def _print_Function(self, expr):
        mod = expr.func.__module__ or ''
        return '%s(%s)' % (self._module_format(mod + ('.' if mod else '') + expr.func.__name__),
                           ', '.join((self._print(arg) for arg in expr.args)))

    def _print_Pow(self, expr, rational=False):
        return self._hprint_Pow(expr, rational=rational, sqrt='sympy.sqrt')

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\errors\__init__.py ===
"""
Expose public exceptions & warnings
"""
from __future__ import annotations

import ctypes

from pandas._config.config import OptionError

from pandas._libs.tslibs import (
    OutOfBoundsDatetime,
    OutOfBoundsTimedelta,
)

from pandas.util.version import InvalidVersion


class IntCastingNaNError(ValueError):
    """
    Exception raised when converting (``astype``) an array with NaN to an integer type.

    Examples
    --------
    >>> pd.DataFrame(np.array([[1, np.nan], [2, 3]]), dtype="i8")
    Traceback (most recent call last):
    IntCastingNaNError: Cannot convert non-finite values (NA or inf) to integer
    """


class NullFrequencyError(ValueError):
    """
    Exception raised when a ``freq`` cannot be null.

    Particularly ``DatetimeIndex.shift``, ``TimedeltaIndex.shift``,
    ``PeriodIndex.shift``.

    Examples
    --------
    >>> df = pd.DatetimeIndex(["2011-01-01 10:00", "2011-01-01"], freq=None)
    >>> df.shift(2)
    Traceback (most recent call last):
    NullFrequencyError: Cannot shift with no freq
    """


class PerformanceWarning(Warning):
    """
    Warning raised when there is a possible performance impact.

    Examples
    --------
    >>> df = pd.DataFrame({"jim": [0, 0, 1, 1],
    ...                    "joe": ["x", "x", "z", "y"],
    ...                    "jolie": [1, 2, 3, 4]})
    >>> df = df.set_index(["jim", "joe"])
    >>> df
              jolie
    jim  joe
    0    x    1
         x    2
    1    z    3
         y    4
    >>> df.loc[(1, 'z')]  # doctest: +SKIP
    # PerformanceWarning: indexing past lexsort depth may impact performance.
    df.loc[(1, 'z')]
              jolie
    jim  joe
    1    z        3
    """


class UnsupportedFunctionCall(ValueError):
    """
    Exception raised when attempting to call a unsupported numpy function.

    For example, ``np.cumsum(groupby_object)``.

    Examples
    --------
    >>> df = pd.DataFrame({"A": [0, 0, 1, 1],
    ...                    "B": ["x", "x", "z", "y"],
    ...                    "C": [1, 2, 3, 4]}
    ...                   )
    >>> np.cumsum(df.groupby(["A"]))
    Traceback (most recent call last):
    UnsupportedFunctionCall: numpy operations are not valid with groupby.
    Use .groupby(...).cumsum() instead
    """


class UnsortedIndexError(KeyError):
    """
    Error raised when slicing a MultiIndex which has not been lexsorted.

    Subclass of `KeyError`.

    Examples
    --------
    >>> df = pd.DataFrame({"cat": [0, 0, 1, 1],
    ...                    "color": ["white", "white", "brown", "black"],
    ...                    "lives": [4, 4, 3, 7]},
    ...                   )
    >>> df = df.set_index(["cat", "color"])
    >>> df
                lives
    cat  color
    0    white    4
         white    4
    1    brown    3
         black    7
    >>> df.loc[(0, "black"):(1, "white")]
    Traceback (most recent call last):
    UnsortedIndexError: 'Key length (2) was greater
    than MultiIndex lexsort depth (1)'
    """


class ParserError(ValueError):
    """
    Exception that is raised by an error encountered in parsing file contents.

    This is a generic error raised for errors encountered when functions like
    `read_csv` or `read_html` are parsing contents of a file.

    See Also
    --------
    read_csv : Read CSV (comma-separated) file into a DataFrame.
    read_html : Read HTML table into a DataFrame.

    Examples
    --------
    >>> data = '''a,b,c
    ... cat,foo,bar
    ... dog,foo,"baz'''
    >>> from io import StringIO
    >>> pd.read_csv(StringIO(data), skipfooter=1, engine='python')
    Traceback (most recent call last):
    ParserError: ',' expected after '"'. Error could possibly be due
    to parsing errors in the skipped footer rows
    """


class DtypeWarning(Warning):
    """
    Warning raised when reading different dtypes in a column from a file.

    Raised for a dtype incompatibility. This can happen whenever `read_csv`
    or `read_table` encounter non-uniform dtypes in a column(s) of a given
    CSV file.

    See Also
    --------
    read_csv : Read CSV (comma-separated) file into a DataFrame.
    read_table : Read general delimited file into a DataFrame.

    Notes
    -----
    This warning is issued when dealing with larger files because the dtype
    checking happens per chunk read.

    Despite the warning, the CSV file is read with mixed types in a single
    column which will be an object type. See the examples below to better
    understand this issue.

    Examples
    --------
    This example creates and reads a large CSV file with a column that contains
    `int` and `str`.

    >>> df = pd.DataFrame({'a': (['1'] * 100000 + ['X'] * 100000 +
    ...                          ['1'] * 100000),
    ...                    'b': ['b'] * 300000})  # doctest: +SKIP
    >>> df.to_csv('test.csv', index=False)  # doctest: +SKIP
    >>> df2 = pd.read_csv('test.csv')  # doctest: +SKIP
    ... # DtypeWarning: Columns (0) have mixed types

    Important to notice that ``df2`` will contain both `str` and `int` for the
    same input, '1'.

    >>> df2.iloc[262140, 0]  # doctest: +SKIP
    '1'
    >>> type(df2.iloc[262140, 0])  # doctest: +SKIP
    <class 'str'>
    >>> df2.iloc[262150, 0]  # doctest: +SKIP
    1
    >>> type(df2.iloc[262150, 0])  # doctest: +SKIP
    <class 'int'>

    One way to solve this issue is using the `dtype` parameter in the
    `read_csv` and `read_table` functions to explicit the conversion:

    >>> df2 = pd.read_csv('test.csv', sep=',', dtype={'a': str})  # doctest: +SKIP

    No warning was issued.
    """


class EmptyDataError(ValueError):
    """
    Exception raised in ``pd.read_csv`` when empty data or header is encountered.

    Examples
    --------
    >>> from io import StringIO
    >>> empty = StringIO()
    >>> pd.read_csv(empty)
    Traceback (most recent call last):
    EmptyDataError: No columns to parse from file
    """


class ParserWarning(Warning):
    """
    Warning raised when reading a file that doesn't use the default 'c' parser.

    Raised by `pd.read_csv` and `pd.read_table` when it is necessary to change
    parsers, generally from the default 'c' parser to 'python'.

    It happens due to a lack of support or functionality for parsing a
    particular attribute of a CSV file with the requested engine.

    Currently, 'c' unsupported options include the following parameters:

    1. `sep` other than a single character (e.g. regex separators)
    2. `skipfooter` higher than 0
    3. `sep=None` with `delim_whitespace=False`

    The warning can be avoided by adding `engine='python'` as a parameter in
    `pd.read_csv` and `pd.read_table` methods.

    See Also
    --------
    pd.read_csv : Read CSV (comma-separated) file into DataFrame.
    pd.read_table : Read general delimited file into DataFrame.

    Examples
    --------
    Using a `sep` in `pd.read_csv` other than a single character:

    >>> import io
    >>> csv = '''a;b;c
    ...           1;1,8
    ...           1;2,1'''
    >>> df = pd.read_csv(io.StringIO(csv), sep='[;,]')  # doctest: +SKIP
    ... # ParserWarning: Falling back to the 'python' engine...

    Adding `engine='python'` to `pd.read_csv` removes the Warning:

    >>> df = pd.read_csv(io.StringIO(csv), sep='[;,]', engine='python')
    """


class MergeError(ValueError):
    """
    Exception raised when merging data.

    Subclass of ``ValueError``.

    Examples
    --------
    >>> left = pd.DataFrame({"a": ["a", "b", "b", "d"],
    ...                     "b": ["cat", "dog", "weasel", "horse"]},
    ...                     index=range(4))
    >>> right = pd.DataFrame({"a": ["a", "b", "c", "d"],
    ...                      "c": ["meow", "bark", "chirp", "nay"]},
    ...                      index=range(4)).set_index("a")
    >>> left.join(right, on="a", validate="one_to_one",)
    Traceback (most recent call last):
    MergeError: Merge keys are not unique in left dataset; not a one-to-one merge
    """


class AbstractMethodError(NotImplementedError):
    """
    Raise this error instead of NotImplementedError for abstract methods.

    Examples
    --------
    >>> class Foo:
    ...     @classmethod
    ...     def classmethod(cls):
    ...         raise pd.errors.AbstractMethodError(cls, methodtype="classmethod")
    ...     def method(self):
    ...         raise pd.errors.AbstractMethodError(self)
    >>> test = Foo.classmethod()
    Traceback (most recent call last):
    AbstractMethodError: This classmethod must be defined in the concrete class Foo

    >>> test2 = Foo().method()
    Traceback (most recent call last):
    AbstractMethodError: This classmethod must be defined in the concrete class Foo
    """

    def __init__(self, class_instance, methodtype: str = "method") -> None:
        types = {"method", "classmethod", "staticmethod", "property"}
        if methodtype not in types:
            raise ValueError(
                f"methodtype must be one of {methodtype}, got {types} instead."
            )
        self.methodtype = methodtype
        self.class_instance = class_instance

    def __str__(self) -> str:
        if self.methodtype == "classmethod":
            name = self.class_instance.__name__
        else:
            name = type(self.class_instance).__name__
        return f"This {self.methodtype} must be defined in the concrete class {name}"


class NumbaUtilError(Exception):
    """
    Error raised for unsupported Numba engine routines.

    Examples
    --------
    >>> df = pd.DataFrame({"key": ["a", "a", "b", "b"], "data": [1, 2, 3, 4]},
    ...                   columns=["key", "data"])
    >>> def incorrect_function(x):
    ...     return sum(x) * 2.7
    >>> df.groupby("key").agg(incorrect_function, engine="numba")
    Traceback (most recent call last):
    NumbaUtilError: The first 2 arguments to incorrect_function
    must be ['values', 'index']
    """


class DuplicateLabelError(ValueError):
    """
    Error raised when an operation would introduce duplicate labels.

    Examples
    --------
    >>> s = pd.Series([0, 1, 2], index=['a', 'b', 'c']).set_flags(
    ...     allows_duplicate_labels=False
    ... )
    >>> s.reindex(['a', 'a', 'b'])
    Traceback (most recent call last):
       ...
    DuplicateLabelError: Index has duplicates.
          positions
    label
    a        [0, 1]
    """


class InvalidIndexError(Exception):
    """
    Exception raised when attempting to use an invalid index key.

    Examples
    --------
    >>> idx = pd.MultiIndex.from_product([["x", "y"], [0, 1]])
    >>> df = pd.DataFrame([[1, 1, 2, 2],
    ...                   [3, 3, 4, 4]], columns=idx)
    >>> df
        x       y
        0   1   0   1
    0   1   1   2   2
    1   3   3   4   4
    >>> df[:, 0]
    Traceback (most recent call last):
    InvalidIndexError: (slice(None, None, None), 0)
    """


class DataError(Exception):
    """
    Exceptionn raised when performing an operation on non-numerical data.

    For example, calling ``ohlc`` on a non-numerical column or a function
    on a rolling window.

    Examples
    --------
    >>> ser = pd.Series(['a', 'b', 'c'])
    >>> ser.rolling(2).sum()
    Traceback (most recent call last):
    DataError: No numeric types to aggregate
    """


class SpecificationError(Exception):
    """
    Exception raised by ``agg`` when the functions are ill-specified.

    The exception raised in two scenarios.

    The first way is calling ``agg`` on a
    Dataframe or Series using a nested renamer (dict-of-dict).

    The second way is calling ``agg`` on a Dataframe with duplicated functions
    names without assigning column name.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 1, 1, 2, 2],
    ...                    'B': range(5),
    ...                    'C': range(5)})
    >>> df.groupby('A').B.agg({'foo': 'count'}) # doctest: +SKIP
    ... # SpecificationError: nested renamer is not supported

    >>> df.groupby('A').agg({'B': {'foo': ['sum', 'max']}}) # doctest: +SKIP
    ... # SpecificationError: nested renamer is not supported

    >>> df.groupby('A').agg(['min', 'min']) # doctest: +SKIP
    ... # SpecificationError: nested renamer is not supported
    """


class SettingWithCopyError(ValueError):
    """
    Exception raised when trying to set on a copied slice from a ``DataFrame``.

    The ``mode.chained_assignment`` needs to be set to set to 'raise.' This can
    happen unintentionally when chained indexing.

    For more information on evaluation order,
    see :ref:`the user guide<indexing.evaluation_order>`.

    For more information on view vs. copy,
    see :ref:`the user guide<indexing.view_versus_copy>`.

    Examples
    --------
    >>> pd.options.mode.chained_assignment = 'raise'
    >>> df = pd.DataFrame({'A': [1, 1, 1, 2, 2]}, columns=['A'])
    >>> df.loc[0:3]['A'] = 'a' # doctest: +SKIP
    ... # SettingWithCopyError: A value is trying to be set on a copy of a...
    """


class SettingWithCopyWarning(Warning):
    """
    Warning raised when trying to set on a copied slice from a ``DataFrame``.

    The ``mode.chained_assignment`` needs to be set to set to 'warn.'
    'Warn' is the default option. This can happen unintentionally when
    chained indexing.

    For more information on evaluation order,
    see :ref:`the user guide<indexing.evaluation_order>`.

    For more information on view vs. copy,
    see :ref:`the user guide<indexing.view_versus_copy>`.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 1, 1, 2, 2]}, columns=['A'])
    >>> df.loc[0:3]['A'] = 'a' # doctest: +SKIP
    ... # SettingWithCopyWarning: A value is trying to be set on a copy of a...
    """


class ChainedAssignmentError(Warning):
    """
    Warning raised when trying to set using chained assignment.

    When the ``mode.copy_on_write`` option is enabled, chained assignment can
    never work. In such a situation, we are always setting into a temporary
    object that is the result of an indexing operation (getitem), which under
    Copy-on-Write always behaves as a copy. Thus, assigning through a chain
    can never update the original Series or DataFrame.

    For more information on view vs. copy,
    see :ref:`the user guide<indexing.view_versus_copy>`.

    Examples
    --------
    >>> pd.options.mode.copy_on_write = True
    >>> df = pd.DataFrame({'A': [1, 1, 1, 2, 2]}, columns=['A'])
    >>> df["A"][0:3] = 10 # doctest: +SKIP
    ... # ChainedAssignmentError: ...
    >>> pd.options.mode.copy_on_write = False
    """


_chained_assignment_msg = (
    "A value is trying to be set on a copy of a DataFrame or Series "
    "through chained assignment.\n"
    "When using the Copy-on-Write mode, such chained assignment never works "
    "to update the original DataFrame or Series, because the intermediate "
    "object on which we are setting values always behaves as a copy.\n\n"
    "Try using '.loc[row_indexer, col_indexer] = value' instead, to perform "
    "the assignment in a single step.\n\n"
    "See the caveats in the documentation: "
    "https://pandas.pydata.org/pandas-docs/stable/user_guide/"
    "indexing.html#returning-a-view-versus-a-copy"
)


_chained_assignment_method_msg = (
    "A value is trying to be set on a copy of a DataFrame or Series "
    "through chained assignment using an inplace method.\n"
    "When using the Copy-on-Write mode, such inplace method never works "
    "to update the original DataFrame or Series, because the intermediate "
    "object on which we are setting values always behaves as a copy.\n\n"
    "For example, when doing 'df[col].method(value, inplace=True)', try "
    "using 'df.method({col: value}, inplace=True)' instead, to perform "
    "the operation inplace on the original object.\n\n"
)


_chained_assignment_warning_msg = (
    "ChainedAssignmentError: behaviour will change in pandas 3.0!\n"
    "You are setting values through chained assignment. Currently this works "
    "in certain cases, but when using Copy-on-Write (which will become the "
    "default behaviour in pandas 3.0) this will never work to update the "
    "original DataFrame or Series, because the intermediate object on which "
    "we are setting values will behave as a copy.\n"
    "A typical example is when you are setting values in a column of a "
    "DataFrame, like:\n\n"
    'df["col"][row_indexer] = value\n\n'
    'Use `df.loc[row_indexer, "col"] = values` instead, to perform the '
    "assignment in a single step and ensure this keeps updating the original `df`.\n\n"
    "See the caveats in the documentation: "
    "https://pandas.pydata.org/pandas-docs/stable/user_guide/"
    "indexing.html#returning-a-view-versus-a-copy\n"
)


_chained_assignment_warning_method_msg = (
    "A value is trying to be set on a copy of a DataFrame or Series "
    "through chained assignment using an inplace method.\n"
    "The behavior will change in pandas 3.0. This inplace method will "
    "never work because the intermediate object on which we are setting "
    "values always behaves as a copy.\n\n"
    "For example, when doing 'df[col].method(value, inplace=True)', try "
    "using 'df.method({col: value}, inplace=True)' or "
    "df[col] = df[col].method(value) instead, to perform "
    "the operation inplace on the original object.\n\n"
)


def _check_cacher(obj):
    # This is a mess, selection paths that return a view set the _cacher attribute
    # on the Series; most of them also set _item_cache which adds 1 to our relevant
    # reference count, but iloc does not, so we have to check if we are actually
    # in the item cache
    if hasattr(obj, "_cacher"):
        parent = obj._cacher[1]()
        # parent could be dead
        if parent is None:
            return False
        if hasattr(parent, "_item_cache"):
            if obj._cacher[0] in parent._item_cache:
                # Check if we are actually the item from item_cache, iloc creates a
                # new object
                return obj is parent._item_cache[obj._cacher[0]]
    return False


class NumExprClobberingError(NameError):
    """
    Exception raised when trying to use a built-in numexpr name as a variable name.

    ``eval`` or ``query`` will throw the error if the engine is set
    to 'numexpr'. 'numexpr' is the default engine value for these methods if the
    numexpr package is installed.

    Examples
    --------
    >>> df = pd.DataFrame({'abs': [1, 1, 1]})
    >>> df.query("abs > 2") # doctest: +SKIP
    ... # NumExprClobberingError: Variables in expression "(abs) > (2)" overlap...
    >>> sin, a = 1, 2
    >>> pd.eval("sin + a", engine='numexpr') # doctest: +SKIP
    ... # NumExprClobberingError: Variables in expression "(sin) + (a)" overlap...
    """


class UndefinedVariableError(NameError):
    """
    Exception raised by ``query`` or ``eval`` when using an undefined variable name.

    It will also specify whether the undefined variable is local or not.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 1, 1]})
    >>> df.query("A > x") # doctest: +SKIP
    ... # UndefinedVariableError: name 'x' is not defined
    >>> df.query("A > @y") # doctest: +SKIP
    ... # UndefinedVariableError: local variable 'y' is not defined
    >>> pd.eval('x + 1') # doctest: +SKIP
    ... # UndefinedVariableError: name 'x' is not defined
    """

    def __init__(self, name: str, is_local: bool | None = None) -> None:
        base_msg = f"{repr(name)} is not defined"
        if is_local:
            msg = f"local variable {base_msg}"
        else:
            msg = f"name {base_msg}"
        super().__init__(msg)


class IndexingError(Exception):
    """
    Exception is raised when trying to index and there is a mismatch in dimensions.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 1, 1]})
    >>> df.loc[..., ..., 'A'] # doctest: +SKIP
    ... # IndexingError: indexer may only contain one '...' entry
    >>> df = pd.DataFrame({'A': [1, 1, 1]})
    >>> df.loc[1, ..., ...] # doctest: +SKIP
    ... # IndexingError: Too many indexers
    >>> df[pd.Series([True], dtype=bool)] # doctest: +SKIP
    ... # IndexingError: Unalignable boolean Series provided as indexer...
    >>> s = pd.Series(range(2),
    ...               index = pd.MultiIndex.from_product([["a", "b"], ["c"]]))
    >>> s.loc["a", "c", "d"] # doctest: +SKIP
    ... # IndexingError: Too many indexers
    """


class PyperclipException(RuntimeError):
    """
    Exception raised when clipboard functionality is unsupported.

    Raised by ``to_clipboard()`` and ``read_clipboard()``.
    """


class PyperclipWindowsException(PyperclipException):
    """
    Exception raised when clipboard functionality is unsupported by Windows.

    Access to the clipboard handle would be denied due to some other
    window process is accessing it.
    """

    def __init__(self, message: str) -> None:
        # attr only exists on Windows, so typing fails on other platforms
        message += f" ({ctypes.WinError()})"  # type: ignore[attr-defined]
        super().__init__(message)


class CSSWarning(UserWarning):
    """
    Warning is raised when converting css styling fails.

    This can be due to the styling not having an equivalent value or because the
    styling isn't properly formatted.

    Examples
    --------
    >>> df = pd.DataFrame({'A': [1, 1, 1]})
    >>> df.style.applymap(
    ...     lambda x: 'background-color: blueGreenRed;'
    ... ).to_excel('styled.xlsx')  # doctest: +SKIP
    CSSWarning: Unhandled color format: 'blueGreenRed'
    >>> df.style.applymap(
    ...     lambda x: 'border: 1px solid red red;'
    ... ).to_excel('styled.xlsx')  # doctest: +SKIP
    CSSWarning: Unhandled color format: 'blueGreenRed'
    """


class PossibleDataLossError(Exception):
    """
    Exception raised when trying to open a HDFStore file when already opened.

    Examples
    --------
    >>> store = pd.HDFStore('my-store', 'a') # doctest: +SKIP
    >>> store.open("w") # doctest: +SKIP
    ... # PossibleDataLossError: Re-opening the file [my-store] with mode [a]...
    """


class ClosedFileError(Exception):
    """
    Exception is raised when trying to perform an operation on a closed HDFStore file.

    Examples
    --------
    >>> store = pd.HDFStore('my-store', 'a') # doctest: +SKIP
    >>> store.close() # doctest: +SKIP
    >>> store.keys() # doctest: +SKIP
    ... # ClosedFileError: my-store file is not open!
    """


class IncompatibilityWarning(Warning):
    """
    Warning raised when trying to use where criteria on an incompatible HDF5 file.
    """


class AttributeConflictWarning(Warning):
    """
    Warning raised when index attributes conflict when using HDFStore.

    Occurs when attempting to append an index with a different
    name than the existing index on an HDFStore or attempting to append an index with a
    different frequency than the existing index on an HDFStore.

    Examples
    --------
    >>> idx1 = pd.Index(['a', 'b'], name='name1')
    >>> df1 = pd.DataFrame([[1, 2], [3, 4]], index=idx1)
    >>> df1.to_hdf('file', 'data', 'w', append=True)  # doctest: +SKIP
    >>> idx2 = pd.Index(['c', 'd'], name='name2')
    >>> df2 = pd.DataFrame([[5, 6], [7, 8]], index=idx2)
    >>> df2.to_hdf('file', 'data', 'a', append=True)  # doctest: +SKIP
    AttributeConflictWarning: the [index_name] attribute of the existing index is
    [name1] which conflicts with the new [name2]...
    """


class DatabaseError(OSError):
    """
    Error is raised when executing sql with bad syntax or sql that throws an error.

    Examples
    --------
    >>> from sqlite3 import connect
    >>> conn = connect(':memory:')
    >>> pd.read_sql('select * test', conn) # doctest: +SKIP
    ... # DatabaseError: Execution failed on sql 'test': near "test": syntax error
    """


class PossiblePrecisionLoss(Warning):
    """
    Warning raised by to_stata on a column with a value outside or equal to int64.

    When the column value is outside or equal to the int64 value the column is
    converted to a float64 dtype.

    Examples
    --------
    >>> df = pd.DataFrame({"s": pd.Series([1, 2**53], dtype=np.int64)})
    >>> df.to_stata('test') # doctest: +SKIP
    ... # PossiblePrecisionLoss: Column converted from int64 to float64...
    """


class ValueLabelTypeMismatch(Warning):
    """
    Warning raised by to_stata on a category column that contains non-string values.

    Examples
    --------
    >>> df = pd.DataFrame({"categories": pd.Series(["a", 2], dtype="category")})
    >>> df.to_stata('test') # doctest: +SKIP
    ... # ValueLabelTypeMismatch: Stata value labels (pandas categories) must be str...
    """


class InvalidColumnName(Warning):
    """
    Warning raised by to_stata the column contains a non-valid stata name.

    Because the column name is an invalid Stata variable, the name needs to be
    converted.

    Examples
    --------
    >>> df = pd.DataFrame({"0categories": pd.Series([2, 2])})
    >>> df.to_stata('test') # doctest: +SKIP
    ... # InvalidColumnName: Not all pandas column names were valid Stata variable...
    """


class CategoricalConversionWarning(Warning):
    """
    Warning is raised when reading a partial labeled Stata file using a iterator.

    Examples
    --------
    >>> from pandas.io.stata import StataReader
    >>> with StataReader('dta_file', chunksize=2) as reader: # doctest: +SKIP
    ...   for i, block in enumerate(reader):
    ...      print(i, block)
    ... # CategoricalConversionWarning: One or more series with value labels...
    """


class LossySetitemError(Exception):
    """
    Raised when trying to do a __setitem__ on an np.ndarray that is not lossless.

    Notes
    -----
    This is an internal error.
    """


class NoBufferPresent(Exception):
    """
    Exception is raised in _get_data_buffer to signal that there is no requested buffer.
    """


class InvalidComparison(Exception):
    """
    Exception is raised by _validate_comparison_value to indicate an invalid comparison.

    Notes
    -----
    This is an internal error.
    """


__all__ = [
    "AbstractMethodError",
    "AttributeConflictWarning",
    "CategoricalConversionWarning",
    "ClosedFileError",
    "CSSWarning",
    "DatabaseError",
    "DataError",
    "DtypeWarning",
    "DuplicateLabelError",
    "EmptyDataError",
    "IncompatibilityWarning",
    "IntCastingNaNError",
    "InvalidColumnName",
    "InvalidComparison",
    "InvalidIndexError",
    "InvalidVersion",
    "IndexingError",
    "LossySetitemError",
    "MergeError",
    "NoBufferPresent",
    "NullFrequencyError",
    "NumbaUtilError",
    "NumExprClobberingError",
    "OptionError",
    "OutOfBoundsDatetime",
    "OutOfBoundsTimedelta",
    "ParserError",
    "ParserWarning",
    "PerformanceWarning",
    "PossibleDataLossError",
    "PossiblePrecisionLoss",
    "PyperclipException",
    "PyperclipWindowsException",
    "SettingWithCopyError",
    "SettingWithCopyWarning",
    "SpecificationError",
    "UndefinedVariableError",
    "UnsortedIndexError",
    "UnsupportedFunctionCall",
    "ValueLabelTypeMismatch",
]

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\stats\drv_types.py ===
"""

Contains
========
FlorySchulz
Geometric
Hermite
Logarithmic
NegativeBinomial
Poisson
Skellam
YuleSimon
Zeta
"""



from sympy.concrete.summations import Sum
from sympy.core.basic import Basic
from sympy.core.function import Lambda
from sympy.core.numbers import I
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.functions.combinatorial.factorials import (binomial, factorial, FallingFactorial)
from sympy.functions.elementary.exponential import (exp, log)
from sympy.functions.elementary.integers import floor
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.piecewise import Piecewise
from sympy.functions.special.bessel import besseli
from sympy.functions.special.beta_functions import beta
from sympy.functions.special.hyper import hyper
from sympy.functions.special.zeta_functions import (polylog, zeta)
from sympy.stats.drv import SingleDiscreteDistribution, SingleDiscretePSpace
from sympy.stats.rv import _value_check, is_random


__all__ = ['FlorySchulz',
'Geometric',
'Hermite',
'Logarithmic',
'NegativeBinomial',
'Poisson',
'Skellam',
'YuleSimon',
'Zeta'
]


def rv(symbol, cls, *args, **kwargs):
    args = list(map(sympify, args))
    dist = cls(*args)
    if kwargs.pop('check', True):
        dist.check(*args)
    pspace = SingleDiscretePSpace(symbol, dist)
    if any(is_random(arg) for arg in args):
        from sympy.stats.compound_rv import CompoundPSpace, CompoundDistribution
        pspace = CompoundPSpace(symbol, CompoundDistribution(dist))
    return pspace.value


class DiscreteDistributionHandmade(SingleDiscreteDistribution):
    _argnames = ('pdf',)

    def __new__(cls, pdf, set=S.Integers):
        return Basic.__new__(cls, pdf, set)

    @property
    def set(self):
        return self.args[1]

    @staticmethod
    def check(pdf, set):
        x = Dummy('x')
        val = Sum(pdf(x), (x, set._inf, set._sup)).doit()
        _value_check(Eq(val, 1) != S.false, "The pdf is incorrect on the given set.")



def DiscreteRV(symbol, density, set=S.Integers, **kwargs):
    """
    Create a Discrete Random Variable given the following:

    Parameters
    ==========

    symbol : Symbol
        Represents name of the random variable.
    density : Expression containing symbol
        Represents probability density function.
    set : set
        Represents the region where the pdf is valid, by default is real line.
    check : bool
        If True, it will check whether the given density
        integrates to 1 over the given set. If False, it
        will not perform this check. Default is False.

    Examples
    ========

    >>> from sympy.stats import DiscreteRV, P, E
    >>> from sympy import Rational, Symbol
    >>> x = Symbol('x')
    >>> n = 10
    >>> density = Rational(1, 10)
    >>> X = DiscreteRV(x, density, set=set(range(n)))
    >>> E(X)
    9/2
    >>> P(X>3)
    3/5

    Returns
    =======

    RandomSymbol

    """
    set = sympify(set)
    pdf = Piecewise((density, set.as_relational(symbol)), (0, True))
    pdf = Lambda(symbol, pdf)
    # have a default of False while `rv` should have a default of True
    kwargs['check'] = kwargs.pop('check', False)
    return rv(symbol.name, DiscreteDistributionHandmade, pdf, set, **kwargs)


#-------------------------------------------------------------------------------
# Flory-Schulz distribution ------------------------------------------------------------

class FlorySchulzDistribution(SingleDiscreteDistribution):
    _argnames = ('a',)
    set = S.Naturals

    @staticmethod
    def check(a):
        _value_check((0 < a, a < 1), "a must be between 0 and 1")

    def pdf(self, k):
        a = self.a
        return (a**2 * k * (1 - a)**(k - 1))

    def _characteristic_function(self, t):
        a = self.a
        return a**2*exp(I*t)/((1 + (a - 1)*exp(I*t))**2)

    def _moment_generating_function(self, t):
        a = self.a
        return a**2*exp(t)/((1 + (a - 1)*exp(t))**2)


def FlorySchulz(name, a):
    r"""
    Create a discrete random variable with a FlorySchulz distribution.

    The density of the FlorySchulz distribution is given by

    .. math::
        f(k) := (a^2) k (1 - a)^{k-1}

    Parameters
    ==========

    a : A real number between 0 and 1

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import density, E, variance, FlorySchulz
    >>> from sympy import Symbol, S

    >>> a = S.One / 5
    >>> z = Symbol("z")

    >>> X = FlorySchulz("x", a)

    >>> density(X)(z)
    (4/5)**(z - 1)*z/25

    >>> E(X)
    9

    >>> variance(X)
    40

    References
    ==========

    https://en.wikipedia.org/wiki/Flory%E2%80%93Schulz_distribution
    """
    return rv(name, FlorySchulzDistribution, a)


#-------------------------------------------------------------------------------
# Geometric distribution ------------------------------------------------------------

class GeometricDistribution(SingleDiscreteDistribution):
    _argnames = ('p',)
    set = S.Naturals

    @staticmethod
    def check(p):
        _value_check((0 < p, p <= 1), "p must be between 0 and 1")

    def pdf(self, k):
        return (1 - self.p)**(k - 1) * self.p

    def _characteristic_function(self, t):
        p = self.p
        return p * exp(I*t) / (1 - (1 - p)*exp(I*t))

    def _moment_generating_function(self, t):
        p = self.p
        return p * exp(t) / (1 - (1 - p) * exp(t))


def Geometric(name, p):
    r"""
    Create a discrete random variable with a Geometric distribution.

    Explanation
    ===========

    The density of the Geometric distribution is given by

    .. math::
        f(k) := p (1 - p)^{k - 1}

    Parameters
    ==========

    p : A probability between 0 and 1

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Geometric, density, E, variance
    >>> from sympy import Symbol, S

    >>> p = S.One / 5
    >>> z = Symbol("z")

    >>> X = Geometric("x", p)

    >>> density(X)(z)
    (4/5)**(z - 1)/5

    >>> E(X)
    5

    >>> variance(X)
    20

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Geometric_distribution
    .. [2] https://mathworld.wolfram.com/GeometricDistribution.html

    """
    return rv(name, GeometricDistribution, p)


#-------------------------------------------------------------------------------
# Hermite distribution ---------------------------------------------------------


class HermiteDistribution(SingleDiscreteDistribution):
    _argnames = ('a1', 'a2')
    set = S.Naturals0

    @staticmethod
    def check(a1, a2):
        _value_check(a1.is_nonnegative, 'Parameter a1 must be >= 0.')
        _value_check(a2.is_nonnegative, 'Parameter a2 must be >= 0.')

    def pdf(self, k):
        a1, a2 = self.a1, self.a2
        term1 = exp(-(a1 + a2))
        j = Dummy("j", integer=True)
        num = a1**(k - 2*j) * a2**j
        den = factorial(k - 2*j) * factorial(j)
        return term1 * Sum(num/den, (j, 0, k//2)).doit()

    def _moment_generating_function(self, t):
        a1, a2 = self.a1, self.a2
        term1 = a1 * (exp(t) - 1)
        term2 = a2 * (exp(2*t) - 1)
        return exp(term1 + term2)

    def _characteristic_function(self, t):
        a1, a2 = self.a1, self.a2
        term1 = a1 * (exp(I*t) - 1)
        term2 = a2 * (exp(2*I*t) - 1)
        return exp(term1 + term2)

def Hermite(name, a1, a2):
    r"""
    Create a discrete random variable with a Hermite distribution.

    Explanation
    ===========

    The density of the Hermite distribution is given by

    .. math::
        f(x):= e^{-a_1 -a_2}\sum_{j=0}^{\left \lfloor x/2 \right \rfloor}
                    \frac{a_{1}^{x-2j}a_{2}^{j}}{(x-2j)!j!}

    Parameters
    ==========

    a1 : A Positive number greater than equal to 0.
    a2 : A Positive number greater than equal to 0.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Hermite, density, E, variance
    >>> from sympy import Symbol

    >>> a1 = Symbol("a1", positive=True)
    >>> a2 = Symbol("a2", positive=True)
    >>> x = Symbol("x")

    >>> H = Hermite("H", a1=5, a2=4)

    >>> density(H)(2)
    33*exp(-9)/2

    >>> E(H)
    13

    >>> variance(H)
    21

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Hermite_distribution

    """

    return rv(name, HermiteDistribution, a1, a2)


#-------------------------------------------------------------------------------
# Logarithmic distribution ------------------------------------------------------------

class LogarithmicDistribution(SingleDiscreteDistribution):
    _argnames = ('p',)

    set = S.Naturals

    @staticmethod
    def check(p):
        _value_check((p > 0, p < 1), "p should be between 0 and 1")

    def pdf(self, k):
        p = self.p
        return (-1) * p**k / (k * log(1 - p))

    def _characteristic_function(self, t):
        p = self.p
        return log(1 - p * exp(I*t)) / log(1 - p)

    def _moment_generating_function(self, t):
        p = self.p
        return log(1 - p * exp(t)) / log(1 - p)


def Logarithmic(name, p):
    r"""
    Create a discrete random variable with a Logarithmic distribution.

    Explanation
    ===========

    The density of the Logarithmic distribution is given by

    .. math::
        f(k) := \frac{-p^k}{k \ln{(1 - p)}}

    Parameters
    ==========

    p : A value between 0 and 1

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Logarithmic, density, E, variance
    >>> from sympy import Symbol, S

    >>> p = S.One / 5
    >>> z = Symbol("z")

    >>> X = Logarithmic("x", p)

    >>> density(X)(z)
    -1/(5**z*z*log(4/5))

    >>> E(X)
    -1/(-4*log(5) + 8*log(2))

    >>> variance(X)
    -1/((-4*log(5) + 8*log(2))*(-2*log(5) + 4*log(2))) + 1/(-64*log(2)*log(5) + 64*log(2)**2 + 16*log(5)**2) - 10/(-32*log(5) + 64*log(2))

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Logarithmic_distribution
    .. [2] https://mathworld.wolfram.com/LogarithmicDistribution.html

    """
    return rv(name, LogarithmicDistribution, p)


#-------------------------------------------------------------------------------
# Negative binomial distribution ------------------------------------------------------------

class NegativeBinomialDistribution(SingleDiscreteDistribution):
    _argnames = ('r', 'p')
    set = S.Naturals0

    @staticmethod
    def check(r, p):
        _value_check(r > 0, 'r should be positive')
        _value_check((p > 0, p < 1), 'p should be between 0 and 1')

    def pdf(self, k):
        r = self.r
        p = self.p

        return binomial(k + r - 1, k) * (1 - p)**k * p**r

    def _characteristic_function(self, t):
        r = self.r
        p = self.p

        return (p / (1 - (1 - p) * exp(I*t)))**r

    def _moment_generating_function(self, t):
        r = self.r
        p = self.p

        return (p / (1 - (1 - p) * exp(t)))**r

def NegativeBinomial(name, r, p):
    r"""
    Create a discrete random variable with a Negative Binomial distribution.

    Explanation
    ===========

    The density of the Negative Binomial distribution is given by

    .. math::
        f(k) := \binom{k + r - 1}{k} (1 - p)^k p^r

    Parameters
    ==========

    r : A positive value
        Number of successes until the experiment is stopped.
    p : A value between 0 and 1
        Probability of success.

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import NegativeBinomial, density, E, variance
    >>> from sympy import Symbol, S

    >>> r = 5
    >>> p = S.One / 3
    >>> z = Symbol("z")

    >>> X = NegativeBinomial("x", r, p)

    >>> density(X)(z)
    (2/3)**z*binomial(z + 4, z)/243

    >>> E(X)
    10

    >>> variance(X)
    30

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Negative_binomial_distribution
    .. [2] https://mathworld.wolfram.com/NegativeBinomialDistribution.html

    """
    return rv(name, NegativeBinomialDistribution, r, p)


#-------------------------------------------------------------------------------
# Poisson distribution ------------------------------------------------------------

class PoissonDistribution(SingleDiscreteDistribution):
    _argnames = ('lamda',)

    set = S.Naturals0

    @staticmethod
    def check(lamda):
        _value_check(lamda > 0, "Lambda must be positive")

    def pdf(self, k):
        return self.lamda**k / factorial(k) * exp(-self.lamda)

    def _characteristic_function(self, t):
        return exp(self.lamda * (exp(I*t) - 1))

    def _moment_generating_function(self, t):
        return exp(self.lamda * (exp(t) - 1))

    def expectation(self, expr, var, evaluate=True, **kwargs):
        if evaluate:
            if expr == var:
                return self.lamda
            if (
                isinstance(expr, FallingFactorial)
                and expr.args[1].is_integer
                and expr.args[1].is_positive
                and expr.args[0] == var
            ):
                return self.lamda ** expr.args[1]
        return super().expectation(expr, var, evaluate, **kwargs)

def Poisson(name, lamda):
    r"""
    Create a discrete random variable with a Poisson distribution.

    Explanation
    ===========

    The density of the Poisson distribution is given by

    .. math::
        f(k) := \frac{\lambda^{k} e^{- \lambda}}{k!}

    Parameters
    ==========

    lamda : Positive number, a rate

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Poisson, density, E, variance
    >>> from sympy import Symbol, simplify

    >>> rate = Symbol("lambda", positive=True)
    >>> z = Symbol("z")

    >>> X = Poisson("x", rate)

    >>> density(X)(z)
    lambda**z*exp(-lambda)/factorial(z)

    >>> E(X)
    lambda

    >>> simplify(variance(X))
    lambda

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Poisson_distribution
    .. [2] https://mathworld.wolfram.com/PoissonDistribution.html

    """
    return rv(name, PoissonDistribution, lamda)


# -----------------------------------------------------------------------------
# Skellam distribution --------------------------------------------------------


class SkellamDistribution(SingleDiscreteDistribution):
    _argnames = ('mu1', 'mu2')
    set = S.Integers

    @staticmethod
    def check(mu1, mu2):
        _value_check(mu1 >= 0, 'Parameter mu1 must be >= 0')
        _value_check(mu2 >= 0, 'Parameter mu2 must be >= 0')

    def pdf(self, k):
        (mu1, mu2) = (self.mu1, self.mu2)
        term1 = exp(-(mu1 + mu2)) * (mu1 / mu2) ** (k / 2)
        term2 = besseli(k, 2 * sqrt(mu1 * mu2))
        return term1 * term2

    def _cdf(self, x):
        raise NotImplementedError(
            "Skellam doesn't have closed form for the CDF.")

    def _characteristic_function(self, t):
        (mu1, mu2) = (self.mu1, self.mu2)
        return exp(-(mu1 + mu2) + mu1 * exp(I * t) + mu2 * exp(-I * t))

    def _moment_generating_function(self, t):
        (mu1, mu2) = (self.mu1, self.mu2)
        return exp(-(mu1 + mu2) + mu1 * exp(t) + mu2 * exp(-t))


def Skellam(name, mu1, mu2):
    r"""
    Create a discrete random variable with a Skellam distribution.

    Explanation
    ===========

    The Skellam is the distribution of the difference N1 - N2
    of two statistically independent random variables N1 and N2
    each Poisson-distributed with respective expected values mu1 and mu2.

    The density of the Skellam distribution is given by

    .. math::
        f(k) := e^{-(\mu_1+\mu_2)}(\frac{\mu_1}{\mu_2})^{k/2}I_k(2\sqrt{\mu_1\mu_2})

    Parameters
    ==========

    mu1 : A non-negative value
    mu2 : A non-negative value

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Skellam, density, E, variance
    >>> from sympy import Symbol, pprint

    >>> z = Symbol("z", integer=True)
    >>> mu1 = Symbol("mu1", positive=True)
    >>> mu2 = Symbol("mu2", positive=True)
    >>> X = Skellam("x", mu1, mu2)

    >>> pprint(density(X)(z), use_unicode=False)
         z
         -
         2
    /mu1\   -mu1 - mu2        /       _____   _____\
    |---| *e          *besseli\z, 2*\/ mu1 *\/ mu2 /
    \mu2/
    >>> E(X)
    mu1 - mu2
    >>> variance(X).expand()
    mu1 + mu2

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Skellam_distribution

    """
    return rv(name, SkellamDistribution, mu1, mu2)


#-------------------------------------------------------------------------------
# Yule-Simon distribution ------------------------------------------------------------

class YuleSimonDistribution(SingleDiscreteDistribution):
    _argnames = ('rho',)
    set = S.Naturals

    @staticmethod
    def check(rho):
        _value_check(rho > 0, 'rho should be positive')

    def pdf(self, k):
        rho = self.rho
        return rho * beta(k, rho + 1)

    def _cdf(self, x):
        return Piecewise((1 - floor(x) * beta(floor(x), self.rho + 1), x >= 1), (0, True))

    def _characteristic_function(self, t):
        rho = self.rho
        return rho * hyper((1, 1), (rho + 2,), exp(I*t)) * exp(I*t) / (rho + 1)

    def _moment_generating_function(self, t):
        rho = self.rho
        return rho * hyper((1, 1), (rho + 2,), exp(t)) * exp(t) / (rho + 1)


def YuleSimon(name, rho):
    r"""
    Create a discrete random variable with a Yule-Simon distribution.

    Explanation
    ===========

    The density of the Yule-Simon distribution is given by

    .. math::
        f(k) := \rho B(k, \rho + 1)

    Parameters
    ==========

    rho : A positive value

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import YuleSimon, density, E, variance
    >>> from sympy import Symbol, simplify

    >>> p = 5
    >>> z = Symbol("z")

    >>> X = YuleSimon("x", p)

    >>> density(X)(z)
    5*beta(z, 6)

    >>> simplify(E(X))
    5/4

    >>> simplify(variance(X))
    25/48

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Yule%E2%80%93Simon_distribution

    """
    return rv(name, YuleSimonDistribution, rho)


#-------------------------------------------------------------------------------
# Zeta distribution ------------------------------------------------------------

class ZetaDistribution(SingleDiscreteDistribution):
    _argnames = ('s',)
    set = S.Naturals

    @staticmethod
    def check(s):
        _value_check(s > 1, 's should be greater than 1')

    def pdf(self, k):
        s = self.s
        return 1 / (k**s * zeta(s))

    def _characteristic_function(self, t):
        return polylog(self.s, exp(I*t)) / zeta(self.s)

    def _moment_generating_function(self, t):
        return polylog(self.s, exp(t)) / zeta(self.s)


def Zeta(name, s):
    r"""
    Create a discrete random variable with a Zeta distribution.

    Explanation
    ===========

    The density of the Zeta distribution is given by

    .. math::
        f(k) := \frac{1}{k^s \zeta{(s)}}

    Parameters
    ==========

    s : A value greater than 1

    Returns
    =======

    RandomSymbol

    Examples
    ========

    >>> from sympy.stats import Zeta, density, E, variance
    >>> from sympy import Symbol

    >>> s = 5
    >>> z = Symbol("z")

    >>> X = Zeta("x", s)

    >>> density(X)(z)
    1/(z**5*zeta(5))

    >>> E(X)
    pi**4/(90*zeta(5))

    >>> variance(X)
    -pi**8/(8100*zeta(5)**2) + zeta(3)/zeta(5)

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Zeta_distribution

    """
    return rv(name, ZetaDistribution, s)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\bs4\builder\__init__.py ===
from __future__ import annotations

# Use of this source code is governed by the MIT license.
__license__ = "MIT"

from collections import defaultdict
import re
from types import ModuleType
from typing import (
    Any,
    cast,
    Dict,
    Iterable,
    List,
    Optional,
    Pattern,
    Set,
    Tuple,
    Type,
    TYPE_CHECKING,
)
import warnings
import sys
from bs4.element import (
    AttributeDict,
    AttributeValueList,
    CharsetMetaAttributeValue,
    ContentMetaAttributeValue,
    RubyParenthesisString,
    RubyTextString,
    Stylesheet,
    Script,
    TemplateString,
    nonwhitespace_re,
)

# Exceptions were moved to their own module in 4.13. Import here for
# backwards compatibility.
from bs4.exceptions import ParserRejectedMarkup

from bs4._typing import (
    _AttributeValues,
    _RawAttributeValue,
)

from bs4._warnings import XMLParsedAsHTMLWarning

if TYPE_CHECKING:
    from bs4 import BeautifulSoup
    from bs4.element import (
        NavigableString,
        Tag,
    )
    from bs4._typing import (
        _AttributeValue,
        _Encoding,
        _Encodings,
        _RawOrProcessedAttributeValues,
        _RawMarkup,
    )

__all__ = [
    "HTMLTreeBuilder",
    "SAXTreeBuilder",
    "TreeBuilder",
    "TreeBuilderRegistry",
]

# Some useful features for a TreeBuilder to have.
FAST = "fast"
PERMISSIVE = "permissive"
STRICT = "strict"
XML = "xml"
HTML = "html"
HTML_5 = "html5"

__all__ = [
    "TreeBuilderRegistry",
    "TreeBuilder",
    "HTMLTreeBuilder",
    "DetectsXMLParsedAsHTML",

    "ParserRejectedMarkup", # backwards compatibility only as of 4.13.0
]

class TreeBuilderRegistry(object):
    """A way of looking up TreeBuilder subclasses by their name or by desired
    features.
    """

    builders_for_feature: Dict[str, List[Type[TreeBuilder]]]
    builders: List[Type[TreeBuilder]]

    def __init__(self) -> None:
        self.builders_for_feature = defaultdict(list)
        self.builders = []

    def register(self, treebuilder_class: type[TreeBuilder]) -> None:
        """Register a treebuilder based on its advertised features.

        :param treebuilder_class: A subclass of `TreeBuilder`. its
           `TreeBuilder.features` attribute should list its features.
        """
        for feature in treebuilder_class.features:
            self.builders_for_feature[feature].insert(0, treebuilder_class)
        self.builders.insert(0, treebuilder_class)

    def lookup(self, *features: str) -> Optional[Type[TreeBuilder]]:
        """Look up a TreeBuilder subclass with the desired features.

        :param features: A list of features to look for. If none are
            provided, the most recently registered TreeBuilder subclass
            will be used.
        :return: A TreeBuilder subclass, or None if there's no
            registered subclass with all the requested features.
        """
        if len(self.builders) == 0:
            # There are no builders at all.
            return None

        if len(features) == 0:
            # They didn't ask for any features. Give them the most
            # recently registered builder.
            return self.builders[0]

        # Go down the list of features in order, and eliminate any builders
        # that don't match every feature.
        feature_list = list(features)
        feature_list.reverse()
        candidates = None
        candidate_set = None
        while len(feature_list) > 0:
            feature = feature_list.pop()
            we_have_the_feature = self.builders_for_feature.get(feature, [])
            if len(we_have_the_feature) > 0:
                if candidates is None:
                    candidates = we_have_the_feature
                    candidate_set = set(candidates)
                else:
                    # Eliminate any candidates that don't have this feature.
                    candidate_set = candidate_set.intersection(set(we_have_the_feature))

        # The only valid candidates are the ones in candidate_set.
        # Go through the original list of candidates and pick the first one
        # that's in candidate_set.
        if candidate_set is None or candidates is None:
            return None
        for candidate in candidates:
            if candidate in candidate_set:
                return candidate
        return None


#: The `BeautifulSoup` constructor will take a list of features
#: and use it to look up `TreeBuilder` classes in this registry.
builder_registry: TreeBuilderRegistry = TreeBuilderRegistry()


class TreeBuilder(object):
    """Turn a textual document into a Beautiful Soup object tree.

    This is an abstract superclass which smooths out the behavior of
    different parser libraries into a single, unified interface.

    :param multi_valued_attributes: If this is set to None, the
     TreeBuilder will not turn any values for attributes like
     'class' into lists. Setting this to a dictionary will
     customize this behavior; look at :py:attr:`bs4.builder.HTMLTreeBuilder.DEFAULT_CDATA_LIST_ATTRIBUTES`
     for an example.

     Internally, these are called "CDATA list attributes", but that
     probably doesn't make sense to an end-user, so the argument name
     is ``multi_valued_attributes``.

    :param preserve_whitespace_tags: A set of tags to treat
     the way <pre> tags are treated in HTML. Tags in this set
     are immune from pretty-printing; their contents will always be
     output as-is.

    :param string_containers: A dictionary mapping tag names to
     the classes that should be instantiated to contain the textual
     contents of those tags. The default is to use NavigableString
     for every tag, no matter what the name. You can override the
     default by changing :py:attr:`DEFAULT_STRING_CONTAINERS`.

    :param store_line_numbers: If the parser keeps track of the line
     numbers and positions of the original markup, that information
     will, by default, be stored in each corresponding
     :py:class:`bs4.element.Tag` object. You can turn this off by
     passing store_line_numbers=False; then Tag.sourcepos and
     Tag.sourceline will always be None. If the parser you're using
     doesn't keep track of this information, then store_line_numbers
     is irrelevant.

    :param attribute_dict_class: The value of a multi-valued attribute
      (such as HTML's 'class') willl be stored in an instance of this
      class.  The default is Beautiful Soup's built-in
      `AttributeValueList`, which is a normal Python list, and you
      will probably never need to change it.
    """

    USE_DEFAULT: Any = object()  #: :meta private:

    def __init__(
        self,
        multi_valued_attributes: Dict[str, Set[str]] = USE_DEFAULT,
        preserve_whitespace_tags: Set[str] = USE_DEFAULT,
        store_line_numbers: bool = USE_DEFAULT,
        string_containers: Dict[str, Type[NavigableString]] = USE_DEFAULT,
        empty_element_tags: Set[str] = USE_DEFAULT,
        attribute_dict_class: Type[AttributeDict] = AttributeDict,
        attribute_value_list_class: Type[AttributeValueList] = AttributeValueList,
    ):
        self.soup = None
        if multi_valued_attributes is self.USE_DEFAULT:
            multi_valued_attributes = self.DEFAULT_CDATA_LIST_ATTRIBUTES
        self.cdata_list_attributes = multi_valued_attributes
        if preserve_whitespace_tags is self.USE_DEFAULT:
            preserve_whitespace_tags = self.DEFAULT_PRESERVE_WHITESPACE_TAGS
        self.preserve_whitespace_tags = preserve_whitespace_tags
        if empty_element_tags is self.USE_DEFAULT:
            self.empty_element_tags = self.DEFAULT_EMPTY_ELEMENT_TAGS
        else:
            self.empty_element_tags = empty_element_tags
        # TODO: store_line_numbers is probably irrelevant now that
        # the behavior of sourceline and sourcepos has been made consistent
        # everywhere.
        if store_line_numbers == self.USE_DEFAULT:
            store_line_numbers = self.TRACKS_LINE_NUMBERS
        self.store_line_numbers = store_line_numbers
        if string_containers == self.USE_DEFAULT:
            string_containers = self.DEFAULT_STRING_CONTAINERS
        self.string_containers = string_containers
        self.attribute_dict_class = attribute_dict_class
        self.attribute_value_list_class = attribute_value_list_class

    NAME: str = "[Unknown tree builder]"
    ALTERNATE_NAMES: Iterable[str] = []
    features: Iterable[str] = []

    is_xml: bool = False
    picklable: bool = False

    soup: Optional[BeautifulSoup]  #: :meta private:

    #: A tag will be considered an empty-element
    #: tag when and only when it has no contents.
    empty_element_tags: Optional[Set[str]] = None  #: :meta private:
    cdata_list_attributes: Dict[str, Set[str]]  #: :meta private:
    preserve_whitespace_tags: Set[str]  #: :meta private:
    string_containers: Dict[str, Type[NavigableString]]  #: :meta private:
    tracks_line_numbers: bool  #: :meta private:

    #: A value for these tag/attribute combinations is a space- or
    #: comma-separated list of CDATA, rather than a single CDATA.
    DEFAULT_CDATA_LIST_ATTRIBUTES: Dict[str, Set[str]] = defaultdict(set)

    #: Whitespace should be preserved inside these tags.
    DEFAULT_PRESERVE_WHITESPACE_TAGS: Set[str] = set()

    #: The textual contents of tags with these names should be
    #: instantiated with some class other than `bs4.element.NavigableString`.
    DEFAULT_STRING_CONTAINERS: Dict[str, Type[bs4.element.NavigableString]] = {}

    #: By default, tags are treated as empty-element tags if they have
    #: no contents--that is, using XML rules. HTMLTreeBuilder
    #: defines a different set of DEFAULT_EMPTY_ELEMENT_TAGS based on the
    #: HTML 4 and HTML5 standards.
    DEFAULT_EMPTY_ELEMENT_TAGS: Optional[Set[str]] = None

    #: Most parsers don't keep track of line numbers.
    TRACKS_LINE_NUMBERS: bool = False

    def initialize_soup(self, soup: BeautifulSoup) -> None:
        """The BeautifulSoup object has been initialized and is now
        being associated with the TreeBuilder.

        :param soup: A BeautifulSoup object.
        """
        self.soup = soup

    def reset(self) -> None:
        """Do any work necessary to reset the underlying parser
        for a new document.

        By default, this does nothing.
        """
        pass

    def can_be_empty_element(self, tag_name: str) -> bool:
        """Might a tag with this name be an empty-element tag?

        The final markup may or may not actually present this tag as
        self-closing.

        For instance: an HTMLBuilder does not consider a <p> tag to be
        an empty-element tag (it's not in
        HTMLBuilder.empty_element_tags). This means an empty <p> tag
        will be presented as "<p></p>", not "<p/>" or "<p>".

        The default implementation has no opinion about which tags are
        empty-element tags, so a tag will be presented as an
        empty-element tag if and only if it has no children.
        "<foo></foo>" will become "<foo/>", and "<foo>bar</foo>" will
        be left alone.

        :param tag_name: The name of a markup tag.
        """
        if self.empty_element_tags is None:
            return True
        return tag_name in self.empty_element_tags

    def feed(self, markup: _RawMarkup) -> None:
        """Run incoming markup through some parsing process."""
        raise NotImplementedError()

    def prepare_markup(
        self,
        markup: _RawMarkup,
        user_specified_encoding: Optional[_Encoding] = None,
        document_declared_encoding: Optional[_Encoding] = None,
        exclude_encodings: Optional[_Encodings] = None,
    ) -> Iterable[Tuple[_RawMarkup, Optional[_Encoding], Optional[_Encoding], bool]]:
        """Run any preliminary steps necessary to make incoming markup
        acceptable to the parser.

        :param markup: The markup that's about to be parsed.
        :param user_specified_encoding: The user asked to try this encoding
           to convert the markup into a Unicode string.
        :param document_declared_encoding: The markup itself claims to be
            in this encoding. NOTE: This argument is not used by the
            calling code and can probably be removed.
        :param exclude_encodings: The user asked *not* to try any of
            these encodings.

        :yield: A series of 4-tuples: (markup, encoding, declared encoding,
            has undergone character replacement)

            Each 4-tuple represents a strategy that the parser can try
            to convert the document to Unicode and parse it. Each
            strategy will be tried in turn.

         By default, the only strategy is to parse the markup
         as-is. See `LXMLTreeBuilderForXML` and
         `HTMLParserTreeBuilder` for implementations that take into
         account the quirks of particular parsers.

        :meta private:

        """
        yield markup, None, None, False

    def test_fragment_to_document(self, fragment: str) -> str:
        """Wrap an HTML fragment to make it look like a document.

        Different parsers do this differently. For instance, lxml
        introduces an empty <head> tag, and html5lib
        doesn't. Abstracting this away lets us write simple tests
        which run HTML fragments through the parser and compare the
        results against other HTML fragments.

        This method should not be used outside of unit tests.

        :param fragment: A fragment of HTML.
        :return: A full HTML document.
        :meta private:
        """
        return fragment

    def set_up_substitutions(self, tag: Tag) -> bool:
        """Set up any substitutions that will need to be performed on
        a `Tag` when it's output as a string.

        By default, this does nothing. See `HTMLTreeBuilder` for a
        case where this is used.

        :return: Whether or not a substitution was performed.
        :meta private:
        """
        return False

    def _replace_cdata_list_attribute_values(
        self, tag_name: str, attrs: _RawOrProcessedAttributeValues
    ) -> _AttributeValues:
        """When an attribute value is associated with a tag that can
        have multiple values for that attribute, convert the string
        value to a list of strings.

        Basically, replaces class="foo bar" with class=["foo", "bar"]

        NOTE: This method modifies its input in place.

        :param tag_name: The name of a tag.
        :param attrs: A dictionary containing the tag's attributes.
           Any appropriate attribute values will be modified in place.
        :return: The modified dictionary that was originally passed in.
        """

        # First, cast the attrs dict to _AttributeValues. This might
        # not be accurate yet, but it will be by the time this method
        # returns.
        modified_attrs = cast(_AttributeValues, attrs)
        if not modified_attrs or not self.cdata_list_attributes:
            # Nothing to do.
            return modified_attrs

        # There is at least a possibility that we need to modify one of
        # the attribute values.
        universal: Set[str] = self.cdata_list_attributes.get("*", set())
        tag_specific = self.cdata_list_attributes.get(tag_name.lower(), None)
        for attr in list(modified_attrs.keys()):
            modified_value: _AttributeValue
            if attr in universal or (tag_specific and attr in tag_specific):
                # We have a "class"-type attribute whose string
                # value is a whitespace-separated list of
                # values. Split it into a list.
                original_value: _AttributeValue = modified_attrs[attr]
                if isinstance(original_value, _RawAttributeValue):
                    # This is a _RawAttributeValue (a string) that
                    # needs to be split and converted to a
                    # AttributeValueList so it can be an
                    # _AttributeValue.
                    modified_value = self.attribute_value_list_class(
                        nonwhitespace_re.findall(original_value)
                    )
                else:
                    # html5lib calls setAttributes twice for the
                    # same tag when rearranging the parse tree. On
                    # the second call the attribute value here is
                    # already a list. This can also happen when a
                    # Tag object is cloned. If this happens, leave
                    # the value alone rather than trying to split
                    # it again.
                    modified_value = original_value
                modified_attrs[attr] = modified_value
        return modified_attrs


class SAXTreeBuilder(TreeBuilder):
    """A Beautiful Soup treebuilder that listens for SAX events.

    This is not currently used for anything, and it will be removed
    soon. It was a good idea, but it wasn't properly integrated into the
    rest of Beautiful Soup, so there have been long stretches where it
    hasn't worked properly.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "The SAXTreeBuilder class was deprecated in 4.13.0 and will be removed soon thereafter. It is completely untested and probably doesn't work; do not use it.",
            DeprecationWarning,
            stacklevel=2,
        )
        super(SAXTreeBuilder, self).__init__(*args, **kwargs)

    def feed(self, markup: _RawMarkup) -> None:
        raise NotImplementedError()

    def close(self) -> None:
        pass

    def startElement(self, name: str, attrs: Dict[str, str]) -> None:
        attrs = AttributeDict((key[1], value) for key, value in list(attrs.items()))
        # print("Start %s, %r" % (name, attrs))
        assert self.soup is not None
        self.soup.handle_starttag(name, None, None, attrs)

    def endElement(self, name: str) -> None:
        # print("End %s" % name)
        assert self.soup is not None
        self.soup.handle_endtag(name)

    def startElementNS(
        self, nsTuple: Tuple[str, str], nodeName: str, attrs: Dict[str, str]
    ) -> None:
        # Throw away (ns, nodeName) for now.
        self.startElement(nodeName, attrs)

    def endElementNS(self, nsTuple: Tuple[str, str], nodeName: str) -> None:
        # Throw away (ns, nodeName) for now.
        self.endElement(nodeName)
        # handler.endElementNS((ns, node.nodeName), node.nodeName)

    def startPrefixMapping(self, prefix: str, nodeValue: str) -> None:
        # Ignore the prefix for now.
        pass

    def endPrefixMapping(self, prefix: str) -> None:
        # Ignore the prefix for now.
        # handler.endPrefixMapping(prefix)
        pass

    def characters(self, content: str) -> None:
        assert self.soup is not None
        self.soup.handle_data(content)

    def startDocument(self) -> None:
        pass

    def endDocument(self) -> None:
        pass


class HTMLTreeBuilder(TreeBuilder):
    """This TreeBuilder knows facts about HTML, such as which tags are treated
    specially by the HTML standard.
    """

    #: Some HTML tags are defined as having no contents. Beautiful Soup
    #: treats these specially.
    DEFAULT_EMPTY_ELEMENT_TAGS: Set[str] = set(
        [
            # These are from HTML5.
            "area",
            "base",
            "br",
            "col",
            "embed",
            "hr",
            "img",
            "input",
            "keygen",
            "link",
            "menuitem",
            "meta",
            "param",
            "source",
            "track",
            "wbr",
            # These are from earlier versions of HTML and are removed in HTML5.
            "basefont",
            "bgsound",
            "command",
            "frame",
            "image",
            "isindex",
            "nextid",
            "spacer",
        ]
    )

    #: The HTML standard defines these tags as block-level elements. Beautiful
    #: Soup does not treat these elements differently from other elements,
    #: but it may do so eventually, and this information is available if
    #: you need to use it.
    DEFAULT_BLOCK_ELEMENTS: Set[str] = set(
        [
            "address",
            "article",
            "aside",
            "blockquote",
            "canvas",
            "dd",
            "div",
            "dl",
            "dt",
            "fieldset",
            "figcaption",
            "figure",
            "footer",
            "form",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "header",
            "hr",
            "li",
            "main",
            "nav",
            "noscript",
            "ol",
            "output",
            "p",
            "pre",
            "section",
            "table",
            "tfoot",
            "ul",
            "video",
        ]
    )

    #: These HTML tags need special treatment so they can be
    #: represented by a string class other than `bs4.element.NavigableString`.
    #:
    #: For some of these tags, it's because the HTML standard defines
    #: an unusual content model for them. I made this list by going
    #: through the HTML spec
    #: (https://html.spec.whatwg.org/#metadata-content) and looking for
    #: "metadata content" elements that can contain strings.
    #:
    #: The Ruby tags (<rt> and <rp>) are here despite being normal
    #: "phrasing content" tags, because the content they contain is
    #: qualitatively different from other text in the document, and it
    #: can be useful to be able to distinguish it.
    #:
    #: TODO: Arguably <noscript> could go here but it seems
    #: qualitatively different from the other tags.
    DEFAULT_STRING_CONTAINERS: Dict[str, Type[bs4.element.NavigableString]] = {
        "rt": RubyTextString,
        "rp": RubyParenthesisString,
        "style": Stylesheet,
        "script": Script,
        "template": TemplateString,
    }

    #: The HTML standard defines these attributes as containing a
    #: space-separated list of values, not a single value. That is,
    #: class="foo bar" means that the 'class' attribute has two values,
    #: 'foo' and 'bar', not the single value 'foo bar'.  When we
    #: encounter one of these attributes, we will parse its value into
    #: a list of values if possible. Upon output, the list will be
    #: converted back into a string.
    DEFAULT_CDATA_LIST_ATTRIBUTES: Dict[str, Set[str]] = {
        "*": {"class", "accesskey", "dropzone"},
        "a": {"rel", "rev"},
        "link": {"rel", "rev"},
        "td": {"headers"},
        "th": {"headers"},
        "form": {"accept-charset"},
        "object": {"archive"},
        # These are HTML5 specific, as are *.accesskey and *.dropzone above.
        "area": {"rel"},
        "icon": {"sizes"},
        "iframe": {"sandbox"},
        "output": {"for"},
    }

    #: By default, whitespace inside these HTML tags will be
    #: preserved rather than being collapsed.
    DEFAULT_PRESERVE_WHITESPACE_TAGS: set[str] = set(["pre", "textarea"])

    def set_up_substitutions(self, tag: Tag) -> bool:
        """Replace the declared encoding in a <meta> tag with a placeholder,
        to be substituted when the tag is output to a string.

        An HTML document may come in to Beautiful Soup as one
        encoding, but exit in a different encoding, and the <meta> tag
        needs to be changed to reflect this.

        :return: Whether or not a substitution was performed.

        :meta private:
        """
        # We are only interested in <meta> tags
        if tag.name != "meta":
            return False

        # TODO: This cast will fail in the (very unlikely) scenario
        # that the programmer who instantiates the TreeBuilder
        # specifies meta['content'] or meta['charset'] as
        # cdata_list_attributes.
        content: Optional[str] = cast(Optional[str], tag.get("content"))
        charset: Optional[str] = cast(Optional[str], tag.get("charset"))

        # But we can accommodate meta['http-equiv'] being made a
        # cdata_list_attribute (again, very unlikely) without much
        # trouble.
        http_equiv: List[str] = tag.get_attribute_list("http-equiv")

        # We are interested in <meta> tags that say what encoding the
        # document was originally in. This means HTML 5-style <meta>
        # tags that provide the "charset" attribute. It also means
        # HTML 4-style <meta> tags that provide the "content"
        # attribute and have "http-equiv" set to "content-type".
        #
        # In both cases we will replace the value of the appropriate
        # attribute with a standin object that can take on any
        # encoding.
        substituted = False
        if charset is not None:
            # HTML 5 style:
            # <meta charset="utf8">
            tag["charset"] = CharsetMetaAttributeValue(charset)
            substituted = True

        elif content is not None and any(
            x.lower() == "content-type" for x in http_equiv
        ):
            # HTML 4 style:
            # <meta http-equiv="content-type" content="text/html; charset=utf8">
            tag["content"] = ContentMetaAttributeValue(content)
            substituted = True

        return substituted


class DetectsXMLParsedAsHTML(object):
    """A mixin class for any class (a TreeBuilder, or some class used by a
    TreeBuilder) that's in a position to detect whether an XML
    document is being incorrectly parsed as HTML, and issue an
    appropriate warning.

    This requires being able to observe an incoming processing
    instruction that might be an XML declaration, and also able to
    observe tags as they're opened. If you can't do that for a given
    `TreeBuilder`, there's a less reliable implementation based on
    examining the raw markup.
    """

    #: Regular expression for seeing if string markup has an <html> tag.
    LOOKS_LIKE_HTML: Pattern[str] = re.compile("<[^ +]html", re.I)

    #: Regular expression for seeing if byte markup has an <html> tag.
    LOOKS_LIKE_HTML_B: Pattern[bytes] = re.compile(b"<[^ +]html", re.I)

    #: The start of an XML document string.
    XML_PREFIX: str = "<?xml"

    #: The start of an XML document bytestring.
    XML_PREFIX_B: bytes = b"<?xml"

    # This is typed as str, not `ProcessingInstruction`, because this
    # check may be run before any Beautiful Soup objects are created.
    _first_processing_instruction: Optional[str]  #: :meta private:
    _root_tag_name: Optional[str]  #: :meta private:

    @classmethod
    def warn_if_markup_looks_like_xml(
        cls, markup: Optional[_RawMarkup], stacklevel: int = 3
    ) -> bool:
        """Perform a check on some markup to see if it looks like XML
        that's not XHTML. If so, issue a warning.

        This is much less reliable than doing the check while parsing,
        but some of the tree builders can't do that.

        :param stacklevel: The stacklevel of the code calling this\
         function.

        :return: True if the markup looks like non-XHTML XML, False
         otherwise.
        """
        if markup is None:
            return False
        markup = markup[:500]
        if isinstance(markup, bytes):
            markup_b: bytes = markup
            looks_like_xml = markup_b.startswith(
                cls.XML_PREFIX_B
            ) and not cls.LOOKS_LIKE_HTML_B.search(markup)
        else:
            markup_s: str = markup
            looks_like_xml = markup_s.startswith(
                cls.XML_PREFIX
            ) and not cls.LOOKS_LIKE_HTML.search(markup)

        if looks_like_xml:
            cls._warn(stacklevel=stacklevel + 2)
            return True
        return False

    @classmethod
    def _warn(cls, stacklevel: int = 5) -> None:
        """Issue a warning about XML being parsed as HTML."""
        warnings.warn(
            XMLParsedAsHTMLWarning.MESSAGE,
            XMLParsedAsHTMLWarning,
            stacklevel=stacklevel,
        )

    def _initialize_xml_detector(self) -> None:
        """Call this method before parsing a document."""
        self._first_processing_instruction = None
        self._root_tag_name = None

    def _document_might_be_xml(self, processing_instruction: str) -> None:
        """Call this method when encountering an XML declaration, or a
        "processing instruction" that might be an XML declaration.

        This helps Beautiful Soup detect potential issues later, if
        the XML document turns out to be a non-XHTML document that's
        being parsed as XML.
        """
        if (
            self._first_processing_instruction is not None
            or self._root_tag_name is not None
        ):
            # The document has already started. Don't bother checking
            # anymore.
            return

        self._first_processing_instruction = processing_instruction

        # We won't know until we encounter the first tag whether or
        # not this is actually a problem.

    def _root_tag_encountered(self, name: str) -> None:
        """Call this when you encounter the document's root tag.

        This is where we actually check whether an XML document is
        being incorrectly parsed as HTML, and issue the warning.
        """
        if self._root_tag_name is not None:
            # This method was incorrectly called multiple times. Do
            # nothing.
            return

        self._root_tag_name = name

        if (
            name != "html"
            and self._first_processing_instruction is not None
            and self._first_processing_instruction.lower().startswith("xml ")
        ):
            # We encountered an XML declaration and then a tag other
            # than 'html'. This is a reliable indicator that a
            # non-XHTML document is being parsed as XML.
            self._warn(stacklevel=10)


def register_treebuilders_from(module: ModuleType) -> None:
    """Copy TreeBuilders from the given module into this module."""
    this_module = sys.modules[__name__]
    for name in module.__all__:
        obj = getattr(module, name)

        if issubclass(obj, TreeBuilder):
            setattr(this_module, name, obj)
            this_module.__all__.append(name)
            # Register the builder while we're at it.
            this_module.builder_registry.register(obj)


# Builders are registered in reverse order of priority, so that custom
# builder registrations will take precedence. In general, we want lxml
# to take precedence over html5lib, because it's faster. And we only
# want to use HTMLParser as a last resort.
from . import _htmlparser # noqa: E402

register_treebuilders_from(_htmlparser)
try:
    from . import _html5lib

    register_treebuilders_from(_html5lib)
except ImportError:
    # They don't have html5lib installed.
    pass
try:
    from . import _lxml

    register_treebuilders_from(_lxml)
except ImportError:
    # They don't have lxml installed.
    pass

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\socks.py ===
from base64 import b64encode
try:
    from collections.abc import Callable
except ImportError:
    from collections import Callable
from errno import EOPNOTSUPP, EINVAL, EAGAIN
import functools
from io import BytesIO
import logging
import os
from os import SEEK_CUR
import socket
import struct
import sys

__version__ = "1.7.1"


if os.name == "nt" and sys.version_info < (3, 0):
    try:
        import win_inet_pton
    except ImportError:
        raise ImportError(
            "To run PySocks on Windows you must install win_inet_pton")

log = logging.getLogger(__name__)

PROXY_TYPE_SOCKS4 = SOCKS4 = 1
PROXY_TYPE_SOCKS5 = SOCKS5 = 2
PROXY_TYPE_HTTP = HTTP = 3

PROXY_TYPES = {"SOCKS4": SOCKS4, "SOCKS5": SOCKS5, "HTTP": HTTP}
PRINTABLE_PROXY_TYPES = dict(zip(PROXY_TYPES.values(), PROXY_TYPES.keys()))

_orgsocket = _orig_socket = socket.socket


def set_self_blocking(function):

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        self = args[0]
        try:
            _is_blocking = self.gettimeout()
            if _is_blocking == 0:
                self.setblocking(True)
            return function(*args, **kwargs)
        except Exception as e:
            raise
        finally:
            # set orgin blocking
            if _is_blocking == 0:
                self.setblocking(False)
    return wrapper


class ProxyError(IOError):
    """Socket_err contains original socket.error exception."""
    def __init__(self, msg, socket_err=None):
        self.msg = msg
        self.socket_err = socket_err

        if socket_err:
            self.msg += ": {}".format(socket_err)

    def __str__(self):
        return self.msg


class GeneralProxyError(ProxyError):
    pass


class ProxyConnectionError(ProxyError):
    pass


class SOCKS5AuthError(ProxyError):
    pass


class SOCKS5Error(ProxyError):
    pass


class SOCKS4Error(ProxyError):
    pass


class HTTPError(ProxyError):
    pass

SOCKS4_ERRORS = {
    0x5B: "Request rejected or failed",
    0x5C: ("Request rejected because SOCKS server cannot connect to identd on"
           " the client"),
    0x5D: ("Request rejected because the client program and identd report"
           " different user-ids")
}

SOCKS5_ERRORS = {
    0x01: "General SOCKS server failure",
    0x02: "Connection not allowed by ruleset",
    0x03: "Network unreachable",
    0x04: "Host unreachable",
    0x05: "Connection refused",
    0x06: "TTL expired",
    0x07: "Command not supported, or protocol error",
    0x08: "Address type not supported"
}

DEFAULT_PORTS = {SOCKS4: 1080, SOCKS5: 1080, HTTP: 8080}


def set_default_proxy(proxy_type=None, addr=None, port=None, rdns=True,
                      username=None, password=None):
    """Sets a default proxy.

    All further socksocket objects will use the default unless explicitly
    changed. All parameters are as for socket.set_proxy()."""
    socksocket.default_proxy = (proxy_type, addr, port, rdns,
                                username.encode() if username else None,
                                password.encode() if password else None)


def setdefaultproxy(*args, **kwargs):
    if "proxytype" in kwargs:
        kwargs["proxy_type"] = kwargs.pop("proxytype")
    return set_default_proxy(*args, **kwargs)


def get_default_proxy():
    """Returns the default proxy, set by set_default_proxy."""
    return socksocket.default_proxy

getdefaultproxy = get_default_proxy


def wrap_module(module):
    """Attempts to replace a module's socket library with a SOCKS socket.

    Must set a default proxy using set_default_proxy(...) first. This will
    only work on modules that import socket directly into the namespace;
    most of the Python Standard Library falls into this category."""
    if socksocket.default_proxy:
        module.socket.socket = socksocket
    else:
        raise GeneralProxyError("No default proxy specified")

wrapmodule = wrap_module


def create_connection(dest_pair,
                      timeout=None, source_address=None,
                      proxy_type=None, proxy_addr=None,
                      proxy_port=None, proxy_rdns=True,
                      proxy_username=None, proxy_password=None,
                      socket_options=None):
    """create_connection(dest_pair, *[, timeout], **proxy_args) -> socket object

    Like socket.create_connection(), but connects to proxy
    before returning the socket object.

    dest_pair - 2-tuple of (IP/hostname, port).
    **proxy_args - Same args passed to socksocket.set_proxy() if present.
    timeout - Optional socket timeout value, in seconds.
    source_address - tuple (host, port) for the socket to bind to as its source
    address before connecting (only for compatibility)
    """
    # Remove IPv6 brackets on the remote address and proxy address.
    remote_host, remote_port = dest_pair
    if remote_host.startswith("["):
        remote_host = remote_host.strip("[]")
    if proxy_addr and proxy_addr.startswith("["):
        proxy_addr = proxy_addr.strip("[]")

    err = None

    # Allow the SOCKS proxy to be on IPv4 or IPv6 addresses.
    for r in socket.getaddrinfo(proxy_addr, proxy_port, 0, socket.SOCK_STREAM):
        family, socket_type, proto, canonname, sa = r
        sock = None
        try:
            sock = socksocket(family, socket_type, proto)

            if socket_options:
                for opt in socket_options:
                    sock.setsockopt(*opt)

            if isinstance(timeout, (int, float)):
                sock.settimeout(timeout)

            if proxy_type:
                sock.set_proxy(proxy_type, proxy_addr, proxy_port, proxy_rdns,
                               proxy_username, proxy_password)
            if source_address:
                sock.bind(source_address)

            sock.connect((remote_host, remote_port))
            return sock

        except (socket.error, ProxyError) as e:
            err = e
            if sock:
                sock.close()
                sock = None

    if err:
        raise err

    raise socket.error("gai returned empty list.")


class _BaseSocket(socket.socket):
    """Allows Python 2 delegated methods such as send() to be overridden."""
    def __init__(self, *pos, **kw):
        _orig_socket.__init__(self, *pos, **kw)

        self._savedmethods = dict()
        for name in self._savenames:
            self._savedmethods[name] = getattr(self, name)
            delattr(self, name)  # Allows normal overriding mechanism to work

    _savenames = list()


def _makemethod(name):
    return lambda self, *pos, **kw: self._savedmethods[name](*pos, **kw)
for name in ("sendto", "send", "recvfrom", "recv"):
    method = getattr(_BaseSocket, name, None)

    # Determine if the method is not defined the usual way
    # as a function in the class.
    # Python 2 uses __slots__, so there are descriptors for each method,
    # but they are not functions.
    if not isinstance(method, Callable):
        _BaseSocket._savenames.append(name)
        setattr(_BaseSocket, name, _makemethod(name))


class socksocket(_BaseSocket):
    """socksocket([family[, type[, proto]]]) -> socket object

    Open a SOCKS enabled socket. The parameters are the same as
    those of the standard socket init. In order for SOCKS to work,
    you must specify family=AF_INET and proto=0.
    The "type" argument must be either SOCK_STREAM or SOCK_DGRAM.
    """

    default_proxy = None

    def __init__(self, family=socket.AF_INET, type=socket.SOCK_STREAM,
                 proto=0, *args, **kwargs):
        if type not in (socket.SOCK_STREAM, socket.SOCK_DGRAM):
            msg = "Socket type must be stream or datagram, not {!r}"
            raise ValueError(msg.format(type))

        super(socksocket, self).__init__(family, type, proto, *args, **kwargs)
        self._proxyconn = None  # TCP connection to keep UDP relay alive

        if self.default_proxy:
            self.proxy = self.default_proxy
        else:
            self.proxy = (None, None, None, None, None, None)
        self.proxy_sockname = None
        self.proxy_peername = None

        self._timeout = None

    def _readall(self, file, count):
        """Receive EXACTLY the number of bytes requested from the file object.

        Blocks until the required number of bytes have been received."""
        data = b""
        while len(data) < count:
            d = file.read(count - len(data))
            if not d:
                raise GeneralProxyError("Connection closed unexpectedly")
            data += d
        return data

    def settimeout(self, timeout):
        self._timeout = timeout
        try:
            # test if we're connected, if so apply timeout
            peer = self.get_proxy_peername()
            super(socksocket, self).settimeout(self._timeout)
        except socket.error:
            pass

    def gettimeout(self):
        return self._timeout

    def setblocking(self, v):
        if v:
            self.settimeout(None)
        else:
            self.settimeout(0.0)

    def set_proxy(self, proxy_type=None, addr=None, port=None, rdns=True,
                  username=None, password=None):
        """ Sets the proxy to be used.

        proxy_type -  The type of the proxy to be used. Three types
                        are supported: PROXY_TYPE_SOCKS4 (including socks4a),
                        PROXY_TYPE_SOCKS5 and PROXY_TYPE_HTTP
        addr -        The address of the server (IP or DNS).
        port -        The port of the server. Defaults to 1080 for SOCKS
                        servers and 8080 for HTTP proxy servers.
        rdns -        Should DNS queries be performed on the remote side
                       (rather than the local side). The default is True.
                       Note: This has no effect with SOCKS4 servers.
        username -    Username to authenticate with to the server.
                       The default is no authentication.
        password -    Password to authenticate with to the server.
                       Only relevant when username is also provided."""
        self.proxy = (proxy_type, addr, port, rdns,
                      username.encode() if username else None,
                      password.encode() if password else None)

    def setproxy(self, *args, **kwargs):
        if "proxytype" in kwargs:
            kwargs["proxy_type"] = kwargs.pop("proxytype")
        return self.set_proxy(*args, **kwargs)

    def bind(self, *pos, **kw):
        """Implements proxy connection for UDP sockets.

        Happens during the bind() phase."""
        (proxy_type, proxy_addr, proxy_port, rdns, username,
         password) = self.proxy
        if not proxy_type or self.type != socket.SOCK_DGRAM:
            return _orig_socket.bind(self, *pos, **kw)

        if self._proxyconn:
            raise socket.error(EINVAL, "Socket already bound to an address")
        if proxy_type != SOCKS5:
            msg = "UDP only supported by SOCKS5 proxy type"
            raise socket.error(EOPNOTSUPP, msg)
        super(socksocket, self).bind(*pos, **kw)

        # Need to specify actual local port because
        # some relays drop packets if a port of zero is specified.
        # Avoid specifying host address in case of NAT though.
        _, port = self.getsockname()
        dst = ("0", port)

        self._proxyconn = _orig_socket()
        proxy = self._proxy_addr()
        self._proxyconn.connect(proxy)

        UDP_ASSOCIATE = b"\x03"
        _, relay = self._SOCKS5_request(self._proxyconn, UDP_ASSOCIATE, dst)

        # The relay is most likely on the same host as the SOCKS proxy,
        # but some proxies return a private IP address (10.x.y.z)
        host, _ = proxy
        _, port = relay
        super(socksocket, self).connect((host, port))
        super(socksocket, self).settimeout(self._timeout)
        self.proxy_sockname = ("0.0.0.0", 0)  # Unknown

    def sendto(self, bytes, *args, **kwargs):
        if self.type != socket.SOCK_DGRAM:
            return super(socksocket, self).sendto(bytes, *args, **kwargs)
        if not self._proxyconn:
            self.bind(("", 0))

        address = args[-1]
        flags = args[:-1]

        header = BytesIO()
        RSV = b"\x00\x00"
        header.write(RSV)
        STANDALONE = b"\x00"
        header.write(STANDALONE)
        self._write_SOCKS5_address(address, header)

        sent = super(socksocket, self).send(header.getvalue() + bytes, *flags,
                                            **kwargs)
        return sent - header.tell()

    def send(self, bytes, flags=0, **kwargs):
        if self.type == socket.SOCK_DGRAM:
            return self.sendto(bytes, flags, self.proxy_peername, **kwargs)
        else:
            return super(socksocket, self).send(bytes, flags, **kwargs)

    def recvfrom(self, bufsize, flags=0):
        if self.type != socket.SOCK_DGRAM:
            return super(socksocket, self).recvfrom(bufsize, flags)
        if not self._proxyconn:
            self.bind(("", 0))

        buf = BytesIO(super(socksocket, self).recv(bufsize + 1024, flags))
        buf.seek(2, SEEK_CUR)
        frag = buf.read(1)
        if ord(frag):
            raise NotImplementedError("Received UDP packet fragment")
        fromhost, fromport = self._read_SOCKS5_address(buf)

        if self.proxy_peername:
            peerhost, peerport = self.proxy_peername
            if fromhost != peerhost or peerport not in (0, fromport):
                raise socket.error(EAGAIN, "Packet filtered")

        return (buf.read(bufsize), (fromhost, fromport))

    def recv(self, *pos, **kw):
        bytes, _ = self.recvfrom(*pos, **kw)
        return bytes

    def close(self):
        if self._proxyconn:
            self._proxyconn.close()
        return super(socksocket, self).close()

    def get_proxy_sockname(self):
        """Returns the bound IP address and port number at the proxy."""
        return self.proxy_sockname

    getproxysockname = get_proxy_sockname

    def get_proxy_peername(self):
        """
        Returns the IP and port number of the proxy.
        """
        return self.getpeername()

    getproxypeername = get_proxy_peername

    def get_peername(self):
        """Returns the IP address and port number of the destination machine.

        Note: get_proxy_peername returns the proxy."""
        return self.proxy_peername

    getpeername = get_peername

    def _negotiate_SOCKS5(self, *dest_addr):
        """Negotiates a stream connection through a SOCKS5 server."""
        CONNECT = b"\x01"
        self.proxy_peername, self.proxy_sockname = self._SOCKS5_request(
            self, CONNECT, dest_addr)

    def _SOCKS5_request(self, conn, cmd, dst):
        """
        Send SOCKS5 request with given command (CMD field) and
        address (DST field). Returns resolved DST address that was used.
        """
        proxy_type, addr, port, rdns, username, password = self.proxy

        writer = conn.makefile("wb")
        reader = conn.makefile("rb", 0)  # buffering=0 renamed in Python 3
        try:
            # First we'll send the authentication packages we support.
            if username and password:
                # The username/password details were supplied to the
                # set_proxy method so we support the USERNAME/PASSWORD
                # authentication (in addition to the standard none).
                writer.write(b"\x05\x02\x00\x02")
            else:
                # No username/password were entered, therefore we
                # only support connections with no authentication.
                writer.write(b"\x05\x01\x00")

            # We'll receive the server's response to determine which
            # method was selected
            writer.flush()
            chosen_auth = self._readall(reader, 2)

            if chosen_auth[0:1] != b"\x05":
                # Note: string[i:i+1] is used because indexing of a bytestring
                # via bytestring[i] yields an integer in Python 3
                raise GeneralProxyError(
                    "SOCKS5 proxy server sent invalid data")

            # Check the chosen authentication method

            if chosen_auth[1:2] == b"\x02":
                # Okay, we need to perform a basic username/password
                # authentication.
                if not (username and password):
                    # Although we said we don't support authentication, the
                    # server may still request basic username/password
                    # authentication
                    raise SOCKS5AuthError("No username/password supplied. "
                                          "Server requested username/password"
                                          " authentication")

                writer.write(b"\x01" + chr(len(username)).encode()
                             + username
                             + chr(len(password)).encode()
                             + password)
                writer.flush()
                auth_status = self._readall(reader, 2)
                if auth_status[0:1] != b"\x01":
                    # Bad response
                    raise GeneralProxyError(
                        "SOCKS5 proxy server sent invalid data")
                if auth_status[1:2] != b"\x00":
                    # Authentication failed
                    raise SOCKS5AuthError("SOCKS5 authentication failed")

                # Otherwise, authentication succeeded

            # No authentication is required if 0x00
            elif chosen_auth[1:2] != b"\x00":
                # Reaching here is always bad
                if chosen_auth[1:2] == b"\xFF":
                    raise SOCKS5AuthError(
                        "All offered SOCKS5 authentication methods were"
                        " rejected")
                else:
                    raise GeneralProxyError(
                        "SOCKS5 proxy server sent invalid data")

            # Now we can request the actual connection
            writer.write(b"\x05" + cmd + b"\x00")
            resolved = self._write_SOCKS5_address(dst, writer)
            writer.flush()

            # Get the response
            resp = self._readall(reader, 3)
            if resp[0:1] != b"\x05":
                raise GeneralProxyError(
                    "SOCKS5 proxy server sent invalid data")

            status = ord(resp[1:2])
            if status != 0x00:
                # Connection failed: server returned an error
                error = SOCKS5_ERRORS.get(status, "Unknown error")
                raise SOCKS5Error("{:#04x}: {}".format(status, error))

            # Get the bound address/port
            bnd = self._read_SOCKS5_address(reader)

            super(socksocket, self).settimeout(self._timeout)
            return (resolved, bnd)
        finally:
            reader.close()
            writer.close()

    def _write_SOCKS5_address(self, addr, file):
        """
        Return the host and port packed for the SOCKS5 protocol,
        and the resolved address as a tuple object.
        """
        host, port = addr
        proxy_type, _, _, rdns, username, password = self.proxy
        family_to_byte = {socket.AF_INET: b"\x01", socket.AF_INET6: b"\x04"}

        # If the given destination address is an IP address, we'll
        # use the IP address request even if remote resolving was specified.
        # Detect whether the address is IPv4/6 directly.
        for family in (socket.AF_INET, socket.AF_INET6):
            try:
                addr_bytes = socket.inet_pton(family, host)
                file.write(family_to_byte[family] + addr_bytes)
                host = socket.inet_ntop(family, addr_bytes)
                file.write(struct.pack(">H", port))
                return host, port
            except socket.error:
                continue

        # Well it's not an IP number, so it's probably a DNS name.
        if rdns:
            # Resolve remotely
            host_bytes = host.encode("idna")
            file.write(b"\x03" + chr(len(host_bytes)).encode() + host_bytes)
        else:
            # Resolve locally
            addresses = socket.getaddrinfo(host, port, socket.AF_UNSPEC,
                                           socket.SOCK_STREAM,
                                           socket.IPPROTO_TCP,
                                           socket.AI_ADDRCONFIG)
            # We can't really work out what IP is reachable, so just pick the
            # first.
            target_addr = addresses[0]
            family = target_addr[0]
            host = target_addr[4][0]

            addr_bytes = socket.inet_pton(family, host)
            file.write(family_to_byte[family] + addr_bytes)
            host = socket.inet_ntop(family, addr_bytes)
        file.write(struct.pack(">H", port))
        return host, port

    def _read_SOCKS5_address(self, file):
        atyp = self._readall(file, 1)
        if atyp == b"\x01":
            addr = socket.inet_ntoa(self._readall(file, 4))
        elif atyp == b"\x03":
            length = self._readall(file, 1)
            addr = self._readall(file, ord(length))
        elif atyp == b"\x04":
            addr = socket.inet_ntop(socket.AF_INET6, self._readall(file, 16))
        else:
            raise GeneralProxyError("SOCKS5 proxy server sent invalid data")

        port = struct.unpack(">H", self._readall(file, 2))[0]
        return addr, port

    def _negotiate_SOCKS4(self, dest_addr, dest_port):
        """Negotiates a connection through a SOCKS4 server."""
        proxy_type, addr, port, rdns, username, password = self.proxy

        writer = self.makefile("wb")
        reader = self.makefile("rb", 0)  # buffering=0 renamed in Python 3
        try:
            # Check if the destination address provided is an IP address
            remote_resolve = False
            try:
                addr_bytes = socket.inet_aton(dest_addr)
            except socket.error:
                # It's a DNS name. Check where it should be resolved.
                if rdns:
                    addr_bytes = b"\x00\x00\x00\x01"
                    remote_resolve = True
                else:
                    addr_bytes = socket.inet_aton(
                        socket.gethostbyname(dest_addr))

            # Construct the request packet
            writer.write(struct.pack(">BBH", 0x04, 0x01, dest_port))
            writer.write(addr_bytes)

            # The username parameter is considered userid for SOCKS4
            if username:
                writer.write(username)
            writer.write(b"\x00")

            # DNS name if remote resolving is required
            # NOTE: This is actually an extension to the SOCKS4 protocol
            # called SOCKS4A and may not be supported in all cases.
            if remote_resolve:
                writer.write(dest_addr.encode("idna") + b"\x00")
            writer.flush()

            # Get the response from the server
            resp = self._readall(reader, 8)
            if resp[0:1] != b"\x00":
                # Bad data
                raise GeneralProxyError(
                    "SOCKS4 proxy server sent invalid data")

            status = ord(resp[1:2])
            if status != 0x5A:
                # Connection failed: server returned an error
                error = SOCKS4_ERRORS.get(status, "Unknown error")
                raise SOCKS4Error("{:#04x}: {}".format(status, error))

            # Get the bound address/port
            self.proxy_sockname = (socket.inet_ntoa(resp[4:]),
                                   struct.unpack(">H", resp[2:4])[0])
            if remote_resolve:
                self.proxy_peername = socket.inet_ntoa(addr_bytes), dest_port
            else:
                self.proxy_peername = dest_addr, dest_port
        finally:
            reader.close()
            writer.close()

    def _negotiate_HTTP(self, dest_addr, dest_port):
        """Negotiates a connection through an HTTP server.

        NOTE: This currently only supports HTTP CONNECT-style proxies."""
        proxy_type, addr, port, rdns, username, password = self.proxy

        # If we need to resolve locally, we do this now
        addr = dest_addr if rdns else socket.gethostbyname(dest_addr)

        http_headers = [
            (b"CONNECT " + addr.encode("idna") + b":"
             + str(dest_port).encode() + b" HTTP/1.1"),
            b"Host: " + dest_addr.encode("idna")
        ]

        if username and password:
            http_headers.append(b"Proxy-Authorization: basic "
                                + b64encode(username + b":" + password))

        http_headers.append(b"\r\n")

        self.sendall(b"\r\n".join(http_headers))

        # We just need the first line to check if the connection was successful
        fobj = self.makefile()
        status_line = fobj.readline()
        fobj.close()

        if not status_line:
            raise GeneralProxyError("Connection closed unexpectedly")

        try:
            proto, status_code, status_msg = status_line.split(" ", 2)
        except ValueError:
            raise GeneralProxyError("HTTP proxy server sent invalid response")

        if not proto.startswith("HTTP/"):
            raise GeneralProxyError(
                "Proxy server does not appear to be an HTTP proxy")

        try:
            status_code = int(status_code)
        except ValueError:
            raise HTTPError(
                "HTTP proxy server did not return a valid HTTP status")

        if status_code != 200:
            error = "{}: {}".format(status_code, status_msg)
            if status_code in (400, 403, 405):
                # It's likely that the HTTP proxy server does not support the
                # CONNECT tunneling method
                error += ("\n[*] Note: The HTTP proxy server may not be"
                          " supported by PySocks (must be a CONNECT tunnel"
                          " proxy)")
            raise HTTPError(error)

        self.proxy_sockname = (b"0.0.0.0", 0)
        self.proxy_peername = addr, dest_port

    _proxy_negotiators = {
                           SOCKS4: _negotiate_SOCKS4,
                           SOCKS5: _negotiate_SOCKS5,
                           HTTP: _negotiate_HTTP
                         }

    @set_self_blocking
    def connect(self, dest_pair, catch_errors=None):
        """
        Connects to the specified destination through a proxy.
        Uses the same API as socket's connect().
        To select the proxy server, use set_proxy().

        dest_pair - 2-tuple of (IP/hostname, port).
        """
        if len(dest_pair) != 2 or dest_pair[0].startswith("["):
            # Probably IPv6, not supported -- raise an error, and hope
            # Happy Eyeballs (RFC6555) makes sure at least the IPv4
            # connection works...
            raise socket.error("PySocks doesn't support IPv6: %s"
                               % str(dest_pair))

        dest_addr, dest_port = dest_pair

        if self.type == socket.SOCK_DGRAM:
            if not self._proxyconn:
                self.bind(("", 0))
            dest_addr = socket.gethostbyname(dest_addr)

            # If the host address is INADDR_ANY or similar, reset the peer
            # address so that packets are received from any peer
            if dest_addr == "0.0.0.0" and not dest_port:
                self.proxy_peername = None
            else:
                self.proxy_peername = (dest_addr, dest_port)
            return

        (proxy_type, proxy_addr, proxy_port, rdns, username,
         password) = self.proxy

        # Do a minimal input check first
        if (not isinstance(dest_pair, (list, tuple))
                or len(dest_pair) != 2
                or not dest_addr
                or not isinstance(dest_port, int)):
            # Inputs failed, raise an error
            raise GeneralProxyError(
                "Invalid destination-connection (host, port) pair")

        # We set the timeout here so that we don't hang in connection or during
        # negotiation.
        super(socksocket, self).settimeout(self._timeout)

        if proxy_type is None:
            # Treat like regular socket object
            self.proxy_peername = dest_pair
            super(socksocket, self).settimeout(self._timeout)
            super(socksocket, self).connect((dest_addr, dest_port))
            return

        proxy_addr = self._proxy_addr()

        try:
            # Initial connection to proxy server.
            super(socksocket, self).connect(proxy_addr)

        except socket.error as error:
            # Error while connecting to proxy
            self.close()
            if not catch_errors:
                proxy_addr, proxy_port = proxy_addr
                proxy_server = "{}:{}".format(proxy_addr, proxy_port)
                printable_type = PRINTABLE_PROXY_TYPES[proxy_type]

                msg = "Error connecting to {} proxy {}".format(printable_type,
                                                                    proxy_server)
                log.debug("%s due to: %s", msg, error)
                raise ProxyConnectionError(msg, error)
            else:
                raise error

        else:
            # Connected to proxy server, now negotiate
            try:
                # Calls negotiate_{SOCKS4, SOCKS5, HTTP}
                negotiate = self._proxy_negotiators[proxy_type]
                negotiate(self, dest_addr, dest_port)
            except socket.error as error:
                if not catch_errors:
                    # Wrap socket errors
                    self.close()
                    raise GeneralProxyError("Socket error", error)
                else:
                    raise error
            except ProxyError:
                # Protocol error while negotiating with proxy
                self.close()
                raise
                
    @set_self_blocking
    def connect_ex(self, dest_pair):
        """ https://docs.python.org/3/library/socket.html#socket.socket.connect_ex
        Like connect(address), but return an error indicator instead of raising an exception for errors returned by the C-level connect() call (other problems, such as "host not found" can still raise exceptions).
        """
        try:
            self.connect(dest_pair, catch_errors=True)
            return 0
        except OSError as e:
            # If the error is numeric (socket errors are numeric), then return number as 
            # connect_ex expects. Otherwise raise the error again (socket timeout for example)
            if e.errno:
                return e.errno
            else:
                raise

    def _proxy_addr(self):
        """
        Return proxy address to connect to as tuple object
        """
        (proxy_type, proxy_addr, proxy_port, rdns, username,
         password) = self.proxy
        proxy_port = proxy_port or DEFAULT_PORTS.get(proxy_type)
        if not proxy_port:
            raise GeneralProxyError("Invalid proxy type")
        return proxy_addr, proxy_port

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\mpmath\identification.py ===
"""
Implements the PSLQ algorithm for integer relation detection,
and derivative algorithms for constant recognition.
"""

from .libmp.backend import xrange
from .libmp import int_types, sqrt_fixed

# round to nearest integer (can be done more elegantly...)
def round_fixed(x, prec):
    return ((x + (1<<(prec-1))) >> prec) << prec

class IdentificationMethods(object):
    pass


def pslq(ctx, x, tol=None, maxcoeff=1000, maxsteps=100, verbose=False):
    r"""
    Given a vector of real numbers `x = [x_0, x_1, ..., x_n]`, ``pslq(x)``
    uses the PSLQ algorithm to find a list of integers
    `[c_0, c_1, ..., c_n]` such that

    .. math ::

        |c_1 x_1 + c_2 x_2 + ... + c_n x_n| < \mathrm{tol}

    and such that `\max |c_k| < \mathrm{maxcoeff}`. If no such vector
    exists, :func:`~mpmath.pslq` returns ``None``. The tolerance defaults to
    3/4 of the working precision.

    **Examples**

    Find rational approximations for `\pi`::

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> pslq([-1, pi], tol=0.01)
        [22, 7]
        >>> pslq([-1, pi], tol=0.001)
        [355, 113]
        >>> mpf(22)/7; mpf(355)/113; +pi
        3.14285714285714
        3.14159292035398
        3.14159265358979

    Pi is not a rational number with denominator less than 1000::

        >>> pslq([-1, pi])
        >>>

    To within the standard precision, it can however be approximated
    by at least one rational number with denominator less than `10^{12}`::

        >>> p, q = pslq([-1, pi], maxcoeff=10**12)
        >>> print(p); print(q)
        238410049439
        75888275702
        >>> mpf(p)/q
        3.14159265358979

    The PSLQ algorithm can be applied to long vectors. For example,
    we can investigate the rational (in)dependence of integer square
    roots::

        >>> mp.dps = 30
        >>> pslq([sqrt(n) for n in range(2, 5+1)])
        >>>
        >>> pslq([sqrt(n) for n in range(2, 6+1)])
        >>>
        >>> pslq([sqrt(n) for n in range(2, 8+1)])
        [2, 0, 0, 0, 0, 0, -1]

    **Machin formulas**

    A famous formula for `\pi` is Machin's,

    .. math ::

        \frac{\pi}{4} = 4 \operatorname{acot} 5 - \operatorname{acot} 239

    There are actually infinitely many formulas of this type. Two
    others are

    .. math ::

        \frac{\pi}{4} = \operatorname{acot} 1

        \frac{\pi}{4} = 12 \operatorname{acot} 49 + 32 \operatorname{acot} 57
            + 5 \operatorname{acot} 239 + 12 \operatorname{acot} 110443

    We can easily verify the formulas using the PSLQ algorithm::

        >>> mp.dps = 30
        >>> pslq([pi/4, acot(1)])
        [1, -1]
        >>> pslq([pi/4, acot(5), acot(239)])
        [1, -4, 1]
        >>> pslq([pi/4, acot(49), acot(57), acot(239), acot(110443)])
        [1, -12, -32, 5, -12]

    We could try to generate a custom Machin-like formula by running
    the PSLQ algorithm with a few inverse cotangent values, for example
    acot(2), acot(3) ... acot(10). Unfortunately, there is a linear
    dependence among these values, resulting in only that dependence
    being detected, with a zero coefficient for `\pi`::

        >>> pslq([pi] + [acot(n) for n in range(2,11)])
        [0, 1, -1, 0, 0, 0, -1, 0, 0, 0]

    We get better luck by removing linearly dependent terms::

        >>> pslq([pi] + [acot(n) for n in range(2,11) if n not in (3, 5)])
        [1, -8, 0, 0, 4, 0, 0, 0]

    In other words, we found the following formula::

        >>> 8*acot(2) - 4*acot(7)
        3.14159265358979323846264338328
        >>> +pi
        3.14159265358979323846264338328

    **Algorithm**

    This is a fairly direct translation to Python of the pseudocode given by
    David Bailey, "The PSLQ Integer Relation Algorithm":
    http://www.cecm.sfu.ca/organics/papers/bailey/paper/html/node3.html

    The present implementation uses fixed-point instead of floating-point
    arithmetic, since this is significantly (about 7x) faster.
    """

    n = len(x)
    if n < 2:
        raise ValueError("n cannot be less than 2")

    # At too low precision, the algorithm becomes meaningless
    prec = ctx.prec
    if prec < 53:
        raise ValueError("prec cannot be less than 53")

    if verbose and prec // max(2,n) < 5:
        print("Warning: precision for PSLQ may be too low")

    target = int(prec * 0.75)

    if tol is None:
        tol = ctx.mpf(2)**(-target)
    else:
        tol = ctx.convert(tol)

    extra = 60
    prec += extra

    if verbose:
        print("PSLQ using prec %i and tol %s" % (prec, ctx.nstr(tol)))

    tol = ctx.to_fixed(tol, prec)
    assert tol

    # Convert to fixed-point numbers. The dummy None is added so we can
    # use 1-based indexing. (This just allows us to be consistent with
    # Bailey's indexing. The algorithm is 100 lines long, so debugging
    # a single wrong index can be painful.)
    x = [None] + [ctx.to_fixed(ctx.mpf(xk), prec) for xk in x]

    # Sanity check on magnitudes
    minx = min(abs(xx) for xx in x[1:])
    if not minx:
        raise ValueError("PSLQ requires a vector of nonzero numbers")
    if minx < tol//100:
        if verbose:
            print("STOPPING: (one number is too small)")
        return None

    g = sqrt_fixed((4<<prec)//3, prec)
    A = {}
    B = {}
    H = {}
    # Initialization
    # step 1
    for i in xrange(1, n+1):
        for j in xrange(1, n+1):
            A[i,j] = B[i,j] = (i==j) << prec
            H[i,j] = 0
    # step 2
    s = [None] + [0] * n
    for k in xrange(1, n+1):
        t = 0
        for j in xrange(k, n+1):
            t += (x[j]**2 >> prec)
        s[k] = sqrt_fixed(t, prec)
    t = s[1]
    y = x[:]
    for k in xrange(1, n+1):
        y[k] = (x[k] << prec) // t
        s[k] = (s[k] << prec) // t
    # step 3
    for i in xrange(1, n+1):
        for j in xrange(i+1, n):
            H[i,j] = 0
        if i <= n-1:
            if s[i]:
                H[i,i] = (s[i+1] << prec) // s[i]
            else:
                H[i,i] = 0
        for j in range(1, i):
            sjj1 = s[j]*s[j+1]
            if sjj1:
                H[i,j] = ((-y[i]*y[j])<<prec)//sjj1
            else:
                H[i,j] = 0
    # step 4
    for i in xrange(2, n+1):
        for j in xrange(i-1, 0, -1):
            #t = floor(H[i,j]/H[j,j] + 0.5)
            if H[j,j]:
                t = round_fixed((H[i,j] << prec)//H[j,j], prec)
            else:
                #t = 0
                continue
            y[j] = y[j] + (t*y[i] >> prec)
            for k in xrange(1, j+1):
                H[i,k] = H[i,k] - (t*H[j,k] >> prec)
            for k in xrange(1, n+1):
                A[i,k] = A[i,k] - (t*A[j,k] >> prec)
                B[k,j] = B[k,j] + (t*B[k,i] >> prec)
    # Main algorithm
    for REP in range(maxsteps):
        # Step 1
        m = -1
        szmax = -1
        for i in range(1, n):
            h = H[i,i]
            sz = (g**i * abs(h)) >> (prec*(i-1))
            if sz > szmax:
                m = i
                szmax = sz
        # Step 2
        y[m], y[m+1] = y[m+1], y[m]
        for i in xrange(1,n+1): H[m,i], H[m+1,i] = H[m+1,i], H[m,i]
        for i in xrange(1,n+1): A[m,i], A[m+1,i] = A[m+1,i], A[m,i]
        for i in xrange(1,n+1): B[i,m], B[i,m+1] = B[i,m+1], B[i,m]
        # Step 3
        if m <= n - 2:
            t0 = sqrt_fixed((H[m,m]**2 + H[m,m+1]**2)>>prec, prec)
            # A zero element probably indicates that the precision has
            # been exhausted. XXX: this could be spurious, due to
            # using fixed-point arithmetic
            if not t0:
                break
            t1 = (H[m,m] << prec) // t0
            t2 = (H[m,m+1] << prec) // t0
            for i in xrange(m, n+1):
                t3 = H[i,m]
                t4 = H[i,m+1]
                H[i,m] = (t1*t3+t2*t4) >> prec
                H[i,m+1] = (-t2*t3+t1*t4) >> prec
        # Step 4
        for i in xrange(m+1, n+1):
            for j in xrange(min(i-1, m+1), 0, -1):
                try:
                    t = round_fixed((H[i,j] << prec)//H[j,j], prec)
                # Precision probably exhausted
                except ZeroDivisionError:
                    break
                y[j] = y[j] + ((t*y[i]) >> prec)
                for k in xrange(1, j+1):
                    H[i,k] = H[i,k] - (t*H[j,k] >> prec)
                for k in xrange(1, n+1):
                    A[i,k] = A[i,k] - (t*A[j,k] >> prec)
                    B[k,j] = B[k,j] + (t*B[k,i] >> prec)
        # Until a relation is found, the error typically decreases
        # slowly (e.g. a factor 1-10) with each step TODO: we could
        # compare err from two successive iterations. If there is a
        # large drop (several orders of magnitude), that indicates a
        # "high quality" relation was detected. Reporting this to
        # the user somehow might be useful.
        best_err = maxcoeff<<prec
        for i in xrange(1, n+1):
            err = abs(y[i])
            # Maybe we are done?
            if err < tol:
                # We are done if the coefficients are acceptable
                vec = [int(round_fixed(B[j,i], prec) >> prec) for j in \
                range(1,n+1)]
                if max(abs(v) for v in vec) < maxcoeff:
                    if verbose:
                        print("FOUND relation at iter %i/%i, error: %s" % \
                            (REP, maxsteps, ctx.nstr(err / ctx.mpf(2)**prec, 1)))
                    return vec
            best_err = min(err, best_err)
        # Calculate a lower bound for the norm. We could do this
        # more exactly (using the Euclidean norm) but there is probably
        # no practical benefit.
        recnorm = max(abs(h) for h in H.values())
        if recnorm:
            norm = ((1 << (2*prec)) // recnorm) >> prec
            norm //= 100
        else:
            norm = ctx.inf
        if verbose:
            print("%i/%i:  Error: %8s   Norm: %s" % \
                (REP, maxsteps, ctx.nstr(best_err / ctx.mpf(2)**prec, 1), norm))
        if norm >= maxcoeff:
            break
    if verbose:
        print("CANCELLING after step %i/%i." % (REP, maxsteps))
        print("Could not find an integer relation. Norm bound: %s" % norm)
    return None

def findpoly(ctx, x, n=1, **kwargs):
    r"""
    ``findpoly(x, n)`` returns the coefficients of an integer
    polynomial `P` of degree at most `n` such that `P(x) \approx 0`.
    If no polynomial having `x` as a root can be found,
    :func:`~mpmath.findpoly` returns ``None``.

    :func:`~mpmath.findpoly` works by successively calling :func:`~mpmath.pslq` with
    the vectors `[1, x]`, `[1, x, x^2]`, `[1, x, x^2, x^3]`, ...,
    `[1, x, x^2, .., x^n]` as input. Keyword arguments given to
    :func:`~mpmath.findpoly` are forwarded verbatim to :func:`~mpmath.pslq`. In
    particular, you can specify a tolerance for `P(x)` with ``tol``
    and a maximum permitted coefficient size with ``maxcoeff``.

    For large values of `n`, it is recommended to run :func:`~mpmath.findpoly`
    at high precision; preferably 50 digits or more.

    **Examples**

    By default (degree `n = 1`), :func:`~mpmath.findpoly` simply finds a linear
    polynomial with a rational root::

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> findpoly(0.7)
        [-10, 7]

    The generated coefficient list is valid input to ``polyval`` and
    ``polyroots``::

        >>> nprint(polyval(findpoly(phi, 2), phi), 1)
        -2.0e-16
        >>> for r in polyroots(findpoly(phi, 2)):
        ...     print(r)
        ...
        -0.618033988749895
        1.61803398874989

    Numbers of the form `m + n \sqrt p` for integers `(m, n, p)` are
    solutions to quadratic equations. As we find here, `1+\sqrt 2`
    is a root of the polynomial `x^2 - 2x - 1`::

        >>> findpoly(1+sqrt(2), 2)
        [1, -2, -1]
        >>> findroot(lambda x: x**2 - 2*x - 1, 1)
        2.4142135623731

    Despite only containing square roots, the following number results
    in a polynomial of degree 4::

        >>> findpoly(sqrt(2)+sqrt(3), 4)
        [1, 0, -10, 0, 1]

    In fact, `x^4 - 10x^2 + 1` is the *minimal polynomial* of
    `r = \sqrt 2 + \sqrt 3`, meaning that a rational polynomial of
    lower degree having `r` as a root does not exist. Given sufficient
    precision, :func:`~mpmath.findpoly` will usually find the correct
    minimal polynomial of a given algebraic number.

    **Non-algebraic numbers**

    If :func:`~mpmath.findpoly` fails to find a polynomial with given
    coefficient size and tolerance constraints, that means no such
    polynomial exists.

    We can verify that `\pi` is not an algebraic number of degree 3 with
    coefficients less than 1000::

        >>> mp.dps = 15
        >>> findpoly(pi, 3)
        >>>

    It is always possible to find an algebraic approximation of a number
    using one (or several) of the following methods:

        1. Increasing the permitted degree
        2. Allowing larger coefficients
        3. Reducing the tolerance

    One example of each method is shown below::

        >>> mp.dps = 15
        >>> findpoly(pi, 4)
        [95, -545, 863, -183, -298]
        >>> findpoly(pi, 3, maxcoeff=10000)
        [836, -1734, -2658, -457]
        >>> findpoly(pi, 3, tol=1e-7)
        [-4, 22, -29, -2]

    It is unknown whether Euler's constant is transcendental (or even
    irrational). We can use :func:`~mpmath.findpoly` to check that if is
    an algebraic number, its minimal polynomial must have degree
    at least 7 and a coefficient of magnitude at least 1000000::

        >>> mp.dps = 200
        >>> findpoly(euler, 6, maxcoeff=10**6, tol=1e-100, maxsteps=1000)
        >>>

    Note that the high precision and strict tolerance is necessary
    for such high-degree runs, since otherwise unwanted low-accuracy
    approximations will be detected. It may also be necessary to set
    maxsteps high to prevent a premature exit (before the coefficient
    bound has been reached). Running with ``verbose=True`` to get an
    idea what is happening can be useful.
    """
    x = ctx.mpf(x)
    if n < 1:
        raise ValueError("n cannot be less than 1")
    if x == 0:
        return [1, 0]
    xs = [ctx.mpf(1)]
    for i in range(1,n+1):
        xs.append(x**i)
        a = ctx.pslq(xs, **kwargs)
        if a is not None:
            return a[::-1]

def fracgcd(p, q):
    x, y = p, q
    while y:
        x, y = y, x % y
    if x != 1:
        p //= x
        q //= x
    if q == 1:
        return p
    return p, q

def pslqstring(r, constants):
    q = r[0]
    r = r[1:]
    s = []
    for i in range(len(r)):
        p = r[i]
        if p:
            z = fracgcd(-p,q)
            cs = constants[i][1]
            if cs == '1':
                cs = ''
            else:
                cs = '*' + cs
            if isinstance(z, int_types):
                if z > 0: term = str(z) + cs
                else:     term = ("(%s)" % z) + cs
            else:
                term = ("(%s/%s)" % z) + cs
            s.append(term)
    s = ' + '.join(s)
    if '+' in s or '*' in s:
        s = '(' + s + ')'
    return s or '0'

def prodstring(r, constants):
    q = r[0]
    r = r[1:]
    num = []
    den = []
    for i in range(len(r)):
        p = r[i]
        if p:
            z = fracgcd(-p,q)
            cs = constants[i][1]
            if isinstance(z, int_types):
                if abs(z) == 1: t = cs
                else:           t = '%s**%s' % (cs, abs(z))
                ([num,den][z<0]).append(t)
            else:
                t = '%s**(%s/%s)' % (cs, abs(z[0]), z[1])
                ([num,den][z[0]<0]).append(t)
    num = '*'.join(num)
    den = '*'.join(den)
    if num and den: return "(%s)/(%s)" % (num, den)
    if num: return num
    if den: return "1/(%s)" % den

def quadraticstring(ctx,t,a,b,c):
    if c < 0:
        a,b,c = -a,-b,-c
    u1 = (-b+ctx.sqrt(b**2-4*a*c))/(2*c)
    u2 = (-b-ctx.sqrt(b**2-4*a*c))/(2*c)
    if abs(u1-t) < abs(u2-t):
        if b:  s = '((%s+sqrt(%s))/%s)' % (-b,b**2-4*a*c,2*c)
        else:  s = '(sqrt(%s)/%s)' % (-4*a*c,2*c)
    else:
        if b:  s = '((%s-sqrt(%s))/%s)' % (-b,b**2-4*a*c,2*c)
        else:  s = '(-sqrt(%s)/%s)' % (-4*a*c,2*c)
    return s

# Transformation y = f(x,c), with inverse function x = f(y,c)
# The third entry indicates whether the transformation is
# redundant when c = 1
transforms = [
  (lambda ctx,x,c: x*c, '$y/$c', 0),
  (lambda ctx,x,c: x/c, '$c*$y', 1),
  (lambda ctx,x,c: c/x, '$c/$y', 0),
  (lambda ctx,x,c: (x*c)**2, 'sqrt($y)/$c', 0),
  (lambda ctx,x,c: (x/c)**2, '$c*sqrt($y)', 1),
  (lambda ctx,x,c: (c/x)**2, '$c/sqrt($y)', 0),
  (lambda ctx,x,c: c*x**2, 'sqrt($y)/sqrt($c)', 1),
  (lambda ctx,x,c: x**2/c, 'sqrt($c)*sqrt($y)', 1),
  (lambda ctx,x,c: c/x**2, 'sqrt($c)/sqrt($y)', 1),
  (lambda ctx,x,c: ctx.sqrt(x*c), '$y**2/$c', 0),
  (lambda ctx,x,c: ctx.sqrt(x/c), '$c*$y**2', 1),
  (lambda ctx,x,c: ctx.sqrt(c/x), '$c/$y**2', 0),
  (lambda ctx,x,c: c*ctx.sqrt(x), '$y**2/$c**2', 1),
  (lambda ctx,x,c: ctx.sqrt(x)/c, '$c**2*$y**2', 1),
  (lambda ctx,x,c: c/ctx.sqrt(x), '$c**2/$y**2', 1),
  (lambda ctx,x,c: ctx.exp(x*c), 'log($y)/$c', 0),
  (lambda ctx,x,c: ctx.exp(x/c), '$c*log($y)', 1),
  (lambda ctx,x,c: ctx.exp(c/x), '$c/log($y)', 0),
  (lambda ctx,x,c: c*ctx.exp(x), 'log($y/$c)', 1),
  (lambda ctx,x,c: ctx.exp(x)/c, 'log($c*$y)', 1),
  (lambda ctx,x,c: c/ctx.exp(x), 'log($c/$y)', 0),
  (lambda ctx,x,c: ctx.ln(x*c), 'exp($y)/$c', 0),
  (lambda ctx,x,c: ctx.ln(x/c), '$c*exp($y)', 1),
  (lambda ctx,x,c: ctx.ln(c/x), '$c/exp($y)', 0),
  (lambda ctx,x,c: c*ctx.ln(x), 'exp($y/$c)', 1),
  (lambda ctx,x,c: ctx.ln(x)/c, 'exp($c*$y)', 1),
  (lambda ctx,x,c: c/ctx.ln(x), 'exp($c/$y)', 0),
]

def identify(ctx, x, constants=[], tol=None, maxcoeff=1000, full=False,
    verbose=False):
    r"""
    Given a real number `x`, ``identify(x)`` attempts to find an exact
    formula for `x`. This formula is returned as a string. If no match
    is found, ``None`` is returned. With ``full=True``, a list of
    matching formulas is returned.

    As a simple example, :func:`~mpmath.identify` will find an algebraic
    formula for the golden ratio::

        >>> from mpmath import *
        >>> mp.dps = 15; mp.pretty = True
        >>> identify(phi)
        '((1+sqrt(5))/2)'

    :func:`~mpmath.identify` can identify simple algebraic numbers and simple
    combinations of given base constants, as well as certain basic
    transformations thereof. More specifically, :func:`~mpmath.identify`
    looks for the following:

        1. Fractions
        2. Quadratic algebraic numbers
        3. Rational linear combinations of the base constants
        4. Any of the above after first transforming `x` into `f(x)` where
           `f(x)` is `1/x`, `\sqrt x`, `x^2`, `\log x` or `\exp x`, either
           directly or with `x` or `f(x)` multiplied or divided by one of
           the base constants
        5. Products of fractional powers of the base constants and
           small integers

    Base constants can be given as a list of strings representing mpmath
    expressions (:func:`~mpmath.identify` will ``eval`` the strings to numerical
    values and use the original strings for the output), or as a dict of
    formula:value pairs.

    In order not to produce spurious results, :func:`~mpmath.identify` should
    be used with high precision; preferably 50 digits or more.

    **Examples**

    Simple identifications can be performed safely at standard
    precision. Here the default recognition of rational, algebraic,
    and exp/log of algebraic numbers is demonstrated::

        >>> mp.dps = 15
        >>> identify(0.22222222222222222)
        '(2/9)'
        >>> identify(1.9662210973805663)
        'sqrt(((24+sqrt(48))/8))'
        >>> identify(4.1132503787829275)
        'exp((sqrt(8)/2))'
        >>> identify(0.881373587019543)
        'log(((2+sqrt(8))/2))'

    By default, :func:`~mpmath.identify` does not recognize `\pi`. At standard
    precision it finds a not too useful approximation. At slightly
    increased precision, this approximation is no longer accurate
    enough and :func:`~mpmath.identify` more correctly returns ``None``::

        >>> identify(pi)
        '(2**(176/117)*3**(20/117)*5**(35/39))/(7**(92/117))'
        >>> mp.dps = 30
        >>> identify(pi)
        >>>

    Numbers such as `\pi`, and simple combinations of user-defined
    constants, can be identified if they are provided explicitly::

        >>> identify(3*pi-2*e, ['pi', 'e'])
        '(3*pi + (-2)*e)'

    Here is an example using a dict of constants. Note that the
    constants need not be "atomic"; :func:`~mpmath.identify` can just
    as well express the given number in terms of expressions
    given by formulas::

        >>> identify(pi+e, {'a':pi+2, 'b':2*e})
        '((-2) + 1*a + (1/2)*b)'

    Next, we attempt some identifications with a set of base constants.
    It is necessary to increase the precision a bit.

        >>> mp.dps = 50
        >>> base = ['sqrt(2)','pi','log(2)']
        >>> identify(0.25, base)
        '(1/4)'
        >>> identify(3*pi + 2*sqrt(2) + 5*log(2)/7, base)
        '(2*sqrt(2) + 3*pi + (5/7)*log(2))'
        >>> identify(exp(pi+2), base)
        'exp((2 + 1*pi))'
        >>> identify(1/(3+sqrt(2)), base)
        '((3/7) + (-1/7)*sqrt(2))'
        >>> identify(sqrt(2)/(3*pi+4), base)
        'sqrt(2)/(4 + 3*pi)'
        >>> identify(5**(mpf(1)/3)*pi*log(2)**2, base)
        '5**(1/3)*pi*log(2)**2'

    An example of an erroneous solution being found when too low
    precision is used::

        >>> mp.dps = 15
        >>> identify(1/(3*pi-4*e+sqrt(8)), ['pi', 'e', 'sqrt(2)'])
        '((11/25) + (-158/75)*pi + (76/75)*e + (44/15)*sqrt(2))'
        >>> mp.dps = 50
        >>> identify(1/(3*pi-4*e+sqrt(8)), ['pi', 'e', 'sqrt(2)'])
        '1/(3*pi + (-4)*e + 2*sqrt(2))'

    **Finding approximate solutions**

    The tolerance ``tol`` defaults to 3/4 of the working precision.
    Lowering the tolerance is useful for finding approximate matches.
    We can for example try to generate approximations for pi::

        >>> mp.dps = 15
        >>> identify(pi, tol=1e-2)
        '(22/7)'
        >>> identify(pi, tol=1e-3)
        '(355/113)'
        >>> identify(pi, tol=1e-10)
        '(5**(339/269))/(2**(64/269)*3**(13/269)*7**(92/269))'

    With ``full=True``, and by supplying a few base constants,
    ``identify`` can generate almost endless lists of approximations
    for any number (the output below has been truncated to show only
    the first few)::

        >>> for p in identify(pi, ['e', 'catalan'], tol=1e-5, full=True):
        ...     print(p)
        ...  # doctest: +ELLIPSIS
        e/log((6 + (-4/3)*e))
        (3**3*5*e*catalan**2)/(2*7**2)
        sqrt(((-13) + 1*e + 22*catalan))
        log(((-6) + 24*e + 4*catalan)/e)
        exp(catalan*((-1/5) + (8/15)*e))
        catalan*(6 + (-6)*e + 15*catalan)
        sqrt((5 + 26*e + (-3)*catalan))/e
        e*sqrt(((-27) + 2*e + 25*catalan))
        log(((-1) + (-11)*e + 59*catalan))
        ((3/20) + (21/20)*e + (3/20)*catalan)
        ...

    The numerical values are roughly as close to `\pi` as permitted by the
    specified tolerance:

        >>> e/log(6-4*e/3)
        3.14157719846001
        >>> 135*e*catalan**2/98
        3.14166950419369
        >>> sqrt(e-13+22*catalan)
        3.14158000062992
        >>> log(24*e-6+4*catalan)-1
        3.14158791577159

    **Symbolic processing**

    The output formula can be evaluated as a Python expression.
    Note however that if fractions (like '2/3') are present in
    the formula, Python's :func:`~mpmath.eval()` may erroneously perform
    integer division. Note also that the output is not necessarily
    in the algebraically simplest form::

        >>> identify(sqrt(2))
        '(sqrt(8)/2)'

    As a solution to both problems, consider using SymPy's
    :func:`~mpmath.sympify` to convert the formula into a symbolic expression.
    SymPy can be used to pretty-print or further simplify the formula
    symbolically::

        >>> from sympy import sympify # doctest: +SKIP
        >>> sympify(identify(sqrt(2))) # doctest: +SKIP
        2**(1/2)

    Sometimes :func:`~mpmath.identify` can simplify an expression further than
    a symbolic algorithm::

        >>> from sympy import simplify # doctest: +SKIP
        >>> x = sympify('-1/(-3/2+(1/2)*5**(1/2))*(3/2-1/2*5**(1/2))**(1/2)') # doctest: +SKIP
        >>> x # doctest: +SKIP
        (3/2 - 5**(1/2)/2)**(-1/2)
        >>> x = simplify(x) # doctest: +SKIP
        >>> x # doctest: +SKIP
        2/(6 - 2*5**(1/2))**(1/2)
        >>> mp.dps = 30 # doctest: +SKIP
        >>> x = sympify(identify(x.evalf(30))) # doctest: +SKIP
        >>> x # doctest: +SKIP
        1/2 + 5**(1/2)/2

    (In fact, this functionality is available directly in SymPy as the
    function :func:`~mpmath.nsimplify`, which is essentially a wrapper for
    :func:`~mpmath.identify`.)

    **Miscellaneous issues and limitations**

    The input `x` must be a real number. All base constants must be
    positive real numbers and must not be rationals or rational linear
    combinations of each other.

    The worst-case computation time grows quickly with the number of
    base constants. Already with 3 or 4 base constants,
    :func:`~mpmath.identify` may require several seconds to finish. To search
    for relations among a large number of constants, you should
    consider using :func:`~mpmath.pslq` directly.

    The extended transformations are applied to x, not the constants
    separately. As a result, ``identify`` will for example be able to
    recognize ``exp(2*pi+3)`` with ``pi`` given as a base constant, but
    not ``2*exp(pi)+3``. It will be able to recognize the latter if
    ``exp(pi)`` is given explicitly as a base constant.

    """

    solutions = []

    def addsolution(s):
        if verbose: print("Found: ", s)
        solutions.append(s)

    x = ctx.mpf(x)

    # Further along, x will be assumed positive
    if x == 0:
        if full: return ['0']
        else:    return '0'
    if x < 0:
        sol = ctx.identify(-x, constants, tol, maxcoeff, full, verbose)
        if sol is None:
            return sol
        if full:
            return ["-(%s)"%s for s in sol]
        else:
            return "-(%s)" % sol

    if tol:
        tol = ctx.mpf(tol)
    else:
        tol = ctx.eps**0.7
    M = maxcoeff

    if constants:
        if isinstance(constants, dict):
            constants = [(ctx.mpf(v), name) for (name, v) in sorted(constants.items())]
        else:
            namespace = dict((name, getattr(ctx,name)) for name in dir(ctx))
            constants = [(eval(p, namespace), p) for p in constants]
    else:
        constants = []

    # We always want to find at least rational terms
    if 1 not in [value for (name, value) in constants]:
        constants = [(ctx.mpf(1), '1')] + constants

    # PSLQ with simple algebraic and functional transformations
    for ft, ftn, red in transforms:
        for c, cn in constants:
            if red and cn == '1':
                continue
            t = ft(ctx,x,c)
            # Prevent exponential transforms from wreaking havoc
            if abs(t) > M**2 or abs(t) < tol:
                continue
            # Linear combination of base constants
            r = ctx.pslq([t] + [a[0] for a in constants], tol, M)
            s = None
            if r is not None and max(abs(uw) for uw in r) <= M and r[0]:
                s = pslqstring(r, constants)
            # Quadratic algebraic numbers
            else:
                q = ctx.pslq([ctx.one, t, t**2], tol, M)
                if q is not None and len(q) == 3 and q[2]:
                    aa, bb, cc = q
                    if max(abs(aa),abs(bb),abs(cc)) <= M:
                        s = quadraticstring(ctx,t,aa,bb,cc)
            if s:
                if cn == '1' and ('/$c' in ftn):
                    s = ftn.replace('$y', s).replace('/$c', '')
                else:
                    s = ftn.replace('$y', s).replace('$c', cn)
                addsolution(s)
                if not full: return solutions[0]

            if verbose:
                print(".")

    # Check for a direct multiplicative formula
    if x != 1:
        # Allow fractional powers of fractions
        ilogs = [2,3,5,7]
        # Watch out for existing fractional powers of fractions
        logs = []
        for a, s in constants:
            if not sum(bool(ctx.findpoly(ctx.ln(a)/ctx.ln(i),1)) for i in ilogs):
                logs.append((ctx.ln(a), s))
        logs = [(ctx.ln(i),str(i)) for i in ilogs] + logs
        r = ctx.pslq([ctx.ln(x)] + [a[0] for a in logs], tol, M)
        if r is not None and max(abs(uw) for uw in r) <= M and r[0]:
            addsolution(prodstring(r, logs))
            if not full: return solutions[0]

    if full:
        return sorted(solutions, key=len)
    else:
        return None

IdentificationMethods.pslq = pslq
IdentificationMethods.findpoly = findpoly
IdentificationMethods.identify = identify


if __name__ == '__main__':
    import doctest
    doctest.testmod()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pandas\core\indexes\datetimelike.py ===
"""
Base and utility classes for tseries type pandas objects.
"""
from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    cast,
    final,
)
import warnings

import numpy as np

from pandas._config import using_copy_on_write

from pandas._libs import (
    NaT,
    Timedelta,
    lib,
)
from pandas._libs.tslibs import (
    BaseOffset,
    Resolution,
    Tick,
    parsing,
    to_offset,
)
from pandas._libs.tslibs.dtypes import freq_to_period_freqstr
from pandas.compat.numpy import function as nv
from pandas.errors import (
    InvalidIndexError,
    NullFrequencyError,
)
from pandas.util._decorators import (
    Appender,
    cache_readonly,
    doc,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import (
    is_integer,
    is_list_like,
)
from pandas.core.dtypes.concat import concat_compat
from pandas.core.dtypes.dtypes import CategoricalDtype

from pandas.core.arrays import (
    DatetimeArray,
    ExtensionArray,
    PeriodArray,
    TimedeltaArray,
)
from pandas.core.arrays.datetimelike import DatetimeLikeArrayMixin
import pandas.core.common as com
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import (
    Index,
    _index_shared_docs,
)
from pandas.core.indexes.extension import NDArrayBackedExtensionIndex
from pandas.core.indexes.range import RangeIndex
from pandas.core.tools.timedeltas import to_timedelta

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime

    from pandas._typing import (
        Axis,
        Self,
        npt,
    )

    from pandas import CategoricalIndex

_index_doc_kwargs = dict(ibase._index_doc_kwargs)


class DatetimeIndexOpsMixin(NDArrayBackedExtensionIndex, ABC):
    """
    Common ops mixin to support a unified interface datetimelike Index.
    """

    _can_hold_strings = False
    _data: DatetimeArray | TimedeltaArray | PeriodArray

    @doc(DatetimeLikeArrayMixin.mean)
    def mean(self, *, skipna: bool = True, axis: int | None = 0):
        return self._data.mean(skipna=skipna, axis=axis)

    @property
    def freq(self) -> BaseOffset | None:
        return self._data.freq

    @freq.setter
    def freq(self, value) -> None:
        # error: Property "freq" defined in "PeriodArray" is read-only  [misc]
        self._data.freq = value  # type: ignore[misc]

    @property
    def asi8(self) -> npt.NDArray[np.int64]:
        return self._data.asi8

    @property
    @doc(DatetimeLikeArrayMixin.freqstr)
    def freqstr(self) -> str:
        from pandas import PeriodIndex

        if self._data.freqstr is not None and isinstance(
            self._data, (PeriodArray, PeriodIndex)
        ):
            freq = freq_to_period_freqstr(self._data.freq.n, self._data.freq.name)
            return freq
        else:
            return self._data.freqstr  # type: ignore[return-value]

    @cache_readonly
    @abstractmethod
    def _resolution_obj(self) -> Resolution:
        ...

    @cache_readonly
    @doc(DatetimeLikeArrayMixin.resolution)
    def resolution(self) -> str:
        return self._data.resolution

    # ------------------------------------------------------------------------

    @cache_readonly
    def hasnans(self) -> bool:
        return self._data._hasna

    def equals(self, other: Any) -> bool:
        """
        Determines if two Index objects contain the same elements.
        """
        if self.is_(other):
            return True

        if not isinstance(other, Index):
            return False
        elif other.dtype.kind in "iufc":
            return False
        elif not isinstance(other, type(self)):
            should_try = False
            inferable = self._data._infer_matches
            if other.dtype == object:
                should_try = other.inferred_type in inferable
            elif isinstance(other.dtype, CategoricalDtype):
                other = cast("CategoricalIndex", other)
                should_try = other.categories.inferred_type in inferable

            if should_try:
                try:
                    other = type(self)(other)
                except (ValueError, TypeError, OverflowError):
                    # e.g.
                    #  ValueError -> cannot parse str entry, or OutOfBoundsDatetime
                    #  TypeError  -> trying to convert IntervalIndex to DatetimeIndex
                    #  OverflowError -> Index([very_large_timedeltas])
                    return False

        if self.dtype != other.dtype:
            # have different timezone
            return False

        return np.array_equal(self.asi8, other.asi8)

    @Appender(Index.__contains__.__doc__)
    def __contains__(self, key: Any) -> bool:
        hash(key)
        try:
            self.get_loc(key)
        except (KeyError, TypeError, ValueError, InvalidIndexError):
            return False
        return True

    def _convert_tolerance(self, tolerance, target):
        tolerance = np.asarray(to_timedelta(tolerance).to_numpy())
        return super()._convert_tolerance(tolerance, target)

    # --------------------------------------------------------------------
    # Rendering Methods
    _default_na_rep = "NaT"

    def format(
        self,
        name: bool = False,
        formatter: Callable | None = None,
        na_rep: str = "NaT",
        date_format: str | None = None,
    ) -> list[str]:
        """
        Render a string representation of the Index.
        """
        warnings.warn(
            # GH#55413
            f"{type(self).__name__}.format is deprecated and will be removed "
            "in a future version. Convert using index.astype(str) or "
            "index.map(formatter) instead.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
        header = []
        if name:
            header.append(
                ibase.pprint_thing(self.name, escape_chars=("\t", "\r", "\n"))
                if self.name is not None
                else ""
            )

        if formatter is not None:
            return header + list(self.map(formatter))

        return self._format_with_header(
            header=header, na_rep=na_rep, date_format=date_format
        )

    def _format_with_header(
        self, *, header: list[str], na_rep: str, date_format: str | None = None
    ) -> list[str]:
        # TODO: not reached in tests 2023-10-11
        # matches base class except for whitespace padding and date_format
        return header + list(
            self._get_values_for_csv(na_rep=na_rep, date_format=date_format)
        )

    @property
    def _formatter_func(self):
        return self._data._formatter()

    def _format_attrs(self):
        """
        Return a list of tuples of the (attr,formatted_value).
        """
        attrs = super()._format_attrs()
        for attrib in self._attributes:
            # iterating over _attributes prevents us from doing this for PeriodIndex
            if attrib == "freq":
                freq = self.freqstr
                if freq is not None:
                    freq = repr(freq)  # e.g. D -> 'D'
                attrs.append(("freq", freq))
        return attrs

    @Appender(Index._summary.__doc__)
    def _summary(self, name=None) -> str:
        result = super()._summary(name=name)
        if self.freq:
            result += f"\nFreq: {self.freqstr}"

        return result

    # --------------------------------------------------------------------
    # Indexing Methods

    @final
    def _can_partial_date_slice(self, reso: Resolution) -> bool:
        # e.g. test_getitem_setitem_periodindex
        # History of conversation GH#3452, GH#3931, GH#2369, GH#14826
        return reso > self._resolution_obj
        # NB: for DTI/PI, not TDI

    def _parsed_string_to_bounds(self, reso: Resolution, parsed):
        raise NotImplementedError

    def _parse_with_reso(self, label: str):
        # overridden by TimedeltaIndex
        try:
            if self.freq is None or hasattr(self.freq, "rule_code"):
                freq = self.freq
        except NotImplementedError:
            freq = getattr(self, "freqstr", getattr(self, "inferred_freq", None))

        freqstr: str | None
        if freq is not None and not isinstance(freq, str):
            freqstr = freq.rule_code
        else:
            freqstr = freq

        if isinstance(label, np.str_):
            # GH#45580
            label = str(label)

        parsed, reso_str = parsing.parse_datetime_string_with_reso(label, freqstr)
        reso = Resolution.from_attrname(reso_str)
        return parsed, reso

    def _get_string_slice(self, key: str):
        # overridden by TimedeltaIndex
        parsed, reso = self._parse_with_reso(key)
        try:
            return self._partial_date_slice(reso, parsed)
        except KeyError as err:
            raise KeyError(key) from err

    @final
    def _partial_date_slice(
        self,
        reso: Resolution,
        parsed: datetime,
    ) -> slice | npt.NDArray[np.intp]:
        """
        Parameters
        ----------
        reso : Resolution
        parsed : datetime

        Returns
        -------
        slice or ndarray[intp]
        """
        if not self._can_partial_date_slice(reso):
            raise ValueError

        t1, t2 = self._parsed_string_to_bounds(reso, parsed)
        vals = self._data._ndarray
        unbox = self._data._unbox

        if self.is_monotonic_increasing:
            if len(self) and (
                (t1 < self[0] and t2 < self[0]) or (t1 > self[-1] and t2 > self[-1])
            ):
                # we are out of range
                raise KeyError

            # TODO: does this depend on being monotonic _increasing_?

            # a monotonic (sorted) series can be sliced
            left = vals.searchsorted(unbox(t1), side="left")
            right = vals.searchsorted(unbox(t2), side="right")
            return slice(left, right)

        else:
            lhs_mask = vals >= unbox(t1)
            rhs_mask = vals <= unbox(t2)

            # try to find the dates
            return (lhs_mask & rhs_mask).nonzero()[0]

    def _maybe_cast_slice_bound(self, label, side: str):
        """
        If label is a string, cast it to scalar type according to resolution.

        Parameters
        ----------
        label : object
        side : {'left', 'right'}

        Returns
        -------
        label : object

        Notes
        -----
        Value of `side` parameter should be validated in caller.
        """
        if isinstance(label, str):
            try:
                parsed, reso = self._parse_with_reso(label)
            except ValueError as err:
                # DTI -> parsing.DateParseError
                # TDI -> 'unit abbreviation w/o a number'
                # PI -> string cannot be parsed as datetime-like
                self._raise_invalid_indexer("slice", label, err)

            lower, upper = self._parsed_string_to_bounds(reso, parsed)
            return lower if side == "left" else upper
        elif not isinstance(label, self._data._recognized_scalars):
            self._raise_invalid_indexer("slice", label)

        return label

    # --------------------------------------------------------------------
    # Arithmetic Methods

    def shift(self, periods: int = 1, freq=None) -> Self:
        """
        Shift index by desired number of time frequency increments.

        This method is for shifting the values of datetime-like indexes
        by a specified time increment a given number of times.

        Parameters
        ----------
        periods : int, default 1
            Number of periods (or increments) to shift by,
            can be positive or negative.
        freq : pandas.DateOffset, pandas.Timedelta or string, optional
            Frequency increment to shift by.
            If None, the index is shifted by its own `freq` attribute.
            Offset aliases are valid strings, e.g., 'D', 'W', 'M' etc.

        Returns
        -------
        pandas.DatetimeIndex
            Shifted index.

        See Also
        --------
        Index.shift : Shift values of Index.
        PeriodIndex.shift : Shift values of PeriodIndex.
        """
        raise NotImplementedError

    # --------------------------------------------------------------------

    @doc(Index._maybe_cast_listlike_indexer)
    def _maybe_cast_listlike_indexer(self, keyarr):
        try:
            res = self._data._validate_listlike(keyarr, allow_object=True)
        except (ValueError, TypeError):
            if not isinstance(keyarr, ExtensionArray):
                # e.g. we don't want to cast DTA to ndarray[object]
                res = com.asarray_tuplesafe(keyarr)
                # TODO: com.asarray_tuplesafe shouldn't cast e.g. DatetimeArray
            else:
                res = keyarr
        return Index(res, dtype=res.dtype)


class DatetimeTimedeltaMixin(DatetimeIndexOpsMixin, ABC):
    """
    Mixin class for methods shared by DatetimeIndex and TimedeltaIndex,
    but not PeriodIndex
    """

    _data: DatetimeArray | TimedeltaArray
    _comparables = ["name", "freq"]
    _attributes = ["name", "freq"]

    # Compat for frequency inference, see GH#23789
    _is_monotonic_increasing = Index.is_monotonic_increasing
    _is_monotonic_decreasing = Index.is_monotonic_decreasing
    _is_unique = Index.is_unique

    @property
    def unit(self) -> str:
        return self._data.unit

    def as_unit(self, unit: str) -> Self:
        """
        Convert to a dtype with the given unit resolution.

        Parameters
        ----------
        unit : {'s', 'ms', 'us', 'ns'}

        Returns
        -------
        same type as self

        Examples
        --------
        For :class:`pandas.DatetimeIndex`:

        >>> idx = pd.DatetimeIndex(['2020-01-02 01:02:03.004005006'])
        >>> idx
        DatetimeIndex(['2020-01-02 01:02:03.004005006'],
                      dtype='datetime64[ns]', freq=None)
        >>> idx.as_unit('s')
        DatetimeIndex(['2020-01-02 01:02:03'], dtype='datetime64[s]', freq=None)

        For :class:`pandas.TimedeltaIndex`:

        >>> tdelta_idx = pd.to_timedelta(['1 day 3 min 2 us 42 ns'])
        >>> tdelta_idx
        TimedeltaIndex(['1 days 00:03:00.000002042'],
                        dtype='timedelta64[ns]', freq=None)
        >>> tdelta_idx.as_unit('s')
        TimedeltaIndex(['1 days 00:03:00'], dtype='timedelta64[s]', freq=None)
        """
        arr = self._data.as_unit(unit)
        return type(self)._simple_new(arr, name=self.name)

    def _with_freq(self, freq):
        arr = self._data._with_freq(freq)
        return type(self)._simple_new(arr, name=self._name)

    @property
    def values(self) -> np.ndarray:
        # NB: For Datetime64TZ this is lossy
        data = self._data._ndarray
        if using_copy_on_write():
            data = data.view()
            data.flags.writeable = False
        return data

    @doc(DatetimeIndexOpsMixin.shift)
    def shift(self, periods: int = 1, freq=None) -> Self:
        if freq is not None and freq != self.freq:
            if isinstance(freq, str):
                freq = to_offset(freq)
            offset = periods * freq
            return self + offset

        if periods == 0 or len(self) == 0:
            # GH#14811 empty case
            return self.copy()

        if self.freq is None:
            raise NullFrequencyError("Cannot shift with no freq")

        start = self[0] + periods * self.freq
        end = self[-1] + periods * self.freq

        # Note: in the DatetimeTZ case, _generate_range will infer the
        #  appropriate timezone from `start` and `end`, so tz does not need
        #  to be passed explicitly.
        result = self._data._generate_range(
            start=start, end=end, periods=None, freq=self.freq, unit=self.unit
        )
        return type(self)._simple_new(result, name=self.name)

    @cache_readonly
    @doc(DatetimeLikeArrayMixin.inferred_freq)
    def inferred_freq(self) -> str | None:
        return self._data.inferred_freq

    # --------------------------------------------------------------------
    # Set Operation Methods

    @cache_readonly
    def _as_range_index(self) -> RangeIndex:
        # Convert our i8 representations to RangeIndex
        # Caller is responsible for checking isinstance(self.freq, Tick)
        freq = cast(Tick, self.freq)
        tick = Timedelta(freq).as_unit("ns")._value
        rng = range(self[0]._value, self[-1]._value + tick, tick)
        return RangeIndex(rng)

    def _can_range_setop(self, other) -> bool:
        return isinstance(self.freq, Tick) and isinstance(other.freq, Tick)

    def _wrap_range_setop(self, other, res_i8) -> Self:
        new_freq = None
        if not len(res_i8):
            # RangeIndex defaults to step=1, which we don't want.
            new_freq = self.freq
        elif isinstance(res_i8, RangeIndex):
            new_freq = to_offset(Timedelta(res_i8.step))

        # TODO(GH#41493): we cannot just do
        #  type(self._data)(res_i8.values, dtype=self.dtype, freq=new_freq)
        # because test_setops_preserve_freq fails with _validate_frequency raising.
        # This raising is incorrect, as 'on_freq' is incorrect. This will
        # be fixed by GH#41493
        res_values = res_i8.values.view(self._data._ndarray.dtype)
        result = type(self._data)._simple_new(
            # error: Argument "dtype" to "_simple_new" of "DatetimeArray" has
            # incompatible type "Union[dtype[Any], ExtensionDtype]"; expected
            # "Union[dtype[datetime64], DatetimeTZDtype]"
            res_values,
            dtype=self.dtype,  # type: ignore[arg-type]
            freq=new_freq,  # type: ignore[arg-type]
        )
        return cast("Self", self._wrap_setop_result(other, result))

    def _range_intersect(self, other, sort) -> Self:
        # Dispatch to RangeIndex intersection logic.
        left = self._as_range_index
        right = other._as_range_index
        res_i8 = left.intersection(right, sort=sort)
        return self._wrap_range_setop(other, res_i8)

    def _range_union(self, other, sort) -> Self:
        # Dispatch to RangeIndex union logic.
        left = self._as_range_index
        right = other._as_range_index
        res_i8 = left.union(right, sort=sort)
        return self._wrap_range_setop(other, res_i8)

    def _intersection(self, other: Index, sort: bool = False) -> Index:
        """
        intersection specialized to the case with matching dtypes and both non-empty.
        """
        other = cast("DatetimeTimedeltaMixin", other)

        if self._can_range_setop(other):
            return self._range_intersect(other, sort=sort)

        if not self._can_fast_intersect(other):
            result = Index._intersection(self, other, sort=sort)
            # We need to invalidate the freq because Index._intersection
            #  uses _shallow_copy on a view of self._data, which will preserve
            #  self.freq if we're not careful.
            # At this point we should have result.dtype == self.dtype
            #  and type(result) is type(self._data)
            result = self._wrap_setop_result(other, result)
            return result._with_freq(None)._with_freq("infer")

        else:
            return self._fast_intersect(other, sort)

    def _fast_intersect(self, other, sort):
        # to make our life easier, "sort" the two ranges
        if self[0] <= other[0]:
            left, right = self, other
        else:
            left, right = other, self

        # after sorting, the intersection always starts with the right index
        # and ends with the index of which the last elements is smallest
        end = min(left[-1], right[-1])
        start = right[0]

        if end < start:
            result = self[:0]
        else:
            lslice = slice(*left.slice_locs(start, end))
            result = left._values[lslice]

        return result

    def _can_fast_intersect(self, other: Self) -> bool:
        # Note: we only get here with len(self) > 0 and len(other) > 0
        if self.freq is None:
            return False

        elif other.freq != self.freq:
            return False

        elif not self.is_monotonic_increasing:
            # Because freq is not None, we must then be monotonic decreasing
            return False

        # this along with matching freqs ensure that we "line up",
        #  so intersection will preserve freq
        # Note we are assuming away Ticks, as those go through _range_intersect
        # GH#42104
        return self.freq.n == 1

    def _can_fast_union(self, other: Self) -> bool:
        # Assumes that type(self) == type(other), as per the annotation
        # The ability to fast_union also implies that `freq` should be
        #  retained on union.
        freq = self.freq

        if freq is None or freq != other.freq:
            return False

        if not self.is_monotonic_increasing:
            # Because freq is not None, we must then be monotonic decreasing
            # TODO: do union on the reversed indexes?
            return False

        if len(self) == 0 or len(other) == 0:
            # only reached via union_many
            return True

        # to make our life easier, "sort" the two ranges
        if self[0] <= other[0]:
            left, right = self, other
        else:
            left, right = other, self

        right_start = right[0]
        left_end = left[-1]

        # Only need to "adjoin", not overlap
        return (right_start == left_end + freq) or right_start in left

    def _fast_union(self, other: Self, sort=None) -> Self:
        # Caller is responsible for ensuring self and other are non-empty

        # to make our life easier, "sort" the two ranges
        if self[0] <= other[0]:
            left, right = self, other
        elif sort is False:
            # TDIs are not in the "correct" order and we don't want
            #  to sort but want to remove overlaps
            left, right = self, other
            left_start = left[0]
            loc = right.searchsorted(left_start, side="left")
            right_chunk = right._values[:loc]
            dates = concat_compat((left._values, right_chunk))
            result = type(self)._simple_new(dates, name=self.name)
            return result
        else:
            left, right = other, self

        left_end = left[-1]
        right_end = right[-1]

        # concatenate
        if left_end < right_end:
            loc = right.searchsorted(left_end, side="right")
            right_chunk = right._values[loc:]
            dates = concat_compat([left._values, right_chunk])
            # The can_fast_union check ensures that the result.freq
            #  should match self.freq
            assert isinstance(dates, type(self._data))
            # error: Item "ExtensionArray" of "ExtensionArray |
            # ndarray[Any, Any]" has no attribute "_freq"
            assert dates._freq == self.freq  # type: ignore[union-attr]
            result = type(self)._simple_new(dates)
            return result
        else:
            return left

    def _union(self, other, sort):
        # We are called by `union`, which is responsible for this validation
        assert isinstance(other, type(self))
        assert self.dtype == other.dtype

        if self._can_range_setop(other):
            return self._range_union(other, sort=sort)

        if self._can_fast_union(other):
            result = self._fast_union(other, sort=sort)
            # in the case with sort=None, the _can_fast_union check ensures
            #  that result.freq == self.freq
            return result
        else:
            return super()._union(other, sort)._with_freq("infer")

    # --------------------------------------------------------------------
    # Join Methods

    def _get_join_freq(self, other):
        """
        Get the freq to attach to the result of a join operation.
        """
        freq = None
        if self._can_fast_union(other):
            freq = self.freq
        return freq

    def _wrap_joined_index(
        self, joined, other, lidx: npt.NDArray[np.intp], ridx: npt.NDArray[np.intp]
    ):
        assert other.dtype == self.dtype, (other.dtype, self.dtype)
        result = super()._wrap_joined_index(joined, other, lidx, ridx)
        result._data._freq = self._get_join_freq(other)
        return result

    def _get_engine_target(self) -> np.ndarray:
        # engine methods and libjoin methods need dt64/td64 values cast to i8
        return self._data._ndarray.view("i8")

    def _from_join_target(self, result: np.ndarray):
        # view e.g. i8 back to M8[ns]
        result = result.view(self._data._ndarray.dtype)
        return self._data._from_backing_data(result)

    # --------------------------------------------------------------------
    # List-like Methods

    def _get_delete_freq(self, loc: int | slice | Sequence[int]):
        """
        Find the `freq` for self.delete(loc).
        """
        freq = None
        if self.freq is not None:
            if is_integer(loc):
                if loc in (0, -len(self), -1, len(self) - 1):
                    freq = self.freq
            else:
                if is_list_like(loc):
                    # error: Incompatible types in assignment (expression has
                    # type "Union[slice, ndarray]", variable has type
                    # "Union[int, slice, Sequence[int]]")
                    loc = lib.maybe_indices_to_slice(  # type: ignore[assignment]
                        np.asarray(loc, dtype=np.intp), len(self)
                    )
                if isinstance(loc, slice) and loc.step in (1, None):
                    if loc.start in (0, None) or loc.stop in (len(self), None):
                        freq = self.freq
        return freq

    def _get_insert_freq(self, loc: int, item):
        """
        Find the `freq` for self.insert(loc, item).
        """
        value = self._data._validate_scalar(item)
        item = self._data._box_func(value)

        freq = None
        if self.freq is not None:
            # freq can be preserved on edge cases
            if self.size:
                if item is NaT:
                    pass
                elif loc in (0, -len(self)) and item + self.freq == self[0]:
                    freq = self.freq
                elif (loc == len(self)) and item - self.freq == self[-1]:
                    freq = self.freq
            else:
                # Adding a single item to an empty index may preserve freq
                if isinstance(self.freq, Tick):
                    # all TimedeltaIndex cases go through here; is_on_offset
                    #  would raise TypeError
                    freq = self.freq
                elif self.freq.is_on_offset(item):
                    freq = self.freq
        return freq

    @doc(NDArrayBackedExtensionIndex.delete)
    def delete(self, loc) -> Self:
        result = super().delete(loc)
        result._data._freq = self._get_delete_freq(loc)
        return result

    @doc(NDArrayBackedExtensionIndex.insert)
    def insert(self, loc: int, item):
        result = super().insert(loc, item)
        if isinstance(result, type(self)):
            # i.e. parent class method did not cast
            result._data._freq = self._get_insert_freq(loc, item)
        return result

    # --------------------------------------------------------------------
    # NDArray-Like Methods

    @Appender(_index_shared_docs["take"] % _index_doc_kwargs)
    def take(
        self,
        indices,
        axis: Axis = 0,
        allow_fill: bool = True,
        fill_value=None,
        **kwargs,
    ) -> Self:
        nv.validate_take((), kwargs)
        indices = np.asarray(indices, dtype=np.intp)

        result = NDArrayBackedExtensionIndex.take(
            self, indices, axis, allow_fill, fill_value, **kwargs
        )

        maybe_slice = lib.maybe_indices_to_slice(indices, len(self))
        if isinstance(maybe_slice, slice):
            freq = self._data._get_getitem_freq(maybe_slice)
            result._data._freq = freq
        return result