
# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\operations\install\wheel.py ===
"""Support for installing and building the "wheel" binary package format."""

import collections
import compileall
import contextlib
import csv
import importlib
import logging
import os.path
import re
import shutil
import sys
import warnings
from base64 import urlsafe_b64encode
from email.message import Message
from itertools import chain, filterfalse, starmap
from typing import (
    IO,
    Any,
    BinaryIO,
    Callable,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    NewType,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)
from zipfile import ZipFile, ZipInfo

from pip._vendor.distlib.scripts import ScriptMaker
from pip._vendor.distlib.util import get_export_entry
from pip._vendor.packaging.utils import canonicalize_name

from pip._internal.exceptions import InstallationError
from pip._internal.locations import get_major_minor_version
from pip._internal.metadata import (
    BaseDistribution,
    FilesystemWheel,
    get_wheel_distribution,
)
from pip._internal.models.direct_url import DIRECT_URL_METADATA_NAME, DirectUrl
from pip._internal.models.scheme import SCHEME_KEYS, Scheme
from pip._internal.utils.filesystem import adjacent_tmp_file, replace
from pip._internal.utils.misc import StreamWrapper, ensure_dir, hash_file, partition
from pip._internal.utils.unpacking import (
    current_umask,
    is_within_directory,
    set_extracted_file_to_default_mode_plus_executable,
    zip_item_is_executable,
)
from pip._internal.utils.wheel import parse_wheel


class File(Protocol):
    src_record_path: "RecordPath"
    dest_path: str
    changed: bool

    def save(self) -> None:
        pass


logger = logging.getLogger(__name__)

RecordPath = NewType("RecordPath", str)
InstalledCSVRow = Tuple[RecordPath, str, Union[int, str]]


def rehash(path: str, blocksize: int = 1 << 20) -> Tuple[str, str]:
    """Return (encoded_digest, length) for path using hashlib.sha256()"""
    h, length = hash_file(path, blocksize)
    digest = "sha256=" + urlsafe_b64encode(h.digest()).decode("latin1").rstrip("=")
    return (digest, str(length))


def csv_io_kwargs(mode: str) -> Dict[str, Any]:
    """Return keyword arguments to properly open a CSV file
    in the given mode.
    """
    return {"mode": mode, "newline": "", "encoding": "utf-8"}


def fix_script(path: str) -> bool:
    """Replace #!python with #!/path/to/python
    Return True if file was changed.
    """
    # XXX RECORD hashes will need to be updated
    assert os.path.isfile(path)

    with open(path, "rb") as script:
        firstline = script.readline()
        if not firstline.startswith(b"#!python"):
            return False
        exename = sys.executable.encode(sys.getfilesystemencoding())
        firstline = b"#!" + exename + os.linesep.encode("ascii")
        rest = script.read()
    with open(path, "wb") as script:
        script.write(firstline)
        script.write(rest)
    return True


def wheel_root_is_purelib(metadata: Message) -> bool:
    return metadata.get("Root-Is-Purelib", "").lower() == "true"


def get_entrypoints(dist: BaseDistribution) -> Tuple[Dict[str, str], Dict[str, str]]:
    console_scripts = {}
    gui_scripts = {}
    for entry_point in dist.iter_entry_points():
        if entry_point.group == "console_scripts":
            console_scripts[entry_point.name] = entry_point.value
        elif entry_point.group == "gui_scripts":
            gui_scripts[entry_point.name] = entry_point.value
    return console_scripts, gui_scripts


def message_about_scripts_not_on_PATH(scripts: Sequence[str]) -> Optional[str]:
    """Determine if any scripts are not on PATH and format a warning.
    Returns a warning message if one or more scripts are not on PATH,
    otherwise None.
    """
    if not scripts:
        return None

    # Group scripts by the path they were installed in
    grouped_by_dir: Dict[str, Set[str]] = collections.defaultdict(set)
    for destfile in scripts:
        parent_dir = os.path.dirname(destfile)
        script_name = os.path.basename(destfile)
        grouped_by_dir[parent_dir].add(script_name)

    # We don't want to warn for directories that are on PATH.
    not_warn_dirs = [
        os.path.normcase(os.path.normpath(i)).rstrip(os.sep)
        for i in os.environ.get("PATH", "").split(os.pathsep)
    ]
    # If an executable sits with sys.executable, we don't warn for it.
    #     This covers the case of venv invocations without activating the venv.
    not_warn_dirs.append(
        os.path.normcase(os.path.normpath(os.path.dirname(sys.executable)))
    )
    warn_for: Dict[str, Set[str]] = {
        parent_dir: scripts
        for parent_dir, scripts in grouped_by_dir.items()
        if os.path.normcase(os.path.normpath(parent_dir)) not in not_warn_dirs
    }
    if not warn_for:
        return None

    # Format a message
    msg_lines = []
    for parent_dir, dir_scripts in warn_for.items():
        sorted_scripts: List[str] = sorted(dir_scripts)
        if len(sorted_scripts) == 1:
            start_text = f"script {sorted_scripts[0]} is"
        else:
            start_text = "scripts {} are".format(
                ", ".join(sorted_scripts[:-1]) + " and " + sorted_scripts[-1]
            )

        msg_lines.append(
            f"The {start_text} installed in '{parent_dir}' which is not on PATH."
        )

    last_line_fmt = (
        "Consider adding {} to PATH or, if you prefer "
        "to suppress this warning, use --no-warn-script-location."
    )
    if len(msg_lines) == 1:
        msg_lines.append(last_line_fmt.format("this directory"))
    else:
        msg_lines.append(last_line_fmt.format("these directories"))

    # Add a note if any directory starts with ~
    warn_for_tilde = any(
        i[0] == "~" for i in os.environ.get("PATH", "").split(os.pathsep) if i
    )
    if warn_for_tilde:
        tilde_warning_msg = (
            "NOTE: The current PATH contains path(s) starting with `~`, "
            "which may not be expanded by all applications."
        )
        msg_lines.append(tilde_warning_msg)

    # Returns the formatted multiline message
    return "\n".join(msg_lines)


def _normalized_outrows(
    outrows: Iterable[InstalledCSVRow],
) -> List[Tuple[str, str, str]]:
    """Normalize the given rows of a RECORD file.

    Items in each row are converted into str. Rows are then sorted to make
    the value more predictable for tests.

    Each row is a 3-tuple (path, hash, size) and corresponds to a record of
    a RECORD file (see PEP 376 and PEP 427 for details).  For the rows
    passed to this function, the size can be an integer as an int or string,
    or the empty string.
    """
    # Normally, there should only be one row per path, in which case the
    # second and third elements don't come into play when sorting.
    # However, in cases in the wild where a path might happen to occur twice,
    # we don't want the sort operation to trigger an error (but still want
    # determinism).  Since the third element can be an int or string, we
    # coerce each element to a string to avoid a TypeError in this case.
    # For additional background, see--
    # https://github.com/pypa/pip/issues/5868
    return sorted(
        (record_path, hash_, str(size)) for record_path, hash_, size in outrows
    )


def _record_to_fs_path(record_path: RecordPath, lib_dir: str) -> str:
    return os.path.join(lib_dir, record_path)


def _fs_to_record_path(path: str, lib_dir: str) -> RecordPath:
    # On Windows, do not handle relative paths if they belong to different
    # logical disks
    if os.path.splitdrive(path)[0].lower() == os.path.splitdrive(lib_dir)[0].lower():
        path = os.path.relpath(path, lib_dir)

    path = path.replace(os.path.sep, "/")
    return cast("RecordPath", path)


def get_csv_rows_for_installed(
    old_csv_rows: List[List[str]],
    installed: Dict[RecordPath, RecordPath],
    changed: Set[RecordPath],
    generated: List[str],
    lib_dir: str,
) -> List[InstalledCSVRow]:
    """
    :param installed: A map from archive RECORD path to installation RECORD
        path.
    """
    installed_rows: List[InstalledCSVRow] = []
    for row in old_csv_rows:
        if len(row) > 3:
            logger.warning("RECORD line has more than three elements: %s", row)
        old_record_path = cast("RecordPath", row[0])
        new_record_path = installed.pop(old_record_path, old_record_path)
        if new_record_path in changed:
            digest, length = rehash(_record_to_fs_path(new_record_path, lib_dir))
        else:
            digest = row[1] if len(row) > 1 else ""
            length = row[2] if len(row) > 2 else ""
        installed_rows.append((new_record_path, digest, length))
    for f in generated:
        path = _fs_to_record_path(f, lib_dir)
        digest, length = rehash(f)
        installed_rows.append((path, digest, length))
    return installed_rows + [
        (installed_record_path, "", "") for installed_record_path in installed.values()
    ]


def get_console_script_specs(console: Dict[str, str]) -> List[str]:
    """
    Given the mapping from entrypoint name to callable, return the relevant
    console script specs.
    """
    # Don't mutate caller's version
    console = console.copy()

    scripts_to_generate = []

    # Special case pip and setuptools to generate versioned wrappers
    #
    # The issue is that some projects (specifically, pip and setuptools) use
    # code in setup.py to create "versioned" entry points - pip2.7 on Python
    # 2.7, pip3.3 on Python 3.3, etc. But these entry points are baked into
    # the wheel metadata at build time, and so if the wheel is installed with
    # a *different* version of Python the entry points will be wrong. The
    # correct fix for this is to enhance the metadata to be able to describe
    # such versioned entry points.
    # Currently, projects using versioned entry points will either have
    # incorrect versioned entry points, or they will not be able to distribute
    # "universal" wheels (i.e., they will need a wheel per Python version).
    #
    # Because setuptools and pip are bundled with _ensurepip and virtualenv,
    # we need to use universal wheels. As a workaround, we
    # override the versioned entry points in the wheel and generate the
    # correct ones.
    #
    # To add the level of hack in this section of code, in order to support
    # ensurepip this code will look for an ``ENSUREPIP_OPTIONS`` environment
    # variable which will control which version scripts get installed.
    #
    # ENSUREPIP_OPTIONS=altinstall
    #   - Only pipX.Y and easy_install-X.Y will be generated and installed
    # ENSUREPIP_OPTIONS=install
    #   - pipX.Y, pipX, easy_install-X.Y will be generated and installed. Note
    #     that this option is technically if ENSUREPIP_OPTIONS is set and is
    #     not altinstall
    # DEFAULT
    #   - The default behavior is to install pip, pipX, pipX.Y, easy_install
    #     and easy_install-X.Y.
    pip_script = console.pop("pip", None)
    if pip_script:
        if "ENSUREPIP_OPTIONS" not in os.environ:
            scripts_to_generate.append("pip = " + pip_script)

        if os.environ.get("ENSUREPIP_OPTIONS", "") != "altinstall":
            scripts_to_generate.append(f"pip{sys.version_info[0]} = {pip_script}")

        scripts_to_generate.append(f"pip{get_major_minor_version()} = {pip_script}")
        # Delete any other versioned pip entry points
        pip_ep = [k for k in console if re.match(r"pip(\d+(\.\d+)?)?$", k)]
        for k in pip_ep:
            del console[k]
    easy_install_script = console.pop("easy_install", None)
    if easy_install_script:
        if "ENSUREPIP_OPTIONS" not in os.environ:
            scripts_to_generate.append("easy_install = " + easy_install_script)

        scripts_to_generate.append(
            f"easy_install-{get_major_minor_version()} = {easy_install_script}"
        )
        # Delete any other versioned easy_install entry points
        easy_install_ep = [
            k for k in console if re.match(r"easy_install(-\d+\.\d+)?$", k)
        ]
        for k in easy_install_ep:
            del console[k]

    # Generate the console entry points specified in the wheel
    scripts_to_generate.extend(starmap("{} = {}".format, console.items()))

    return scripts_to_generate


class ZipBackedFile:
    def __init__(
        self, src_record_path: RecordPath, dest_path: str, zip_file: ZipFile
    ) -> None:
        self.src_record_path = src_record_path
        self.dest_path = dest_path
        self._zip_file = zip_file
        self.changed = False

    def _getinfo(self) -> ZipInfo:
        return self._zip_file.getinfo(self.src_record_path)

    def save(self) -> None:
        # When we open the output file below, any existing file is truncated
        # before we start writing the new contents. This is fine in most
        # cases, but can cause a segfault if pip has loaded a shared
        # object (e.g. from pyopenssl through its vendored urllib3)
        # Since the shared object is mmap'd an attempt to call a
        # symbol in it will then cause a segfault. Unlinking the file
        # allows writing of new contents while allowing the process to
        # continue to use the old copy.
        if os.path.exists(self.dest_path):
            os.unlink(self.dest_path)

        zipinfo = self._getinfo()

        # optimization: the file is created by open(),
        # skip the decompression when there is 0 bytes to decompress.
        with open(self.dest_path, "wb") as dest:
            if zipinfo.file_size > 0:
                with self._zip_file.open(zipinfo) as f:
                    blocksize = min(zipinfo.file_size, 1024 * 1024)
                    shutil.copyfileobj(f, dest, blocksize)

        if zip_item_is_executable(zipinfo):
            set_extracted_file_to_default_mode_plus_executable(self.dest_path)


class ScriptFile:
    def __init__(self, file: "File") -> None:
        self._file = file
        self.src_record_path = self._file.src_record_path
        self.dest_path = self._file.dest_path
        self.changed = False

    def save(self) -> None:
        self._file.save()
        self.changed = fix_script(self.dest_path)


class MissingCallableSuffix(InstallationError):
    def __init__(self, entry_point: str) -> None:
        super().__init__(
            f"Invalid script entry point: {entry_point} - A callable "
            "suffix is required. Cf https://packaging.python.org/"
            "specifications/entry-points/#use-for-scripts for more "
            "information."
        )


def _raise_for_invalid_entrypoint(specification: str) -> None:
    entry = get_export_entry(specification)
    if entry is not None and entry.suffix is None:
        raise MissingCallableSuffix(str(entry))


class PipScriptMaker(ScriptMaker):
    def make(
        self, specification: str, options: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        _raise_for_invalid_entrypoint(specification)
        return super().make(specification, options)


def _install_wheel(  # noqa: C901, PLR0915 function is too long
    name: str,
    wheel_zip: ZipFile,
    wheel_path: str,
    scheme: Scheme,
    pycompile: bool = True,
    warn_script_location: bool = True,
    direct_url: Optional[DirectUrl] = None,
    requested: bool = False,
) -> None:
    """Install a wheel.

    :param name: Name of the project to install
    :param wheel_zip: open ZipFile for wheel being installed
    :param scheme: Distutils scheme dictating the install directories
    :param req_description: String used in place of the requirement, for
        logging
    :param pycompile: Whether to byte-compile installed Python files
    :param warn_script_location: Whether to check that scripts are installed
        into a directory on PATH
    :raises UnsupportedWheel:
        * when the directory holds an unpacked wheel with incompatible
          Wheel-Version
        * when the .dist-info dir does not match the wheel
    """
    info_dir, metadata = parse_wheel(wheel_zip, name)

    if wheel_root_is_purelib(metadata):
        lib_dir = scheme.purelib
    else:
        lib_dir = scheme.platlib

    # Record details of the files moved
    #   installed = files copied from the wheel to the destination
    #   changed = files changed while installing (scripts #! line typically)
    #   generated = files newly generated during the install (script wrappers)
    installed: Dict[RecordPath, RecordPath] = {}
    changed: Set[RecordPath] = set()
    generated: List[str] = []

    def record_installed(
        srcfile: RecordPath, destfile: str, modified: bool = False
    ) -> None:
        """Map archive RECORD paths to installation RECORD paths."""
        newpath = _fs_to_record_path(destfile, lib_dir)
        installed[srcfile] = newpath
        if modified:
            changed.add(newpath)

    def is_dir_path(path: RecordPath) -> bool:
        return path.endswith("/")

    def assert_no_path_traversal(dest_dir_path: str, target_path: str) -> None:
        if not is_within_directory(dest_dir_path, target_path):
            message = (
                "The wheel {!r} has a file {!r} trying to install"
                " outside the target directory {!r}"
            )
            raise InstallationError(
                message.format(wheel_path, target_path, dest_dir_path)
            )

    def root_scheme_file_maker(
        zip_file: ZipFile, dest: str
    ) -> Callable[[RecordPath], "File"]:
        def make_root_scheme_file(record_path: RecordPath) -> "File":
            normed_path = os.path.normpath(record_path)
            dest_path = os.path.join(dest, normed_path)
            assert_no_path_traversal(dest, dest_path)
            return ZipBackedFile(record_path, dest_path, zip_file)

        return make_root_scheme_file

    def data_scheme_file_maker(
        zip_file: ZipFile, scheme: Scheme
    ) -> Callable[[RecordPath], "File"]:
        scheme_paths = {key: getattr(scheme, key) for key in SCHEME_KEYS}

        def make_data_scheme_file(record_path: RecordPath) -> "File":
            normed_path = os.path.normpath(record_path)
            try:
                _, scheme_key, dest_subpath = normed_path.split(os.path.sep, 2)
            except ValueError:
                message = (
                    f"Unexpected file in {wheel_path}: {record_path!r}. .data directory"
                    " contents should be named like: '<scheme key>/<path>'."
                )
                raise InstallationError(message)

            try:
                scheme_path = scheme_paths[scheme_key]
            except KeyError:
                valid_scheme_keys = ", ".join(sorted(scheme_paths))
                message = (
                    f"Unknown scheme key used in {wheel_path}: {scheme_key} "
                    f"(for file {record_path!r}). .data directory contents "
                    f"should be in subdirectories named with a valid scheme "
                    f"key ({valid_scheme_keys})"
                )
                raise InstallationError(message)

            dest_path = os.path.join(scheme_path, dest_subpath)
            assert_no_path_traversal(scheme_path, dest_path)
            return ZipBackedFile(record_path, dest_path, zip_file)

        return make_data_scheme_file

    def is_data_scheme_path(path: RecordPath) -> bool:
        return path.split("/", 1)[0].endswith(".data")

    paths = cast(List[RecordPath], wheel_zip.namelist())
    file_paths = filterfalse(is_dir_path, paths)
    root_scheme_paths, data_scheme_paths = partition(is_data_scheme_path, file_paths)

    make_root_scheme_file = root_scheme_file_maker(wheel_zip, lib_dir)
    files: Iterator[File] = map(make_root_scheme_file, root_scheme_paths)

    def is_script_scheme_path(path: RecordPath) -> bool:
        parts = path.split("/", 2)
        return len(parts) > 2 and parts[0].endswith(".data") and parts[1] == "scripts"

    other_scheme_paths, script_scheme_paths = partition(
        is_script_scheme_path, data_scheme_paths
    )

    make_data_scheme_file = data_scheme_file_maker(wheel_zip, scheme)
    other_scheme_files = map(make_data_scheme_file, other_scheme_paths)
    files = chain(files, other_scheme_files)

    # Get the defined entry points
    distribution = get_wheel_distribution(
        FilesystemWheel(wheel_path),
        canonicalize_name(name),
    )
    console, gui = get_entrypoints(distribution)

    def is_entrypoint_wrapper(file: "File") -> bool:
        # EP, EP.exe and EP-script.py are scripts generated for
        # entry point EP by setuptools
        path = file.dest_path
        name = os.path.basename(path)
        if name.lower().endswith(".exe"):
            matchname = name[:-4]
        elif name.lower().endswith("-script.py"):
            matchname = name[:-10]
        elif name.lower().endswith(".pya"):
            matchname = name[:-4]
        else:
            matchname = name
        # Ignore setuptools-generated scripts
        return matchname in console or matchname in gui

    script_scheme_files: Iterator[File] = map(
        make_data_scheme_file, script_scheme_paths
    )
    script_scheme_files = filterfalse(is_entrypoint_wrapper, script_scheme_files)
    script_scheme_files = map(ScriptFile, script_scheme_files)
    files = chain(files, script_scheme_files)

    existing_parents = set()
    for file in files:
        # directory creation is lazy and after file filtering
        # to ensure we don't install empty dirs; empty dirs can't be
        # uninstalled.
        parent_dir = os.path.dirname(file.dest_path)
        if parent_dir not in existing_parents:
            ensure_dir(parent_dir)
            existing_parents.add(parent_dir)
        file.save()
        record_installed(file.src_record_path, file.dest_path, file.changed)

    def pyc_source_file_paths() -> Generator[str, None, None]:
        # We de-duplicate installation paths, since there can be overlap (e.g.
        # file in .data maps to same location as file in wheel root).
        # Sorting installation paths makes it easier to reproduce and debug
        # issues related to permissions on existing files.
        for installed_path in sorted(set(installed.values())):
            full_installed_path = os.path.join(lib_dir, installed_path)
            if not os.path.isfile(full_installed_path):
                continue
            if not full_installed_path.endswith(".py"):
                continue
            yield full_installed_path

    def pyc_output_path(path: str) -> str:
        """Return the path the pyc file would have been written to."""
        return importlib.util.cache_from_source(path)

    # Compile all of the pyc files for the installed files
    if pycompile:
        with contextlib.redirect_stdout(
            StreamWrapper.from_stream(sys.stdout)
        ) as stdout:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                for path in pyc_source_file_paths():
                    success = compileall.compile_file(path, force=True, quiet=True)
                    if success:
                        pyc_path = pyc_output_path(path)
                        assert os.path.exists(pyc_path)
                        pyc_record_path = cast(
                            "RecordPath", pyc_path.replace(os.path.sep, "/")
                        )
                        record_installed(pyc_record_path, pyc_path)
        logger.debug(stdout.getvalue())

    maker = PipScriptMaker(None, scheme.scripts)

    # Ensure old scripts are overwritten.
    # See https://github.com/pypa/pip/issues/1800
    maker.clobber = True

    # Ensure we don't generate any variants for scripts because this is almost
    # never what somebody wants.
    # See https://bitbucket.org/pypa/distlib/issue/35/
    maker.variants = {""}

    # This is required because otherwise distlib creates scripts that are not
    # executable.
    # See https://bitbucket.org/pypa/distlib/issue/32/
    maker.set_mode = True

    # Generate the console and GUI entry points specified in the wheel
    scripts_to_generate = get_console_script_specs(console)

    gui_scripts_to_generate = list(starmap("{} = {}".format, gui.items()))

    generated_console_scripts = maker.make_multiple(scripts_to_generate)
    generated.extend(generated_console_scripts)

    generated.extend(maker.make_multiple(gui_scripts_to_generate, {"gui": True}))

    if warn_script_location:
        msg = message_about_scripts_not_on_PATH(generated_console_scripts)
        if msg is not None:
            logger.warning(msg)

    generated_file_mode = 0o666 & ~current_umask()

    @contextlib.contextmanager
    def _generate_file(path: str, **kwargs: Any) -> Generator[BinaryIO, None, None]:
        with adjacent_tmp_file(path, **kwargs) as f:
            yield f
        os.chmod(f.name, generated_file_mode)
        replace(f.name, path)

    dest_info_dir = os.path.join(lib_dir, info_dir)

    # Record pip as the installer
    installer_path = os.path.join(dest_info_dir, "INSTALLER")
    with _generate_file(installer_path) as installer_file:
        installer_file.write(b"pip\n")
    generated.append(installer_path)

    # Record the PEP 610 direct URL reference
    if direct_url is not None:
        direct_url_path = os.path.join(dest_info_dir, DIRECT_URL_METADATA_NAME)
        with _generate_file(direct_url_path) as direct_url_file:
            direct_url_file.write(direct_url.to_json().encode("utf-8"))
        generated.append(direct_url_path)

    # Record the REQUESTED file
    if requested:
        requested_path = os.path.join(dest_info_dir, "REQUESTED")
        with open(requested_path, "wb"):
            pass
        generated.append(requested_path)

    record_text = distribution.read_text("RECORD")
    record_rows = list(csv.reader(record_text.splitlines()))

    rows = get_csv_rows_for_installed(
        record_rows,
        installed=installed,
        changed=changed,
        generated=generated,
        lib_dir=lib_dir,
    )

    # Record details of all files installed
    record_path = os.path.join(dest_info_dir, "RECORD")

    with _generate_file(record_path, **csv_io_kwargs("w")) as record_file:
        # Explicitly cast to typing.IO[str] as a workaround for the mypy error:
        # "writer" has incompatible type "BinaryIO"; expected "_Writer"
        writer = csv.writer(cast("IO[str]", record_file))
        writer.writerows(_normalized_outrows(rows))


@contextlib.contextmanager
def req_error_context(req_description: str) -> Generator[None, None, None]:
    try:
        yield
    except InstallationError as e:
        message = f"For req: {req_description}. {e.args[0]}"
        raise InstallationError(message) from e


def install_wheel(
    name: str,
    wheel_path: str,
    scheme: Scheme,
    req_description: str,
    pycompile: bool = True,
    warn_script_location: bool = True,
    direct_url: Optional[DirectUrl] = None,
    requested: bool = False,
) -> None:
    with ZipFile(wheel_path, allowZip64=True) as z:
        with req_error_context(req_description):
            _install_wheel(
                name=name,
                wheel_zip=z,
                wheel_path=wheel_path,
                scheme=scheme,
                pycompile=pycompile,
                warn_script_location=warn_script_location,
                direct_url=direct_url,
                requested=requested,
            )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_internal\operations\prepare.py ===
"""Prepares a distribution for installation"""

# The following comment should be removed at some point in the future.
# mypy: strict-optional=False

import mimetypes
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from pip._vendor.packaging.utils import canonicalize_name

from pip._internal.distributions import make_distribution_for_install_requirement
from pip._internal.distributions.installed import InstalledDistribution
from pip._internal.exceptions import (
    DirectoryUrlHashUnsupported,
    HashMismatch,
    HashUnpinned,
    InstallationError,
    MetadataInconsistent,
    NetworkConnectionError,
    VcsHashUnsupported,
)
from pip._internal.index.package_finder import PackageFinder
from pip._internal.metadata import BaseDistribution, get_metadata_distribution
from pip._internal.models.direct_url import ArchiveInfo
from pip._internal.models.link import Link
from pip._internal.models.wheel import Wheel
from pip._internal.network.download import BatchDownloader, Downloader
from pip._internal.network.lazy_wheel import (
    HTTPRangeRequestUnsupported,
    dist_from_wheel_url,
)
from pip._internal.network.session import PipSession
from pip._internal.operations.build.build_tracker import BuildTracker
from pip._internal.req.req_install import InstallRequirement
from pip._internal.utils._log import getLogger
from pip._internal.utils.direct_url_helpers import (
    direct_url_for_editable,
    direct_url_from_link,
)
from pip._internal.utils.hashes import Hashes, MissingHashes
from pip._internal.utils.logging import indent_log
from pip._internal.utils.misc import (
    display_path,
    hash_file,
    hide_url,
    redact_auth_from_requirement,
)
from pip._internal.utils.temp_dir import TempDirectory
from pip._internal.utils.unpacking import unpack_file
from pip._internal.vcs import vcs

logger = getLogger(__name__)


def _get_prepared_distribution(
    req: InstallRequirement,
    build_tracker: BuildTracker,
    finder: PackageFinder,
    build_isolation: bool,
    check_build_deps: bool,
) -> BaseDistribution:
    """Prepare a distribution for installation."""
    abstract_dist = make_distribution_for_install_requirement(req)
    tracker_id = abstract_dist.build_tracker_id
    if tracker_id is not None:
        with build_tracker.track(req, tracker_id):
            abstract_dist.prepare_distribution_metadata(
                finder, build_isolation, check_build_deps
            )
    return abstract_dist.get_metadata_distribution()


def unpack_vcs_link(link: Link, location: str, verbosity: int) -> None:
    vcs_backend = vcs.get_backend_for_scheme(link.scheme)
    assert vcs_backend is not None
    vcs_backend.unpack(location, url=hide_url(link.url), verbosity=verbosity)


@dataclass
class File:
    path: str
    content_type: Optional[str] = None

    def __post_init__(self) -> None:
        if self.content_type is None:
            # Try to guess the file's MIME type. If the system MIME tables
            # can't be loaded, give up.
            try:
                self.content_type = mimetypes.guess_type(self.path)[0]
            except OSError:
                pass


def get_http_url(
    link: Link,
    download: Downloader,
    download_dir: Optional[str] = None,
    hashes: Optional[Hashes] = None,
) -> File:
    temp_dir = TempDirectory(kind="unpack", globally_managed=True)
    # If a download dir is specified, is the file already downloaded there?
    already_downloaded_path = None
    if download_dir:
        already_downloaded_path = _check_download_dir(link, download_dir, hashes)

    if already_downloaded_path:
        from_path = already_downloaded_path
        content_type = None
    else:
        # let's download to a tmp dir
        from_path, content_type = download(link, temp_dir.path)
        if hashes:
            hashes.check_against_path(from_path)

    return File(from_path, content_type)


def get_file_url(
    link: Link, download_dir: Optional[str] = None, hashes: Optional[Hashes] = None
) -> File:
    """Get file and optionally check its hash."""
    # If a download dir is specified, is the file already there and valid?
    already_downloaded_path = None
    if download_dir:
        already_downloaded_path = _check_download_dir(link, download_dir, hashes)

    if already_downloaded_path:
        from_path = already_downloaded_path
    else:
        from_path = link.file_path

    # If --require-hashes is off, `hashes` is either empty, the
    # link's embedded hash, or MissingHashes; it is required to
    # match. If --require-hashes is on, we are satisfied by any
    # hash in `hashes` matching: a URL-based or an option-based
    # one; no internet-sourced hash will be in `hashes`.
    if hashes:
        hashes.check_against_path(from_path)
    return File(from_path, None)


def unpack_url(
    link: Link,
    location: str,
    download: Downloader,
    verbosity: int,
    download_dir: Optional[str] = None,
    hashes: Optional[Hashes] = None,
) -> Optional[File]:
    """Unpack link into location, downloading if required.

    :param hashes: A Hashes object, one of whose embedded hashes must match,
        or HashMismatch will be raised. If the Hashes is empty, no matches are
        required, and unhashable types of requirements (like VCS ones, which
        would ordinarily raise HashUnsupported) are allowed.
    """
    # non-editable vcs urls
    if link.is_vcs:
        unpack_vcs_link(link, location, verbosity=verbosity)
        return None

    assert not link.is_existing_dir()

    # file urls
    if link.is_file:
        file = get_file_url(link, download_dir, hashes=hashes)

    # http urls
    else:
        file = get_http_url(
            link,
            download,
            download_dir,
            hashes=hashes,
        )

    # unpack the archive to the build dir location. even when only downloading
    # archives, they have to be unpacked to parse dependencies, except wheels
    if not link.is_wheel:
        unpack_file(file.path, location, file.content_type)

    return file


def _check_download_dir(
    link: Link,
    download_dir: str,
    hashes: Optional[Hashes],
    warn_on_hash_mismatch: bool = True,
) -> Optional[str]:
    """Check download_dir for previously downloaded file with correct hash
    If a correct file is found return its path else None
    """
    download_path = os.path.join(download_dir, link.filename)

    if not os.path.exists(download_path):
        return None

    # If already downloaded, does its hash match?
    logger.info("File was already downloaded %s", download_path)
    if hashes:
        try:
            hashes.check_against_path(download_path)
        except HashMismatch:
            if warn_on_hash_mismatch:
                logger.warning(
                    "Previously-downloaded file %s has bad hash. Re-downloading.",
                    download_path,
                )
            os.unlink(download_path)
            return None
    return download_path


class RequirementPreparer:
    """Prepares a Requirement"""

    def __init__(
        self,
        build_dir: str,
        download_dir: Optional[str],
        src_dir: str,
        build_isolation: bool,
        check_build_deps: bool,
        build_tracker: BuildTracker,
        session: PipSession,
        progress_bar: str,
        finder: PackageFinder,
        require_hashes: bool,
        use_user_site: bool,
        lazy_wheel: bool,
        verbosity: int,
        legacy_resolver: bool,
        resume_retries: int,
    ) -> None:
        super().__init__()

        self.src_dir = src_dir
        self.build_dir = build_dir
        self.build_tracker = build_tracker
        self._session = session
        self._download = Downloader(session, progress_bar, resume_retries)
        self._batch_download = BatchDownloader(session, progress_bar, resume_retries)
        self.finder = finder

        # Where still-packed archives should be written to. If None, they are
        # not saved, and are deleted immediately after unpacking.
        self.download_dir = download_dir

        # Is build isolation allowed?
        self.build_isolation = build_isolation

        # Should check build dependencies?
        self.check_build_deps = check_build_deps

        # Should hash-checking be required?
        self.require_hashes = require_hashes

        # Should install in user site-packages?
        self.use_user_site = use_user_site

        # Should wheels be downloaded lazily?
        self.use_lazy_wheel = lazy_wheel

        # How verbose should underlying tooling be?
        self.verbosity = verbosity

        # Are we using the legacy resolver?
        self.legacy_resolver = legacy_resolver

        # Memoized downloaded files, as mapping of url: path.
        self._downloaded: Dict[str, str] = {}

        # Previous "header" printed for a link-based InstallRequirement
        self._previous_requirement_header = ("", "")

    def _log_preparing_link(self, req: InstallRequirement) -> None:
        """Provide context for the requirement being prepared."""
        if req.link.is_file and not req.is_wheel_from_cache:
            message = "Processing %s"
            information = str(display_path(req.link.file_path))
        else:
            message = "Collecting %s"
            information = redact_auth_from_requirement(req.req) if req.req else str(req)

        # If we used req.req, inject requirement source if available (this
        # would already be included if we used req directly)
        if req.req and req.comes_from:
            if isinstance(req.comes_from, str):
                comes_from: Optional[str] = req.comes_from
            else:
                comes_from = req.comes_from.from_path()
            if comes_from:
                information += f" (from {comes_from})"

        if (message, information) != self._previous_requirement_header:
            self._previous_requirement_header = (message, information)
            logger.info(message, information)

        if req.is_wheel_from_cache:
            with indent_log():
                logger.info("Using cached %s", req.link.filename)

    def _ensure_link_req_src_dir(
        self, req: InstallRequirement, parallel_builds: bool
    ) -> None:
        """Ensure source_dir of a linked InstallRequirement."""
        # Since source_dir is only set for editable requirements.
        if req.link.is_wheel:
            # We don't need to unpack wheels, so no need for a source
            # directory.
            return
        assert req.source_dir is None
        if req.link.is_existing_dir():
            # build local directories in-tree
            req.source_dir = req.link.file_path
            return

        # We always delete unpacked sdists after pip runs.
        req.ensure_has_source_dir(
            self.build_dir,
            autodelete=True,
            parallel_builds=parallel_builds,
        )
        req.ensure_pristine_source_checkout()

    def _get_linked_req_hashes(self, req: InstallRequirement) -> Hashes:
        # By the time this is called, the requirement's link should have
        # been checked so we can tell what kind of requirements req is
        # and raise some more informative errors than otherwise.
        # (For example, we can raise VcsHashUnsupported for a VCS URL
        # rather than HashMissing.)
        if not self.require_hashes:
            return req.hashes(trust_internet=True)

        # We could check these first 2 conditions inside unpack_url
        # and save repetition of conditions, but then we would
        # report less-useful error messages for unhashable
        # requirements, complaining that there's no hash provided.
        if req.link.is_vcs:
            raise VcsHashUnsupported()
        if req.link.is_existing_dir():
            raise DirectoryUrlHashUnsupported()

        # Unpinned packages are asking for trouble when a new version
        # is uploaded.  This isn't a security check, but it saves users
        # a surprising hash mismatch in the future.
        # file:/// URLs aren't pinnable, so don't complain about them
        # not being pinned.
        if not req.is_direct and not req.is_pinned:
            raise HashUnpinned()

        # If known-good hashes are missing for this requirement,
        # shim it with a facade object that will provoke hash
        # computation and then raise a HashMissing exception
        # showing the user what the hash should be.
        return req.hashes(trust_internet=False) or MissingHashes()

    def _fetch_metadata_only(
        self,
        req: InstallRequirement,
    ) -> Optional[BaseDistribution]:
        if self.legacy_resolver:
            logger.debug(
                "Metadata-only fetching is not used in the legacy resolver",
            )
            return None
        if self.require_hashes:
            logger.debug(
                "Metadata-only fetching is not used as hash checking is required",
            )
            return None
        # Try PEP 658 metadata first, then fall back to lazy wheel if unavailable.
        return self._fetch_metadata_using_link_data_attr(
            req
        ) or self._fetch_metadata_using_lazy_wheel(req.link)

    def _fetch_metadata_using_link_data_attr(
        self,
        req: InstallRequirement,
    ) -> Optional[BaseDistribution]:
        """Fetch metadata from the data-dist-info-metadata attribute, if possible."""
        # (1) Get the link to the metadata file, if provided by the backend.
        metadata_link = req.link.metadata_link()
        if metadata_link is None:
            return None
        assert req.req is not None
        logger.verbose(
            "Obtaining dependency information for %s from %s",
            req.req,
            metadata_link,
        )
        # (2) Download the contents of the METADATA file, separate from the dist itself.
        metadata_file = get_http_url(
            metadata_link,
            self._download,
            hashes=metadata_link.as_hashes(),
        )
        with open(metadata_file.path, "rb") as f:
            metadata_contents = f.read()
        # (3) Generate a dist just from those file contents.
        metadata_dist = get_metadata_distribution(
            metadata_contents,
            req.link.filename,
            req.req.name,
        )
        # (4) Ensure the Name: field from the METADATA file matches the name from the
        #     install requirement.
        #
        #     NB: raw_name will fall back to the name from the install requirement if
        #     the Name: field is not present, but it's noted in the raw_name docstring
        #     that that should NEVER happen anyway.
        if canonicalize_name(metadata_dist.raw_name) != canonicalize_name(req.req.name):
            raise MetadataInconsistent(
                req, "Name", req.req.name, metadata_dist.raw_name
            )
        return metadata_dist

    def _fetch_metadata_using_lazy_wheel(
        self,
        link: Link,
    ) -> Optional[BaseDistribution]:
        """Fetch metadata using lazy wheel, if possible."""
        # --use-feature=fast-deps must be provided.
        if not self.use_lazy_wheel:
            return None
        if link.is_file or not link.is_wheel:
            logger.debug(
                "Lazy wheel is not used as %r does not point to a remote wheel",
                link,
            )
            return None

        wheel = Wheel(link.filename)
        name = canonicalize_name(wheel.name)
        logger.info(
            "Obtaining dependency information from %s %s",
            name,
            wheel.version,
        )
        url = link.url.split("#", 1)[0]
        try:
            return dist_from_wheel_url(name, url, self._session)
        except HTTPRangeRequestUnsupported:
            logger.debug("%s does not support range requests", url)
            return None

    def _complete_partial_requirements(
        self,
        partially_downloaded_reqs: Iterable[InstallRequirement],
        parallel_builds: bool = False,
    ) -> None:
        """Download any requirements which were only fetched by metadata."""
        # Download to a temporary directory. These will be copied over as
        # needed for downstream 'download', 'wheel', and 'install' commands.
        temp_dir = TempDirectory(kind="unpack", globally_managed=True).path

        # Map each link to the requirement that owns it. This allows us to set
        # `req.local_file_path` on the appropriate requirement after passing
        # all the links at once into BatchDownloader.
        links_to_fully_download: Dict[Link, InstallRequirement] = {}
        for req in partially_downloaded_reqs:
            assert req.link
            links_to_fully_download[req.link] = req

        batch_download = self._batch_download(
            links_to_fully_download.keys(),
            temp_dir,
        )
        for link, (filepath, _) in batch_download:
            logger.debug("Downloading link %s to %s", link, filepath)
            req = links_to_fully_download[link]
            # Record the downloaded file path so wheel reqs can extract a Distribution
            # in .get_dist().
            req.local_file_path = filepath
            # Record that the file is downloaded so we don't do it again in
            # _prepare_linked_requirement().
            self._downloaded[req.link.url] = filepath

            # If this is an sdist, we need to unpack it after downloading, but the
            # .source_dir won't be set up until we are in _prepare_linked_requirement().
            # Add the downloaded archive to the install requirement to unpack after
            # preparing the source dir.
            if not req.is_wheel:
                req.needs_unpacked_archive(Path(filepath))

        # This step is necessary to ensure all lazy wheels are processed
        # successfully by the 'download', 'wheel', and 'install' commands.
        for req in partially_downloaded_reqs:
            self._prepare_linked_requirement(req, parallel_builds)

    def prepare_linked_requirement(
        self, req: InstallRequirement, parallel_builds: bool = False
    ) -> BaseDistribution:
        """Prepare a requirement to be obtained from req.link."""
        assert req.link
        self._log_preparing_link(req)
        with indent_log():
            # Check if the relevant file is already available
            # in the download directory
            file_path = None
            if self.download_dir is not None and req.link.is_wheel:
                hashes = self._get_linked_req_hashes(req)
                file_path = _check_download_dir(
                    req.link,
                    self.download_dir,
                    hashes,
                    # When a locally built wheel has been found in cache, we don't warn
                    # about re-downloading when the already downloaded wheel hash does
                    # not match. This is because the hash must be checked against the
                    # original link, not the cached link. It that case the already
                    # downloaded file will be removed and re-fetched from cache (which
                    # implies a hash check against the cache entry's origin.json).
                    warn_on_hash_mismatch=not req.is_wheel_from_cache,
                )

            if file_path is not None:
                # The file is already available, so mark it as downloaded
                self._downloaded[req.link.url] = file_path
            else:
                # The file is not available, attempt to fetch only metadata
                metadata_dist = self._fetch_metadata_only(req)
                if metadata_dist is not None:
                    req.needs_more_preparation = True
                    return metadata_dist

            # None of the optimizations worked, fully prepare the requirement
            return self._prepare_linked_requirement(req, parallel_builds)

    def prepare_linked_requirements_more(
        self, reqs: Iterable[InstallRequirement], parallel_builds: bool = False
    ) -> None:
        """Prepare linked requirements more, if needed."""
        reqs = [req for req in reqs if req.needs_more_preparation]
        for req in reqs:
            # Determine if any of these requirements were already downloaded.
            if self.download_dir is not None and req.link.is_wheel:
                hashes = self._get_linked_req_hashes(req)
                file_path = _check_download_dir(req.link, self.download_dir, hashes)
                if file_path is not None:
                    self._downloaded[req.link.url] = file_path
                    req.needs_more_preparation = False

        # Prepare requirements we found were already downloaded for some
        # reason. The other downloads will be completed separately.
        partially_downloaded_reqs: List[InstallRequirement] = []
        for req in reqs:
            if req.needs_more_preparation:
                partially_downloaded_reqs.append(req)
            else:
                self._prepare_linked_requirement(req, parallel_builds)

        # TODO: separate this part out from RequirementPreparer when the v1
        # resolver can be removed!
        self._complete_partial_requirements(
            partially_downloaded_reqs,
            parallel_builds=parallel_builds,
        )

    def _prepare_linked_requirement(
        self, req: InstallRequirement, parallel_builds: bool
    ) -> BaseDistribution:
        assert req.link
        link = req.link

        hashes = self._get_linked_req_hashes(req)

        if hashes and req.is_wheel_from_cache:
            assert req.download_info is not None
            assert link.is_wheel
            assert link.is_file
            # We need to verify hashes, and we have found the requirement in the cache
            # of locally built wheels.
            if (
                isinstance(req.download_info.info, ArchiveInfo)
                and req.download_info.info.hashes
                and hashes.has_one_of(req.download_info.info.hashes)
            ):
                # At this point we know the requirement was built from a hashable source
                # artifact, and we verified that the cache entry's hash of the original
                # artifact matches one of the hashes we expect. We don't verify hashes
                # against the cached wheel, because the wheel is not the original.
                hashes = None
            else:
                logger.warning(
                    "The hashes of the source archive found in cache entry "
                    "don't match, ignoring cached built wheel "
                    "and re-downloading source."
                )
                req.link = req.cached_wheel_source_link
                link = req.link

        self._ensure_link_req_src_dir(req, parallel_builds)

        if link.is_existing_dir():
            local_file = None
        elif link.url not in self._downloaded:
            try:
                local_file = unpack_url(
                    link,
                    req.source_dir,
                    self._download,
                    self.verbosity,
                    self.download_dir,
                    hashes,
                )
            except NetworkConnectionError as exc:
                raise InstallationError(
                    f"Could not install requirement {req} because of HTTP "
                    f"error {exc} for URL {link}"
                )
        else:
            file_path = self._downloaded[link.url]
            if hashes:
                hashes.check_against_path(file_path)
            local_file = File(file_path, content_type=None)

        # If download_info is set, we got it from the wheel cache.
        if req.download_info is None:
            # Editables don't go through this function (see
            # prepare_editable_requirement).
            assert not req.editable
            req.download_info = direct_url_from_link(link, req.source_dir)
            # Make sure we have a hash in download_info. If we got it as part of the
            # URL, it will have been verified and we can rely on it. Otherwise we
            # compute it from the downloaded file.
            # FIXME: https://github.com/pypa/pip/issues/11943
            if (
                isinstance(req.download_info.info, ArchiveInfo)
                and not req.download_info.info.hashes
                and local_file
            ):
                hash = hash_file(local_file.path)[0].hexdigest()
                # We populate info.hash for backward compatibility.
                # This will automatically populate info.hashes.
                req.download_info.info.hash = f"sha256={hash}"

        # For use in later processing,
        # preserve the file path on the requirement.
        if local_file:
            req.local_file_path = local_file.path

        dist = _get_prepared_distribution(
            req,
            self.build_tracker,
            self.finder,
            self.build_isolation,
            self.check_build_deps,
        )
        return dist

    def save_linked_requirement(self, req: InstallRequirement) -> None:
        assert self.download_dir is not None
        assert req.link is not None
        link = req.link
        if link.is_vcs or (link.is_existing_dir() and req.editable):
            # Make a .zip of the source_dir we already created.
            req.archive(self.download_dir)
            return

        if link.is_existing_dir():
            logger.debug(
                "Not copying link to destination directory "
                "since it is a directory: %s",
                link,
            )
            return
        if req.local_file_path is None:
            # No distribution was downloaded for this requirement.
            return

        download_location = os.path.join(self.download_dir, link.filename)
        if not os.path.exists(download_location):
            shutil.copy(req.local_file_path, download_location)
            download_path = display_path(download_location)
            logger.info("Saved %s", download_path)

    def prepare_editable_requirement(
        self,
        req: InstallRequirement,
    ) -> BaseDistribution:
        """Prepare an editable requirement."""
        assert req.editable, "cannot prepare a non-editable req as editable"

        logger.info("Obtaining %s", req)

        with indent_log():
            if self.require_hashes:
                raise InstallationError(
                    f"The editable requirement {req} cannot be installed when "
                    "requiring hashes, because there is no single file to "
                    "hash."
                )
            req.ensure_has_source_dir(self.src_dir)
            req.update_editable()
            assert req.source_dir
            req.download_info = direct_url_for_editable(req.unpacked_source_directory)

            dist = _get_prepared_distribution(
                req,
                self.build_tracker,
                self.finder,
                self.build_isolation,
                self.check_build_deps,
            )

            req.check_if_exists(self.use_user_site)

        return dist

    def prepare_installed_requirement(
        self,
        req: InstallRequirement,
        skip_reason: str,
    ) -> BaseDistribution:
        """Prepare an already-installed requirement."""
        assert req.satisfied_by, "req should have been satisfied but isn't"
        assert skip_reason is not None, (
            "did not get skip reason skipped but req.satisfied_by "
            f"is set to {req.satisfied_by}"
        )
        logger.info(
            "Requirement %s: %s (%s)", skip_reason, req, req.satisfied_by.version
        )
        with indent_log():
            if self.require_hashes:
                logger.debug(
                    "Since it is already installed, we are trusting this "
                    "package without checking its hash. To ensure a "
                    "completely repeatable environment, install into an "
                    "empty virtualenv."
                )
            return InstalledDistribution(req).get_metadata_distribution()

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\mechanics\functions.py ===
from sympy.utilities import dict_merge
from sympy.utilities.iterables import iterable
from sympy.physics.vector import (Dyadic, Vector, ReferenceFrame,
                                  Point, dynamicsymbols)
from sympy.physics.vector.printing import (vprint, vsprint, vpprint, vlatex,
                                           init_vprinting)
from sympy.physics.mechanics.particle import Particle
from sympy.physics.mechanics.rigidbody import RigidBody
from sympy.simplify.simplify import simplify
from sympy import Matrix, Mul, Derivative, sin, cos, tan, S
from sympy.core.function import AppliedUndef
from sympy.physics.mechanics.inertia import (inertia as _inertia,
    inertia_of_point_mass as _inertia_of_point_mass)
from sympy.utilities.exceptions import sympy_deprecation_warning

__all__ = ['linear_momentum',
           'angular_momentum',
           'kinetic_energy',
           'potential_energy',
           'Lagrangian',
           'mechanics_printing',
           'mprint',
           'msprint',
           'mpprint',
           'mlatex',
           'msubs',
           'find_dynamicsymbols']

# These are functions that we've moved and renamed during extracting the
# basic vector calculus code from the mechanics packages.

mprint = vprint
msprint = vsprint
mpprint = vpprint
mlatex = vlatex


def mechanics_printing(**kwargs):
    """
    Initializes time derivative printing for all SymPy objects in
    mechanics module.
    """

    init_vprinting(**kwargs)

mechanics_printing.__doc__ = init_vprinting.__doc__


def inertia(frame, ixx, iyy, izz, ixy=0, iyz=0, izx=0):
    sympy_deprecation_warning(
        """
        The inertia function has been moved.
        Import it from "sympy.physics.mechanics".
        """,
        deprecated_since_version="1.13",
        active_deprecations_target="moved-mechanics-functions"
    )
    return _inertia(frame, ixx, iyy, izz, ixy, iyz, izx)


def inertia_of_point_mass(mass, pos_vec, frame):
    sympy_deprecation_warning(
        """
        The inertia_of_point_mass function has been moved.
        Import it from "sympy.physics.mechanics".
        """,
        deprecated_since_version="1.13",
        active_deprecations_target="moved-mechanics-functions"
    )
    return _inertia_of_point_mass(mass, pos_vec, frame)


def linear_momentum(frame, *body):
    """Linear momentum of the system.

    Explanation
    ===========

    This function returns the linear momentum of a system of Particle's and/or
    RigidBody's. The linear momentum of a system is equal to the vector sum of
    the linear momentum of its constituents. Consider a system, S, comprised of
    a rigid body, A, and a particle, P. The linear momentum of the system, L,
    is equal to the vector sum of the linear momentum of the particle, L1, and
    the linear momentum of the rigid body, L2, i.e.

    L = L1 + L2

    Parameters
    ==========

    frame : ReferenceFrame
        The frame in which linear momentum is desired.
    body1, body2, body3... : Particle and/or RigidBody
        The body (or bodies) whose linear momentum is required.

    Examples
    ========

    >>> from sympy.physics.mechanics import Point, Particle, ReferenceFrame
    >>> from sympy.physics.mechanics import RigidBody, outer, linear_momentum
    >>> N = ReferenceFrame('N')
    >>> P = Point('P')
    >>> P.set_vel(N, 10 * N.x)
    >>> Pa = Particle('Pa', P, 1)
    >>> Ac = Point('Ac')
    >>> Ac.set_vel(N, 25 * N.y)
    >>> I = outer(N.x, N.x)
    >>> A = RigidBody('A', Ac, N, 20, (I, Ac))
    >>> linear_momentum(N, A, Pa)
    10*N.x + 500*N.y

    """

    if not isinstance(frame, ReferenceFrame):
        raise TypeError('Please specify a valid ReferenceFrame')
    else:
        linear_momentum_sys = Vector(0)
        for e in body:
            if isinstance(e, (RigidBody, Particle)):
                linear_momentum_sys += e.linear_momentum(frame)
            else:
                raise TypeError('*body must have only Particle or RigidBody')
    return linear_momentum_sys


def angular_momentum(point, frame, *body):
    """Angular momentum of a system.

    Explanation
    ===========

    This function returns the angular momentum of a system of Particle's and/or
    RigidBody's. The angular momentum of such a system is equal to the vector
    sum of the angular momentum of its constituents. Consider a system, S,
    comprised of a rigid body, A, and a particle, P. The angular momentum of
    the system, H, is equal to the vector sum of the angular momentum of the
    particle, H1, and the angular momentum of the rigid body, H2, i.e.

    H = H1 + H2

    Parameters
    ==========

    point : Point
        The point about which angular momentum of the system is desired.
    frame : ReferenceFrame
        The frame in which angular momentum is desired.
    body1, body2, body3... : Particle and/or RigidBody
        The body (or bodies) whose angular momentum is required.

    Examples
    ========

    >>> from sympy.physics.mechanics import Point, Particle, ReferenceFrame
    >>> from sympy.physics.mechanics import RigidBody, outer, angular_momentum
    >>> N = ReferenceFrame('N')
    >>> O = Point('O')
    >>> O.set_vel(N, 0 * N.x)
    >>> P = O.locatenew('P', 1 * N.x)
    >>> P.set_vel(N, 10 * N.x)
    >>> Pa = Particle('Pa', P, 1)
    >>> Ac = O.locatenew('Ac', 2 * N.y)
    >>> Ac.set_vel(N, 5 * N.y)
    >>> a = ReferenceFrame('a')
    >>> a.set_ang_vel(N, 10 * N.z)
    >>> I = outer(N.z, N.z)
    >>> A = RigidBody('A', Ac, a, 20, (I, Ac))
    >>> angular_momentum(O, N, Pa, A)
    10*N.z

    """

    if not isinstance(frame, ReferenceFrame):
        raise TypeError('Please enter a valid ReferenceFrame')
    if not isinstance(point, Point):
        raise TypeError('Please specify a valid Point')
    else:
        angular_momentum_sys = Vector(0)
        for e in body:
            if isinstance(e, (RigidBody, Particle)):
                angular_momentum_sys += e.angular_momentum(point, frame)
            else:
                raise TypeError('*body must have only Particle or RigidBody')
    return angular_momentum_sys


def kinetic_energy(frame, *body):
    """Kinetic energy of a multibody system.

    Explanation
    ===========

    This function returns the kinetic energy of a system of Particle's and/or
    RigidBody's. The kinetic energy of such a system is equal to the sum of
    the kinetic energies of its constituents. Consider a system, S, comprising
    a rigid body, A, and a particle, P. The kinetic energy of the system, T,
    is equal to the vector sum of the kinetic energy of the particle, T1, and
    the kinetic energy of the rigid body, T2, i.e.

    T = T1 + T2

    Kinetic energy is a scalar.

    Parameters
    ==========

    frame : ReferenceFrame
        The frame in which the velocity or angular velocity of the body is
        defined.
    body1, body2, body3... : Particle and/or RigidBody
        The body (or bodies) whose kinetic energy is required.

    Examples
    ========

    >>> from sympy.physics.mechanics import Point, Particle, ReferenceFrame
    >>> from sympy.physics.mechanics import RigidBody, outer, kinetic_energy
    >>> N = ReferenceFrame('N')
    >>> O = Point('O')
    >>> O.set_vel(N, 0 * N.x)
    >>> P = O.locatenew('P', 1 * N.x)
    >>> P.set_vel(N, 10 * N.x)
    >>> Pa = Particle('Pa', P, 1)
    >>> Ac = O.locatenew('Ac', 2 * N.y)
    >>> Ac.set_vel(N, 5 * N.y)
    >>> a = ReferenceFrame('a')
    >>> a.set_ang_vel(N, 10 * N.z)
    >>> I = outer(N.z, N.z)
    >>> A = RigidBody('A', Ac, a, 20, (I, Ac))
    >>> kinetic_energy(N, Pa, A)
    350

    """

    if not isinstance(frame, ReferenceFrame):
        raise TypeError('Please enter a valid ReferenceFrame')
    ke_sys = S.Zero
    for e in body:
        if isinstance(e, (RigidBody, Particle)):
            ke_sys += e.kinetic_energy(frame)
        else:
            raise TypeError('*body must have only Particle or RigidBody')
    return ke_sys


def potential_energy(*body):
    """Potential energy of a multibody system.

    Explanation
    ===========

    This function returns the potential energy of a system of Particle's and/or
    RigidBody's. The potential energy of such a system is equal to the sum of
    the potential energy of its constituents. Consider a system, S, comprising
    a rigid body, A, and a particle, P. The potential energy of the system, V,
    is equal to the vector sum of the potential energy of the particle, V1, and
    the potential energy of the rigid body, V2, i.e.

    V = V1 + V2

    Potential energy is a scalar.

    Parameters
    ==========

    body1, body2, body3... : Particle and/or RigidBody
        The body (or bodies) whose potential energy is required.

    Examples
    ========

    >>> from sympy.physics.mechanics import Point, Particle, ReferenceFrame
    >>> from sympy.physics.mechanics import RigidBody, outer, potential_energy
    >>> from sympy import symbols
    >>> M, m, g, h = symbols('M m g h')
    >>> N = ReferenceFrame('N')
    >>> O = Point('O')
    >>> O.set_vel(N, 0 * N.x)
    >>> P = O.locatenew('P', 1 * N.x)
    >>> Pa = Particle('Pa', P, m)
    >>> Ac = O.locatenew('Ac', 2 * N.y)
    >>> a = ReferenceFrame('a')
    >>> I = outer(N.z, N.z)
    >>> A = RigidBody('A', Ac, a, M, (I, Ac))
    >>> Pa.potential_energy = m * g * h
    >>> A.potential_energy = M * g * h
    >>> potential_energy(Pa, A)
    M*g*h + g*h*m

    """

    pe_sys = S.Zero
    for e in body:
        if isinstance(e, (RigidBody, Particle)):
            pe_sys += e.potential_energy
        else:
            raise TypeError('*body must have only Particle or RigidBody')
    return pe_sys


def gravity(acceleration, *bodies):
    from sympy.physics.mechanics.loads import gravity as _gravity
    sympy_deprecation_warning(
        """
        The gravity function has been moved.
        Import it from "sympy.physics.mechanics.loads".
        """,
        deprecated_since_version="1.13",
        active_deprecations_target="moved-mechanics-functions"
    )
    return _gravity(acceleration, *bodies)


def center_of_mass(point, *bodies):
    """
    Returns the position vector from the given point to the center of mass
    of the given bodies(particles or rigidbodies).

    Example
    =======

    >>> from sympy import symbols, S
    >>> from sympy.physics.vector import Point
    >>> from sympy.physics.mechanics import Particle, ReferenceFrame, RigidBody, outer
    >>> from sympy.physics.mechanics.functions import center_of_mass
    >>> a = ReferenceFrame('a')
    >>> m = symbols('m', real=True)
    >>> p1 = Particle('p1', Point('p1_pt'), S(1))
    >>> p2 = Particle('p2', Point('p2_pt'), S(2))
    >>> p3 = Particle('p3', Point('p3_pt'), S(3))
    >>> p4 = Particle('p4', Point('p4_pt'), m)
    >>> b_f = ReferenceFrame('b_f')
    >>> b_cm = Point('b_cm')
    >>> mb = symbols('mb')
    >>> b = RigidBody('b', b_cm, b_f, mb, (outer(b_f.x, b_f.x), b_cm))
    >>> p2.point.set_pos(p1.point, a.x)
    >>> p3.point.set_pos(p1.point, a.x + a.y)
    >>> p4.point.set_pos(p1.point, a.y)
    >>> b.masscenter.set_pos(p1.point, a.y + a.z)
    >>> point_o=Point('o')
    >>> point_o.set_pos(p1.point, center_of_mass(p1.point, p1, p2, p3, p4, b))
    >>> expr = 5/(m + mb + 6)*a.x + (m + mb + 3)/(m + mb + 6)*a.y + mb/(m + mb + 6)*a.z
    >>> point_o.pos_from(p1.point)
    5/(m + mb + 6)*a.x + (m + mb + 3)/(m + mb + 6)*a.y + mb/(m + mb + 6)*a.z

    """
    if not bodies:
        raise TypeError("No bodies(instances of Particle or Rigidbody) were passed.")

    total_mass = 0
    vec = Vector(0)
    for i in bodies:
        total_mass += i.mass

        masscenter = getattr(i, 'masscenter', None)
        if masscenter is None:
            masscenter = i.point
        vec += i.mass*masscenter.pos_from(point)

    return vec/total_mass


def Lagrangian(frame, *body):
    """Lagrangian of a multibody system.

    Explanation
    ===========

    This function returns the Lagrangian of a system of Particle's and/or
    RigidBody's. The Lagrangian of such a system is equal to the difference
    between the kinetic energies and potential energies of its constituents. If
    T and V are the kinetic and potential energies of a system then it's
    Lagrangian, L, is defined as

    L = T - V

    The Lagrangian is a scalar.

    Parameters
    ==========

    frame : ReferenceFrame
        The frame in which the velocity or angular velocity of the body is
        defined to determine the kinetic energy.

    body1, body2, body3... : Particle and/or RigidBody
        The body (or bodies) whose Lagrangian is required.

    Examples
    ========

    >>> from sympy.physics.mechanics import Point, Particle, ReferenceFrame
    >>> from sympy.physics.mechanics import RigidBody, outer, Lagrangian
    >>> from sympy import symbols
    >>> M, m, g, h = symbols('M m g h')
    >>> N = ReferenceFrame('N')
    >>> O = Point('O')
    >>> O.set_vel(N, 0 * N.x)
    >>> P = O.locatenew('P', 1 * N.x)
    >>> P.set_vel(N, 10 * N.x)
    >>> Pa = Particle('Pa', P, 1)
    >>> Ac = O.locatenew('Ac', 2 * N.y)
    >>> Ac.set_vel(N, 5 * N.y)
    >>> a = ReferenceFrame('a')
    >>> a.set_ang_vel(N, 10 * N.z)
    >>> I = outer(N.z, N.z)
    >>> A = RigidBody('A', Ac, a, 20, (I, Ac))
    >>> Pa.potential_energy = m * g * h
    >>> A.potential_energy = M * g * h
    >>> Lagrangian(N, Pa, A)
    -M*g*h - g*h*m + 350

    """

    if not isinstance(frame, ReferenceFrame):
        raise TypeError('Please supply a valid ReferenceFrame')
    for e in body:
        if not isinstance(e, (RigidBody, Particle)):
            raise TypeError('*body must have only Particle or RigidBody')
    return kinetic_energy(frame, *body) - potential_energy(*body)


def find_dynamicsymbols(expression, exclude=None, reference_frame=None):
    """Find all dynamicsymbols in expression.

    Explanation
    ===========

    If the optional ``exclude`` kwarg is used, only dynamicsymbols
    not in the iterable ``exclude`` are returned.
    If we intend to apply this function on a vector, the optional
    ``reference_frame`` is also used to inform about the corresponding frame
    with respect to which the dynamic symbols of the given vector is to be
    determined.

    Parameters
    ==========

    expression : SymPy expression

    exclude : iterable of dynamicsymbols, optional

    reference_frame : ReferenceFrame, optional
        The frame with respect to which the dynamic symbols of the
        given vector is to be determined.

    Examples
    ========

    >>> from sympy.physics.mechanics import dynamicsymbols, find_dynamicsymbols
    >>> from sympy.physics.mechanics import ReferenceFrame
    >>> x, y = dynamicsymbols('x, y')
    >>> expr = x + x.diff()*y
    >>> find_dynamicsymbols(expr)
    {x(t), y(t), Derivative(x(t), t)}
    >>> find_dynamicsymbols(expr, exclude=[x, y])
    {Derivative(x(t), t)}
    >>> a, b, c = dynamicsymbols('a, b, c')
    >>> A = ReferenceFrame('A')
    >>> v = a * A.x + b * A.y + c * A.z
    >>> find_dynamicsymbols(v, reference_frame=A)
    {a(t), b(t), c(t)}

    """
    t_set = {dynamicsymbols._t}
    if exclude:
        if iterable(exclude):
            exclude_set = set(exclude)
        else:
            raise TypeError("exclude kwarg must be iterable")
    else:
        exclude_set = set()
    if isinstance(expression, Vector):
        if reference_frame is None:
            raise ValueError("You must provide reference_frame when passing a "
                             "vector expression, got %s." % reference_frame)
        else:
            expression = expression.to_matrix(reference_frame)
    return {i for i in expression.atoms(AppliedUndef, Derivative) if
            i.free_symbols == t_set} - exclude_set


def msubs(expr, *sub_dicts, smart=False, **kwargs):
    """A custom subs for use on expressions derived in physics.mechanics.

    Traverses the expression tree once, performing the subs found in sub_dicts.
    Terms inside ``Derivative`` expressions are ignored:

    Examples
    ========

    >>> from sympy.physics.mechanics import dynamicsymbols, msubs
    >>> x = dynamicsymbols('x')
    >>> msubs(x.diff() + x, {x: 1})
    Derivative(x(t), t) + 1

    Note that sub_dicts can be a single dictionary, or several dictionaries:

    >>> x, y, z = dynamicsymbols('x, y, z')
    >>> sub1 = {x: 1, y: 2}
    >>> sub2 = {z: 3, x.diff(): 4}
    >>> msubs(x.diff() + x + y + z, sub1, sub2)
    10

    If smart=True (default False), also checks for conditions that may result
    in ``nan``, but if simplified would yield a valid expression. For example:

    >>> from sympy import sin, tan
    >>> (sin(x)/tan(x)).subs(x, 0)
    nan
    >>> msubs(sin(x)/tan(x), {x: 0}, smart=True)
    1

    It does this by first replacing all ``tan`` with ``sin/cos``. Then each
    node is traversed. If the node is a fraction, subs is first evaluated on
    the denominator. If this results in 0, simplification of the entire
    fraction is attempted. Using this selective simplification, only
    subexpressions that result in 1/0 are targeted, resulting in faster
    performance.

    """

    sub_dict = dict_merge(*sub_dicts)
    if smart:
        func = _smart_subs
    elif hasattr(expr, 'msubs'):
        return expr.msubs(sub_dict)
    else:
        func = lambda expr, sub_dict: _crawl(expr, _sub_func, sub_dict)
    if isinstance(expr, (Matrix, Vector, Dyadic)):
        return expr.applyfunc(lambda x: func(x, sub_dict))
    else:
        return func(expr, sub_dict)


def _crawl(expr, func, *args, **kwargs):
    """Crawl the expression tree, and apply func to every node."""
    val = func(expr, *args, **kwargs)
    if val is not None:
        return val
    new_args = (_crawl(arg, func, *args, **kwargs) for arg in expr.args)
    return expr.func(*new_args)


def _sub_func(expr, sub_dict):
    """Perform direct matching substitution, ignoring derivatives."""
    if expr in sub_dict:
        return sub_dict[expr]
    elif not expr.args or expr.is_Derivative:
        return expr


def _tan_repl_func(expr):
    """Replace tan with sin/cos."""
    if isinstance(expr, tan):
        return sin(*expr.args) / cos(*expr.args)
    elif not expr.args or expr.is_Derivative:
        return expr


def _smart_subs(expr, sub_dict):
    """Performs subs, checking for conditions that may result in `nan` or
    `oo`, and attempts to simplify them out.

    The expression tree is traversed twice, and the following steps are
    performed on each expression node:
    - First traverse:
        Replace all `tan` with `sin/cos`.
    - Second traverse:
        If node is a fraction, check if the denominator evaluates to 0.
        If so, attempt to simplify it out. Then if node is in sub_dict,
        sub in the corresponding value.

    """
    expr = _crawl(expr, _tan_repl_func)

    def _recurser(expr, sub_dict):
        # Decompose the expression into num, den
        num, den = _fraction_decomp(expr)
        if den != 1:
            # If there is a non trivial denominator, we need to handle it
            denom_subbed = _recurser(den, sub_dict)
            if denom_subbed.evalf() == 0:
                # If denom is 0 after this, attempt to simplify the bad expr
                expr = simplify(expr)
            else:
                # Expression won't result in nan, find numerator
                num_subbed = _recurser(num, sub_dict)
                return num_subbed / denom_subbed
        # We have to crawl the tree manually, because `expr` may have been
        # modified in the simplify step. First, perform subs as normal:
        val = _sub_func(expr, sub_dict)
        if val is not None:
            return val
        new_args = (_recurser(arg, sub_dict) for arg in expr.args)
        return expr.func(*new_args)
    return _recurser(expr, sub_dict)


def _fraction_decomp(expr):
    """Return num, den such that expr = num/den."""
    if not isinstance(expr, Mul):
        return expr, 1
    num = []
    den = []
    for a in expr.args:
        if a.is_Pow and a.args[1] < 0:
            den.append(1 / a)
        else:
            num.append(a)
    if not den:
        return expr, 1
    num = Mul(*num)
    den = Mul(*den)
    return num, den


def _f_list_parser(fl, ref_frame):
    """Parses the provided forcelist composed of items
    of the form (obj, force).
    Returns a tuple containing:
        vel_list: The velocity (ang_vel for Frames, vel for Points) in
                the provided reference frame.
        f_list: The forces.

    Used internally in the KanesMethod and LagrangesMethod classes.

    """
    def flist_iter():
        for pair in fl:
            obj, force = pair
            if isinstance(obj, ReferenceFrame):
                yield obj.ang_vel_in(ref_frame), force
            elif isinstance(obj, Point):
                yield obj.vel(ref_frame), force
            else:
                raise TypeError('First entry in each forcelist pair must '
                                'be a point or frame.')

    if not fl:
        vel_list, f_list = (), ()
    else:
        unzip = lambda l: list(zip(*l)) if l[0] else [(), ()]
        vel_list, f_list = unzip(list(flist_iter()))
    return vel_list, f_list


def _validate_coordinates(coordinates=None, speeds=None, check_duplicates=True,
                          is_dynamicsymbols=True, u_auxiliary=None):
    """Validate the generalized coordinates and generalized speeds.

    Parameters
    ==========
    coordinates : iterable, optional
        Generalized coordinates to be validated.
    speeds : iterable, optional
        Generalized speeds to be validated.
    check_duplicates : bool, optional
        Checks if there are duplicates in the generalized coordinates and
        generalized speeds. If so it will raise a ValueError. The default is
        True.
    is_dynamicsymbols : iterable, optional
        Checks if all the generalized coordinates and generalized speeds are
        dynamicsymbols. If any is not a dynamicsymbol, a ValueError will be
        raised. The default is True.
    u_auxiliary : iterable, optional
        Auxiliary generalized speeds to be validated.

    """
    t_set = {dynamicsymbols._t}
    # Convert input to iterables
    if coordinates is None:
        coordinates = []
    elif not iterable(coordinates):
        coordinates = [coordinates]
    if speeds is None:
        speeds = []
    elif not iterable(speeds):
        speeds = [speeds]
    if u_auxiliary is None:
        u_auxiliary = []
    elif not iterable(u_auxiliary):
        u_auxiliary = [u_auxiliary]

    msgs = []
    if check_duplicates:  # Check for duplicates
        seen = set()
        coord_duplicates = {x for x in coordinates if x in seen or seen.add(x)}
        seen = set()
        speed_duplicates = {x for x in speeds if x in seen or seen.add(x)}
        seen = set()
        aux_duplicates = {x for x in u_auxiliary if x in seen or seen.add(x)}
        overlap_coords = set(coordinates).intersection(speeds)
        overlap_aux = set(coordinates).union(speeds).intersection(u_auxiliary)
        if coord_duplicates:
            msgs.append(f'The generalized coordinates {coord_duplicates} are '
                        f'duplicated, all generalized coordinates should be '
                        f'unique.')
        if speed_duplicates:
            msgs.append(f'The generalized speeds {speed_duplicates} are '
                        f'duplicated, all generalized speeds should be unique.')
        if aux_duplicates:
            msgs.append(f'The auxiliary speeds {aux_duplicates} are duplicated,'
                        f' all auxiliary speeds should be unique.')
        if overlap_coords:
            msgs.append(f'{overlap_coords} are defined as both generalized '
                        f'coordinates and generalized speeds.')
        if overlap_aux:
            msgs.append(f'The auxiliary speeds {overlap_aux} are also defined '
                        f'as generalized coordinates or generalized speeds.')
    if is_dynamicsymbols:  # Check whether all coordinates are dynamicsymbols
        for coordinate in coordinates:
            if not (isinstance(coordinate, (AppliedUndef, Derivative)) and
                    coordinate.free_symbols == t_set):
                msgs.append(f'Generalized coordinate "{coordinate}" is not a '
                            f'dynamicsymbol.')
        for speed in speeds:
            if not (isinstance(speed, (AppliedUndef, Derivative)) and
                    speed.free_symbols == t_set):
                msgs.append(
                    f'Generalized speed "{speed}" is not a dynamicsymbol.')
        for aux in u_auxiliary:
            if not (isinstance(aux, (AppliedUndef, Derivative)) and
                    aux.free_symbols == t_set):
                msgs.append(
                    f'Auxiliary speed "{aux}" is not a dynamicsymbol.')
    if msgs:
        raise ValueError('\n'.join(msgs))


def _parse_linear_solver(linear_solver):
    """Helper function to retrieve a specified linear solver."""
    if callable(linear_solver):
        return linear_solver
    return lambda A, b: Matrix.solve(A, b, method=linear_solver)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\wtforms\validators.py ===
import ipaddress
import math
import re
import uuid

__all__ = (
    "DataRequired",
    "data_required",
    "Email",
    "email",
    "EqualTo",
    "equal_to",
    "IPAddress",
    "ip_address",
    "InputRequired",
    "input_required",
    "Length",
    "length",
    "NumberRange",
    "number_range",
    "Optional",
    "optional",
    "Regexp",
    "regexp",
    "URL",
    "url",
    "AnyOf",
    "any_of",
    "NoneOf",
    "none_of",
    "MacAddress",
    "mac_address",
    "UUID",
    "ValidationError",
    "StopValidation",
    "readonly",
    "ReadOnly",
    "disabled",
    "Disabled",
)


class ValidationError(ValueError):
    """
    Raised when a validator fails to validate its input.
    """

    def __init__(self, message="", *args, **kwargs):
        ValueError.__init__(self, message, *args, **kwargs)


class StopValidation(Exception):
    """
    Causes the validation chain to stop.

    If StopValidation is raised, no more validators in the validation chain are
    called. If raised with a message, the message will be added to the errors
    list.
    """

    def __init__(self, message="", *args, **kwargs):
        Exception.__init__(self, message, *args, **kwargs)


class EqualTo:
    """
    Compares the values of two fields.

    :param fieldname:
        The name of the other field to compare to.
    :param message:
        Error message to raise in case of a validation error. Can be
        interpolated with `%(other_label)s` and `%(other_name)s` to provide a
        more helpful error.
    """

    def __init__(self, fieldname, message=None):
        self.fieldname = fieldname
        self.message = message

    def __call__(self, form, field):
        try:
            other = form[self.fieldname]
        except KeyError as exc:
            raise ValidationError(
                field.gettext("Invalid field name '%s'.") % self.fieldname
            ) from exc
        if field.data == other.data:
            return

        d = {
            "other_label": hasattr(other, "label")
            and other.label.text
            or self.fieldname,
            "other_name": self.fieldname,
        }
        message = self.message
        if message is None:
            message = field.gettext("Field must be equal to %(other_name)s.")

        raise ValidationError(message % d)


class Length:
    """
    Validates the length of a string.

    :param min:
        The minimum required length of the string. If not provided, minimum
        length will not be checked.
    :param max:
        The maximum length of the string. If not provided, maximum length
        will not be checked.
    :param message:
        Error message to raise in case of a validation error. Can be
        interpolated using `%(min)d` and `%(max)d` if desired. Useful defaults
        are provided depending on the existence of min and max.

    When supported, sets the `minlength` and `maxlength` attributes on widgets.
    """

    def __init__(self, min=-1, max=-1, message=None):
        assert (
            min != -1 or max != -1
        ), "At least one of `min` or `max` must be specified."
        assert max == -1 or min <= max, "`min` cannot be more than `max`."
        self.min = min
        self.max = max
        self.message = message
        self.field_flags = {}
        if self.min != -1:
            self.field_flags["minlength"] = self.min
        if self.max != -1:
            self.field_flags["maxlength"] = self.max

    def __call__(self, form, field):
        length = field.data and len(field.data) or 0
        if length >= self.min and (self.max == -1 or length <= self.max):
            return

        if self.message is not None:
            message = self.message

        elif self.max == -1:
            message = field.ngettext(
                "Field must be at least %(min)d character long.",
                "Field must be at least %(min)d characters long.",
                self.min,
            )
        elif self.min == -1:
            message = field.ngettext(
                "Field cannot be longer than %(max)d character.",
                "Field cannot be longer than %(max)d characters.",
                self.max,
            )
        elif self.min == self.max:
            message = field.ngettext(
                "Field must be exactly %(max)d character long.",
                "Field must be exactly %(max)d characters long.",
                self.max,
            )
        else:
            message = field.gettext(
                "Field must be between %(min)d and %(max)d characters long."
            )

        raise ValidationError(message % dict(min=self.min, max=self.max, length=length))


class NumberRange:
    """
    Validates that a number is of a minimum and/or maximum value, inclusive.
    This will work with any comparable number type, such as floats and
    decimals, not just integers.

    :param min:
        The minimum required value of the number. If not provided, minimum
        value will not be checked.
    :param max:
        The maximum value of the number. If not provided, maximum value
        will not be checked.
    :param message:
        Error message to raise in case of a validation error. Can be
        interpolated using `%(min)s` and `%(max)s` if desired. Useful defaults
        are provided depending on the existence of min and max.

    When supported, sets the `min` and `max` attributes on widgets.
    """

    def __init__(self, min=None, max=None, message=None):
        self.min = min
        self.max = max
        self.message = message
        self.field_flags = {}
        if self.min is not None:
            self.field_flags["min"] = self.min
        if self.max is not None:
            self.field_flags["max"] = self.max

    def __call__(self, form, field):
        data = field.data
        if (
            data is not None
            and not math.isnan(data)
            and (self.min is None or data >= self.min)
            and (self.max is None or data <= self.max)
        ):
            return

        if self.message is not None:
            message = self.message

        # we use %(min)s interpolation to support floats, None, and
        # Decimals without throwing a formatting exception.
        elif self.max is None:
            message = field.gettext("Number must be at least %(min)s.")

        elif self.min is None:
            message = field.gettext("Number must be at most %(max)s.")

        else:
            message = field.gettext("Number must be between %(min)s and %(max)s.")

        raise ValidationError(message % dict(min=self.min, max=self.max))


class Optional:
    """
    Allows empty input and stops the validation chain from continuing.

    If input is empty, also removes prior errors (such as processing errors)
    from the field.

    :param strip_whitespace:
        If True (the default) also stop the validation chain on input which
        consists of only whitespace.

    Sets the `optional` attribute on widgets.
    """

    def __init__(self, strip_whitespace=True):
        if strip_whitespace:
            self.string_check = lambda s: s.strip()
        else:
            self.string_check = lambda s: s

        self.field_flags = {"optional": True}

    def __call__(self, form, field):
        if (
            not field.raw_data
            or isinstance(field.raw_data[0], str)
            and not self.string_check(field.raw_data[0])
        ):
            field.errors[:] = []
            raise StopValidation()


class DataRequired:
    """
    Checks the field's data is 'truthy' otherwise stops the validation chain.

    This validator checks that the ``data`` attribute on the field is a 'true'
    value (effectively, it does ``if field.data``.) Furthermore, if the data
    is a string type, a string containing only whitespace characters is
    considered false.

    If the data is empty, also removes prior errors (such as processing errors)
    from the field.

    **NOTE** this validator used to be called `Required` but the way it behaved
    (requiring coerced data, not input data) meant it functioned in a way
    which was not symmetric to the `Optional` validator and furthermore caused
    confusion with certain fields which coerced data to 'falsey' values like
    ``0``, ``Decimal(0)``, ``time(0)`` etc. Unless a very specific reason
    exists, we recommend using the :class:`InputRequired` instead.

    :param message:
        Error message to raise in case of a validation error.

    Sets the `required` attribute on widgets.
    """

    def __init__(self, message=None):
        self.message = message
        self.field_flags = {"required": True}

    def __call__(self, form, field):
        if field.data and (not isinstance(field.data, str) or field.data.strip()):
            return

        if self.message is None:
            message = field.gettext("This field is required.")
        else:
            message = self.message

        field.errors[:] = []
        raise StopValidation(message)


class InputRequired:
    """
    Validates that input was provided for this field.

    Note there is a distinction between this and DataRequired in that
    InputRequired looks that form-input data was provided, and DataRequired
    looks at the post-coercion data. This means that this validator only checks
    whether non-empty data was sent, not whether non-empty data was coerced
    from that data. Initially populated data is not considered sent.

    Sets the `required` attribute on widgets.
    """

    def __init__(self, message=None):
        self.message = message
        self.field_flags = {"required": True}

    def __call__(self, form, field):
        if field.raw_data and field.raw_data[0]:
            return

        if self.message is None:
            message = field.gettext("This field is required.")
        else:
            message = self.message

        field.errors[:] = []
        raise StopValidation(message)


class Regexp:
    """
    Validates the field against a user provided regexp.

    :param regex:
        The regular expression string to use. Can also be a compiled regular
        expression pattern.
    :param flags:
        The regexp flags to use, for example re.IGNORECASE. Ignored if
        `regex` is not a string.
    :param message:
        Error message to raise in case of a validation error.
    """

    def __init__(self, regex, flags=0, message=None):
        if isinstance(regex, str):
            regex = re.compile(regex, flags)
        self.regex = regex
        self.message = message

    def __call__(self, form, field, message=None):
        match = self.regex.match(field.data or "")
        if match:
            return match

        if message is None:
            if self.message is None:
                message = field.gettext("Invalid input.")
            else:
                message = self.message

        raise ValidationError(message)


class Email:
    """
    Validates an email address. Requires email_validator package to be
    installed. For ex: pip install wtforms[email].

    :param message:
        Error message to raise in case of a validation error.
    :param granular_message:
        Use validation failed message from email_validator library
        (Default False).
    :param check_deliverability:
        Perform domain name resolution check (Default False).
    :param allow_smtputf8:
        Fail validation for addresses that would require SMTPUTF8
        (Default True).
    :param allow_empty_local:
        Allow an empty local part (i.e. @example.com), e.g. for validating
        Postfix aliases (Default False).
    """

    def __init__(
        self,
        message=None,
        granular_message=False,
        check_deliverability=False,
        allow_smtputf8=True,
        allow_empty_local=False,
    ):
        self.message = message
        self.granular_message = granular_message
        self.check_deliverability = check_deliverability
        self.allow_smtputf8 = allow_smtputf8
        self.allow_empty_local = allow_empty_local

    def __call__(self, form, field):
        try:
            import email_validator
        except ImportError as exc:  # pragma: no cover
            raise Exception(
                "Install 'email_validator' for email validation support."
            ) from exc

        try:
            if field.data is None:
                raise email_validator.EmailNotValidError()
            email_validator.validate_email(
                field.data,
                check_deliverability=self.check_deliverability,
                allow_smtputf8=self.allow_smtputf8,
                allow_empty_local=self.allow_empty_local,
            )
        except email_validator.EmailNotValidError as e:
            message = self.message
            if message is None:
                if self.granular_message:
                    message = field.gettext(e)
                else:
                    message = field.gettext("Invalid email address.")
            raise ValidationError(message) from e


class IPAddress:
    """
    Validates an IP address.

    :param ipv4:
        If True, accept IPv4 addresses as valid (default True)
    :param ipv6:
        If True, accept IPv6 addresses as valid (default False)
    :param message:
        Error message to raise in case of a validation error.
    """

    def __init__(self, ipv4=True, ipv6=False, message=None):
        if not ipv4 and not ipv6:
            raise ValueError(
                "IP Address Validator must have at least one of ipv4 or ipv6 enabled."
            )
        self.ipv4 = ipv4
        self.ipv6 = ipv6
        self.message = message

    def __call__(self, form, field):
        value = field.data
        valid = False
        if value:
            valid = (self.ipv4 and self.check_ipv4(value)) or (
                self.ipv6 and self.check_ipv6(value)
            )

        if valid:
            return

        message = self.message
        if message is None:
            message = field.gettext("Invalid IP address.")
        raise ValidationError(message)

    @classmethod
    def check_ipv4(cls, value):
        try:
            address = ipaddress.ip_address(value)
        except ValueError:
            return False

        if not isinstance(address, ipaddress.IPv4Address):
            return False

        return True

    @classmethod
    def check_ipv6(cls, value):
        try:
            address = ipaddress.ip_address(value)
        except ValueError:
            return False

        if not isinstance(address, ipaddress.IPv6Address):
            return False

        return True


class MacAddress(Regexp):
    """
    Validates a MAC address.

    :param message:
        Error message to raise in case of a validation error.
    """

    def __init__(self, message=None):
        pattern = r"^(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}$"
        super().__init__(pattern, message=message)

    def __call__(self, form, field):
        message = self.message
        if message is None:
            message = field.gettext("Invalid Mac address.")

        super().__call__(form, field, message)


class URL(Regexp):
    """
    Simple regexp based url validation. Much like the email validator, you
    probably want to validate the url later by other means if the url must
    resolve.

    :param require_tld:
        If true, then the domain-name portion of the URL must contain a .tld
        suffix.  Set this to false if you want to allow domains like
        `localhost`.
    :param allow_ip:
        If false, then give ip as host will fail validation
    :param message:
        Error message to raise in case of a validation error.
    """

    def __init__(self, require_tld=True, allow_ip=True, message=None):
        regex = (
            r"^[a-z]+://"
            r"(?P<host>[^\/\?:]+)"
            r"(?P<port>:[0-9]+)?"
            r"(?P<path>\/.*?)?"
            r"(?P<query>\?.*)?$"
        )
        super().__init__(regex, re.IGNORECASE, message)
        self.validate_hostname = HostnameValidation(
            require_tld=require_tld, allow_ip=allow_ip
        )

    def __call__(self, form, field):
        message = self.message
        if message is None:
            message = field.gettext("Invalid URL.")

        match = super().__call__(form, field, message)
        if not self.validate_hostname(match.group("host")):
            raise ValidationError(message)


class UUID:
    """
    Validates a UUID.

    :param message:
        Error message to raise in case of a validation error.
    """

    def __init__(self, message=None):
        self.message = message

    def __call__(self, form, field):
        message = self.message
        if message is None:
            message = field.gettext("Invalid UUID.")
        try:
            uuid.UUID(field.data)
        except ValueError as exc:
            raise ValidationError(message) from exc


class AnyOf:
    """
    Compares the incoming data to a sequence of valid inputs.

    :param values:
        A sequence of valid inputs.
    :param message:
        Error message to raise in case of a validation error. `%(values)s`
        contains the list of values.
    :param values_formatter:
        Function used to format the list of values in the error message.
    """

    def __init__(self, values, message=None, values_formatter=None):
        self.values = values
        self.message = message
        if values_formatter is None:
            values_formatter = self.default_values_formatter
        self.values_formatter = values_formatter

    def __call__(self, form, field):
        data = field.data if isinstance(field.data, list) else [field.data]
        if any(d in self.values for d in data):
            return

        message = self.message
        if message is None:
            message = field.gettext("Invalid value, must be one of: %(values)s.")

        raise ValidationError(message % dict(values=self.values_formatter(self.values)))

    @staticmethod
    def default_values_formatter(values):
        return ", ".join(str(x) for x in values)


class NoneOf:
    """
    Compares the incoming data to a sequence of invalid inputs.

    :param values:
        A sequence of invalid inputs.
    :param message:
        Error message to raise in case of a validation error. `%(values)s`
        contains the list of values.
    :param values_formatter:
        Function used to format the list of values in the error message.
    """

    def __init__(self, values, message=None, values_formatter=None):
        self.values = values
        self.message = message
        if values_formatter is None:
            values_formatter = self.default_values_formatter
        self.values_formatter = values_formatter

    def __call__(self, form, field):
        data = field.data if isinstance(field.data, list) else [field.data]
        if not any(d in self.values for d in data):
            return

        message = self.message
        if message is None:
            message = field.gettext("Invalid value, can't be any of: %(values)s.")

        raise ValidationError(message % dict(values=self.values_formatter(self.values)))

    @staticmethod
    def default_values_formatter(v):
        return ", ".join(str(x) for x in v)


class HostnameValidation:
    """
    Helper class for checking hostnames for validation.

    This is not a validator in and of itself, and as such is not exported.
    """

    hostname_part = re.compile(r"^(xn-|[a-z0-9_]+)(-[a-z0-9_-]+)*$", re.IGNORECASE)
    tld_part = re.compile(r"^([a-z]{2,20}|xn--([a-z0-9]+-)*[a-z0-9]+)$", re.IGNORECASE)

    def __init__(self, require_tld=True, allow_ip=False):
        self.require_tld = require_tld
        self.allow_ip = allow_ip

    def __call__(self, hostname):
        if self.allow_ip and (
            IPAddress.check_ipv4(hostname) or IPAddress.check_ipv6(hostname)
        ):
            return True

        # Encode out IDNA hostnames. This makes further validation easier.
        try:
            hostname = hostname.encode("idna")
        except UnicodeError:
            pass

        # Turn back into a string in Python 3x
        if not isinstance(hostname, str):
            hostname = hostname.decode("ascii")

        if len(hostname) > 253:
            return False

        # Check that all labels in the hostname are valid
        parts = hostname.split(".")
        for part in parts:
            if not part or len(part) > 63:
                return False
            if not self.hostname_part.match(part):
                return False

        if self.require_tld and (len(parts) < 2 or not self.tld_part.match(parts[-1])):
            return False

        return True


class ReadOnly:
    """
    Set a field readonly.

    Validation fails if the form data is different than the
    field object data, or if unset, from the field default data.
    """

    def __init__(self):
        self.field_flags = {"readonly": True}

    def __call__(self, form, field):
        if field.data != field.object_data:
            raise ValidationError(field.gettext("This field cannot be edited."))


class Disabled:
    """
    Set a field disabled.

    Validation fails if the form data has any value.
    """

    def __init__(self):
        self.field_flags = {"disabled": True}

    def __call__(self, form, field):
        if field.raw_data is not None:
            raise ValidationError(
                field.gettext("This field is disabled and cannot have a value.")
            )


email = Email
equal_to = EqualTo
ip_address = IPAddress
mac_address = MacAddress
length = Length
number_range = NumberRange
optional = Optional
input_required = InputRequired
data_required = DataRequired
regexp = Regexp
url = URL
any_of = AnyOf
none_of = NoneOf
readonly = ReadOnly
disabled = Disabled

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sortedcontainers\sortedset.py ===
"""Sorted Set
=============

:doc:`Sorted Containers<index>` is an Apache2 licensed Python sorted
collections library, written in pure-Python, and fast as C-extensions. The
:doc:`introduction<introduction>` is the best way to get started.

Sorted set implementations:

.. currentmodule:: sortedcontainers

* :class:`SortedSet`

"""

from itertools import chain
from operator import eq, ne, gt, ge, lt, le
from textwrap import dedent

from .sortedlist import SortedList, recursive_repr

###############################################################################
# BEGIN Python 2/3 Shims
###############################################################################

try:
    from collections.abc import MutableSet, Sequence, Set
except ImportError:
    from collections import MutableSet, Sequence, Set

###############################################################################
# END Python 2/3 Shims
###############################################################################


class SortedSet(MutableSet, Sequence):
    """Sorted set is a sorted mutable set.

    Sorted set values are maintained in sorted order. The design of sorted set
    is simple: sorted set uses a set for set-operations and maintains a sorted
    list of values.

    Sorted set values must be hashable and comparable. The hash and total
    ordering of values must not change while they are stored in the sorted set.

    Mutable set methods:

    * :func:`SortedSet.__contains__`
    * :func:`SortedSet.__iter__`
    * :func:`SortedSet.__len__`
    * :func:`SortedSet.add`
    * :func:`SortedSet.discard`

    Sequence methods:

    * :func:`SortedSet.__getitem__`
    * :func:`SortedSet.__delitem__`
    * :func:`SortedSet.__reversed__`

    Methods for removing values:

    * :func:`SortedSet.clear`
    * :func:`SortedSet.pop`
    * :func:`SortedSet.remove`

    Set-operation methods:

    * :func:`SortedSet.difference`
    * :func:`SortedSet.difference_update`
    * :func:`SortedSet.intersection`
    * :func:`SortedSet.intersection_update`
    * :func:`SortedSet.symmetric_difference`
    * :func:`SortedSet.symmetric_difference_update`
    * :func:`SortedSet.union`
    * :func:`SortedSet.update`

    Methods for miscellany:

    * :func:`SortedSet.copy`
    * :func:`SortedSet.count`
    * :func:`SortedSet.__repr__`
    * :func:`SortedSet._check`

    Sorted list methods available:

    * :func:`SortedList.bisect_left`
    * :func:`SortedList.bisect_right`
    * :func:`SortedList.index`
    * :func:`SortedList.irange`
    * :func:`SortedList.islice`
    * :func:`SortedList._reset`

    Additional sorted list methods available, if key-function used:

    * :func:`SortedKeyList.bisect_key_left`
    * :func:`SortedKeyList.bisect_key_right`
    * :func:`SortedKeyList.irange_key`

    Sorted set comparisons use subset and superset relations. Two sorted sets
    are equal if and only if every element of each sorted set is contained in
    the other (each is a subset of the other). A sorted set is less than
    another sorted set if and only if the first sorted set is a proper subset
    of the second sorted set (is a subset, but is not equal). A sorted set is
    greater than another sorted set if and only if the first sorted set is a
    proper superset of the second sorted set (is a superset, but is not equal).

    """
    def __init__(self, iterable=None, key=None):
        """Initialize sorted set instance.

        Optional `iterable` argument provides an initial iterable of values to
        initialize the sorted set.

        Optional `key` argument defines a callable that, like the `key`
        argument to Python's `sorted` function, extracts a comparison key from
        each value. The default, none, compares values directly.

        Runtime complexity: `O(n*log(n))`

        >>> ss = SortedSet([3, 1, 2, 5, 4])
        >>> ss
        SortedSet([1, 2, 3, 4, 5])
        >>> from operator import neg
        >>> ss = SortedSet([3, 1, 2, 5, 4], neg)
        >>> ss
        SortedSet([5, 4, 3, 2, 1], key=<built-in function neg>)

        :param iterable: initial values (optional)
        :param key: function used to extract comparison key (optional)

        """
        self._key = key

        # SortedSet._fromset calls SortedSet.__init__ after initializing the
        # _set attribute. So only create a new set if the _set attribute is not
        # already present.

        if not hasattr(self, '_set'):
            self._set = set()

        self._list = SortedList(self._set, key=key)

        # Expose some set methods publicly.

        _set = self._set
        self.isdisjoint = _set.isdisjoint
        self.issubset = _set.issubset
        self.issuperset = _set.issuperset

        # Expose some sorted list methods publicly.

        _list = self._list
        self.bisect_left = _list.bisect_left
        self.bisect = _list.bisect
        self.bisect_right = _list.bisect_right
        self.index = _list.index
        self.irange = _list.irange
        self.islice = _list.islice
        self._reset = _list._reset

        if key is not None:
            self.bisect_key_left = _list.bisect_key_left
            self.bisect_key_right = _list.bisect_key_right
            self.bisect_key = _list.bisect_key
            self.irange_key = _list.irange_key

        if iterable is not None:
            self._update(iterable)


    @classmethod
    def _fromset(cls, values, key=None):
        """Initialize sorted set from existing set.

        Used internally by set operations that return a new set.

        """
        sorted_set = object.__new__(cls)
        sorted_set._set = values
        sorted_set.__init__(key=key)
        return sorted_set


    @property
    def key(self):
        """Function used to extract comparison key from values.

        Sorted set compares values directly when the key function is none.

        """
        return self._key


    def __contains__(self, value):
        """Return true if `value` is an element of the sorted set.

        ``ss.__contains__(value)`` <==> ``value in ss``

        Runtime complexity: `O(1)`

        >>> ss = SortedSet([1, 2, 3, 4, 5])
        >>> 3 in ss
        True

        :param value: search for value in sorted set
        :return: true if `value` in sorted set

        """
        return value in self._set


    def __getitem__(self, index):
        """Lookup value at `index` in sorted set.

        ``ss.__getitem__(index)`` <==> ``ss[index]``

        Supports slicing.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> ss = SortedSet('abcde')
        >>> ss[2]
        'c'
        >>> ss[-1]
        'e'
        >>> ss[2:5]
        ['c', 'd', 'e']

        :param index: integer or slice for indexing
        :return: value or list of values
        :raises IndexError: if index out of range

        """
        return self._list[index]


    def __delitem__(self, index):
        """Remove value at `index` from sorted set.

        ``ss.__delitem__(index)`` <==> ``del ss[index]``

        Supports slicing.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> ss = SortedSet('abcde')
        >>> del ss[2]
        >>> ss
        SortedSet(['a', 'b', 'd', 'e'])
        >>> del ss[:2]
        >>> ss
        SortedSet(['d', 'e'])

        :param index: integer or slice for indexing
        :raises IndexError: if index out of range

        """
        _set = self._set
        _list = self._list
        if isinstance(index, slice):
            values = _list[index]
            _set.difference_update(values)
        else:
            value = _list[index]
            _set.remove(value)
        del _list[index]


    def __make_cmp(set_op, symbol, doc):
        "Make comparator method."
        def comparer(self, other):
            "Compare method for sorted set and set."
            if isinstance(other, SortedSet):
                return set_op(self._set, other._set)
            elif isinstance(other, Set):
                return set_op(self._set, other)
            return NotImplemented

        set_op_name = set_op.__name__
        comparer.__name__ = '__{0}__'.format(set_op_name)
        doc_str = """Return true if and only if sorted set is {0} `other`.

        ``ss.__{1}__(other)`` <==> ``ss {2} other``

        Comparisons use subset and superset semantics as with sets.

        Runtime complexity: `O(n)`

        :param other: `other` set
        :return: true if sorted set is {0} `other`

        """
        comparer.__doc__ = dedent(doc_str.format(doc, set_op_name, symbol))
        return comparer


    __eq__ = __make_cmp(eq, '==', 'equal to')
    __ne__ = __make_cmp(ne, '!=', 'not equal to')
    __lt__ = __make_cmp(lt, '<', 'a proper subset of')
    __gt__ = __make_cmp(gt, '>', 'a proper superset of')
    __le__ = __make_cmp(le, '<=', 'a subset of')
    __ge__ = __make_cmp(ge, '>=', 'a superset of')
    __make_cmp = staticmethod(__make_cmp)


    def __len__(self):
        """Return the size of the sorted set.

        ``ss.__len__()`` <==> ``len(ss)``

        :return: size of sorted set

        """
        return len(self._set)


    def __iter__(self):
        """Return an iterator over the sorted set.

        ``ss.__iter__()`` <==> ``iter(ss)``

        Iterating the sorted set while adding or deleting values may raise a
        :exc:`RuntimeError` or fail to iterate over all values.

        """
        return iter(self._list)


    def __reversed__(self):
        """Return a reverse iterator over the sorted set.

        ``ss.__reversed__()`` <==> ``reversed(ss)``

        Iterating the sorted set while adding or deleting values may raise a
        :exc:`RuntimeError` or fail to iterate over all values.

        """
        return reversed(self._list)


    def add(self, value):
        """Add `value` to sorted set.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> ss = SortedSet()
        >>> ss.add(3)
        >>> ss.add(1)
        >>> ss.add(2)
        >>> ss
        SortedSet([1, 2, 3])

        :param value: value to add to sorted set

        """
        _set = self._set
        if value not in _set:
            _set.add(value)
            self._list.add(value)

    _add = add


    def clear(self):
        """Remove all values from sorted set.

        Runtime complexity: `O(n)`

        """
        self._set.clear()
        self._list.clear()


    def copy(self):
        """Return a shallow copy of the sorted set.

        Runtime complexity: `O(n)`

        :return: new sorted set

        """
        return self._fromset(set(self._set), key=self._key)

    __copy__ = copy


    def count(self, value):
        """Return number of occurrences of `value` in the sorted set.

        Runtime complexity: `O(1)`

        >>> ss = SortedSet([1, 2, 3, 4, 5])
        >>> ss.count(3)
        1

        :param value: value to count in sorted set
        :return: count

        """
        return 1 if value in self._set else 0


    def discard(self, value):
        """Remove `value` from sorted set if it is a member.

        If `value` is not a member, do nothing.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> ss = SortedSet([1, 2, 3, 4, 5])
        >>> ss.discard(5)
        >>> ss.discard(0)
        >>> ss == set([1, 2, 3, 4])
        True

        :param value: `value` to discard from sorted set

        """
        _set = self._set
        if value in _set:
            _set.remove(value)
            self._list.remove(value)

    _discard = discard


    def pop(self, index=-1):
        """Remove and return value at `index` in sorted set.

        Raise :exc:`IndexError` if the sorted set is empty or index is out of
        range.

        Negative indices are supported.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> ss = SortedSet('abcde')
        >>> ss.pop()
        'e'
        >>> ss.pop(2)
        'c'
        >>> ss
        SortedSet(['a', 'b', 'd'])

        :param int index: index of value (default -1)
        :return: value
        :raises IndexError: if index is out of range

        """
        # pylint: disable=arguments-differ
        value = self._list.pop(index)
        self._set.remove(value)
        return value


    def remove(self, value):
        """Remove `value` from sorted set; `value` must be a member.

        If `value` is not a member, raise :exc:`KeyError`.

        Runtime complexity: `O(log(n))` -- approximate.

        >>> ss = SortedSet([1, 2, 3, 4, 5])
        >>> ss.remove(5)
        >>> ss == set([1, 2, 3, 4])
        True
        >>> ss.remove(0)
        Traceback (most recent call last):
          ...
        KeyError: 0

        :param value: `value` to remove from sorted set
        :raises KeyError: if `value` is not in sorted set

        """
        self._set.remove(value)
        self._list.remove(value)


    def difference(self, *iterables):
        """Return the difference of two or more sets as a new sorted set.

        The `difference` method also corresponds to operator ``-``.

        ``ss.__sub__(iterable)`` <==> ``ss - iterable``

        The difference is all values that are in this sorted set but not the
        other `iterables`.

        >>> ss = SortedSet([1, 2, 3, 4, 5])
        >>> ss.difference([4, 5, 6, 7])
        SortedSet([1, 2, 3])

        :param iterables: iterable arguments
        :return: new sorted set

        """
        diff = self._set.difference(*iterables)
        return self._fromset(diff, key=self._key)

    __sub__ = difference


    def difference_update(self, *iterables):
        """Remove all values of `iterables` from this sorted set.

        The `difference_update` method also corresponds to operator ``-=``.

        ``ss.__isub__(iterable)`` <==> ``ss -= iterable``

        >>> ss = SortedSet([1, 2, 3, 4, 5])
        >>> _ = ss.difference_update([4, 5, 6, 7])
        >>> ss
        SortedSet([1, 2, 3])

        :param iterables: iterable arguments
        :return: itself

        """
        _set = self._set
        _list = self._list
        values = set(chain(*iterables))
        if (4 * len(values)) > len(_set):
            _set.difference_update(values)
            _list.clear()
            _list.update(_set)
        else:
            _discard = self._discard
            for value in values:
                _discard(value)
        return self

    __isub__ = difference_update


    def intersection(self, *iterables):
        """Return the intersection of two or more sets as a new sorted set.

        The `intersection` method also corresponds to operator ``&``.

        ``ss.__and__(iterable)`` <==> ``ss & iterable``

        The intersection is all values that are in this sorted set and each of
        the other `iterables`.

        >>> ss = SortedSet([1, 2, 3, 4, 5])
        >>> ss.intersection([4, 5, 6, 7])
        SortedSet([4, 5])

        :param iterables: iterable arguments
        :return: new sorted set

        """
        intersect = self._set.intersection(*iterables)
        return self._fromset(intersect, key=self._key)

    __and__ = intersection
    __rand__ = __and__


    def intersection_update(self, *iterables):
        """Update the sorted set with the intersection of `iterables`.

        The `intersection_update` method also corresponds to operator ``&=``.

        ``ss.__iand__(iterable)`` <==> ``ss &= iterable``

        Keep only values found in itself and all `iterables`.

        >>> ss = SortedSet([1, 2, 3, 4, 5])
        >>> _ = ss.intersection_update([4, 5, 6, 7])
        >>> ss
        SortedSet([4, 5])

        :param iterables: iterable arguments
        :return: itself

        """
        _set = self._set
        _list = self._list
        _set.intersection_update(*iterables)
        _list.clear()
        _list.update(_set)
        return self

    __iand__ = intersection_update


    def symmetric_difference(self, other):
        """Return the symmetric difference with `other` as a new sorted set.

        The `symmetric_difference` method also corresponds to operator ``^``.

        ``ss.__xor__(other)`` <==> ``ss ^ other``

        The symmetric difference is all values tha are in exactly one of the
        sets.

        >>> ss = SortedSet([1, 2, 3, 4, 5])
        >>> ss.symmetric_difference([4, 5, 6, 7])
        SortedSet([1, 2, 3, 6, 7])

        :param other: `other` iterable
        :return: new sorted set

        """
        diff = self._set.symmetric_difference(other)
        return self._fromset(diff, key=self._key)

    __xor__ = symmetric_difference
    __rxor__ = __xor__


    def symmetric_difference_update(self, other):
        """Update the sorted set with the symmetric difference with `other`.

        The `symmetric_difference_update` method also corresponds to operator
        ``^=``.

        ``ss.__ixor__(other)`` <==> ``ss ^= other``

        Keep only values found in exactly one of itself and `other`.

        >>> ss = SortedSet([1, 2, 3, 4, 5])
        >>> _ = ss.symmetric_difference_update([4, 5, 6, 7])
        >>> ss
        SortedSet([1, 2, 3, 6, 7])

        :param other: `other` iterable
        :return: itself

        """
        _set = self._set
        _list = self._list
        _set.symmetric_difference_update(other)
        _list.clear()
        _list.update(_set)
        return self

    __ixor__ = symmetric_difference_update


    def union(self, *iterables):
        """Return new sorted set with values from itself and all `iterables`.

        The `union` method also corresponds to operator ``|``.

        ``ss.__or__(iterable)`` <==> ``ss | iterable``

        >>> ss = SortedSet([1, 2, 3, 4, 5])
        >>> ss.union([4, 5, 6, 7])
        SortedSet([1, 2, 3, 4, 5, 6, 7])

        :param iterables: iterable arguments
        :return: new sorted set

        """
        return self.__class__(chain(iter(self), *iterables), key=self._key)

    __or__ = union
    __ror__ = __or__


    def update(self, *iterables):
        """Update the sorted set adding values from all `iterables`.

        The `update` method also corresponds to operator ``|=``.

        ``ss.__ior__(iterable)`` <==> ``ss |= iterable``

        >>> ss = SortedSet([1, 2, 3, 4, 5])
        >>> _ = ss.update([4, 5, 6, 7])
        >>> ss
        SortedSet([1, 2, 3, 4, 5, 6, 7])

        :param iterables: iterable arguments
        :return: itself

        """
        _set = self._set
        _list = self._list
        values = set(chain(*iterables))
        if (4 * len(values)) > len(_set):
            _list = self._list
            _set.update(values)
            _list.clear()
            _list.update(_set)
        else:
            _add = self._add
            for value in values:
                _add(value)
        return self

    __ior__ = update
    _update = update


    def __reduce__(self):
        """Support for pickle.

        The tricks played with exposing methods in :func:`SortedSet.__init__`
        confuse pickle so customize the reducer.

        """
        return (type(self), (self._set, self._key))


    @recursive_repr()
    def __repr__(self):
        """Return string representation of sorted set.

        ``ss.__repr__()`` <==> ``repr(ss)``

        :return: string representation

        """
        _key = self._key
        key = '' if _key is None else ', key={0!r}'.format(_key)
        type_name = type(self).__name__
        return '{0}({1!r}{2})'.format(type_name, list(self), key)


    def _check(self):
        """Check invariants of sorted set.

        Runtime complexity: `O(n)`

        """
        _set = self._set
        _list = self._list
        _list._check()
        assert len(_set) == len(_list)
        assert all(value in _set for value in _list)

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sqlalchemy\util\typing.py ===
# util/typing.py
# Copyright (C) 2022-2025 the SQLAlchemy authors and contributors
# <see AUTHORS file>
#
# This module is part of SQLAlchemy and is released under
# the MIT License: https://www.opensource.org/licenses/mit-license.php
# mypy: allow-untyped-defs, allow-untyped-calls

from __future__ import annotations

import builtins
from collections import deque
import collections.abc as collections_abc
import re
import sys
import typing
from typing import Any
from typing import Callable
from typing import Dict
from typing import ForwardRef
from typing import Generic
from typing import Iterable
from typing import Mapping
from typing import NewType
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Set
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union

import typing_extensions

from . import compat

if True:  # zimports removes the tailing comments
    from typing_extensions import Annotated as Annotated  # 3.8
    from typing_extensions import Concatenate as Concatenate  # 3.10
    from typing_extensions import (
        dataclass_transform as dataclass_transform,  # 3.11,
    )
    from typing_extensions import Final as Final  # 3.8
    from typing_extensions import final as final  # 3.8
    from typing_extensions import get_args as get_args  # 3.10
    from typing_extensions import get_origin as get_origin  # 3.10
    from typing_extensions import Literal as Literal  # 3.8
    from typing_extensions import NotRequired as NotRequired  # 3.11
    from typing_extensions import ParamSpec as ParamSpec  # 3.10
    from typing_extensions import Protocol as Protocol  # 3.8
    from typing_extensions import SupportsIndex as SupportsIndex  # 3.8
    from typing_extensions import TypeAlias as TypeAlias  # 3.10
    from typing_extensions import TypedDict as TypedDict  # 3.8
    from typing_extensions import TypeGuard as TypeGuard  # 3.10
    from typing_extensions import Self as Self  # 3.11
    from typing_extensions import TypeAliasType as TypeAliasType  # 3.12
    from typing_extensions import Never as Never  # 3.11
    from typing_extensions import LiteralString as LiteralString  # 3.11

_T = TypeVar("_T", bound=Any)
_KT = TypeVar("_KT")
_KT_co = TypeVar("_KT_co", covariant=True)
_KT_contra = TypeVar("_KT_contra", contravariant=True)
_VT = TypeVar("_VT")
_VT_co = TypeVar("_VT_co", covariant=True)

if compat.py310:
    # why they took until py310 to put this in stdlib is beyond me,
    # I've been wanting it since py27
    from types import NoneType as NoneType
else:
    NoneType = type(None)  # type: ignore


def is_fwd_none(typ: Any) -> bool:
    return isinstance(typ, ForwardRef) and typ.__forward_arg__ == "None"


_AnnotationScanType = Union[
    Type[Any], str, ForwardRef, NewType, TypeAliasType, "GenericProtocol[Any]"
]


class ArgsTypeProtocol(Protocol):
    """protocol for types that have ``__args__``

    there's no public interface for this AFAIK

    """

    __args__: Tuple[_AnnotationScanType, ...]


class GenericProtocol(Protocol[_T]):
    """protocol for generic types.

    this since Python.typing _GenericAlias is private

    """

    __args__: Tuple[_AnnotationScanType, ...]
    __origin__: Type[_T]

    # Python's builtin _GenericAlias has this method, however builtins like
    # list, dict, etc. do not, even though they have ``__origin__`` and
    # ``__args__``
    #
    # def copy_with(self, params: Tuple[_AnnotationScanType, ...]) -> Type[_T]:
    #     ...


# copied from TypeShed, required in order to implement
# MutableMapping.update()
class SupportsKeysAndGetItem(Protocol[_KT, _VT_co]):
    def keys(self) -> Iterable[_KT]: ...

    def __getitem__(self, __k: _KT) -> _VT_co: ...


# work around https://github.com/microsoft/pyright/issues/3025
_LiteralStar = Literal["*"]


def de_stringify_annotation(
    cls: Type[Any],
    annotation: _AnnotationScanType,
    originating_module: str,
    locals_: Mapping[str, Any],
    *,
    str_cleanup_fn: Optional[Callable[[str, str], str]] = None,
    include_generic: bool = False,
    _already_seen: Optional[Set[Any]] = None,
) -> Type[Any]:
    """Resolve annotations that may be string based into real objects.

    This is particularly important if a module defines "from __future__ import
    annotations", as everything inside of __annotations__ is a string. We want
    to at least have generic containers like ``Mapped``, ``Union``, ``List``,
    etc.

    """
    # looked at typing.get_type_hints(), looked at pydantic.  We need much
    # less here, and we here try to not use any private typing internals
    # or construct ForwardRef objects which is documented as something
    # that should be avoided.

    original_annotation = annotation

    if is_fwd_ref(annotation):
        annotation = annotation.__forward_arg__

    if isinstance(annotation, str):
        if str_cleanup_fn:
            annotation = str_cleanup_fn(annotation, originating_module)

        annotation = eval_expression(
            annotation, originating_module, locals_=locals_, in_class=cls
        )

    if (
        include_generic
        and is_generic(annotation)
        and not is_literal(annotation)
    ):
        if _already_seen is None:
            _already_seen = set()

        if annotation in _already_seen:
            # only occurs recursively.  outermost return type
            # will always be Type.
            # the element here will be either ForwardRef or
            # Optional[ForwardRef]
            return original_annotation  # type: ignore
        else:
            _already_seen.add(annotation)

        elements = tuple(
            de_stringify_annotation(
                cls,
                elem,
                originating_module,
                locals_,
                str_cleanup_fn=str_cleanup_fn,
                include_generic=include_generic,
                _already_seen=_already_seen,
            )
            for elem in annotation.__args__
        )

        return _copy_generic_annotation_with(annotation, elements)

    return annotation  # type: ignore


def fixup_container_fwd_refs(
    type_: _AnnotationScanType,
) -> _AnnotationScanType:
    """Correct dict['x', 'y'] into dict[ForwardRef('x'), ForwardRef('y')]
    and similar for list, set

    """

    if (
        is_generic(type_)
        and get_origin(type_)
        in (
            dict,
            set,
            list,
            collections_abc.MutableSet,
            collections_abc.MutableMapping,
            collections_abc.MutableSequence,
            collections_abc.Mapping,
            collections_abc.Sequence,
        )
        # fight, kick and scream to struggle to tell the difference between
        # dict[] and typing.Dict[] which DO NOT compare the same and DO NOT
        # behave the same yet there is NO WAY to distinguish between which type
        # it is using public attributes
        and not re.match(
            "typing.(?:Dict|List|Set|.*Mapping|.*Sequence|.*Set)", repr(type_)
        )
    ):
        # compat with py3.10 and earlier
        return get_origin(type_).__class_getitem__(  # type: ignore
            tuple(
                [
                    ForwardRef(elem) if isinstance(elem, str) else elem
                    for elem in get_args(type_)
                ]
            )
        )
    return type_


def _copy_generic_annotation_with(
    annotation: GenericProtocol[_T], elements: Tuple[_AnnotationScanType, ...]
) -> Type[_T]:
    if hasattr(annotation, "copy_with"):
        # List, Dict, etc. real generics
        return annotation.copy_with(elements)  # type: ignore
    else:
        # Python builtins list, dict, etc.
        return annotation.__origin__[elements]  # type: ignore


def eval_expression(
    expression: str,
    module_name: str,
    *,
    locals_: Optional[Mapping[str, Any]] = None,
    in_class: Optional[Type[Any]] = None,
) -> Any:
    try:
        base_globals: Dict[str, Any] = sys.modules[module_name].__dict__
    except KeyError as ke:
        raise NameError(
            f"Module {module_name} isn't present in sys.modules; can't "
            f"evaluate expression {expression}"
        ) from ke

    try:
        if in_class is not None:
            cls_namespace = dict(in_class.__dict__)
            cls_namespace.setdefault(in_class.__name__, in_class)

            # see #10899.  We want the locals/globals to take precedence
            # over the class namespace in this context, even though this
            # is not the usual way variables would resolve.
            cls_namespace.update(base_globals)

            annotation = eval(expression, cls_namespace, locals_)
        else:
            annotation = eval(expression, base_globals, locals_)
    except Exception as err:
        raise NameError(
            f"Could not de-stringify annotation {expression!r}"
        ) from err
    else:
        return annotation


def eval_name_only(
    name: str,
    module_name: str,
    *,
    locals_: Optional[Mapping[str, Any]] = None,
) -> Any:
    if "." in name:
        return eval_expression(name, module_name, locals_=locals_)

    try:
        base_globals: Dict[str, Any] = sys.modules[module_name].__dict__
    except KeyError as ke:
        raise NameError(
            f"Module {module_name} isn't present in sys.modules; can't "
            f"resolve name {name}"
        ) from ke

    # name only, just look in globals.  eval() works perfectly fine here,
    # however we are seeking to have this be faster, as this occurs for
    # every Mapper[] keyword, etc. depending on configuration
    try:
        return base_globals[name]
    except KeyError as ke:
        # check in builtins as well to handle `list`, `set` or `dict`, etc.
        try:
            return builtins.__dict__[name]
        except KeyError:
            pass

        raise NameError(
            f"Could not locate name {name} in module {module_name}"
        ) from ke


def resolve_name_to_real_class_name(name: str, module_name: str) -> str:
    try:
        obj = eval_name_only(name, module_name)
    except NameError:
        return name
    else:
        return getattr(obj, "__name__", name)


def is_pep593(type_: Optional[Any]) -> bool:
    return type_ is not None and get_origin(type_) in _type_tuples.Annotated


def is_non_string_iterable(obj: Any) -> TypeGuard[Iterable[Any]]:
    return isinstance(obj, collections_abc.Iterable) and not isinstance(
        obj, (str, bytes)
    )


def is_literal(type_: Any) -> bool:
    return get_origin(type_) in _type_tuples.Literal


def is_newtype(type_: Optional[_AnnotationScanType]) -> TypeGuard[NewType]:
    return hasattr(type_, "__supertype__")

    # doesn't work in 3.8, 3.7 as it passes a closure, not an
    # object instance
    # isinstance(type, type_instances.NewType)


def is_generic(type_: _AnnotationScanType) -> TypeGuard[GenericProtocol[Any]]:
    return hasattr(type_, "__args__") and hasattr(type_, "__origin__")


def is_pep695(type_: _AnnotationScanType) -> TypeGuard[TypeAliasType]:
    # NOTE: a generic TAT does not instance check as TypeAliasType outside of
    # python 3.10. For sqlalchemy use cases it's fine to consider it a TAT
    # though.
    # NOTE: things seems to work also without this additional check
    if is_generic(type_):
        return is_pep695(type_.__origin__)
    return isinstance(type_, _type_instances.TypeAliasType)


def flatten_newtype(type_: NewType) -> Type[Any]:
    super_type = type_.__supertype__
    while is_newtype(super_type):
        super_type = super_type.__supertype__
    return super_type  # type: ignore[return-value]


def pep695_values(type_: _AnnotationScanType) -> Set[Any]:
    """Extracts the value from a TypeAliasType, recursively exploring unions
    and inner TypeAliasType to flatten them into a single set.

    Forward references are not evaluated, so no recursive exploration happens
    into them.
    """
    _seen = set()

    def recursive_value(inner_type):
        if inner_type in _seen:
            # recursion are not supported (at least it's flagged as
            # an error by pyright). Just avoid infinite loop
            return inner_type
        _seen.add(inner_type)
        if not is_pep695(inner_type):
            return inner_type
        value = inner_type.__value__
        if not is_union(value):
            return value
        return [recursive_value(t) for t in value.__args__]

    res = recursive_value(type_)
    if isinstance(res, list):
        types = set()
        stack = deque(res)
        while stack:
            t = stack.popleft()
            if isinstance(t, list):
                stack.extend(t)
            else:
                types.add(None if t is NoneType or is_fwd_none(t) else t)
        return types
    else:
        return {res}


def is_fwd_ref(
    type_: _AnnotationScanType,
    check_generic: bool = False,
    check_for_plain_string: bool = False,
) -> TypeGuard[ForwardRef]:
    if check_for_plain_string and isinstance(type_, str):
        return True
    elif isinstance(type_, _type_instances.ForwardRef):
        return True
    elif check_generic and is_generic(type_):
        return any(
            is_fwd_ref(
                arg, True, check_for_plain_string=check_for_plain_string
            )
            for arg in type_.__args__
        )
    else:
        return False


@overload
def de_optionalize_union_types(type_: str) -> str: ...


@overload
def de_optionalize_union_types(type_: Type[Any]) -> Type[Any]: ...


@overload
def de_optionalize_union_types(
    type_: _AnnotationScanType,
) -> _AnnotationScanType: ...


def de_optionalize_union_types(
    type_: _AnnotationScanType,
) -> _AnnotationScanType:
    """Given a type, filter out ``Union`` types that include ``NoneType``
    to not include the ``NoneType``.

    Contains extra logic to work on non-flattened unions, unions that contain
    ``None`` (seen in py38, 37)

    """

    if is_fwd_ref(type_):
        return _de_optionalize_fwd_ref_union_types(type_, False)

    elif is_union(type_) and includes_none(type_):
        if compat.py39:
            typ = set(type_.__args__)
        else:
            # py38, 37 - unions are not automatically flattened, can contain
            # None rather than NoneType
            stack_of_unions = deque([type_])
            typ = set()
            while stack_of_unions:
                u_typ = stack_of_unions.popleft()
                for elem in u_typ.__args__:
                    if is_union(elem):
                        stack_of_unions.append(elem)
                    else:
                        typ.add(elem)

            typ.discard(None)  # type: ignore

        typ = {t for t in typ if t is not NoneType and not is_fwd_none(t)}

        return make_union_type(*typ)

    else:
        return type_


@overload
def _de_optionalize_fwd_ref_union_types(
    type_: ForwardRef, return_has_none: Literal[True]
) -> bool: ...


@overload
def _de_optionalize_fwd_ref_union_types(
    type_: ForwardRef, return_has_none: Literal[False]
) -> _AnnotationScanType: ...


def _de_optionalize_fwd_ref_union_types(
    type_: ForwardRef, return_has_none: bool
) -> Union[_AnnotationScanType, bool]:
    """return the non-optional type for Optional[], Union[None, ...], x|None,
    etc. without de-stringifying forward refs.

    unfortunately this seems to require lots of hardcoded heuristics

    """

    annotation = type_.__forward_arg__

    mm = re.match(r"^(.+?)\[(.+)\]$", annotation)
    if mm:
        g1 = mm.group(1).split(".")[-1]
        if g1 == "Optional":
            return True if return_has_none else ForwardRef(mm.group(2))
        elif g1 == "Union":
            if "[" in mm.group(2):
                # cases like "Union[Dict[str, int], int, None]"
                elements: list[str] = []
                current: list[str] = []
                ignore_comma = 0
                for char in mm.group(2):
                    if char == "[":
                        ignore_comma += 1
                    elif char == "]":
                        ignore_comma -= 1
                    elif ignore_comma == 0 and char == ",":
                        elements.append("".join(current).strip())
                        current.clear()
                        continue
                    current.append(char)
            else:
                elements = re.split(r",\s*", mm.group(2))
            parts = [ForwardRef(elem) for elem in elements if elem != "None"]
            if return_has_none:
                return len(elements) != len(parts)
            else:
                return make_union_type(*parts) if parts else Never  # type: ignore[return-value] # noqa: E501
        else:
            return False if return_has_none else type_

    pipe_tokens = re.split(r"\s*\|\s*", annotation)
    has_none = "None" in pipe_tokens
    if return_has_none:
        return has_none
    if has_none:
        anno_str = "|".join(p for p in pipe_tokens if p != "None")
        return ForwardRef(anno_str) if anno_str else Never  # type: ignore[return-value] # noqa: E501

    return type_


def make_union_type(*types: _AnnotationScanType) -> Type[Any]:
    """Make a Union type."""

    return Union[types]  # type: ignore


def includes_none(type_: Any) -> bool:
    """Returns if the type annotation ``type_`` allows ``None``.

    This function supports:
    * forward refs
    * unions
    * pep593 - Annotated
    * pep695 - TypeAliasType (does not support looking into
    fw reference of other pep695)
    * NewType
    * plain types like ``int``, ``None``, etc
    """
    if is_fwd_ref(type_):
        return _de_optionalize_fwd_ref_union_types(type_, True)
    if is_union(type_):
        return any(includes_none(t) for t in get_args(type_))
    if is_pep593(type_):
        return includes_none(get_args(type_)[0])
    if is_pep695(type_):
        return any(includes_none(t) for t in pep695_values(type_))
    if is_newtype(type_):
        return includes_none(type_.__supertype__)
    try:
        return type_ in (NoneType, None) or is_fwd_none(type_)
    except TypeError:
        # if type_ is Column, mapped_column(), etc. the use of "in"
        # resolves to ``__eq__()`` which then gives us an expression object
        # that can't resolve to boolean.  just catch it all via exception
        return False


def is_a_type(type_: Any) -> bool:
    return (
        isinstance(type_, type)
        or hasattr(type_, "__origin__")
        or type_.__module__ in ("typing", "typing_extensions")
        or type(type_).__mro__[0].__module__ in ("typing", "typing_extensions")
    )


def is_union(type_: Any) -> TypeGuard[ArgsTypeProtocol]:
    return is_origin_of(type_, "Union", "UnionType")


def is_origin_of_cls(
    type_: Any, class_obj: Union[Tuple[Type[Any], ...], Type[Any]]
) -> bool:
    """return True if the given type has an __origin__ that shares a base
    with the given class"""

    origin = get_origin(type_)
    if origin is None:
        return False

    return isinstance(origin, type) and issubclass(origin, class_obj)


def is_origin_of(
    type_: Any, *names: str, module: Optional[str] = None
) -> bool:
    """return True if the given type has an __origin__ with the given name
    and optional module."""

    origin = get_origin(type_)
    if origin is None:
        return False

    return _get_type_name(origin) in names and (
        module is None or origin.__module__.startswith(module)
    )


def _get_type_name(type_: Type[Any]) -> str:
    if compat.py310:
        return type_.__name__
    else:
        typ_name = getattr(type_, "__name__", None)
        if typ_name is None:
            typ_name = getattr(type_, "_name", None)

        return typ_name  # type: ignore


class DescriptorProto(Protocol):
    def __get__(self, instance: object, owner: Any) -> Any: ...

    def __set__(self, instance: Any, value: Any) -> None: ...

    def __delete__(self, instance: Any) -> None: ...


_DESC = TypeVar("_DESC", bound=DescriptorProto)


class DescriptorReference(Generic[_DESC]):
    """a descriptor that refers to a descriptor.

    used for cases where we need to have an instance variable referring to an
    object that is itself a descriptor, which typically confuses typing tools
    as they don't know when they should use ``__get__`` or not when referring
    to the descriptor assignment as an instance variable. See
    sqlalchemy.orm.interfaces.PropComparator.prop

    """

    if TYPE_CHECKING:

        def __get__(self, instance: object, owner: Any) -> _DESC: ...

        def __set__(self, instance: Any, value: _DESC) -> None: ...

        def __delete__(self, instance: Any) -> None: ...


_DESC_co = TypeVar("_DESC_co", bound=DescriptorProto, covariant=True)


class RODescriptorReference(Generic[_DESC_co]):
    """a descriptor that refers to a descriptor.

    same as :class:`.DescriptorReference` but is read-only, so that subclasses
    can define a subtype as the generically contained element

    """

    if TYPE_CHECKING:

        def __get__(self, instance: object, owner: Any) -> _DESC_co: ...

        def __set__(self, instance: Any, value: Any) -> NoReturn: ...

        def __delete__(self, instance: Any) -> NoReturn: ...


_FN = TypeVar("_FN", bound=Optional[Callable[..., Any]])


class CallableReference(Generic[_FN]):
    """a descriptor that refers to a callable.

    works around mypy's limitation of not allowing callables assigned
    as instance variables


    """

    if TYPE_CHECKING:

        def __get__(self, instance: object, owner: Any) -> _FN: ...

        def __set__(self, instance: Any, value: _FN) -> None: ...

        def __delete__(self, instance: Any) -> None: ...


class _TypingInstances:
    def __getattr__(self, key: str) -> tuple[type, ...]:
        types = tuple(
            {
                t
                for t in [
                    getattr(typing, key, None),
                    getattr(typing_extensions, key, None),
                ]
                if t is not None
            }
        )
        if not types:
            raise AttributeError(key)
        self.__dict__[key] = types
        return types


_type_tuples = _TypingInstances()
if TYPE_CHECKING:
    _type_instances = typing_extensions
else:
    _type_instances = _type_tuples

LITERAL_TYPES = _type_tuples.Literal

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\bidi\browsing_context.py ===
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

from typing import Dict, List, Optional, Union

from selenium.webdriver.common.bidi.common import command_builder

from .session import Session


class ReadinessState:
    """Represents the stage of document loading at which a navigation command will return."""

    NONE = "none"
    INTERACTIVE = "interactive"
    COMPLETE = "complete"


class UserPromptType:
    """Represents the possible user prompt types."""

    ALERT = "alert"
    BEFORE_UNLOAD = "beforeunload"
    CONFIRM = "confirm"
    PROMPT = "prompt"


class NavigationInfo:
    """Provides details of an ongoing navigation."""

    def __init__(
        self,
        context: str,
        navigation: Optional[str],
        timestamp: int,
        url: str,
    ):
        self.context = context
        self.navigation = navigation
        self.timestamp = timestamp
        self.url = url

    @classmethod
    def from_json(cls, json: Dict) -> "NavigationInfo":
        """Creates a NavigationInfo instance from a dictionary.

        Parameters:
        -----------
            json: A dictionary containing the navigation information.

        Returns:
        -------
            NavigationInfo: A new instance of NavigationInfo.
        """
        return cls(
            context=json.get("context"),
            navigation=json.get("navigation"),
            timestamp=json.get("timestamp"),
            url=json.get("url"),
        )


class BrowsingContextInfo:
    """Represents the properties of a navigable."""

    def __init__(
        self,
        context: str,
        url: str,
        children: Optional[List["BrowsingContextInfo"]],
        parent: Optional[str] = None,
        user_context: Optional[str] = None,
        original_opener: Optional[str] = None,
        client_window: Optional[str] = None,
    ):
        self.context = context
        self.url = url
        self.children = children
        self.parent = parent
        self.user_context = user_context
        self.original_opener = original_opener
        self.client_window = client_window

    @classmethod
    def from_json(cls, json: Dict) -> "BrowsingContextInfo":
        """Creates a BrowsingContextInfo instance from a dictionary.

        Parameters:
        -----------
            json: A dictionary containing the browsing context information.

        Returns:
        -------
            BrowsingContextInfo: A new instance of BrowsingContextInfo.
        """
        children = None
        if json.get("children") is not None:
            children = [BrowsingContextInfo.from_json(child) for child in json.get("children")]

        return cls(
            context=json.get("context"),
            url=json.get("url"),
            children=children,
            parent=json.get("parent"),
            user_context=json.get("userContext"),
            original_opener=json.get("originalOpener"),
            client_window=json.get("clientWindow"),
        )


class DownloadWillBeginParams(NavigationInfo):
    """Parameters for the downloadWillBegin event."""

    def __init__(
        self,
        context: str,
        navigation: Optional[str],
        timestamp: int,
        url: str,
        suggested_filename: str,
    ):
        super().__init__(context, navigation, timestamp, url)
        self.suggested_filename = suggested_filename

    @classmethod
    def from_json(cls, json: Dict) -> "DownloadWillBeginParams":
        """Creates a DownloadWillBeginParams instance from a dictionary.

        Parameters:
        -----------
            json: A dictionary containing the download parameters.

        Returns:
        -------
            DownloadWillBeginParams: A new instance of DownloadWillBeginParams.
        """
        return cls(
            context=json.get("context"),
            navigation=json.get("navigation"),
            timestamp=json.get("timestamp"),
            url=json.get("url"),
            suggested_filename=json.get("suggestedFilename"),
        )


class UserPromptOpenedParams:
    """Parameters for the userPromptOpened event."""

    def __init__(
        self,
        context: str,
        handler: str,
        message: str,
        type: str,
        default_value: Optional[str] = None,
    ):
        self.context = context
        self.handler = handler
        self.message = message
        self.type = type
        self.default_value = default_value

    @classmethod
    def from_json(cls, json: Dict) -> "UserPromptOpenedParams":
        """Creates a UserPromptOpenedParams instance from a dictionary.

        Parameters:
        -----------
            json: A dictionary containing the user prompt parameters.

        Returns:
        -------
            UserPromptOpenedParams: A new instance of UserPromptOpenedParams.
        """
        return cls(
            context=json.get("context"),
            handler=json.get("handler"),
            message=json.get("message"),
            type=json.get("type"),
            default_value=json.get("defaultValue"),
        )


class UserPromptClosedParams:
    """Parameters for the userPromptClosed event."""

    def __init__(
        self,
        context: str,
        accepted: bool,
        type: str,
        user_text: Optional[str] = None,
    ):
        self.context = context
        self.accepted = accepted
        self.type = type
        self.user_text = user_text

    @classmethod
    def from_json(cls, json: Dict) -> "UserPromptClosedParams":
        """Creates a UserPromptClosedParams instance from a dictionary.

        Parameters:
        -----------
            json: A dictionary containing the user prompt closed parameters.

        Returns:
        -------
            UserPromptClosedParams: A new instance of UserPromptClosedParams.
        """
        return cls(
            context=json.get("context"),
            accepted=json.get("accepted"),
            type=json.get("type"),
            user_text=json.get("userText"),
        )


class HistoryUpdatedParams:
    """Parameters for the historyUpdated event."""

    def __init__(
        self,
        context: str,
        url: str,
    ):
        self.context = context
        self.url = url

    @classmethod
    def from_json(cls, json: Dict) -> "HistoryUpdatedParams":
        """Creates a HistoryUpdatedParams instance from a dictionary.

        Parameters:
        -----------
            json: A dictionary containing the history updated parameters.

        Returns:
        -------
            HistoryUpdatedParams: A new instance of HistoryUpdatedParams.
        """
        return cls(
            context=json.get("context"),
            url=json.get("url"),
        )


class BrowsingContextEvent:
    """Base class for browsing context events."""

    def __init__(self, event_class: str, **kwargs):
        self.event_class = event_class
        self.params = kwargs

    @classmethod
    def from_json(cls, json: Dict) -> "BrowsingContextEvent":
        """Creates a BrowsingContextEvent instance from a dictionary.

        Parameters:
        -----------
            json: A dictionary containing the event information.

        Returns:
        -------
            BrowsingContextEvent: A new instance of BrowsingContextEvent.
        """
        return cls(event_class=json.get("event_class"), **json)


class BrowsingContext:
    """BiDi implementation of the browsingContext module."""

    EVENTS = {
        "context_created": "browsingContext.contextCreated",
        "context_destroyed": "browsingContext.contextDestroyed",
        "dom_content_loaded": "browsingContext.domContentLoaded",
        "download_will_begin": "browsingContext.downloadWillBegin",
        "fragment_navigated": "browsingContext.fragmentNavigated",
        "history_updated": "browsingContext.historyUpdated",
        "load": "browsingContext.load",
        "navigation_aborted": "browsingContext.navigationAborted",
        "navigation_committed": "browsingContext.navigationCommitted",
        "navigation_failed": "browsingContext.navigationFailed",
        "navigation_started": "browsingContext.navigationStarted",
        "user_prompt_closed": "browsingContext.userPromptClosed",
        "user_prompt_opened": "browsingContext.userPromptOpened",
    }

    def __init__(self, conn):
        self.conn = conn
        self.subscriptions = {}
        self.callbacks = {}

    def activate(self, context: str) -> None:
        """Activates and focuses the given top-level traversable.

        Parameters:
        -----------
            context: The browsing context ID to activate.

        Raises:
        ------
            Exception: If the browsing context is not a top-level traversable.
        """
        params = {"context": context}
        self.conn.execute(command_builder("browsingContext.activate", params))

    def capture_screenshot(
        self,
        context: str,
        origin: str = "viewport",
        format: Optional[Dict] = None,
        clip: Optional[Dict] = None,
    ) -> str:
        """Captures an image of the given navigable, and returns it as a Base64-encoded string.

        Parameters:
        -----------
            context: The browsing context ID to capture.
            origin: The origin of the screenshot, either "viewport" or "document".
            format: The format of the screenshot.
            clip: The clip rectangle of the screenshot.

        Returns:
        -------
            str: The Base64-encoded screenshot.
        """
        params = {"context": context, "origin": origin}
        if format is not None:
            params["format"] = format
        if clip is not None:
            params["clip"] = clip

        result = self.conn.execute(command_builder("browsingContext.captureScreenshot", params))
        return result["data"]

    def close(self, context: str, prompt_unload: bool = False) -> None:
        """Closes a top-level traversable.

        Parameters:
        -----------
            context: The browsing context ID to close.
            prompt_unload: Whether to prompt to unload.

        Raises:
        ------
            Exception: If the browsing context is not a top-level traversable.
        """
        params = {"context": context, "promptUnload": prompt_unload}
        self.conn.execute(command_builder("browsingContext.close", params))

    def create(
        self,
        type: str,
        reference_context: Optional[str] = None,
        background: bool = False,
        user_context: Optional[str] = None,
    ) -> str:
        """Creates a new navigable, either in a new tab or in a new window, and returns its navigable id.

        Parameters:
        -----------
            type: The type of the new navigable, either "tab" or "window".
            reference_context: The reference browsing context ID.
            background: Whether to create the new navigable in the background.
            user_context: The user context ID.

        Returns:
        -------
            str: The browsing context ID of the created navigable.
        """
        params = {"type": type}
        if reference_context is not None:
            params["referenceContext"] = reference_context
        if background is not None:
            params["background"] = background
        if user_context is not None:
            params["userContext"] = user_context

        result = self.conn.execute(command_builder("browsingContext.create", params))
        return result["context"]

    def get_tree(
        self,
        max_depth: Optional[int] = None,
        root: Optional[str] = None,
    ) -> List[BrowsingContextInfo]:
        """Returns a tree of all descendent navigables including the given parent itself, or all top-level contexts
        when no parent is provided.

        Parameters:
        -----------
            max_depth: The maximum depth of the tree.
            root: The root browsing context ID.

        Returns:
        -------
            List[BrowsingContextInfo]: A list of browsing context information.
        """
        params = {}
        if max_depth is not None:
            params["maxDepth"] = max_depth
        if root is not None:
            params["root"] = root

        result = self.conn.execute(command_builder("browsingContext.getTree", params))
        return [BrowsingContextInfo.from_json(context) for context in result["contexts"]]

    def handle_user_prompt(
        self,
        context: str,
        accept: Optional[bool] = None,
        user_text: Optional[str] = None,
    ) -> None:
        """Allows closing an open prompt.

        Parameters:
        -----------
            context: The browsing context ID.
            accept: Whether to accept the prompt.
            user_text: The text to enter in the prompt.
        """
        params = {"context": context}
        if accept is not None:
            params["accept"] = accept
        if user_text is not None:
            params["userText"] = user_text

        self.conn.execute(command_builder("browsingContext.handleUserPrompt", params))

    def locate_nodes(
        self,
        context: str,
        locator: Dict,
        max_node_count: Optional[int] = None,
        serialization_options: Optional[Dict] = None,
        start_nodes: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """Returns a list of all nodes matching the specified locator.

        Parameters:
        -----------
            context: The browsing context ID.
            locator: The locator to use.
            max_node_count: The maximum number of nodes to return.
            serialization_options: The serialization options.
            start_nodes: The start nodes.

        Returns:
        -------
            List[Dict]: A list of nodes.
        """
        params = {"context": context, "locator": locator}
        if max_node_count is not None:
            params["maxNodeCount"] = max_node_count
        if serialization_options is not None:
            params["serializationOptions"] = serialization_options
        if start_nodes is not None:
            params["startNodes"] = start_nodes

        result = self.conn.execute(command_builder("browsingContext.locateNodes", params))
        return result["nodes"]

    def navigate(
        self,
        context: str,
        url: str,
        wait: Optional[str] = None,
    ) -> Dict:
        """Navigates a navigable to the given URL.

        Parameters:
        -----------
            context: The browsing context ID.
            url: The URL to navigate to.
            wait: The readiness state to wait for.

        Returns:
        -------
            Dict: A dictionary containing the navigation result.
        """
        params = {"context": context, "url": url}
        if wait is not None:
            params["wait"] = wait

        result = self.conn.execute(command_builder("browsingContext.navigate", params))
        return result

    def print(
        self,
        context: str,
        background: bool = False,
        margin: Optional[Dict] = None,
        orientation: str = "portrait",
        page: Optional[Dict] = None,
        page_ranges: Optional[List[Union[int, str]]] = None,
        scale: float = 1.0,
        shrink_to_fit: bool = True,
    ) -> str:
        """Creates a paginated representation of a document, and returns it as a PDF document represented as a
        Base64-encoded string.

        Parameters:
        -----------
            context: The browsing context ID.
            background: Whether to include the background.
            margin: The margin parameters.
            orientation: The orientation, either "portrait" or "landscape".
            page: The page parameters.
            page_ranges: The page ranges.
            scale: The scale.
            shrink_to_fit: Whether to shrink to fit.

        Returns:
        -------
            str: The Base64-encoded PDF document.
        """
        params = {
            "context": context,
            "background": background,
            "orientation": orientation,
            "scale": scale,
            "shrinkToFit": shrink_to_fit,
        }
        if margin is not None:
            params["margin"] = margin
        if page is not None:
            params["page"] = page
        if page_ranges is not None:
            params["pageRanges"] = page_ranges

        result = self.conn.execute(command_builder("browsingContext.print", params))
        return result["data"]

    def reload(
        self,
        context: str,
        ignore_cache: Optional[bool] = None,
        wait: Optional[str] = None,
    ) -> Dict:
        """Reloads a navigable.

        Parameters:
        -----------
            context: The browsing context ID.
            ignore_cache: Whether to ignore the cache.
            wait: The readiness state to wait for.

        Returns:
        -------
            Dict: A dictionary containing the navigation result.
        """
        params = {"context": context}
        if ignore_cache is not None:
            params["ignoreCache"] = ignore_cache
        if wait is not None:
            params["wait"] = wait

        result = self.conn.execute(command_builder("browsingContext.reload", params))
        return result

    def set_viewport(
        self,
        context: Optional[str] = None,
        viewport: Optional[Dict] = None,
        device_pixel_ratio: Optional[float] = None,
        user_contexts: Optional[List[str]] = None,
    ) -> None:
        """Modifies specific viewport characteristics on the given top-level traversable.

        Parameters:
        -----------
            context: The browsing context ID.
            viewport: The viewport parameters.
            device_pixel_ratio: The device pixel ratio.
            user_contexts: The user context IDs.

        Raises:
        ------
            Exception: If the browsing context is not a top-level traversable.
        """
        params = {}
        if context is not None:
            params["context"] = context
        if viewport is not None:
            params["viewport"] = viewport
        if device_pixel_ratio is not None:
            params["devicePixelRatio"] = device_pixel_ratio
        if user_contexts is not None:
            params["userContexts"] = user_contexts

        self.conn.execute(command_builder("browsingContext.setViewport", params))

    def traverse_history(self, context: str, delta: int) -> Dict:
        """Traverses the history of a given navigable by a delta.

        Parameters:
        -----------
            context: The browsing context ID.
            delta: The delta to traverse by.

        Returns:
        -------
            Dict: A dictionary containing the traverse history result.
        """
        params = {"context": context, "delta": delta}
        result = self.conn.execute(command_builder("browsingContext.traverseHistory", params))
        return result

    def _on_event(self, event_name: str, callback: callable) -> int:
        """Set a callback function to subscribe to a browsing context event.

        Parameters:
        ----------
            event_name: The event to subscribe to.
            callback: The callback function to execute on event.

        Returns:
        -------
            int: callback id
        """
        event = BrowsingContextEvent(event_name)

        def _callback(event_data):
            if event_name == self.EVENTS["context_created"] or event_name == self.EVENTS["context_destroyed"]:
                info = BrowsingContextInfo.from_json(event_data.params)
                callback(info)
            elif event_name == self.EVENTS["download_will_begin"]:
                params = DownloadWillBeginParams.from_json(event_data.params)
                callback(params)
            elif event_name == self.EVENTS["user_prompt_opened"]:
                params = UserPromptOpenedParams.from_json(event_data.params)
                callback(params)
            elif event_name == self.EVENTS["user_prompt_closed"]:
                params = UserPromptClosedParams.from_json(event_data.params)
                callback(params)
            elif event_name == self.EVENTS["history_updated"]:
                params = HistoryUpdatedParams.from_json(event_data.params)
                callback(params)
            else:
                # For navigation events
                info = NavigationInfo.from_json(event_data.params)
                callback(info)

        callback_id = self.conn.add_callback(event, _callback)

        if event_name in self.callbacks:
            self.callbacks[event_name].append(callback_id)
        else:
            self.callbacks[event_name] = [callback_id]

        return callback_id

    def add_event_handler(self, event: str, callback: callable, contexts: Optional[List[str]] = None) -> int:
        """Add an event handler to the browsing context.

        Parameters:
        ----------
            event: The event to subscribe to.
            callback: The callback function to execute on event.
            contexts: The browsing context IDs to subscribe to.

        Returns:
        -------
            int: callback id
        """
        try:
            event_name = self.EVENTS[event]
        except KeyError:
            raise Exception(f"Event {event} not found")

        callback_id = self._on_event(event_name, callback)

        if event_name in self.subscriptions:
            self.subscriptions[event_name].append(callback_id)
        else:
            params = {"events": [event_name]}
            if contexts is not None:
                params["browsingContexts"] = contexts
            session = Session(self.conn)
            self.conn.execute(session.subscribe(**params))
            self.subscriptions[event_name] = [callback_id]

        return callback_id

    def remove_event_handler(self, event: str, callback_id: int) -> None:
        """Remove an event handler from the browsing context.

        Parameters:
        ----------
            event: The event to unsubscribe from.
            callback_id: The callback id to remove.
        """
        try:
            event_name = self.EVENTS[event]
        except KeyError:
            raise Exception(f"Event {event} not found")

        event = BrowsingContextEvent(event_name)

        self.conn.remove_callback(event, callback_id)
        self.subscriptions[event_name].remove(callback_id)
        if len(self.subscriptions[event_name]) == 0:
            params = {"events": [event_name]}
            session = Session(self.conn)
            self.conn.execute(session.unsubscribe(**params))
            del self.subscriptions[event_name]

    def clear_event_handlers(self) -> None:
        """Clear all event handlers from the browsing context."""
        for event_name in self.subscriptions:
            event = BrowsingContextEvent(event_name)
            for callback_id in self.subscriptions[event_name]:
                self.conn.remove_callback(event, callback_id)
            params = {"events": [event_name]}
            session = Session(self.conn)
            self.conn.execute(session.unsubscribe(**params))
        self.subscriptions = {}

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\selenium\webdriver\common\devtools\v135\browser.py ===
# DO NOT EDIT THIS FILE!
#
# This file is generated from the CDP specification. If you need to make
# changes, edit the generator and regenerate all of the modules.
#
# CDP domain: Browser
from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import target


class BrowserContextID(str):
    def to_json(self) -> str:
        return self

    @classmethod
    def from_json(cls, json: str) -> BrowserContextID:
        return cls(json)

    def __repr__(self):
        return 'BrowserContextID({})'.format(super().__repr__())


class WindowID(int):
    def to_json(self) -> int:
        return self

    @classmethod
    def from_json(cls, json: int) -> WindowID:
        return cls(json)

    def __repr__(self):
        return 'WindowID({})'.format(super().__repr__())


class WindowState(enum.Enum):
    '''
    The state of the browser window.
    '''
    NORMAL = "normal"
    MINIMIZED = "minimized"
    MAXIMIZED = "maximized"
    FULLSCREEN = "fullscreen"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class Bounds:
    '''
    Browser window bounds information
    '''
    #: The offset from the left edge of the screen to the window in pixels.
    left: typing.Optional[int] = None

    #: The offset from the top edge of the screen to the window in pixels.
    top: typing.Optional[int] = None

    #: The window width in pixels.
    width: typing.Optional[int] = None

    #: The window height in pixels.
    height: typing.Optional[int] = None

    #: The window state. Default to normal.
    window_state: typing.Optional[WindowState] = None

    def to_json(self):
        json = dict()
        if self.left is not None:
            json['left'] = self.left
        if self.top is not None:
            json['top'] = self.top
        if self.width is not None:
            json['width'] = self.width
        if self.height is not None:
            json['height'] = self.height
        if self.window_state is not None:
            json['windowState'] = self.window_state.to_json()
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            left=int(json['left']) if 'left' in json else None,
            top=int(json['top']) if 'top' in json else None,
            width=int(json['width']) if 'width' in json else None,
            height=int(json['height']) if 'height' in json else None,
            window_state=WindowState.from_json(json['windowState']) if 'windowState' in json else None,
        )


class PermissionType(enum.Enum):
    AR = "ar"
    AUDIO_CAPTURE = "audioCapture"
    AUTOMATIC_FULLSCREEN = "automaticFullscreen"
    BACKGROUND_FETCH = "backgroundFetch"
    BACKGROUND_SYNC = "backgroundSync"
    CAMERA_PAN_TILT_ZOOM = "cameraPanTiltZoom"
    CAPTURED_SURFACE_CONTROL = "capturedSurfaceControl"
    CLIPBOARD_READ_WRITE = "clipboardReadWrite"
    CLIPBOARD_SANITIZED_WRITE = "clipboardSanitizedWrite"
    DISPLAY_CAPTURE = "displayCapture"
    DURABLE_STORAGE = "durableStorage"
    GEOLOCATION = "geolocation"
    HAND_TRACKING = "handTracking"
    IDLE_DETECTION = "idleDetection"
    KEYBOARD_LOCK = "keyboardLock"
    LOCAL_FONTS = "localFonts"
    MIDI = "midi"
    MIDI_SYSEX = "midiSysex"
    NFC = "nfc"
    NOTIFICATIONS = "notifications"
    PAYMENT_HANDLER = "paymentHandler"
    PERIODIC_BACKGROUND_SYNC = "periodicBackgroundSync"
    POINTER_LOCK = "pointerLock"
    PROTECTED_MEDIA_IDENTIFIER = "protectedMediaIdentifier"
    SENSORS = "sensors"
    SMART_CARD = "smartCard"
    SPEAKER_SELECTION = "speakerSelection"
    STORAGE_ACCESS = "storageAccess"
    TOP_LEVEL_STORAGE_ACCESS = "topLevelStorageAccess"
    VIDEO_CAPTURE = "videoCapture"
    VR = "vr"
    WAKE_LOCK_SCREEN = "wakeLockScreen"
    WAKE_LOCK_SYSTEM = "wakeLockSystem"
    WEB_APP_INSTALLATION = "webAppInstallation"
    WEB_PRINTING = "webPrinting"
    WINDOW_MANAGEMENT = "windowManagement"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


class PermissionSetting(enum.Enum):
    GRANTED = "granted"
    DENIED = "denied"
    PROMPT = "prompt"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class PermissionDescriptor:
    '''
    Definition of PermissionDescriptor defined in the Permissions API:
    https://w3c.github.io/permissions/#dom-permissiondescriptor.
    '''
    #: Name of permission.
    #: See https://cs.chromium.org/chromium/src/third_party/blink/renderer/modules/permissions/permission_descriptor.idl for valid permission names.
    name: str

    #: For "midi" permission, may also specify sysex control.
    sysex: typing.Optional[bool] = None

    #: For "push" permission, may specify userVisibleOnly.
    #: Note that userVisibleOnly = true is the only currently supported type.
    user_visible_only: typing.Optional[bool] = None

    #: For "clipboard" permission, may specify allowWithoutSanitization.
    allow_without_sanitization: typing.Optional[bool] = None

    #: For "fullscreen" permission, must specify allowWithoutGesture:true.
    allow_without_gesture: typing.Optional[bool] = None

    #: For "camera" permission, may specify panTiltZoom.
    pan_tilt_zoom: typing.Optional[bool] = None

    def to_json(self):
        json = dict()
        json['name'] = self.name
        if self.sysex is not None:
            json['sysex'] = self.sysex
        if self.user_visible_only is not None:
            json['userVisibleOnly'] = self.user_visible_only
        if self.allow_without_sanitization is not None:
            json['allowWithoutSanitization'] = self.allow_without_sanitization
        if self.allow_without_gesture is not None:
            json['allowWithoutGesture'] = self.allow_without_gesture
        if self.pan_tilt_zoom is not None:
            json['panTiltZoom'] = self.pan_tilt_zoom
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=str(json['name']),
            sysex=bool(json['sysex']) if 'sysex' in json else None,
            user_visible_only=bool(json['userVisibleOnly']) if 'userVisibleOnly' in json else None,
            allow_without_sanitization=bool(json['allowWithoutSanitization']) if 'allowWithoutSanitization' in json else None,
            allow_without_gesture=bool(json['allowWithoutGesture']) if 'allowWithoutGesture' in json else None,
            pan_tilt_zoom=bool(json['panTiltZoom']) if 'panTiltZoom' in json else None,
        )


class BrowserCommandId(enum.Enum):
    '''
    Browser command ids used by executeBrowserCommand.
    '''
    OPEN_TAB_SEARCH = "openTabSearch"
    CLOSE_TAB_SEARCH = "closeTabSearch"

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)


@dataclass
class Bucket:
    '''
    Chrome histogram bucket.
    '''
    #: Minimum value (inclusive).
    low: int

    #: Maximum value (exclusive).
    high: int

    #: Number of samples.
    count: int

    def to_json(self):
        json = dict()
        json['low'] = self.low
        json['high'] = self.high
        json['count'] = self.count
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            low=int(json['low']),
            high=int(json['high']),
            count=int(json['count']),
        )


@dataclass
class Histogram:
    '''
    Chrome histogram.
    '''
    #: Name.
    name: str

    #: Sum of sample values.
    sum_: int

    #: Total number of samples.
    count: int

    #: Buckets.
    buckets: typing.List[Bucket]

    def to_json(self):
        json = dict()
        json['name'] = self.name
        json['sum'] = self.sum_
        json['count'] = self.count
        json['buckets'] = [i.to_json() for i in self.buckets]
        return json

    @classmethod
    def from_json(cls, json):
        return cls(
            name=str(json['name']),
            sum_=int(json['sum']),
            count=int(json['count']),
            buckets=[Bucket.from_json(i) for i in json['buckets']],
        )


def set_permission(
        permission: PermissionDescriptor,
        setting: PermissionSetting,
        origin: typing.Optional[str] = None,
        browser_context_id: typing.Optional[BrowserContextID] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Set permission settings for given origin.

    **EXPERIMENTAL**

    :param permission: Descriptor of permission to override.
    :param setting: Setting of the permission.
    :param origin: *(Optional)* Origin the permission applies to, all origins if not specified.
    :param browser_context_id: *(Optional)* Context to override. When omitted, default browser context is used.
    '''
    params: T_JSON_DICT = dict()
    params['permission'] = permission.to_json()
    params['setting'] = setting.to_json()
    if origin is not None:
        params['origin'] = origin
    if browser_context_id is not None:
        params['browserContextId'] = browser_context_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Browser.setPermission',
        'params': params,
    }
    json = yield cmd_dict


def grant_permissions(
        permissions: typing.List[PermissionType],
        origin: typing.Optional[str] = None,
        browser_context_id: typing.Optional[BrowserContextID] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Grant specific permissions to the given origin and reject all others.

    **EXPERIMENTAL**

    :param permissions:
    :param origin: *(Optional)* Origin the permission applies to, all origins if not specified.
    :param browser_context_id: *(Optional)* BrowserContext to override permissions. When omitted, default browser context is used.
    '''
    params: T_JSON_DICT = dict()
    params['permissions'] = [i.to_json() for i in permissions]
    if origin is not None:
        params['origin'] = origin
    if browser_context_id is not None:
        params['browserContextId'] = browser_context_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Browser.grantPermissions',
        'params': params,
    }
    json = yield cmd_dict


def reset_permissions(
        browser_context_id: typing.Optional[BrowserContextID] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Reset all permission management for all origins.

    :param browser_context_id: *(Optional)* BrowserContext to reset permissions. When omitted, default browser context is used.
    '''
    params: T_JSON_DICT = dict()
    if browser_context_id is not None:
        params['browserContextId'] = browser_context_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Browser.resetPermissions',
        'params': params,
    }
    json = yield cmd_dict


def set_download_behavior(
        behavior: str,
        browser_context_id: typing.Optional[BrowserContextID] = None,
        download_path: typing.Optional[str] = None,
        events_enabled: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Set the behavior when downloading a file.

    **EXPERIMENTAL**

    :param behavior: Whether to allow all or deny all download requests, or use default Chrome behavior if available (otherwise deny). ``allowAndName`` allows download and names files according to their download guids.
    :param browser_context_id: *(Optional)* BrowserContext to set download behavior. When omitted, default browser context is used.
    :param download_path: *(Optional)* The default path to save downloaded files to. This is required if behavior is set to 'allow' or 'allowAndName'.
    :param events_enabled: *(Optional)* Whether to emit download events (defaults to false).
    '''
    params: T_JSON_DICT = dict()
    params['behavior'] = behavior
    if browser_context_id is not None:
        params['browserContextId'] = browser_context_id.to_json()
    if download_path is not None:
        params['downloadPath'] = download_path
    if events_enabled is not None:
        params['eventsEnabled'] = events_enabled
    cmd_dict: T_JSON_DICT = {
        'method': 'Browser.setDownloadBehavior',
        'params': params,
    }
    json = yield cmd_dict


def cancel_download(
        guid: str,
        browser_context_id: typing.Optional[BrowserContextID] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Cancel a download if in progress

    **EXPERIMENTAL**

    :param guid: Global unique identifier of the download.
    :param browser_context_id: *(Optional)* BrowserContext to perform the action in. When omitted, default browser context is used.
    '''
    params: T_JSON_DICT = dict()
    params['guid'] = guid
    if browser_context_id is not None:
        params['browserContextId'] = browser_context_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Browser.cancelDownload',
        'params': params,
    }
    json = yield cmd_dict


def close() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Close browser gracefully.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Browser.close',
    }
    json = yield cmd_dict


def crash() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Crashes browser on the main thread.

    **EXPERIMENTAL**
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Browser.crash',
    }
    json = yield cmd_dict


def crash_gpu_process() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Crashes GPU process.

    **EXPERIMENTAL**
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Browser.crashGpuProcess',
    }
    json = yield cmd_dict


def get_version() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[str, str, str, str, str]]:
    '''
    Returns version information.

    :returns: A tuple with the following items:

        0. **protocolVersion** - Protocol version.
        1. **product** - Product name.
        2. **revision** - Product revision.
        3. **userAgent** - User-Agent.
        4. **jsVersion** - V8 version.
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Browser.getVersion',
    }
    json = yield cmd_dict
    return (
        str(json['protocolVersion']),
        str(json['product']),
        str(json['revision']),
        str(json['userAgent']),
        str(json['jsVersion'])
    )


def get_browser_command_line() -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[str]]:
    '''
    Returns the command line switches for the browser process if, and only if
    --enable-automation is on the commandline.

    **EXPERIMENTAL**

    :returns: Commandline parameters
    '''
    cmd_dict: T_JSON_DICT = {
        'method': 'Browser.getBrowserCommandLine',
    }
    json = yield cmd_dict
    return [str(i) for i in json['arguments']]


def get_histograms(
        query: typing.Optional[str] = None,
        delta: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.List[Histogram]]:
    '''
    Get Chrome histograms.

    **EXPERIMENTAL**

    :param query: *(Optional)* Requested substring in name. Only histograms which have query as a substring in their name are extracted. An empty or absent query returns all histograms.
    :param delta: *(Optional)* If true, retrieve delta since last delta call.
    :returns: Histograms.
    '''
    params: T_JSON_DICT = dict()
    if query is not None:
        params['query'] = query
    if delta is not None:
        params['delta'] = delta
    cmd_dict: T_JSON_DICT = {
        'method': 'Browser.getHistograms',
        'params': params,
    }
    json = yield cmd_dict
    return [Histogram.from_json(i) for i in json['histograms']]


def get_histogram(
        name: str,
        delta: typing.Optional[bool] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,Histogram]:
    '''
    Get a Chrome histogram by name.

    **EXPERIMENTAL**

    :param name: Requested histogram name.
    :param delta: *(Optional)* If true, retrieve delta since last delta call.
    :returns: Histogram.
    '''
    params: T_JSON_DICT = dict()
    params['name'] = name
    if delta is not None:
        params['delta'] = delta
    cmd_dict: T_JSON_DICT = {
        'method': 'Browser.getHistogram',
        'params': params,
    }
    json = yield cmd_dict
    return Histogram.from_json(json['histogram'])


def get_window_bounds(
        window_id: WindowID
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,Bounds]:
    '''
    Get position and size of the browser window.

    **EXPERIMENTAL**

    :param window_id: Browser window id.
    :returns: Bounds information of the window. When window state is 'minimized', the restored window position and size are returned.
    '''
    params: T_JSON_DICT = dict()
    params['windowId'] = window_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Browser.getWindowBounds',
        'params': params,
    }
    json = yield cmd_dict
    return Bounds.from_json(json['bounds'])


def get_window_for_target(
        target_id: typing.Optional[target.TargetID] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,typing.Tuple[WindowID, Bounds]]:
    '''
    Get the browser window that contains the devtools target.

    **EXPERIMENTAL**

    :param target_id: *(Optional)* Devtools agent host id. If called as a part of the session, associated targetId is used.
    :returns: A tuple with the following items:

        0. **windowId** - Browser window id.
        1. **bounds** - Bounds information of the window. When window state is 'minimized', the restored window position and size are returned.
    '''
    params: T_JSON_DICT = dict()
    if target_id is not None:
        params['targetId'] = target_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Browser.getWindowForTarget',
        'params': params,
    }
    json = yield cmd_dict
    return (
        WindowID.from_json(json['windowId']),
        Bounds.from_json(json['bounds'])
    )


def set_window_bounds(
        window_id: WindowID,
        bounds: Bounds
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Set position and/or size of the browser window.

    **EXPERIMENTAL**

    :param window_id: Browser window id.
    :param bounds: New window bounds. The 'minimized', 'maximized' and 'fullscreen' states cannot be combined with 'left', 'top', 'width' or 'height'. Leaves unspecified fields unchanged.
    '''
    params: T_JSON_DICT = dict()
    params['windowId'] = window_id.to_json()
    params['bounds'] = bounds.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Browser.setWindowBounds',
        'params': params,
    }
    json = yield cmd_dict


def set_dock_tile(
        badge_label: typing.Optional[str] = None,
        image: typing.Optional[str] = None
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Set dock tile details, platform-specific.

    **EXPERIMENTAL**

    :param badge_label: *(Optional)*
    :param image: *(Optional)* Png encoded image.
    '''
    params: T_JSON_DICT = dict()
    if badge_label is not None:
        params['badgeLabel'] = badge_label
    if image is not None:
        params['image'] = image
    cmd_dict: T_JSON_DICT = {
        'method': 'Browser.setDockTile',
        'params': params,
    }
    json = yield cmd_dict


def execute_browser_command(
        command_id: BrowserCommandId
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Invoke custom browser commands used by telemetry.

    **EXPERIMENTAL**

    :param command_id:
    '''
    params: T_JSON_DICT = dict()
    params['commandId'] = command_id.to_json()
    cmd_dict: T_JSON_DICT = {
        'method': 'Browser.executeBrowserCommand',
        'params': params,
    }
    json = yield cmd_dict


def add_privacy_sandbox_enrollment_override(
        url: str
    ) -> typing.Generator[T_JSON_DICT,T_JSON_DICT,None]:
    '''
    Allows a site to use privacy sandbox features that require enrollment
    without the site actually being enrolled. Only supported on page targets.

    :param url:
    '''
    params: T_JSON_DICT = dict()
    params['url'] = url
    cmd_dict: T_JSON_DICT = {
        'method': 'Browser.addPrivacySandboxEnrollmentOverride',
        'params': params,
    }
    json = yield cmd_dict


@event_class('Browser.downloadWillBegin')
@dataclass
class DownloadWillBegin:
    '''
    **EXPERIMENTAL**

    Fired when page is about to start a download.
    '''
    #: Id of the frame that caused the download to begin.
    frame_id: page.FrameId
    #: Global unique identifier of the download.
    guid: str
    #: URL of the resource being downloaded.
    url: str
    #: Suggested file name of the resource (the actual name of the file saved on disk may differ).
    suggested_filename: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DownloadWillBegin:
        return cls(
            frame_id=page.FrameId.from_json(json['frameId']),
            guid=str(json['guid']),
            url=str(json['url']),
            suggested_filename=str(json['suggestedFilename'])
        )


@event_class('Browser.downloadProgress')
@dataclass
class DownloadProgress:
    '''
    **EXPERIMENTAL**

    Fired when download makes progress. Last call has ``done`` == true.
    '''
    #: Global unique identifier of the download.
    guid: str
    #: Total expected bytes to download.
    total_bytes: float
    #: Total bytes received.
    received_bytes: float
    #: Download status.
    state: str

    @classmethod
    def from_json(cls, json: T_JSON_DICT) -> DownloadProgress:
        return cls(
            guid=str(json['guid']),
            total_bytes=float(json['totalBytes']),
            received_bytes=float(json['receivedBytes']),
            state=str(json['state'])
        )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\physics\optics\polarization.py ===
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The module implements routines to model the polarization of optical fields
and can be used to calculate the effects of polarization optical elements on
the fields.

- Jones vectors.

- Stokes vectors.

- Jones matrices.

- Mueller matrices.

Examples
========

We calculate a generic Jones vector:

>>> from sympy import symbols, pprint, zeros, simplify
>>> from sympy.physics.optics.polarization import (jones_vector, stokes_vector,
...     half_wave_retarder, polarizing_beam_splitter, jones_2_stokes)

>>> psi, chi, p, I0 = symbols("psi, chi, p, I0", real=True)
>>> x0 = jones_vector(psi, chi)
>>> pprint(x0, use_unicode=True)
⎡-ⅈ⋅sin(χ)⋅sin(ψ) + cos(χ)⋅cos(ψ)⎤
⎢                                ⎥
⎣ⅈ⋅sin(χ)⋅cos(ψ) + sin(ψ)⋅cos(χ) ⎦

And the more general Stokes vector:

>>> s0 = stokes_vector(psi, chi, p, I0)
>>> pprint(s0, use_unicode=True)
⎡          I₀          ⎤
⎢                      ⎥
⎢I₀⋅p⋅cos(2⋅χ)⋅cos(2⋅ψ)⎥
⎢                      ⎥
⎢I₀⋅p⋅sin(2⋅ψ)⋅cos(2⋅χ)⎥
⎢                      ⎥
⎣    I₀⋅p⋅sin(2⋅χ)     ⎦

We calculate how the Jones vector is modified by a half-wave plate:

>>> alpha = symbols("alpha", real=True)
>>> HWP = half_wave_retarder(alpha)
>>> x1 = simplify(HWP*x0)

We calculate the very common operation of passing a beam through a half-wave
plate and then through a polarizing beam-splitter. We do this by putting this
Jones vector as the first entry of a two-Jones-vector state that is transformed
by a 4x4 Jones matrix modelling the polarizing beam-splitter to get the
transmitted and reflected Jones vectors:

>>> PBS = polarizing_beam_splitter()
>>> X1 = zeros(4, 1)
>>> X1[:2, :] = x1
>>> X2 = PBS*X1
>>> transmitted_port = X2[:2, :]
>>> reflected_port = X2[2:, :]

This allows us to calculate how the power in both ports depends on the initial
polarization:

>>> transmitted_power = jones_2_stokes(transmitted_port)[0]
>>> reflected_power = jones_2_stokes(reflected_port)[0]
>>> print(transmitted_power)
cos(-2*alpha + chi + psi)**2/2 + cos(2*alpha + chi - psi)**2/2


>>> print(reflected_power)
sin(-2*alpha + chi + psi)**2/2 + sin(2*alpha + chi - psi)**2/2

Please see the description of the individual functions for further
details and examples.

References
==========

.. [1] https://en.wikipedia.org/wiki/Jones_calculus
.. [2] https://en.wikipedia.org/wiki/Mueller_calculus
.. [3] https://en.wikipedia.org/wiki/Stokes_parameters

"""

from sympy.core.numbers import (I, pi)
from sympy.functions.elementary.complexes import (Abs, im, re)
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.matrices.dense import Matrix
from sympy.simplify.simplify import simplify
from sympy.physics.quantum import TensorProduct


def jones_vector(psi, chi):
    """A Jones vector corresponding to a polarization ellipse with `psi` tilt,
    and `chi` circularity.

    Parameters
    ==========

    psi : numeric type or SymPy Symbol
        The tilt of the polarization relative to the `x` axis.

    chi : numeric type or SymPy Symbol
        The angle adjacent to the mayor axis of the polarization ellipse.


    Returns
    =======

    Matrix :
        A Jones vector.

    Examples
    ========

    The axes on the Poincaré sphere.

    >>> from sympy import pprint, symbols, pi
    >>> from sympy.physics.optics.polarization import jones_vector
    >>> psi, chi = symbols("psi, chi", real=True)

    A general Jones vector.

    >>> pprint(jones_vector(psi, chi), use_unicode=True)
    ⎡-ⅈ⋅sin(χ)⋅sin(ψ) + cos(χ)⋅cos(ψ)⎤
    ⎢                                ⎥
    ⎣ⅈ⋅sin(χ)⋅cos(ψ) + sin(ψ)⋅cos(χ) ⎦

    Horizontal polarization.

    >>> pprint(jones_vector(0, 0), use_unicode=True)
    ⎡1⎤
    ⎢ ⎥
    ⎣0⎦

    Vertical polarization.

    >>> pprint(jones_vector(pi/2, 0), use_unicode=True)
    ⎡0⎤
    ⎢ ⎥
    ⎣1⎦

    Diagonal polarization.

    >>> pprint(jones_vector(pi/4, 0), use_unicode=True)
    ⎡√2⎤
    ⎢──⎥
    ⎢2 ⎥
    ⎢  ⎥
    ⎢√2⎥
    ⎢──⎥
    ⎣2 ⎦

    Anti-diagonal polarization.

    >>> pprint(jones_vector(-pi/4, 0), use_unicode=True)
    ⎡ √2 ⎤
    ⎢ ── ⎥
    ⎢ 2  ⎥
    ⎢    ⎥
    ⎢-√2 ⎥
    ⎢────⎥
    ⎣ 2  ⎦

    Right-hand circular polarization.

    >>> pprint(jones_vector(0, pi/4), use_unicode=True)
    ⎡ √2 ⎤
    ⎢ ── ⎥
    ⎢ 2  ⎥
    ⎢    ⎥
    ⎢√2⋅ⅈ⎥
    ⎢────⎥
    ⎣ 2  ⎦

    Left-hand circular polarization.

    >>> pprint(jones_vector(0, -pi/4), use_unicode=True)
    ⎡  √2  ⎤
    ⎢  ──  ⎥
    ⎢  2   ⎥
    ⎢      ⎥
    ⎢-√2⋅ⅈ ⎥
    ⎢──────⎥
    ⎣  2   ⎦

    """
    return Matrix([-I*sin(chi)*sin(psi) + cos(chi)*cos(psi),
                   I*sin(chi)*cos(psi) + sin(psi)*cos(chi)])


def stokes_vector(psi, chi, p=1, I=1):
    """A Stokes vector corresponding to a polarization ellipse with ``psi``
    tilt, and ``chi`` circularity.

    Parameters
    ==========

    psi : numeric type or SymPy Symbol
        The tilt of the polarization relative to the ``x`` axis.
    chi : numeric type or SymPy Symbol
        The angle adjacent to the mayor axis of the polarization ellipse.
    p : numeric type or SymPy Symbol
        The degree of polarization.
    I : numeric type or SymPy Symbol
        The intensity of the field.


    Returns
    =======

    Matrix :
        A Stokes vector.

    Examples
    ========

    The axes on the Poincaré sphere.

    >>> from sympy import pprint, symbols, pi
    >>> from sympy.physics.optics.polarization import stokes_vector
    >>> psi, chi, p, I = symbols("psi, chi, p, I", real=True)
    >>> pprint(stokes_vector(psi, chi, p, I), use_unicode=True)
    ⎡          I          ⎤
    ⎢                     ⎥
    ⎢I⋅p⋅cos(2⋅χ)⋅cos(2⋅ψ)⎥
    ⎢                     ⎥
    ⎢I⋅p⋅sin(2⋅ψ)⋅cos(2⋅χ)⎥
    ⎢                     ⎥
    ⎣    I⋅p⋅sin(2⋅χ)     ⎦


    Horizontal polarization

    >>> pprint(stokes_vector(0, 0), use_unicode=True)
    ⎡1⎤
    ⎢ ⎥
    ⎢1⎥
    ⎢ ⎥
    ⎢0⎥
    ⎢ ⎥
    ⎣0⎦

    Vertical polarization

    >>> pprint(stokes_vector(pi/2, 0), use_unicode=True)
    ⎡1 ⎤
    ⎢  ⎥
    ⎢-1⎥
    ⎢  ⎥
    ⎢0 ⎥
    ⎢  ⎥
    ⎣0 ⎦

    Diagonal polarization

    >>> pprint(stokes_vector(pi/4, 0), use_unicode=True)
    ⎡1⎤
    ⎢ ⎥
    ⎢0⎥
    ⎢ ⎥
    ⎢1⎥
    ⎢ ⎥
    ⎣0⎦

    Anti-diagonal polarization

    >>> pprint(stokes_vector(-pi/4, 0), use_unicode=True)
    ⎡1 ⎤
    ⎢  ⎥
    ⎢0 ⎥
    ⎢  ⎥
    ⎢-1⎥
    ⎢  ⎥
    ⎣0 ⎦

    Right-hand circular polarization

    >>> pprint(stokes_vector(0, pi/4), use_unicode=True)
    ⎡1⎤
    ⎢ ⎥
    ⎢0⎥
    ⎢ ⎥
    ⎢0⎥
    ⎢ ⎥
    ⎣1⎦

    Left-hand circular polarization

    >>> pprint(stokes_vector(0, -pi/4), use_unicode=True)
    ⎡1 ⎤
    ⎢  ⎥
    ⎢0 ⎥
    ⎢  ⎥
    ⎢0 ⎥
    ⎢  ⎥
    ⎣-1⎦

    Unpolarized light

    >>> pprint(stokes_vector(0, 0, 0), use_unicode=True)
    ⎡1⎤
    ⎢ ⎥
    ⎢0⎥
    ⎢ ⎥
    ⎢0⎥
    ⎢ ⎥
    ⎣0⎦

    """
    S0 = I
    S1 = I*p*cos(2*psi)*cos(2*chi)
    S2 = I*p*sin(2*psi)*cos(2*chi)
    S3 = I*p*sin(2*chi)
    return Matrix([S0, S1, S2, S3])


def jones_2_stokes(e):
    """Return the Stokes vector for a Jones vector ``e``.

    Parameters
    ==========

    e : SymPy Matrix
        A Jones vector.

    Returns
    =======

    SymPy Matrix
        A Jones vector.

    Examples
    ========

    The axes on the Poincaré sphere.

    >>> from sympy import pprint, pi
    >>> from sympy.physics.optics.polarization import jones_vector
    >>> from sympy.physics.optics.polarization import jones_2_stokes
    >>> H = jones_vector(0, 0)
    >>> V = jones_vector(pi/2, 0)
    >>> D = jones_vector(pi/4, 0)
    >>> A = jones_vector(-pi/4, 0)
    >>> R = jones_vector(0, pi/4)
    >>> L = jones_vector(0, -pi/4)
    >>> pprint([jones_2_stokes(e) for e in [H, V, D, A, R, L]],
    ...         use_unicode=True)
    ⎡⎡1⎤  ⎡1 ⎤  ⎡1⎤  ⎡1 ⎤  ⎡1⎤  ⎡1 ⎤⎤
    ⎢⎢ ⎥  ⎢  ⎥  ⎢ ⎥  ⎢  ⎥  ⎢ ⎥  ⎢  ⎥⎥
    ⎢⎢1⎥  ⎢-1⎥  ⎢0⎥  ⎢0 ⎥  ⎢0⎥  ⎢0 ⎥⎥
    ⎢⎢ ⎥, ⎢  ⎥, ⎢ ⎥, ⎢  ⎥, ⎢ ⎥, ⎢  ⎥⎥
    ⎢⎢0⎥  ⎢0 ⎥  ⎢1⎥  ⎢-1⎥  ⎢0⎥  ⎢0 ⎥⎥
    ⎢⎢ ⎥  ⎢  ⎥  ⎢ ⎥  ⎢  ⎥  ⎢ ⎥  ⎢  ⎥⎥
    ⎣⎣0⎦  ⎣0 ⎦  ⎣0⎦  ⎣0 ⎦  ⎣1⎦  ⎣-1⎦⎦

    """
    ex, ey = e
    return Matrix([Abs(ex)**2 + Abs(ey)**2,
                   Abs(ex)**2 - Abs(ey)**2,
                   2*re(ex*ey.conjugate()),
                   -2*im(ex*ey.conjugate())])


def linear_polarizer(theta=0):
    """A linear polarizer Jones matrix with transmission axis at
    an angle ``theta``.

    Parameters
    ==========

    theta : numeric type or SymPy Symbol
        The angle of the transmission axis relative to the horizontal plane.

    Returns
    =======

    SymPy Matrix
        A Jones matrix representing the polarizer.

    Examples
    ========

    A generic polarizer.

    >>> from sympy import pprint, symbols
    >>> from sympy.physics.optics.polarization import linear_polarizer
    >>> theta = symbols("theta", real=True)
    >>> J = linear_polarizer(theta)
    >>> pprint(J, use_unicode=True)
    ⎡      2                     ⎤
    ⎢   cos (θ)     sin(θ)⋅cos(θ)⎥
    ⎢                            ⎥
    ⎢                     2      ⎥
    ⎣sin(θ)⋅cos(θ)     sin (θ)   ⎦


    """
    M = Matrix([[cos(theta)**2, sin(theta)*cos(theta)],
                [sin(theta)*cos(theta), sin(theta)**2]])
    return M


def phase_retarder(theta=0, delta=0):
    """A phase retarder Jones matrix with retardance ``delta`` at angle ``theta``.

    Parameters
    ==========

    theta : numeric type or SymPy Symbol
        The angle of the fast axis relative to the horizontal plane.
    delta : numeric type or SymPy Symbol
        The phase difference between the fast and slow axes of the
        transmitted light.

    Returns
    =======

    SymPy Matrix :
        A Jones matrix representing the retarder.

    Examples
    ========

    A generic retarder.

    >>> from sympy import pprint, symbols
    >>> from sympy.physics.optics.polarization import phase_retarder
    >>> theta, delta = symbols("theta, delta", real=True)
    >>> R = phase_retarder(theta, delta)
    >>> pprint(R, use_unicode=True)
    ⎡                          -ⅈ⋅δ               -ⅈ⋅δ               ⎤
    ⎢                          ─────              ─────              ⎥
    ⎢⎛ ⅈ⋅δ    2         2   ⎞    2    ⎛     ⅈ⋅δ⎞    2                ⎥
    ⎢⎝ℯ   ⋅sin (θ) + cos (θ)⎠⋅ℯ       ⎝1 - ℯ   ⎠⋅ℯ     ⋅sin(θ)⋅cos(θ)⎥
    ⎢                                                                ⎥
    ⎢            -ⅈ⋅δ                                           -ⅈ⋅δ ⎥
    ⎢            ─────                                          ─────⎥
    ⎢⎛     ⅈ⋅δ⎞    2                  ⎛ ⅈ⋅δ    2         2   ⎞    2  ⎥
    ⎣⎝1 - ℯ   ⎠⋅ℯ     ⋅sin(θ)⋅cos(θ)  ⎝ℯ   ⋅cos (θ) + sin (θ)⎠⋅ℯ     ⎦

    """
    R = Matrix([[cos(theta)**2 + exp(I*delta)*sin(theta)**2,
                (1-exp(I*delta))*cos(theta)*sin(theta)],
                [(1-exp(I*delta))*cos(theta)*sin(theta),
                sin(theta)**2 + exp(I*delta)*cos(theta)**2]])
    return R*exp(-I*delta/2)


def half_wave_retarder(theta):
    """A half-wave retarder Jones matrix at angle ``theta``.

    Parameters
    ==========

    theta : numeric type or SymPy Symbol
        The angle of the fast axis relative to the horizontal plane.

    Returns
    =======

    SymPy Matrix
        A Jones matrix representing the retarder.

    Examples
    ========

    A generic half-wave plate.

    >>> from sympy import pprint, symbols
    >>> from sympy.physics.optics.polarization import half_wave_retarder
    >>> theta= symbols("theta", real=True)
    >>> HWP = half_wave_retarder(theta)
    >>> pprint(HWP, use_unicode=True)
    ⎡   ⎛     2         2   ⎞                        ⎤
    ⎢-ⅈ⋅⎝- sin (θ) + cos (θ)⎠    -2⋅ⅈ⋅sin(θ)⋅cos(θ)  ⎥
    ⎢                                                ⎥
    ⎢                             ⎛   2         2   ⎞⎥
    ⎣   -2⋅ⅈ⋅sin(θ)⋅cos(θ)     -ⅈ⋅⎝sin (θ) - cos (θ)⎠⎦

    """
    return phase_retarder(theta, pi)


def quarter_wave_retarder(theta):
    """A quarter-wave retarder Jones matrix at angle ``theta``.

    Parameters
    ==========

    theta : numeric type or SymPy Symbol
        The angle of the fast axis relative to the horizontal plane.

    Returns
    =======

    SymPy Matrix
        A Jones matrix representing the retarder.

    Examples
    ========

    A generic quarter-wave plate.

    >>> from sympy import pprint, symbols
    >>> from sympy.physics.optics.polarization import quarter_wave_retarder
    >>> theta= symbols("theta", real=True)
    >>> QWP = quarter_wave_retarder(theta)
    >>> pprint(QWP, use_unicode=True)
    ⎡                       -ⅈ⋅π            -ⅈ⋅π               ⎤
    ⎢                       ─────           ─────              ⎥
    ⎢⎛     2         2   ⎞    4               4                ⎥
    ⎢⎝ⅈ⋅sin (θ) + cos (θ)⎠⋅ℯ       (1 - ⅈ)⋅ℯ     ⋅sin(θ)⋅cos(θ)⎥
    ⎢                                                          ⎥
    ⎢         -ⅈ⋅π                                        -ⅈ⋅π ⎥
    ⎢         ─────                                       ─────⎥
    ⎢           4                  ⎛   2           2   ⎞    4  ⎥
    ⎣(1 - ⅈ)⋅ℯ     ⋅sin(θ)⋅cos(θ)  ⎝sin (θ) + ⅈ⋅cos (θ)⎠⋅ℯ     ⎦

    """
    return phase_retarder(theta, pi/2)


def transmissive_filter(T):
    """An attenuator Jones matrix with transmittance ``T``.

    Parameters
    ==========

    T : numeric type or SymPy Symbol
        The transmittance of the attenuator.

    Returns
    =======

    SymPy Matrix
        A Jones matrix representing the filter.

    Examples
    ========

    A generic filter.

    >>> from sympy import pprint, symbols
    >>> from sympy.physics.optics.polarization import transmissive_filter
    >>> T = symbols("T", real=True)
    >>> NDF = transmissive_filter(T)
    >>> pprint(NDF, use_unicode=True)
    ⎡√T  0 ⎤
    ⎢      ⎥
    ⎣0   √T⎦

    """
    return Matrix([[sqrt(T), 0], [0, sqrt(T)]])


def reflective_filter(R):
    """A reflective filter Jones matrix with reflectance ``R``.

    Parameters
    ==========

    R : numeric type or SymPy Symbol
        The reflectance of the filter.

    Returns
    =======

    SymPy Matrix
        A Jones matrix representing the filter.

    Examples
    ========

    A generic filter.

    >>> from sympy import pprint, symbols
    >>> from sympy.physics.optics.polarization import reflective_filter
    >>> R = symbols("R", real=True)
    >>> pprint(reflective_filter(R), use_unicode=True)
    ⎡√R   0 ⎤
    ⎢       ⎥
    ⎣0   -√R⎦

    """
    return Matrix([[sqrt(R), 0], [0, -sqrt(R)]])


def mueller_matrix(J):
    """The Mueller matrix corresponding to Jones matrix `J`.

    Parameters
    ==========

    J : SymPy Matrix
        A Jones matrix.

    Returns
    =======

    SymPy Matrix
        The corresponding Mueller matrix.

    Examples
    ========

    Generic optical components.

    >>> from sympy import pprint, symbols
    >>> from sympy.physics.optics.polarization import (mueller_matrix,
    ...     linear_polarizer, half_wave_retarder, quarter_wave_retarder)
    >>> theta = symbols("theta", real=True)

    A linear_polarizer

    >>> pprint(mueller_matrix(linear_polarizer(theta)), use_unicode=True)
    ⎡            cos(2⋅θ)      sin(2⋅θ)     ⎤
    ⎢  1/2       ────────      ────────    0⎥
    ⎢               2             2         ⎥
    ⎢                                       ⎥
    ⎢cos(2⋅θ)  cos(4⋅θ)   1    sin(4⋅θ)     ⎥
    ⎢────────  ──────── + ─    ────────    0⎥
    ⎢   2         4       4       4         ⎥
    ⎢                                       ⎥
    ⎢sin(2⋅θ)    sin(4⋅θ)    1   cos(4⋅θ)   ⎥
    ⎢────────    ────────    ─ - ────────  0⎥
    ⎢   2           4        4      4       ⎥
    ⎢                                       ⎥
    ⎣   0           0             0        0⎦

    A half-wave plate

    >>> pprint(mueller_matrix(half_wave_retarder(theta)), use_unicode=True)
    ⎡1              0                           0               0 ⎤
    ⎢                                                             ⎥
    ⎢        4           2                                        ⎥
    ⎢0  8⋅sin (θ) - 8⋅sin (θ) + 1           sin(4⋅θ)            0 ⎥
    ⎢                                                             ⎥
    ⎢                                     4           2           ⎥
    ⎢0          sin(4⋅θ)           - 8⋅sin (θ) + 8⋅sin (θ) - 1  0 ⎥
    ⎢                                                             ⎥
    ⎣0              0                           0               -1⎦

    A quarter-wave plate

    >>> pprint(mueller_matrix(quarter_wave_retarder(theta)), use_unicode=True)
    ⎡1       0             0            0    ⎤
    ⎢                                        ⎥
    ⎢   cos(4⋅θ)   1    sin(4⋅θ)             ⎥
    ⎢0  ──────── + ─    ────────    -sin(2⋅θ)⎥
    ⎢      2       2       2                 ⎥
    ⎢                                        ⎥
    ⎢     sin(4⋅θ)    1   cos(4⋅θ)           ⎥
    ⎢0    ────────    ─ - ────────  cos(2⋅θ) ⎥
    ⎢        2        2      2               ⎥
    ⎢                                        ⎥
    ⎣0    sin(2⋅θ)     -cos(2⋅θ)        0    ⎦

    """
    A = Matrix([[1, 0, 0, 1],
                [1, 0, 0, -1],
                [0, 1, 1, 0],
                [0, -I, I, 0]])

    return simplify(A*TensorProduct(J, J.conjugate())*A.inv())


def polarizing_beam_splitter(Tp=1, Rs=1, Ts=0, Rp=0, phia=0, phib=0):
    r"""A polarizing beam splitter Jones matrix at angle `theta`.

    Parameters
    ==========

    J : SymPy Matrix
        A Jones matrix.
    Tp : numeric type or SymPy Symbol
        The transmissivity of the P-polarized component.
    Rs : numeric type or SymPy Symbol
        The reflectivity of the S-polarized component.
    Ts : numeric type or SymPy Symbol
        The transmissivity of the S-polarized component.
    Rp : numeric type or SymPy Symbol
        The reflectivity of the P-polarized component.
    phia : numeric type or SymPy Symbol
        The phase difference between transmitted and reflected component for
        output mode a.
    phib : numeric type or SymPy Symbol
        The phase difference between transmitted and reflected component for
        output mode b.


    Returns
    =======

    SymPy Matrix
        A 4x4 matrix representing the PBS. This matrix acts on a 4x1 vector
        whose first two entries are the Jones vector on one of the PBS ports,
        and the last two entries the Jones vector on the other port.

    Examples
    ========

    Generic polarizing beam-splitter.

    >>> from sympy import pprint, symbols
    >>> from sympy.physics.optics.polarization import polarizing_beam_splitter
    >>> Ts, Rs, Tp, Rp = symbols(r"Ts, Rs, Tp, Rp", positive=True)
    >>> phia, phib = symbols("phi_a, phi_b", real=True)
    >>> PBS = polarizing_beam_splitter(Tp, Rs, Ts, Rp, phia, phib)
    >>> pprint(PBS, use_unicode=False)
    [   ____                           ____                    ]
    [ \/ Tp            0           I*\/ Rp           0         ]
    [                                                          ]
    [                  ____                       ____  I*phi_a]
    [   0            \/ Ts            0      -I*\/ Rs *e       ]
    [                                                          ]
    [    ____                         ____                     ]
    [I*\/ Rp           0            \/ Tp            0         ]
    [                                                          ]
    [               ____  I*phi_b                    ____      ]
    [   0      -I*\/ Rs *e            0            \/ Ts       ]

    """
    PBS = Matrix([[sqrt(Tp), 0, I*sqrt(Rp), 0],
                  [0, sqrt(Ts), 0, -I*sqrt(Rs)*exp(I*phia)],
                  [I*sqrt(Rp), 0, sqrt(Tp), 0],
                  [0, -I*sqrt(Rs)*exp(I*phib), 0, sqrt(Ts)]])
    return PBS

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\geometry\util.py ===
"""Utility functions for geometrical entities.

Contains
========
intersection
convex_hull
closest_points
farthest_points
are_coplanar
are_similar

"""

from collections import deque
from math import sqrt as _sqrt

from sympy import nsimplify
from .entity import GeometryEntity
from .exceptions import GeometryError
from .point import Point, Point2D, Point3D
from sympy.core.containers import OrderedSet
from sympy.core.exprtools import factor_terms
from sympy.core.function import Function, expand_mul
from sympy.core.numbers import Float
from sympy.core.sorting import ordered
from sympy.core.symbol import Symbol
from sympy.core.singleton import S
from sympy.polys.polytools import cancel
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.utilities.iterables import is_sequence

from mpmath.libmp.libmpf import prec_to_dps


def find(x, equation):
    """
    Checks whether a Symbol matching ``x`` is present in ``equation``
    or not. If present, the matching symbol is returned, else a
    ValueError is raised. If ``x`` is a string the matching symbol
    will have the same name; if ``x`` is a Symbol then it will be
    returned if found.

    Examples
    ========

    >>> from sympy.geometry.util import find
    >>> from sympy import Dummy
    >>> from sympy.abc import x
    >>> find('x', x)
    x
    >>> find('x', Dummy('x'))
    _x

    The dummy symbol is returned since it has a matching name:

    >>> _.name == 'x'
    True
    >>> find(x, Dummy('x'))
    Traceback (most recent call last):
    ...
    ValueError: could not find x
    """

    free = equation.free_symbols
    xs = [i for i in free if (i.name if isinstance(x, str) else i) == x]
    if not xs:
        raise ValueError('could not find %s' % x)
    if len(xs) != 1:
        raise ValueError('ambiguous %s' % x)
    return xs[0]


def _ordered_points(p):
    """Return the tuple of points sorted numerically according to args"""
    return tuple(sorted(p, key=lambda x: x.args))


def are_coplanar(*e):
    """ Returns True if the given entities are coplanar otherwise False

    Parameters
    ==========

    e: entities to be checked for being coplanar

    Returns
    =======

    Boolean

    Examples
    ========

    >>> from sympy import Point3D, Line3D
    >>> from sympy.geometry.util import are_coplanar
    >>> a = Line3D(Point3D(5, 0, 0), Point3D(1, -1, 1))
    >>> b = Line3D(Point3D(0, -2, 0), Point3D(3, 1, 1))
    >>> c = Line3D(Point3D(0, -1, 0), Point3D(5, -1, 9))
    >>> are_coplanar(a, b, c)
    False

    """
    from .line import LinearEntity3D
    from .plane import Plane
    # XXX update tests for coverage

    e = set(e)
    # first work with a Plane if present
    for i in list(e):
        if isinstance(i, Plane):
            e.remove(i)
            return all(p.is_coplanar(i) for p in e)

    if all(isinstance(i, Point3D) for i in e):
        if len(e) < 3:
            return False

        # remove pts that are collinear with 2 pts
        a, b = e.pop(), e.pop()
        for i in list(e):
            if Point3D.are_collinear(a, b, i):
                e.remove(i)

        if not e:
            return False
        else:
            # define a plane
            p = Plane(a, b, e.pop())
            for i in e:
                if i not in p:
                    return False
            return True
    else:
        pt3d = []
        for i in e:
            if isinstance(i, Point3D):
                pt3d.append(i)
            elif isinstance(i, LinearEntity3D):
                pt3d.extend(i.args)
            elif isinstance(i, GeometryEntity):  # XXX we should have a GeometryEntity3D class so we can tell the difference between 2D and 3D -- here we just want to deal with 2D objects; if new 3D objects are encountered that we didn't handle above, an error should be raised
                # all 2D objects have some Point that defines them; so convert those points to 3D pts by making z=0
                for p in i.args:
                    if isinstance(p, Point):
                        pt3d.append(Point3D(*(p.args + (0,))))
        return are_coplanar(*pt3d)


def are_similar(e1, e2):
    """Are two geometrical entities similar.

    Can one geometrical entity be uniformly scaled to the other?

    Parameters
    ==========

    e1 : GeometryEntity
    e2 : GeometryEntity

    Returns
    =======

    are_similar : boolean

    Raises
    ======

    GeometryError
        When `e1` and `e2` cannot be compared.

    Notes
    =====

    If the two objects are equal then they are similar.

    See Also
    ========

    sympy.geometry.entity.GeometryEntity.is_similar

    Examples
    ========

    >>> from sympy import Point, Circle, Triangle, are_similar
    >>> c1, c2 = Circle(Point(0, 0), 4), Circle(Point(1, 4), 3)
    >>> t1 = Triangle(Point(0, 0), Point(1, 0), Point(0, 1))
    >>> t2 = Triangle(Point(0, 0), Point(2, 0), Point(0, 2))
    >>> t3 = Triangle(Point(0, 0), Point(3, 0), Point(0, 1))
    >>> are_similar(t1, t2)
    True
    >>> are_similar(t1, t3)
    False

    """
    if e1 == e2:
        return True
    is_similar1 = getattr(e1, 'is_similar', None)
    if is_similar1:
        return is_similar1(e2)
    is_similar2 = getattr(e2, 'is_similar', None)
    if is_similar2:
        return is_similar2(e1)
    n1 = e1.__class__.__name__
    n2 = e2.__class__.__name__
    raise GeometryError(
        "Cannot test similarity between %s and %s" % (n1, n2))


def centroid(*args):
    """Find the centroid (center of mass) of the collection containing only Points,
    Segments or Polygons. The centroid is the weighted average of the individual centroid
    where the weights are the lengths (of segments) or areas (of polygons).
    Overlapping regions will add to the weight of that region.

    If there are no objects (or a mixture of objects) then None is returned.

    See Also
    ========

    sympy.geometry.point.Point, sympy.geometry.line.Segment,
    sympy.geometry.polygon.Polygon

    Examples
    ========

    >>> from sympy import Point, Segment, Polygon
    >>> from sympy.geometry.util import centroid
    >>> p = Polygon((0, 0), (10, 0), (10, 10))
    >>> q = p.translate(0, 20)
    >>> p.centroid, q.centroid
    (Point2D(20/3, 10/3), Point2D(20/3, 70/3))
    >>> centroid(p, q)
    Point2D(20/3, 40/3)
    >>> p, q = Segment((0, 0), (2, 0)), Segment((0, 0), (2, 2))
    >>> centroid(p, q)
    Point2D(1, 2 - sqrt(2))
    >>> centroid(Point(0, 0), Point(2, 0))
    Point2D(1, 0)

    Stacking 3 polygons on top of each other effectively triples the
    weight of that polygon:

    >>> p = Polygon((0, 0), (1, 0), (1, 1), (0, 1))
    >>> q = Polygon((1, 0), (3, 0), (3, 1), (1, 1))
    >>> centroid(p, q)
    Point2D(3/2, 1/2)
    >>> centroid(p, p, p, q) # centroid x-coord shifts left
    Point2D(11/10, 1/2)

    Stacking the squares vertically above and below p has the same
    effect:

    >>> centroid(p, p.translate(0, 1), p.translate(0, -1), q)
    Point2D(11/10, 1/2)

    """
    from .line import Segment
    from .polygon import Polygon
    if args:
        if all(isinstance(g, Point) for g in args):
            c = Point(0, 0)
            for g in args:
                c += g
            den = len(args)
        elif all(isinstance(g, Segment) for g in args):
            c = Point(0, 0)
            L = 0
            for g in args:
                l = g.length
                c += g.midpoint*l
                L += l
            den = L
        elif all(isinstance(g, Polygon) for g in args):
            c = Point(0, 0)
            A = 0
            for g in args:
                a = g.area
                c += g.centroid*a
                A += a
            den = A
        c /= den
        return c.func(*[i.simplify() for i in c.args])


def closest_points(*args):
    """Return the subset of points from a set of points that were
    the closest to each other in the 2D plane.

    Parameters
    ==========

    args
        A collection of Points on 2D plane.

    Notes
    =====

    This can only be performed on a set of points whose coordinates can
    be ordered on the number line. If there are no ties then a single
    pair of Points will be in the set.

    Examples
    ========

    >>> from sympy import closest_points, Triangle
    >>> Triangle(sss=(3, 4, 5)).args
    (Point2D(0, 0), Point2D(3, 0), Point2D(3, 4))
    >>> closest_points(*_)
    {(Point2D(0, 0), Point2D(3, 0))}

    References
    ==========

    .. [1] https://www.cs.mcgill.ca/~cs251/ClosestPair/ClosestPairPS.html

    .. [2] Sweep line algorithm
        https://en.wikipedia.org/wiki/Sweep_line_algorithm

    """
    p = [Point2D(i) for i in set(args)]
    if len(p) < 2:
        raise ValueError('At least 2 distinct points must be given.')

    try:
        p.sort(key=lambda x: x.args)
    except TypeError:
        raise ValueError("The points could not be sorted.")

    if not all(i.is_Rational for j in p for i in j.args):
        def hypot(x, y):
            arg = x*x + y*y
            if arg.is_Rational:
                return _sqrt(arg)
            return sqrt(arg)
    else:
        from math import hypot

    rv = [(0, 1)]
    best_dist = hypot(p[1].x - p[0].x, p[1].y - p[0].y)
    left = 0
    box = deque([0, 1])
    for i in range(2, len(p)):
        while left < i and p[i][0] - p[left][0] > best_dist:
            box.popleft()
            left += 1

        for j in box:
            d = hypot(p[i].x - p[j].x, p[i].y - p[j].y)
            if d < best_dist:
                rv = [(j, i)]
            elif d == best_dist:
                rv.append((j, i))
            else:
                continue
            best_dist = d
        box.append(i)

    return {tuple([p[i] for i in pair]) for pair in rv}


def convex_hull(*args, polygon=True):
    """The convex hull surrounding the Points contained in the list of entities.

    Parameters
    ==========

    args : a collection of Points, Segments and/or Polygons

    Optional parameters
    ===================

    polygon : Boolean. If True, returns a Polygon, if false a tuple, see below.
              Default is True.

    Returns
    =======

    convex_hull : Polygon if ``polygon`` is True else as a tuple `(U, L)` where
                  ``L`` and ``U`` are the lower and upper hulls, respectively.

    Notes
    =====

    This can only be performed on a set of points whose coordinates can
    be ordered on the number line.

    See Also
    ========

    sympy.geometry.point.Point, sympy.geometry.polygon.Polygon

    Examples
    ========

    >>> from sympy import convex_hull
    >>> points = [(1, 1), (1, 2), (3, 1), (-5, 2), (15, 4)]
    >>> convex_hull(*points)
    Polygon(Point2D(-5, 2), Point2D(1, 1), Point2D(3, 1), Point2D(15, 4))
    >>> convex_hull(*points, **dict(polygon=False))
    ([Point2D(-5, 2), Point2D(15, 4)],
     [Point2D(-5, 2), Point2D(1, 1), Point2D(3, 1), Point2D(15, 4)])

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Graham_scan

    .. [2] Andrew's Monotone Chain Algorithm
      (A.M. Andrew,
      "Another Efficient Algorithm for Convex Hulls in Two Dimensions", 1979)
      https://web.archive.org/web/20210511015444/http://geomalgorithms.com/a10-_hull-1.html

    """
    from .line import Segment
    from .polygon import Polygon
    p = OrderedSet()
    for e in args:
        if not isinstance(e, GeometryEntity):
            try:
                e = Point(e)
            except NotImplementedError:
                raise ValueError('%s is not a GeometryEntity and cannot be made into Point' % str(e))
        if isinstance(e, Point):
            p.add(e)
        elif isinstance(e, Segment):
            p.update(e.points)
        elif isinstance(e, Polygon):
            p.update(e.vertices)
        else:
            raise NotImplementedError(
                'Convex hull for %s not implemented.' % type(e))

    # make sure all our points are of the same dimension
    if any(len(x) != 2 for x in p):
        raise ValueError('Can only compute the convex hull in two dimensions')

    p = list(p)
    if len(p) == 1:
        return p[0] if polygon else (p[0], None)
    elif len(p) == 2:
        s = Segment(p[0], p[1])
        return s if polygon else (s, None)

    def _orientation(p, q, r):
        '''Return positive if p-q-r are clockwise, neg if ccw, zero if
        collinear.'''
        return (q.y - p.y)*(r.x - p.x) - (q.x - p.x)*(r.y - p.y)

    # scan to find upper and lower convex hulls of a set of 2d points.
    U = []
    L = []
    try:
        p.sort(key=lambda x: x.args)
    except TypeError:
        raise ValueError("The points could not be sorted.")
    for p_i in p:
        while len(U) > 1 and _orientation(U[-2], U[-1], p_i) <= 0:
            U.pop()
        while len(L) > 1 and _orientation(L[-2], L[-1], p_i) >= 0:
            L.pop()
        U.append(p_i)
        L.append(p_i)
    U.reverse()
    convexHull = tuple(L + U[1:-1])

    if len(convexHull) == 2:
        s = Segment(convexHull[0], convexHull[1])
        return s if polygon else (s, None)
    if polygon:
        return Polygon(*convexHull)
    else:
        U.reverse()
        return (U, L)

def farthest_points(*args):
    """Return the subset of points from a set of points that were
    the furthest apart from each other in the 2D plane.

    Parameters
    ==========

    args
        A collection of Points on 2D plane.

    Notes
    =====

    This can only be performed on a set of points whose coordinates can
    be ordered on the number line. If there are no ties then a single
    pair of Points will be in the set.

    Examples
    ========

    >>> from sympy.geometry import farthest_points, Triangle
    >>> Triangle(sss=(3, 4, 5)).args
    (Point2D(0, 0), Point2D(3, 0), Point2D(3, 4))
    >>> farthest_points(*_)
    {(Point2D(0, 0), Point2D(3, 4))}

    References
    ==========

    .. [1] https://code.activestate.com/recipes/117225-convex-hull-and-diameter-of-2d-point-sets/

    .. [2] Rotating Callipers Technique
        https://en.wikipedia.org/wiki/Rotating_calipers

    """

    def rotatingCalipers(Points):
        U, L = convex_hull(*Points, **{"polygon": False})

        if L is None:
            if isinstance(U, Point):
                raise ValueError('At least two distinct points must be given.')
            yield U.args
        else:
            i = 0
            j = len(L) - 1
            while i < len(U) - 1 or j > 0:
                yield U[i], L[j]
                # if all the way through one side of hull, advance the other side
                if i == len(U) - 1:
                    j -= 1
                elif j == 0:
                    i += 1
                # still points left on both lists, compare slopes of next hull edges
                # being careful to avoid divide-by-zero in slope calculation
                elif (U[i+1].y - U[i].y) * (L[j].x - L[j-1].x) > \
                        (L[j].y - L[j-1].y) * (U[i+1].x - U[i].x):
                    i += 1
                else:
                    j -= 1

    p = [Point2D(i) for i in set(args)]

    if not all(i.is_Rational for j in p for i in j.args):
        def hypot(x, y):
            arg = x*x + y*y
            if arg.is_Rational:
                return _sqrt(arg)
            return sqrt(arg)
    else:
        from math import hypot

    rv = []
    diam = 0
    for pair in rotatingCalipers(args):
        h, q = _ordered_points(pair)
        d = hypot(h.x - q.x, h.y - q.y)
        if d > diam:
            rv = [(h, q)]
        elif d == diam:
            rv.append((h, q))
        else:
            continue
        diam = d

    return set(rv)


def idiff(eq, y, x, n=1):
    """Return ``dy/dx`` assuming that ``eq == 0``.

    Parameters
    ==========

    y : the dependent variable or a list of dependent variables (with y first)
    x : the variable that the derivative is being taken with respect to
    n : the order of the derivative (default is 1)

    Examples
    ========

    >>> from sympy.abc import x, y, a
    >>> from sympy.geometry.util import idiff

    >>> circ = x**2 + y**2 - 4
    >>> idiff(circ, y, x)
    -x/y
    >>> idiff(circ, y, x, 2).simplify()
    (-x**2 - y**2)/y**3

    Here, ``a`` is assumed to be independent of ``x``:

    >>> idiff(x + a + y, y, x)
    -1

    Now the x-dependence of ``a`` is made explicit by listing ``a`` after
    ``y`` in a list.

    >>> idiff(x + a + y, [y, a], x)
    -Derivative(a, x) - 1

    See Also
    ========

    sympy.core.function.Derivative: represents unevaluated derivatives
    sympy.core.function.diff: explicitly differentiates wrt symbols

    """
    if is_sequence(y):
        dep = set(y)
        y = y[0]
    elif isinstance(y, Symbol):
        dep = {y}
    elif isinstance(y, Function):
        pass
    else:
        raise ValueError("expecting x-dependent symbol(s) or function(s) but got: %s" % y)

    f = {s: Function(s.name)(x) for s in eq.free_symbols
        if s != x and s in dep}

    if isinstance(y, Symbol):
        dydx = Function(y.name)(x).diff(x)
    else:
        dydx = y.diff(x)

    eq = eq.subs(f)
    derivs = {}
    for i in range(n):
        # equation will be linear in dydx, a*dydx + b, so dydx = -b/a
        deq = eq.diff(x)
        b = deq.xreplace({dydx: S.Zero})
        a = (deq - b).xreplace({dydx: S.One})
        yp = factor_terms(expand_mul(cancel((-b/a).subs(derivs)), deep=False))
        if i == n - 1:
            return yp.subs([(v, k) for k, v in f.items()])
        derivs[dydx] = yp
        eq = dydx - yp
        dydx = dydx.diff(x)


def intersection(*entities, pairwise=False, **kwargs):
    """The intersection of a collection of GeometryEntity instances.

    Parameters
    ==========
    entities : sequence of GeometryEntity
    pairwise (keyword argument) : Can be either True or False

    Returns
    =======
    intersection : list of GeometryEntity

    Raises
    ======
    NotImplementedError
        When unable to calculate intersection.

    Notes
    =====
    The intersection of any geometrical entity with itself should return
    a list with one item: the entity in question.
    An intersection requires two or more entities. If only a single
    entity is given then the function will return an empty list.
    It is possible for `intersection` to miss intersections that one
    knows exists because the required quantities were not fully
    simplified internally.
    Reals should be converted to Rationals, e.g. Rational(str(real_num))
    or else failures due to floating point issues may result.

    Case 1: When the keyword argument 'pairwise' is False (default value):
    In this case, the function returns a list of intersections common to
    all entities.

    Case 2: When the keyword argument 'pairwise' is True:
    In this case, the functions returns a list intersections that occur
    between any pair of entities.

    See Also
    ========

    sympy.geometry.entity.GeometryEntity.intersection

    Examples
    ========

    >>> from sympy import Ray, Circle, intersection
    >>> c = Circle((0, 1), 1)
    >>> intersection(c, c.center)
    []
    >>> right = Ray((0, 0), (1, 0))
    >>> up = Ray((0, 0), (0, 1))
    >>> intersection(c, right, up)
    [Point2D(0, 0)]
    >>> intersection(c, right, up, pairwise=True)
    [Point2D(0, 0), Point2D(0, 2)]
    >>> left = Ray((1, 0), (0, 0))
    >>> intersection(right, left)
    [Segment2D(Point2D(0, 0), Point2D(1, 0))]

    """
    if len(entities) <= 1:
        return []

    entities = list(entities)
    prec = None
    for i, e in enumerate(entities):
        if not isinstance(e, GeometryEntity):
            # entities may be an immutable tuple
            e = Point(e)
        # convert to exact Rationals
        d = {}
        for f in e.atoms(Float):
            prec = f._prec if prec is None else min(f._prec, prec)
            d.setdefault(f, nsimplify(f, rational=True))
        entities[i] = e.xreplace(d)

    if not pairwise:
        # find the intersection common to all objects
        res = entities[0].intersection(entities[1])
        for entity in entities[2:]:
            newres = []
            for x in res:
                newres.extend(x.intersection(entity))
            res = newres
    else:
        # find all pairwise intersections
        ans = []
        for j in range(len(entities)):
            for k in range(j + 1, len(entities)):
                ans.extend(intersection(entities[j], entities[k]))
        res = list(ordered(set(ans)))

    # convert back to Floats
    if prec is not None:
        p = prec_to_dps(prec)
        res = [i.n(p) for i in res]
    return res

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\printing\pretty\pretty_symbology.py ===
"""Symbolic primitives + unicode/ASCII abstraction for pretty.py"""

import sys
import warnings
from string import ascii_lowercase, ascii_uppercase
import unicodedata

unicode_warnings = ''

def U(name):
    """
    Get a unicode character by name or, None if not found.

    This exists because older versions of Python use older unicode databases.
    """
    try:
        return unicodedata.lookup(name)
    except KeyError:
        global unicode_warnings
        unicode_warnings += 'No \'%s\' in unicodedata\n' % name
        return None

from sympy.printing.conventions import split_super_sub
from sympy.core.alphabets import greeks
from sympy.utilities.exceptions import sympy_deprecation_warning

# prefix conventions when constructing tables
# L   - LATIN     i
# G   - GREEK     beta
# D   - DIGIT     0
# S   - SYMBOL    +


__all__ = ['greek_unicode', 'sub', 'sup', 'xsym', 'vobj', 'hobj', 'pretty_symbol',
           'annotated', 'center_pad', 'center']


_use_unicode = False


def pretty_use_unicode(flag=None):
    """Set whether pretty-printer should use unicode by default"""
    global _use_unicode, unicode_warnings
    if flag is None:
        return _use_unicode

    if flag and unicode_warnings:
        # print warnings (if any) on first unicode usage
        warnings.warn(unicode_warnings)
        unicode_warnings = ''

    use_unicode_prev = _use_unicode
    _use_unicode = flag
    return use_unicode_prev


def pretty_try_use_unicode():
    """See if unicode output is available and leverage it if possible"""

    encoding = getattr(sys.stdout, 'encoding', None)

    # this happens when e.g. stdout is redirected through a pipe, or is
    # e.g. a cStringIO.StringO
    if encoding is None:
        return  # sys.stdout has no encoding

    symbols = []

    # see if we can represent greek alphabet
    symbols += greek_unicode.values()

    # and atoms
    symbols += atoms_table.values()

    for s in symbols:
        if s is None:
            return  # common symbols not present!

        try:
            s.encode(encoding)
        except UnicodeEncodeError:
            return

    # all the characters were present and encodable
    pretty_use_unicode(True)


def xstr(*args):
    sympy_deprecation_warning(
        """
        The sympy.printing.pretty.pretty_symbology.xstr() function is
        deprecated. Use str() instead.
        """,
        deprecated_since_version="1.7",
        active_deprecations_target="deprecated-pretty-printing-functions"
    )
    return str(*args)

# GREEK
g = lambda l: U('GREEK SMALL LETTER %s' % l.upper())
G = lambda l: U('GREEK CAPITAL LETTER %s' % l.upper())

greek_letters = list(greeks) # make a copy
# deal with Unicode's funny spelling of lambda
greek_letters[greek_letters.index('lambda')] = 'lamda'

# {}  greek letter -> (g,G)
greek_unicode = {L: g(L) for L in greek_letters}
greek_unicode.update((L[0].upper() + L[1:], G(L)) for L in greek_letters)

# aliases
greek_unicode['lambda'] = greek_unicode['lamda']
greek_unicode['Lambda'] = greek_unicode['Lamda']
greek_unicode['varsigma'] = '\N{GREEK SMALL LETTER FINAL SIGMA}'

# BOLD
b = lambda l: U('MATHEMATICAL BOLD SMALL %s' % l.upper())
B = lambda l: U('MATHEMATICAL BOLD CAPITAL %s' % l.upper())

bold_unicode = {l: b(l) for l in ascii_lowercase}
bold_unicode.update((L, B(L)) for L in ascii_uppercase)

# GREEK BOLD
gb = lambda l: U('MATHEMATICAL BOLD SMALL %s' % l.upper())
GB = lambda l: U('MATHEMATICAL BOLD CAPITAL  %s' % l.upper())

greek_bold_letters = list(greeks) # make a copy, not strictly required here
# deal with Unicode's funny spelling of lambda
greek_bold_letters[greek_bold_letters.index('lambda')] = 'lamda'

# {}  greek letter -> (g,G)
greek_bold_unicode = {L: g(L) for L in greek_bold_letters}
greek_bold_unicode.update((L[0].upper() + L[1:], G(L)) for L in greek_bold_letters)
greek_bold_unicode['lambda'] = greek_unicode['lamda']
greek_bold_unicode['Lambda'] = greek_unicode['Lamda']
greek_bold_unicode['varsigma'] = '\N{MATHEMATICAL BOLD SMALL FINAL SIGMA}'

digit_2txt = {
    '0':    'ZERO',
    '1':    'ONE',
    '2':    'TWO',
    '3':    'THREE',
    '4':    'FOUR',
    '5':    'FIVE',
    '6':    'SIX',
    '7':    'SEVEN',
    '8':    'EIGHT',
    '9':    'NINE',
}

symb_2txt = {
    '+':    'PLUS SIGN',
    '-':    'MINUS',
    '=':    'EQUALS SIGN',
    '(':    'LEFT PARENTHESIS',
    ')':    'RIGHT PARENTHESIS',
    '[':    'LEFT SQUARE BRACKET',
    ']':    'RIGHT SQUARE BRACKET',
    '{':    'LEFT CURLY BRACKET',
    '}':    'RIGHT CURLY BRACKET',

    # non-std
    '{}':   'CURLY BRACKET',
    'sum':  'SUMMATION',
    'int':  'INTEGRAL',
}

# SUBSCRIPT & SUPERSCRIPT
LSUB = lambda letter: U('LATIN SUBSCRIPT SMALL LETTER %s' % letter.upper())
GSUB = lambda letter: U('GREEK SUBSCRIPT SMALL LETTER %s' % letter.upper())
DSUB = lambda digit:  U('SUBSCRIPT %s' % digit_2txt[digit])
SSUB = lambda symb:   U('SUBSCRIPT %s' % symb_2txt[symb])

LSUP = lambda letter: U('SUPERSCRIPT LATIN SMALL LETTER %s' % letter.upper())
DSUP = lambda digit:  U('SUPERSCRIPT %s' % digit_2txt[digit])
SSUP = lambda symb:   U('SUPERSCRIPT %s' % symb_2txt[symb])

sub = {}    # symb -> subscript symbol
sup = {}    # symb -> superscript symbol

# latin subscripts
for l in 'aeioruvxhklmnpst':
    sub[l] = LSUB(l)

for l in 'in':
    sup[l] = LSUP(l)

for gl in ['beta', 'gamma', 'rho', 'phi', 'chi']:
    sub[gl] = GSUB(gl)

for d in [str(i) for i in range(10)]:
    sub[d] = DSUB(d)
    sup[d] = DSUP(d)

for s in '+-=()':
    sub[s] = SSUB(s)
    sup[s] = SSUP(s)

# Variable modifiers
# TODO: Make brackets adjust to height of contents
modifier_dict = {
    # Accents
    'mathring': lambda s: center_accent(s, '\N{COMBINING RING ABOVE}'),
    'ddddot': lambda s: center_accent(s, '\N{COMBINING FOUR DOTS ABOVE}'),
    'dddot': lambda s: center_accent(s, '\N{COMBINING THREE DOTS ABOVE}'),
    'ddot': lambda s: center_accent(s, '\N{COMBINING DIAERESIS}'),
    'dot': lambda s: center_accent(s, '\N{COMBINING DOT ABOVE}'),
    'check': lambda s: center_accent(s, '\N{COMBINING CARON}'),
    'breve': lambda s: center_accent(s, '\N{COMBINING BREVE}'),
    'acute': lambda s: center_accent(s, '\N{COMBINING ACUTE ACCENT}'),
    'grave': lambda s: center_accent(s, '\N{COMBINING GRAVE ACCENT}'),
    'tilde': lambda s: center_accent(s, '\N{COMBINING TILDE}'),
    'hat': lambda s: center_accent(s, '\N{COMBINING CIRCUMFLEX ACCENT}'),
    'bar': lambda s: center_accent(s, '\N{COMBINING OVERLINE}'),
    'vec': lambda s: center_accent(s, '\N{COMBINING RIGHT ARROW ABOVE}'),
    'prime': lambda s: s+'\N{PRIME}',
    'prm': lambda s: s+'\N{PRIME}',
    # # Faces -- these are here for some compatibility with latex printing
    # 'bold': lambda s: s,
    # 'bm': lambda s: s,
    # 'cal': lambda s: s,
    # 'scr': lambda s: s,
    # 'frak': lambda s: s,
    # Brackets
    'norm': lambda s: '\N{DOUBLE VERTICAL LINE}'+s+'\N{DOUBLE VERTICAL LINE}',
    'avg': lambda s: '\N{MATHEMATICAL LEFT ANGLE BRACKET}'+s+'\N{MATHEMATICAL RIGHT ANGLE BRACKET}',
    'abs': lambda s: '\N{VERTICAL LINE}'+s+'\N{VERTICAL LINE}',
    'mag': lambda s: '\N{VERTICAL LINE}'+s+'\N{VERTICAL LINE}',
}

# VERTICAL OBJECTS
HUP = lambda symb: U('%s UPPER HOOK' % symb_2txt[symb])
CUP = lambda symb: U('%s UPPER CORNER' % symb_2txt[symb])
MID = lambda symb: U('%s MIDDLE PIECE' % symb_2txt[symb])
EXT = lambda symb: U('%s EXTENSION' % symb_2txt[symb])
HLO = lambda symb: U('%s LOWER HOOK' % symb_2txt[symb])
CLO = lambda symb: U('%s LOWER CORNER' % symb_2txt[symb])
TOP = lambda symb: U('%s TOP' % symb_2txt[symb])
BOT = lambda symb: U('%s BOTTOM' % symb_2txt[symb])

# {} '('  ->  (extension, start, end, middle) 1-character
_xobj_unicode = {

    # vertical symbols
    #                       (( ext, top, bot, mid ), c1)
    '(':                    (( EXT('('), HUP('('), HLO('(') ), '('),
    ')':                    (( EXT(')'), HUP(')'), HLO(')') ), ')'),
    '[':                    (( EXT('['), CUP('['), CLO('[') ), '['),
    ']':                    (( EXT(']'), CUP(']'), CLO(']') ), ']'),
    '{':                    (( EXT('{}'), HUP('{'), HLO('{'), MID('{') ), '{'),
    '}':                    (( EXT('{}'), HUP('}'), HLO('}'), MID('}') ), '}'),
    '|':                    U('BOX DRAWINGS LIGHT VERTICAL'),
    'Tee':                  U('BOX DRAWINGS LIGHT UP AND HORIZONTAL'),
    'UpTack':               U('BOX DRAWINGS LIGHT DOWN AND HORIZONTAL'),
    'corner_up_centre'
    '(_ext':                U('LEFT PARENTHESIS EXTENSION'),
    ')_ext':                U('RIGHT PARENTHESIS EXTENSION'),
    '(_lower_hook':         U('LEFT PARENTHESIS LOWER HOOK'),
    ')_lower_hook':         U('RIGHT PARENTHESIS LOWER HOOK'),
    '(_upper_hook':         U('LEFT PARENTHESIS UPPER HOOK'),
    ')_upper_hook':         U('RIGHT PARENTHESIS UPPER HOOK'),
    '<':                  ((U('BOX DRAWINGS LIGHT VERTICAL'),
                            U('BOX DRAWINGS LIGHT DIAGONAL UPPER RIGHT TO LOWER LEFT'),
                            U('BOX DRAWINGS LIGHT DIAGONAL UPPER LEFT TO LOWER RIGHT')), '<'),

    '>':                  ((U('BOX DRAWINGS LIGHT VERTICAL'),
                            U('BOX DRAWINGS LIGHT DIAGONAL UPPER LEFT TO LOWER RIGHT'),
                            U('BOX DRAWINGS LIGHT DIAGONAL UPPER RIGHT TO LOWER LEFT')), '>'),

    'lfloor':               (( EXT('['), EXT('['), CLO('[') ), U('LEFT FLOOR')),
    'rfloor':               (( EXT(']'), EXT(']'), CLO(']') ), U('RIGHT FLOOR')),
    'lceil':                (( EXT('['), CUP('['), EXT('[') ), U('LEFT CEILING')),
    'rceil':                (( EXT(']'), CUP(']'), EXT(']') ), U('RIGHT CEILING')),

    'int':                  (( EXT('int'), U('TOP HALF INTEGRAL'), U('BOTTOM HALF INTEGRAL') ), U('INTEGRAL')),
    'sum':                  (( U('BOX DRAWINGS LIGHT DIAGONAL UPPER LEFT TO LOWER RIGHT'), '_', U('OVERLINE'), U('BOX DRAWINGS LIGHT DIAGONAL UPPER RIGHT TO LOWER LEFT')), U('N-ARY SUMMATION')),

    # horizontal objects
    #'-':   '-',
    '-':    U('BOX DRAWINGS LIGHT HORIZONTAL'),
    '_':    U('LOW LINE'),
    # We used to use this, but LOW LINE looks better for roots, as it's a
    # little lower (i.e., it lines up with the / perfectly.  But perhaps this
    # one would still be wanted for some cases?
    # '_':    U('HORIZONTAL SCAN LINE-9'),

    # diagonal objects '\' & '/' ?
    '/':    U('BOX DRAWINGS LIGHT DIAGONAL UPPER RIGHT TO LOWER LEFT'),
    '\\':   U('BOX DRAWINGS LIGHT DIAGONAL UPPER LEFT TO LOWER RIGHT'),
}

_xobj_ascii = {
    # vertical symbols
    #       (( ext, top, bot, mid ), c1)
    '(':    (( '|', '/', '\\' ), '('),
    ')':    (( '|', '\\', '/' ), ')'),

# XXX this looks ugly
#   '[':    (( '|', '-', '-' ), '['),
#   ']':    (( '|', '-', '-' ), ']'),
# XXX not so ugly :(
    '[':    (( '[', '[', '[' ), '['),
    ']':    (( ']', ']', ']' ), ']'),

    '{':    (( '|', '/', '\\', '<' ), '{'),
    '}':    (( '|', '\\', '/', '>' ), '}'),
    '|':    '|',

    '<':    (( '|', '/', '\\' ), '<'),
    '>':    (( '|', '\\', '/' ), '>'),

    'int':  ( ' | ', '  /', '/  ' ),

    # horizontal objects
    '-':    '-',
    '_':    '_',

    # diagonal objects '\' & '/' ?
    '/':    '/',
    '\\':   '\\',
}


def xobj(symb, length):
    """Construct spatial object of given length.

    return: [] of equal-length strings
    """

    if length <= 0:
        raise ValueError("Length should be greater than 0")

    # TODO robustify when no unicodedat available
    if _use_unicode:
        _xobj = _xobj_unicode
    else:
        _xobj = _xobj_ascii

    vinfo = _xobj[symb]

    c1 = top = bot = mid = None

    if not isinstance(vinfo, tuple):        # 1 entry
        ext = vinfo
    else:
        if isinstance(vinfo[0], tuple):     # (vlong), c1
            vlong = vinfo[0]
            c1 = vinfo[1]
        else:                               # (vlong), c1
            vlong = vinfo

        ext = vlong[0]

        try:
            top = vlong[1]
            bot = vlong[2]
            mid = vlong[3]
        except IndexError:
            pass

    if c1 is None:
        c1 = ext
    if top is None:
        top = ext
    if bot is None:
        bot = ext
    if mid is not None:
        if (length % 2) == 0:
            # even height, but we have to print it somehow anyway...
            # XXX is it ok?
            length += 1

    else:
        mid = ext

    if length == 1:
        return c1

    res = []
    next = (length - 2)//2
    nmid = (length - 2) - next*2

    res += [top]
    res += [ext]*next
    res += [mid]*nmid
    res += [ext]*next
    res += [bot]

    return res


def vobj(symb, height):
    """Construct vertical object of a given height

       see: xobj
    """
    return '\n'.join( xobj(symb, height) )


def hobj(symb, width):
    """Construct horizontal object of a given width

       see: xobj
    """
    return ''.join( xobj(symb, width) )

# RADICAL
# n -> symbol
root = {
    2: U('SQUARE ROOT'),   # U('RADICAL SYMBOL BOTTOM')
    3: U('CUBE ROOT'),
    4: U('FOURTH ROOT'),
}


# RATIONAL
VF = lambda txt: U('VULGAR FRACTION %s' % txt)

# (p,q) -> symbol
frac = {
    (1, 2): VF('ONE HALF'),
    (1, 3): VF('ONE THIRD'),
    (2, 3): VF('TWO THIRDS'),
    (1, 4): VF('ONE QUARTER'),
    (3, 4): VF('THREE QUARTERS'),
    (1, 5): VF('ONE FIFTH'),
    (2, 5): VF('TWO FIFTHS'),
    (3, 5): VF('THREE FIFTHS'),
    (4, 5): VF('FOUR FIFTHS'),
    (1, 6): VF('ONE SIXTH'),
    (5, 6): VF('FIVE SIXTHS'),
    (1, 8): VF('ONE EIGHTH'),
    (3, 8): VF('THREE EIGHTHS'),
    (5, 8): VF('FIVE EIGHTHS'),
    (7, 8): VF('SEVEN EIGHTHS'),
}


# atom symbols
_xsym = {
    '==':  ('=', '='),
    '<':   ('<', '<'),
    '>':   ('>', '>'),
    '<=':  ('<=', U('LESS-THAN OR EQUAL TO')),
    '>=':  ('>=', U('GREATER-THAN OR EQUAL TO')),
    '!=':  ('!=', U('NOT EQUAL TO')),
    ':=':  (':=', ':='),
    '+=':  ('+=', '+='),
    '-=':  ('-=', '-='),
    '*=':  ('*=', '*='),
    '/=':  ('/=', '/='),
    '%=':  ('%=', '%='),
    '*':   ('*', U('DOT OPERATOR')),
    '-->': ('-->', U('EM DASH') + U('EM DASH') +
            U('BLACK RIGHT-POINTING TRIANGLE') if U('EM DASH')
            and U('BLACK RIGHT-POINTING TRIANGLE') else None),
    '==>': ('==>', U('BOX DRAWINGS DOUBLE HORIZONTAL') +
            U('BOX DRAWINGS DOUBLE HORIZONTAL') +
            U('BLACK RIGHT-POINTING TRIANGLE') if
            U('BOX DRAWINGS DOUBLE HORIZONTAL') and
            U('BOX DRAWINGS DOUBLE HORIZONTAL') and
            U('BLACK RIGHT-POINTING TRIANGLE') else None),
    '.':   ('*', U('RING OPERATOR')),
}


def xsym(sym):
    """get symbology for a 'character'"""
    op = _xsym[sym]

    if _use_unicode:
        return op[1]
    else:
        return op[0]


# SYMBOLS

atoms_table = {
    # class                    how-to-display
    'Exp1':                    U('SCRIPT SMALL E'),
    'Pi':                      U('GREEK SMALL LETTER PI'),
    'Infinity':                U('INFINITY'),
    'NegativeInfinity':        U('INFINITY') and ('-' + U('INFINITY')),  # XXX what to do here
    #'ImaginaryUnit':          U('GREEK SMALL LETTER IOTA'),
    #'ImaginaryUnit':          U('MATHEMATICAL ITALIC SMALL I'),
    'ImaginaryUnit':           U('DOUBLE-STRUCK ITALIC SMALL I'),
    'EmptySet':                U('EMPTY SET'),
    'Naturals':                U('DOUBLE-STRUCK CAPITAL N'),
    'Naturals0':              (U('DOUBLE-STRUCK CAPITAL N') and
                              (U('DOUBLE-STRUCK CAPITAL N') +
                               U('SUBSCRIPT ZERO'))),
    'Integers':                U('DOUBLE-STRUCK CAPITAL Z'),
    'Rationals':               U('DOUBLE-STRUCK CAPITAL Q'),
    'Reals':                   U('DOUBLE-STRUCK CAPITAL R'),
    'Complexes':               U('DOUBLE-STRUCK CAPITAL C'),
    'Universe':                U('MATHEMATICAL DOUBLE-STRUCK CAPITAL U'),
    'IdentityMatrix':          U('MATHEMATICAL DOUBLE-STRUCK CAPITAL I'),
    'ZeroMatrix':              U('MATHEMATICAL DOUBLE-STRUCK DIGIT ZERO'),
    'OneMatrix':               U('MATHEMATICAL DOUBLE-STRUCK DIGIT ONE'),
    'Differential':            U('DOUBLE-STRUCK ITALIC SMALL D'),
    'Union':                   U('UNION'),
    'ElementOf':               U('ELEMENT OF'),
    'SmallElementOf':          U('SMALL ELEMENT OF'),
    'SymmetricDifference':     U('INCREMENT'),
    'Intersection':            U('INTERSECTION'),
    'Ring':                    U('RING OPERATOR'),
    'Multiplication':          U('MULTIPLICATION SIGN'),
    'TensorProduct':           U('N-ARY CIRCLED TIMES OPERATOR'),
    'Dots':                    U('HORIZONTAL ELLIPSIS'),
    'Modifier Letter Low Ring':U('Modifier Letter Low Ring'),
    'EmptySequence':           'EmptySequence',
    'SuperscriptPlus':         U('SUPERSCRIPT PLUS SIGN'),
    'SuperscriptMinus':        U('SUPERSCRIPT MINUS'),
    'Dagger':                  U('DAGGER'),
    'Degree':                  U('DEGREE SIGN'),
    #Logic Symbols
    'And':                     U('LOGICAL AND'),
    'Or':                      U('LOGICAL OR'),
    'Not':                     U('NOT SIGN'),
    'Nor':                     U('NOR'),
    'Nand':                    U('NAND'),
    'Xor':                     U('XOR'),
    'Equiv':                   U('LEFT RIGHT DOUBLE ARROW'),
    'NotEquiv':                U('LEFT RIGHT DOUBLE ARROW WITH STROKE'),
    'Implies':                 U('LEFT RIGHT DOUBLE ARROW'),
    'NotImplies':              U('LEFT RIGHT DOUBLE ARROW WITH STROKE'),
    'Arrow':                   U('RIGHTWARDS ARROW'),
    'ArrowFromBar':            U('RIGHTWARDS ARROW FROM BAR'),
    'NotArrow':                U('RIGHTWARDS ARROW WITH STROKE'),
    'Tautology':               U('BOX DRAWINGS LIGHT UP AND HORIZONTAL'),
    'Contradiction':           U('BOX DRAWINGS LIGHT DOWN AND HORIZONTAL')
}


def pretty_atom(atom_name, default=None, printer=None):
    """return pretty representation of an atom"""
    if _use_unicode:
        if printer is not None and atom_name == 'ImaginaryUnit' and printer._settings['imaginary_unit'] == 'j':
            return U('DOUBLE-STRUCK ITALIC SMALL J')
        else:
            return atoms_table[atom_name]
    else:
        if default is not None:
            return default

        raise KeyError('only unicode')  # send it default printer


def pretty_symbol(symb_name, bold_name=False):
    """return pretty representation of a symbol"""
    # let's split symb_name into symbol + index
    # UC: beta1
    # UC: f_beta

    if not _use_unicode:
        return symb_name

    name, sups, subs = split_super_sub(symb_name)

    def translate(s, bold_name) :
        if bold_name:
            gG = greek_bold_unicode.get(s)
        else:
            gG = greek_unicode.get(s)
        if gG is not None:
            return gG
        for key in sorted(modifier_dict.keys(), key=lambda k:len(k), reverse=True) :
            if s.lower().endswith(key) and len(s)>len(key):
                return modifier_dict[key](translate(s[:-len(key)], bold_name))
        if bold_name:
            return ''.join([bold_unicode[c] for c in s])
        return s

    name = translate(name, bold_name)

    # Let's prettify sups/subs. If it fails at one of them, pretty sups/subs are
    # not used at all.
    def pretty_list(l, mapping):
        result = []
        for s in l:
            pretty = mapping.get(s)
            if pretty is None:
                try:  # match by separate characters
                    pretty = ''.join([mapping[c] for c in s])
                except (TypeError, KeyError):
                    return None
            result.append(pretty)
        return result

    pretty_sups = pretty_list(sups, sup)
    if pretty_sups is not None:
        pretty_subs = pretty_list(subs, sub)
    else:
        pretty_subs = None

    # glue the results into one string
    if pretty_subs is None:  # nice formatting of sups/subs did not work
        if subs:
            name += '_'+'_'.join([translate(s, bold_name) for s in subs])
        if sups:
            name += '__'+'__'.join([translate(s, bold_name) for s in sups])
        return name
    else:
        sups_result = ' '.join(pretty_sups)
        subs_result = ' '.join(pretty_subs)

    return ''.join([name, sups_result, subs_result])


def annotated(letter):
    """
    Return a stylised drawing of the letter ``letter``, together with
    information on how to put annotations (super- and subscripts to the
    left and to the right) on it.

    See pretty.py functions _print_meijerg, _print_hyper on how to use this
    information.
    """
    ucode_pics = {
        'F': (2, 0, 2, 0, '\N{BOX DRAWINGS LIGHT DOWN AND RIGHT}\N{BOX DRAWINGS LIGHT HORIZONTAL}\n'
                          '\N{BOX DRAWINGS LIGHT VERTICAL AND RIGHT}\N{BOX DRAWINGS LIGHT HORIZONTAL}\n'
                          '\N{BOX DRAWINGS LIGHT UP}'),
        'G': (3, 0, 3, 1, '\N{BOX DRAWINGS LIGHT ARC DOWN AND RIGHT}\N{BOX DRAWINGS LIGHT HORIZONTAL}\N{BOX DRAWINGS LIGHT ARC DOWN AND LEFT}\n'
                          '\N{BOX DRAWINGS LIGHT VERTICAL}\N{BOX DRAWINGS LIGHT RIGHT}\N{BOX DRAWINGS LIGHT DOWN AND LEFT}\n'
                          '\N{BOX DRAWINGS LIGHT ARC UP AND RIGHT}\N{BOX DRAWINGS LIGHT HORIZONTAL}\N{BOX DRAWINGS LIGHT ARC UP AND LEFT}')
    }
    ascii_pics = {
        'F': (3, 0, 3, 0, ' _\n|_\n|\n'),
        'G': (3, 0, 3, 1, ' __\n/__\n\\_|')
    }

    if _use_unicode:
        return ucode_pics[letter]
    else:
        return ascii_pics[letter]

_remove_combining = dict.fromkeys(list(range(ord('\N{COMBINING GRAVE ACCENT}'), ord('\N{COMBINING LATIN SMALL LETTER X}')))
                            + list(range(ord('\N{COMBINING LEFT HARPOON ABOVE}'), ord('\N{COMBINING ASTERISK ABOVE}'))))

def is_combining(sym):
    """Check whether symbol is a unicode modifier. """

    return ord(sym) in _remove_combining


def center_accent(string, accent):
    """
    Returns a string with accent inserted on the middle character. Useful to
    put combining accents on symbol names, including multi-character names.

    Parameters
    ==========

    string : string
        The string to place the accent in.
    accent : string
        The combining accent to insert

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Combining_character
    .. [2] https://en.wikipedia.org/wiki/Combining_Diacritical_Marks

    """

    # Accent is placed on the previous character, although it may not always look
    # like that depending on console
    midpoint = len(string) // 2 + 1
    firstpart = string[:midpoint]
    secondpart = string[midpoint:]
    return firstpart + accent + secondpart


def line_width(line):
    """Unicode combining symbols (modifiers) are not ever displayed as
    separate symbols and thus should not be counted
    """
    return len(line.translate(_remove_combining))


def is_subscriptable_in_unicode(subscript):
    """
    Checks whether a string is subscriptable in unicode or not.

    Parameters
    ==========

    subscript: the string which needs to be checked

    Examples
    ========

    >>> from sympy.printing.pretty.pretty_symbology import is_subscriptable_in_unicode
    >>> is_subscriptable_in_unicode('abc')
    False
    >>> is_subscriptable_in_unicode('123')
    True

    """
    return all(character in sub for character in subscript)


def center_pad(wstring, wtarget, fillchar=' '):
    """
    Return the padding strings necessary to center a string of
    wstring characters wide in a wtarget wide space.

    The line_width wstring should always be less or equal to wtarget
    or else a ValueError will be raised.
    """
    if wstring > wtarget:
        raise ValueError('not enough space for string')
    wdelta = wtarget - wstring

    wleft = wdelta // 2  # favor left '1 '
    wright = wdelta - wleft

    left = fillchar * wleft
    right = fillchar * wright

    return left, right


def center(string, width, fillchar=' '):
    """Return a centered string of length determined by `line_width`
    that uses `fillchar` for padding.
    """
    left, right = center_pad(line_width(string), width, fillchar)
    return ''.join([left, string, right])

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\sympy\parsing\latex\lark\transformer.py ===
import re

import sympy
from sympy.external import import_module
from sympy.parsing.latex.errors import LaTeXParsingError

lark = import_module("lark")

if lark:
    from lark import Transformer, Token, Tree  # type: ignore
else:
    class Transformer:  # type: ignore
        def transform(self, *args):
            pass


    class Token:  # type: ignore
        pass


    class Tree:  # type: ignore
        pass


# noinspection PyPep8Naming,PyMethodMayBeStatic
class TransformToSymPyExpr(Transformer):
    """Returns a SymPy expression that is generated by traversing the ``lark.Tree``
    passed to the ``.transform()`` function.

    Notes
    =====

    **This class is never supposed to be used directly.**

    In order to tweak the behavior of this class, it has to be subclassed and then after
    the required modifications are made, the name of the new class should be passed to
    the :py:class:`LarkLaTeXParser` class by using the ``transformer`` argument in the
    constructor.

    Parameters
    ==========

    visit_tokens : bool, optional
        For information about what this option does, see `here
        <https://lark-parser.readthedocs.io/en/latest/visitors.html#lark.visitors.Transformer>`_.

        Note that the option must be set to ``True`` for the default parser to work.
    """

    SYMBOL = sympy.Symbol
    DIGIT = sympy.core.numbers.Integer

    def CMD_INFTY(self, tokens):
        return sympy.oo

    def GREEK_SYMBOL_WITH_PRIMES(self, tokens):
        # we omit the first character because it is a backslash. Also, if the variable name has "var" in it,
        # like "varphi" or "varepsilon", we remove that too
        variable_name = re.sub("var", "", tokens[1:])

        return sympy.Symbol(variable_name)

    def LATIN_SYMBOL_WITH_LATIN_SUBSCRIPT(self, tokens):
        base, sub = tokens.value.split("_")
        if sub.startswith("{"):
            return sympy.Symbol("%s_{%s}" % (base, sub[1:-1]))
        else:
            return sympy.Symbol("%s_{%s}" % (base, sub))

    def GREEK_SYMBOL_WITH_LATIN_SUBSCRIPT(self, tokens):
        base, sub = tokens.value.split("_")
        greek_letter = re.sub("var", "", base[1:])

        if sub.startswith("{"):
            return sympy.Symbol("%s_{%s}" % (greek_letter, sub[1:-1]))
        else:
            return sympy.Symbol("%s_{%s}" % (greek_letter, sub))

    def LATIN_SYMBOL_WITH_GREEK_SUBSCRIPT(self, tokens):
        base, sub = tokens.value.split("_")
        if sub.startswith("{"):
            greek_letter = sub[2:-1]
        else:
            greek_letter = sub[1:]

        greek_letter = re.sub("var", "", greek_letter)
        return sympy.Symbol("%s_{%s}" % (base, greek_letter))


    def GREEK_SYMBOL_WITH_GREEK_SUBSCRIPT(self, tokens):
        base, sub = tokens.value.split("_")
        greek_base = re.sub("var", "", base[1:])

        if sub.startswith("{"):
            greek_sub = sub[2:-1]
        else:
            greek_sub = sub[1:]

        greek_sub = re.sub("var", "", greek_sub)
        return sympy.Symbol("%s_{%s}" % (greek_base, greek_sub))

    def multi_letter_symbol(self, tokens):
        if len(tokens) == 4: # no primes (single quotes) on symbol
            return sympy.Symbol(tokens[2])
        if len(tokens) == 5: # there are primes on the symbol
            return sympy.Symbol(tokens[2] + tokens[4])

    def number(self, tokens):
        if tokens[0].type == "CMD_IMAGINARY_UNIT":
            return sympy.I

        if "." in tokens[0]:
            return sympy.core.numbers.Float(tokens[0])
        else:
            return sympy.core.numbers.Integer(tokens[0])

    def latex_string(self, tokens):
        return tokens[0]

    def group_round_parentheses(self, tokens):
        return tokens[1]

    def group_square_brackets(self, tokens):
        return tokens[1]

    def group_curly_parentheses(self, tokens):
        return tokens[1]

    def eq(self, tokens):
        return sympy.Eq(tokens[0], tokens[2])

    def ne(self, tokens):
        return sympy.Ne(tokens[0], tokens[2])

    def lt(self, tokens):
        return sympy.Lt(tokens[0], tokens[2])

    def lte(self, tokens):
        return sympy.Le(tokens[0], tokens[2])

    def gt(self, tokens):
        return sympy.Gt(tokens[0], tokens[2])

    def gte(self, tokens):
        return sympy.Ge(tokens[0], tokens[2])

    def add(self, tokens):
        if len(tokens) == 2: # +a
            return tokens[1]
        if len(tokens) == 3: # a + b
            lh = tokens[0]
            rh = tokens[2]

            if self._obj_is_sympy_Matrix(lh) or self._obj_is_sympy_Matrix(rh):
                return sympy.MatAdd(lh, rh)

            return sympy.Add(lh, rh)

    def sub(self, tokens):
        if len(tokens) == 2: # -a
            x = tokens[1]

            if self._obj_is_sympy_Matrix(x):
                return sympy.MatMul(-1, x)

            return -x
        if len(tokens) == 3: # a - b
            lh = tokens[0]
            rh = tokens[2]

            if self._obj_is_sympy_Matrix(lh) or self._obj_is_sympy_Matrix(rh):
                return sympy.MatAdd(lh, sympy.MatMul(-1, rh))

            return sympy.Add(lh, -rh)

    def mul(self, tokens):
        lh = tokens[0]
        rh = tokens[2]

        if self._obj_is_sympy_Matrix(lh) or self._obj_is_sympy_Matrix(rh):
            return sympy.MatMul(lh, rh)

        return sympy.Mul(lh, rh)

    def div(self, tokens):
        return self._handle_division(tokens[0], tokens[2])

    def adjacent_expressions(self, tokens):
        # Most of the time, if two expressions are next to each other, it means implicit multiplication,
        # but not always
        from sympy.physics.quantum import Bra, Ket
        if isinstance(tokens[0], Ket) and isinstance(tokens[1], Bra):
            from sympy.physics.quantum import OuterProduct
            return OuterProduct(tokens[0], tokens[1])
        elif tokens[0] == sympy.Symbol("d"):
            # If the leftmost token is a "d", then it is highly likely that this is a differential
            return tokens[0], tokens[1]
        elif isinstance(tokens[0], tuple):
            # then we have a derivative
            return sympy.Derivative(tokens[1], tokens[0][1])
        else:
            return sympy.Mul(tokens[0], tokens[1])

    def superscript(self, tokens):
        def isprime(x):
            return isinstance(x, Token) and x.type == "PRIMES"

        def iscmdprime(x):
            return isinstance(x, Token) and (x.type == "PRIMES_VIA_CMD"
                                             or x.type == "CMD_PRIME")

        def isstar(x):
            return isinstance(x, Token) and x.type == "STARS"

        def iscmdstar(x):
            return isinstance(x, Token) and (x.type == "STARS_VIA_CMD"
                                             or x.type == "CMD_ASTERISK")

        base = tokens[0]
        if len(tokens) == 3: # a^b OR a^\prime OR a^\ast
            sup = tokens[2]
        if len(tokens) == 5:
            # a^{'}, a^{''}, ... OR
            # a^{*}, a^{**}, ... OR
            # a^{\prime}, a^{\prime\prime}, ... OR
            # a^{\ast}, a^{\ast\ast}, ...
            sup = tokens[3]

        if self._obj_is_sympy_Matrix(base):
            if sup == sympy.Symbol("T"):
                return sympy.Transpose(base)
            if sup == sympy.Symbol("H"):
                return sympy.adjoint(base)
            if isprime(sup):
                sup = sup.value
                if len(sup) % 2 == 0:
                    return base
                return sympy.Transpose(base)
            if iscmdprime(sup):
                sup = sup.value
                if (len(sup)/len(r"\prime")) % 2 == 0:
                    return base
                return sympy.Transpose(base)
            if isstar(sup):
                sup = sup.value
                # need .doit() in order to be consistent with
                # sympy.adjoint() which returns the evaluated adjoint
                # of a matrix
                if len(sup) % 2 == 0:
                    return base.doit()
                return sympy.adjoint(base)
            if iscmdstar(sup):
                sup = sup.value
                # need .doit() for same reason as above
                if (len(sup)/len(r"\ast")) % 2 == 0:
                    return base.doit()
                return sympy.adjoint(base)

        if isprime(sup) or iscmdprime(sup) or isstar(sup) or iscmdstar(sup):
            raise LaTeXParsingError(f"{base} with superscript {sup} is not understood.")

        return sympy.Pow(base, sup)

    def matrix_prime(self, tokens):
        base = tokens[0]
        primes = tokens[1].value

        if not self._obj_is_sympy_Matrix(base):
            raise LaTeXParsingError(f"({base}){primes} is not understood.")

        if len(primes) % 2 == 0:
            return base

        return sympy.Transpose(base)

    def symbol_prime(self, tokens):
        base = tokens[0]
        primes = tokens[1].value

        return sympy.Symbol(f"{base.name}{primes}")

    def fraction(self, tokens):
        numerator = tokens[1]
        if isinstance(tokens[2], tuple):
            # we only need the variable w.r.t. which we are differentiating
            _, variable = tokens[2]

            # we will pass this information upwards
            return "derivative", variable
        else:
            denominator = tokens[2]
            return self._handle_division(numerator, denominator)

    def binomial(self, tokens):
        return sympy.binomial(tokens[1], tokens[2])

    def normal_integral(self, tokens):
        underscore_index = None
        caret_index = None

        if "_" in tokens:
            # we need to know the index because the next item in the list is the
            # arguments for the lower bound of the integral
            underscore_index = tokens.index("_")

        if "^" in tokens:
            # we need to know the index because the next item in the list is the
            # arguments for the upper bound of the integral
            caret_index = tokens.index("^")

        lower_bound = tokens[underscore_index + 1] if underscore_index else None
        upper_bound = tokens[caret_index + 1] if caret_index else None

        differential_symbol = self._extract_differential_symbol(tokens)

        if differential_symbol is None:
            raise LaTeXParsingError("Differential symbol was not found in the expression."
                                    "Valid differential symbols are \"d\", \"\\text{d}, and \"\\mathrm{d}\".")

        # else we can assume that a differential symbol was found
        differential_variable_index = tokens.index(differential_symbol) + 1
        differential_variable = tokens[differential_variable_index]

        # we can't simply do something like `if (lower_bound and not upper_bound) ...` because this would
        # evaluate to `True` if the `lower_bound` is 0 and upper bound is non-zero
        if lower_bound is not None and upper_bound is None:
            # then one was given and the other wasn't
            raise LaTeXParsingError("Lower bound for the integral was found, but upper bound was not found.")

        if upper_bound is not None and lower_bound is None:
            # then one was given and the other wasn't
            raise LaTeXParsingError("Upper bound for the integral was found, but lower bound was not found.")

        # check if any expression was given or not. If it wasn't, then set the integrand to 1.
        if underscore_index is not None and underscore_index == differential_variable_index - 3:
            # The Token at differential_variable_index - 2 should be the integrand. However, if going one more step
            # backwards after that gives us the underscore, then that means that there _was_ no integrand.
            # Example: \int^7_0 dx
            integrand = 1
        elif caret_index is not None and caret_index == differential_variable_index - 3:
            # The Token at differential_variable_index - 2 should be the integrand. However, if going one more step
            # backwards after that gives us the caret, then that means that there _was_ no integrand.
            # Example: \int_0^7 dx
            integrand = 1
        elif differential_variable_index == 2:
            # this means we have something like "\int dx", because the "\int" symbol will always be
            # at index 0 in `tokens`
            integrand = 1
        else:
            # The Token at differential_variable_index - 1 is the differential symbol itself, so we need to go one
            # more step before that.
            integrand = tokens[differential_variable_index - 2]

        if lower_bound is not None:
            # then we have a definite integral

            # we can assume that either both the lower and upper bounds are given, or
            # neither of them are
            return sympy.Integral(integrand, (differential_variable, lower_bound, upper_bound))
        else:
            # we have an indefinite integral
            return sympy.Integral(integrand, differential_variable)

    def group_curly_parentheses_int(self, tokens):
        # return signature is a tuple consisting of the expression in the numerator, along with the variable of
        # integration
        if len(tokens) == 3:
            return 1, tokens[1]
        elif len(tokens) == 4:
            return tokens[1], tokens[2]
        # there are no other possibilities

    def special_fraction(self, tokens):
        numerator, variable = tokens[1]
        denominator = tokens[2]

        # We pass the integrand, along with information about the variable of integration, upw
        return sympy.Mul(numerator, sympy.Pow(denominator, -1)), variable

    def integral_with_special_fraction(self, tokens):
        underscore_index = None
        caret_index = None

        if "_" in tokens:
            # we need to know the index because the next item in the list is the
            # arguments for the lower bound of the integral
            underscore_index = tokens.index("_")

        if "^" in tokens:
            # we need to know the index because the next item in the list is the
            # arguments for the upper bound of the integral
            caret_index = tokens.index("^")

        lower_bound = tokens[underscore_index + 1] if underscore_index else None
        upper_bound = tokens[caret_index + 1] if caret_index else None

        # we can't simply do something like `if (lower_bound and not upper_bound) ...` because this would
        # evaluate to `True` if the `lower_bound` is 0 and upper bound is non-zero
        if lower_bound is not None and upper_bound is None:
            # then one was given and the other wasn't
            raise LaTeXParsingError("Lower bound for the integral was found, but upper bound was not found.")

        if upper_bound is not None and lower_bound is None:
            # then one was given and the other wasn't
            raise LaTeXParsingError("Upper bound for the integral was found, but lower bound was not found.")

        integrand, differential_variable = tokens[-1]

        if lower_bound is not None:
            # then we have a definite integral

            # we can assume that either both the lower and upper bounds are given, or
            # neither of them are
            return sympy.Integral(integrand, (differential_variable, lower_bound, upper_bound))
        else:
            # we have an indefinite integral
            return sympy.Integral(integrand, differential_variable)

    def group_curly_parentheses_special(self, tokens):
        underscore_index = tokens.index("_")
        caret_index = tokens.index("^")

        # given the type of expressions we are parsing, we can assume that the lower limit
        # will always use braces around its arguments. This is because we don't support
        # converting unconstrained sums into SymPy expressions.

        # first we isolate the bottom limit
        left_brace_index = tokens.index("{", underscore_index)
        right_brace_index = tokens.index("}", underscore_index)

        bottom_limit = tokens[left_brace_index + 1: right_brace_index]

        # next, we isolate the upper limit
        top_limit = tokens[caret_index + 1:]

        # the code below will be useful for supporting things like `\sum_{n = 0}^{n = 5} n^2`
        # if "{" in top_limit:
        #     left_brace_index = tokens.index("{", caret_index)
        #     if left_brace_index != -1:
        #         # then there's a left brace in the string, and we need to find the closing right brace
        #         right_brace_index = tokens.index("}", caret_index)
        #         top_limit = tokens[left_brace_index + 1: right_brace_index]

        # print(f"top  limit = {top_limit}")

        index_variable = bottom_limit[0]
        lower_limit = bottom_limit[-1]
        upper_limit = top_limit[0]  # for now, the index will always be 0

        # print(f"return value = ({index_variable}, {lower_limit}, {upper_limit})")

        return index_variable, lower_limit, upper_limit

    def summation(self, tokens):
        return sympy.Sum(tokens[2], tokens[1])

    def product(self, tokens):
        return sympy.Product(tokens[2], tokens[1])

    def limit_dir_expr(self, tokens):
        caret_index = tokens.index("^")

        if "{" in tokens:
            left_curly_brace_index = tokens.index("{", caret_index)
            direction = tokens[left_curly_brace_index + 1]
        else:
            direction = tokens[caret_index + 1]

        if direction == "+":
            return tokens[0], "+"
        elif direction == "-":
            return tokens[0], "-"
        else:
            return tokens[0], "+-"

    def group_curly_parentheses_lim(self, tokens):
        limit_variable = tokens[1]
        if isinstance(tokens[3], tuple):
            destination, direction = tokens[3]
        else:
            destination = tokens[3]
            direction = "+-"

        return limit_variable, destination, direction

    def limit(self, tokens):
        limit_variable, destination, direction = tokens[2]

        return sympy.Limit(tokens[-1], limit_variable, destination, direction)

    def differential(self, tokens):
        return tokens[1]

    def derivative(self, tokens):
        return sympy.Derivative(tokens[-1], tokens[5])

    def list_of_expressions(self, tokens):
        if len(tokens) == 1:
            # we return it verbatim because the function_applied node expects
            # a list
            return tokens
        else:
            def remove_tokens(args):
                if isinstance(args, Token):
                    if args.type != "COMMA":
                        # An unexpected token was encountered
                        raise LaTeXParsingError("A comma token was expected, but some other token was encountered.")
                    return False
                return True

            return filter(remove_tokens, tokens)

    def function_applied(self, tokens):
        return sympy.Function(tokens[0])(*tokens[2])

    def min(self, tokens):
        return sympy.Min(*tokens[2])

    def max(self, tokens):
        return sympy.Max(*tokens[2])

    def bra(self, tokens):
        from sympy.physics.quantum import Bra
        return Bra(tokens[1])

    def ket(self, tokens):
        from sympy.physics.quantum import Ket
        return Ket(tokens[1])

    def inner_product(self, tokens):
        from sympy.physics.quantum import Bra, Ket, InnerProduct
        return InnerProduct(Bra(tokens[1]), Ket(tokens[3]))

    def sin(self, tokens):
        return sympy.sin(tokens[1])

    def cos(self, tokens):
        return sympy.cos(tokens[1])

    def tan(self, tokens):
        return sympy.tan(tokens[1])

    def csc(self, tokens):
        return sympy.csc(tokens[1])

    def sec(self, tokens):
        return sympy.sec(tokens[1])

    def cot(self, tokens):
        return sympy.cot(tokens[1])

    def sin_power(self, tokens):
        exponent = tokens[2]
        if exponent == -1:
            return sympy.asin(tokens[-1])
        else:
            return sympy.Pow(sympy.sin(tokens[-1]), exponent)

    def cos_power(self, tokens):
        exponent = tokens[2]
        if exponent == -1:
            return sympy.acos(tokens[-1])
        else:
            return sympy.Pow(sympy.cos(tokens[-1]), exponent)

    def tan_power(self, tokens):
        exponent = tokens[2]
        if exponent == -1:
            return sympy.atan(tokens[-1])
        else:
            return sympy.Pow(sympy.tan(tokens[-1]), exponent)

    def csc_power(self, tokens):
        exponent = tokens[2]
        if exponent == -1:
            return sympy.acsc(tokens[-1])
        else:
            return sympy.Pow(sympy.csc(tokens[-1]), exponent)

    def sec_power(self, tokens):
        exponent = tokens[2]
        if exponent == -1:
            return sympy.asec(tokens[-1])
        else:
            return sympy.Pow(sympy.sec(tokens[-1]), exponent)

    def cot_power(self, tokens):
        exponent = tokens[2]
        if exponent == -1:
            return sympy.acot(tokens[-1])
        else:
            return sympy.Pow(sympy.cot(tokens[-1]), exponent)

    def arcsin(self, tokens):
        return sympy.asin(tokens[1])

    def arccos(self, tokens):
        return sympy.acos(tokens[1])

    def arctan(self, tokens):
        return sympy.atan(tokens[1])

    def arccsc(self, tokens):
        return sympy.acsc(tokens[1])

    def arcsec(self, tokens):
        return sympy.asec(tokens[1])

    def arccot(self, tokens):
        return sympy.acot(tokens[1])

    def sinh(self, tokens):
        return sympy.sinh(tokens[1])

    def cosh(self, tokens):
        return sympy.cosh(tokens[1])

    def tanh(self, tokens):
        return sympy.tanh(tokens[1])

    def asinh(self, tokens):
        return sympy.asinh(tokens[1])

    def acosh(self, tokens):
        return sympy.acosh(tokens[1])

    def atanh(self, tokens):
        return sympy.atanh(tokens[1])

    def abs(self, tokens):
        return sympy.Abs(tokens[1])

    def floor(self, tokens):
        return sympy.floor(tokens[1])

    def ceil(self, tokens):
        return sympy.ceiling(tokens[1])

    def factorial(self, tokens):
        return sympy.factorial(tokens[0])

    def conjugate(self, tokens):
        return sympy.conjugate(tokens[1])

    def square_root(self, tokens):
        if len(tokens) == 2:
            # then there was no square bracket argument
            return sympy.sqrt(tokens[1])
        elif len(tokens) == 3:
            # then there _was_ a square bracket argument
            return sympy.root(tokens[2], tokens[1])

    def exponential(self, tokens):
        return sympy.exp(tokens[1])

    def log(self, tokens):
        if tokens[0].type == "FUNC_LG":
            # we don't need to check if there's an underscore or not because having one
            # in this case would be meaningless
            # TODO: ANTLR refers to ISO 80000-2:2019. should we keep base 10 or base 2?
            return sympy.log(tokens[1], 10)
        elif tokens[0].type == "FUNC_LN":
            return sympy.log(tokens[1])
        elif tokens[0].type == "FUNC_LOG":
            # we check if a base was specified or not
            if "_" in tokens:
                # then a base was specified
                return sympy.log(tokens[3], tokens[2])
            else:
                # a base was not specified
                return sympy.log(tokens[1])

    def _extract_differential_symbol(self, s: str):
        differential_symbols = {"d", r"\text{d}", r"\mathrm{d}"}

        differential_symbol = next((symbol for symbol in differential_symbols if symbol in s), None)

        return differential_symbol

    def matrix(self, tokens):
        def is_matrix_row(x):
            return (isinstance(x, Tree) and x.data == "matrix_row")

        def is_not_col_delim(y):
            return (not isinstance(y, Token) or y.type != "MATRIX_COL_DELIM")

        matrix_body = tokens[1].children
        return sympy.Matrix([[y for y in x.children if is_not_col_delim(y)]
                             for x in matrix_body if is_matrix_row(x)])

    def determinant(self, tokens):
        if len(tokens) == 2: # \det A
            if not self._obj_is_sympy_Matrix(tokens[1]):
                raise LaTeXParsingError("Cannot take determinant of non-matrix.")

            return tokens[1].det()

        if len(tokens) == 3: # | A |
            return self.matrix(tokens).det()

    def trace(self, tokens):
        if not self._obj_is_sympy_Matrix(tokens[1]):
            raise LaTeXParsingError("Cannot take trace of non-matrix.")

        return sympy.Trace(tokens[1])

    def adjugate(self, tokens):
        if not self._obj_is_sympy_Matrix(tokens[1]):
            raise LaTeXParsingError("Cannot take adjugate of non-matrix.")

        # need .doit() since MatAdd does not support .adjugate() method
        return tokens[1].doit().adjugate()

    def _obj_is_sympy_Matrix(self, obj):
        if hasattr(obj, "is_Matrix"):
            return obj.is_Matrix

        return isinstance(obj, sympy.Matrix)

    def _handle_division(self, numerator, denominator):
        if self._obj_is_sympy_Matrix(denominator):
            raise LaTeXParsingError("Cannot divide by matrices like this since "
                                    "it is not clear if left or right multiplication "
                                    "by the inverse is intended. Try explicitly "
                                    "multiplying by the inverse instead.")

        if self._obj_is_sympy_Matrix(numerator):
            return sympy.MatMul(numerator, sympy.Pow(denominator, -1))

        return sympy.Mul(numerator, sympy.Pow(denominator, -1))

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\onnxruntime\transformers\onnx_exporter.py ===
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import logging
import os
from pathlib import Path

import numpy
import torch
from affinity_helper import AffinitySetting
from benchmark_helper import OptimizerInfo, Precision, create_onnxruntime_session
from huggingface_models import MODEL_CLASSES
from quantize_helper import QuantizeHelper
from torch_onnx_export_helper import torch_onnx_export
from transformers import AutoConfig, AutoFeatureExtractor, AutoTokenizer, LxmertConfig, TransfoXLConfig

from onnxruntime.transformers.models.gpt2.gpt2_helper import (
    PRETRAINED_GPT2_MODELS,
    GPT2ModelNoPastState,
    TFGPT2ModelNoPastState,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

logger = logging.getLogger(__name__)

# Workaround by replacing torch.triu using self-defined op
# Since torch.triu cannot be exported to ONNX. See https://github.com/pytorch/pytorch/issues/32968
torch_func = {"triu": torch.triu}


def triu_onnx(x, diagonal=0, out=None):
    assert out is None
    assert len(x.shape) == 2 and x.size(0) == x.size(1)

    torch_triu = torch_func["triu"]
    template = torch_triu(torch.ones((1024, 1024), dtype=torch.uint8), diagonal)
    mask = template[: x.size(0), : x.size(1)]
    return torch.where(mask.bool(), x, torch.zeros_like(x))


def replace_torch_functions():
    torch.triu = triu_onnx


def restore_torch_functions():
    torch.triu = torch_func["triu"]


def create_onnxruntime_input(vocab_size, batch_size, sequence_length, input_names, config, data_type=numpy.int64):
    if config.model_type in ["vit", "swin"]:
        input_ids = numpy.random.rand(batch_size, 3, config.image_size, config.image_size).astype(numpy.float32)
        inputs = {"pixel_values": input_ids}
        return inputs

    input_ids = numpy.random.randint(low=0, high=vocab_size - 1, size=(batch_size, sequence_length), dtype=data_type)
    inputs = {"input_ids": input_ids}

    if "attention_mask" in input_names:
        attention_mask = numpy.ones([batch_size, sequence_length], dtype=data_type)
        inputs["attention_mask"] = attention_mask

    if "token_type_ids" in input_names:
        segment_ids = numpy.zeros([batch_size, sequence_length], dtype=data_type)
        inputs["token_type_ids"] = segment_ids

    if config.is_encoder_decoder:
        inputs["decoder_input_ids"] = input_ids

    if isinstance(config, LxmertConfig):
        inputs["visual_feats"] = numpy.random.randn(1, 1, config.visual_feat_dim).astype(numpy.float32)
        inputs["visual_pos"] = numpy.random.randn(1, 1, config.visual_pos_dim).astype(numpy.float32)
    if isinstance(config, TransfoXLConfig):
        inputs["tf_transfo_xl_model/transformer/pos_emb/einsum/Einsum/inputs_1:0"] = numpy.zeros(
            [config.hidden_size], dtype=numpy.float32
        )
    return inputs


def filter_inputs(inputs, input_names):
    remaining_model_inputs = {}
    for input_name in input_names:
        if input_name in inputs:
            remaining_model_inputs[input_name] = inputs[input_name]
    return remaining_model_inputs


def flatten(inputs):
    return [[flatten(i) for i in inputs] if isinstance(inputs, (list, tuple)) else inputs]


def update_flatten_list(inputs, res_list):
    for i in inputs:
        res_list.append(i) if not isinstance(i, (list, tuple)) else update_flatten_list(i, res_list)
    return res_list


def build_dynamic_axes(example_inputs, outputs_flatten):
    sequence_length = example_inputs["input_ids"].shape[-1]

    dynamic_axes = {key: {0: "batch_size", 1: "seq_len"} for key in example_inputs}

    output_names = ["output_" + str(i + 1) for i in range(len(outputs_flatten))]
    for i, output_name in enumerate(output_names):
        dynamic_axes[output_name] = {0: "batch_size"}
        dims = outputs_flatten[i].shape
        for j, dim in enumerate(dims):
            if dim == sequence_length:
                dynamic_axes[output_name].update({j: "seq_len"})
    return dynamic_axes, output_names


def validate_onnx_model(
    onnx_model_path,
    example_inputs,
    example_outputs_flatten,
    use_gpu,
    fp16,
    output_names=None,
):
    test_session = create_onnxruntime_session(onnx_model_path, use_gpu, enable_all_optimization=False)
    if test_session is None:
        logger.error(f"{onnx_model_path} is an invalid ONNX model")
        return False

    logger.info(f"{onnx_model_path} is a valid ONNX model")

    # Compare the inference result with PyTorch or Tensorflow
    example_ort_inputs = {k: t.numpy() for k, t in example_inputs.items()}
    example_ort_outputs = test_session.run(output_names, example_ort_inputs)
    if len(example_outputs_flatten) != len(example_ort_outputs):
        logger.error(
            f"Number of output tensors expected {len(example_outputs_flatten)}, got {len(example_ort_outputs)}"
        )
        return False

    for i in range(len(example_outputs_flatten)):
        abs_diff = numpy.amax(numpy.abs(example_ort_outputs[i] - example_outputs_flatten[i].cpu().numpy()))
        if abs_diff > 1e-4:
            logger.info(f"Max absolute diff={abs_diff} for output tensor {i}")

        rtol = 5e-02 if fp16 else 1e-4
        atol = 1e-01 if fp16 else 1e-4
        if not numpy.allclose(
            example_ort_outputs[i],
            example_outputs_flatten[i].cpu().numpy(),
            rtol=rtol,
            atol=atol,
        ):
            logger.error(f"Output tensor {i} is not close: rtol={rtol}, atol={atol}")
            return False

    logger.info(f"inference result of onnxruntime is validated on {onnx_model_path}")
    return True


def get_onnx_file_path(
    onnx_dir: str,
    model_name: str,
    input_count: int,
    optimized_by_script: bool,
    use_gpu: bool,
    precision: Precision,
    optimized_by_onnxruntime: bool,
    use_external_data: bool,
):
    from re import sub

    normalized_model_name = sub(r"[^a-zA-Z0-9_]", "_", model_name)

    if not optimized_by_script:
        filename = f"{normalized_model_name}_{input_count}"
    else:
        device = "gpu" if use_gpu else "cpu"
        filename = f"{normalized_model_name}_{input_count}_{precision}_{device}"

    if optimized_by_onnxruntime:
        filename += "_ort"

    directory = onnx_dir
    # ONNXRuntime will not write external data so the raw and optimized models shall be in same directory.
    if use_external_data and not optimized_by_onnxruntime:
        directory = os.path.join(onnx_dir, filename)
        if not os.path.exists(directory):
            os.makedirs(directory)

    return os.path.join(directory, f"{filename}.onnx")


def add_filename_suffix(file_path: str, suffix: str) -> str:
    """
    Append a suffix at the filename (before the extension).
    Args:
        path: pathlib.Path The actual path object we would like to add a suffix
        suffix: The suffix to add
    Returns: path with suffix appended at the end of the filename and before extension
    """
    path = Path(file_path)
    return str(path.parent.joinpath(path.stem + suffix).with_suffix(path.suffix))


def optimize_onnx_model_by_ort(onnx_model_path, ort_model_path, use_gpu, overwrite, model_fusion_statistics):
    if overwrite or not os.path.exists(ort_model_path):
        Path(ort_model_path).parent.mkdir(parents=True, exist_ok=True)
        from optimizer import get_fusion_statistics, optimize_by_onnxruntime

        # Use onnxruntime to optimize model, which will be saved to *_ort.onnx
        _ = optimize_by_onnxruntime(
            onnx_model_path,
            use_gpu=use_gpu,
            optimized_model_path=ort_model_path,
            opt_level=99,
        )
        model_fusion_statistics[ort_model_path] = get_fusion_statistics(ort_model_path)
    else:
        logger.info(f"Skip optimization since model existed: {ort_model_path}")


def optimize_onnx_model(
    onnx_model_path,
    optimized_model_path,
    model_type,
    num_attention_heads,
    hidden_size,
    use_gpu,
    precision,
    use_raw_attention_mask,
    overwrite,
    model_fusion_statistics,
    use_external_data_format,
    optimization_options=None,
):
    if overwrite or not os.path.exists(optimized_model_path):
        Path(optimized_model_path).parent.mkdir(parents=True, exist_ok=True)

        from fusion_options import FusionOptions
        from optimizer import optimize_model

        if optimization_options is None:
            optimization_options = FusionOptions(model_type)
        optimization_options.use_raw_attention_mask(use_raw_attention_mask)
        if precision == Precision.FLOAT16:
            optimization_options.enable_gelu_approximation = True
        if precision == Precision.INT8:
            optimization_options.enable_embed_layer_norm = False

        # For swin models, the num_attention_heads is a list, which isn't supported yet, so set to 0 for now
        if model_type == "swin":
            num_attention_heads = 0
            hidden_size = 0

        # Use script to optimize model.
        # Use opt_level <= 1 for models to be converted to fp16, because some fused op (like FusedGemm) has only fp32 and no fp16.
        # It is better to be conservative so we use opt_level=0 here, in case MemcpyFromHost is added to the graph by OnnxRuntime.
        opt_model = optimize_model(
            onnx_model_path,
            model_type,
            num_heads=num_attention_heads,
            hidden_size=hidden_size,
            opt_level=0,
            optimization_options=optimization_options,
            use_gpu=use_gpu,
            only_onnxruntime=False,
        )
        if model_type == "bert_keras" or model_type == "bert_tf":
            opt_model.use_dynamic_axes()

        model_fusion_statistics[optimized_model_path] = opt_model.get_fused_operator_statistics()

        if precision == Precision.FLOAT16:
            opt_model.convert_float_to_float16(keep_io_types=True)

        opt_model.save_model_to_file(optimized_model_path, use_external_data_format)
    else:
        logger.info(f"Skip optimization since model existed: {optimized_model_path}")


def modelclass_dispatcher(model_name, custom_model_class):
    if custom_model_class is not None:
        if custom_model_class in MODEL_CLASSES:
            return custom_model_class
        else:
            raise Exception("Valid model class: " + " ".join(MODEL_CLASSES))

    if model_name in PRETRAINED_GPT2_MODELS:
        return "GPT2ModelNoPastState"

    import re

    if re.search("-squad$", model_name) is not None:
        return "AutoModelForQuestionAnswering"
    elif re.search("-mprc$", model_name) is not None:
        return "AutoModelForSequenceClassification"
    elif re.search("gpt2", model_name) is not None:
        return "AutoModelWithLMHead"

    return "AutoModel"


def load_pretrained_model(model_name, config, cache_dir, custom_model_class, is_tf_model=False):
    model_class_name = modelclass_dispatcher(model_name, custom_model_class)

    if model_class_name == "GPT2ModelNoPastState":
        if is_tf_model:
            return TFGPT2ModelNoPastState.from_pretrained(model_name, config=config, cache_dir=cache_dir)
        else:
            return GPT2ModelNoPastState.from_pretrained(model_name, config=config, cache_dir=cache_dir)

    if is_tf_model:
        model_class_name = "TF" + model_class_name

    transformers_module = __import__("transformers", fromlist=[model_class_name])
    logger.info(f"Model class name: {model_class_name}")
    model_class = getattr(transformers_module, model_class_name)

    return model_class.from_pretrained(model_name, config=config, cache_dir=cache_dir)


def load_pt_model(model_name, model_class, cache_dir, config_modifier):
    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    if hasattr(config, "return_dict"):
        config.return_dict = False

    config_modifier.modify(config)

    model = load_pretrained_model(model_name, config=config, cache_dir=cache_dir, custom_model_class=model_class)

    return config, model


def load_tf_model(model_name, model_class, cache_dir, config_modifier):
    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)

    config_modifier.modify(config)
    # Loading tf model from transformers limits the cpu affinity to {0} when KMP_AFFINITY is set
    # Restore the affinity after model loading for expected ORT performance
    affinity_setting = AffinitySetting()
    affinity_setting.get_affinity()
    model = load_pretrained_model(
        model_name,
        config=config,
        cache_dir=cache_dir,
        custom_model_class=model_class,
        is_tf_model=True,
    )
    affinity_setting.set_affinity()

    return config, model


# For test only
def load_pt_model_from_tf(model_name):
    # Note that we could get pt model from tf, but model source and its structure in this case is different from directly using
    # load_pt_model() and load_tf_model() even with the same name. Therefore it should not be used for comparing with them
    from convert_tf_models_to_pytorch import tf2pt_pipeline

    config, model = tf2pt_pipeline(model_name)

    return config, model


def validate_and_optimize_onnx(
    model_name,
    use_external_data_format,
    model_type,
    onnx_dir,
    input_names,
    use_gpu,
    precision,
    optimize_info,
    validate_onnx,
    use_raw_attention_mask,
    overwrite,
    config,
    model_fusion_statistics,
    onnx_model_path,
    example_inputs,
    example_outputs_flatten,
    output_names,
    fusion_options,
):
    is_valid_onnx_model = True
    if validate_onnx:
        is_valid_onnx_model = validate_onnx_model(
            onnx_model_path,
            example_inputs,
            example_outputs_flatten,
            use_gpu,
            False,
            output_names,
        )
    if optimize_info.name == OptimizerInfo.NOOPT.name:
        return onnx_model_path, is_valid_onnx_model, config.vocab_size

    if (
        optimize_info.name == OptimizerInfo.BYSCRIPT.name
        or precision == Precision.FLOAT16
        or precision == Precision.INT8
    ):  # Use script (optimizer.py) to optimize
        optimized_model_path = get_onnx_file_path(
            onnx_dir,
            model_name,
            len(input_names),
            True,
            use_gpu,
            precision,
            False,
            use_external_data_format,
        )
        optimize_onnx_model(
            onnx_model_path,
            optimized_model_path,
            model_type,
            config.num_attention_heads,
            config.hidden_size,
            use_gpu,
            precision,
            use_raw_attention_mask,
            overwrite,
            model_fusion_statistics,
            use_external_data_format,
            fusion_options,
        )

        onnx_model_path = optimized_model_path
        if validate_onnx:
            is_valid_onnx_model = validate_onnx_model(
                onnx_model_path,
                example_inputs,
                example_outputs_flatten,
                use_gpu,
                precision == Precision.FLOAT16,
                output_names,
            )

        if precision == Precision.INT8:
            logger.info(f"Quantizing model: {onnx_model_path}")
            QuantizeHelper.quantize_onnx_model(onnx_model_path, onnx_model_path, use_external_data_format)
            logger.info(f"Finished quantizing model: {onnx_model_path}")

    if optimize_info.name == OptimizerInfo.BYORT.name:  # Use OnnxRuntime to optimize
        if is_valid_onnx_model:
            ort_model_path = add_filename_suffix(onnx_model_path, "_ort")
            optimize_onnx_model_by_ort(
                onnx_model_path,
                ort_model_path,
                use_gpu,
                overwrite,
                model_fusion_statistics,
            )

    return (
        onnx_model_path,
        is_valid_onnx_model,
        config.num_labels if model_type in ["vit", "swin"] else config.vocab_size,
    )


def export_onnx_model_from_pt(
    model_name,
    opset_version,
    use_external_data_format,
    model_type,
    model_class,
    config_modifier,
    cache_dir,
    onnx_dir,
    input_names,
    use_gpu,
    precision,
    optimizer_info,
    validate_onnx,
    use_raw_attention_mask,
    overwrite,
    model_fusion_statistics,
    fusion_options,
):
    config, model = load_pt_model(model_name, model_class, cache_dir, config_modifier)
    # config, model = load_pt_model_from_tf(model_name)
    model.cpu()

    example_inputs = None
    max_input_size = None

    if model_type in ["vit", "swin"]:
        image_processor = AutoFeatureExtractor.from_pretrained(model_name, cache_dir=cache_dir)
        data = numpy.random.randint(
            low=0, high=256, size=config.image_size * config.image_size * 3, dtype=numpy.uint8
        ).reshape(config.image_size, config.image_size, 3)

        example_inputs = image_processor(data, return_tensors="pt")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        max_input_size = tokenizer.model_max_length
        example_inputs = tokenizer.encode_plus("This is a sample input", return_tensors="pt")

    example_inputs = filter_inputs(example_inputs, input_names)

    example_outputs = model(**example_inputs)

    assert isinstance(example_outputs, (list, tuple)), f"type of output is not list or tuple: {type(example_outputs)}"

    # Flatten is needed for gpt2 and distilgpt2.
    example_outputs_flatten = flatten(example_outputs)
    example_outputs_flatten = update_flatten_list(example_outputs_flatten, [])

    onnx_model_path = get_onnx_file_path(
        onnx_dir,
        model_name,
        len(input_names),
        False,
        use_gpu,
        precision,
        False,
        use_external_data_format,
    )

    if overwrite or not os.path.exists(onnx_model_path):
        logger.info(f"Exporting ONNX model to {onnx_model_path}")
        Path(onnx_model_path).parent.mkdir(parents=True, exist_ok=True)

        dynamic_axes = None
        output_names = None

        if model_type in ["vit", "swin"]:
            dynamic_axes, output_names = {key: {0: "pixel_values"} for key in example_inputs}, ["logits"]
        else:
            dynamic_axes, output_names = build_dynamic_axes(example_inputs, example_outputs_flatten)

        replace_torch_functions()
        torch_onnx_export(
            model=model,
            args=tuple(example_inputs.values()),
            f=onnx_model_path,
            input_names=list(example_inputs.keys()),
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
            opset_version=opset_version,
            use_external_data_format=use_external_data_format,
        )
        restore_torch_functions()
    else:
        logger.info(f"Skip export since model existed: {onnx_model_path}")

    onnx_model_file, is_valid_onnx_model, vocab_size = validate_and_optimize_onnx(
        model_name,
        use_external_data_format,
        model_type,
        onnx_dir,
        input_names,
        use_gpu,
        precision,
        optimizer_info,
        validate_onnx,
        use_raw_attention_mask,
        overwrite,
        config,
        model_fusion_statistics,
        onnx_model_path,
        example_inputs,
        example_outputs_flatten,
        None,
        fusion_options,
    )

    return onnx_model_file, is_valid_onnx_model, vocab_size, max_input_size


def export_onnx_model_from_tf(
    model_name,
    opset_version,
    use_external_data_format,
    model_type,
    model_class,
    config_modifier,
    cache_dir,
    onnx_dir,
    input_names,
    use_gpu,
    precision,
    optimizer_info,
    validate_onnx,
    use_raw_attention_mask,
    overwrite,
    model_fusion_statistics,
    fusion_options,
):
    # Use CPU to export
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    # Fix "Using pad_token, but it is not set yet" error.
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    max_input_size = tokenizer.model_max_length

    config, model = load_tf_model(model_name, model_class, cache_dir, config_modifier)
    model.resize_token_embeddings(len(tokenizer))

    example_inputs = tokenizer.encode_plus(
        "This is a sample input",
        return_tensors="tf",
        max_length=max_input_size,
        padding="max_length",
        truncation=True,
    )
    example_inputs = filter_inputs(example_inputs, input_names)

    if config.is_encoder_decoder:
        example_inputs["decoder_input_ids"] = tokenizer.encode_plus(
            "This is a sample input",
            return_tensors="tf",
            max_length=max_input_size,
            padding="max_length",
            truncation=True,
        ).input_ids
    if model_name == "unc-nlp/lxmert-base-uncased":
        example_inputs["visual_feats"] = tf.random.normal([1, 1, config.visual_feat_dim])
        example_inputs["visual_pos"] = tf.random.normal([1, 1, config.visual_pos_dim])

    try:
        # Use no past state for these models
        if config.use_cache:
            config.use_cache = False
    except Exception:
        pass

    example_outputs = model(example_inputs, training=False)
    output_names = None

    # For xlnet models, only compare the last_hidden_state output.
    if model_name == "xlnet-base-cased" or model_name == "xlnet-large-cased":
        output_names = ["last_hidden_state"]
        example_outputs = example_outputs["last_hidden_state"]

    # Flatten is needed for gpt2 and distilgpt2. Output name sorting is needed for tf2onnx outputs to match onnx outputs.
    from tensorflow.python.util import nest

    example_outputs_flatten = nest.flatten(example_outputs)

    onnx_model_path = get_onnx_file_path(
        onnx_dir,
        model_name,
        len(input_names),
        False,
        use_gpu,
        precision,
        False,
        use_external_data_format,
    )
    tf_internal_model_path = onnx_model_path[:-5] if use_external_data_format else onnx_model_path

    if overwrite or not os.path.exists(tf_internal_model_path):
        logger.info(f"Exporting ONNX model to {onnx_model_path}")
        if not use_external_data_format:
            Path(tf_internal_model_path).parent.mkdir(parents=True, exist_ok=True)

        import zipfile

        import tf2onnx

        tf2onnx.logging.set_level(tf2onnx.logging.ERROR)
        specs = []
        for name, value in example_inputs.items():
            dims = [None] * len(value.shape)
            specs.append(tf.TensorSpec(tuple(dims), value.dtype, name=name))
        _, _ = tf2onnx.convert.from_keras(
            model,
            input_signature=tuple(specs),
            opset=opset_version,
            large_model=use_external_data_format,
            output_path=tf_internal_model_path,
        )
        if use_external_data_format:
            # need to unpack the zip for run_onnxruntime()
            with zipfile.ZipFile(tf_internal_model_path, "r") as z:
                z.extractall(os.path.dirname(tf_internal_model_path))
            tf_internal_model_path = os.path.join(os.path.dirname(tf_internal_model_path), "__MODEL_PROTO.onnx")
            if os.path.exists(onnx_model_path):
                os.remove(onnx_model_path)
            os.rename(tf_internal_model_path, onnx_model_path)

    else:
        logger.info(f"Skip export since model existed: {onnx_model_path}")

    model_type = model_type + "_tf"
    optimized_onnx_path, is_valid_onnx_model, vocab_size = validate_and_optimize_onnx(
        model_name,
        use_external_data_format,
        model_type,
        onnx_dir,
        input_names,
        use_gpu,
        precision,
        optimizer_info,
        validate_onnx,
        use_raw_attention_mask,
        overwrite,
        config,
        model_fusion_statistics,
        onnx_model_path,
        example_inputs,
        example_outputs_flatten,
        output_names,
        fusion_options,
    )

    return (
        optimized_onnx_path,
        is_valid_onnx_model,
        vocab_size,
        max_input_size,
    )

# === atelier-kyo-manager/.venv_backup\Lib\site-packages\pip\_vendor\requests\adapters.py ===
"""
requests.adapters
~~~~~~~~~~~~~~~~~

This module contains the transport adapters that Requests uses to define
and maintain connections.
"""

import os.path
import socket  # noqa: F401
import typing
import warnings

from pip._vendor.urllib3.exceptions import ClosedPoolError, ConnectTimeoutError
from pip._vendor.urllib3.exceptions import HTTPError as _HTTPError
from pip._vendor.urllib3.exceptions import InvalidHeader as _InvalidHeader
from pip._vendor.urllib3.exceptions import (
    LocationValueError,
    MaxRetryError,
    NewConnectionError,
    ProtocolError,
)
from pip._vendor.urllib3.exceptions import ProxyError as _ProxyError
from pip._vendor.urllib3.exceptions import ReadTimeoutError, ResponseError
from pip._vendor.urllib3.exceptions import SSLError as _SSLError
from pip._vendor.urllib3.poolmanager import PoolManager, proxy_from_url
from pip._vendor.urllib3.util import Timeout as TimeoutSauce
from pip._vendor.urllib3.util import parse_url
from pip._vendor.urllib3.util.retry import Retry
from pip._vendor.urllib3.util.ssl_ import create_urllib3_context

from .auth import _basic_auth_str
from .compat import basestring, urlparse
from .cookies import extract_cookies_to_jar
from .exceptions import (
    ConnectionError,
    ConnectTimeout,
    InvalidHeader,
    InvalidProxyURL,
    InvalidSchema,
    InvalidURL,
    ProxyError,
    ReadTimeout,
    RetryError,
    SSLError,
)
from .models import Response
from .structures import CaseInsensitiveDict
from .utils import (
    DEFAULT_CA_BUNDLE_PATH,
    extract_zipped_paths,
    get_auth_from_url,
    get_encoding_from_headers,
    prepend_scheme_if_needed,
    select_proxy,
    urldefragauth,
)

try:
    from pip._vendor.urllib3.contrib.socks import SOCKSProxyManager
except ImportError:

    def SOCKSProxyManager(*args, **kwargs):
        raise InvalidSchema("Missing dependencies for SOCKS support.")


if typing.TYPE_CHECKING:
    from .models import PreparedRequest


DEFAULT_POOLBLOCK = False
DEFAULT_POOLSIZE = 10
DEFAULT_RETRIES = 0
DEFAULT_POOL_TIMEOUT = None


try:
    import ssl  # noqa: F401

    _preloaded_ssl_context = create_urllib3_context()
    _preloaded_ssl_context.load_verify_locations(
        extract_zipped_paths(DEFAULT_CA_BUNDLE_PATH)
    )
except ImportError:
    # Bypass default SSLContext creation when Python
    # interpreter isn't built with the ssl module.
    _preloaded_ssl_context = None


def _urllib3_request_context(
    request: "PreparedRequest",
    verify: "bool | str | None",
    client_cert: "typing.Tuple[str, str] | str | None",
    poolmanager: "PoolManager",
) -> "(typing.Dict[str, typing.Any], typing.Dict[str, typing.Any])":
    host_params = {}
    pool_kwargs = {}
    parsed_request_url = urlparse(request.url)
    scheme = parsed_request_url.scheme.lower()
    port = parsed_request_url.port

    # Determine if we have and should use our default SSLContext
    # to optimize performance on standard requests.
    poolmanager_kwargs = getattr(poolmanager, "connection_pool_kw", {})
    has_poolmanager_ssl_context = poolmanager_kwargs.get("ssl_context")
    should_use_default_ssl_context = (
        _preloaded_ssl_context is not None and not has_poolmanager_ssl_context
    )

    cert_reqs = "CERT_REQUIRED"
    if verify is False:
        cert_reqs = "CERT_NONE"
    elif verify is True and should_use_default_ssl_context:
        pool_kwargs["ssl_context"] = _preloaded_ssl_context
    elif isinstance(verify, str):
        if not os.path.isdir(verify):
            pool_kwargs["ca_certs"] = verify
        else:
            pool_kwargs["ca_cert_dir"] = verify
    pool_kwargs["cert_reqs"] = cert_reqs
    if client_cert is not None:
        if isinstance(client_cert, tuple) and len(client_cert) == 2:
            pool_kwargs["cert_file"] = client_cert[0]
            pool_kwargs["key_file"] = client_cert[1]
        else:
            # According to our docs, we allow users to specify just the client
            # cert path
            pool_kwargs["cert_file"] = client_cert
    host_params = {
        "scheme": scheme,
        "host": parsed_request_url.hostname,
        "port": port,
    }
    return host_params, pool_kwargs


class BaseAdapter:
    """The Base Transport Adapter"""

    def __init__(self):
        super().__init__()

    def send(
        self, request, stream=False, timeout=None, verify=True, cert=None, proxies=None
    ):
        """Sends PreparedRequest object. Returns Response object.

        :param request: The :class:`PreparedRequest <PreparedRequest>` being sent.
        :param stream: (optional) Whether to stream the request content.
        :param timeout: (optional) How long to wait for the server to send
            data before giving up, as a float, or a :ref:`(connect timeout,
            read timeout) <timeouts>` tuple.
        :type timeout: float or tuple
        :param verify: (optional) Either a boolean, in which case it controls whether we verify
            the server's TLS certificate, or a string, in which case it must be a path
            to a CA bundle to use
        :param cert: (optional) Any user-provided SSL certificate to be trusted.
        :param proxies: (optional) The proxies dictionary to apply to the request.
        """
        raise NotImplementedError

    def close(self):
        """Cleans up adapter specific items."""
        raise NotImplementedError


class HTTPAdapter(BaseAdapter):
    """The built-in HTTP Adapter for urllib3.

    Provides a general-case interface for Requests sessions to contact HTTP and
    HTTPS urls by implementing the Transport Adapter interface. This class will
    usually be created by the :class:`Session <Session>` class under the
    covers.

    :param pool_connections: The number of urllib3 connection pools to cache.
    :param pool_maxsize: The maximum number of connections to save in the pool.
    :param max_retries: The maximum number of retries each connection
        should attempt. Note, this applies only to failed DNS lookups, socket
        connections and connection timeouts, never to requests where data has
        made it to the server. By default, Requests does not retry failed
        connections. If you need granular control over the conditions under
        which we retry a request, import urllib3's ``Retry`` class and pass
        that instead.
    :param pool_block: Whether the connection pool should block for connections.

    Usage::

      >>> import requests
      >>> s = requests.Session()
      >>> a = requests.adapters.HTTPAdapter(max_retries=3)
      >>> s.mount('http://', a)
    """

    __attrs__ = [
        "max_retries",
        "config",
        "_pool_connections",
        "_pool_maxsize",
        "_pool_block",
    ]

    def __init__(
        self,
        pool_connections=DEFAULT_POOLSIZE,
        pool_maxsize=DEFAULT_POOLSIZE,
        max_retries=DEFAULT_RETRIES,
        pool_block=DEFAULT_POOLBLOCK,
    ):
        if max_retries == DEFAULT_RETRIES:
            self.max_retries = Retry(0, read=False)
        else:
            self.max_retries = Retry.from_int(max_retries)
        self.config = {}
        self.proxy_manager = {}

        super().__init__()

        self._pool_connections = pool_connections
        self._pool_maxsize = pool_maxsize
        self._pool_block = pool_block

        self.init_poolmanager(pool_connections, pool_maxsize, block=pool_block)

    def __getstate__(self):
        return {attr: getattr(self, attr, None) for attr in self.__attrs__}

    def __setstate__(self, state):
        # Can't handle by adding 'proxy_manager' to self.__attrs__ because
        # self.poolmanager uses a lambda function, which isn't pickleable.
        self.proxy_manager = {}
        self.config = {}

        for attr, value in state.items():
            setattr(self, attr, value)

        self.init_poolmanager(
            self._pool_connections, self._pool_maxsize, block=self._pool_block
        )

    def init_poolmanager(
        self, connections, maxsize, block=DEFAULT_POOLBLOCK, **pool_kwargs
    ):
        """Initializes a urllib3 PoolManager.

        This method should not be called from user code, and is only
        exposed for use when subclassing the
        :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.

        :param connections: The number of urllib3 connection pools to cache.
        :param maxsize: The maximum number of connections to save in the pool.
        :param block: Block when no free connections are available.
        :param pool_kwargs: Extra keyword arguments used to initialize the Pool Manager.
        """
        # save these values for pickling
        self._pool_connections = connections
        self._pool_maxsize = maxsize
        self._pool_block = block

        self.poolmanager = PoolManager(
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            **pool_kwargs,
        )

    def proxy_manager_for(self, proxy, **proxy_kwargs):
        """Return urllib3 ProxyManager for the given proxy.

        This method should not be called from user code, and is only
        exposed for use when subclassing the
        :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.

        :param proxy: The proxy to return a urllib3 ProxyManager for.
        :param proxy_kwargs: Extra keyword arguments used to configure the Proxy Manager.
        :returns: ProxyManager
        :rtype: urllib3.ProxyManager
        """
        if proxy in self.proxy_manager:
            manager = self.proxy_manager[proxy]
        elif proxy.lower().startswith("socks"):
            username, password = get_auth_from_url(proxy)
            manager = self.proxy_manager[proxy] = SOCKSProxyManager(
                proxy,
                username=username,
                password=password,
                num_pools=self._pool_connections,
                maxsize=self._pool_maxsize,
                block=self._pool_block,
                **proxy_kwargs,
            )
        else:
            proxy_headers = self.proxy_headers(proxy)
            manager = self.proxy_manager[proxy] = proxy_from_url(
                proxy,
                proxy_headers=proxy_headers,
                num_pools=self._pool_connections,
                maxsize=self._pool_maxsize,
                block=self._pool_block,
                **proxy_kwargs,
            )

        return manager

    def cert_verify(self, conn, url, verify, cert):
        """Verify a SSL certificate. This method should not be called from user
        code, and is only exposed for use when subclassing the
        :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.

        :param conn: The urllib3 connection object associated with the cert.
        :param url: The requested URL.
        :param verify: Either a boolean, in which case it controls whether we verify
            the server's TLS certificate, or a string, in which case it must be a path
            to a CA bundle to use
        :param cert: The SSL certificate to verify.
        """
        if url.lower().startswith("https") and verify:
            conn.cert_reqs = "CERT_REQUIRED"

            # Only load the CA certificates if 'verify' is a string indicating the CA bundle to use.
            # Otherwise, if verify is a boolean, we don't load anything since
            # the connection will be using a context with the default certificates already loaded,
            # and this avoids a call to the slow load_verify_locations()
            if verify is not True:
                # `verify` must be a str with a path then
                cert_loc = verify

                if not os.path.exists(cert_loc):
                    raise OSError(
                        f"Could not find a suitable TLS CA certificate bundle, "
                        f"invalid path: {cert_loc}"
                    )

                if not os.path.isdir(cert_loc):
                    conn.ca_certs = cert_loc
                else:
                    conn.ca_cert_dir = cert_loc
        else:
            conn.cert_reqs = "CERT_NONE"
            conn.ca_certs = None
            conn.ca_cert_dir = None

        if cert:
            if not isinstance(cert, basestring):
                conn.cert_file = cert[0]
                conn.key_file = cert[1]
            else:
                conn.cert_file = cert
                conn.key_file = None
            if conn.cert_file and not os.path.exists(conn.cert_file):
                raise OSError(
                    f"Could not find the TLS certificate file, "
                    f"invalid path: {conn.cert_file}"
                )
            if conn.key_file and not os.path.exists(conn.key_file):
                raise OSError(
                    f"Could not find the TLS key file, invalid path: {conn.key_file}"
                )

    def build_response(self, req, resp):
        """Builds a :class:`Response <requests.Response>` object from a urllib3
        response. This should not be called from user code, and is only exposed
        for use when subclassing the
        :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`

        :param req: The :class:`PreparedRequest <PreparedRequest>` used to generate the response.
        :param resp: The urllib3 response object.
        :rtype: requests.Response
        """
        response = Response()

        # Fallback to None if there's no status_code, for whatever reason.
        response.status_code = getattr(resp, "status", None)

        # Make headers case-insensitive.
        response.headers = CaseInsensitiveDict(getattr(resp, "headers", {}))

        # Set encoding.
        response.encoding = get_encoding_from_headers(response.headers)
        response.raw = resp
        response.reason = response.raw.reason

        if isinstance(req.url, bytes):
            response.url = req.url.decode("utf-8")
        else:
            response.url = req.url

        # Add new cookies from the server.
        extract_cookies_to_jar(response.cookies, req, resp)

        # Give the Response some context.
        response.request = req
        response.connection = self

        return response

    def build_connection_pool_key_attributes(self, request, verify, cert=None):
        """Build the PoolKey attributes used by urllib3 to return a connection.

        This looks at the PreparedRequest, the user-specified verify value,
        and the value of the cert parameter to determine what PoolKey values
        to use to select a connection from a given urllib3 Connection Pool.

        The SSL related pool key arguments are not consistently set. As of
        this writing, use the following to determine what keys may be in that
        dictionary:

        * If ``verify`` is ``True``, ``"ssl_context"`` will be set and will be the
          default Requests SSL Context
        * If ``verify`` is ``False``, ``"ssl_context"`` will not be set but
          ``"cert_reqs"`` will be set
        * If ``verify`` is a string, (i.e., it is a user-specified trust bundle)
          ``"ca_certs"`` will be set if the string is not a directory recognized
          by :py:func:`os.path.isdir`, otherwise ``"ca_certs_dir"`` will be
          set.
        * If ``"cert"`` is specified, ``"cert_file"`` will always be set. If
          ``"cert"`` is a tuple with a second item, ``"key_file"`` will also
          be present

        To override these settings, one may subclass this class, call this
        method and use the above logic to change parameters as desired. For
        example, if one wishes to use a custom :py:class:`ssl.SSLContext` one
        must both set ``"ssl_context"`` and based on what else they require,
        alter the other keys to ensure the desired behaviour.

        :param request:
            The PreparedReqest being sent over the connection.
        :type request:
            :class:`~requests.models.PreparedRequest`
        :param verify:
            Either a boolean, in which case it controls whether
            we verify the server's TLS certificate, or a string, in which case it
            must be a path to a CA bundle to use.
        :param cert:
            (optional) Any user-provided SSL certificate for client
            authentication (a.k.a., mTLS). This may be a string (i.e., just
            the path to a file which holds both certificate and key) or a
            tuple of length 2 with the certificate file path and key file
            path.
        :returns:
            A tuple of two dictionaries. The first is the "host parameters"
            portion of the Pool Key including scheme, hostname, and port. The
            second is a dictionary of SSLContext related parameters.
        """
        return _urllib3_request_context(request, verify, cert, self.poolmanager)

    def get_connection_with_tls_context(self, request, verify, proxies=None, cert=None):
        """Returns a urllib3 connection for the given request and TLS settings.
        This should not be called from user code, and is only exposed for use
        when subclassing the :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.

        :param request:
            The :class:`PreparedRequest <PreparedRequest>` object to be sent
            over the connection.
        :param verify:
            Either a boolean, in which case it controls whether we verify the
            server's TLS certificate, or a string, in which case it must be a
            path to a CA bundle to use.
        :param proxies:
            (optional) The proxies dictionary to apply to the request.
        :param cert:
            (optional) Any user-provided SSL certificate to be used for client
            authentication (a.k.a., mTLS).
        :rtype:
            urllib3.ConnectionPool
        """
        proxy = select_proxy(request.url, proxies)
        try:
            host_params, pool_kwargs = self.build_connection_pool_key_attributes(
                request,
                verify,
                cert,
            )
        except ValueError as e:
            raise InvalidURL(e, request=request)
        if proxy:
            proxy = prepend_scheme_if_needed(proxy, "http")
            proxy_url = parse_url(proxy)
            if not proxy_url.host:
                raise InvalidProxyURL(
                    "Please check proxy URL. It is malformed "
                    "and could be missing the host."
                )
            proxy_manager = self.proxy_manager_for(proxy)
            conn = proxy_manager.connection_from_host(
                **host_params, pool_kwargs=pool_kwargs
            )
        else:
            # Only scheme should be lower case
            conn = self.poolmanager.connection_from_host(
                **host_params, pool_kwargs=pool_kwargs
            )

        return conn

    def get_connection(self, url, proxies=None):
        """DEPRECATED: Users should move to `get_connection_with_tls_context`
        for all subclasses of HTTPAdapter using Requests>=2.32.2.

        Returns a urllib3 connection for the given URL. This should not be
        called from user code, and is only exposed for use when subclassing the
        :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.

        :param url: The URL to connect to.
        :param proxies: (optional) A Requests-style dictionary of proxies used on this request.
        :rtype: urllib3.ConnectionPool
        """
        warnings.warn(
            (
                "`get_connection` has been deprecated in favor of "
                "`get_connection_with_tls_context`. Custom HTTPAdapter subclasses "
                "will need to migrate for Requests>=2.32.2. Please see "
                "https://github.com/psf/requests/pull/6710 for more details."
            ),
            DeprecationWarning,
        )
        proxy = select_proxy(url, proxies)

        if proxy:
            proxy = prepend_scheme_if_needed(proxy, "http")
            proxy_url = parse_url(proxy)
            if not proxy_url.host:
                raise InvalidProxyURL(
                    "Please check proxy URL. It is malformed "
                    "and could be missing the host."
                )
            proxy_manager = self.proxy_manager_for(proxy)
            conn = proxy_manager.connection_from_url(url)
        else:
            # Only scheme should be lower case
            parsed = urlparse(url)
            url = parsed.geturl()
            conn = self.poolmanager.connection_from_url(url)

        return conn

    def close(self):
        """Disposes of any internal state.

        Currently, this closes the PoolManager and any active ProxyManager,
        which closes any pooled connections.
        """
        self.poolmanager.clear()
        for proxy in self.proxy_manager.values():
            proxy.clear()

    def request_url(self, request, proxies):
        """Obtain the url to use when making the final request.

        If the message is being sent through a HTTP proxy, the full URL has to
        be used. Otherwise, we should only use the path portion of the URL.

        This should not be called from user code, and is only exposed for use
        when subclassing the
        :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.

        :param request: The :class:`PreparedRequest <PreparedRequest>` being sent.
        :param proxies: A dictionary of schemes or schemes and hosts to proxy URLs.
        :rtype: str
        """
        proxy = select_proxy(request.url, proxies)
        scheme = urlparse(request.url).scheme

        is_proxied_http_request = proxy and scheme != "https"
        using_socks_proxy = False
        if proxy:
            proxy_scheme = urlparse(proxy).scheme.lower()
            using_socks_proxy = proxy_scheme.startswith("socks")

        url = request.path_url
        if url.startswith("//"):  # Don't confuse urllib3
            url = f"/{url.lstrip('/')}"

        if is_proxied_http_request and not using_socks_proxy:
            url = urldefragauth(request.url)

        return url

    def add_headers(self, request, **kwargs):
        """Add any headers needed by the connection. As of v2.0 this does
        nothing by default, but is left for overriding by users that subclass
        the :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.

        This should not be called from user code, and is only exposed for use
        when subclassing the
        :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.

        :param request: The :class:`PreparedRequest <PreparedRequest>` to add headers to.
        :param kwargs: The keyword arguments from the call to send().
        """
        pass

    def proxy_headers(self, proxy):
        """Returns a dictionary of the headers to add to any request sent
        through a proxy. This works with urllib3 magic to ensure that they are
        correctly sent to the proxy, rather than in a tunnelled request if
        CONNECT is being used.

        This should not be called from user code, and is only exposed for use
        when subclassing the
        :class:`HTTPAdapter <requests.adapters.HTTPAdapter>`.

        :param proxy: The url of the proxy being used for this request.
        :rtype: dict
        """
        headers = {}
        username, password = get_auth_from_url(proxy)

        if username:
            headers["Proxy-Authorization"] = _basic_auth_str(username, password)

        return headers

    def send(
        self, request, stream=False, timeout=None, verify=True, cert=None, proxies=None
    ):
        """Sends PreparedRequest object. Returns Response object.

        :param request: The :class:`PreparedRequest <PreparedRequest>` being sent.
        :param stream: (optional) Whether to stream the request content.
        :param timeout: (optional) How long to wait for the server to send
            data before giving up, as a float, or a :ref:`(connect timeout,
            read timeout) <timeouts>` tuple.
        :type timeout: float or tuple or urllib3 Timeout object
        :param verify: (optional) Either a boolean, in which case it controls whether
            we verify the server's TLS certificate, or a string, in which case it
            must be a path to a CA bundle to use
        :param cert: (optional) Any user-provided SSL certificate to be trusted.
        :param proxies: (optional) The proxies dictionary to apply to the request.
        :rtype: requests.Response
        """

        try:
            conn = self.get_connection_with_tls_context(
                request, verify, proxies=proxies, cert=cert
            )
        except LocationValueError as e:
            raise InvalidURL(e, request=request)

        self.cert_verify(conn, request.url, verify, cert)
        url = self.request_url(request, proxies)
        self.add_headers(
            request,
            stream=stream,
            timeout=timeout,
            verify=verify,
            cert=cert,
            proxies=proxies,
        )

        chunked = not (request.body is None or "Content-Length" in request.headers)

        if isinstance(timeout, tuple):
            try:
                connect, read = timeout
                timeout = TimeoutSauce(connect=connect, read=read)
            except ValueError:
                raise ValueError(
                    f"Invalid timeout {timeout}. Pass a (connect, read) timeout tuple, "
                    f"or a single float to set both timeouts to the same value."
                )
        elif isinstance(timeout, TimeoutSauce):
            pass
        else:
            timeout = TimeoutSauce(connect=timeout, read=timeout)

        try:
            resp = conn.urlopen(
                method=request.method,
                url=url,
                body=request.body,
                headers=request.headers,
                redirect=False,
                assert_same_host=False,
                preload_content=False,
                decode_content=False,
                retries=self.max_retries,
                timeout=timeout,
                chunked=chunked,
            )

        except (ProtocolError, OSError) as err:
            raise ConnectionError(err, request=request)

        except MaxRetryError as e:
            if isinstance(e.reason, ConnectTimeoutError):
                # TODO: Remove this in 3.0.0: see #2811
                if not isinstance(e.reason, NewConnectionError):
                    raise ConnectTimeout(e, request=request)

            if isinstance(e.reason, ResponseError):
                raise RetryError(e, request=request)

            if isinstance(e.reason, _ProxyError):
                raise ProxyError(e, request=request)

            if isinstance(e.reason, _SSLError):
                # This branch is for urllib3 v1.22 and later.
                raise SSLError(e, request=request)

            raise ConnectionError(e, request=request)

        except ClosedPoolError as e:
            raise ConnectionError(e, request=request)

        except _ProxyError as e:
            raise ProxyError(e)

        except (_SSLError, _HTTPError) as e:
            if isinstance(e, _SSLError):
                # This branch is for urllib3 versions earlier than v1.22
                raise SSLError(e, request=request)
            elif isinstance(e, ReadTimeoutError):
                raise ReadTimeout(e, request=request)
            elif isinstance(e, _InvalidHeader):
                raise InvalidHeader(e, request=request)
            else:
                raise

        return self.build_response(request, resp)